import logging
import math
from datasets import load_dataset
import tqdm
from transformers import TrainerCallback
import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


class HellaswagCallback(TrainerCallback):
    """Get hellaswag performance which measure common sense reasoning. """

    def __init__(self, tokenizer, params, num_prompts=32):
        self.tokenizer = tokenizer
        self.params = params
        self.num_prompts = num_prompts
        self.ds = load_hellaswag_dataset()

    def on_evaluate(self, args, state, control, **kwargs):
        # if not float(state.epoch).is_integer():
        #     return

        logger.info('calculating hellaswag metrics')
        model = kwargs.get('model')

        with torch.no_grad():
            perplexity, token_accuracy, sentence_accuracy, mean_probability, probabilities = score_hellaswag(
                self.ds, model, self.tokenizer, self.num_prompts, self.params
            )

        # if using deepspeed only log for the main process
        if state.is_world_process_zero:
            wandb.log({
                'hellaswag/sentence_accuracy': sentence_accuracy,
                'hellaswag/token_accuracy': token_accuracy,
                'hellaswag/perplexity': perplexity,
                'hellaswag/mean_probability': mean_probability,
                'hellaswag/probabilities': wandb.Histogram(np.array(probabilities)),
            })


def load_hellaswag_dataset():
    return load_dataset('hellaswag')['train']


def score_hellaswag(dataset, model, tokenizer, num_prompts, params):
    likelihoods = []
    is_token_match = []
    is_sentence_match = []
    probabilities = []
    for row_id in tqdm.trange(num_prompts, desc='HellaSwag'):
        row = dataset[row_id]
        endings_likelihoods = [[]] * len(row["endings"])
        for i, ending in enumerate(row["endings"]):
            input_text = row["ctx_a"]
            output_text = ending
            inputs = tokenizer(input_text + output_text, return_tensors="pt").to(model.device)
            output_len = len(tokenizer(output_text, return_tensors="pt").input_ids[0])

            input_ids = inputs.input_ids[0][1:]
            output = model(**inputs)
            output_probs = output.logits.softmax(-1)[0][:-1]
            # output_len = len(output_probs)
            for probs, input_id in zip(output_probs[-output_len:], input_ids[-output_len:]):
                prob = probs[input_id]
                # print(input_id, prob)
                likelihood = math.log(prob) if prob > 0 else -np.inf
                likelihoods.append(likelihood)
                endings_likelihoods[i].append(likelihood)
                is_token_match.append(probs.argmax() == input_id)

                if i == int(row["label"]):
                    probabilities.append(float(prob.cpu()))
        endings_likelihoods = [sum(endings_likelihood) for endings_likelihood in endings_likelihoods]
        ending_index = endings_likelihoods.index(max(endings_likelihoods))
        is_sentence_match.append(ending_index == int(row["label"]))

    sentence_accuracy = sum(is_sentence_match) / len(is_sentence_match)
    token_accuracy = float(sum(is_token_match) / len(is_token_match))
    perplexity = np.exp(-np.mean(likelihoods))
    mean_probability = np.mean(probabilities)

    return perplexity, token_accuracy, sentence_accuracy, mean_probability, probabilities
