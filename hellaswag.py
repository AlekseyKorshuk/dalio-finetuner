import logging
import math

from datasets import load_dataset
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
        print("Model device:", kwargs.get('model').device)
        if state.global_step != state.max_steps:
            return

        logger.info('calculating hellaswag metrics')
        model = kwargs.get('model')

        with torch.no_grad():
            accuracy, perplexity = score_hellaswag(
                self.ds, model, self.tokenizer, self.num_prompts, self.params
            )

        # if using deepspeed only log for the main process
        if state.is_world_process_zero:
            wandb.log({
                'hellaswag/accuracy': accuracy,
                'hellaswag/perplexity': perplexity
            })


def load_hellaswag_dataset():
    return load_dataset('hellaswag')['train']


def score_hellaswag(dataset, model, tokenizer, num_prompts, params):
    inputs = dataset['ctx_a'][:num_prompts]
    endings = dataset['endings'][:num_prompts]
    labels = dataset['label'][:num_prompts]

    likelihoods = []
    for ins, ends in zip(inputs, endings):
        scores = [log_likelihood(model, tokenizer, ins, e, params) for e in ends]
        likelihoods.append(scores)

    accuracy = calculate_accuracy(labels, likelihoods)
    perplexity = calculate_perplexity(labels, likelihoods)

    return accuracy, perplexity


def calculate_perplexity(labels, likelihoods):
    scores = []
    for label, likelihood in zip(labels, likelihoods):
        completion = likelihood[int(label)]
        scores.append(np.exp(-np.mean(completion)))
    return np.mean(scores)


def calculate_accuracy(correct_labels, log_likelihoods):
    correct_count = count_correct_labels(correct_labels, log_likelihoods)
    return correct_count / len(correct_labels)


def count_correct_labels(correct_labels, log_likelihoods):
    correct_count = 0
    for label, likelihoods in zip(correct_labels, log_likelihoods):
        correct_count += int(label) == np.argmax(likelihoods)
    return correct_count


def log_likelihood(model, tokenizer, text, output_text, params):
    input_tokens = tokenizer(text, return_tensors="pt").to(model.device)['input_ids']
    output_tokens = tokenizer(output_text, return_tensors="pt").to(model.device)['input_ids']

    args = params.copy()
    args = {
        'return_dict_in_generate': True,
        'output_scores': True,
        'max_new_tokens': 1
    }

    logprobs = []
    for token in output_tokens[0]:
        output = model.generate(input_tokens, **args)
        output_probs = torch.stack(output.scores, dim=1).softmax(-1)[0][0]
        prob = output_probs[token]
        logprobs.append(math.log(prob) if prob > 0 else -np.inf)
        input_tokens = torch.cat((input_tokens, token.resize(1, 1)), dim=1)
    return logprobs
