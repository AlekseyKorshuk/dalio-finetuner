import logging
import math

from datasets import load_dataset
from tqdm import tqdm
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
        if not float(state.epoch).is_integer():
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

    likelihoods = []
    is_match = []
    for input_text in tqdm(inputs, desc='HellaSwag'):
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids[0][1:]
        output = model(**inputs)
        output_probs = output.logits.softmax(-1)[0]
        for probs, input_id in zip(output_probs, input_ids):
            prob = probs[input_id]
            likelihoods.append(math.log(prob) if prob > 0 else -np.inf)
            is_match.append(probs.argmax() == input_id)

    accuracy = sum(is_match) / len(is_match)
    perplexity = np.exp(-np.mean(likelihoods))

    return accuracy, perplexity
