"""
Evaluate a model against a fixed set of prompts to get a sense of its performance
during training
"""
import tqdm
from pandas import DataFrame
from transformers import TrainerCallback
import json
import pandas as pd
import torch
import logging
import wandb
from wandb import Table

logger = logging.getLogger(__name__)


class RecordExampleAnswersCallback(TrainerCallback):
    """Log responses to fixed questions to wandb. """

    def __init__(self, dataset, tokenizer, params):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.params = params

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get('model')

        with torch.no_grad():
            table = generate_table(
                model,
                self.tokenizer,
                self.dataset,
                self.params,
            )

        # if using deepspeed only log for the main process
        if state.is_world_process_zero:
            wandb.log({'eval_table': table}, step=state.global_step)


def generate_table(model, tokenizer, test_dataset, params):
    input_texts = test_dataset["input_text"]
    output_texts = test_dataset["output_text"]
    table = {
        "input": input_texts,
        "output": [],
        "target": output_texts
    }
    for sample in tqdm.tqdm(input_texts, desc="Generating table"):
        inputs = tokenizer(sample, return_tensors="pt")
        inputs.to(model.device)
        output_ids = model.generate(**inputs, **params)
        output = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):])
        table["output"].append(output)
        del inputs
    df = DataFrame(table)
    torch.cuda.empty_cache()
    return Table(data=df)
