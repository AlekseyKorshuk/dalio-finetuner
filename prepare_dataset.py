from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", padding_side="left", pad_token_id=0)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", device_map="auto")

dataset1 = load_dataset("AlekseyKorshuk/dalio-handwritten-io")
dataset2 = load_dataset("Jellywibble/dalio-finetune-principles_book")

dataset2_io = DatasetDict(
    {
        "train": Dataset.from_dict(
            {
                "input_text": [""] * len(dataset2["train"]),
                "output_text": dataset2["train"]["text"],
            }
        ),
        "validation": Dataset.from_dict(
            {
                "input_text": [""] * len(dataset2["train"]),
                "output_text": dataset2["train"]["text"],
            }
        )
    }
)

dataset = DatasetDict(
    {
        "train": concatenate_datasets([dataset1["train"], dataset2_io["train"]]),
        "validation": concatenate_datasets([dataset1["validation"], dataset2_io["validation"]]),
        "test": dataset1["test"],
    }
)

print(dataset)

# dataset.push_to_hub("AlekseyKorshuk/dalio-book-handwritten-io")

block_size = 1024
from copy import deepcopy

input_column_name = "input_text"
output_column_name = "output_text"


def tokenize_function(examples):
    if "text" in dataset["train"].column_names:
        inputs = tokenizer(examples["text"], padding="longest", max_length=block_size,
                           truncation=True)
        # inputs = tokenizer(examples[text_column_name])
        inputs["labels"] = deepcopy(inputs.input_ids)
        for i in range(len(inputs["labels"])):
            for j in range(len(inputs["labels"][i])):
                if inputs["labels"][i][j] == tokenizer.pad_token_id:
                    inputs["labels"][i][j] = -100
        return inputs
    input_texts = examples[input_column_name]
    output_texts = examples[output_column_name]
    data = [input_ + output_ for input_, output_ in zip(input_texts, output_texts)]
    inputs = tokenizer(data, padding="longest", max_length=block_size, truncation=True)
    inputs["labels"] = deepcopy(inputs.input_ids)
    output_lengths = [len(tokenizer(output_string).input_ids) for output_string in output_texts]
    for i in range(len(inputs["labels"])):
        for j in range(0, len(inputs["labels"][i]) - output_lengths[i]):
            inputs["labels"][i][j] = -100
    return inputs


tokenized_datasets = dataset2_io.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

import torch
import tqdm

losses = []

for data in tqdm.tqdm(tokenized_datasets["train"]):
    for key in data.keys():
        data[key] = torch.tensor(data[key]).unsqueeze(0).to(model.device)
    # print(data)
    print(data)
    input("Press Enter to continue...")
    with torch.no_grad():
        output = model(**data)
    # print(output.loss)
    losses.append(float(output.loss))

# import numpy as np
#
# print(np.mean(losses))
# import pandas as pd
#
# print(pd.Series(losses).describe())
#
# dataset["train"] = dataset["train"].add_column("loss", losses).sort('loss')
#
# dataset["train"] = dataset["train"].remove_columns(["loss"])
#
# dataset.push_to_hub("AlekseyKorshuk/dalio-book-handwritten-io-sorted")
