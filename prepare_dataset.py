from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b").to(0)

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
dataset.push_to_hub("AlekseyKorshuk/dalio-book-handwritten-io")
