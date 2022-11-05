from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = {
    "input_ids": torch.tensor([0, 0, 0, 0, 0], device="cuda:0"),
    "attention_mask": torch.tensor([1, 1, 1, 1, 1], device="cuda:0"),
    "labels": torch.tensor([0, 0, 0, 0, 0], device="cuda:0"),
}

outputs = model(**inputs)
print(outputs)