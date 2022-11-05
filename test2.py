from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_string = "User"

inputs = tokenizer(input_string, return_tensors="pt").to(0)
input_len = len(inputs["input_ids"][0])

inputs["labels"] = inputs["input_ids"]
