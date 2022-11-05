from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_str = "User: Hello, how are you?\nBot:"
output_str = ""

inputs = tokenizer(input_str, return_tensors="pt", return_token_type_ids=True).to(0)

# inputs = {
#     "input_ids": torch.tensor([0, 0, 0, 0, 0], device="cuda:0"),
#     "attention_mask": torch.tensor([1, 1, 1, 1, 1], device="cuda:0"),
#     "labels": torch.tensor([0, 0, 0, 0, 0], device="cuda:0"),
# }
print(inputs)
outputs = model(**inputs)
print(outputs.loss)
labels = inputs.labels
lm_logits = outputs.logits
shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
# Flatten the tokens
loss_fct = CrossEntropyLoss()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print(loss)

#
# inputs = {
#     "input_ids": torch.tensor([0, 0, 0, 0, 0], device="cuda:0"),
#     "attention_mask": torch.tensor([1, 1, 1, 1, 1], device="cuda:0"),
#     "labels": torch.tensor([0, 0, 0, 0, 123], device="cuda:0"),
# }
# print(inputs)
# outputs = model(**inputs)
# print(outputs.loss)
#
# inputs = {
#     "input_ids": torch.tensor([0, 0, 0, 0, 0], device="cuda:0"),
#     "attention_mask": torch.tensor([1, 1, 1, 1, 0], device="cuda:0"),
#     "labels": torch.tensor([0, 0, 0, 0, 123], device="cuda:0"),
# }
# print(inputs)
# outputs = model(**inputs)
# print(outputs.loss)