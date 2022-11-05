from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_str = "User: Hello, how are you?\nBot:"
output_str = " I am fine"

inputs = tokenizer(input_str, return_tensors="pt").to(0)
input_len = len(inputs["input_ids"][0])

inputs = tokenizer(output_str, return_tensors="pt").to(0)
output_len = len(inputs["input_ids"][0])

inputs = {
    'input_ids': tensor([[12982, 25, 18435, 11, 703, 389, 345, 30, 198, 20630, 25,
                          314, 716, 3734]], device='cuda:0'),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1]], device='cuda:0'),
    'labels': tensor([[12982, 25, 18435, 11, 703, 389, 345, 30, 198, 20630, 25,
                       314, 716, 3734]], device='cuda:0')
}
print(inputs)
outputs = model(**inputs)
print(outputs.loss)
labels = inputs["labels"]
lm_logits = outputs.logits
print(lm_logits.size())
shift_logits = lm_logits[..., :-1, :].contiguous()
print(shift_logits.size())
shift_labels = labels[..., 1:].contiguous()
print(shift_labels.size())
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
