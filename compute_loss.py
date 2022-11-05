from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

input_string = ["User: How are you?\nBot:", "User: Hello, how old are you?\nBot:"]
output_strings = [" I am fine, thanks for asking\n", " I am 20\n"]

data = [input_ + output_ for input_, output_ in zip(input_string, output_strings)]
inputs = tokenizer(data, return_tensors="pt", padding=True).to(0)
inputs["labels"] = tensor(inputs.input_ids.tolist().copy(), device="cuda:0")
print(inputs)

output_lengths = [len(tokenizer(output_string).input_ids) for output_string in output_strings]

outputs = model(**inputs)


print(outputs.loss)

labels = inputs["labels"]
lm_logits = outputs.logits
print(lm_logits)


shift_logits = lm_logits[..., :-1, :].contiguous()
shift_logits = shift_logits.tolist()
for i in range(len(shift_logits)):
    shift_logits[i] = shift_logits[i][-output_lengths[i]:]
shift_logits = tensor(shift_logits, device="cuda:0")

shift_labels = labels[..., 1:].contiguous()  # Flatten the tokens
shift_labels = shift_labels.tolist()
for i in range(len(shift_labels)):
    shift_labels[i] = shift_labels[i][-output_lengths[i]:]
shift_labels = tensor(shift_labels, device="cuda:0")
# print(shift_labels_test)
# print(shift_labels.size())
loss_fct = CrossEntropyLoss(ignore_index=-1)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print(loss)

#
# outputs = tokenizer(output_strings, return_tensors="pt", padding=False).to(0)
# print(inputs)
# output_len = len(outputs["input_ids"][0])
#
# generated_output_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False, eos_token_id=198)
# print(generated_output_ids)
#
# outputs = model(**inputs)
#
# print(outputs.loss)
#
# labels = inputs["labels"]
# lm_logits = outputs.logits
#
# shift_logits = lm_logits[..., :-1, :].contiguous()
# shift_labels = labels[..., 1:].contiguous()  # Flatten the tokens
# loss_fct = CrossEntropyLoss(ignore_index=-1)
# loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
# print(loss)
#
#
# shift_logits = lm_logits[..., input_len - 1:-1, :].contiguous()
# print(shift_logits)
# shift_labels = labels[..., input_len:].contiguous()  # Flatten the tokens
# print(shift_labels)
# loss_fct = CrossEntropyLoss(ignore_index=-1)
# loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
# print("Output loss:", loss)
