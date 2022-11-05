from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor, cat

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

input_string = ["User: How are you?\nBot:", "User: Hello, how old are you?\nBot:"]
output_strings = [" I am fine, thanks for asking\n", " I am 20\n"]

data = [input_ + output_ for input_, output_ in zip(input_string, output_strings)]
inputs = tokenizer(data, return_tensors="pt", padding=True).to(0)
inputs["labels"] = tensor(inputs.input_ids.tolist().copy(), device="cuda:0")
output_lengths = [len(tokenizer(output_string).input_ids) for output_string in output_strings]
for i in range(len(inputs["labels"])):
    for j in range(0, len(inputs["labels"]) - output_lengths[i]):
        inputs["labels"][i][j] = -100
print(inputs)
outputs = model(**inputs)
print(outputs.loss)

labels = inputs["labels"]
lm_logits = outputs.logits

shift_logits = lm_logits[..., :-1, :].contiguous()
shift_logits = shift_logits.tolist()
for i in range(len(shift_logits)):
    shift_logits[i] = shift_logits[i][-output_lengths[i]:]
shift_logits = cat([tensor(data, device="cuda:0") for data in shift_logits])

shift_labels = labels[..., 1:].contiguous()
shift_labels = shift_labels.tolist()
for i in range(len(shift_labels)):
    shift_labels[i] = shift_labels[i][-output_lengths[i]:]
shift_labels = cat([tensor(data, device="cuda:0") for data in shift_labels])

loss_fct = CrossEntropyLoss(ignore_index=-1)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print(loss)

labels = inputs["labels"]
lm_logits = outputs.logits
shift_logits = lm_logits[..., :-1, :].contiguous()

shift_labels = labels[..., 1:].contiguous()
for i in range(len(shift_labels)):
    for j in range(0, len(shift_labels[i]) - output_lengths[i]):
        shift_labels[i][j] = -100

loss_fct = CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print(loss)
