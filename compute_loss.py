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
    for j in range(0, len(inputs["labels"][i]) - output_lengths[i]):
        inputs["labels"][i][j] = -100
outputs = model(**inputs)
print(outputs.loss)
