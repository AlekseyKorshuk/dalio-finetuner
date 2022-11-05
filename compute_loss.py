from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

data = ["User", "User:"]
inputs = tokenizer(data, return_tensors="pt", padding=True).to(0)
inputs["labels"] = tensor(inputs["input_ids"].tolist().copy(), device="cuda:0")
print(inputs)


generated_output_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False)
print(generated_output_ids)

outputs = model(**inputs)

print(outputs.loss)

labels = inputs["labels"]
lm_logits = outputs.logits

shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()  # Flatten the tokens
loss_fct = CrossEntropyLoss(ignore_index=-1)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print(loss)
