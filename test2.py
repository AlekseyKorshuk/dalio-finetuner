from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_string = "User:"

inputs = tokenizer(input_string, return_tensors="pt").to(0)
input_len = len(inputs["input_ids"][0])
print("Input length:", input_len)
inputs["labels"] = inputs["input_ids"]

outputs = model.generate(**inputs, do_sample=False, max_new_tokens=3)
output_len = len(outputs[0][input_len:])
print("Output length:", output_len)

inputs = {
    'input_ids': outputs,
    'attention_mask': tensor([[1] * (input_len + output_len)], device='cuda:0'),
    'labels': outputs
}


def test(inputs, input_len, output_len):
    outputs = model(**inputs)
    print("Default loss:", outputs.loss)
    labels = inputs["labels"]
    lm_logits = outputs.logits
    print(lm_logits[0])

    shift_logits = lm_logits[..., :-1, :].contiguous()
    print(shift_logits)
    shift_labels = labels[..., 1:].contiguous()  # Flatten the tokens
    print(shift_labels)
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print("Total loss:", loss)

    shift_logits = lm_logits[..., input_len:-1, :].contiguous()
    print(shift_logits)
    shift_labels = labels[..., input_len:].contiguous()  # Flatten the tokens
    print(shift_labels)
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print("Output loss:", loss)

test(inputs, input_len, output_len)