from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import tensor

model = AutoModelForCausalLM.from_pretrained("gpt2").to(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_str = "User: Hello, how are you?\nBot:"
output_str = " I'm a bot.\n"

inputs = tokenizer(input_str, return_tensors="pt").to(0)
input_len = len(inputs["input_ids"][0])
outputs = model.generate(**inputs, do_sample=False, eos_token_id=198)
print(outputs)
print(tokenizer.decode(outputs[0]))

inputs = tokenizer(output_str, return_tensors="pt").to(0)
output_len = len(inputs["input_ids"][0])

inputs = {
    'input_ids': tensor([[12982, 25, 18435, 11, 703, 389, 345, 30, 198, 20630, 25,
                          314, 1101, 257, 10214, 13, 198]],
                        device='cuda:0'),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1]], device='cuda:0'),
    'labels': tensor([[12982, 25, 18435, 11, 703, 389, 345, 30, 198, 20630, 25,
                       314, 1101, 257, 10214, 13, 198]], device='cuda:0')
}
outputs = model(**inputs)
print(outputs.logits.size())
print(outputs.logits[0])

inputs = {
    'input_ids': tensor([[12982, 25, 18435, 11, 703, 389, 345, 30, 198, 20630, 25,
                          314, 1101, 257, 10214, 13]],
                        device='cuda:0'),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1]], device='cuda:0'),
    'labels': tensor([[12982, 25, 18435, 11, 703, 389, 345, 30, 198, 20630, 25,
                       314, 1101, 257, 10214, 13]], device='cuda:0')
}
outputs = model(**inputs)
print(outputs.logits.size())
print(outputs.logits[0])


def test(inputs, input_len):
    outputs = model(**inputs)
    print("Default loss:", outputs.loss)
    labels = inputs["labels"]
    lm_logits = outputs.logits
    print(lm_logits[0])

    shift_logits = lm_logits[..., :-1, :].contiguous()
    print(shift_logits)
    shift_labels = labels[..., 1:].contiguous()  # Flatten the tokens
    print(shift_labels)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print("Total loss:", loss)

    shift_logits = lm_logits[..., :-input_len, :].contiguous()
    shift_labels = labels[..., input_len:].contiguous()  # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print("Output loss:", loss)


test(inputs, input_len)
