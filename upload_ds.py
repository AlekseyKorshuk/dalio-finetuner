import json

with open("dataset.txt", "r") as f:
    conversations = json.load(f)


def get_data_from_conversation(conversation):
    data = conversation.split("\n")
    pairs = []
    for i, line in enumerate(data):
        if line.startswith("Ray:") and not line.startswith("Ray: No problem"):
            input_text = "\n".join(data[:i]) + "\nRay:"
            output_text = line.split("Ray:")[1] + "\n"
            pairs.append((input_text, output_text))
    return pairs


pairs = []
for convo in conversations:
    # print(convo)
    pairs += get_data_from_conversation(convo)

train_data = pairs[:int(len(pairs) * 0.8)]
validation_data = pairs[int(len(pairs) * 0.8):int(len(pairs) * 0.95)]
test_data = pairs[int(len(pairs) * 0.95):]

from datasets import Dataset, DatasetDict

list1, list2 = zip(*train_data)
train_split = Dataset.from_dict({"input_text": list1, "output_text": list2})

list1, list2 = zip(*validation_data)
validation_split = Dataset.from_dict({"input_text": list1, "output_text": list2})

list1, list2 = zip(*test_data)
test_split = Dataset.from_dict({"input_text": list1, "output_text": list2})



dataset = DatasetDict({"train": train_split, "validation": validation_split, "test": test_split})

print(dataset)

dataset.push_to_hub("AlekseyKorshuk/dalio-v1")
