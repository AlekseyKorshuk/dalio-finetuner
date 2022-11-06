from datasets import load_dataset, Dataset, DatasetDict

dataset = load_dataset("amazon_reviews_multi", "en")

train_data = {
    "input_text": [],
    "output_text": []
}
for i in range(1000):
    input_text = f'Title: {dataset["train"][i]["review_title"]}\nBody:'
    train_data["input_text"].append(input_text)
    output_text = f" {dataset['train'][i]['review_body']}\n"
    train_data["output_text"].append(output_text)

validation_data = {
    "input_text": [],
    "output_text": []
}
for i in range(100):
    input_text = f'Title: {dataset["validation"][i]["review_title"]}\nBody:'
    validation_data["input_text"].append(input_text)
    output_text = f" {dataset['validation'][i]['review_body']}\n"
    validation_data["output_text"].append(output_text)

test_data = {
    "input_text": [],
    "output_text": []
}
for i in range(10):
    input_text = f'Title: {dataset["test"][i]["review_title"]}\nBody:'
    test_data["input_text"].append(input_text)
    output_text = f" {dataset['test'][i]['review_body']}\n"
    test_data["output_text"].append(output_text)


ds = DatasetDict(
    {
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(validation_data),
        "test": Dataset.from_dict(test_data),
    }
)

ds.push_to_hub("AlekseyKorshuk/amazon-reviews-input-output")

