from datasets import load_dataset, Dataset, DatasetDict

dataset = load_dataset("amazon_reviews_multi", "en")

train_data = {
    "input_string": [],
    "output_string": []
}
for i in range(1000):
    input_string = f'Title: {dataset["train"][i]["review_title"]}\nBody:'
    train_data["input_string"].append(input_string)
    output_string = f" {dataset['train'][i]['review_body']}\n"
    train_data["output_string"].append(output_string)

validation_data = {
    "input_string": [],
    "output_string": []
}
for i in range(100):
    input_string = f'Title: {dataset["validation"][i]["review_title"]}\nBody:'
    validation_data["input_string"].append(input_string)
    output_string = f" {dataset['validation'][i]['review_body']}\n"
    validation_data["output_string"].append(output_string)

test_data = {
    "input_string": [],
    "output_string": []
}
for i in range(10):
    input_string = f'Title: {dataset["test"][i]["review_title"]}\nBody:'
    test_data["input_string"].append(input_string)
    output_string = f" {dataset['test'][i]['review_body']}\n"
    test_data["output_string"].append(output_string)


ds = DatasetDict(
    {
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(validation_data),
        "test": Dataset.from_dict(test_data),
    }
)

ds.push_to_hub("AlekseyKorshuk/amazon-reviews-input-output")

