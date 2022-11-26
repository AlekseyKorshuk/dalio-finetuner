from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

ds_name = "Jellywibble/dalio_convo_scores"

dataset = load_dataset(ds_name)

if "validation" not in dataset.keys():
    dataset["validation"] = load_dataset(
        ds_name,
        split=f"train[:{20}%]",
    )
    dataset["train"] = load_dataset(
        ds_name,
        split=f"train[{20}%:]",
    )

dataset = DatasetDict(
    {
        "train": Dataset.from_dict(
            {
                "text": dataset["train"]["text"],
                "label": dataset["train"]["score"],
            }
        ),
        "validation": Dataset.from_dict(
            {
                "text": dataset["validation"]["text"],
                "label": dataset["validation"]["score"],
            }
        )
    }
)

print(dataset)

dataset.push_to_hub("AlekseyKorshuk/dalio_convo_scores")
