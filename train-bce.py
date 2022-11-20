import torch
import wandb
import random
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_metric, load_dataset, concatenate_datasets
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from utils import ValidationCallback

AUTH_TOKEN = "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"
WANDB_KEY = "95713a8419108e9246736f20a564e81559a8e80f"

LR = 1e-6
MODEL = "EleutherAI/gpt-neo-1.3B"
NUM_LABELS = 1
OUTPUT_DIR = "/tmp/gptneo1p3B-large-wills-loss-function-tr-lr1e-6-seed0"
SEED = 0
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token


def reduce_randomness(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def get_train_eval_datasets():
    dataset = load_dataset("Jellywibble/dalio_convo_scores")['train']
    dataset = preprocess(dataset)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset['train'], dataset['test']


def preprocess(dataset):
    dataset = dataset.remove_columns("index")
    dataset = dataset.rename_column("score", "label")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def compute_regression_metrics(eval_pred):
    logits, labels = eval_pred
    mse = mean_squared_error(labels/3., np_sigmoid(logits))
    mae = mean_absolute_error(labels/3., np_sigmoid(logits))
    return {"mse": mse, "mae": mae}


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        labels = labels / 3.0
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = sigmoid(logits)
        loss = torch.mean(torch.square(logits.squeeze() - labels.squeeze()))
        return (loss, outputs) if return_outputs else loss


def sigmoid(z):
    return 1/(1+torch.exp(-z))


def np_sigmoid(z):
    return 1/(1+np.exp(-z))

def training_loop():
    train_dataset, eval_dataset = get_train_eval_datasets()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        learning_rate=LR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        save_steps=50,
        report_to="wandb",
        deepspeed="ds_config.json",
    )

    # log_callback = ValidationCallback(eval_dataset)

    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_regression_metrics,
        # callbacks=[log_callback],
    )

    trainer.train()
    trainer.save_model('/tmp/gptneo1p3B-large-wills-loss-function-tr-lr1e-6-seed0')


reduce_randomness()


wandb.login(key=WANDB_KEY)

if __name__ == "__main__":
    training_loop()
    model.push_to_hub('Jellywibble/gptneo1p3B-large-wills-loss-function-tr-lr1e-6-seed0')
    tokenizer.push_to_hub('Jellywibble/gptneo1p3B-large-wills-loss-function-tr-lr1e-6-seed0')
