#!/usr/bin/env python3

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_jobs.csv", engine='python')
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

dataset_train = Dataset.from_pandas(train_df.reset_index(drop=True))
dataset_val = Dataset.from_pandas(val_df.reset_index(drop=True))

checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

max_input_length = 1024
max_target_length = 128

def preprocess(batch):
    inputs = tokenizer(batch["description"], padding="max_length", truncation=True, max_length=max_input_length)
    targets = tokenizer(batch["description"], padding="max_length", truncation=True, max_length=max_target_length)
    labels = np.array(targets["input_ids"])
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels.tolist()
    }

dataset_train = dataset_train.map(preprocess, batched=True, remove_columns=["id", "description"])
dataset_val = dataset_val.map(preprocess, batched=True, remove_columns=["id", "description"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

from tqdm.auto import tqdm
summaries = []
for text in tqdm(df["description"], desc="Generando todos"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length, padding="longest").to(model.device)
    summary_ids = model.generate(**inputs, max_length=128, num_beams=4)
    summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

df["generated_summary"] = summaries
df.to_csv("job_descriptions_with_summaries.csv", index=False)
