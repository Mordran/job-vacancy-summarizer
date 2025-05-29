#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import torch
from transformers import AutoModelForSequenceClassification

df = pd.read_csv("job_descriptions_with_summaries.csv", engine='python')
val_texts = df["description"].tolist()
val_preds = df["generated_summary"].tolist()

max_input_length = 1024

bertscore = evaluate.load("bertscore")
bs = bertscore.compute(predictions=val_preds, references=val_texts, lang="en")
avg_bs_f1 = np.mean(bs["f1"])
print(f"Validation BERTScore F1: {avg_bs_f1:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)

entail_scores = []
for premise, hypothesis in tqdm(zip(val_texts, val_preds), desc="Calculando NLI", total=len(val_texts)):
    enc = nli_tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)
    with torch.no_grad():
        logits = nli_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    entail_prob = probs[2]
    entail_scores.append(entail_prob)

avg_entail = np.mean(entail_scores)
print(f"Validation NLI entailment score: {avg_entail:.4f}")
