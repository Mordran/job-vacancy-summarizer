#!/usr/bin/env python3
# preprocess.py

import pandas as pd
import re

df = pd.read_csv('scrapped_jobs.csv')

df.drop(columns=[
    'site', 'job_url', 'job_url_direct', 'title', 'company',
    'location', 'date_posted', 'job_type', 'salary_source', 'interval',
    'min_amount', 'max_amount', 'currency', 'is_remote', 'job_level',
    'job_function', 'listing_type', 'emails', 'company_industry',
    'company_url', 'company_logo', 'company_url_direct', 'company_addresses',
    'company_num_employees', 'company_revenue', 'company_description',
    'search_term'
], inplace=True)


def clean_job_text_deep(text):
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[*•\-–—●■▶]+", " ", text)
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    text = re.sub(r"\b[A-Z]{2,}\b", " ", text)
    text = re.sub(r"\b\d{5,}\b", " ", text)
    text = re.sub(r"\b(apply now|click here|learn more|send resume|join us|hiring now)\b", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:!?() ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df['description'] = df['description'].apply(clean_job_text_deep)

df.to_csv('cleaned_jobs.csv', index=False)
