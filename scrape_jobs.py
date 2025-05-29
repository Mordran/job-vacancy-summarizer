#!/usr/bin/env python3
# scrape_jobs.py

"""
Scrapes job listings from LinkedIn, Indeed, and Glassdoor using the jobspy library.
Stores results in a CSV file, avoiding duplicates and retrying failed requests.

Requirements:
- jobspy
- pandas

Author: Mordran
Date: 2025
"""

import pandas as pd
import time
import random
import os
from jobspy import scrape_jobs

# Constants
CSV_FILE = "scrapped_jobs.csv"
SEARCH_TERMS = ["data science", "data analyst", "data engineer"]
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def load_existing_jobs(csv_file):
    if os.path.exists(csv_file):
        print(f"Loading existing jobs from {csv_file}...")
        return pd.read_csv(csv_file)
    else:
        print("No existing jobs file found. Starting from scratch...")
        return pd.DataFrame()


def scrape_and_save_jobs():
    existing_jobs_df = load_existing_jobs(CSV_FILE)

    for search_term in SEARCH_TERMS:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                print(f"Scraping jobs for search term: '{search_term}'...")

                jobs = scrape_jobs(
                    site_name=["indeed", "glassdoor", "linkedin"],
                    search_term=search_term,
                    location="Mexico",
                    results_wanted=300,
                    linkedin_fetch_description=True,
                    country_indeed="Mexico"
                )

                print(f"Found {len(jobs)} new jobs for '{search_term}'")

                new_jobs_df = pd.DataFrame(jobs)
                new_jobs_df["search_term"] = search_term

                if not existing_jobs_df.empty:
                    all_jobs_df = pd.concat(
                        [existing_jobs_df, new_jobs_df]
                    ).drop_duplicates(subset=["id"], keep="first")
                else:
                    all_jobs_df = new_jobs_df

                all_jobs_df.to_csv(CSV_FILE, index=False)
                print(f"Saved {len(all_jobs_df)} jobs to {CSV_FILE}")

                existing_jobs_df = all_jobs_df

                sleep_time = random.uniform(30, 60)
                print(f"Sleeping for {sleep_time:.2f} seconds before the next search term...")
                time.sleep(sleep_time)

                break  # Success, exit retry loop

            except Exception as e:
                retries += 1
                print(f"Attempt {retries} failed for '{search_term}': {e}")
                if retries < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Max retries reached for '{search_term}'. Skipping...")


if __name__ == "__main__":
    scrape_and_save_jobs()
