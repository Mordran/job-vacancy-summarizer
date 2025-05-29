# Job Description Summarization with Transformers

This project focuses on summarizing job descriptions using a transformer-based language model to generate concise summaries of technical roles in the data field. The process includes data collection, preprocessing, training a summarization model, and evaluating its performance with semantic metrics.

## Dataset

Job data was collected using the [`python-jobspy`](https://pypi.org/project/python-jobspy/) Python library, which allows structured scraping from job platforms. In this case, job postings were gathered for roles in:

- Data Science  
- Data Engineering  
- Data Analysis  

from the following platforms:

- LinkedIn  
- Glassdoor  
- Indeed  

> ⚠️ Out of respect for the platforms' terms of use, the `scrapped_jobs.csv` file is not included in this repository. You can generate your own data by running `scrape_jobs.py`.

## Preprocessing

The raw job descriptions were cleaned by removing HTML, emails, URLs, marketing noise, special characters, and irrelevant metadata. The cleaned dataset was saved as `cleaned_jobs.csv`.

## Summarization

We fine-tuned the `sshleifer/distilbart-cnn-12-6` model using the job descriptions as both input and pseudo-summary (self-supervised learning). After training, summaries were generated and stored in `job_descriptions_with_summaries.csv`.

### Example

**Before:**  
> full stack engineer mandatory: strong proficiency in front end development technologies (react, angular, vue.js) solid understanding of back end development languages (java, python, node.js) experience with relational databases (mysql, postgresql) and nosql databases (mongodb, cassandra) knowledge of cloud platforms (aws, azure, gcp) familiarity with devops practices and tools (ci cd pipelines, containerization, infrastructure as code) understanding of software development methodologies (agile, scrum) strong problem solving and analytical skills excellent communication and collaboration skills nice to have: experience with microservices architecture knowledge of security best practices proficiency in testing frameworks (jest, mocha, cypress) experience with data engineering and data science concepts familiarity with machine learning and ai techniques key responsibilities: design, develop, and maintain scalable and high performance web applications collaborate with cross functional teams to understand business requirements and translate them into technical solutions write clean, well structured, and maintainable code perform unit, integration, and end to end testing to ensure code quality deploy and maintain applications in production environments troubleshoot and resolve technical issues stay up to date with the latest technologies and industry trends contribute to a culture of continuous improvement and innovation location: toluca cdmx hybrid scheme 3 days per week green credit bureau report.

**After:**  
> full stack engineer mandatory: strong proficiency in front end development technologies (react, angular, vue.js) solid understanding of back end development languages (java, python, node.js), experience with relational databases (mysql, postgresql) and nosql databases (mongodb, cassandra) knowledge of cloud platforms (aws, azure, gcp) familiarity with devops practices and tools (ci cd pipelines, containerization, infrastructure as code) understanding of software development methodologies (agile, scrum) strong problem solving and analytical skills excellent communication and collaboration skills nice to have

## Evaluation

Two evaluation methods were used to assess summary quality:

- **BERTScore F1**: 0.89  
- **NLI Entailment Score**: 0.4093  
  *(Model: `roberta-large-mnli`)*

While the high BERTScore indicates strong semantic similarity between the generated summaries and the original descriptions, the relatively low NLI entailment score suggests that coherence and logical consistency may be lacking in some summaries.

## Repository Structure

├── scrape_jobs.py # Scrapes job listings and saves to CSV

├── preprocess_jobs.py # Cleans and filters job descriptions

├── summarize.py # Trains model and generates summaries

├── evaluate.py # Evaluates summaries using BERTScore and NLI

└── README.md

## Requirements

- Python ≥ 3.8  
- Transformers (Hugging Face)  
- Datasets  
- Evaluate  
- PyTorch  
- tqdm  

