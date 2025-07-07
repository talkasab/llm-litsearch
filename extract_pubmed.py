import asyncio
import csv
import json
import os
import random
from typing import Literal
import zipfile

import dotenv
import instructor
import openai
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ArticleInfo(BaseModel):
    pmid: str = Field(description="Medline PMID of the article, if available")
    title: str = Field(description="Title of the article")
    about_imaging_followup_recommendations : bool | Literal['unknown'] = Field(description="Does the article discuss imaging follow-up recommendations? True or False; if you can't tell from the information, say 'unknown'")
    has_abstract: bool | Literal['unknown'] = Field(description="Does the article have an abstract? True or False; if you can't tell from the information, say 'unknown'")
    about_breast_imaging_followup: bool | Literal['unknown'] = Field(description="Does the article discuss breast imaging follow-up recommendations? True or False; if you can't tell from the information, say 'unknown'")
    followup_guideline: bool | Literal['unknown'] = Field(description="Does the article discuss a follow-up guideline? True or False; if you can't tell from the information, say 'unknown'")
    management_guideline: bool | Literal['unknown'] = Field(description="Does the article discuss a management guideline? True or False; if you can't tell from the information, say 'unknown'")
    followup_detection: bool | Literal['unknown'] = Field(description="Does the article discuss follow-up detection? True or False; if you can't tell from the information, say 'unknown'")
    system_or_program: bool | Literal['unknown'] = Field(description="Does the article discuss a system or program to improve follow-up? True or False; if you can't tell from the information, say 'unknown'")
    influencing_factors: bool | Literal['unknown'] = Field(description="Does the article describe factors, other those in the radiology report, that influence completion or other outcomes of follow up recommendations? True or False; if you can't tell from the information, say 'unknown'")

articles: dict[str, str] = {}

pmids = list(articles.keys())

# for article in get_random_articles(5):
#     print(article_text(article))

PROMPT = """You are a medical expert. You are given information about an article in a medical journal. 
Based on the information given, please answer the following questions: 
- Is the article about imaging follow-up recommendations, including the identification, communication, tracking, management, and outcomes of these recommendations? For this question, synonyms for follow up recommendations include incidental findings in radiology or other "clinically significant" findings/results in imaging exams.  
- Does the article have an abstract? 
- Is the article primarily about follow-up in breast imaging, such as mammography, breast ultrasound, or breast MRI? 
- If the article is about imaging follow-up, please answer the following questions: 
  - *follow-up guideline*: Does the article describe what the appropriate follow-up is for a particular imaging finding? 
  - *management guideline*: Does the article describe what the appropriate follow-up is for a particular diagnosis or following a particular therapy? 
  - *follow-up detection*: Does the article describe how do to identify a follow up recommendation in a radiology report? 
  - *system or program*: Does the article describe a system or program to improve adherence to follow-up recommendations from radiology reports?
  - *influencing factors*: Does the article describe factors, other those in the radiology report, that influence completion or other outcomes of follow up recommendations? 
"""

INSTRUCTIONS = """Answer the questions based ONLY on the given information. If you can't tell from the information, say 'unknown'."""

def generate_prompt(article_pmid: str) -> str:
    article_text_content = articles[article_pmid]
    return f"{PROMPT}\n\n{INSTRUCTIONS}\n\n{article_text_content}"


async def extract_study_info(llm: instructor.AsyncInstructor, article_pmid: str, semaphore: asyncio.Semaphore) -> ArticleInfo:
    async with semaphore:
        prompt = generate_prompt(article_pmid)
        response = await llm.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_model=ArticleInfo,
        )
        return response

def write_csv_file(articles: list[ArticleInfo], filename: str):
    with open(filename, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=articles[0].model_dump().keys())
        writer.writeheader()
        for info in articles:
            writer.writerow(info.model_dump())

async def main():
    CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", 25))  # Adjust this number based on your needs and API limits
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", None)
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", 0))  # Default to 100 if not set

    # Load articles from the zip file
    articles_zip = zipfile.ZipFile("data/entries_with_abstracts.zip", "r")
    for filename in articles_zip.namelist():
        pmid = filename[0:-4]
        with articles_zip.open(filename) as f:
            articles[pmid] = f.read().decode("utf-8")
    print(f"Loaded {len(articles)} articles from zipfile.")

    # Do the extractions
    llm = instructor.from_openai(openai.AsyncOpenAI())
    selected_pmids = random.sample(list(articles.keys()), SAMPLE_SIZE) if SAMPLE_SIZE > 0 else articles.keys()
    print(f"Selected {len(selected_pmids)} articles for processing")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)  # Limit concurrent requests
    tasks = [extract_study_info(llm, pmid, semaphore) for pmid in selected_pmids]
    results = await tqdm.gather(*tasks)

    # Write results to file or print them
    written = False
    if OUTPUT_FILE:
        if OUTPUT_FILE.endswith(".json"):
            with open(OUTPUT_FILE, "w") as f:
                json.dump([result.model_dump(exclude_none=True) for result in results], f, indent=2)
            written = True
        elif OUTPUT_FILE.endswith(".csv"):
            write_csv_file([result for result in results if result is not None], OUTPUT_FILE)
            written = True
    if written:
        print(f"Results written to {OUTPUT_FILE}")
    else:
        for result in results:
            print(result.model_dump_json(exclude_none=True, indent=2))
        if OUTPUT_FILE:
            print(f"Results not written to {OUTPUT_FILE} due to unsupported format. Please use .json or .csv.")

if __name__ == "__main__":
    asyncio.run(main())
