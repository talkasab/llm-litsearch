import csv
import json

fields = [
    "Title",
    "Author Names",
    "Source",
    "Abstract",
    "Author Keywords",
    "Emtree Medical Index Terms",
    "Embase Classification",
    "Medline PMID",
    "PUI",
]

with open("data/FinalEmbaseWithoutPubMedOverlap.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=fields)
    articles = [row for row in reader]

with open("data/embase_articles_full.json", "w", encoding="utf-8") as jsonfile:
    json.dump(articles[1:], jsonfile, indent=2, ensure_ascii=True)
