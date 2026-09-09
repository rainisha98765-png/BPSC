import csv
import os
import re
from typing import Dict, List


QUESTION_FILE = "questions_check.csv"
CLASSIFIED_FILE = "questions_classified.csv"
COMBINED_FILE = "questions_combined.csv"


def load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def qid_key(row: Dict[str, str]) -> str:
    return str(row.get("Question ID", "")).strip()


def sort_key(row: Dict[str, str]):
    m = re.search(r"\d+", qid_key(row))
    return int(m.group()) if m else 10**9


def combine_files() -> None:
    questions = load_csv(QUESTION_FILE)
    classifications = load_csv(CLASSIFIED_FILE)

    if not questions:
        raise ValueError(f"{QUESTION_FILE} is empty.")
    if not classifications:
        raise ValueError(f"{CLASSIFIED_FILE} is empty.")

    classified_by_id = {
        qid_key(row): row
        for row in classifications
        if qid_key(row)
    }

    combined = []

    for question in questions:
        qid = qid_key(question)
        classification = classified_by_id.get(qid, {})

        # Start with every original question/source column.
        row = dict(question)

        # Add/update the generated classification columns.
        row["Subject"] = classification.get("Subject", "")
        row["Topic"] = classification.get("Topic", "")
        row["Subtopic"] = classification.get("Subtopic", "")

        # Helpful status when a question has no classification yet.
        row["Classification Status"] = (
            "CLASSIFIED" if qid in classified_by_id else "MISSING"
        )

        combined.append(row)

    combined.sort(key=sort_key)

    # Preserve the original question-file column order first.
    original_fields = list(questions[0].keys())

    classification_fields = [
        "Subject",
        "Topic",
        "Subtopic",
        "Classification Status",
    ]

    extra_fields = []
    for row in combined:
        for field in row.keys():
            if field not in original_fields + classification_fields:
                extra_fields.append(field)

    fieldnames = original_fields + classification_fields + extra_fields

    with open(COMBINED_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(combined)

    classified_count = sum(
        1 for row in combined
        if row["Classification Status"] == "CLASSIFIED"
    )
    missing_count = len(combined) - classified_count

    print("=" * 75)
    print("COMBINE COMPLETE")
    print("=" * 75)
    print(f"Questions in source:       {len(questions)}")
    print(f"Classifications available: {len(classified_by_id)}")
    print(f"Rows in combined file:     {len(combined)}")
    print(f"Classified rows:           {classified_count}")
    print(f"Missing classifications:   {missing_count}")
    print(f"Created:                   {COMBINED_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    combine_files()
