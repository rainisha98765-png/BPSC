"""
Accuracy check for questions_combined.csv, using a manually-reviewed
random sample + a Subject-level confusion matrix.

A confusion matrix needs ground truth to compare against. Nothing in the
pipeline has ever recorded a "correct" Subject/Topic, so step 1 below is
where that ground truth gets created -- by you, marking a sample by hand.

Usage:
    python check_accuracy.py sample
        Draws a stratified random sample from questions_combined.csv and
        writes review_sample.csv for you to fill in by hand.

    python check_accuracy.py audit
        Reads your filled-in review_sample.csv and reports accuracy %,
        a Subject-level confusion matrix (saved to confusion_matrix.csv),
        and per-subject precision/recall/F1.

You can also just run `python check_accuracy.py` with no argument -- it
falls back to MODE below.
"""

import csv
import random
import sys
from collections import defaultdict, Counter
from typing import Dict, List


MODE = "sample"  # "sample" | "audit"

COMBINED_FILE = "questions_combined.csv"
REVIEW_FILE = "review_sample.csv"
CONFUSION_MATRIX_FILE = "confusion_matrix.csv"

SAMPLE_SIZE = 150       # total rows to pull for manual review
MIN_PER_SUBJECT = 3     # floor, so rare subjects still get reviewed
RANDOM_SEED = 42        # fixed seed -> same sample every time you draw one


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MODE 1: DRAW A STRATIFIED SAMPLE
# ============================================================

def cmd_sample() -> None:
    rows = read_csv(COMBINED_FILE)
    classified = [r for r in rows if r.get("Subject", "").strip()]

    by_subject: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in classified:
        by_subject[r["Subject"]].append(r)

    random.seed(RANDOM_SEED)
    selected: List[Dict[str, str]] = []
    selected_ids = set()

    # Floor: every subject present gets at least MIN_PER_SUBJECT rows (or
    # all of its rows, if it has fewer than that) -- otherwise a random
    # sample of 150 out of 3000+ rows will barely touch your rarer
    # subjects (Essay -- Mains, Language -- Mains, etc.).
    for subject, subject_rows in by_subject.items():
        take = min(MIN_PER_SUBJECT, len(subject_rows))
        for r in random.sample(subject_rows, take):
            selected.append(r)
            selected_ids.add(r["Question ID"])

    # Fill remaining slots from the rest, proportional to how common each
    # subject actually is.
    remaining_slots = max(0, SAMPLE_SIZE - len(selected))
    remaining_pool = [r for r in classified if r["Question ID"] not in selected_ids]
    if remaining_slots and remaining_pool:
        take = min(remaining_slots, len(remaining_pool))
        selected.extend(random.sample(remaining_pool, take))

    random.shuffle(selected)

    fieldnames = [
        "Question ID", "Questions",
        "Option A", "Option B", "Option C", "Option D",
        "Predicted Subject", "Predicted Topic", "Predicted Subtopic",
        "Subject Correct (Y/N)", "Correct Subject (if N)",
        "Topic Correct (Y/N)", "Correct Topic (if N)",
        "Notes",
    ]

    out_rows = [{
        "Question ID": r["Question ID"],
        "Questions": r["Questions"],
        "Option A": r["Option A"],
        "Option B": r["Option B"],
        "Option C": r["Option C"],
        "Option D": r["Option D"],
        "Predicted Subject": r["Subject"],
        "Predicted Topic": r["Topic"],
        "Predicted Subtopic": r["Subtopic"],
        "Subject Correct (Y/N)": "",
        "Correct Subject (if N)": "",
        "Topic Correct (Y/N)": "",
        "Correct Topic (if N)": "",
        "Notes": "",
    } for r in selected]

    write_csv(REVIEW_FILE, out_rows, fieldnames)

    print("=" * 60)
    print("SAMPLE")
    print("=" * 60)
    print(f"Classified rows available: {len(classified)}")
    print(f"Subjects represented: {len(by_subject)}")
    print(f"Sample drawn: {len(out_rows)} rows -> {REVIEW_FILE}")
    print()
    print("Next: open review_sample.csv (Excel/Sheets), and for every row:")
    print("  - put Y or N in 'Subject Correct (Y/N)'")
    print("    if N, put the right Subject in 'Correct Subject (if N)'")
    print("  - put Y or N in 'Topic Correct (Y/N)'")
    print("    if N, put the right Topic in 'Correct Topic (if N)'")
    print("  - leave a row's Y/N columns BLANK to skip it (won't count)")
    print("Then run:  python check_accuracy.py audit")


# ============================================================
# MODE 2: AUDIT THE FILLED-IN SAMPLE
# ============================================================

def cmd_audit() -> None:
    rows = read_csv(REVIEW_FILE)

    subject_confusion: Dict[str, Counter] = defaultdict(Counter)
    subject_reviewed = 0
    subject_correct = 0
    subject_skipped_no_truth = 0

    topic_reviewed = 0
    topic_correct = 0

    for row in rows:
        predicted_subject = row.get("Predicted Subject", "").strip()
        subj_flag = row.get("Subject Correct (Y/N)", "").strip().upper()

        if subj_flag in ("Y", "N"):
            subject_reviewed += 1
            if subj_flag == "Y":
                true_subject = predicted_subject
                subject_correct += 1
            else:
                true_subject = row.get("Correct Subject (if N)", "").strip()
                if not true_subject:
                    subject_skipped_no_truth += 1
                    true_subject = None
            if true_subject:
                subject_confusion[true_subject][predicted_subject] += 1

        topic_flag = row.get("Topic Correct (Y/N)", "").strip().upper()
        if topic_flag in ("Y", "N"):
            topic_reviewed += 1
            if topic_flag == "Y":
                topic_correct += 1

    print("=" * 60)
    print("AUDIT")
    print("=" * 60)

    if subject_reviewed == 0:
        print(f"No reviewed rows found in {REVIEW_FILE}.")
        print("Fill in 'Subject Correct (Y/N)' / 'Topic Correct (Y/N)' first.")
        return

    subject_accuracy = 100 * subject_correct / subject_reviewed
    print(f"Subject-level: {subject_correct}/{subject_reviewed} correct "
          f"({subject_accuracy:.1f}% accuracy)")
    if subject_skipped_no_truth:
        print(
            f"  ({subject_skipped_no_truth} row(s) marked N with no "
            f"'Correct Subject' filled in -- excluded from the confusion "
            f"matrix, still counted as wrong above)"
        )

    if topic_reviewed:
        topic_accuracy = 100 * topic_correct / topic_reviewed
        print(f"Topic-level:   {topic_correct}/{topic_reviewed} correct "
              f"({topic_accuracy:.1f}% accuracy)")

    # --- confusion matrix -------------------------------------------------
    all_subjects = sorted(
        set(subject_confusion.keys())
        | {p for preds in subject_confusion.values() for p in preds}
    )

    matrix_rows = []
    for true_subject in all_subjects:
        row_out = {"True Subject \\ Predicted": true_subject}
        for pred_subject in all_subjects:
            row_out[pred_subject] = subject_confusion[true_subject].get(pred_subject, 0)
        matrix_rows.append(row_out)

    write_csv(
        CONFUSION_MATRIX_FILE,
        matrix_rows,
        fieldnames=["True Subject \\ Predicted"] + all_subjects,
    )
    print()
    print(f"Confusion matrix ({len(all_subjects)} subjects) saved -> "
          f"{CONFUSION_MATRIX_FILE}")

    # --- per-subject precision / recall / F1 -------------------------------
    print()
    print("Per-subject precision / recall / F1 (subjects seen in the sample):")
    print(f"  {'Subject':<32} {'Prec':>6} {'Recall':>7} {'F1':>6}  (n true)")
    for subject in all_subjects:
        tp = subject_confusion[subject].get(subject, 0)
        fn = sum(v for k, v in subject_confusion[subject].items() if k != subject)
        fp = sum(
            subject_confusion[other].get(subject, 0)
            for other in all_subjects if other != subject
        )
        n_true = tp + fn
        if n_true == 0 and (tp + fp) == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        flag = "  <-- weak" if n_true >= 3 and (precision < 0.7 or recall < 0.7) else ""
        print(f"  {subject:<32} {precision:6.2f} {recall:7.2f} {f1:6.2f}  ({n_true}){flag}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    mode = mode.strip().lower()

    if mode == "sample":
        cmd_sample()
    elif mode == "audit":
        cmd_audit()
    else:
        print(f"Unknown mode '{mode}'. Choose 'sample' or 'audit'.")
        sys.exit(1)


if __name__ == "__main__":
    main()