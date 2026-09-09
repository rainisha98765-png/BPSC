import json
import re
import csv


# ============================================================
# STEP 1: LOAD BPSC JSON  (existing "pdf_info" format)
# ============================================================

with open("BPSC.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pages = data["pdf_info"]

print("Number of pages (BPSC.json):", len(pages))


# ============================================================
# STEP 2: EXTRACT TEXT FROM BLOCKS  (BPSC.json format)
# ============================================================

def get_text(block):
    text = ""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text += span.get("content", "") + " "
    if not text and block.get("blocks"):
        text = "\n".join(get_text(sub) for sub in block["blocks"])
    return text.strip()


combined_lines = []

for page in pages:
    for block in page["para_blocks"]:
        text = get_text(block)
        if text:
            combined_lines.extend(text.split("\n"))

print("Lines from BPSC.json:", len(combined_lines))


# ============================================================
# STEP 2b: PARSE "content_list_v2" FILES AT BLOCK LEVEL
#
# Unlike BPSC.json, v2 files carry rich structure that lets us
# group Answer + Explanation together. We process them into a
# list of tagged segments:
#
#   {"kind": "line",        "text": "..."}   <- question / option
#   {"kind": "answer",      "text": "..."}   <- full answer line
#   {"kind": "explanation", "text": "..."}   <- explanation body
#
# The BPSC.json lines are wrapped as "line" segments so everything
# flows through a single unified parser in Step 4.
# ============================================================

ADDITIONAL_FILES = [
    "B71EnglishPart2sub1_content_list_v2.json",
    "B71EnglishPart2sub2_content_list_v2.json",
    "B71EnglishPart3sub1_content_list_v2.json",
    "B71EnglishPart3sub2_content_list_v2.json",
    "B71EnglishPart4sub1_content_list_v2.json",
    "B71EnglishPart4sub2_content_list_v2.json",
]

SKIP_BLOCK_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "image",
}

# Regex patterns (compiled once)
answer_line_re  = re.compile(r"^Answer\s*[-–]\s*", re.I)
num_re          = re.compile(r"^\d{1,3}\.\s")
option_re       = re.compile(r"^\([A-D]\)\s*")


def _collect_texts(node):
    """Recursively pull every {"type":"text","content":"..."} string."""
    texts = []
    if isinstance(node, dict):
        for value in node.values():
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        texts.append(item.get("content", ""))
                    else:
                        texts.extend(_collect_texts(item))
            elif isinstance(value, dict):
                texts.extend(_collect_texts(value))
    return texts


def _block_text(block):
    content = block.get("content")
    if not isinstance(content, dict):
        return ""
    texts = _collect_texts(content)
    return "\n".join(t for t in texts if t)


def _is_question_or_option(text):
    """True if this text looks like a question stem or option line."""
    first_line = text.split("\n")[0].strip()
    return bool(num_re.match(first_line) or option_re.match(first_line))


def parse_v2_file(filename):
    """
    Return a list of tagged segments from one content_list_v2 file.

    Each segment is a dict:
        {"kind": "line"|"answer"|"explanation", "text": str}

    Strategy
    --------
    We scan blocks in reading order.  The moment we see a block
    whose first meaningful text matches "Answer - ...", we mark it
    as "answer".  Every subsequent block — until we hit the next
    question stem or option block — is tagged "explanation".
    """
    with open(filename, "r", encoding="utf-8") as f:
        v2_pages = json.load(f)

    segments = []
    in_explanation = False

    for v2_page in v2_pages:
        for block in v2_page:
            if block.get("type") in SKIP_BLOCK_TYPES:
                continue

            text = _block_text(block)
            if not text:
                continue

            first_line = text.split("\n")[0].strip()

            # ------------------------------------------------
            # Answer block
            # ------------------------------------------------
            if answer_line_re.match(first_line):
                segments.append({"kind": "answer", "text": text})
                in_explanation = True
                continue

            # ------------------------------------------------
            # New question / option → end of explanation zone
            # ------------------------------------------------
            if _is_question_or_option(first_line):
                in_explanation = False

            # ------------------------------------------------
            # Explanation block (anything after an answer line
            # until the next question)
            # ------------------------------------------------
            if in_explanation:
                segments.append({"kind": "explanation", "text": text})
                continue

            # ------------------------------------------------
            # Normal content (question stem, options, headers…)
            # ------------------------------------------------
            segments.append({"kind": "line", "text": text})

    return segments


# ============================================================
# STEP 3: BUILD UNIFIED SEGMENT LIST
#
# BPSC.json lines are wrapped as {"kind":"line"} segments first,
# then v2 segments are appended in reading order.
# ============================================================

segments = [{"kind": "line", "text": t} for t in combined_lines]

for filename in ADDITIONAL_FILES:
    before = len(segments)
    new_segs = parse_v2_file(filename)
    segments.extend(new_segs)
    print(f"Segments from {filename}: {len(segments) - before}")

print("Total segments (all sources):", len(segments))


# ============================================================
# STEP 4: EXAM HEADER DETECTION
# ============================================================

def is_exam_header(line):
    clean = re.sub(r"<[^>]+>", "", line).strip()

    pattern = re.compile(
        r"^("
        r"\d{2,3}(?:st|nd|rd|th)?"
        r"(?:\s*-\s*\d{2,3}(?:st|nd|rd|th)?)?"
        r"\s*"
        r"(?:RE-?EXAM\s*|CANCELLED\s*)?"
        r"BPSC"
        r")",
        re.I
    )
    match = pattern.match(clean)
    if match:
        return match.group(1).strip()

    match = re.match(r"^(CDPO\s*\([^)]*\))", clean, re.I)
    if match:
        return match.group(1).strip()

    return None


# ============================================================
# STEP 5: ANSWER PARSING HELPERS
# ============================================================

# Captures the letter from:  Answer - (C) Some text
# or the full note from:      Answer - This question was removed...
answer_letter_re = re.compile(
    r"^Answer\s*[-–]\s*\(?([A-D])\)?",
    re.I
)
answer_full_re = re.compile(
    r"^Answer\s*[-–]\s*(.*)",
    re.I
)


def parse_answer_segment(text):
    """
    Returns (letter, answer_text) from an answer segment.

    letter      → "A"/"B"/"C"/"D" or "" if not a lettered answer
    answer_text → the full answer string (letter + label, or the
                  full note for non-standard answers like removals)
    """
    first = text.split("\n")[0].strip()
    first_clean = re.sub(r"<[^>]+>", "", first).strip()

    m_letter = answer_letter_re.match(first_clean)
    m_full   = answer_full_re.match(first_clean)

    letter      = m_letter.group(1).upper() if m_letter else ""
    answer_text = m_full.group(1).strip()   if m_full   else first_clean

    return letter, answer_text


# ============================================================
# STEP 6: QUESTION EXTRACTION  (single pass over segments)
# ============================================================

current_exam   = "71st BPSC (main)"
questions      = []
current        = None
pending_stem   = []
collecting_stem = False

num_pattern    = re.compile(r"^\d{1,3}\.\s")
option_pattern = re.compile(r"^\(([A-D])\)\s*(.*)")

# For BPSC.json lines that carry the answer inline
answer_pattern_inline = re.compile(
    r"^Answer\s*[-–]\s*\(?([A-D])\)?",
    re.I
)


for seg in segments:

    kind = seg["kind"]
    text = seg["text"]

    # --------------------------------------------------------
    # ANSWER segment  (tagged by parse_v2_file)
    # --------------------------------------------------------
    if kind == "answer":
        if current:
            letter, ans_text = parse_answer_segment(text)
            if not current["answer"]:
                current["answer"]      = letter
                current["answer_text"] = ans_text
        continue

    # --------------------------------------------------------
    # EXPLANATION segment
    # --------------------------------------------------------
    if kind == "explanation":
        if current:
            current["explanation_parts"].append(text)
        continue

    # --------------------------------------------------------
    # LINE segment  (from BPSC.json or v2 non-answer blocks)
    # --------------------------------------------------------
    line = text.strip()

    for raw_line in line.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # -- Exam header --
        header = is_exam_header(raw_line)
        if header:
            if current:
                questions.append(current)
                current = None
            current_exam    = header
            pending_stem    = []
            collecting_stem = False
            continue

        # -- Option --
        option_match = option_pattern.match(raw_line)
        if option_match:
            opt_letter = option_match.group(1)
            opt_text   = option_match.group(2)

            if opt_letter == "A":
                if current:
                    questions.append(current)
                current = {
                    "exam":             current_exam,
                    "stem":             " ".join(pending_stem),
                    "options":          {},
                    "answer":           "",
                    "answer_text":      "",
                    "explanation_parts": [],
                }
                current["options"]["A"] = opt_text
                pending_stem    = []
                collecting_stem = False
            elif current:
                current["options"][opt_letter] = opt_text
            continue

        # -- Inline answer (BPSC.json style) --
        ans_inline = answer_pattern_inline.match(raw_line)
        if ans_inline and current:
            if not current["answer"]:
                current["answer"] = ans_inline.group(1).upper()
                # Build answer_text from the raw line
                m = answer_full_re.match(raw_line)
                current["answer_text"] = m.group(1).strip() if m else raw_line
            continue

        # -- New question number --
        if num_pattern.match(raw_line) and not collecting_stem:
            pending_stem    = [raw_line]
            collecting_stem = True
            continue

        # -- Stem continuation --
        if collecting_stem:
            pending_stem.append(raw_line)
            continue

        # -- Everything else: ignore --


# ============================================================
# STEP 7: SAVE LAST QUESTION
# ============================================================

if current:
    questions.append(current)


# ============================================================
# STEP 8: CLEAN STEMS AND JOIN EXPLANATIONS
# ============================================================

for q in questions:
    q["stem"] = re.sub(r"^\d{1,3}\.\s*", "", q["stem"]).strip()

    # Strip HTML tags from answer_text
    q["answer_text"] = re.sub(r"<[^>]+>", "", q.get("answer_text", "")).strip()

    # Join explanation paragraphs with a separator
    raw_parts = q.pop("explanation_parts", [])
    q["explanation"] = "\n\n".join(
        re.sub(r"<[^>]+>", "", part).strip()
        for part in raw_parts
        if part.strip()
    )


print("Questions found:", len(questions))


# ============================================================
# STEP 9: BUILD OUTPUT ROWS
# ============================================================

output_rows = []

for index, q in enumerate(questions, start=1):
    # Combine letter + label into one field, e.g. "(C) STETHOSCOPE"
    # Fall back to answer_text alone for non-lettered answers (e.g. removed questions)
    letter    = q["answer"]
    ans_text  = q["answer_text"]
    if letter and ans_text:
        combined_answer = f"({letter}) {ans_text}"
    elif letter:
        combined_answer = f"({letter})"
    else:
        combined_answer = ans_text  # e.g. "This question was removed by BPSC..."

    output_rows.append({
        "Question ID": f"Q{index:06d}",
        "Exam":        q["exam"],
        "Questions":   q["stem"],
        "Option A":    q["options"].get("A", ""),
        "Option B":    q["options"].get("B", ""),
        "Option C":    q["options"].get("C", ""),
        "Option D":    q["options"].get("D", ""),
        "Answer":      combined_answer,       # e.g. "(C) STETHOSCOPE"
        "Explanation": q["explanation"],      # full explanation body
    })


# ============================================================
# STEP 10: SAVE CSV
# ============================================================

CSV_FIELDS = [
    "Question ID",
    "Exam",
    "Questions",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "Answer",
    "Explanation",
]

with open("questions_check.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(output_rows)


# ============================================================
# STEP 11: SUMMARY
# ============================================================

print()
print(f"Saved questions_check.csv — {len(output_rows)} rows")
print()
print("CSV fields:")
for field in CSV_FIELDS:
    print(" -", field)
print()
if output_rows:
    print("Question IDs generated:")
    print(f"  {output_rows[0]['Question ID']} → {output_rows[-1]['Question ID']}")
    # Quick stats
    with_answer = sum(1 for r in output_rows if r["Answer"])
    with_expl   = sum(1 for r in output_rows if r["Explanation"])
    print(f"\nQuestions with a letter answer : {with_answer}")
    print(f"Questions with an explanation  : {with_expl}")
print()
print("This is the raw extraction file.")
print("Classification will be performed later by Ollama.")