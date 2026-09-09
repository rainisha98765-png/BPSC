"""
classify_with_rag.py
====================
RAG-enhanced version of the classifier.
For each question it retrieves the 3 most relevant PDF chunks
from ChromaDB and injects them into the Ollama prompt as context.

Run AFTER build_rag.py has finished building the database.

Usage:
    cd bpsc_project/scripts
    python classify_with_rag.py
"""

import csv
import json
import os
import re
import time
from typing import Dict, List, Tuple

import chromadb
import ollama

# ── PATH SETUP ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RAG_DB_DIR  = os.path.join(PROJECT_DIR, "rag_db")

os.chdir(DATA_DIR)

# ── SETTINGS ─────────────────────────────────────────────────────────────────
MODEL        = "gemma3:latest"
EMBED_MODEL  = "nomic-embed-text"

INPUT_FILE   = "questions_check.csv"
OUTPUT_FILE  = "questions_classified_rag.csv"
FAILED_FILE  = "questions_failed_rag.csv"

START_QUESTION       = 1
END_QUESTION         = 3100
ONLY_QUESTIONS: List[int] = []
RETRY_FAILED_ONLY    = False
FORCE_RECLASSIFY_ALL = False

BATCH_SIZE           = 8    # smaller than before — each prompt is larger with context
RECOVERY_BATCH_SIZE  = 2
INDIVIDUAL_RETRIES   = 2
NUM_CTX              = 10000  # bumped up to fit PDF context chunks
KEEP_ALIVE           = "10m"

# RAG settings
PDF_CHUNKS_PER_Q     = 3    # how many PDF chunks to retrieve per question


# ════════════════════════════════════════════════════════════════════════════
# TAXONOMY  (same 9 subjects as classify_questions.py)
# ════════════════════════════════════════════════════════════════════════════

SUBJECT_TOPICS: Dict[str, List[str]] = {
    "History of India": [
        "Prehistoric India", "Indus Valley Civilization", "Vedic Age",
        "Mahajanapadas", "Buddhism", "Jainism", "Mauryan Empire",
        "Post-Mauryan India", "Gupta Age", "Harshavardhana",
        "South Indian Kingdoms", "Sangam Age", "Early Medieval India",
        "Delhi Sultanate", "Mughal Empire", "Bhakti Movement", "Sufi Movement",
        "Marathas", "Later Mughals", "Vijayanagara Empire", "Rajput Kingdoms",
        "European Penetration", "British Expansion",
        "British Administrative Policies", "Economic Impact of British Rule",
        "Socio-Religious Reform Movements", "Tribal and Peasant Movements",
        "Revolt of 1857", "Indian National Movement",
        "Constitutional Development", "Independence and Partition",
        "Magadha", "Mauryan Bihar", "Gupta Period in Bihar", "Pala Dynasty",
        "Sena Influence", "Sher Shah Suri", "Sasaram", "Mughal Bihar",
        "Permanent Settlement", "Indigo Cultivation", "Champaran Satyagraha",
        "Bihar's Role in 1857", "Kunwar Singh",
        "Bihar during Non-Cooperation Movement",
        "Civil Disobedience in Bihar", "Quit India Movement in Bihar",
        "Peasant Movements in Bihar", "Formation of Bihar and Orissa Province",
        "Bihar's Separation from Bengal", "Creation of Jharkhand",
        "Important Historical Personalities", "Vaishali", "Nalanda",
        "Vikramshila",
    ],
    "Geography": [
        "Earth and Universe", "Geomorphology", "Climatology", "Oceanography",
        "Soils", "Natural Vegetation", "Biodiversity and Ecosystem",
        "Physiographic Divisions of India", "Himalayan Rivers",
        "Peninsular Rivers", "Indian Monsoon", "Indian Agriculture",
        "Minerals and Mining", "Industries", "Transport and Communication",
        "Population and Census", "Migration", "Urbanisation",
        "Location and Boundaries of Bihar", "North Bihar Plains",
        "South Bihar Plateau", "Floods in Bihar", "Drought in Bihar",
        "Soil Types in Bihar", "Climate of Bihar",
        "Irrigation Projects in Bihar", "Forests of Bihar",
        "Wildlife in Bihar", "District-wise Geography of Bihar",
        "Major Infrastructure Projects in Bihar",
        "Kosi River", "Gandak River", "Sone River",
    ],
    "Indian Polity": [
        "Making of the Constitution", "Preamble", "Fundamental Rights",
        "Directive Principles", "Fundamental Duties", "Union Government",
        "Parliament", "President and Vice President",
        "Prime Minister and Cabinet", "Supreme Court", "High Courts",
        "Federalism", "Centre-State Relations", "Emergency Provisions",
        "Amendment of Constitution", "Schedules of the Constitution",
        "State Government", "State Legislature", "Local Government",
        "Panchayati Raj", "Election Commission", "UPSC and State PSCs",
        "CAG", "Finance Commission", "National Commission for SC/ST",
        "Attorney General", "Advocate General", "Constitutional Bodies",
        "Non-constitutional Bodies", "Bihar Legislature",
        "Bihar Panchayati Raj", "Bihar Administration",
        "Bihar Government Schemes", "Bihar-specific Governance",
        "Governance and Public Policy", "RTI", "Lokpal and Lokayukta",
        "Inter-State Disputes",
    ],
    "Indian Economy": [
        "Basic Economic Concepts", "National Income and GDP",
        "Economic Planning", "NITI Aayog", "Five Year Plans",
        "Agricultural Sector", "Industrial Sector", "Service Sector",
        "Major Crops and Crop Patterns", "Agricultural Reforms",
        "Land Reforms", "Green Revolution", "Banking System",
        "Reserve Bank of India", "Monetary Policy", "Fiscal Policy",
        "Union Budget", "Taxation", "GST", "External Sector",
        "Balance of Payments", "Foreign Trade Policy", "WTO", "IMF",
        "World Bank", "Human Development", "Poverty and Inequality",
        "Employment and Unemployment", "Social Sector Schemes",
        "Health Economy", "Education Economy", "Infrastructure",
        "Public Distribution System", "Economic Indices and Reports",
        "HDI", "GDP Growth", "Bihar Economy Overview", "Bihar Budget",
        "Bihar Economic Survey", "Bihar Agriculture", "Bihar Industries",
        "Bihar Employment",
    ],
    "General Science": [
        "Motion and Laws of Motion", "Work Energy and Power", "Gravitation",
        "Heat and Thermodynamics", "Light and Optics", "Sound",
        "Electricity and Magnetism", "Modern Physics", "Nuclear Physics",
        "Matter and Its Properties", "Atomic Structure", "Chemical Bonding",
        "Periodic Table", "Acids Bases and Salts", "Metals and Non-metals",
        "Carbon and Its Compounds", "Chemical Reactions", "Electrochemistry",
        "Fuels", "Cell Biology", "Genetics and Heredity", "Evolution",
        "Plant Kingdom", "Animal Kingdom", "Human Physiology",
        "Nutrition and Diseases", "Microorganisms", "Biotechnology",
        "Ecology Basics", "Science and Technology", "Space Technology",
        "Defence Technology", "Medical Science", "Inventions and Discoveries",
        "Scientific Instruments", "Computer and IT Basics",
    ],
    "Environment & Ecology": [
        "Ecosystem and Food Chain", "Biodiversity",
        "Conservation of Wildlife", "Protected Areas",
        "National Parks and Sanctuaries", "Climate Change and Global Warming",
        "Greenhouse Effect", "Ozone Layer Depletion", "Air Pollution",
        "Water Pollution", "Soil Pollution", "Noise Pollution",
        "Solid Waste Management", "Environmental Laws and Acts",
        "International Environmental Agreements", "UNFCCC and COP",
        "Paris Agreement", "Sustainable Development", "Green Economy",
        "Environmental Bodies and Organisations", "Bihar Environment and Ecology",
    ],
    "Current Affairs": [
        "National Current Affairs", "International Current Affairs",
        "Science and Technology Current Affairs",
        "Environment and Ecology Current Affairs",
        "Economy and Business Current Affairs", "Sports Current Affairs",
        "Awards and Honours", "Books and Authors",
        "Important Days and Events", "Summits and Conferences",
        "Defence and Security", "Appointments and Resignations",
        "Bihar Current Affairs",
    ],
    "Art & Culture": [
        "Indian Architecture", "Indian Sculpture", "Indian Painting",
        "Indian Music", "Indian Dance", "Indian Literature",
        "Indian Languages", "Festivals of India", "Indian Cuisine",
        "Folk Arts and Crafts", "Hinduism", "Buddhism and Jainism",
        "Islam in India", "Sikhism", "Indian Philosophy",
        "UNESCO World Heritage Sites", "ASI and Heritage Conservation",
        "Indian Museums", "Bihar Art and Architecture", "Bihar Folk Culture",
        "Madhubani Painting", "Chhau Dance", "Festivals of Bihar",
        "Bihar Literature", "Languages of Bihar", "Bihari Cuisine",
        "Famous Personalities of Bihar",
    ],
    "General Mental Ability & CSAT": [
        "Coding-Decoding", "Blood Relations", "Direction Sense",
        "Ranking and Ordering", "Alphabetical and Dictionary Order",
        "Number Series", "Letter Series", "Classification and Odd One Out",
        "Analogy", "Syllogism and Logical Reasoning", "Seating Arrangement",
        "Puzzle", "Calendar and Clock Problems", "Age Problems",
        "Time and Work", "Time Speed and Distance",
        "Profit Loss and Discount", "Simple and Compound Interest",
        "Ratio Proportion and Partnership", "Percentage", "Average",
        "Mathematical and Symbol Operations", "Reading Comprehension",
        "Decision Making", "Basic Numeracy", "Data Interpretation",
        "Venn Diagrams", "Statement and Conclusions",
        "Statement and Assumptions",
    ],
    "Miscellaneous": ["Miscellaneous"],
}

BPSC_SUBJECTS: List[str] = list(SUBJECT_TOPICS.keys())

# Ghost subject remapping — applied before validation
GHOST_SUBJECT_MAP = {
    "History of Bihar":          "History of India",
    "Bihar Polity & Governance": "Indian Polity",
    "Bihar Economy":             "Indian Economy",
    "Bihar Polity":              "Indian Polity",
    "Society":                   "Current Affairs",
    "Agriculture":               "Indian Economy",
    "Disaster Management":       "Geography",
    "Internal Security":         "Current Affairs",
    "Mathematics — Mains":       "General Mental Ability & CSAT",
    "Mathematics - Mains":       "General Mental Ability & CSAT",
    "Mathematics":               "General Mental Ability & CSAT",
    "Ethics & Integrity — Mains":"Current Affairs",
}

# Subtopics that indicate a hallucinated copy-paste
HALLUCINATED_SUBTOPICS = [
    "kosi-mechi link project", "kosi mechi link project",
    "dakshin gangotri", "satat jivikoparjan yojana",
    "national biodiversity 2022", "india sacep mou",
    "rudraksh balasaheb patil", "collins dictionary word of the year",
    "india egypt bilateral", "bhagirathi devi padma shri",
]


# ════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC HINTS  (same logic as classify_questions.py)
# ════════════════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

def _contains(text: str, phrases: List[str]) -> bool:
    n = _norm(text)
    return any(_norm(p) in n for p in phrases)

GMA_PHRASES = [
    "coding decoding", "coding-decoding", "substitution cipher",
    "code language", "code breaking", "coded as",
    "blood relation", "blood relations", "family relationship",
    "direction sense", "facing north", "facing south",
    "turned left", "turned right", "walks north", "how far",
    "ranking", "from the top", "from the bottom", "in a row",
    "alphabetical order", "dictionary order", "arrange the letters",
    "number series", "letter series", "next number", "missing number",
    "next in the series", "odd one out", "which is different",
    "does not belong", "find the odd", "analogy", " is to ",
    "same relationship", "syllogism", "conclusions follow",
    "seating arrangement", "sits between", "sits to the left",
    "calendar problem", "clock problem", "what day", "day of the week",
    "if + means", "if - means", "if × means", "if ÷ means",
    "means multiplication", "means addition", "operators are interchanged",
    "times as old", "as old as", "years hence", "present age",
    "cost price", "selling price", "profit percent", "loss percent",
    "simple interest", "compound interest",
    "average speed", "km per hour", "km/h",
    "sum of the numbers", "difference of the numbers",
    "product of two numbers", "value of x", "value of y",
    "pipes and cisterns", "pipe can fill",
    "dearness allowance", "basic salary",
    "time and work", "can complete the work",
    "speed of the train", "upstream", "downstream",
    "discount percent", "lcm", "hcf",
    "venn diagram", "statement and conclusion",
]

GMA_EXCLUSION_PHRASES = [
    "censure motion", "no-confidence", "presiding officer",
    "non-sense codon", "codon", "isotope", "polio",
    "preamble", "socialist", "secular", "sovereign",
    "arjuna award", "dadasaheb phalke", "bharat ratna",
    "nobel prize", "member of nato", "member of quad",
]

CURRENT_AFFAIRS_PHRASES = [
    "2022", "2023", "2024", "2025", "2026",
    "recently appointed", "recently launched", "recently declared",
    "g20 summit", "g7 summit", "cop28", "cop29", "baku",
    "nobel prize", "padma vibhushan", "padma bhushan", "padma shri",
    "bharat ratna", "national film award", "booker prize",
    "commonwealth games", "asian games", "olympic",
    "aditya l1", "chandrayaan", "gaganyaan", "gslv",
    "central vista", "new parliament", "quad", "nato",
    "cabinet secretary", "chief minister appointed",
]

NATIONAL_NOT_BIHAR_PHRASES = [
    "union budget", "national education policy", "nep 2020",
    "yuva 2.0", "central government", "prime minister inaugurated",
    "union ministry", "secretary general", "uk head of state",
    "aiims", "ndma", "nobel prize", "quad", "nato",
]

HARD_OVERRIDE_SUBJECTS = {"General Mental Ability & CSAT", "Current Affairs"}

def _is_gma(text: str) -> bool:
    if _contains(text, GMA_EXCLUSION_PHRASES):
        return False
    family_words = ["mother","father","son","daughter","brother","sister",
                    "husband","wife","uncle","aunt","cousin","nephew","niece",
                    "grandfather","grandmother","grandson","granddaughter"]
    n    = _norm(text)
    hits = sum(1 for w in family_words if _norm(w) in n)
    relation = any(k in n for k in ("related","relation","introduce","how is"))
    if hits >= 3 or (hits >= 1 and relation):
        return True
    symbols = len(re.findall(r"[+\-×÷/*]", text))
    if "means" in text.lower() and symbols >= 2:
        return True
    return _contains(text, GMA_PHRASES)

def subject_hint(question: Dict[str, str]) -> str:
    q    = str(question.get("Questions", ""))
    all_ = " ".join(str(question.get(k, "")) for k in
                    ("Questions","Option A","Option B","Option C","Option D"))
    if _is_gma(q):   return "General Mental Ability & CSAT"
    if _contains(q, CURRENT_AFFAIRS_PHRASES) and not _contains(q, NATIONAL_NOT_BIHAR_PHRASES):
        return "Current Affairs"
    if _is_gma(all_): return "General Mental Ability & CSAT"
    if _contains(all_, CURRENT_AFFAIRS_PHRASES): return "Current Affairs"
    return "NONE"


# ════════════════════════════════════════════════════════════════════════════
# RAG RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════

_chroma_client = None
_pdf_col       = None

def get_collection():
    global _chroma_client, _pdf_col
    if _pdf_col is None:
        if not os.path.exists(RAG_DB_DIR):
            raise FileNotFoundError(
                f"RAG database not found at {RAG_DB_DIR}.\n"
                "Run build_rag.py first."
            )
        _chroma_client = chromadb.PersistentClient(path=RAG_DB_DIR)
        _pdf_col       = _chroma_client.get_collection("pdf_chunks")
    return _pdf_col


def retrieve_context(question: Dict[str, str], n: int = PDF_CHUNKS_PER_Q) -> str:
    """
    Embed the question and retrieve the n most similar PDF chunks.
    Returns a formatted string ready to inject into the prompt.
    If RAG fails for any reason, returns an empty string gracefully.
    """
    try:
        col  = get_collection()
        text = str(question.get("Questions", ""))
        hint = subject_hint(question)

        emb_resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        emb      = emb_resp["embedding"]

        # Filter by subject when we have a hard-override hint
        where = {"subject": hint} if hint in HARD_OVERRIDE_SUBJECTS else None

        results = col.query(
            query_embeddings=[emb],
            n_results=n,
            where=where,
        )

        chunks = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]

        if not chunks:
            return ""

        parts = []
        for chunk, meta in zip(chunks, metas):
            src = meta.get("source", "unknown")
            sub = meta.get("subject", "")
            parts.append(f"[Source: {src} | Subject: {sub}]\n{chunk.strip()}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        # RAG failure is non-fatal — fall back to no context
        print(f"    RAG warning: {e}")
        return ""


# ════════════════════════════════════════════════════════════════════════════
# TAXONOMY BLOCK
# ════════════════════════════════════════════════════════════════════════════

def build_taxonomy_block() -> str:
    lines = []
    for subject in BPSC_SUBJECTS:
        if subject == "Miscellaneous":
            continue
        topics = ", ".join(SUBJECT_TOPICS[subject])
        lines.append(f"- {subject}\n    Topics: {topics}")
    return "\n".join(lines)

TAXONOMY_BLOCK = build_taxonomy_block()


# ════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ════════════════════════════════════════════════════════════════════════════

def build_prompt(questions: List[Dict[str, str]]) -> str:
    blocks = []
    for q in questions:
        hint    = subject_hint(q)
        context = retrieve_context(q)

        hint_label = (
            f"HARD OVERRIDE — you MUST use Subject: {hint}"
            if hint in HARD_OVERRIDE_SUBJECTS
            else f"Strong hint: {hint}"
            if hint != "NONE"
            else "No hint — classify freely"
        )

        block = (
            f"Question ID: {q['Question ID']}\n"
            f"Subject Hint: {hint_label}\n"
            f"Q: {q['Questions']}\n"
            f"(A) {q.get('Option A','')}  "
            f"(B) {q.get('Option B','')}  "
            f"(C) {q.get('Option C','')}  "
            f"(D) {q.get('Option D','')}"
        )

        if context:
            block += f"\n\nREFERENCE MATERIAL:\n{context}"

        blocks.append(block)

    return f"""You are an expert BPSC previous-year question classifier.
Classify every question using the reference material provided.
Return ONLY valid JSON, nothing else.

OUTPUT: {{"classifications":[{{"Question ID":"...","Subject":"...","Topic":"...","Subtopic":"..."}}]}}

━━━ SUBJECTS AND TOPICS (copy EXACTLY) ━━━
{TAXONOMY_BLOCK}

━━━ RULES ━━━
1. Only these 9 subjects exist. Never invent a new subject.
   "History of Bihar" → use "History of India"
   "Bihar Polity" → use "Indian Polity"
   "Bihar Economy" → use "Indian Economy"
   "Mathematics — Mains" → use "General Mental Ability & CSAT"
   "Agriculture" (standalone) → use "Indian Economy"
   "Society" → use "Current Affairs"
   "Disaster Management" → use "Geography"

2. USE THE REFERENCE MATERIAL. Each question has PDF excerpts from
   standard textbooks. Read them — they tell you the correct subject
   and topic. The subtopic should come from the question's own text,
   not from the reference material.

3. Topic must be copied exactly from that Subject's Topics list.

4. Subtopic = the specific fact/person/event THIS question tests.
   Never copy a subtopic from another question or from the reference.

5. Bihar content stays inside the national subject:
   Bihar history → History of India
   Bihar polity → Indian Polity
   Bihar economy → Indian Economy

6. Reasoning puzzles → ALWAYS "General Mental Ability & CSAT":
   coding-decoding, blood relations, direction, series, analogy,
   seating arrangement, age problems, profit/loss, speed-time-distance,
   symbol substitution, LCM/HCF, percentage, pipes & cisterns.

7. "Miscellaneous" only for corrupted/unreadable questions (rare).

━━━ QUESTIONS ━━━
{chr(10).join(blocks)}

Return ONLY the JSON. Every Question ID exactly once.
""".strip()


def build_retry_prompt(questions: List[Dict[str, str]]) -> str:
    blocks = []
    for q in questions:
        hint    = subject_hint(q)
        context = retrieve_context(q)
        block   = (
            f"Question ID: {q['Question ID']}\n"
            f"Hint: {hint}\n"
            f"Q: {q['Questions']}\n"
            f"(A) {q.get('Option A','')}  (B) {q.get('Option B','')}  "
            f"(C) {q.get('Option C','')}  (D) {q.get('Option D','')}"
        )
        if context:
            block += f"\n\nREFERENCE:\n{context}"
        blocks.append(block)

    return f"""RETRY. JSON only. One row per Question ID.
Subjects (copy exactly): {', '.join(s for s in BPSC_SUBJECTS if s != 'Miscellaneous')}
Topic from that subject's list. Subtopic = specific fact from question text.
Reasoning/math puzzles → "General Mental Ability & CSAT".
Bihar content → national subject (History of India / Indian Polity / Indian Economy).

{chr(10).join(blocks)}

JSON: {{"classifications":[{{"Question ID":"...","Subject":"...","Topic":"...","Subtopic":"..."}}]}}
""".strip()


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def normalize_row(row: Dict) -> Dict[str, str]:
    if not isinstance(row, dict):
        return {"Question ID":"","Subject":"","Topic":"","Subtopic":""}
    return {k: str(row.get(k,"")).strip()
            for k in ("Question ID","Subject","Topic","Subtopic")}


def validate_results(
    results: List[Dict],
    expected_ids: List[str],
    question_by_id: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], List[str]]:

    expected = set(expected_ids)
    accepted: Dict[str, Dict] = {}
    errors:   List[str]       = []

    for raw in results:
        row = normalize_row(raw)
        qid = row["Question ID"]

        if qid not in expected:  continue
        if qid in accepted:
            errors.append(f"{qid}: duplicate"); continue

        # Remap ghost subjects
        if row["Subject"] in GHOST_SUBJECT_MAP:
            print(f"  GHOST {qid}: {row['Subject']} → {GHOST_SUBJECT_MAP[row['Subject']]}")
            row["Subject"] = GHOST_SUBJECT_MAP[row["Subject"]]

        # Hard-override hints
        hint = subject_hint(question_by_id[qid])
        if hint in HARD_OVERRIDE_SUBJECTS and row["Subject"] != hint:
            print(f"  OVERRIDE {qid}: {row['Subject']} → {hint}")
            row["Subject"] = hint

        # Validate subject
        if row["Subject"] not in BPSC_SUBJECTS:
            errors.append(f"{qid}: invalid Subject '{row['Subject']}'"); continue

        # Validate topic
        if not row["Topic"]:
            errors.append(f"{qid}: empty Topic"); continue

        # Validate subtopic
        if not row["Subtopic"]:
            errors.append(f"{qid}: empty Subtopic"); continue
        if row["Subtopic"].casefold() == row["Subject"].casefold():
            errors.append(f"{qid}: Subtopic = Subject"); continue
        if row["Subtopic"].casefold() == row["Topic"].casefold():
            errors.append(f"{qid}: Subtopic = Topic"); continue

        # Reject hallucinated subtopics
        if any(h in row["Subtopic"].lower() for h in HALLUCINATED_SUBTOPICS):
            errors.append(f"{qid}: hallucinated subtopic '{row['Subtopic']}'"); continue

        accepted[qid] = row

    missing = sorted(expected - set(accepted))
    errors.extend(f"{qid}: missing" for qid in missing)
    return accepted, errors


# ════════════════════════════════════════════════════════════════════════════
# OLLAMA CALL
# ════════════════════════════════════════════════════════════════════════════

def make_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Question ID": {"type": "string"},
                        "Subject":     {"type": "string", "enum": BPSC_SUBJECTS},
                        "Topic":       {"type": "string"},
                        "Subtopic":    {"type": "string"},
                    },
                    "required": ["Question ID","Subject","Topic","Subtopic"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*","",text,flags=re.I)
        text = re.sub(r"\s*```$","",text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in [("[","]"),("{","}")]:
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:    return json.loads(text[s:e+1])
            except: continue
    raise ValueError("No valid JSON found")


def call_ollama(questions: List[Dict], retry: bool = False) -> List[Dict]:
    prompt   = build_retry_prompt(questions) if retry else build_prompt(questions)
    response = ollama.chat(
        model    = MODEL,
        messages = [{"role":"user","content":prompt}],
        format   = make_schema(),
        options  = {
            "num_ctx":     NUM_CTX,
            "temperature": 0,
            "top_p":       0.2,
            "num_predict": max(256, len(questions) * 80),
        },
        keep_alive = KEEP_ALIVE,
    )
    content = response.get("message",{}).get("content","")
    if not content:
        raise ValueError("Ollama returned empty content")
    payload = extract_json(content)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("classifications"), list):
        return payload["classifications"]
    raise ValueError("Unexpected JSON structure")


def classify_batch(
    questions: List[Dict], retry: bool = False
) -> Tuple[Dict[str,Dict], List[str]]:
    ids            = [q["Question ID"] for q in questions]
    question_by_id = {q["Question ID"]: q for q in questions}
    raw            = call_ollama(questions, retry=retry)
    return validate_results(raw, ids, question_by_id)


def process_one(q: Dict) -> Tuple[Dict | None, str | None]:
    last = "classification failed"
    for attempt in range(INDIVIDUAL_RETRIES + 1):
        try:
            accepted, errors = classify_batch([q], retry=(attempt > 0))
            if q["Question ID"] in accepted:
                return accepted[q["Question ID"]], None
            last = "; ".join(errors) or last
        except Exception as e:
            last = str(e)
        if attempt < INDIVIDUAL_RETRIES:
            time.sleep(0.2)
    return None, last


# ════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ════════════════════════════════════════════════════════════════════════════

def load_questions() -> List[Dict]:
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_existing() -> Dict[str, Dict]:
    if not os.path.exists(OUTPUT_FILE): return {}
    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {str(r.get("Question ID","")).strip(): r for r in rows
            if str(r.get("Question ID","")).strip()}

def load_failed() -> Dict[str, str]:
    if not os.path.exists(FAILED_FILE): return {}
    with open(FAILED_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {str(r.get("Question ID","")).strip(): str(r.get("Reason","")).strip()
            for r in rows if str(r.get("Question ID","")).strip()}

def save_output(classifications: Dict[str, Dict]) -> None:
    ordered = sorted(
        classifications.values(),
        key=lambda r: int(re.search(r"\d+", r["Question ID"]).group())
        if re.search(r"\d+", r["Question ID"]) else 10**9,
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Question ID","Subject","Topic","Subtopic"]
        )
        writer.writeheader()
        writer.writerows(ordered)

def save_failures(failures: Dict[str, str]) -> None:
    ordered = sorted(
        failures.items(),
        key=lambda x: int(re.search(r"\d+", x[0]).group())
        if re.search(r"\d+", x[0]) else 10**9,
    )
    with open(FAILED_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Question ID","Reason"])
        writer.writeheader()
        for qid, reason in ordered:
            writer.writerow({"Question ID": qid, "Reason": reason})

def select_questions(rows: List[Dict]) -> List[Dict]:
    if ONLY_QUESTIONS:
        allowed = {int(x) for x in ONLY_QUESTIONS}
        return [r for r in rows
                if re.search(r"\d+", r["Question ID"])
                and int(re.search(r"\d+", r["Question ID"]).group()) in allowed]
    selected = []
    for r in rows:
        m = re.search(r"\d+", r["Question ID"])
        if m and START_QUESTION <= int(m.group()) <= END_QUESTION:
            selected.append(r)
    return selected


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"BPSC RAG CLASSIFIER")
    print(f"Model: {MODEL}  |  Embed: {EMBED_MODEL}")
    print(f"Batch: {BATCH_SIZE}  |  PDF chunks per Q: {PDF_CHUNKS_PER_Q}")
    print("=" * 60)

    # Verify RAG DB
    try:
        col = get_collection()
        print(f"RAG DB: {col.count()} chunks available\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    questions = load_questions()
    existing  = load_existing()
    failures  = load_failed()

    print(f"Source questions : {len(questions)}")
    print(f"Already done     : {len(existing)}")
    print(f"Currently failed : {len(failures)}")

    if RETRY_FAILED_ONLY:
        failed_ids = set(failures.keys())
        if not failed_ids:
            print("Nothing in failed file. Done.")
            return
        selected = [q for q in questions if q["Question ID"] in failed_ids]
    else:
        selected = select_questions(questions)
        if not FORCE_RECLASSIFY_ALL:
            already_done = set(existing.keys())
            before   = len(selected)
            selected = [q for q in selected if q["Question ID"] not in already_done]
            skipped  = before - len(selected)
            if skipped:
                print(f"Skipping {skipped} already classified")

    print(f"Selected for run : {len(selected)}\n")
    if not selected:
        print("Nothing to classify. Done.")
        return

    for batch_start in range(0, len(selected), BATCH_SIZE):
        batch       = selected[batch_start : batch_start + BATCH_SIZE]
        ids         = [q["Question ID"] for q in batch]
        batch_num   = batch_start // BATCH_SIZE + 1
        total_batches = (len(selected) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"{'='*60}")
        print(f"BATCH {batch_num}/{total_batches}: {ids[0]} – {ids[-1]}")
        print(f"{'='*60}")

        accepted: Dict[str,Dict] = {}
        errors:   List[str]      = []

        try:
            accepted, errors = classify_batch(batch, retry=False)
        except Exception as exc:
            print(f"  Main call failed: {exc}")
            try:
                accepted, errors = classify_batch(batch, retry=True)
                print("  Retry succeeded.")
            except Exception as exc2:
                print(f"  Retry failed: {exc2}")
                accepted, errors = {}, [str(exc2)]

        for qid, row in accepted.items():
            existing[qid] = row
            failures.pop(qid, None)
            print(f"  OK  {qid}: {row['Subject']} | {row['Topic']} | {row['Subtopic']}")

        missing = [qid for qid in ids if qid not in accepted]
        if missing:
            print(f"  Recovery needed: {len(missing)}")
            for err in errors[:5]:
                print(f"    {err}")

            missing_qs = [q for q in batch if q["Question ID"] in set(missing)]
            recovered: Dict[str,Dict] = {}
            for chunk_start in range(0, len(missing_qs), RECOVERY_BATCH_SIZE):
                chunk = missing_qs[chunk_start : chunk_start + RECOVERY_BATCH_SIZE]
                try:
                    part, _ = classify_batch(chunk, retry=True)
                    recovered.update(part)
                except Exception as exc:
                    print(f"  Recovery chunk failed: {exc}")

            for qid, row in recovered.items():
                existing[qid] = row
                failures.pop(qid, None)
                print(f"  REC {qid}: {row['Subject']} | {row['Topic']} | {row['Subtopic']}")

            still_missing = [qid for qid in missing if qid not in recovered]
            for qid in still_missing:
                q = next(q for q in batch if q["Question ID"] == qid)
                row, reason = process_one(q)
                if row:
                    existing[qid] = row
                    failures.pop(qid, None)
                    print(f"  IND {qid}: {row['Subject']} | {row['Topic']} | {row['Subtopic']}")
                else:
                    failures[qid] = reason or "unresolved"
                    print(f"  FAIL {qid}: {failures[qid]}")

        save_output(existing)
        save_failures(failures)
        unresolved = sum(1 for qid in ids if qid not in existing)
        print(f"  Batch {batch_num}: {len(ids)-unresolved} OK, {unresolved} unresolved\n")

    print("=" * 60)
    print("DONE")
    print(f"Output  : {OUTPUT_FILE}")
    print(f"Failures: {FAILED_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
