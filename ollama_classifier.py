import csv
import json
import os
import re
import time
from typing import Dict, List, Tuple

import ollama


MODEL = "gemma3:latest"
SCRIPT_VERSION = "v29"

INPUT_FILE = "questions_check.csv"
OUTPUT_FILE = "questions_classified.csv"
FAILED_FILE = "questions_failed.csv"
TOPIC_REVIEW_FILE = "questions_topic_review.csv"

START_QUESTION = 1
END_QUESTION = 3100
ONLY_QUESTIONS: List[int] = []

RETRY_FAILED_ONLY = False
FORCE_RECLASSIFY_ALL = True

BATCH_SIZE = 12
RECOVERY_BATCH_SIZE = 4
INDIVIDUAL_RETRIES = 2
NUM_CTX = 8192
KEEP_ALIVE = "10m"


# ============================================================
# SUBJECT / TOPIC TAXONOMY  (9 subjects + Miscellaneous)
#
# Mains-only subjects (Ethics, Essay, Mathematics — Mains,
# Language — Mains) are REMOVED. Every BPSC Prelims MCQ must
# land in one of the 9 subjects below, or Miscellaneous.
# ============================================================

SUBJECT_TOPICS: Dict[str, List[str]] = {

    # ----------------------------------------------------------
    # 1. HISTORY OF INDIA
    # ----------------------------------------------------------
    "History of India": [
        # Ancient India
        "Prehistoric India", "Indus Valley Civilization", "Vedic Age",
        "Mahajanapadas", "Buddhism", "Jainism", "Mauryan Empire",
        "Post-Mauryan India", "Gupta Age", "Harshavardhana",
        "South Indian Kingdoms", "Sangam Age",
        # Medieval India
        "Early Medieval India", "Delhi Sultanate", "Mughal Empire",
        "Bhakti Movement", "Sufi Movement", "Marathas", "Later Mughals",
        "Vijayanagara Empire", "Rajput Kingdoms",
        # Modern India
        "European Penetration", "British Expansion",
        "British Administrative Policies", "Economic Impact of British Rule",
        "Socio-Religious Reform Movements", "Tribal and Peasant Movements",
        "Revolt of 1857", "Indian National Movement",
        "Constitutional Development", "Independence and Partition",
        # Bihar-specific history (merged here — no separate Bihar subject)
        "Magadha", "Mauryan Bihar", "Gupta Period in Bihar",
        "Pala Dynasty", "Sena Influence", "Sher Shah Suri",
        "Sasaram", "Mughal Bihar", "Permanent Settlement",
        "Indigo Cultivation", "Champaran Satyagraha",
        "Bihar's Role in 1857", "Kunwar Singh",
        "Bihar during Non-Cooperation Movement",
        "Civil Disobedience in Bihar", "Quit India Movement in Bihar",
        "Peasant Movements in Bihar",
        "Formation of Bihar and Orissa Province",
        "Bihar's Separation from Bengal", "Creation of Jharkhand",
        "Important Historical Personalities",
        "Vaishali", "Nalanda", "Vikramshila",
    ],

    # ----------------------------------------------------------
    # 2. GEOGRAPHY
    # ----------------------------------------------------------
    "Geography": [
        # Physical / World
        "Earth and Universe", "Geomorphology", "Climatology",
        "Oceanography", "Soils", "Natural Vegetation",
        "Biodiversity and Ecosystem",
        # Indian Geography
        "Physiographic Divisions of India", "Himalayan Rivers",
        "Peninsular Rivers", "Indian Monsoon", "Indian Agriculture",
        "Minerals and Mining", "Industries", "Transport and Communication",
        "Population and Census", "Migration", "Urbanisation",
        # Bihar Geography
        "Location and Boundaries of Bihar", "North Bihar Plains",
        "South Bihar Plateau", "Floods in Bihar", "Drought in Bihar",
        "Soil Types in Bihar", "Climate of Bihar",
        "Irrigation Projects in Bihar", "Forests of Bihar",
        "Wildlife in Bihar", "District-wise Geography of Bihar",
        "Major Infrastructure Projects in Bihar",
        "Kosi River", "Gandak River", "Sone River",
    ],

    # ----------------------------------------------------------
    # 3. INDIAN POLITY
    # ----------------------------------------------------------
    "Indian Polity": [
        # Constitution & Centre
        "Making of the Constitution", "Preamble",
        "Fundamental Rights", "Directive Principles",
        "Fundamental Duties", "Union Government", "Parliament",
        "President and Vice President", "Prime Minister and Cabinet",
        "Supreme Court", "High Courts",
        "Federalism", "Centre-State Relations",
        "Emergency Provisions", "Amendment of Constitution",
        "Schedules of the Constitution",
        # State & Local
        "State Government", "State Legislature",
        "Local Government", "Panchayati Raj",
        # Constitutional & Statutory Bodies
        "Election Commission", "UPSC and State PSCs",
        "CAG", "Finance Commission", "National Commission for SC/ST",
        "Attorney General", "Advocate General",
        "Constitutional Bodies", "Non-constitutional Bodies",
        # Bihar-specific (merged here)
        "Bihar Legislature", "Bihar Panchayati Raj",
        "Bihar Administration", "Bihar Government Schemes",
        "Bihar-specific Governance",
        # Governance & Policy
        "Governance and Public Policy", "RTI", "Lokpal and Lokayukta",
        "Inter-State Disputes",
    ],

    # ----------------------------------------------------------
    # 4. INDIAN ECONOMY
    # ----------------------------------------------------------
    "Indian Economy": [
        # Basics
        "Basic Economic Concepts", "National Income and GDP",
        "Economic Planning", "NITI Aayog", "Five Year Plans",
        # Sectors
        "Agricultural Sector", "Industrial Sector", "Service Sector",
        "Major Crops and Crop Patterns", "Agricultural Reforms",
        "Land Reforms", "Green Revolution",
        # Finance & Banking
        "Banking System", "Reserve Bank of India",
        "Monetary Policy", "Fiscal Policy", "Union Budget",
        "Taxation", "GST",
        # External
        "External Sector", "Balance of Payments", "Foreign Trade Policy",
        "WTO", "IMF", "World Bank",
        # Development & Welfare
        "Human Development", "Poverty and Inequality",
        "Employment and Unemployment",
        "Social Sector Schemes", "Health Economy", "Education Economy",
        "Infrastructure", "Public Distribution System",
        # Indices & Reports
        "Economic Indices and Reports", "HDI", "GDP Growth",
        # Bihar Economy (merged here)
        "Bihar Economy Overview", "Bihar Budget",
        "Bihar Economic Survey", "Bihar Agriculture",
        "Bihar Industries", "Bihar Employment",
    ],

    # ----------------------------------------------------------
    # 5. GENERAL SCIENCE
    # ----------------------------------------------------------
    "General Science": [
        # Physics
        "Motion and Laws of Motion", "Work Energy and Power",
        "Gravitation", "Heat and Thermodynamics",
        "Light and Optics", "Sound", "Electricity and Magnetism",
        "Modern Physics", "Nuclear Physics",
        # Chemistry
        "Matter and Its Properties", "Atomic Structure",
        "Chemical Bonding", "Periodic Table",
        "Acids Bases and Salts", "Metals and Non-metals",
        "Carbon and Its Compounds", "Chemical Reactions",
        "Electrochemistry", "Fuels",
        # Biology
        "Cell Biology", "Genetics and Heredity", "Evolution",
        "Plant Kingdom", "Animal Kingdom", "Human Physiology",
        "Nutrition and Diseases", "Microorganisms",
        "Biotechnology", "Ecology Basics",
        # Applied Science
        "Science and Technology", "Space Technology",
        "Defence Technology", "Medical Science",
        "Inventions and Discoveries", "Scientific Instruments",
        "Computer and IT Basics",
    ],

    # ----------------------------------------------------------
    # 6. ENVIRONMENT & ECOLOGY
    # ----------------------------------------------------------
    "Environment & Ecology": [
        "Ecosystem and Food Chain", "Biodiversity",
        "Conservation of Wildlife", "Protected Areas",
        "National Parks and Sanctuaries",
        "Climate Change and Global Warming",
        "Greenhouse Effect", "Ozone Layer Depletion",
        "Air Pollution", "Water Pollution", "Soil Pollution",
        "Noise Pollution", "Solid Waste Management",
        "Environmental Laws and Acts",
        "International Environmental Agreements",
        "UNFCCC and COP", "Paris Agreement",
        "Sustainable Development", "Green Economy",
        "Environmental Bodies and Organisations",
        "Bihar Environment and Ecology",
    ],

    # ----------------------------------------------------------
    # 7. CURRENT AFFAIRS
    # ----------------------------------------------------------
    "Current Affairs": [
        "National Current Affairs", "International Current Affairs",
        "Science and Technology Current Affairs",
        "Environment and Ecology Current Affairs",
        "Economy and Business Current Affairs",
        "Sports Current Affairs", "Awards and Honours",
        "Books and Authors", "Important Days and Events",
        "Summits and Conferences", "Defence and Security",
        "Appointments and Resignations",
        "Bihar Current Affairs",
    ],

    # ----------------------------------------------------------
    # 8. ART & CULTURE
    # ----------------------------------------------------------
    "Art & Culture": [
        # Indian Culture
        "Indian Architecture", "Indian Sculpture",
        "Indian Painting", "Indian Music", "Indian Dance",
        "Indian Literature", "Indian Languages",
        "Festivals of India", "Indian Cuisine",
        "Folk Arts and Crafts",
        # Religion & Philosophy
        "Hinduism", "Buddhism and Jainism",
        "Islam in India", "Sikhism",
        "Indian Philosophy",
        # Heritage
        "UNESCO World Heritage Sites", "ASI and Heritage Conservation",
        "Indian Museums",
        # Bihar Culture (merged here)
        "Bihar Art and Architecture", "Bihar Folk Culture",
        "Madhubani Painting", "Chhau Dance",
        "Festivals of Bihar", "Bihar Literature",
        "Languages of Bihar", "Bihari Cuisine",
        "Famous Personalities of Bihar",
    ],

    # ----------------------------------------------------------
    # 9. GENERAL MENTAL ABILITY & CSAT
    # ----------------------------------------------------------
    "General Mental Ability & CSAT": [
        "Coding-Decoding", "Blood Relations", "Direction Sense",
        "Ranking and Ordering", "Alphabetical and Dictionary Order",
        "Number Series", "Letter Series",
        "Classification and Odd One Out",
        "Analogy", "Syllogism and Logical Reasoning",
        "Seating Arrangement", "Puzzle",
        "Calendar and Clock Problems", "Age Problems",
        "Time and Work", "Time Speed and Distance",
        "Profit Loss and Discount",
        "Simple and Compound Interest",
        "Ratio Proportion and Partnership",
        "Percentage", "Average",
        "Mathematical and Symbol Operations",
        "Reading Comprehension", "Decision Making",
        "Basic Numeracy", "Data Interpretation",
        "Venn Diagrams", "Statement and Conclusions",
        "Statement and Assumptions",
    ],

    # ----------------------------------------------------------
    # FALLBACK
    # ----------------------------------------------------------
    "Miscellaneous": [
        "Miscellaneous",
    ],
}

BPSC_SUBJECTS: List[str] = list(SUBJECT_TOPICS.keys())


# ============================================================
# HELPER UTILITIES
# ============================================================

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _contains_phrase(text: str, phrases: List[str]) -> bool:
    n = _norm(text)
    return any(_norm(p) in n for p in phrases)


# ============================================================
# DETERMINISTIC HINT KEYWORDS
#
# These are checked BEFORE sending to Ollama.  A match forces
# the Subject — Ollama still picks Topic + Subtopic freely.
# Keywords are kept highly specific to avoid false positives.
# ============================================================

# ---- General Mental Ability & CSAT -------------------------

GMA_PHRASES = [
    # Coding / cipher
    "coding decoding", "coding-decoding", "substitution cipher",
    "code language", "code breaking", "coded as", "written in a code",
    "written as a code", "code for", "coded language",
    # Blood relations
    "blood relation", "blood relations", "family relationship",
    "how is", "related to", "is the mother of", "is the father of",
    "is the son of", "is the daughter of", "is the brother of",
    "is the sister of", "is the wife of", "is the husband of",
    "introduce him", "introduce her", "introduce",
    # Direction sense
    "direction sense", "facing north", "facing south", "facing east",
    "facing west", "turned left", "turned right", "walks north",
    "walks south", "walks east", "walks west", "distance between",
    "how far", "which direction",
    # Ranking & ordering
    "ranking", "rank from", "position from", "from the top",
    "from the bottom", "from the left", "from the right",
    "in a row", "in a queue", "in a line",
    # Alphabetical / dictionary
    "alphabetical order", "dictionary order", "alphabetically",
    "arrange the letters", "rearrange the letters",
    "meaningful word", "which letter",
    # Series
    "number series", "letter series", "next number",
    "next term", "missing number", "missing term",
    "next in the series", "find the missing",
    # Odd one out / classification
    "odd one out", "which is different", "does not belong",
    "find the odd", "select the odd",
    # Analogy
    "analogy", "is to", "as :: ", "as :", " : : ",
    "same relationship",
    # Syllogism
    "syllogism", "all are", "some are", "no are",
    "conclusions follow", "conclusion follows",
    "which conclusion", "all students", "some students",
    # Seating arrangement
    "seating arrangement", "sitting arrangement",
    "sits between", "seated between",
    "sits to the left", "sits to the right",
    # Calendar / clock
    "calendar problem", "clock problem", "what day",
    "what date", "day of the week", "angle between the hands",
    "time on the clock",
    # Puzzles
    "puzzle", "if red means", "if blue means",
    # Symbol / operator substitution
    "if + means", "if - means", "if × means", "if ÷ means",
    "means multiplication", "means addition", "means subtraction",
    "means division", "replace +", "replace -",
    "operators are interchanged", "interchanged",
    "mathematical operations", "symbol substitution",
    # Age problems
    "times as old", "as old as", "years ago",
    "years hence", "present age", "present ages",
    "age of", "age will be", "how old",
    # Quantitative word problems
    "cost price", "selling price", "marked price",
    "profit percent", "loss percent", "profit and loss",
    "simple interest", "compound interest",
    "average speed", "km per hour", "kilometers per hour", "km/h",
    "sum of the numbers", "sum of two numbers",
    "difference of the numbers", "difference of two numbers",
    "product of two numbers", "ratio of their",
    "value of x", "value of y", "find x", "find y",
    "pipes and cisterns", "pipe can fill", "cistern can be filled",
    "dearness allowance", "basic salary",
    "square of a number", "squares of two numbers",
    "time and work", "can complete the work",
    "speed of the train", "length of the train",
    "upstream", "downstream",
    "discount percent", "successive discount",
    "partnership profit", "capital invested",
    "lcm", "hcf", "highest common factor", "lowest common multiple",
    # Reading comprehension (CSAT)
    "passage", "according to the passage", "the author says",
    "the passage states",
    # Venn / statement
    "venn diagram", "statement and conclusion",
    "statement and assumption",
]

FAMILY_RELATION_WORDS = [
    "mother", "father", "son", "daughter", "brother", "sister",
    "husband", "wife", "uncle", "aunt", "cousin",
    "nephew", "niece", "grandfather", "grandmother",
    "grandson", "granddaughter", "father in law", "mother in law",
    "brother in law", "sister in law", "son in law", "daughter in law",
    "parent", "child", "sibling", "relative", "relation",
]


def _looks_like_blood_relation_puzzle(text: str) -> bool:
    n = _norm(text)
    hits = sum(1 for w in FAMILY_RELATION_WORDS if _norm(w) in n)
    mentions_relation = any(k in n for k in (
        "related", "relation", "relationship", "introduce",
        "how is", "pointing", "referring"
    ))
    return hits >= 3 or (hits >= 1 and mentions_relation)


def _looks_like_symbol_substitution_puzzle(text: str) -> bool:
    symbol_count = len(re.findall(r"[+\-×÷/*]", text))
    return "means" in text.lower() and symbol_count >= 2


def _looks_like_quant_word_problem(text: str) -> bool:
    quant_phrases = [
        "cost price", "selling price", "marked price",
        "profit percent", "loss percent",
        "simple interest", "compound interest",
        "average speed", "km per hour", "km/h",
        "sum of the numbers", "difference of the numbers",
        "product of two numbers", "ratio of their",
        "value of x", "value of y",
        "pipes and cisterns", "pipe can fill",
        "dearness allowance", "basic salary",
        "square of a number",
        "time and work", "can complete the work",
        "speed of the train", "upstream", "downstream",
        "discount percent", "successive discount",
        "partnership profit", "lcm", "hcf",
    ]
    return _contains_phrase(text, quant_phrases)


def _is_gma(text: str) -> bool:
    return (
        _contains_phrase(text, GMA_PHRASES)
        or _looks_like_blood_relation_puzzle(text)
        or _looks_like_symbol_substitution_puzzle(text)
        or _looks_like_quant_word_problem(text)
    )


# ---- Current Affairs ---------------------------------------

CURRENT_AFFAIRS_PHRASES = [
    # Year anchors — any recent year in the question signals CA
    "2022", "2023", "2024", "2025", "2026",
    # Awards
    "nobel prize", "ramon magsaysay", "academy award", "oscar",
    "national film award", "padma vibhushan", "padma bhushan",
    "padma shri", "bharat ratna", "international booker",
    "booker prize", "pulitzer", "man booker",
    "dada saheb phalke", "filmfare award",
    # Sports
    "national junior athletics", "athletics championship",
    "commonwealth games", "asian games", "olympic",
    "world cup 2023", "world cup 2024", "ipl 2024",
    "cricket world cup",
    # Summits & conferences
    "g20 summit", "g7 summit", "cop28", "cop29", "baku",
    "ai action summit", "quad summit", "brics summit",
    "climate conference",
    # Launches / missions
    "aditya l1", "chandrayaan", "gaganyaan", "gslv",
    "lunar trailblazer", "akashteer",
    "artemis", "james webb",
    # Appointments
    "recently appointed", "new chief minister",
    "new governor", "new president of",
    "new prime minister of", "newly elected",
    # Schemes / policies
    "national urban innovation", "nuis", "u-win",
    "bharatmala", "mission daksh",
    # Misc recent events
    "eurovision", "panchayati raj day",
    "holocaust memorial day",
    "yala glacier", "poor things", "kutch express",
    "classical languages recognized", "classical languages recognised",
    "recently conserved", "recently declared", "recently launched",
    "recent award", "recent summit", "recent appointment",
    "current event", "current events",
]


# ---- History of India keyword anchors ----------------------
# (used to rescue questions that might otherwise drift to
#  General Science or Geography)

HISTORY_PHRASES = [
    "maurya", "mauryan", "chandragupta", "ashoka", "asoka",
    "gupta", "harsha", "harshavardhana", "pala", "sena",
    "mughal", "akbar", "babur", "humayun", "jahangir",
    "shah jahan", "aurangzeb", "sher shah", "delhi sultanate",
    "iltutmish", "balban", "alauddin", "tughlaq", "lodhi",
    "vijayanagara", "krishnadevaraya", "bahmani",
    "marathas", "shivaji", "peshwa",
    "vedic", "rigveda", "upanishad", "aryan",
    "indus valley", "harappan", "mohenjo daro", "lothal",
    "buddhism", "gautam buddha", "bodh gaya", "sarnath",
    "jainism", "mahavira", "tirthankara",
    "bhakti movement", "sufi", "kabir", "mirabai", "nanak",
    "revolt of 1857", "sepoy mutiny", "kunwar singh",
    "indian national congress", "gandhi", "nehru", "bose",
    "non-cooperation", "civil disobedience", "quit india",
    "champaran", "partition of india", "independence 1947",
    "permanent settlement", "indigo", "zamindari",
    "nalanda", "vikramshila", "vaishali", "pataliputra",
    "taxila", "sanchi", "ajanta", "ellora", "mahabalipuram",
    "battle of plassey", "battle of buxar", "battle of panipat",
    "simon commission", "montagu chelmsford", "morley minto",
    "government of india act", "cripps mission",
    "cabinet mission", "mountbatten",
]


# ---- Geography keyword anchors -----------------------------

GEOGRAPHY_PHRASES = [
    "latitude", "longitude", "tropic of cancer",
    "meridian", "equator", "international date line",
    "peninsula", "plateau", "delta", "estuary",
    "himalaya", "western ghats", "eastern ghats",
    "deccan plateau", "thar desert",
    "river basin", "catchment area", "watershed",
    "monsoon", "rainfall", "cyclone", "tornado",
    "earthquake", "volcano", "tectonic",
    "continental drift", "sea floor spreading",
    "biodiversity hotspot",
    "national park", "wildlife sanctuary", "biosphere reserve",
    "soil erosion", "desertification",
    "population density", "census",
    "urbanisation", "metropolitan",
    "kosi", "gandak", "sone", "ganga", "yamuna", "brahmaputra",
    "kaveri", "godavari", "krishna", "mahanadi", "narmada",
    "tapti", "indus", "chenab", "jhelum", "ravi", "beas", "sutlej",
    "bay of bengal", "arabian sea", "lakshadweep sea",
    "andaman", "nicobar", "lakshadweep",
    "mineral", "iron ore", "coal", "bauxite", "mica",
    "green revolution", "agriculture zone",
]


# ---- Indian Polity keyword anchors -------------------------

POLITY_PHRASES = [
    "article", "constitution", "amendment", "schedule",
    "fundamental right", "directive principle", "fundamental duty",
    "parliament", "lok sabha", "rajya sabha", "speaker",
    "president", "vice president", "prime minister", "cabinet",
    "supreme court", "high court", "judicial review",
    "federalism", "centre state", "concurrent list",
    "union list", "state list",
    "election commission", "panchayati raj", "local government",
    "municipality", "gram panchayat",
    "upsc", "state psc", "cag", "finance commission",
    "attorney general", "advocate general",
    "governor", "chief minister", "state legislature",
    "legislative assembly", "legislative council",
    "emergency", "proclamation", "president rule",
    "rti", "lokpal", "lokayukta",
    "constituent assembly", "drafting committee",
    "bhimrao ambedkar", "b.r. ambedkar",
    "comptroller", "auditor general",
]


# ---- Indian Economy keyword anchors ------------------------

ECONOMY_PHRASES = [
    "gdp", "gnp", "national income", "per capita income",
    "economic growth", "inflation", "deflation", "stagflation",
    "repo rate", "reverse repo", "crr", "slr",
    "rbi", "reserve bank", "monetary policy",
    "fiscal policy", "budget", "deficit", "surplus",
    "tax", "gst", "direct tax", "indirect tax", "income tax",
    "five year plan", "niti aayog", "planning commission",
    "poverty line", "below poverty line", "bpl",
    "hdi", "human development index", "gini coefficient",
    "unemployment", "employment", "labour",
    "wto", "imf", "world bank", "asian development bank",
    "sebi", "stock exchange", "sensex", "nifty",
    "foreign direct investment", "fdi", "fii",
    "balance of payment", "current account", "capital account",
    "export", "import", "trade deficit",
    "public distribution", "food security", "pds",
    "green revolution", "land reform", "cooperative farming",
    "pradhan mantri", "jan dhan", "mudra", "pmgsy",
    "bihar budget", "bihar economic",
]


# ---- General Science keyword anchors -----------------------

SCIENCE_PHRASES = [
    # Physics
    "newton", "law of motion", "inertia", "momentum",
    "velocity", "acceleration", "force", "friction",
    "gravitation", "gravitational", "escape velocity",
    "work energy", "kinetic energy", "potential energy",
    "thermodynamics", "heat", "temperature", "specific heat",
    "conduction", "convection", "radiation",
    "light", "reflection", "refraction", "lens", "mirror",
    "wavelength", "frequency", "amplitude",
    "electricity", "current", "voltage", "resistance", "ohm",
    "magnetic", "electromagnetic", "transformer",
    "nuclear", "radioactive", "fission", "fusion",
    "quantum", "photon", "electron",
    # Chemistry
    "atom", "molecule", "element", "compound", "mixture",
    "periodic table", "valency", "atomic number", "atomic mass",
    "acid", "base", "salt", "ph",
    "oxidation", "reduction", "redox",
    "metal", "non-metal", "alloy",
    "carbon", "hydrocarbon", "organic", "polymer",
    "catalyst", "enzyme",
    "electrolysis", "electrochemistry",
    # Biology
    "cell", "nucleus", "chromosome", "dna", "rna",
    "gene", "genetics", "heredity", "mutation",
    "photosynthesis", "respiration", "transpiration",
    "blood group", "blood type", "rh factor",
    "vitamin", "mineral", "protein", "carbohydrate", "fat",
    "bacteria", "virus", "fungi", "protozoa",
    "vaccine", "antibiotic", "immunity",
    "digestion", "excretion", "nervous system", "endocrine",
    "hormone", "insulin", "adrenaline",
    "taxonomy", "kingdom", "phylum", "class", "order",
    "mammal", "reptile", "amphibian", "bird", "fish",
    "plant", "flower", "seed", "pollination",
    # Applied
    "telescope", "microscope", "periscope", "stethoscope",
    "laser", "radar", "sonar",
    "satellite", "space", "rocket", "isro",
    "computer", "internet", "artificial intelligence",
    "nanotechnology", "biotechnology",
    "nuclear reactor", "solar energy", "wind energy",
    "semiconductor", "transistor",
    "invention", "inventor", "discovery",
    "scientific instrument",
]


# ---- Environment & Ecology keyword anchors -----------------

ENVIRONMENT_PHRASES = [
    "ecosystem", "food chain", "food web", "trophic level",
    "producer", "consumer", "decomposer",
    "biodiversity", "endemic species", "invasive species",
    "endangered", "extinct", "red list", "iucn",
    "wildlife protection act", "forest act",
    "national park", "wildlife sanctuary", "biosphere reserve",
    "tiger reserve", "project tiger", "project elephant",
    "climate change", "global warming", "greenhouse gas",
    "carbon dioxide", "methane", "nitrous oxide",
    "ozone layer", "ozone depletion", "cfc",
    "air pollution", "water pollution", "soil pollution",
    "noise pollution", "solid waste", "e-waste",
    "unfccc", "kyoto protocol", "paris agreement",
    "cop", "cop26", "cop27", "cop28",
    "ramsar", "wetland", "mangrove", "coral reef",
    "deforestation", "afforestation", "reforestation",
    "sustainable development", "sdg", "green economy",
    "environmental impact", "eia",
    "pollution control board", "cpcb", "spcb",
    "ecology", "habitat", "niche",
]


# ---- Art & Culture keyword anchors -------------------------

CULTURE_PHRASES = [
    # Architecture & Art
    "architecture", "temple", "mosque", "church", "stupa",
    "fort", "palace", "mausoleum", "tomb",
    "sculpture", "painting", "mural", "fresco",
    "madhubani", "warli", "pattachitra", "kalamkari",
    "miniature painting", "mughal painting", "rajput painting",
    # Music & Dance
    "classical music", "carnatic", "hindustani",
    "raga", "tala", "raag",
    "bharatanatyam", "kathak", "odissi", "kuchipudi",
    "manipuri", "mohiniyattam", "sattriya", "chhau",
    "folk dance", "folk music", "folk art",
    # Literature
    "literature", "poet", "poetry", "novel", "author",
    "ramayana", "mahabharata", "vedas", "upanishads",
    "kalidas", "tulsidas", "kabir", "mirabai", "surdas",
    "rabindranath tagore", "premchand",
    # Religion & Philosophy
    "hinduism", "buddhism", "jainism", "sikhism", "islam",
    "vedanta", "yoga", "meditation",
    "pilgrimage", "shrine", "sacred",
    # Festivals & Traditions
    "festival", "diwali", "holi", "eid", "christmas",
    "durga puja", "chhath puja", "bihu", "pongal",
    "navratri", "dussehra",
    # Heritage
    "world heritage", "unesco", "asi",
    "museum", "gallery",
    # Bihar culture
    "chhath", "sonepur fair", "rajgir", "nalanda",
    "vikramshila", "bodh gaya", "patna sahib",
    "bihar sharif", "darbhanga",
    "maithili", "bhojpuri", "magahi", "angika",
]


def _is_history(text: str) -> bool:
    return _contains_phrase(text, HISTORY_PHRASES)


def _is_geography(text: str) -> bool:
    return _contains_phrase(text, GEOGRAPHY_PHRASES)


def _is_polity(text: str) -> bool:
    return _contains_phrase(text, POLITY_PHRASES)


def _is_economy(text: str) -> bool:
    return _contains_phrase(text, ECONOMY_PHRASES)


def _is_science(text: str) -> bool:
    return _contains_phrase(text, SCIENCE_PHRASES)


def _is_environment(text: str) -> bool:
    return _contains_phrase(text, ENVIRONMENT_PHRASES)


def _is_culture(text: str) -> bool:
    return _contains_phrase(text, CULTURE_PHRASES)


def _is_current_affairs(text: str) -> bool:
    return _contains_phrase(text, CURRENT_AFFAIRS_PHRASES)


def subject_hint(question: Dict[str, str]) -> str:
    """
    Deterministic high-confidence subject routing.

    Checked BEFORE Ollama. Returns a subject name when confident,
    or 'NONE' to let Ollama decide.

    Priority order:
      1. GMA (reasoning puzzles) — very specific patterns, rarely wrong
      2. Current Affairs — year anchors + event-specific phrases
      3. Everything else — checked but NOT overriding, just returned
         so Ollama can use as a strong hint.
    """
    q_only = str(question.get("Questions", ""))
    all_text = " ".join(str(question.get(k, "")) for k in (
        "Questions", "Option A", "Option B", "Option C", "Option D"
    ))

    # GMA check on question stem first (highest confidence)
    if _is_gma(q_only):
        return "General Mental Ability & CSAT"

    # Current Affairs check
    if _is_current_affairs(q_only):
        return "Current Affairs"

    # GMA check on all text (catches options like "(A) 15 km/h")
    if _is_gma(all_text):
        return "General Mental Ability & CSAT"

    if _is_current_affairs(all_text):
        return "Current Affairs"

    # Softer hints — still useful to guide Ollama
    if _is_history(all_text):
        return "History of India"
    if _is_polity(all_text):
        return "Indian Polity"
    if _is_economy(all_text):
        return "Indian Economy"
    if _is_science(all_text):
        return "General Science"
    if _is_environment(all_text):
        return "Environment & Ecology"
    if _is_geography(all_text):
        return "Geography"
    if _is_culture(all_text):
        return "Art & Culture"

    return "NONE"


# ============================================================
# SUBJECT HINT CONFIDENCE
#
# GMA and Current Affairs hints are HARD overrides (followed
# even if Ollama disagrees). All other hints are SOFT — shown
# to Ollama but not forced. This avoids wrong overrides when
# a science question happens to mention an article number, etc.
# ============================================================

HARD_OVERRIDE_SUBJECTS = {"General Mental Ability & CSAT", "Current Affairs"}


# ============================================================
# TAXONOMY BLOCK FOR PROMPTS
# ============================================================

def build_taxonomy_block() -> str:
    lines = []
    for subject in BPSC_SUBJECTS:
        if subject == "Miscellaneous":
            continue
        topics = ", ".join(SUBJECT_TOPICS[subject])
        lines.append(f"- {subject}\n    Topics: {topics}")
    return "\n".join(lines)


TAXONOMY_BLOCK = build_taxonomy_block()


GOOD_PATTERNS_WARNING = """
SUBTOPIC RULES:
- Subtopic must be the specific fact/event/ruler/item/scheme/technique
  actually named or tested in THIS question — not a generic label.
- Subtopic must be more specific than Topic, never identical to Topic or Subject.
- NEVER copy a Subtopic from another question in this batch.
- NEVER use a memorised stock phrase ("Kosi-Mechi Link Project",
  "Sher Shah Suri", "Human Development Index") for a question that
  does not literally mention it. Ground Subtopic only in the question's
  own text.
""".strip()


# ============================================================
# FILE I/O
# ============================================================

def load_questions() -> List[Dict[str, str]]:
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = ["Question ID", "Questions", "Option A", "Option B", "Option C", "Option D"]
    for col in required:
        if col not in rows[0]:
            raise ValueError(f"Missing column in {INPUT_FILE}: {col}")
    return rows


def load_existing() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(OUTPUT_FILE):
        return {}
    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        qid = str(row.get("Question ID", "")).strip()
        if qid:
            out[qid] = {
                "Question ID": qid,
                "Subject": row.get("Subject", "").strip(),
                "Topic": row.get("Topic", "").strip(),
                "Subtopic": row.get("Subtopic", "").strip(),
            }
    return out


def load_failed() -> Dict[str, str]:
    if not os.path.exists(FAILED_FILE):
        return {}
    with open(FAILED_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        str(r.get("Question ID", "")).strip(): str(r.get("Reason", "")).strip()
        for r in rows if str(r.get("Question ID", "")).strip()
    }


def save_output(classifications: Dict[str, Dict[str, str]]) -> None:
    ordered = sorted(
        classifications.values(),
        key=lambda r: int(re.search(r"\d+", r["Question ID"]).group())
        if re.search(r"\d+", r["Question ID"]) else 10**9,
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Question ID", "Subject", "Topic", "Subtopic"]
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
        writer = csv.DictWriter(f, fieldnames=["Question ID", "Reason"])
        writer.writeheader()
        for qid, reason in ordered:
            writer.writerow({"Question ID": qid, "Reason": reason})


_topic_mismatches: List[Dict[str, str]] = []


def save_topic_review(rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    file_exists = os.path.exists(TOPIC_REVIEW_FILE)
    with open(TOPIC_REVIEW_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Question ID", "Subject", "Topic", "Subtopic", "Question"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt(questions: List[Dict[str, str]]) -> str:
    blocks = []
    for q in questions:
        hint = subject_hint(q)
        hint_instruction = (
            f"HARD OVERRIDE — you MUST use Subject: {hint}"
            if hint in HARD_OVERRIDE_SUBJECTS
            else f"Strong hint (use unless clearly wrong): {hint}"
            if hint != "NONE"
            else "No hint — classify freely from the taxonomy"
        )
        blocks.append(
            f"Question ID: {q['Question ID']}\n"
            f"Subject Hint: {hint_instruction}\n"
            f"Q: {q['Questions']}\n"
            f"(A) {q.get('Option A','')}  "
            f"(B) {q.get('Option B','')}  "
            f"(C) {q.get('Option C','')}  "
            f"(D) {q.get('Option D','')}"
        )

    return f"""You are an expert BPSC previous-year question classifier.
Classify every question into EXACTLY one Subject, one Topic, and one Subtopic.

OUTPUT FORMAT — return ONLY valid JSON, nothing else:
{{"classifications":[{{"Question ID":"...","Subject":"...","Topic":"...","Subtopic":"..."}}]}}

━━━ SUBJECTS AND TOPICS (copy EXACTLY) ━━━

{TAXONOMY_BLOCK}

━━━ RULES ━━━

1. ONLY 9 SUBJECTS EXIST. You MUST pick Subject from the list above.
   "Ethics & Integrity", "Essay", "Mathematics — Mains", "Language — Mains",
   "History of Bihar", "Bihar Polity & Governance", "Bihar Economy" and any
   other subject NOT listed above do NOT exist. Never invent a new subject.
   If you would have said "History of Bihar", use "History of India" instead.
   If you would have said "Bihar Polity", use "Indian Polity" instead.
   If you would have said "Bihar Economy", use "Indian Economy" instead.

2. Topic must be copied EXACTLY from that Subject's own Topics list above.
   Never put a Topic under the wrong Subject.

3. Subtopic is free text — the specific fact/person/event/method tested.
   Must be more specific than Topic. Never identical to Subject or Topic.
   Built only from words in THIS question's own text.

4. Bihar-specific content goes INSIDE the relevant subject (not a separate
   Bihar subject). History about Bihar → "History of India". Economy of
   Bihar → "Indian Economy". Polity of Bihar → "Indian Polity".

5. REASONING QUESTIONS → ALWAYS "General Mental Ability & CSAT":
   • Coding-decoding, cipher, code language
   • Blood relation puzzles ("X is the mother of Y's son...")
   • Direction sense, ranking, alphabetical/dictionary order
   • Number/letter series, odd-one-out, analogy, syllogism
   • Seating arrangement, calendar, clock, puzzle
   • Age word-problems ("X is 3 times as old as Y...")
   • Symbol/operator substitution ("if + means ×...")
   • Quantitative word-problems: profit/loss, simple/compound interest,
     speed-distance-time, ratio/proportion, time and work, pipes & cisterns,
     LCM/HCF, percentage, average, discount, partnership
   These are NEVER History, Geography, Economy, Science, or any Mains subject,
   even when they contain numbers, family words, or currency terms.

6. "Miscellaneous" is ONLY for corrupted/garbled text that cannot be
   classified. It should appear in fewer than 1% of rows.

7. SUBJECT HINTS: Each question has a Subject Hint.
   "HARD OVERRIDE" hints MUST be followed exactly.
   "Strong hint" hints should be followed unless clearly wrong.
   "No hint" means use your own judgement.

{GOOD_PATTERNS_WARNING}

━━━ QUESTIONS ━━━
{chr(10).join(blocks)}

Return ONLY the JSON. Every input Question ID must appear exactly once.
""".strip()


def build_retry_prompt(questions: List[Dict[str, str]]) -> str:
    blocks = []
    for q in questions:
        hint = subject_hint(q)
        blocks.append(
            f"Question ID: {q['Question ID']}\n"
            f"Subject Hint: {hint}\n"
            f"Q: {q['Questions']}\n"
            f"(A) {q.get('Option A','')}  "
            f"(B) {q.get('Option B','')}  "
            f"(C) {q.get('Option C','')}  "
            f"(D) {q.get('Option D','')}"
        )
    return f"""RETRY. Return ONLY valid JSON. One row per Question ID.

ONLY these 9 Subjects exist (copy exactly):
{chr(10).join(f'- {s}' for s in BPSC_SUBJECTS if s != 'Miscellaneous')}
- Miscellaneous (only for corrupted questions)

Topic must come from that Subject's own list. Subtopic = specific fact from the question.
Reasoning/puzzle/arithmetic word-problems → "General Mental Ability & CSAT" always.
Bihar-specific content stays inside the matching national subject (History of India,
Indian Polity, Indian Economy), NOT a separate Bihar subject.

{chr(10).join(blocks)}

JSON: {{"classifications":[{{"Question ID":"...","Subject":"...","Topic":"...","Subtopic":"..."}}]}}
""".strip()


# ============================================================
# OLLAMA CALL
# ============================================================

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
                        "Subject": {"type": "string", "enum": BPSC_SUBJECTS},
                        "Topic": {"type": "string"},
                        "Subtopic": {"type": "string"},
                    },
                    "required": ["Question ID", "Subject", "Topic", "Subtopic"],
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
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in [("[", "]"), ("{", "}")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in Ollama response")


def normalize_row(row: Dict) -> Dict[str, str]:
    if not isinstance(row, dict):
        return {"Question ID": "", "Subject": "", "Topic": "", "Subtopic": ""}
    return {
        "Question ID": str(row.get("Question ID", "")).strip(),
        "Subject": str(row.get("Subject", "")).strip(),
        "Topic": str(row.get("Topic", "")).strip(),
        "Subtopic": str(row.get("Subtopic", "")).strip(),
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    results: List[Dict[str, str]],
    expected_ids: List[str],
    question_by_id: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    expected = set(expected_ids)
    accepted: Dict[str, Dict[str, str]] = {}
    errors: List[str] = []

    for raw in results:
        row = normalize_row(raw)
        qid = row["Question ID"]

        if qid not in expected:
            continue
        if qid in accepted:
            errors.append(f"{qid}: duplicate")
            continue

        # Apply hard overrides (GMA and Current Affairs only)
        hint = subject_hint(question_by_id[qid])
        if hint in HARD_OVERRIDE_SUBJECTS and row["Subject"] != hint:
            print(f"  OVERRIDE {qid}: {row['Subject']} → {hint}")
            row["Subject"] = hint
            # Also fix Topic if it's no longer valid for the new Subject
            valid_topics = {t.casefold() for t in SUBJECT_TOPICS.get(hint, [])}
            if row["Topic"].casefold() not in valid_topics:
                row["Topic"] = SUBJECT_TOPICS[hint][0]

        if row["Subject"] not in BPSC_SUBJECTS:
            errors.append(f"{qid}: invalid Subject '{row['Subject']}'")
            continue

        if not row["Topic"]:
            errors.append(f"{qid}: empty Topic")
            continue

        if not row["Subtopic"]:
            errors.append(f"{qid}: empty Subtopic")
            continue

        if row["Subtopic"].casefold() == row["Subject"].casefold():
            errors.append(f"{qid}: Subtopic duplicates Subject")
            continue

        if row["Subtopic"].casefold() == row["Topic"].casefold():
            errors.append(f"{qid}: Subtopic duplicates Topic")
            continue

        # Log Topic mismatches (soft — not a failure, just for review)
        valid_topics = {t.casefold() for t in SUBJECT_TOPICS.get(row["Subject"], [])}
        if row["Topic"].casefold() not in valid_topics:
            _topic_mismatches.append({
                "Question ID": qid,
                "Subject": row["Subject"],
                "Topic": row["Topic"],
                "Subtopic": row["Subtopic"],
                "Question": question_by_id[qid].get("Questions", ""),
            })

        accepted[qid] = row

    missing = sorted(expected - set(accepted))
    errors.extend(f"{qid}: missing" for qid in missing)
    return accepted, errors


# ============================================================
# OLLAMA DISPATCH
# ============================================================

def call_ollama(questions: List[Dict[str, str]], retry: bool = False) -> List[Dict[str, str]]:
    prompt = build_retry_prompt(questions) if retry else build_prompt(questions)
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=make_schema(),
        options={
            "num_ctx": NUM_CTX,
            "temperature": 0,
            "top_p": 0.2,
            "num_predict": max(256, len(questions) * 80),
        },
        keep_alive=KEEP_ALIVE,
    )
    content = response.get("message", {}).get("content", "")
    if not content:
        raise ValueError("Ollama returned empty content")
    payload = extract_json(content)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("classifications"), list):
        return payload["classifications"]
    raise ValueError("Unexpected JSON structure")


def classify_batch(
    questions: List[Dict[str, str]],
    retry: bool = False,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    ids = [q["Question ID"] for q in questions]
    question_by_id = {q["Question ID"]: q for q in questions}
    raw = call_ollama(questions, retry=retry)
    return validate_results(raw, ids, question_by_id)


def process_one(q: Dict[str, str]) -> Tuple[Dict[str, str] | None, str | None]:
    last_reason = "classification failed"
    for attempt in range(INDIVIDUAL_RETRIES + 1):
        try:
            accepted, errors = classify_batch([q], retry=(attempt > 0))
            if q["Question ID"] in accepted:
                return accepted[q["Question ID"]], None
            last_reason = "; ".join(errors) or last_reason
        except Exception as exc:
            last_reason = str(exc)
        if attempt < INDIVIDUAL_RETRIES:
            time.sleep(0.15)
    return None, last_reason


# ============================================================
# QUESTION SELECTION
# ============================================================

def select_questions(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if ONLY_QUESTIONS:
        allowed = {int(x) for x in ONLY_QUESTIONS}
        return [
            r for r in rows
            if re.search(r"\d+", r["Question ID"])
            and int(re.search(r"\d+", r["Question ID"]).group()) in allowed
        ]
    selected = []
    for r in rows:
        m = re.search(r"\d+", r["Question ID"])
        if not m:
            continue
        if START_QUESTION <= int(m.group()) <= END_QUESTION:
            selected.append(r)
    return selected


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(f"BPSC OLLAMA CLASSIFIER {SCRIPT_VERSION}")
    print(f"Model: {MODEL}")
    print(f"Subjects: {len(BPSC_SUBJECTS)}  |  "
          f"Total Topics: {sum(len(v) for v in SUBJECT_TOPICS.values())}")
    print(f"Batch: {BATCH_SIZE}  |  Recovery: {RECOVERY_BATCH_SIZE}")
    print("=" * 60)

    questions = load_questions()
    existing = load_existing()
    failures = load_failed()

    print(f"Source questions     : {len(questions)}")
    print(f"Already classified  : {len(existing)}")
    print(f"Currently failed    : {len(failures)}")

    if RETRY_FAILED_ONLY:
        failed_ids = set(failures.keys())
        if not failed_ids:
            print("RETRY_FAILED_ONLY is on but failures file is empty.")
            return
        selected = [q for q in questions if q["Question ID"] in failed_ids]
        print(f"RETRY_FAILED_ONLY: retrying {len(selected)} question(s)")
    else:
        selected = select_questions(questions)
        if not FORCE_RECLASSIFY_ALL:
            already_done = set(existing.keys())
            before = len(selected)
            selected = [q for q in selected if q["Question ID"] not in already_done]
            skipped = before - len(selected)
            if skipped:
                print(f"Skipping {skipped} already-classified (set FORCE_RECLASSIFY_ALL=True to redo)")

    print(f"Selected for run    : {len(selected)}")
    if not selected:
        print("Nothing to classify. Done.")
        return

    for batch_start in range(0, len(selected), BATCH_SIZE):
        batch = selected[batch_start: batch_start + BATCH_SIZE]
        ids = [q["Question ID"] for q in batch]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(selected) + BATCH_SIZE - 1) // BATCH_SIZE

        print("=" * 60)
        print(f"BATCH {batch_num}/{total_batches}: {ids[0]} – {ids[-1]}")
        print("=" * 60)

        accepted: Dict[str, Dict[str, str]] = {}
        errors: List[str] = []

        try:
            accepted, errors = classify_batch(batch, retry=False)
        except Exception as exc:
            print(f"  Main call failed: {exc}")
            try:
                accepted, errors = classify_batch(batch, retry=True)
                print("  Compact retry succeeded.")
            except Exception as exc2:
                print(f"  Compact retry failed: {exc2}")
                accepted, errors = {}, [str(exc2)]

        for qid, row in accepted.items():
            existing[qid] = row
            failures.pop(qid, None)
            print(f"  OK  {qid}: {row['Subject']} | {row['Topic']} | {row['Subtopic']}")

        missing = [qid for qid in ids if qid not in accepted]
        if missing:
            print(f"  Recovery needed for {len(missing)} item(s).")
            for err in errors[:10]:
                print(f"    {err}")

            missing_qs = [q for q in batch if q["Question ID"] in set(missing)]
            recovered: Dict[str, Dict[str, str]] = {}

            for chunk_start in range(0, len(missing_qs), RECOVERY_BATCH_SIZE):
                chunk = missing_qs[chunk_start: chunk_start + RECOVERY_BATCH_SIZE]
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
        if _topic_mismatches:
            save_topic_review(_topic_mismatches)
            _topic_mismatches.clear()

        unresolved = sum(1 for qid in ids if qid not in existing)
        print(f"  Batch {batch_num}: {len(ids) - unresolved} OK, {unresolved} unresolved")

    print("=" * 60)
    print("DONE")
    print(f"Output  : {OUTPUT_FILE}")
    print(f"Failures: {FAILED_FILE}")
    if os.path.exists(TOPIC_REVIEW_FILE):
        print(f"Review  : {TOPIC_REVIEW_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()