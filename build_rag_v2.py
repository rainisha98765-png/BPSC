


import os, sys, json, hashlib
import fitz       # pymupdf
import chromadb
import ollama

# ── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR    = os.path.join(SCRIPT_DIR, "reference materials")
RAG_DB_DIR = os.path.join(SCRIPT_DIR, "rag_db")
STATE_FILE = os.path.join(RAG_DB_DIR, "indexed_files.json")

CHUNK_SIZE      = 300
CHUNK_OVERLAP   = 50
MIN_CHUNK_WORDS = 25
EMBED_MODEL     = "nomic-embed-text"
BATCH           = 50

# ── PDF MANIFEST ─────────────────────────────────────────────────────────────
# Run with LIST_ONLY=True first to see your exact filenames,
# then update this list to match.
LIST_ONLY = False  # <-- change to False after confirming filenames

PDF_MANIFEST = [
    # HISTORY OF INDIA
    ("A Brief History of Modern India 2020-21 Edition_nodrm.pdf",               "History of India"),
    ("Copy of our past class 7 social science.pdf",               "History of India"),
    ("History6.pdf",               "History of India"),
    ("Introduction to Indian Art  _.pdf",               "History of India"),
    ("NCERT-Class-10-History.pdf",               "History of India"),
    ("NCERT-Class-11-Heritage-Crafts.pdf",               "History of India"),
    ("spectrum_modern_history.pdf","History of India"),
    ("NCERT-Class-11-History.pdf",               "History of India"),
    ("NCERT-Class-12-History-Part-1 (1).pdf",               "History of India"),
    ("NCERT-Class-12-History-Part-2.pdf",               "History of India"),
    ("NCERT-Class-12-History-Part-3.pdf",               "History of India"),
    ("OUR PAST PART 2 CLASS 8 SOCIAL SCIENCE.pdf",               "History of India"),
    
    
    # INDIAN POLITY
    ("INDIAN POLITY M. LAXMIKANT pdf.pdf",             "Indian Polity"),
    
    # INDIAN ECONOMY
    ("NCERT-Class-10-Economics.pdf",                "Indian Economy"),
    ("NCERT-Class-11-Economics.pdf",                "Indian Economy"),
    ("NCERT-Class-12-Economics-Part-1.pdf",                "Indian Economy"),
    ("NCERT-Class-12-Economics-Part-2.pdf",          "Indian Economy"),
    ("economics class 9.pdf" ,                      "Indian Economy"), 


    
    # GEOGRAPHY
    ("Fundamental of Physical Geography (Class XI) 2.pdf",               "Geography"),
    ("Fundamentals of Human Geography (Class XII) 1.pdf",               "Geography"),
    ("Geography.pdf",                   "Geography" ),
    ("India People and Economy (Class XII).pdf",                       "Geography" ),
    ("India Physical Environment (Class XI) 2.pdf",                       "Geography"),
    ("NCERT-Class-10-Geography.pdf",                         "Geography"),
    (" NCERT-Class-9-Geography-1.pdf",                             "Geography"),
 ("g-c-leong-geography-freeupscmaterials.org_.pdf",                 "Geography"),
    # GENERAL SCIENCE
    (" Copy of science class 7.pdf",               "General Science"),
    ("NCERT-Class-10-Science.pdf",               "General Science"),
    (" NCERT-Class-11-Biology.pdf",               "General Science"),
    (" NCERT-Class-11-Chemistry-Part-1.pdf",               "General Science"),
    ("NCERT-Class-11-Physics-Part-1.pdf",               "General Science"),
    (" NCERT-Class-11-Physics-Part-2.pdf",               "General Science"),
    (" NCERT-Class-12-Biology.pdf",               "General Science"),
    ("NCERT-Class-12-Chemistry-Part-1.pdf",               "General Science"),
    (" NCERT-Class-12-Chemistry-Part-2.pdf",               "General Science"),
    ("NCERT-Class-12-Physics-Part-1.pdf",               "General Science"),
    ("NCERT-Class-12-Physics-Part-2.pdf",               "General Science"),
    (" class 9 science.pdf",                             "General Science"),
    ( "science class 6.pdf",                     "General Science"),
    ("science class 8.pdf",                     "General Science" ),
    # ENVIRONMENT & ECOLOGY
    ("shankar_environment.pdf",   "Environment & Ecology"),
    # ART & CULTURE
    ("nitin_singhania.pdf",       "Art & Culture"),
]

# ── HELPERS ──────────────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE,"r",encoding="utf-8") as f: return json.load(f)
    return {"indexed":[]}

def save_state(s):
    os.makedirs(RAG_DB_DIR,exist_ok=True)
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(s,f,indent=2)

def chunk_text(text):
    words,chunks,i = text.split(),[],0
    while i<len(words):
        c=" ".join(words[i:i+CHUNK_SIZE])
        if len(c.split())>=MIN_CHUNK_WORDS: chunks.append(c)
        i+=CHUNK_SIZE-CHUNK_OVERLAP
    return chunks

def clean_text(text):
    return " ".join(l.strip() for l in text.splitlines()
                    if l.strip() and not l.strip().isdigit() and len(l.strip())>=3)

def embed(text):
    return ollama.embeddings(model=EMBED_MODEL,prompt=text)["embedding"]

def stable_id(fname,idx):
    return hashlib.md5(f"{fname}::{idx}".encode()).hexdigest()

def flush(col,ids,docs,metas):
    embs=[]
    for d in docs:
        try: embs.append(embed(d))
        except: embs.append([0.0]*768)
    col.upsert(ids=ids,embeddings=embs,documents=docs,metadatas=metas)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("BPSC RAG DATABASE BUILDER")
    print(f"PDF folder : {PDF_DIR}")
    print(f"DB folder  : {RAG_DB_DIR}")
    print("="*60)

    if not os.path.exists(PDF_DIR):
        print(f"\nERROR: Folder not found: {PDF_DIR}")
        print("Make sure your PDFs are in a folder called 'reference materials'")
        print("next to this script.")
        sys.exit(1)

    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    print(f"\nFound {len(pdfs)} PDFs in reference materials:\n")
    for p in pdfs: print(f"  {p}")
    print()

    if LIST_ONLY:
        print("LIST_ONLY=True — update PDF_MANIFEST with the filenames above,")
        print("then set LIST_ONLY=False and rerun.")
        return

    try:
        ollama.embeddings(model=EMBED_MODEL,prompt="test")
        print(f"Ollama OK — {EMBED_MODEL} ready\n")
    except Exception as e:
        print(f"ERROR: Ollama not reachable — {e}")
        print("Open the Ollama app and run: ollama pull nomic-embed-text")
        sys.exit(1)

    os.makedirs(RAG_DB_DIR,exist_ok=True)
    client  = chromadb.PersistentClient(path=RAG_DB_DIR)
    col     = client.get_or_create_collection("pdf_chunks",metadata={"hnsw:space":"cosine"})
    print(f"ChromaDB: {col.count()} chunks already stored\n")

    state     = load_state()
    done_set  = set(state.get("indexed",[]))
    total_new = 0

    for fname, subject in PDF_MANIFEST:
        path = os.path.join(PDF_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {fname}")
            continue
        if fname in done_set:
            print(f"  SKIP (indexed):   {fname}")
            continue

        print(f"\n  Processing: {fname}  [{subject}]")
        try: doc = fitz.open(path)
        except Exception as e:
            print(f"    ERROR: {e}"); continue

        fc=0; ids,docs,metas=[],[],[]
        for pn,page in enumerate(doc):
            text=clean_text(page.get_text())
            if len(text.split())<MIN_CHUNK_WORDS: continue
            for ci,chunk in enumerate(chunk_text(text)):
                ids.append(stable_id(fname,pn*1000+ci))
                docs.append(chunk)
                metas.append({"source":fname,"subject":subject,"page":pn+1})
                fc+=1
                if len(ids)>=BATCH:
                    flush(col,ids,docs,metas)
                    total_new+=len(ids)
                    print(f"    ... {fc} chunks",end="\r")
                    ids,docs,metas=[],[],[]
        if ids:
            flush(col,ids,docs,metas)
            total_new+=len(ids)
    
        print(f"    Done: {fc} chunks from {len(doc)} pages     ")
        doc.close()
        state["indexed"].append(fname)
        save_state(state)

    print("\n"+"="*60)
    print(f"Build complete. New: {total_new}  Total: {col.count()}")
    print("="*60)
    print("\nNext step: python classify_with_rag.py")

if __name__=="__main__":
    main()
