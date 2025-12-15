# ingest_ipc.py
import re
import os
from datetime import datetime
import uuid
from tqdm import tqdm
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------- CONFIG ----------------
PDF_PATH = "Indian Penal Code Book.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "legal_knowledge"
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
CHUNK_CHAR_SIZE = 2000  

# ---------------- SETUP ----------------
print(" Initializing embedding model and Chroma client...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f" Chroma collection ready at {CHROMA_DIR}/{COLLECTION_NAME}")

# ---------------- HELPERS ----------------
def clean_text(raw: str) -> str:
    text = re.sub(r'\r\n?', '\n', raw)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace to single spaces
    text = text.replace('\x0c', ' ')            # form feed from pdf maybe
    text = re.sub(r'Page\s*\d+', '', text, flags=re.I)  # remove simple page markers
    text = text.strip()
    return text

# def split_into_sections(text: str):
#     """
#     Finds occurrences like 'Section 420.—Cheating...' or 'Sec. 420. - Cheating' etc.
#     Returns list of dicts: {"section": "420", "chunk_id": 0, "text": "..."}
#     """
#     # pattern: capture numeric/alpha section number and stop char (., —, -)
#     pattern = re.compile(r'(?:Section|SECTION|Sec(?:tion)?|Sec\.|S\.)\s*(\d+[A-Za-z]?)\s*[\.\-–—\s]')
#     matches = list(pattern.finditer(text))
#     sections = []

#     if not matches:
#         # fallback: attempt to find '420.' style occurrences
#         pattern2 = re.compile(r'\b(\d{1,4}[A-Za-z]?)\.\s+')
#         matches2 = list(pattern2.finditer(text))
#         if matches2:
#             matches = matches2

#     for i, m in enumerate(matches):
#         sec_num = m.group(1)
#         start = m.end()
#         end = matches[i+1].start() if i+1 < len(matches) else len(text)
#         sec_body = text[start:end].strip()

#         # Normalize the section number to canonical form: only numbers and trailing letter (e.g., 420, 498A)
#         sec_norm = re.sub(r'\W+', '', sec_num).upper()

#         # split long section body into smaller chunks
#         if len(sec_body) <= CHUNK_CHAR_SIZE:
#             sections.append({"section": sec_norm, "chunk_id": 0, "text": sec_body})
#         else:
#             # chunk by chars, try to break at sentence boundaries
#             start_idx = 0
#             chunk_id = 0
#             while start_idx < len(sec_body):
#                 end_idx = start_idx + CHUNK_CHAR_SIZE
#                 # try to expand to next sentence end if possible ('. ')
#                 if end_idx < len(sec_body):
#                     next_period = sec_body.rfind('.', start_idx, end_idx)
#                     if next_period > start_idx + 100:  # reasonable sentence end
#                         end_idx = next_period + 1
#                 chunk_text = sec_body[start_idx:end_idx].strip()
#                 sections.append({"section": sec_norm, "chunk_id": chunk_id, "text": chunk_text})
#                 chunk_id += 1
#                 start_idx = end_idx

#     return sections

def split_into_sections(text: str):
    """
    Improved splitter: captures all section formats like
    'Section 220.', 'SECTION 220 –', 'Sec. 220', 'S. 220', etc.
    """
    pattern = re.compile(
        r'(?:Section|SECTION|Sec\.?|S\.)\s*(\d+[A-Za-z]?)\s*(?:[–\-\.]|\b)',
        flags=re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    sections = []

    for i, m in enumerate(matches):
        sec_num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_body = text[start:end].strip()

        # normalize numeric id
        sec_norm = re.sub(r'\W+', '', sec_num).upper()

        # skip broken captures (e.g. '22' misread as '220')
        if not re.match(r'^\d+[A-Z]?$' , sec_norm):
            continue

        if len(sec_body) < 50:
            continue

        sections.append({
            "section": sec_norm,
            "chunk_id": 0,
            "text": sec_body
        })

    print(f" Extracted {len(sections)} IPC sections.")
    return sections


def ingest_pdf(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    print(f" Extracting text from {path} ...")
    raw = extract_text(path)
    cleaned = clean_text(raw)
    print(" Splitting into sections...")
    sections = split_into_sections(cleaned)
    print(f" Found {len(sections)} section-chunks.")

    print(" Embedding and uploading to Chroma...")
    batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []
    for s in tqdm(sections, desc="Embedding sections"):
        text = s["text"]
        if not text or len(text.strip()) < 10:
            continue
        emb = embedding_model.encode(text).tolist()
        doc_id = f"{s['section']}_{s['chunk_id']}_{uuid.uuid4().hex}"

        meta = {
            "section": s["section"],
            "chunk": s["chunk_id"],
            "source": "IPC",
            "ingested_at": datetime.utcnow().isoformat()
        }
        # add to batch
        batch_ids.append(doc_id)
        batch_docs.append(text)
        batch_embs.append(emb)
        batch_metas.append(meta)

        # flush in small batches to avoid memory explosion
        if len(batch_ids) >= 64:
            collection.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs, metadatas=batch_metas)
            batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    # flush remaining
    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs, metadatas=batch_metas)

    print(" Ingestion complete. Total uploaded chunks:", len(collection.get().get("ids", [])))

# ---------------- RUN ----------------
if __name__ == "__main__":
    ingest_pdf(PDF_PATH)
    print("All done.")
