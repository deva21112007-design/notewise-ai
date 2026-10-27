import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from document_processor import extract_text, SUPPORTED_EXTENSIONS
from chunker import chunk_pages

# ── Project settings ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT_DIR / "chroma_db")
OUTLINE_FILE = ROOT_DIR / "chroma_db" / "document_outline.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "notewise_notes"

# Distance threshold: ChromaDB uses cosine distance (0 = identical).
# Results with distance > this value are considered too weak to use.
RELEVANCE_THRESHOLD = 0.65


# ── Document Outline Extraction ──────────────────────────────────────────────
def extract_document_outline(pages: list[dict], file_name: str) -> dict:
    """
    Scan pages to detect units, syllabus topics, and high-level structure.
    Saves and returns a structured outline dictionary.
    """
    unit_regex = re.compile(
        r"^\s*(UNIT\s*[-–:]?\s*(?:[I|V|X]+|\d+))\b\s*(.*)",
        re.IGNORECASE | re.MULTILINE,
    )
    roman_conv = {
        "UNIT1": "UNIT I", "UNIT2": "UNIT II", "UNIT3": "UNIT III",
        "UNIT4": "UNIT IV", "UNIT5": "UNIT V",
        "UNITI": "UNIT I", "UNITII": "UNIT II", "UNITIII": "UNIT III",
        "UNITIV": "UNIT IV", "UNITV": "UNIT V",
    }

    outline_units = []
    seen = set()

    for p in pages:
        lines = [l.strip() for l in p.get("text", "").split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = unit_regex.match(line)
            if m:
                u_tag = m.group(1).upper()
                u_norm = re.sub(r"[^A-Z0-9]", "", u_tag)
                canonical = roman_conv.get(u_norm, u_tag)
                if canonical in seen:
                    continue
                seen.add(canonical)

                title = m.group(2).strip()
                if not title and i + 1 < len(lines):
                    title = lines[i + 1]
                if len(title) > 80:
                    title = title[:80]

                sub = lines[i + 2] if i + 2 < len(lines) else ""
                if len(sub) > 100:
                    sub = sub[:100]

                outline_units.append({
                    "unit": canonical,
                    "title": title or "Module",
                    "preview": sub,
                    "page": p["page"],
                })

    outline_data = {
        "file_name": Path(file_name).name,
        "total_pages": len(pages),
        "units": outline_units,
    }

    try:
        OUTLINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTLINE_FILE, "w", encoding="utf-8") as f:
            json.dump(outline_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save outline: {e}")

    return outline_data


def get_document_outline() -> dict | None:
    """Retrieve saved document outline if it exists."""
    if OUTLINE_FILE.exists():
        try:
            with open(OUTLINE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── Indexing ─────────────────────────────────────────────────────────────────
def build_index(file_paths, api_key: str | None = None):
    """
    Extract, chunk, embed, and store one or multiple documents in ChromaDB.
    Supports a single file path or a list of file paths: PDF, DOCX, PPTX, TXT, images.
    Pass api_key to enable Gemini Vision OCR for scanned/image files.
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [Path(file_paths)]
    else:
        file_paths = [Path(fp) for fp in file_paths]

    all_chunks = []
    all_outlines = []
    total_pages_count = 0

    for fp in file_paths:
        file_name = fp.name
        print(f"\n1. Extracting document text from {file_name}...")
        pages = extract_text(fp, api_key=api_key)
        total_pages_count += len(pages)

        # Extract outline
        outline = extract_document_outline(pages, file_name)
        all_outlines.append(outline)

        # Chunk pages and tag source
        file_chunks = chunk_pages(pages)
        for c in file_chunks:
            c["source"] = file_name
        all_chunks.extend(file_chunks)

    # Save combined outline if multiple documents
    if len(all_outlines) > 1:
        combined_outline = {
            "files": [o["file_name"] for o in all_outlines],
            "total_files": len(all_outlines),
            "total_pages": total_pages_count,
            "documents": all_outlines,
        }
        try:
            with open(OUTLINE_FILE, "w", encoding="utf-8") as f:
                json.dump(combined_outline, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save multi-file outline: {e}")

    if not all_chunks:
        raise ValueError("No text chunks were created from the uploaded document(s).")

    print(f"\nTotal Pages across all files: {total_pages_count}")
    print(f"Total Chunks across all files: {len(all_chunks)}")

    print("\nLoading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nCreating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
    ).tolist()

    print("\nSaving embeddings to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection so rerunning this script
    # does not create duplicate records.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(all_chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "page": chunk["page"],
                "chunk_id": chunk.get("chunk_id", i),
                "source": chunk.get("source", "Notes"),
            }
            for i, chunk in enumerate(all_chunks)
        ],
    )

    names_str = ", ".join(fp.name for fp in file_paths)
    print(f"\nSUCCESS! Indexed {len(all_chunks)} chunks from {len(file_paths)} file(s): {names_str}.")
    return collection, model


# ── Retrieval ────────────────────────────────────────────────────────────────
def search_notes(query, collection, model, top_k=10):
    """
    Advanced Hybrid Retrieval:
    1. Query normalization (handling apostrophes, e.g. k'map -> k-map, kmap).
    2. Document Structure & Overview detection:
       If the query asks about units, syllabus, chapters, topics, or index,
       retrieve the Unit introduction chunks directly.
    3. Specific Unit queries:
       If the query targets a specific unit (e.g. Unit 3 / Unit III),
       prioritize that unit's pages.
    4. Keyword Boosting:
       Exact match boosting for technical terms (e.g., k-map, flip-flop, adder).
    5. Dense Semantic Search:
       Cosine similarity using all-MiniLM-L6-v2 embeddings.
    6. Fusion & Deduplication:
       Merges and ranks the most informative passages.
    """
    count = collection.count()
    if count == 0:
        return []

    q_lower = query.lower()
    # Normalize variants like k'map, k-map, kmap
    q_normalized = q_lower.replace("'", "").replace("’", "")
    terms = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_normalized)]

    stored = None
    passages_dict = {}  # key -> dict

    def get_stored_data():
        nonlocal stored
        if stored is None:
            stored = collection.get(include=["documents", "metadatas"])
        return stored

    # ── Strategy A: Overview / Unit-list detection ───────────────────────────
    is_overview = any(term in q_lower for term in [
        "unit name", "units in", "what units", "how many units", "list the units",
        "what are the units", "all units", "syllabus", "table of contents",
        "index", "chapters", "outline", "what does this note cover", "overview",
        "topics covered", "how many chapters"
    ])

    if is_overview:
        outline = get_document_outline()
        unit_pages = set()
        if outline:
            if "documents" in outline:
                for d in outline["documents"]:
                    unit_pages.update({u["page"] for u in d.get("units", [])})
            else:
                unit_pages.update({u["page"] for u in outline.get("units", [])})
        data = get_stored_data()
        for doc, meta in zip(data["documents"], data["metadatas"]):
            p = meta.get("page", 1)
            # Include detected unit pages or beginning pages
            if p in unit_pages or p in {1, 2, 3}:
                key = (meta.get("source", "Notes"), p, meta.get("chunk_id", 0))
                passages_dict[key] = {
                    "text": doc,
                    "page": p,
                    "source": meta.get("source", "Notes"),
                    "score": 0.98,
                }

    # ── Strategy B: Specific Unit query ──────────────────────────────────────
    unit_match = re.search(r"\bunit\s*(i{1,3}|iv|v|[1-5])\b", q_lower)
    if unit_match:
        unit_str = unit_match.group(1)
        roman_map = {
            "i": (1, "I"), "1": (1, "I"),
            "ii": (2, "II"), "2": (2, "II"),
            "iii": (3, "III"), "3": (3, "III"),
            "iv": (4, "IV"), "4": (4, "IV"),
            "v": (5, "V"), "5": (5, "V"),
        }
        number, roman = roman_map.get(unit_str, (1, "I"))
        pattern = re.compile(rf"\bUNIT\s*[-:]?\s*(?:{roman}|{number})\b", re.IGNORECASE)
        data = get_stored_data()
        for doc, meta in zip(data["documents"], data["metadatas"]):
            if pattern.search(doc[:300]):
                key = (meta.get("source", "Notes"), meta["page"], meta.get("chunk_id", 0))
                passages_dict[key] = {
                    "text": doc,
                    "page": meta["page"],
                    "source": meta.get("source", "Notes"),
                    "score": 0.95,
                }

    # ── Strategy C: Key Technical Term Boost ─────────────────────────────────
    tech_keywords = []
    if "kmap" in q_normalized or "k map" in q_lower or "karnaugh" in q_lower:
        tech_keywords.extend(["kmap", "k-map", "karnaugh"])
    if "flip flop" in q_lower or "flip-flop" in q_lower:
        tech_keywords.extend(["flip-flop", "flip flop"])
    if "multiplexer" in q_lower or "mux" in q_lower:
        tech_keywords.extend(["multiplexer", "mux"])

    if tech_keywords:
        data = get_stored_data()
        for doc, meta in zip(data["documents"], data["metadatas"]):
            doc_lower = doc.lower()
            if any(tk in doc_lower for tk in tech_keywords):
                key = (meta.get("source", "Notes"), meta["page"], meta.get("chunk_id", 0))
                if key not in passages_dict:
                    passages_dict[key] = {
                        "text": doc,
                        "page": meta["page"],
                        "source": meta.get("source", "Notes"),
                        "score": 0.88,
                    }

    # ── Strategy D: Dense Semantic Vector Search ─────────────────────────────
    # Search with original query + normalized variant
    query_variants = [query]
    if "k'map" in query.lower() or "k map" in query.lower():
        query_variants.append("Karnaugh Map K-map Boolean simplification logic diagram")

    for q_var in query_variants:
        embedding = model.encode([q_var]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=min(top_k * 2, count),
            include=["documents", "metadatas", "distances"],
        )

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - distance
            if distance > RELEVANCE_THRESHOLD:
                continue

            key = (meta.get("source", "Notes"), meta["page"], meta.get("chunk_id", 0))
            if key not in passages_dict or passages_dict[key]["score"] < similarity:
                passages_dict[key] = {
                    "text": doc,
                    "page": meta["page"],
                    "source": meta.get("source", "Notes"),
                    "score": round(similarity, 3),
                }

    # Sort all candidate passages by score descending
    sorted_passages = sorted(
        passages_dict.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return sorted_passages[:top_k]


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    default_pdf = ROOT_DIR / "data" / "dpco notes.pdf"
    entered_path = input(
        f"Enter file path (press Enter for {default_pdf}): "
    ).strip().strip('"')

    pdf_path = Path(entered_path) if entered_path else default_pdf

    if not pdf_path.is_absolute():
        pdf_path = ROOT_DIR / pdf_path

    api_key = os.getenv("GEMINI_API_KEY")
    try:
        collection, model = build_index(pdf_path, api_key=api_key)

        while True:
            query = input(
                "\nAsk a question about your notes (or type 'exit'): "
            ).strip()

            if query.lower() == "exit":
                break

            if query:
                hits = search_notes(query, collection, model)
                for i, h in enumerate(hits, 1):
                    print(
                        f"\n[{i}] Page {h['page']} | Score {h['score']:.3f}\n"
                        f"{h['text'][:300]}"
                    )

    except Exception as error:
        print(f"\nERROR: {error}")