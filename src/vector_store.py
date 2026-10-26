import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from document_processor import extract_text, SUPPORTED_EXTENSIONS
from chunker import chunk_pages

# ── Project settings ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT_DIR / "chroma_db")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "notewise_notes"

# Distance threshold: ChromaDB uses cosine distance (0 = identical).
# Results with distance > this value are considered too weak to use.
RELEVANCE_THRESHOLD = 0.65


# ── Indexing ─────────────────────────────────────────────────────────────────
def build_index(file_path, api_key: str | None = None):
    """
    Extract, chunk, embed, and store a document in ChromaDB.
    Supports PDF (text + scanned), DOCX, PPTX, TXT, and images.
    Pass api_key to enable Gemini Vision OCR for scanned/image files.
    """
    print("\n1. Extracting document text...")
    pages = extract_text(file_path, api_key=api_key)

    print("2. Splitting text into sentence-aware chunks...")
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text chunks were created from this document.")

    print(f"   Pages:  {len(pages)}")
    print(f"   Chunks: {len(chunks)}")

    print("\n3. Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\n4. Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
    ).tolist()

    print("\n5. Saving embeddings to ChromaDB...")
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
        ids=[f"chunk-{chunk['chunk_id']}" for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "source": Path(file_path).name,
            }
            for chunk in chunks
        ],
    )

    print(f"\nSUCCESS! Indexed {len(chunks)} chunks from {Path(file_path).name}.")
    return collection, model


# ── Retrieval ────────────────────────────────────────────────────────────────
def search_notes(query, collection, model, top_k=6):
    """
    Retrieve the most relevant passages for a query.

    Strategy:
    1. If the query mentions a specific unit (e.g. "Unit I", "unit 3"),
       first try a keyword match on the stored text so structured
       curriculum content is always found.
    2. Fall back to semantic (cosine) similarity search for everything else.
    3. Filter out passages whose cosine distance exceeds RELEVANCE_THRESHOLD
       to avoid hallucination from irrelevant context.

    Returns a list of dicts: {text, page, source, score}.
    """
    count = collection.count()
    if count == 0:
        return []

    # ── Step 1: Unit-keyword shortcut ────────────────────────────────────────
    unit_match = re.search(
        r"\bunit\s*(i{1,3}|iv|v|[1-5])\b",
        query.lower(),
    )

    if unit_match:
        unit_str = unit_match.group(1)
        roman_map = {
            "i": (1, "I"), "1": (1, "I"),
            "ii": (2, "II"), "2": (2, "II"),
            "iii": (3, "III"), "3": (3, "III"),
            "iv": (4, "IV"), "4": (4, "IV"),
            "v": (5, "V"), "5": (5, "V"),
        }
        number, roman = roman_map[unit_str]

        stored = collection.get(include=["documents", "metadatas"])
        pattern = re.compile(
            rf"\bUNIT\s*[-:]?\s*(?:{roman}|{number})\b",
            re.IGNORECASE,
        )

        keyword_passages = [
            {
                "text": doc,
                "page": meta["page"],
                "source": meta.get("source", "Notes"),
                "score": 1.0,  # keyword hits treated as perfect match
            }
            for doc, meta in zip(stored["documents"], stored["metadatas"])
            if pattern.search(doc[:300])
        ]

        if keyword_passages:
            return keyword_passages[:top_k]

    # ── Step 2: Semantic search ───────────────────────────────────────────────
    embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Convert cosine distance → similarity score (0–1, higher = better).
        similarity = 1.0 - distance

        # Skip passages that are too dissimilar to be useful.
        if distance > RELEVANCE_THRESHOLD:
            continue

        passages.append({
            "text": doc,
            "page": meta["page"],
            "source": meta.get("source", "Notes"),
            "score": round(similarity, 3),
        })

    return passages


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