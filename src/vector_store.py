
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from document_processor import extract_text_from_pdf
from chunker import chunk_pages

# Project settings
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT_DIR / "chroma_db")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "notewise_notes"


def build_index(pdf_path):
    print("\n1. Extracting PDF text...")
    pages = extract_text_from_pdf(pdf_path)

    print("2. Splitting text into chunks...")
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"   Pages: {len(pages)}")
    print(f"   Chunks: {len(chunks)}")

    print("\n3. Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\n4. Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    ).tolist()

    print("\n5. Saving embeddings to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection so rerunning this script
    # doesn't create duplicate records.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"chunk-{chunk['chunk_id']}" for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "source": Path(pdf_path).name
            }
            for chunk in chunks
        ]
    )

    print(f"\nSUCCESS! Indexed {len(chunks)} chunks.")
    return collection, model



import re


from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from document_processor import extract_text_from_pdf
from chunker import chunk_pages

# Project settings
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT_DIR / "chroma_db")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "notewise_notes"


def build_index(pdf_path):
    print("\n1. Extracting PDF text...")
    pages = extract_text_from_pdf(pdf_path)

    print("2. Splitting text into chunks...")
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"   Pages: {len(pages)}")
    print(f"   Chunks: {len(chunks)}")

    print("\n3. Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\n4. Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    ).tolist()

    print("\n5. Saving embeddings to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection so rerunning this script
    # doesn't create duplicate records.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"chunk-{chunk['chunk_id']}" for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "source": Path(pdf_path).name
            }
            for chunk in chunks
        ]
    )

    print(f"\nSUCCESS! Indexed {len(chunks)} chunks.")
    return collection, model



import re


def search_notes(query, collection, model, top_k=5):
    # Return passages about a requested unit when possible.
    match = re.search(
        r"\bunit\s*(i{1,3}|iv|v|[1-5])\b",
        query.lower()
    )

    if match:
        unit = match.group(1)
        numbers = {
            "i": 1, "1": 1,
            "ii": 2, "2": 2,
            "iii": 3, "3": 3,
            "iv": 4, "4": 4,
            "v": 5, "5": 5
        }
        number = numbers[unit]
        roman = ["", "I", "II", "III", "IV", "V"][number]

        stored = collection.get(
            include=["documents", "metadatas"]
        )

        pattern = re.compile(
            rf"\bUNIT\s*[-:]?\s*(?:{roman}|{number})\b",
            re.IGNORECASE
        )

        passages = []

        for doc, metadata in zip(
            stored["documents"], stored["metadatas"]
        ):
            if pattern.search(doc[:250]):
                passages.append({
                    "text": doc,
                    "page": metadata["page"],
                    "source": metadata.get("source", "Notes")
                })

        if passages:
            return passages[:top_k]

    # Semantic search for other questions.
    count = collection.count()
    if count == 0:
        return []

    embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=min(top_k, count)
    )

    passages = []

    for doc, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        passages.append({
            "text": doc,
            "page": metadata["page"],
            "source": metadata.get("source", "Notes")
        })

    return passages


if __name__ == "__main__":
    default_pdf = ROOT_DIR / "data" / "dpco notes.pdf"
    entered_path = input(
        f"Enter PDF path (press Enter for {default_pdf}): "
    ).strip().strip('"')

    pdf_path = Path(entered_path) if entered_path else default_pdf

    if not pdf_path.is_absolute():
        pdf_path = ROOT_DIR / pdf_path

    try:
        collection, model = build_index(pdf_path)

        while True:
            query = input(
                "\nAsk a question about your notes (or type 'exit'): "
            ).strip()

            if query.lower() == "exit":
                break

            if query:
                search_notes(query, collection, model)

    except Exception as error:
        print(f"\nERROR: {error}")