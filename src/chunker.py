
def chunk_pages(pages, chunk_size=150, overlap=30):
    """Split extracted PDF pages into overlapping word chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    chunk_id = 1
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": " ".join(chunk_words)
            })

            chunk_id += 1

            if start + chunk_size >= len(words):
                break

    return chunks


if __name__ == "__main__":
    from document_processor import extract_text_from_pdf

    pdf_path = input("Enter your PDF path: ").strip().strip('"')

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"\nPages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk['chunk_id']} | Page {chunk['page']} ---")
        print(chunk["text"][:400])

    print("\nSUCCESS: Text chunking completed!")
