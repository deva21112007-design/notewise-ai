import re


def _split_into_sentences(text):
    """Split text into sentences using punctuation boundaries."""
    # Split on . ! ? followed by whitespace or end of string
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(pages, chunk_size=200, overlap=40):
    """
    Split extracted PDF pages into overlapping, sentence-aware chunks.

    Unlike a raw word-window approach, this tries not to break chunks
    in the middle of a sentence. Each chunk targets ~chunk_size words
    but only ever splits at a sentence boundary.

    Args:
        pages:      List of dicts with keys 'page' and 'text'.
        chunk_size: Target maximum words per chunk.
        overlap:    Number of words to repeat between consecutive chunks
                    to preserve context across boundaries.

    Returns:
        List of dicts with keys 'chunk_id', 'page', and 'text'.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    chunk_id = 1

    for page in pages:
        sentences = _split_into_sentences(page["text"])

        current_words = []

        for sentence in sentences:
            sentence_words = sentence.split()

            # If adding this sentence would exceed the limit and we already
            # have content, flush the current chunk first.
            if current_words and (
                len(current_words) + len(sentence_words) > chunk_size
            ):
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": " ".join(current_words),
                })
                chunk_id += 1

                # Keep the last `overlap` words as context for the next chunk.
                current_words = current_words[-overlap:] if overlap else []

            current_words.extend(sentence_words)

            # Hard cap: if a single sentence is longer than chunk_size, force
            # a flush so we never produce unlimited-size chunks.
            while len(current_words) > chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": " ".join(current_words[:chunk_size]),
                })
                chunk_id += 1
                current_words = (
                    current_words[chunk_size - overlap:]
                    if overlap else current_words[chunk_size:]
                )

        # Flush any remaining words for this page.
        if current_words:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": " ".join(current_words),
            })
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    from document_processor import extract_text_from_pdf

    pdf_path = input("Enter your PDF path: ").strip().strip('"')

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"\nPages extracted: {len(pages)}")
    print(f"Chunks created:  {len(chunks)}")

    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk['chunk_id']} | Page {chunk['page']} ---")
        print(chunk["text"][:400])

    print("\nSUCCESS: Sentence-aware text chunking completed!")
