
from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF and preserve its page numbers."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No readable text found. "
            "This PDF may contain scanned images."
        )

    return pages


if __name__ == "__main__":
    pdf_path = input("Enter your PDF path: ").strip().strip('"')

    try:
        pages = extract_text_from_pdf(pdf_path)

        print(f"\nSuccessfully extracted {len(pages)} pages.")

        for page in pages[:2]:
            print(f"\n--- Page {page['page']} ---")
            print(page["text"][:1000])

        print("\nPDF processing completed!")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
