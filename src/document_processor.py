"""
document_processor.py
=====================
Universal document text extractor for NoteWise AI.

Supported formats
-----------------
- PDF  — text-based (PyMuPDF fast path)
- PDF  — scanned / image-based (Gemini Vision OCR fallback)
- DOCX — Microsoft Word (python-docx)
- PPTX — PowerPoint slides (python-pptx)
- TXT  — plain text
- Images — PNG, JPG, JPEG, WEBP (Gemini Vision OCR)

Each extractor returns a list of page dicts:
    [{"page": int, "text": str}, ...]

The master dispatcher `extract_text(file_path, api_key)` chooses the
right extractor automatically based on file extension.
"""

import os
from pathlib import Path

import pymupdf  # PyMuPDF


# ── Threshold: pages with fewer chars than this are treated as image-only ──
_MIN_TEXT_CHARS = 60
# ── Fraction of pages that must be text-based to skip OCR ──
_TEXT_PAGE_RATIO = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# PDF — text extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract text from a native (non-scanned) PDF."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_num, "text": text})
    return pages


def _render_page_as_png(doc, page_index: int, zoom: float = 2.0) -> bytes:
    """Render a single PDF page to PNG bytes at the given zoom level."""
    page = doc[page_index]
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


VISION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]


def _ocr_image_bytes(image_bytes: bytes, api_key: str, page_label: str = "page") -> str:
    """Send raw PNG/JPEG bytes to Gemini Vision and return extracted text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last_err = None
    for model_name in VISION_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    types.Part.from_text(
                        "Extract ALL visible text from this image exactly as written. "
                        "Preserve headings, bullet points, tables, equations, and captions. "
                        "Do not add any commentary or explanation — only the text content."
                    ),
                ],
            )
            return (response.text or "").strip()
        except Exception as err:
            last_err = err
            continue
    if last_err:
        raise last_err
    return ""


def extract_text_from_pdf(pdf_path, api_key: str | None = None) -> list[dict]:
    """
    Extract text from a PDF.

    If the PDF is text-based → fast PyMuPDF extraction.
    If the PDF is scanned / image-only → Gemini Vision OCR (requires api_key).
    Mixed PDFs are handled page-by-page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as doc:
        total = len(doc)

        # First pass: try plain text extraction on every page.
        raw_pages = []
        for i in range(total):
            text = doc[i].get_text("text").strip()
            raw_pages.append(text)

        text_count = sum(1 for t in raw_pages if len(t) >= _MIN_TEXT_CHARS)
        needs_ocr = (text_count / max(total, 1)) < _TEXT_PAGE_RATIO

        if needs_ocr and not api_key:
            raise ValueError(
                "This PDF appears to be scanned (image-based). "
                "OCR requires a Gemini API key — please set GEMINI_API_KEY."
            )

        for i, text in enumerate(raw_pages, start=1):
            if len(text) >= _MIN_TEXT_CHARS:
                # Good text page — use it directly.
                pages.append({"page": i, "text": text})
            elif api_key:
                # Image page — OCR with Gemini Vision.
                img_bytes = _render_page_as_png(doc, i - 1)
                ocr_text = _ocr_image_bytes(img_bytes, api_key, page_label=f"p{i}")
                if ocr_text:
                    pages.append({"page": i, "text": f"[OCR] {ocr_text}"})

    if not pages:
        raise ValueError(
            "No readable text found in this PDF. "
            "If it is a scanned document, ensure your Gemini API key is set."
        )

    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# DOCX — Microsoft Word
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_docx(docx_path) -> list[dict]:
    """Extract text from a .docx file. Each ~50 paragraphs = 1 'page'."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")

    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise ValueError("No readable text found in this DOCX file.")

    # Group every 50 paragraphs into a logical "page".
    chunk_size = 50
    pages = []
    for i in range(0, len(paragraphs), chunk_size):
        page_num = (i // chunk_size) + 1
        text = "\n".join(paragraphs[i: i + chunk_size])
        pages.append({"page": page_num, "text": text})

    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# PPTX — PowerPoint
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pptx(pptx_path) -> list[dict]:
    """Extract text from a .pptx file. Each slide = 1 page."""
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required: pip install python-pptx")

    pptx_path = Path(pptx_path)
    if not pptx_path.exists():
        raise FileNotFoundError(f"PPTX not found: {pptx_path}")

    prs = Presentation(pptx_path)
    pages = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        if texts:
            pages.append({"page": slide_num, "text": "\n".join(texts)})

    if not pages:
        raise ValueError("No readable text found in this PPTX file.")

    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# TXT — Plain text
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_txt(txt_path) -> list[dict]:
    """Extract text from a plain .txt file. Every ~100 lines = 1 page."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT not found: {txt_path}")

    content = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        raise ValueError("The text file is empty.")

    lines = content.splitlines()
    chunk_size = 100
    pages = []

    for i in range(0, len(lines), chunk_size):
        page_num = (i // chunk_size) + 1
        text = "\n".join(lines[i: i + chunk_size]).strip()
        if text:
            pages.append({"page": page_num, "text": text})

    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# Images — PNG, JPG, JPEG, WEBP via Gemini Vision OCR
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_image(image_path, api_key: str) -> list[dict]:
    """OCR a single image file using Gemini Vision."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/png")

    img_bytes = image_path.read_bytes()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last_err = None
    response = None
    for model_name in VISION_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    types.Part.from_text(
                        "Extract ALL visible text from this image exactly as written. "
                        "Preserve headings, bullet points, tables, equations. "
                        "Return only the extracted text."
                    ),
                ],
            )
            break
        except Exception as err:
            last_err = err
            continue

    if response is None and last_err:
        raise last_err

    text = (response.text or "").strip()
    if not text:
        raise ValueError("No text could be extracted from this image.")

    return [{"page": 1, "text": text}]


# ═══════════════════════════════════════════════════════════════════════════════
# Master dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".png", ".jpg", ".jpeg", ".webp"}


def extract_text(file_path, api_key: str | None = None) -> list[dict]:
    """
    Auto-detect the file type and extract text pages.

    Args:
        file_path:  Path to the document.
        api_key:    Gemini API key — required for scanned PDFs and images.

    Returns:
        List of {"page": int, "text": str} dicts.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path, api_key=api_key)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".pptx":
        return extract_text_from_pptx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
        if not api_key:
            raise ValueError("Image OCR requires a Gemini API key.")
        return extract_text_from_image(path, api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    file_path = input("Enter file path: ").strip().strip('"')
    api_key = os.getenv("GEMINI_API_KEY", "")

    try:
        pages = extract_text(file_path, api_key=api_key or None)
        print(f"\nExtracted {len(pages)} page(s).")
        for p in pages[:2]:
            print(f"\n--- Page {p['page']} ---")
            print(p["text"][:500])
        print("\nSUCCESS!")
    except Exception as e:
        print(f"Error: {e}")
