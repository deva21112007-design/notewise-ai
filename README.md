# 📚 NoteWise AI

**Your personal AI study companion — chat with your notes.**

NoteWise AI is a Retrieval-Augmented Generation (RAG) application that lets students ask questions about academic notes and receive conversational, AI-generated explanations.

It combines semantic search with a large language model to retrieve relevant passages from study materials and use them as context for answers.

## ✨ Features

- **PDF text extraction** using PyMuPDF.
- **Text chunking** to divide long documents into searchable sections.
- **Semantic search** using Sentence Transformers.
- **Vector storage and retrieval** with ChromaDB.
- **AI-generated answers** powered by the Gemini API.
- **Interactive chat interface** built with Streamlit.
- **Source references** showing the document pages returned by retrieval.
- **Cached resources** to avoid reloading the embedding model on every interaction.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| User interface | Streamlit |
| LLM | Google Gemini API |
| Embeddings | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| PDF processing | PyMuPDF |

## 🏗️ Architecture

```text
Academic PDF
     ↓
PDF Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer Embeddings
     ↓
ChromaDB Vector Database
     ↓
User Question
     ↓
Semantic Retrieval
     ↓
Retrieved Passages + Question
     ↓
Gemini API
     ↓
Conversational Answer + Page References
     ↓
Streamlit Chat Interface
```

## 📁 Project Structure

```text
notewise-ai/
├── app.py
├── src/
│   ├── document_processor.py
│   ├── chunker.py
│   └── vector_store.py
├── data/
│   └── [your academic PDF]
├── chroma_db/             # Generated locally
├── .streamlit/
│   └── secrets.toml       # Private; never commit
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd notewise-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Set the key temporarily in PowerShell:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Alternatively, create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Never commit your real API key to GitHub.

### 5. Add your academic PDF

Place a PDF you are authorized to use inside the `data/` folder.

The current application expects the local ChromaDB index to exist. Follow the project's indexing instructions to build the index before launching the app.

### 6. Build the vector index

```powershell
python src\vector_store.py
```

Use the default PDF path when prompted, or enter the path to your own PDF.

### 7. Launch NoteWise AI

```powershell
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔐 Security

- Never commit API keys or secrets.
- Keep `.venv/` and generated database files out of version control.
- Do not upload private, confidential, or copyrighted documents without permission.
- Review the Gemini API terms and data handling policies before using sensitive materials.

## ⚠️ Current Limitations

- Retrieval quality depends on the extracted PDF text and chunking strategy.
- Scanned PDFs may require OCR.
- Retrieved page references identify relevant search results; they do not independently verify every generated claim.
- The initial version is configured around a locally indexed document collection.
- Public deployment requires additional configuration for secrets, dependencies, and document indexing.

## 🗺️ Roadmap

- [x] PDF text extraction
- [x] Text chunking
- [x] Embedding generation
- [x] ChromaDB indexing and semantic retrieval
- [x] Gemini API integration
- [x] Streamlit chat interface
- [ ] Improve retrieval relevance and answer grounding
- [ ] Support uploading and indexing PDFs through the UI
- [ ] Improve source citations
- [ ] Add suggested prompts and follow-up questions
- [ ] Deploy on Streamlit Community Cloud

## 🎯 Project Goal

Build a practical AI-powered study assistant that helps students understand their own learning materials through natural-language conversations.

---

**NoteWise AI** — Learn smarter, one question at a time.
