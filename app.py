"""
NoteWise AI — Main Streamlit Application
=========================================
A conversational RAG study assistant.

Improvements in this version
-----------------------------
- PDF upload directly from the sidebar (no manual indexing step needed).
- Streaming Gemini responses (token-by-token display).
- Relevance score shown on every source citation.
- Suggested starter questions on a fresh conversation.
- Polished custom CSS with dark-mode glassmorphism.
- Correct Gemini model name (gemini-2.0-flash).
- Conversation history capped at last 8 turns to keep prompts tight.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import chromadb
from google import genai
from sentence_transformers import SentenceTransformer

# Make src/ importable.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vector_store import (
    build_index,
    search_notes,
    DB_PATH,
    MODEL_NAME,
    COLLECTION_NAME,
)

# ── Constants ─────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
HISTORY_WINDOW = 8          # turns kept in the prompt context
TOP_K_PASSAGES = 6          # passages retrieved per question

SUGGESTED_QUESTIONS = [
    "Summarise the key topics covered in these notes.",
    "Explain the most important concept in simple terms.",
    "Give me 5 practice questions based on these notes.",
    "What are the main differences between the topics discussed?",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NoteWise AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Header ── */
    .notewise-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .notewise-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .notewise-header p {
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
        margin: 0;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
        margin-bottom: 0.75rem;
    }

    /* ── Source pills ── */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.78rem;
        color: #c4b5fd;
        margin: 3px 4px 3px 0;
        font-weight: 500;
    }
    .score-badge {
        background: rgba(52, 211, 153, 0.2);
        border-radius: 999px;
        padding: 1px 7px;
        font-size: 0.72rem;
        color: #6ee7b7;
    }

    /* ── Suggested question buttons ── */
    .stButton button {
        background: rgba(96, 165, 250, 0.1) !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px !important;
        color: #93c5fd !important;
        font-size: 0.85rem !important;
        width: 100%;
        text-align: left !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        background: rgba(96, 165, 250, 0.2) !important;
        border-color: rgba(96, 165, 250, 0.6) !important;
        transform: translateY(-1px);
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 16px !important;
        color: white !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        border: 1px dashed rgba(167, 139, 250, 0.4);
    }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return key


@st.cache_resource
def load_existing_index():
    """Load a pre-built ChromaDB collection and embedding model."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)
    return collection, model


def index_uploaded_pdf(uploaded_file):
    """Save upload to a temp file, run build_index, return (collection, model)."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", dir=ROOT_DIR / "data"
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    # Rename to the original filename for cleaner source citations.
    final_path = ROOT_DIR / "data" / uploaded_file.name
    tmp_path.rename(final_path)

    with st.spinner("⚙️ Indexing your PDF — this takes ~30 seconds..."):
        collection, model = build_index(final_path)

    # Clear the cached resource so the new collection is picked up.
    load_existing_index.clear()
    return collection, model


def build_prompt(question, passages, history):
    """Construct the full prompt sent to Gemini."""
    context = "\n\n".join(
        f"[Page {p['page']} | relevance {p.get('score', '—')}]\n{p['text']}"
        for p in passages
    )

    chat_history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-(HISTORY_WINDOW * 2):]
    )

    return f"""
You are NoteWise, a friendly and knowledgeable AI study tutor.

HOW TO ANSWER:
- Answer naturally, like a patient tutor having a conversation.
- Start with a direct answer, then explain with clear examples.
- Use headings, bullet points, and step-by-step working when helpful.
- Show clear working for numerical or logic problems.
- Adapt explanations to a college student's level.
- Use the retrieved passages as your primary evidence.
- NEVER claim the notes say something unless a passage supports it.
- If passages are insufficient, say so clearly, then offer a general explanation labelled as such.
- Do not invent page numbers or citations.
- Generate useful practice questions and answers when requested.
- Use conversation history to answer follow-ups.

CONVERSATION HISTORY:
{chat_history or "No previous conversation."}

RETRIEVED PASSAGES FROM THE STUDENT'S NOTES:
{context or "No relevant passages found — answer from general knowledge and label it clearly."}

STUDENT'S QUESTION:
{question}

Write a clear, friendly, well-structured answer.
""".strip()


def render_sources(sources):
    """Render source citations as styled HTML pills."""
    if not sources:
        return

    pills_html = ""
    for s in sources:
        score_pct = f"{s.get('score', 0) * 100:.0f}%" if s.get("score") else ""
        score_html = (
            f'<span class="score-badge">{score_pct}</span>' if score_pct else ""
        )
        pills_html += (
            f'<span class="source-pill">'
            f'📄 {s["source"]} · p.{s["page"]} {score_html}'
            f'</span>'
        )

    st.markdown(
        f'<div style="margin-top:0.5rem">{pills_html}</div>',
        unsafe_allow_html=True,
    )


# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ── API key check ─────────────────────────────────────────────────────────────
api_key = get_api_key()

if not api_key:
    st.warning(
        "🔑 **Gemini API key not found.** "
        "Set `GEMINI_API_KEY` in your terminal or in `.streamlit/secrets.toml`."
    )
    st.stop()

gemini = genai.Client(api_key=api_key)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 NoteWise AI")
    st.markdown("---")

    # ── PDF Upload ──
    st.markdown("### 📄 Upload Your Notes")
    uploaded = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        help="Your PDF is indexed locally — nothing is sent to external servers except the Gemini API.",
        label_visibility="collapsed",
    )

    if uploaded:
        if uploaded.name != st.session_state.doc_name:
            try:
                col, mdl = index_uploaded_pdf(uploaded)
                st.session_state.collection = col
                st.session_state.embed_model = mdl
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.success(f"✅ Indexed **{uploaded.name}**")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to index PDF: {e}")

    # ── Try loading pre-built index if no upload yet ──
    if st.session_state.collection is None:
        try:
            col, mdl = load_existing_index()
            st.session_state.collection = col
            st.session_state.embed_model = mdl
            st.session_state.doc_name = "DPCO Notes (pre-loaded)"
        except Exception:
            pass

    st.markdown("---")

    # ── Stats ──
    if st.session_state.collection:
        chunk_count = st.session_state.collection.count()
        col1, col2 = st.columns(2)
        col1.metric("📦 Chunks", chunk_count)
        col2.metric("💬 Messages", len(st.session_state.messages))

        st.markdown(
            f"**Active document:**  \n`{st.session_state.doc_name}`",
        )
    else:
        st.info("Upload a PDF to get started.")

    st.markdown("---")

    # ── Controls ──
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="notewise-header">
        <h1>📚 NoteWise AI</h1>
        <p>Your personal AI study companion — chat with your notes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Gate: need a collection to chat ──────────────────────────────────────────
if st.session_state.collection is None:
    st.info(
        "👈 **Upload a PDF** in the sidebar to begin chatting with your notes. "
        "Or wait a moment — the pre-loaded notes are being indexed."
    )
    st.stop()

# ── Suggested starter questions (shown only on empty conversation) ─────────────
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking…")
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.messages.append(
                {"role": "user", "content": suggestion}
            )
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about your notes…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""

        try:
            # 1. Retrieve relevant passages.
            passages = search_notes(
                question,
                st.session_state.collection,
                st.session_state.embed_model,
                top_k=TOP_K_PASSAGES,
            )

            history = st.session_state.messages[:-1]
            prompt = build_prompt(question, passages, history)

            # 2. Stream the Gemini response.
            stream = gemini.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            for chunk in stream:
                if chunk.text:
                    full_answer += chunk.text
                    answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            # 3. Build de-duplicated source list.
            sources = []
            seen = set()
            for p in passages:
                key = (p.get("source", "Notes"), p["page"])
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source": key[0],
                        "page": key[1],
                        "score": p.get("score"),
                    })

            render_sources(sources)

            # 4. Persist to session state.
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })

        except Exception as error:
            answer_placeholder.error(
                "Something went wrong while generating your answer."
            )
            st.caption(f"Technical details: {error}")
