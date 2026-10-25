
import os
import sys
from pathlib import Path

import streamlit as st
import chromadb
from google import genai
from sentence_transformers import SentenceTransformer

# Make the src folder importable.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vector_store import (
    search_notes,
    DB_PATH,
    MODEL_NAME,
    COLLECTION_NAME,
)

MODEL_NAME_GEMINI = "gemini-3.6-flash"

st.set_page_config(
    page_title="NoteWise AI",
    page_icon="📚",
    layout="centered",
)

st.title("📚 NoteWise AI")
st.caption("Your personal AI study companion")

st.markdown(
    "Ask questions about your notes. Get clear explanations, "
    "examples, and source page references."
)


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None

    return key


@st.cache_resource
def load_resources():
    # Reuse the existing index. Do not rebuild it per question.
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Cache the embedding model in memory.
    model = SentenceTransformer(MODEL_NAME)

    return collection, model


def make_prompt(question, passages, history):
    context = "\n\n".join(
        f"[Page {p['page']}]\n{p['text']}"
        for p in passages
    )

    previous_chat = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-6:]
    )

    return f"""
You are NoteWise, a friendly and helpful AI study tutor.

HOW TO ANSWER:
- Answer naturally, like a patient tutor having a conversation.
- Start with a direct answer, then explain in simple language.
- Use headings, bullet points, and step-by-step examples when useful.
- For numerical or logic problems, show the working clearly.
- Adapt the explanation to a college student.
- Use the supplied notes as the primary source.
- Never claim that the notes say something unless the supplied
  passages support it.
- If the passages do not contain enough information, say so.
  You may then offer a clearly labelled general explanation.
- Do not invent page numbers or citations.
- If the user asks for practice questions, create useful questions
  on the requested topic and provide answers when requested.
- Answer follow-up questions using the conversation history.

PREVIOUS CONVERSATION:
{previous_chat or "No previous conversation."}

RETRIEVED PASSAGES FROM THE STUDENT'S NOTES:
{context or "No relevant passages were retrieved."}

STUDENT'S QUESTION:
{question}

Write the answer in a clear, friendly, helpful style.
""".strip()


api_key = get_api_key()

if not api_key:
    st.warning(
        "Gemini API key not found. Set GEMINI_API_KEY in your "
        "terminal or configure it in .streamlit/secrets.toml."
    )
    st.stop()

try:
    collection, embedding_model = load_resources()
except Exception as error:
    st.error(
        "Could not load the notes index. Check that chroma_db "
        "exists and contains the indexed collection."
    )
    st.caption(f"Technical details: {error}")
    st.stop()

gemini = genai.Client(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Your study space")
    st.write("📄 Indexed document: DPCO notes")
    st.write(f"🔎 Searchable chunks: {collection.count()}")
    st.caption("More document-upload options can be added next.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Display the existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources from your notes"):
                for source in message["sources"]:
                    st.write(
                        f"📄 {source['source']} — Page {source['page']}"
                    )

question = st.chat_input("Ask anything about your notes...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading your notes and preparing an answer..."):
                passages = search_notes(
                    question,
                    collection,
                    embedding_model,
                    top_k=5,
                )

                history = st.session_state.messages[:-1]

                prompt = make_prompt(
                    question,
                    passages,
                    history,
                )

                response = gemini.models.generate_content(
                    model=MODEL_NAME_GEMINI,
                    contents=prompt,
                )

                answer = response.text or (
                    "I couldn't generate an answer. Please try again."
                )

            st.markdown(answer)

            # Remove duplicate source/page pairs.
            sources = []
            seen = set()

            for passage in passages:
                item = (
                    passage.get("source", "Notes"),
                    passage["page"],
                )

                if item not in seen:
                    seen.add(item)
                    sources.append({
                        "source": item[0],
                        "page": item[1],
                    })

            if sources:
                with st.expander("Sources from your notes"):
                    for source in sources:
                        st.write(
                            f"📄 {source['source']} — "
                            f"Page {source['page']}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(
                "Something went wrong while generating your answer."
            )
            st.caption(f"Technical details: {error}")
