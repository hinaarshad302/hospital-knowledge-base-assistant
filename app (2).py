"""
Hospital Knowledge Base Assistant
----------------------------------
A beginner-friendly RAG (Retrieval-Augmented Generation) application that
answers questions about a sample hospital knowledge base using:

    FAISS (vector search) + Sentence Transformers (embeddings)
    + Google Gemini (answer generation) + Streamlit (UI)

Author: Hina Arshad | AI/ML Project
"""

import os
import pickle

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Hospital Knowledge Base Assistant",
    page_icon="🏥",
    layout="wide",
)

FAISS_INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/metadata.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.6-flash"

# Retrieval settings
TOP_K = 5
MIN_SIMILARITY = 0.30
SIMILARITY_GAP = 0.08

SUGGESTED_QUESTIONS = [
    "What documents are required for admission?",
    "What should I do in a medical emergency?",
    "What departments are available?",
    "What are the patient safety guidelines?",
]

DEPARTMENTS = [
    "General Administration",
    "Admissions",
    "Emergency",
    "Patient Safety",
    "Outpatient Services",
]

DISCLAIMER_TEXT = (
    "This assistant answers using a **sample hospital knowledge base** for "
    "demonstration purposes only. It does **not** provide medical advice, "
    "diagnosis, or treatment. In a real emergency, always contact your local "
    "emergency services or visit the nearest hospital immediately."
)

NOT_FOUND_MESSAGE = "I don't have that information in the current knowledge base."


# =========================================================
# RESOURCE LOADING (cached)
# =========================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


def resources_available():
    """Check that both FAISS index and metadata files exist."""
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)


# =========================================================
# GEMINI SETUP
# =========================================================

def configure_gemini():
    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return False

    genai.configure(api_key=api_key)
    return True


# =========================================================
# RETRIEVAL
# =========================================================

def get_relevant_chunks(question, index, metadata, embed_model):
    """Embed the question and retrieve relevant chunks from FAISS."""
    query_embedding = embed_model.encode(
        [question],
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, TOP_K)
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        if score < MIN_SIMILARITY:
            continue
        chunk = metadata[idx]
        results.append({"score": float(score), "chunk": chunk})

    if not results:
        return []

    # Filter out chunks that fall too far behind the best match
    top_score = results[0]["score"]
    filtered = [r for r in results if (top_score - r["score"]) <= SIMILARITY_GAP]

    return filtered


def format_context(chunks):
    context_parts = []
    for r in chunks:
        text = r["chunk"].get("text", "")
        context_parts.append(text)
    return "\n\n---\n\n".join(context_parts)


def get_sources(chunks):
    sources = []
    seen = set()
    for r in chunks:
        source = r["chunk"].get("source", "Unknown source")
        if source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def get_departments_mentioned(chunks):
    departments = []
    seen = set()
    for r in chunks:
        dept = r["chunk"].get("department")
        if dept and dept not in seen:
            departments.append(dept)
            seen.add(dept)
    return departments


# =========================================================
# GEMINI ANSWER GENERATION
# =========================================================

def build_prompt(question, context):
    return f"""You are a helpful hospital knowledge base assistant.

Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, respond exactly with:
"{NOT_FOUND_MESSAGE}"

Do not make up information. Do not answer from general knowledge.
Keep the answer clear, concise, and professional.

Context:
{context}

Question:
{question}

Answer:"""


def generate_answer(question, context):
    prompt = build_prompt(question, context)

    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = gemini_model.generate_content(prompt)
        return response.text.strip(), None

    except Exception as e:
        error_text = str(e).lower()

        if "429" in error_text or "resourceexhausted" in error_text or "quota" in error_text:
            return None, (
                "⚠️ AI service quota is temporarily unavailable.\n\n"
                "The Gemini API request limit has been reached. Please try again later."
            )

        return None, (
            "⚠️ The AI service is temporarily unavailable.\n\n"
            "Please try again later."
        )


def get_related_questions(current_question):
    return [q for q in SUGGESTED_QUESTIONS if q.lower() != current_question.lower()][:3]


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat():
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🏥 Shifa Hospital")
    st.markdown("### Knowledge Base Assistant")

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This assistant answers questions using a sample hospital knowledge "
        "base covering policies, admissions, departments, emergency "
        "procedures, and patient safety guidelines."
    )

    st.markdown("---")
    st.markdown("### Available Departments")
    for dept in DEPARTMENTS:
        st.markdown(f"- {dept}")

    st.markdown("---")
    st.button("🗑️ Clear Chat", on_click=clear_chat, use_container_width=True)

    st.markdown("---")
    st.warning(DISCLAIMER_TEXT)

    st.markdown("---")
    st.caption("Built by Hina Arshad | AI/ML Project")


# =========================================================
# MAIN AREA
# =========================================================

st.markdown("# 🏥 Hospital Knowledge Base Assistant")
st.markdown(
    "Ask questions about hospital policies, departments, "
    "admission procedures and patient safety."
)

# --- Resource / config checks ---
if not resources_available():
    st.error(
        "⚠️ Knowledge base files not found.\n\n"
        "Please make sure `faiss_index/index.faiss` and "
        "`faiss_index/metadata.pkl` exist in the project directory. "
        "These files are generated by the project's indexing pipeline."
    )
    st.stop()

gemini_ready = configure_gemini()
if not gemini_ready:
    st.error(
        "⚠️ Gemini API key not found.\n\n"
        "Please add `GOOGLE_API_KEY` to `.streamlit/secrets.toml` "
        "(see `.streamlit/secrets.toml.example`)."
    )
    st.stop()

embed_model = load_embedding_model()
faiss_index = load_faiss_index()
metadata = load_metadata()

# --- Suggested questions ---
st.markdown("#### 💬 Suggested Questions")
cols = st.columns(2)
suggested_click = None
for i, q in enumerate(SUGGESTED_QUESTIONS):
    if cols[i % 2].button(q, key=f"suggested_{i}", use_container_width=True):
        suggested_click = q

st.markdown("---")

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")
            if msg.get("related_questions"):
                with st.expander("💡 Related Questions"):
                    for rq in msg["related_questions"]:
                        st.markdown(f"- {rq}")

# --- Chat input ---
user_question = st.chat_input("Ask a question about the hospital...")

question_to_process = suggested_click or user_question

if question_to_process:
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    with st.chat_message("user"):
        st.markdown(question_to_process)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            relevant_chunks = get_relevant_chunks(
                question_to_process, faiss_index, metadata, embed_model
            )

        if not relevant_chunks:
            answer = NOT_FOUND_MESSAGE
            sources = []
            departments = []
        else:
            context = format_context(relevant_chunks)
            with st.spinner("Generating answer..."):
                answer, error = generate_answer(question_to_process, context)

            if error:
                answer = error
                sources = []
                departments = []
            else:
                sources = get_sources(relevant_chunks)
                departments = get_departments_mentioned(relevant_chunks)

        st.markdown(answer)

        related_questions = get_related_questions(question_to_process)

        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

        if departments:
            with st.expander("🏥 Related Departments"):
                for d in departments:
                    st.markdown(f"- {d}")

        if related_questions:
            with st.expander("💡 Related Questions"):
                for rq in related_questions:
                    st.markdown(f"- {rq}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "related_questions": related_questions,
        }
    )
