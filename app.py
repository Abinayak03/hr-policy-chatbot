# app.py
# ─────────────────────────────────────────────
# STEP 7: Streamlit Chat UI
# ─────────────────────────────────────────────

# app.py (Final version for deployment)

import os
import streamlit as st
from rag_pipeline import (
    load_vectorstore,
    load_llm,
    build_rag_chain,
    ask_question
)
from ingest import (
    load_document,
    chunk_documents,
    load_embedding_model,
    store_in_chromadb
)

# ── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered"
)

# ── AUTO BUILD CHROMADB IF MISSING ────────────
if not os.path.exists("chroma_db"):
    with st.spinner("⚙️ Building knowledge base (first time setup)..."):
        pages      = load_document("data/hr_policy.pdf")
        chunks     = chunk_documents(pages)
        embeddings = load_embedding_model()
        store_in_chromadb(chunks, embeddings)

# ── rest of your app.py stays exactly the same ──

# ── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered"
)

# ── CUSTOM CSS ────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .source-badge {
        background-color: #e8f4fd;
        border-left: 3px solid #2196F3;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.8em;
        color: #1565C0;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────
st.title("📋 HR Policy Assistant")
st.markdown("Ask any question about your company's HR policies.")
st.divider()


# ── LOAD RAG PIPELINE (cached) ────────────────
# @st.cache_resource means it loads ONCE and reuses
# across all users — no reloading on every message!
@st.cache_resource
def load_pipeline():
    with st.spinner("🔄 Loading HR Policy database..."):
        vectorstore     = load_vectorstore()
        llm             = load_llm()
        rag_chain_tuple = build_rag_chain(vectorstore, llm)
    return rag_chain_tuple


rag_chain_tuple = load_pipeline()


# ── CHAT HISTORY ──────────────────────────────
# st.session_state persists data across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your HR Policy Assistant. Ask me anything about leave, work hours, payroll, or company policies!",
            "sources": []
        }
    ]

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source pages if available
        if msg.get("sources"):
            st.markdown(
                f'<div class="source-badge">📄 Source: Page(s) '
                f'{", ".join(msg["sources"])}</div>',
                unsafe_allow_html=True
            )


# ── CHAT INPUT ────────────────────────────────
if user_input := st.chat_input("Ask about HR policies..."):

    # 1. Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "sources": []
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Get answer from RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching HR policy..."):
            answer, pages = ask_question(rag_chain_tuple, user_input)

        st.markdown(answer)

        # Show source pages
        if pages:
            st.markdown(
                f'<div class="source-badge">📄 Source: Page(s) '
                f'{", ".join(pages)}</div>',
                unsafe_allow_html=True
            )

    # 3. Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": pages
    })


# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **HR Policy Assistant** uses:
    - 🦙 Groq LLaMA 3.3 (LLM)
    - 🗄️ ChromaDB (Vector DB)
    - 🔗 LangChain (RAG Pipeline)
    - 🤗 HuggingFace Embeddings
    """)

    st.divider()

    st.header("💡 Sample Questions")
    sample_questions = [
        "What is the leave policy?",
        "How many sick leaves per year?",
        "What are the work hours?",
        "What is the remote work policy?",
        "How is payroll processed?",
    ]

    # Clicking a sample question sends it automatically
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": q,
                "sources": []
            })
            with st.spinner("🔍 Searching..."):
                answer, pages = ask_question(rag_chain_tuple, q)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": pages
            })
            st.rerun()

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Chat cleared! Ask me anything about HR policies.",
                "sources": []
            }
        ]
        st.rerun()