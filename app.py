# app.py
import os
import streamlit as st
from rag_pipeline import load_llm, build_rag_chain, ask_question
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingest import load_document, chunk_documents, store_in_faiss

FAISS_DIR   = "faiss_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_PATH    = "data/hr_policy_document.pdf"

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered"
)

st.markdown("""
    <style>
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

st.title("📋 HR Policy Assistant")
st.markdown("Ask any question about your company's HR policies.")
st.divider()


@st.cache_resource
def load_pipeline():
    # Always build FAISS from PDF on startup
    with st.spinner("⚙️ Building knowledge base..."):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        pages  = load_document(PDF_PATH)
        chunks = chunk_documents(pages)

        # Build FAISS directly in memory
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

    with st.spinner("🤖 Connecting to Groq LLM..."):
        llm             = load_llm()
        rag_chain_tuple = build_rag_chain(vectorstore, llm)

    return rag_chain_tuple


rag_chain_tuple = load_pipeline()

# ── CHAT HISTORY ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your HR Policy Assistant. Ask me anything about leave, work hours, payroll, or company policies!",
            "sources": []
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                f'<div class="source-badge">📄 Source: Page(s) '
                f'{", ".join(msg["sources"])}</div>',
                unsafe_allow_html=True
            )

# ── CHAT INPUT ────────────────────────────────
if user_input := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "sources": []
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching HR policy..."):
            answer, pages = ask_question(rag_chain_tuple, user_input)
        st.markdown(answer)
        if pages:
            st.markdown(
                f'<div class="source-badge">📄 Source: Page(s) '
                f'{", ".join(pages)}</div>',
                unsafe_allow_html=True
            )

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
    - ⚡ FAISS (Vector DB)
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
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Chat cleared! Ask me anything about HR policies.",
                "sources": []
            }
        ]
        st.rerun()