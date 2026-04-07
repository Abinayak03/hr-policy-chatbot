# rag_pipeline.py (Updated - Modern LangChain LCEL approach)

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Load API key ──────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── CONFIG ────────────────────────────────────
CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K       = 3


# ── LOAD CHROMADB ─────────────────────────────
def load_vectorstore():
    print("\n📂 Loading ChromaDB from disk...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    vectorstore = Chroma(
        client=client,
        collection_name="hr_policy",
        embedding_function=embeddings,
    )

    print(f"✅ ChromaDB loaded successfully!")
    return vectorstore


# ── LOAD GROQ LLM ─────────────────────────────
def load_llm():
    print(f"\n🤖 Connecting to Groq LLM ({LLM_MODEL})...")

    if not GROQ_API_KEY:
        raise ValueError(
            "❌ GROQ_API_KEY not found!\n"
            "   Add it to your .env file:\n"
            "   GROQ_API_KEY=gsk_xxxxxxxxxxxx"
        )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
        max_tokens=1024
    )

    print(f"✅ Groq LLM connected!")
    return llm


# ── BUILD PROMPT ──────────────────────────────
def build_prompt():
    template = """
You are a helpful HR Policy Assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I don't have information about that in the HR policy document."

Context:
{context}

Question: {question}

Answer (be clear and concise):
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# ── BUILD RAG CHAIN ───────────────────────────
def build_rag_chain(vectorstore, llm):
    print(f"\n⛓️  Building RAG chain...")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = build_prompt()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"✅ RAG chain ready!")
    return rag_chain, retriever


# ── ASK A QUESTION ────────────────────────────
def ask_question(rag_chain_tuple, question: str):
    rag_chain, retriever = rag_chain_tuple

    print(f"\n💬 Question: {question}")
    print(f"   Thinking...")

    answer = rag_chain.invoke(question)

    # Get source pages
    docs  = retriever.invoke(question)
    pages = list(set([
        str(doc.metadata.get("page", 0) + 1) for doc in docs
    ]))

    return answer, pages


# ── MAIN TEST ─────────────────────────────────
if __name__ == "__main__":

    vectorstore     = load_vectorstore()
    llm             = load_llm()
    rag_chain_tuple = build_rag_chain(vectorstore, llm)

    test_questions = [
        "What is the leave policy?",
        "How many sick leaves do employees get?",
        "What are the work hours?",
        "What happens if someone is late?",
    ]

    print(f"\n{'='*55}")
    print("🧪 TESTING RAG PIPELINE")
    print(f"{'='*55}")

    for question in test_questions:
        answer, pages = ask_question(rag_chain_tuple, question)
        print(f"\n{'─'*55}")
        print(f"❓ Q: {question}")
        print(f"💡 A: {answer}")
        print(f"📄 Sources: Page(s) {', '.join(pages)}")

    print(f"\n🎉 RAG Pipeline working!")
    print(f"   ➡️  Ready for Step 7: Streamlit UI!")