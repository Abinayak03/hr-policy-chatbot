# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH      = "data/hr_policy_document.pdf"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL   = "all-MiniLM-L6-v2"
FAISS_DIR     = "faiss_db"


def load_document(pdf_path: str):
    print(f"\n📄 Loading PDF from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found at '{pdf_path}'.")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"✅ Loaded {len(pages)} page(s).")
    return pages


def chunk_documents(pages):
    print(f"\n✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


def load_embedding_model():
    print(f"\n🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"✅ Embedding model ready!")
    return embeddings


def store_in_faiss(chunks, embeddings):
    print(f"\n💾 Storing {len(chunks)} chunks in FAISS...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save to disk
    vectorstore.save_local(FAISS_DIR)
    print(f"✅ FAISS saved to '{FAISS_DIR}/'")
    return vectorstore


if __name__ == "__main__":
    pages      = load_document(PDF_PATH)
    chunks     = chunk_documents(pages)
    embeddings = load_embedding_model()
    store_in_faiss(chunks, embeddings)
    print("\n🎉 Ingestion complete!")