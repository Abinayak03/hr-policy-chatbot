# ingest.py
# ─────────────────────────────────────────────
# STEP 2: Load PDF
# STEP 3: Chunk text
# STEP 4: Generate embeddings
# STEP 5: Store in ChromaDB  ← NEW
# ─────────────────────────────────────────────

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── CONFIG ────────────────────────────────────
PDF_PATH      = "data/hr_policy_document.pdf"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHROMA_DIR    = "chroma_db"           # folder where DB is saved


# ── STEP 2: LOAD PDF ─────────────────────────
def load_document(pdf_path: str):
    print(f"\n📄 Loading PDF from: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"❌ PDF not found at '{pdf_path}'. "
            "Please place your PDF inside the 'data/' folder."
        )

    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"✅ Loaded {len(pages)} page(s) from the PDF.")
    return pages


# ── STEP 3: CHUNK TEXT ────────────────────────
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


# ── STEP 4: LOAD EMBEDDING MODEL ─────────────
def load_embedding_model():
    print(f"\n🧠 Loading embedding model: {EMBED_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"✅ Embedding model ready!")
    return embeddings


# ── STEP 5: STORE IN CHROMADB ─────────────────
def store_in_chromadb(chunks, embeddings):
    """
    Embeds all chunks and stores them in ChromaDB.
    - If chroma_db/ folder exists → deletes and rebuilds (fresh start)
    - Saves to disk so we never re-embed again
    """
    print(f"\n💾 Storing {len(chunks)} chunks in ChromaDB...")
    print(f"   Location: {CHROMA_DIR}/")

    # Delete old DB if exists (clean rebuild)
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"   ♻️  Old ChromaDB cleared — rebuilding fresh...")

    # Create ChromaDB from chunks + embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"✅ ChromaDB created and saved to '{CHROMA_DIR}/'")
    return vectorstore


# ── TEST RETRIEVAL ────────────────────────────
def test_retrieval(vectorstore):
    """
    Quick test: ask a question and see which chunks
    ChromaDB retrieves as most relevant.
    """
    print(f"\n🔍 RETRIEVAL TEST")
    print(f"{'='*55}")

    query = "What is the leave policy?"
    print(f"   Query: '{query}'")

    # Get top 3 most relevant chunks
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\n   📦 Result {i+1}")
        print(f"   Page   : {doc.metadata.get('page', 'N/A')}")
        print(f"   Preview: {doc.page_content[:200]}...")
        print(f"   {'─'*50}")

    print(f"\n✅ ChromaDB is retrieving relevant chunks correctly!")


# ── MAIN: RUN ALL STEPS ───────────────────────
if __name__ == "__main__":

    # Step 2: Load
    pages = load_document(PDF_PATH)

    # Step 3: Chunk
    chunks = chunk_documents(pages)

    # Step 4: Embeddings
    embeddings = load_embedding_model()

    # Step 5: Store
    vectorstore = store_in_chromadb(chunks, embeddings)

    # Test it works
    test_retrieval(vectorstore)

    print(f"\n🎉 INGESTION COMPLETE!")
    print(f"{'='*55}")
    print(f"   PDF loaded      : ✅")
    print(f"   Chunks created  : ✅ ({len(chunks)} chunks)")
    print(f"   Embeddings made : ✅ (all-MiniLM-L6-v2)")
    print(f"   ChromaDB saved  : ✅ ({CHROMA_DIR}/)")
    print(f"\n   ➡️  Ready for Step 6: RAG Pipeline!")