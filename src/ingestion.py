import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Configuration ---
DATA_PATH = r"D:\python-project\news-nexus\data\raw_pdfs"
DB_PATH = r"D:\python-project\news-nexus\data\chroma_db"


def ingest_documents():
    # 1. Load Documents
    print(f"Loading PDFs from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Initialize Embeddings (Ollama)
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Create Vector Store
    print("Initializing Vector Store (this may take a few minutes for large PDFs)...")

    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )

    # Batch size
    BATCH_SIZE = 100
    total_chunks = len(chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        print(
            f"   > Processing batch {i // BATCH_SIZE + 1} of "
            f"{(total_chunks - 1) // BATCH_SIZE + 1} ({len(batch)} chunks)..."
        )
        vector_db.add_documents(batch)

    print("✅ Vector Store created successfully.")
    return len(raw_documents), len(chunks)


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)

    if not os.listdir(DATA_PATH):
        print(f"No PDFs found in {DATA_PATH}. Please add files to enable RAG features.")
    else:
        ingest_documents()