import os
import sys
from src.config import (
    PDF_PATH,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    get_embedding_model,
)
from src.document_processor import load_and_split_pdf
from langchain_chroma import Chroma


def ingest():
    print("=" * 50)
    print("HR Policy RAG — Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load and chunk the PDF
    chunks = load_and_split_pdf(PDF_PATH)
    print(f"\nTotal chunks to index: {len(chunks)}")

    # Step 2: Get embedding model from config
    print("\nLoading embedding model...")
    embeddings = get_embedding_model()
    print("Embedding model ready.")

    # Step 3: Store into ChromaDB
    print(f"\nIndexing into ChromaDB at: {CHROMA_DB_PATH}")
    print(f"Collection name: {CHROMA_COLLECTION_NAME}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print(f"\nDone! {len(chunks)} chunks stored in ChromaDB.")
    print("You can now run the app: streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    ingest()