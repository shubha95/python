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

    chunks = load_and_split_pdf(PDF_PATH)
    print(f"\nTotal chunks to index: {len(chunks)}")

    print("\nLoading embedding model...")
    embeddings = get_embedding_model()
    print("Embedding model ready.")

    print(f"\nIndexing into ChromaDB at: {CHROMA_DB_PATH}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print(f"\nDone! {len(chunks)} chunks stored in ChromaDB.")
    print("=" * 50)


if __name__ == "__main__":
    ingest()