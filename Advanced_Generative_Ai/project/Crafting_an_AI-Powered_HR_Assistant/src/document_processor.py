import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str) -> list:
    """
    Load a PDF file and split it into chunks for indexing.

    Args:
        pdf_path: Path to the HR policy PDF file.

    Returns:
        A list of LangChain Document objects (chunks).
    """
    # Check the file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}\n"
            "Please place your HR policy PDF at data/hr_policy.pdf"
        )

    print(f"Loading PDF: {pdf_path}")

    # Load all pages from the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"Loaded {len(pages)} pages from PDF.")

    # Split pages into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk ~ 500 characters
        chunk_overlap=50,     # 50-char overlap so context isn't cut off
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    print(f"Split into {len(chunks)} chunks.")

    return chunks