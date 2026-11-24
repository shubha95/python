import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
_DEFAULT_EMBEDDING_MODEL = {
    "openai": "text-embedding-3-small",
    "google": "models/gemini-embedding-001",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL.get(
    EMBEDDING_PROVIDER, ""
)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hr_policy")

PDF_PATH = os.getenv("PDF_PATH", "./data/hr_policy.pdf")


def get_chat_model():
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            max_retries=6,  # ride out temporary 503 "high demand" errors
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def get_embedding_model(for_query=False):
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    elif EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        task = "retrieval_query" if for_query else "retrieval_document"
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
            task_type=task,
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")