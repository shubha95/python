# Crafting an AI-Powered HR Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers employee questions from an HR policy PDF. It splits the PDF into chunks, stores their embeddings in ChromaDB, and answers questions with an LLM that is told to use **only** the retrieved policy text.

If the answer isn't in the policy, the assistant says so and refers the user to HR instead of guessing.

## How it works

```
PDF  ->  split into chunks  ->  embed  ->  ChromaDB          (ingest.py, run once)

question -> embed -> retrieve top 4 chunks -> prompt + LLM -> answer   (src/rag.py)
```

## Project structure

```
.
├── app.py                  # Streamlit chat UI
├── ingest.py               # Builds the vector database from the PDF
├── requirements.txt
├── .env                    # Your settings and API keys (not committed)
├── data/
│   └── hr_policy.pdf       # The policy document to index (not committed)
├── chroma_db/              # Generated vector store (not committed)
└── src/
    ├── config.py           # Env settings, chat model and embedding model factories
    ├── document_processor.py  # PDF loading and chunking
    ├── ingest.py           # Ingestion pipeline
    └── rag.py              # Retriever, prompt and RAG chain
```

## Setup

1. **Create and activate a virtual environment** (Python 3.10+)

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt streamlit
   ```

3. **Add the policy PDF** at `data/hr_policy.pdf` (or set `PDF_PATH`).

4. **Create a `.env` file** in the project root:

   ```env
   # Chat model: "google" or "openai"
   LLM_PROVIDER=google
   GOOGLE_API_KEY=your-google-api-key
   GEMINI_MODEL=gemini-3.7-flash

   # Embeddings: "google" or "openai"
   EMBEDDING_PROVIDER=google
   EMBEDDING_MODEL=models/gemini-embedding-001

   # For OpenAI instead
   # OPENAI_API_KEY=your-openai-api-key
   # OPENAI_MODEL=gpt-4o-mini

   CHROMA_DB_PATH=./chroma_db
   CHROMA_COLLECTION_NAME=hr_policy
   ```

   Never commit `.env`; it is already in `.gitignore`.

## Usage

**1. Build the vector database** (once, and again whenever the PDF or embedding model changes):

```powershell
python ingest.py
```

If you change the embedding model, delete `chroma_db/` first. Embeddings from different models can't share a collection.

**2. Ask a question from the command line:**

```powershell
python -c "from src.rag import get_rag_chain; print(get_rag_chain().invoke('What is the leave policy?'))"
```

**3. Run the Streamlit app:**

```powershell
streamlit run app.py
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `google` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini chat model |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `google` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` (OpenAI) / `models/gemini-embedding-001` (Google) | Embedding model |
| `CHROMA_DB_PATH` | `./chroma_db` | Where the vector store is saved |
| `CHROMA_COLLECTION_NAME` | `hr_policy` | ChromaDB collection name |
| `PDF_PATH` | `./data/hr_policy.pdf` | PDF to index |

## Status

- Ingestion and the RAG chain work from the command line.
- `app.py` is a chat UI connected to the RAG chain. It streams answers, keeps the conversation on screen and has a "Clear conversation" button in the sidebar.
- The sample PDF is the Nestlé Human Resources Policy, which covers high-level principles. It has no detailed leave entitlements, so questions about specific benefits correctly return the "not found" message. Use a more detailed policy for richer answers.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `404 NOT_FOUND` for a model | Google retires model names. List the models your key can use and update `GEMINI_MODEL` / `EMBEDDING_MODEL`: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; from google import genai; [print(m.name) for m in genai.Client(api_key=os.environ['GOOGLE_API_KEY']).models.list()]"` |
| `429 RESOURCE_EXHAUSTED` | Free-tier daily quota is used up. Wait for the reset, switch to another model, or enable billing. |
| `503 UNAVAILABLE` | The model is overloaded. Retry later or switch models. |
| `No module named ...` | The virtual environment is not active or dependencies are not installed. |
| `UnicodeEncodeError` in the terminal | Run `$env:PYTHONIOENCODING="utf-8"` first. |

## Tech stack

Python, LangChain, ChromaDB, Google Gemini or OpenAI, pypdf, Streamlit.
