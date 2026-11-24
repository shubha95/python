from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    get_chat_model,
    get_embedding_model,
)

# --- Prompt template ---
HR_PROMPT = PromptTemplate.from_template("""
You are a strict HR Policy Assistant. You can ONLY answer questions using the HR policy context provided below.

STRICT RULES:
1. If the question is about HR policy and the answer is in the context → answer it.
2. If the question is about HR policy but the answer is NOT in the context → say exactly: "I'm sorry, the provided HR policy does not contain information about this. Please contact your HR department directly."
3. If the question is NOT related to HR policy at all (e.g. cricket, weather, coding, general knowledge) → say exactly: "I can only answer questions related to the company HR policy. Please ask an HR-related question."
4. NEVER use your own knowledge. ONLY use the context below.
5. NEVER make up or guess any policy details.

HR Policy Context:
{context}

Question: {question}

Answer:
""")


def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """Build and return the RAG chain."""

    # Load embedding model for querying
    embeddings = get_embedding_model(for_query=True)

    # Connect to existing ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # Retriever — fetch top 4 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Load chat model
    llm = get_chat_model()

    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | HR_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain