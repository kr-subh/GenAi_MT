"""
ingest.py — Reads the HR Manual PDF, splits it into chunks,
computes embeddings using Ollama, and stores them in ChromaDB.
"""

import os
import sys
import shutil
import urllib.request
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Load environment configurations
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "data/hr_manual.pdf")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def check_ollama_connection(base_url: str) -> bool:
    """Verify that Ollama server is running and accessible."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    print("=" * 60)
    print("HR Policy Assistant — Document Ingestion Pipeline")
    print("=" * 60)

    # 1. Check if the PDF file exists
    if not os.path.exists(PDF_PATH):
        print(f"\n[ERROR] Missing PDF file: '{PDF_PATH}' was not found.")
        print("Please ensure the HR manual is placed in the 'data/' directory.")
        sys.exit(1)

    print(f"\n[1/4] Checking Ollama server at {OLLAMA_BASE_URL}...")
    if not check_ollama_connection(OLLAMA_BASE_URL):
        print(f"[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}.")
        print("Please start Ollama in your terminal using:")
        print("    brew services start ollama   (or: ollama serve)")
        sys.exit(1)
    print("      Ollama connection verified.")

    # 2. Load the PDF document
    print(f"\n[2/4] Loading PDF document from '{PDF_PATH}'...")
    loader = PyPDFLoader(PDF_PATH)
    raw_docs = loader.load()
    if not raw_docs:
        print("[ERROR] The PDF file could not be read or contains no extractable text.")
        sys.exit(1)

    total_pages = len(raw_docs)
    print(f"      Successfully extracted {total_pages} page(s).")

    # Add 1-based page numbers to document metadata
    for doc in raw_docs:
        page_num = doc.metadata.get("page", 0) + 1
        doc.metadata["page_number"] = page_num
        doc.metadata["source_file"] = os.path.basename(PDF_PATH)

    # 3. Split the text into manageable chunks
    print("\n[3/4] Chunking document text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"      Created {len(chunks)} text chunks from {total_pages} pages.")

    # 4. Generate embeddings and populate ChromaDB
    print(f"\n[4/4] Creating embeddings with '{EMBEDDING_MODEL}' and saving to ChromaDB...")
    
    # Clean previous database if it exists to ensure fresh ingestion
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"      Removing old database at '{CHROMA_PERSIST_DIRECTORY}'...")
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )

    print("\n" + "=" * 60)
    print("Ingestion Completed Successfully!")
    print(f"Total Pages Processed:  {total_pages}")
    print(f"Total Chunks Stored:    {len(chunks)}")
    print(f"Database Directory:     ./{CHROMA_PERSIST_DIRECTORY}")
    print("You can now launch the app with: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
