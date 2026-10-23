"""
ingest.py — Ingestion Pipeline for Enso Group HR Policy Manual
Extracts text from hr_manual.pdf, preserves original PDF page numbers and policy section headings,
computes embeddings using Ollama (nomic-embed-text), and stores them in ChromaDB.
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

# Comprehensive mapping of original PDF page numbers (1-indexed) to Policy Section Headings
PAGE_SECTIONS = {
    1: "Cover",
    2: "Table of Contents",
    3: "Table of Contents",
    4: "Introduction",
    5: "Section 1: Employing Staff",
    6: "Section 1.1: Recruitment / Section 1.2: Induction Policy",
    7: "Section 1.2: Induction Policy",
    8: "Section 1.3: Employee's Employment",
    9: "Section 1.4: Standards of Business Conduct",
    10: "Section 1.4: Standards of Business Conduct",
    11: "Section 1.4: Standards of Business Conduct",
    12: "Section 1.5: Internet and Email Usage Policy and Guidelines",
    13: "Section 1.5: Internet and Email Usage Policy and Guidelines",
    14: "Section 1.5: Internet and Email Usage Policy and Guidelines",
    15: "Section 1.5: Internet and Email Usage Policy and Guidelines",
    16: "Section 2: Managing Employee Performance",
    17: "Section 2.1: Performance Appraisal Policy",
    18: "Section 2.2: Controlling Absence Policy",
    19: "Section 2.3: Disciplinary Procedure",
    20: "Section 2.3: Disciplinary Procedure",
    21: "Section 2.4: Capability Procedure",
    22: "Section 3: Dealing with Employee Concerns",
    23: "Section 3.1: Grievance Procedure",
    24: "Section 3.1: Grievance Procedure",
    25: "Section 3.2: Prevention of Bullying and Harassment at Work",
    26: "Section 3.3: Whistleblowing Policy",
    27: "Section 3.3: Whistleblowing Policy",
    28: "Section 4: Supporting Our Staff in the Workplace",
    29: "Section 4.1: Equality Policy",
    30: "Section 4.1: Equality Policy",
    31: "Section 4.2: Drug and Alcohol Abuse Policy / Section 4.3: Health and Safety Policy",
    32: "Section 4.3: Health and Safety Policy",
    33: "Section 4.3: Health and Safety Policy",
    34: "Company Rules & Regulations: Staff Handbook & Property",
    35: "Company Rules & Regulations: IT Security & Smoking",
    36: "Company Rules & Regulations: Confidentiality & Safety Evacuation",
    37: "Company Rules & Regulations: Conflict of Interest & Acceptance of Advantage",
    38: "Company Rules & Regulations: Theft Attempted",
    39: "Annexure - I",
    40: "Annexure - I: [A] On Boarding Process",
    41: "Annexure - I: [A] On Boarding Process",
    42: "Annexure - I: [B] Probation Policy",
    43: "Annexure - I: [B] Probation Policy",
    44: "Annexure - I: [C] Training and Development",
    45: "Annexure - I: [D] Relocation Policy",
    46: "Annexure - I: [E] Off-Boarding Process",
    47: "Annexure - I: [E] Off-Boarding Process",
    48: "Annexure - II",
    49: "Annexure - II: [A] Equal Opportunities",
    50: "Annexure - II: [A] Equal Opportunities",
    51: "Annexure - II: [B] Data card/Visiting Cards/Emails Policy",
    52: "Annexure - III",
    53: "Annexure - III: [A] Dress Code",
    54: "Annexure - III: [B] Work Timing Policy",
    55: "Annexure - III: [C] Punctuality and Attendance",
    56: "Annexure - III: [C] Punctuality and Attendance",
    57: "Annexure - III: [D] Compensation Policy",
    58: "Annexure - III: [E] Staff Advances",
    59: "Annexure - III: [F] Employment of Relatives Policy",
    60: "Annexure - III: [G] Leave Policy",
    61: "Annexure - III: [G] Leave Policy",
    62: "Annexure - III: [G] Leave Policy",
    63: "Annexure - III: [H] Reward & Recognition",
    64: "Annexure - III: [I] Whistle Blower Policy & Anti Sexual Harassment Policy",
    65: "Annexure - III: [I] Whistle Blower Policy & Anti Sexual Harassment Policy",
    66: "Annexure - IV: Traveling Policies",
    67: "Annexure - IV: [A] Long Distance Travel Policy",
    68: "Annexure - IV: [A] Long Distance Travel Policy",
    69: "Annexure - IV: [A] Long Distance Travel Policy",
    70: "Annexure - IV: [A] Long Distance Travel Policy",
    71: "Annexure - IV: [A] Long Distance Travel Policy",
    72: "Annexure - IV: [A] Long Distance Travel Policy",
    73: "Annexure - IV: [B] Short Distance Travel Policy",
    74: "Annexure - IV: [B] Short Distance Travel Policy",
    75: "Annexure - IV: [B] Short Distance Travel Policy",
    76: "Annexure - IV: [B] Short Distance Travel Policy",
    77: "Annexure - IV: [B] Short Distance Travel Policy",
    78: "Annexure - IV: [B] Short Distance Travel Policy",
    79: "Closing Back Cover"
}


def check_ollama_connection(base_url: str) -> bool:
    """Verify that Ollama server is running and accessible."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    print("=" * 65)
    print("HR Policy Assistant — Document Ingestion Pipeline")
    print("=" * 65)

    # 1. Check if the PDF file exists
    if not os.path.exists(PDF_PATH):
        # Also check root if data/hr_manual.pdf not found
        fallback_path = os.path.basename(PDF_PATH)
        if os.path.exists(fallback_path):
            actual_pdf = fallback_path
        else:
            print(f"\n[ERROR] Missing PDF file: '{PDF_PATH}' was not found.")
            sys.exit(1)
    else:
        actual_pdf = PDF_PATH

    print(f"\n[1/4] Checking Ollama server at {OLLAMA_BASE_URL}...")
    if not check_ollama_connection(OLLAMA_BASE_URL):
        print(f"[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}.")
        print("Please start Ollama in your terminal using: ollama serve")
        sys.exit(1)
    print("      Ollama connection verified.")

    # 2. Load the PDF document
    print(f"\n[2/4] Loading PDF document from '{actual_pdf}'...")
    loader = PyPDFLoader(actual_pdf)
    raw_docs = loader.load()
    if not raw_docs:
        print("[ERROR] The PDF file could not be read or contains no extractable text.")
        sys.exit(1)

    total_pages = len(raw_docs)
    print(f"      Successfully extracted {total_pages} page(s).")

    # Add 1-based page numbers and policy section headings to document metadata
    for doc in raw_docs:
        page_num = doc.metadata.get("page", 0) + 1
        section_name = PAGE_SECTIONS.get(page_num, f"Section Page {page_num}")
        doc.metadata["page_number"] = page_num
        doc.metadata["policy_section"] = section_name
        doc.metadata["source_file"] = os.path.basename(actual_pdf)

    # 3. Split the text into chunks while preserving section heading & page context
    print("\n[3/4] Chunking document text with section awareness...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)

    # Prepend section heading and page number to each chunk's content for optimal retrieval
    for chunk in chunks:
        sec = chunk.metadata.get("policy_section", "General")
        pg = chunk.metadata.get("page_number", 1)
        chunk.page_content = f"[Policy Section: {sec} | Page: {pg}]\n\n{chunk.page_content}"

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

    print("\n" + "=" * 65)
    print("Ingestion Completed Successfully!")
    print(f"Total Pages Processed:  {total_pages}")
    print(f"Total Chunks Stored:    {len(chunks)}")
    print(f"Database Directory:     ./{CHROMA_PERSIST_DIRECTORY}")
    print("Section headings and page numbers preserved in ChromaDB metadata.")
    print("=" * 65)


if __name__ == "__main__":
    main()
