"""
app.py — HR Policy Assistant using RAG
Streamlit chat interface for asking questions grounded in the Enso Group HR Manual.
"""

import os
import urllib.request
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment configurations
load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
PDF_PATH = os.getenv("PDF_PATH", "data/hr_manual.pdf")

# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="wide",
)

st.title("💼 HR Policy Assistant (RAG)")
st.caption("Ask questions about the Enso Group HR Policy Manual. All answers are strictly grounded with page references.")


def check_ollama(base_url: str) -> bool:
    """Verify Ollama server accessibility."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=4) as response:
            return response.status == 200
    except Exception:
        return False


# 1. Error handling: Missing database check
if not os.path.exists(CHROMA_PERSIST_DIRECTORY) or not os.listdir(CHROMA_PERSIST_DIRECTORY):
    st.error("⚠️ **ChromaDB vector database not found!**")
    st.info("The knowledge base has not been created yet. Please run the ingestion command in your terminal first:")
    st.code("python ingest.py", language="bash")
    st.stop()

# 2. Error handling: Ollama connection check
if not check_ollama(OLLAMA_BASE_URL):
    st.error(f"⚠️ **Cannot connect to Ollama at `{OLLAMA_BASE_URL}`.**")
    st.info("Please make sure the Ollama background service is running:")
    st.code("ollama serve   # or brew services start ollama", language="bash")
    st.stop()

# 3. Error handling: Missing PDF warning
if not os.path.exists(PDF_PATH) and not os.path.exists(os.path.basename(PDF_PATH)):
    st.warning(f"Note: Original source PDF '{PDF_PATH}' was not found in directory, but ChromaDB vector index is active.")


# Cached Vector Store & Retriever Initialization
@st.cache_resource(show_spinner=False)
def get_retriever():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    # Retrieve top 4 most relevant chunks
    return vector_store.as_retriever(search_kwargs={"k": 4})


# Cached LLM Initialization
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


try:
    retriever = get_retriever()
    llm = get_llm()
except Exception as e:
    st.error(f"Error initializing RAG components: {e}")
    st.stop()


# Sidebar Configuration & Test Question Shortcuts
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**LLM Model:** `{LLM_MODEL}`")
    st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    st.write(f"**Vector Store:** `ChromaDB ({CHROMA_PERSIST_DIRECTORY})`")
    st.write(f"**Source Document:** `{PDF_PATH}`")
    
    st.divider()
    st.subheader("📋 Assignment Test Questions")
    sample_questions = [
        "How many days of Privilege Leave (PL) does a permanent employee get per year?",
        "What is the probation period for employees up to Senior Manager?",
        "What is the company's health-insurance premium amount?",
        "What is the current GST rate in India?",
        "How many days of maternity leave are given?",
    ]

    selected_sample = None
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            selected_sample = q

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous conversation
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Retrieved Policy Evidence"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}** — *{src['section']}* ({src['source']})")
                    st.caption(src["snippet"])


# Grounded RAG Query Function
def answer_question(question: str):
    # 1. Retrieve relevant chunks
    try:
        docs = retriever.invoke(question)
    except Exception as err:
        return f"Error querying vector store: {err}", []

    if not docs:
        return "Not found", []

    # 2. Format context with explicit page and section tags
    formatted_context_list = []
    sources = []
    for d in docs:
        page_num = d.metadata.get("page_number", d.metadata.get("page", 0) + 1)
        sec_name = d.metadata.get("policy_section", "General Policy")
        src_name = d.metadata.get("source_file", os.path.basename(PDF_PATH))
        content = d.page_content.strip()
        
        formatted_context_list.append(f"--- [Page {page_num} | Policy Section: {sec_name}] ---\n{content}")
        sources.append({
            "page": page_num,
            "section": sec_name,
            "source": src_name,
            "snippet": content[:300] + ("..." if len(content) > 300 else "")
        })

    context_str = "\n\n".join(formatted_context_list)

    # 3. Strict prompt enforcement
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an official HR Policy Assistant for Enso Group.
Answer the user's question using ONLY the provided Document Context.

RULES:
1. If the provided context explicitly contains the facts to answer the question:
   First, provide the factual answer clearly in 1-2 complete sentences.
   Then, on a new line, provide the exact source reference:
   Source: Page <page_number>, Policy Section: <policy_section>

2. If the provided context does NOT explicitly answer the question (or if the topic is absent, such as health insurance premium or GST rate):
   Respond with EXACTLY the words 'Not found' and absolutely nothing else. Do not add explanations, notes, or source lines when not found.

3. Do not assume, guess, extrapolate, or use outside knowledge.
"""),
        ("user", """Document Context:
{context}

Question:
{question}
""")
    ])

    chain = prompt | llm | StrOutputParser()
    
    try:
        raw_reply = chain.invoke({"context": context_str, "question": question}).strip()
    except Exception as err:
        return f"Error communicating with Ollama: {err}", []

    # Normalization: if model indicates absence or replies 'Not found', return exactly 'Not found'
    first_line = raw_reply.split('\n')[0].strip().lower()
    raw_lower = raw_reply.lower()
    if (
        first_line.startswith("not found")
        or "not found" in raw_lower
        or "not mentioned" in raw_lower
        or "does not contain" in raw_lower
        or "no information" in raw_lower
        or "cannot find" in raw_lower
    ):
        return "Not found", []

    return raw_reply, sources


# Handle User Input
user_input = st.chat_input("Ask a question about the HR Policy Manual...")

# If a sidebar shortcut button was clicked, use it
if selected_sample:
    user_input = selected_sample

if user_input:
    # Append user question
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Searching HR Manual..."):
            reply, sources = answer_question(user_input)
            st.markdown(reply)
            if sources:
                with st.expander("📄 Retrieved Policy Evidence"):
                    for src in sources:
                        st.markdown(f"**Page {src['page']}** — *{src['section']}* ({src['source']})")
                        st.caption(src["snippet"])

    # Save assistant response to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": reply,
        "sources": sources
    })
