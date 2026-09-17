# ---- SQLite fix for ChromaDB on Streamlit Cloud (MUST be at the very top) ----
# Streamlit Cloud ships an old sqlite3; ChromaDB needs a newer one.
# This swaps in pysqlite3. Requires 'pysqlite3-binary' in requirements.txt.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb
import pypdf
import os

# ---- Page title + description ----
st.title("Lab 4: Course Information RAG Chatbot")
st.write(
    "Ask a question about the course documents. This chatbot uses a RAG "
    "(Retrieval-Augmented Generation) pipeline: it searches a ChromaDB vector "
    "database built from the course PDFs, then answers using the most relevant "
    "content it finds."
)

# ---- OpenAI client from secrets ----
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API Key not found in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# ---- Folder holding the 7 course PDFs ----
PDF_FOLDER = "Lab4_pdfs"

# ---- Helper: read a single PDF into text ----
def read_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---- Helper: get an OpenAI embedding for a piece of text ----
def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ---- Part A: build the ChromaDB collection (only once) ----
def build_vectordb():
    """Reads the 7 PDFs, embeds them, and stores a ChromaDB collection."""
    chroma_client = chromadb.Client()
    # get_or_create avoids errors if it already exists in this client
    collection = chroma_client.get_or_create_collection(name="Lab4Collection")

    # Read each PDF, embed it, and add to the collection
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            doc_text = read_pdf(file_path)
            embedding = get_embedding(doc_text)
            collection.add(
                documents=[doc_text],
                embeddings=[embedding],
                ids=[filename],                       # filename as the key
                metadatas=[{"filename": filename}]    # metadata
            )
    return collection

# ---- Only build the DB once, then keep it in session_state (saves cost) ----
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building the vector database (first run only)..."):
        st.session_state.Lab4_VectorDB = build_vectordb()

collection = st.session_state.Lab4_VectorDB

# ============================================================================
# PART A TEST CODE — uncomment to verify the DB, then re-comment for Part B
# ============================================================================
# test_query = "Generative AI"
# test_emb = get_embedding(test_query)
# results = collection.query(query_embeddings=[test_emb], n_results=3)
# st.write(f"Top 3 documents for '{test_query}':")
# for i, doc_id in enumerate(results["ids"][0], start=1):
#     st.write(f"{i}. {doc_id}")
# ============================================================================

# ---- Part B: RAG chatbot ----

# Initialize chat history (unique key so it won't collide with other pages)
if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the course documents."}
    ]

# Show conversation so far
for msg in st.session_state.lab4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the course..."):

    # Save + show the user's message
    st.session_state.lab4_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Retrieve relevant documents from the vector DB ----
    query_emb = get_embedding(user_input)
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    retrieved_docs = results["documents"][0]
    retrieved_names = results["ids"][0]

    # Build the context block from retrieved documents
    context = ""
    for name, doc in zip(retrieved_names, retrieved_docs):
        context += f"\n\n--- From {name} ---\n{doc[:3000]}"  # truncate per doc

    # ---- Prompt engineering: give the LLM the retrieved context ----
    system_prompt = (
        "You are a helpful course-information assistant. Answer the user's question "
        "using the course document excerpts provided below. If you use information "
        "from these documents, clearly say so and mention which file(s) it came from. "
        "If the answer is not in the documents, say you couldn't find it in the course "
        "materials.\n\n"
        f"COURSE DOCUMENT EXCERPTS:{context}"
    )

    # ---- Generate the answer (streamed) ----
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-5-mini",   # per the assignment's suggestion
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    # Save assistant reply
    st.session_state.lab4_messages.append({"role": "assistant", "content": response})