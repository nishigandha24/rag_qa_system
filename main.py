# main.py

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
import google.generativeai as genai

# ======================
# Load .env & configure Gemini
# ======================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ======================
# Global Configuration
# ======================
PDF_PATH = "data/Umbrella_Corporation_Employee_Handbook.pdf"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
ROUGE = Rouge()

# ======================
# Document Processing with advanced chunking
# ======================
@st.cache_data(show_spinner=False)
def load_and_chunk_pdf(pdf_path, chunk_size=500, overlap=100):
    reader = PdfReader(pdf_path)
    full_text = ""
    page_starts = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        page_starts.append((len(full_text), i + 1))
        full_text += text + "\n"

    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk_text = full_text[i:i + chunk_size].strip()
        page_num = next(p for (start, p) in reversed(page_starts) if start <= i)
        chunks.append({"text": chunk_text, "page": page_num})

    return chunks

# ======================
# Embedding + Indexing
# ======================
@st.cache_data(show_spinner=False)
def create_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = EMBED_MODEL.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def save_index(index, chunks):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# ======================
# Semantic Search
# ======================
def semantic_search(query, index, chunks, top_k=5):
    query_embedding = EMBED_MODEL.encode([query])
    _, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

# ======================
# Response Generation with Gemini
# ======================
def generate_response(query, retrieved_chunks):
    context = "\n---\n".join(
        [f"(Page {chunk['page']})\n{chunk['text']}" for chunk in retrieved_chunks]
    )

    prompt = f"""
Answer the following question using only the information provided in the context below.
Cite the source using page numbers in parentheses like (Page X).

Context:
{context}

Question: {query}

Answer:
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ======================
# Evaluation (ROUGE + Cosine)
# ======================
def evaluate_response(generated, expected, context_chunks):
    rouge_score = ROUGE.get_scores(generated, expected)[0]["rouge-l"]["f"]
    context_text = " ".join(c["text"] for c in context_chunks)
    vecs = EMBED_MODEL.encode([generated, context_text])
    cosine_sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    return rouge_score, cosine_sim

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Umbrella QA Bot", layout="wide")
st.title("🧠 QA Bot: Umbrella Corporation Employee Handbook")

query = st.text_input("🔍 Ask your question:")
expected = st.text_area("📌 (Optional) Expected Answer for Evaluation:")

if st.sidebar.button("📥 Index PDF Document"):
    with st.spinner("Indexing document..."):
        chunks = load_and_chunk_pdf(PDF_PATH)
        index, _ = create_faiss_index(chunks)
        save_index(index, chunks)
        st.success("✅ Document indexed and saved successfully!")

if query:
    index, chunks = load_index()
    retrieved_chunks = semantic_search(query, index, chunks)
    response = generate_response(query, retrieved_chunks)

    st.subheader("💡 Answer")
    st.markdown(response)

    st.subheader("📚 Source Context")
    cleaned_chunks = []
    for chunk in retrieved_chunks:
        text = chunk['text'].strip().replace('\n', ' ')
        cleaned_chunks.append(f"(Page {chunk['page']}) {text}")
    context_paragraph = " ".join(cleaned_chunks)
    st.text(context_paragraph)


    if expected.strip():
        rouge, cosine = evaluate_response(response, expected, retrieved_chunks)
        st.subheader("📊 Evaluation Metrics")
        st.markdown(f"- **ROUGE-L**: `{rouge:.4f}`")
        st.markdown(f"- **Cosine Similarity**: `{cosine:.4f}`")
