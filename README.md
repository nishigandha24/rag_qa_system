✅ README.md

# 🔍 RAG System with Gemini Flash

This project implements an advanced Retrieval-Augmented Generation (RAG) system using Google’s Gemini Flash API (free tier). It enhances Large Language Model (LLM) responses by retrieving and injecting relevant content from the **Umbrella Corporation Employee Handbook** PDF.


## 📦 Features
- **📄 PDF ingestion with overlap-aware chunking**: Efficiently extracts and segments PDF text while preserving page references for citation.
- **🧠 Embedding generation using MiniLM (sentence-transformers)**: Converts document chunks and user queries into semantic vector representations.
- **⚡ High-speed semantic search using FAISS**: Retrieves top-matching chunks from vector index for relevant, real-time responses.
- **🧾 Context-aware prompt construction with source tracking**: Injects retrieved chunks into a structured prompt with inline (Page X) citations.
- **🤖 LLM-powered response generation via Gemini Flash API**: Uses Google's Gemini (free-tier) to generate fluent, grounded answers based on retrieved content.
- **📊 Accuracy evaluation using ROUGE-L and cosine similarity**: Quantifies alignment between generated answers and expected responses for benchmarking.
- **🖥️ Streamlit UI for seamless interaction and evaluation**: Enables document indexing, question input, and answer evaluation in a user-friendly web app.
- **🧠 FAISS cache reuse via file hashing**: Caches processed PDFs to avoid redundant indexing and improve performance on repeat uploads.


## 📁 Project Structure
```bash
rag_system/
├── main.py                      ← Streamlit app
├── .env                         ← contains GOOGLE_API_KEY
├── requirements.txt             ← dependencies
├── rag_env.yaml                 ← (optional, for local use)
├── README.md
└── data/
    └── Umbrella_Corporation_Employee_Handbook.pdf
```


## ⚙️ Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/nishigandha24/rag_qa_system.git
cd rag_qa_system
```

### 2. Set Up Environment
Using Conda:
```bash
conda env create -f rag_env.yml
conda activate rag_env
```
Or manually with pip:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Get your Gemini Flash API key (from https://makersuite.google.com/app/apikey).

Create a .env file from the terminal.

For Linux / macOS / Git Bash / WSL:
```bash
echo "GOOGLE_API_KEY=your_actual_gemini_api_key_here" > .env
```

For Windows Command Prompt:
```cmd
echo GOOGLE_API_KEY=your_actual_gemini_api_key_here > .env
```

For Windows PowerShell:
```powershell
"GOOGLE_API_KEY=your_actual_gemini_api_key_here" | Out-File -Encoding ascii .env
```

### 4. Place Handbook PDF
Make sure `data/Umbrella_Corporation_Employee_Handbook.pdf` exists.


### 5. ▶️ Running the App
```bash
streamlit run main.py
```

Click “Index PDF Document” in the sidebar to process the handbook.

Enter a question like:
“What are the company’s core values?”
Optionally add an expected answer for ROUGE-L and cosine similarity evaluation.


## 💡 Sample Queries & Responses
| Question                            | Sample Answer
| ----------------------------------- | -----------------------------------------------------------------
| What are the company’s core values? | Innovation, Integrity, Collaboration, Excellence, Sustainability (Page 2)
| Describe the disciplinary process   | 4-stage progressive steps: Verbal → Written → Final → Termination (Page 47)
| How much sick leave is provided?    | 10 days per year for full-time employees (Page 38)


## 📊 Evaluation Metrics
| Metric         | Purpose                                                 |
| -------------- | ------------------------------------------------------- |
| **ROUGE-L**    | Measures overlap between generated and expected answers |
| **Cosine Sim** | Measures semantic similarity of retrieved context       |


## 🏗️ System Architecture

| Component                  |  Description                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 📄 **PDF Chunker**         | Extracts text from PDF using `PyPDF2`, splits it into overlapping chunks, and maps text to page numbers.     |
| 🧠 **Embedding Generator** | Uses `sentence-transformers` (MiniLM) to encode chunks and user queries into dense vector representations.   |
| 💾 **FAISS Indexer**       | Stores the document embeddings in a FAISS flat index for efficient top-k vector similarity search.           |
| 🔍 **Semantic Retriever**  | Encodes the user query, searches the FAISS index, and retrieves top-matching document chunks.                |
| 🧾 **Prompt Constructor**  | Formats retrieved chunks along with the question into a context-aware prompt with page-level citations.      |
| 🤖 **Gemini Flash API**    | Uses `google.generativeai` to send the prompt to Gemini Flash (free-tier) and get a context-grounded answer. |
| 📊 **Evaluator**           | Measures response quality using ROUGE-L (textual overlap) and cosine similarity (semantic alignment).        |

```
Mermaid:
graph TD
    A[📄 PDF Input] --> B[🔧 Chunking (with Overlap)]
    B --> C[🧠 Embedding Generator (MiniLM)]
    C --> D[💾 FAISS Vector Store]
    E[❓ User Query] --> F[🧠 Query Encoder (MiniLM)]
    F --> D
    D --> G[🔍 Top-k Semantic Retrieval]
    G --> H[🧾 Prompt Construction]
    H --> I[🤖 LLM API (Gemini Flash)]
    I --> J[✅ Final Answer + Source Citations]
    J --> K[📊 ROUGE-L + Cosine Evaluation]
```

![System Architecture](assets\system_architecture_diagram.png)


## ✅ Advanced Techniques Implemented
- Advanced chunking (with overlap & page tracking)
- Context selection & summarization
- Query optimization for Gemini Flash
- Semantic evaluation: Cosine Similarity + ROUGE-L
- Caching mechanism via local FAISS index


## 🚀 Future Improvements
-  Multi-LLM toggle: Add runtime option to switch between Gemini, Claude, and OpenAI (code partially supports this).
-  Reranking retrieved chunks: Apply embedding-based reordering after FAISS top-k for better relevance.
-  Highlight citations in output: Color-code (Page X) matches and link to source chunks in UI.
-  UI refinements:
    - Add download button for final answer.
    - Add feedback thumbs-up/down per answer.
-  Streamlit Cloud deployment: Automate build with secure secret management.
-  Dockerization: Wrap app into a container for reproducible local & cloud deployment.
-  Feedback loop & logging: Log queries, answers, and ratings for fine-tuning or improvement tracking.


## ⏱️ Time Tracking
- RAG Implementation: ~6 hours (dev + config + evaluation)
- Indexing Time:
    - ~8–9 seconds for a 60-page PDF
    - Cached using file hash for instant reuse
- Avg. Query Runtime (Gemini Flash Free API):
    - ~3.4–4.1 seconds per query
    - Avg. of 10 unique questions, including:
        - simple factual queries (e.g., PTO policy)
        - multi-page retrieval questions (e.g., disciplinary process)
- Evaluation (ROUGE-L + Cosine Similarity): ~0.5s per request


## 🔐 Security & Notes
- Keep .env secure and never commit it
- Free tier of Gemini Flash supports 60 RPM, ~20k tokens per minute
- For production, consider using Gemini Pro with billing enabled