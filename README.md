# ⚖️ Project NyayLens: Agentic Legal AI Platform

NyayLens is an advanced, production-ready AI legal research platform designed specifically for the complexities of the Indian legal system. By moving beyond generic wrapper apps, NyayLens utilizes a **Hybrid RAG architecture**, **two-stage local summarization**, and a **custom FastAPI Agentic Router** to reduce hours of legal research into a few minutes of verifiable, hallucination-free reading.

## ✨ Key Features

- **Agentic Query Router:** Dynamically routes user intents between *Document Q&A*, *Global Precedent Search*, and *Structured Summarization* using a unified API endpoint.
- **Two-Stage Legal Summarization:** Bypasses generic LLMs to use specialized transformers. It first extracts critical facts globally using **Legal-BERT**, then synthesizes them abstractively using **Legal-PEGASUS**.
- **Hybrid RAG Retrieval:** Fuses dense vector search (`FAISS`) with sparse keyword matching (`SQLite FTS5 / BM25`) across **298,000+** indexed Supreme Court paragraphs.
- **Zero-Hallucination Guardrails:** Implements a strict "Legal Auditor" prompting mechanism. The conversational LLM is structurally forced to cite only retrieved paragraphs, preventing invented case laws.
- **Semantic Truncation:** For uploaded PDFs exceeding context limits, the system dynamically scores and extracts the most legally important 30K characters rather than blindly cutting the text.

## 🛠 Tech Stack

- **Backend:** FastAPI, Python, Uvicorn, SQLite FTS5
- **Frontend:** React, Vite, localForage (IndexedDB for stateless session memory)
- **AI / NLP Models:** 
  - *Conversational & RAG Synthesis:* Llama-3.1-8B (via Groq API)
  - *Embeddings:* BAAI/bge-base-en-v1.5
  - *Cross-Encoder Reranking:* BAAI/bge-reranker-base
  - *Extraction & Summarization:* nlpaueb/legal-bert-base-uncased, nsi319/legal-pegasus
- **Vector Database:** FAISS (CPU)
- **Evaluation:** RAGAS framework (Context Relevance, Faithfulness, Answer Relevance)

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Git LFS](https://git-lfs.com/) (Required to pull the database and model weights)
- Groq API Key

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/NyayLens.git
cd NyayLens
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables
echo "GROQ_API_KEY=your_key_here" > .env

# Run the FastAPI server
uvicorn src.api.main:app --reload
```

### 3. Frontend Setup
```bash
cd web
npm install

# Optional: Add .env.local for Vite if needed
# echo "VITE_API_URL=http://127.0.0.1:8000" > .env.local

# Start the Vite development server
npm run dev
```

## 🧠 Architecture Overview

NyayLens was explicitly engineered to handle the nuances of Indian judgments:
1. **Temporal Precedent Awareness:** The retrieval engine is chronologically aware, instructing the AI to prioritize newer Constitution Bench rulings over older decisions, explicitly warning users of conflicting precedents.
2. **Context Preservation:** Standard apps crash when feeding 100-page PDFs to LLMs. NyayLens chunks, scores, and filters the document using Legal-BERT before it ever touches a generative model.
3. **Stateless Scalability:** The frontend uses IndexedDB (`localForage`) to store massive chat histories securely on the client, minimizing backend database bloat.

## 📊 RAGAS Evaluation
The system includes a native benchmarking suite to ensure legal accuracy. By hitting the `/api/evaluate` endpoint, the pipeline self-scores responses for:
- **Faithfulness:** Are all claims verifiable against the source context?
- **Context Relevance:** Did the retriever fetch the correct paragraphs?
- **Answer Relevance:** Does the synthesis actually answer the user's prompt?

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

---
*Built to bring extreme speed and accuracy to legal research.*
