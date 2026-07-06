# ResearchGenie 🧞‍♂️

A production-ready Multi-Agent AI Research Assistant built using LangGraph, FastAPI, Streamlit, and Groq LLM.

## Features
- Multi-agent workflow orchestrated by LangGraph
- Semantic search & document retrieval using FAISS and HuggingFace
- PDF download and text extraction
- Summarization, comparison, citation generation, and research gap identification
- Downloadable PDF report
- Clean Streamlit UI

## Prerequisites
- Python 3.10+
- Groq API Key

## Setup & Run Locally

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

2. Run the FastAPI backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. In a new terminal, run the Streamlit frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Setup & Run via Docker

```bash
docker-compose up --build
```

Access the app at http://localhost:8501.
