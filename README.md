# Agentic AI eBook RAG Chatbot (LangGraph + Pinecone)

A Retrieval-Augmented Generation chatbot that answers **only** from the *Agentic AI eBook*.
Built with Python, LangGraph, Pinecone, OpenAI embeddings and FastAPI (Streamlit UI included).

## Architecture

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (800 / 100 overlap)
    -> OpenAI text-embedding-3-small -> Pinecone (cosine, metadata: page_number, source)

Query -> LangGraph:
   retrieve --(low similarity)--> refuse --> END
      |
      v
   generate (context-only prompt) --> grade (LLM hallucination grader)
      ^                                  |
      |----- score < threshold, retry ---|
                                         |-- score ok --> END
                                         |-- retries exhausted --> refuse --> END
```

| Component | Technology |
|---|---|
| Ingestion | PyPDF + LangChain text splitter |
| Embeddings / Vector DB | OpenAI `text-embedding-3-small` + Pinecone (serverless) |
| Orchestration | LangGraph `StateGraph` (cyclic: generate <-> grade) |
| API / UI | FastAPI (`/chat`) and Streamlit |

### Confidence score
1. **Retrieval gate:** top cosine similarity below `MIN_RELEVANCE` (0.25) -> refuse without calling the generator (handles out-of-scope questions).
2. **Groundedness score:** a grader LLM scores 0-1 how well the answer is supported by the retrieved chunks. This is `confidence_score`.
3. **Retry loop:** if the score is below `GROUNDED_MIN` (0.6), regenerate up to `MAX_ATTEMPTS` (2), then refuse.
4. Refusals return `0.0`.

## Setup

```bash
git clone <this-repo> && cd rag-agentic-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in OPENAI_API_KEY and PINECONE_API_KEY
```

## Ingestion

```bash
python scripts/download_pdf.py  # or place the PDF at data/Ebook-Agentic-AI.pdf manually
python -m src.ingestion
```

## Run

```bash
uvicorn app:app --reload        # API at http://localhost:8000/docs
# or
streamlit run streamlit_app.py  # UI
```

### Example request
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI?"}'
```

### Response format
```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["...", "..."],
  "confidence_score": 0.92
}
```

## Tests

With the API running, in a second terminal:
```bash
python tests_sample_queries.py
```

### Sample output
> PASTE YOUR ACTUAL TEST OUTPUT HERE (5 in-document queries + the "capital of France" refusal).

## Project structure
```
rag-agentic-ai/
├── app.py                  # FastAPI
├── streamlit_app.py        # Streamlit UI
├── src/{config,ingestion,graph}.py
├── scripts/download_pdf.py
├── tests_sample_queries.py
├── requirements.txt
└── .env.example
```
