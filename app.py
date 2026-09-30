from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def health():
    return {"status": "ok", "docs": "/docs"}


@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    r = graph.invoke({
        "query": req.query, "context": [], "pages": [], "retrieval_score": 0.0,
        "answer": "", "grounded_score": 0.0, "attempts": 0,
    })
    return QueryResponse(
        query=req.query,
        final_answer=r["answer"],
        retrieved_context_chunks=r["context"],
        confidence_score=r["grounded_score"],
    )
