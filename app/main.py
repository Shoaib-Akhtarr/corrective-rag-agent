from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from app.graph.crag_workflow import crag_agent
from app.evaluate import run_benchmark

app = FastAPI(
    title="Corrective RAG (CRAG) Agent",
    version="1.0.0",
    description="Self-Reflective Corrective RAG over MultiHop-RAG Adversarial Corpus."
)

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "CorrectiveRAGAgent"}

@app.post("/api/v1/crag/query")
def query_crag(req: QueryRequest) -> Dict[str, Any]:
    return crag_agent.run(req.query)

@app.get("/api/v1/crag/benchmark")
def benchmark():
    return run_benchmark()
