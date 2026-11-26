from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.graph.crag_workflow import crag_agent
from app.retrieval.hybrid_index import hybrid_index

app = FastAPI(
    title="Temporal & Contradiction-Aware Claim-Diff RAG",
    version="2.0.0",
)


class QueryReq(BaseModel):
    query: str


@app.get("/health")
def health():
    docs = {c["doc_id"] for c in hybrid_index.chunks}
    return {
        "status": "healthy",
        "documents_indexed": len(docs),
        "chunks_indexed": len(hybrid_index.chunks),
    }


@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    raw = await file.read()
    added = hybrid_index.ingest_bytes(file.filename, raw)
    return {"filename": file.filename, "chunks_indexed": added, "total_chunks": len(hybrid_index.chunks)}


@app.post("/api/v1/query")
def run_claim_diff_rag(req: QueryReq):
    return crag_agent.run(req.query)
