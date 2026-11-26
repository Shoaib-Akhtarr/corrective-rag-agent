from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.graph.crag_workflow import crag_agent
from app.llm.client import llm_gateway
from app.retrieval.hybrid_index import hybrid_index

app = FastAPI(
    title="Temporal & Contradiction-Aware Claim-Diff RAG",
    version="3.0.0",
    description="LangGraph + Qdrant Hybrid Search + PyTorch Neural NLI Claim-Diff + Constrained LLM Synthesis",
)


class QueryReq(BaseModel):
    query: str
    min_authority_tier: int | None = None


@app.get("/health")
def health():
    docs = {c["doc_id"] for c in hybrid_index.chunks}
    llm_cfg = llm_gateway.resolve_provider()
    return {
        "status": "healthy",
        "documents_indexed": len(docs),
        "chunks_indexed": len(hybrid_index.chunks),
        "qdrant_mode": hybrid_index.qdrant_mode,
        "embedder_backend": hybrid_index.embedder.backend_name,
        "llm_provider": llm_cfg["provider"],
        "llm_model": llm_cfg["model"],
    }


@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    raw = await file.read()
    added = hybrid_index.ingest_bytes(file.filename, raw)
    return {
        "filename": file.filename,
        "chunks_indexed": added,
        "total_chunks": len(hybrid_index.chunks),
        "qdrant_mode": hybrid_index.qdrant_mode,
    }


@app.post("/api/v1/query")
def run_claim_diff_rag(req: QueryReq):
    return crag_agent.run(req.query)


@app.post("/api/v1/query/stream")
def stream_claim_diff_rag(req: QueryReq):
    """Streams constrained synthesis tokens after executing retrieval & Claim-Diff verification."""
    state = crag_agent.run(req.query)
    claims = state["claim_diff"].get("authoritative_claims", [])
    evidence = "\n".join(
        f"[{c['doc_id']} p.{c['page']} | {c['authority_label']} | {c['effective_date']}] {c['text']}"
        for c in claims[:5]
    )

    def _gen():
        yielded = False
        for token in llm_gateway.stream_text(
            "Answer strictly using these authoritative claims with [doc_id p.X] citations.",
            f"Question: {req.query}\n\nClaims:\n{evidence}",
        ):
            yielded = True
            yield token
        if not yielded:
            yield state["draft_answer"]

    return StreamingResponse(_gen(), media_type="text/plain")
