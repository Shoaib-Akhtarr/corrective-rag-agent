from fastapi.testclient import TestClient
from app.main import app
from app.graph.crag_workflow import crag_agent
from app.evaluate import run_benchmark

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_multihop_distractor_rejection():
    q = "Who chairs the HSM encryption quorum responsible for key rotations of the vector engine used by BioNova Therapeutics' VALOR-2 trial?"
    state = crag_agent.run(q)
    graded_ids = {d["doc_id"] for d in state["graded_docs"]}
    assert "MH-DOC-003" not in graded_ids
    assert "MH-DOC-006" not in graded_ids
    assert "MH-DOC-002" in graded_ids or "MH-DOC-007" in graded_ids
    assert state["is_grounded"] is True

def test_crag_benchmark_targets():
    report = run_benchmark()
    assert report["multi_hop_doc_recall"] >= 75.0
    assert report["distractor_rejection_rate"] == 100.0
    assert report["grounding_faithfulness"] == 100.0
