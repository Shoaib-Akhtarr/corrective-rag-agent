from fastapi.testclient import TestClient
from app.evaluate import run_benchmark
from app.graph.crag_workflow import crag_agent, extract_generic_entities
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["documents_indexed"] >= 8
    assert body["chunks_indexed"] >= 10
    assert "qdrant_mode" in body
    assert "embedder_backend" in body


def test_generic_entity_extraction_without_hardcoding():
    # Entities completely outside the seeded PDF corpus must still extract cleanly
    sample = "Under the Geneva Cloud Pact, OrbitalVault-v9 supersedes AlphaStore-v1 across us-gov-1."
    ents = extract_generic_entities(sample)
    assert any("OrbitalVault-v9" in e or "AlphaStore-v1" in e or "Geneva Cloud Pact" in e for e in ents)


def test_live_document_ingestion_and_contradiction_resolution():
    from app.retrieval.hybrid_index import hybrid_index
    addendum = (
        b"Document Authority: REGULATORY_BINDING | Effective Date: 2026-04-01\n\n"
        b"All eu-central-2 HelixDB clusters must rotate encryption keys every 14 days under Zurich HSM Quorum oversight."
    )
    r = client.post(
        "/api/v1/ingest",
        files={"file": ("09_Regulatory_14Day_Rotation_Order.md", addendum, "text/markdown")},
    )
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1
    hybrid_index.reload()


def test_temporal_claim_diff_and_distractor_rejection():
    q = "Who authorizes HelixDB encryption key rotations in eu-central-2, and is the chair compliant with EMA DORA?"
    state = crag_agent.run(q)
    graded_ids = {d["doc_id"] for d in state["graded_docs"]}
    assert "07_DISTRACTOR_Apex_Sports_Apparel_Earnings" not in graded_ids
    assert "03_Zurich_HSM_Board_Resolution_2026_Q1" in graded_ids
    assert state["claim_diff"]["contradictions_detected"] >= 1
    assert state["is_grounded"] is True
    assert state["faithfulness_score"] >= 0.5


def test_claim_diff_benchmark():
    report = run_benchmark()
    assert report["authoritative_doc_recall_pct"] == 100.0
    assert report["distractor_rejection_rate_pct"] == 100.0
    assert report["grounding_faithfulness_pct"] == 100.0
