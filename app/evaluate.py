import time
from app.graph.crag_workflow import crag_agent
from app.retrieval.hybrid_index import hybrid_index

BENCHMARK_CASES = [
    {
        "query": "What is the current default vector storage layer in eu-central-2, and which earlier policy did it supersede?",
        "expected_winner_doc": "02_NovaVector_Acquisition_Addendum_2025_Q3",
        "expected_superseded_doc": "01_Apex_Sovereign_Cloud_Policy_2025_Q1",
        "distractor": "07_DISTRACTOR_Apex_Sports_Apparel_Earnings",
    },
    {
        "query": "Who authorizes HelixDB encryption key rotations in eu-central-2, and is the chair compliant with EMA DORA?",
        "expected_winner_doc": "03_Zurich_HSM_Board_Resolution_2026_Q1",
        "expected_superseded_doc": "01_Apex_Sovereign_Cloud_Policy_2025_Q1",
        "distractor": "07_DISTRACTOR_Apex_Sports_Apparel_Earnings",
    },
    {
        "query": "What GPU unit threshold triggers SLA penalty credits for Meridian AI Labs, and what credit was issued?",
        "expected_winner_doc": "05_Meridian_AI_SLA_Amendment_2026_Q1",
        "expected_superseded_doc": "01_Apex_Sovereign_Cloud_Policy_2025_Q1",
        "distractor": "08_DISTRACTOR_Meridian_Maritime_Fleet_Update",
    },
]


def run_benchmark():
    latencies = []
    winner_hits = 0
    contradiction_hits = 0
    distractor_rejected = 0
    grounded_count = 0

    for case in BENCHMARK_CASES:
        t0 = time.perf_counter()
        state = crag_agent.run(case["query"])
        latencies.append((time.perf_counter() - t0) * 1000)

        graded_docs = {d["doc_id"] for d in state["graded_docs"]}
        if case["expected_winner_doc"] in graded_docs:
            winner_hits += 1
        if case["distractor"] not in graded_docs:
            distractor_rejected += 1
        if state["claim_diff"]["contradictions_detected"] >= 1:
            contradiction_hits += 1
        if state["is_grounded"]:
            grounded_count += 1

    n = len(BENCHMARK_CASES)
    return {
        "documents_indexed": len({c["doc_id"] for c in hybrid_index.chunks}),
        "chunks_indexed": len(hybrid_index.chunks),
        "queries_evaluated": n,
        "authoritative_doc_recall_pct": round(100 * winner_hits / n, 1),
        "contradiction_detection_rate_pct": round(100 * contradiction_hits / n, 1),
        "distractor_rejection_rate_pct": round(100 * distractor_rejected / n, 1),
        "grounding_faithfulness_pct": round(100 * grounded_count / n, 1),
        "avg_latency_ms": round(sum(latencies) / n, 2),
    }


if __name__ == "__main__":
    print(run_benchmark())
