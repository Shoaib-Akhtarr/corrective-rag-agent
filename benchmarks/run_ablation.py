import json
import os
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.retrieval.store import store
from app.graph.crag_workflow import crag_agent
from app.evaluate import run_benchmark


def eval_tfidf_baseline():
    docs = store.documents
    texts = [f"{d['title']} {d['content']}" for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    doc_mat = vec.fit_transform(texts).toarray()
    recalls = []
    distractor_hits = 0
    total_distractors = 0
    for q in store.benchmark_queries:
        q_vec = vec.transform([q["query"]]).toarray()[0]
        sims = doc_mat @ q_vec
        top_idx = np.argsort(sims)[::-1][:5]
        top_ids = {docs[i]["doc_id"] for i in top_idx}
        expected = set(q["expected_docs"])
        recalls.append(len(expected & top_ids) / len(expected))
        distractors = set(q.get("distractor_docs", []))
        total_distractors += len(distractors)
        distractor_hits += len(distractors & top_ids)
    return {
        "method": "Naive TF-IDF Top-5 (No Grader, No Multi-Hop)",
        "multi_hop_doc_recall_pct": round(100 * sum(recalls) / len(recalls), 1),
        "distractor_rejection_rate_pct": round(100 * (1 - distractor_hits / max(1, total_distractors)), 1),
    }


def main():
    baseline = eval_tfidf_baseline()
    crag_report = run_benchmark()
    ablation = {
        "corpus_docs": len(store.documents),
        "benchmark_queries": len(store.benchmark_queries),
        "baseline_naive_tfidf": baseline,
        "crag_multihop_ml_grader": {
            "method": "BM25 + Multi-Hop Entity Decomposer + Trained ML Distractor Grader",
            "multi_hop_doc_recall_pct": crag_report["multi_hop_doc_recall"],
            "distractor_rejection_rate_pct": crag_report["distractor_rejection_rate"],
            "grounding_faithfulness_pct": crag_report["grounding_faithfulness"],
            "avg_latency_ms": crag_report["avg_latency_ms"],
        },
    }
    out_json = os.path.join(ROOT, "benchmarks", "ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)

    md = (
        "# Corrective RAG Ablation Report\n\n"
        f"- **Corpus:** {ablation['corpus_docs']} documents | **Queries:** {ablation['benchmark_queries']}\n\n"
        "| Method | Multi-Hop Recall | Distractor Rejection | Faithfulness |\n"
        "|---|---:|---:|---:|\n"
        f"| {baseline['method']} | {baseline['multi_hop_doc_recall_pct']}% | {baseline['distractor_rejection_rate_pct']}% | N/A |\n"
        f"| {ablation['crag_multihop_ml_grader']['method']} | {crag_report['multi_hop_doc_recall']}% | {crag_report['distractor_rejection_rate']}% | {crag_report['grounding_faithfulness']}% |\n"
    )
    with open(os.path.join(ROOT, "benchmarks", "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
