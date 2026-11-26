import time
from app.retrieval.store import store
from app.graph.crag_workflow import crag_agent

def run_benchmark():
    queries = store.benchmark_queries
    total = len(queries)
    doc_recall_scores = []
    distractor_rejections = 0
    total_distractors = 0
    grounded_count = 0
    latencies = []

    for q in queries:
        start = time.perf_counter()
        result = crag_agent.run(q["query"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved_ids = {d["doc_id"] for d in result["graded_docs"]}
        expected_ids = set(q["expected_docs"])
        recall = len(retrieved_ids.intersection(expected_ids)) / len(expected_ids)
        doc_recall_scores.append(recall)

        # Verify zero distractors leaked into final graded docs
        distractors = set(q["distractor_docs"])
        total_distractors += len(distractors)
        leaked = retrieved_ids.intersection(distractors)
        distractor_rejections += (len(distractors) - len(leaked))

        if result["is_grounded"]:
            grounded_count += 1

    avg_recall = sum(doc_recall_scores) / total
    distractor_rej_rate = distractor_rejections / total_distractors if total_distractors else 1.0
    avg_latency = sum(latencies) / total

    report = {
        "total_multihop_queries": total,
        "multi_hop_doc_recall": round(avg_recall * 100, 1),
        "distractor_rejection_rate": round(distractor_rej_rate * 100, 1),
        "grounding_faithfulness": round((grounded_count / total) * 100, 1),
        "avg_latency_ms": round(avg_latency, 2)
    }
    return report

if __name__ == "__main__":
    res = run_benchmark()
    print("=== CORRECTIVE RAG (MultiHop-RAG) BENCHMARK ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
