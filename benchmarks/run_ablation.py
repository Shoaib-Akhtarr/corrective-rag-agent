import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.evaluate import run_benchmark


def main():
    report = run_benchmark()
    nli_metrics = json.load(open(os.path.join(ROOT, "benchmarks", "nli_training_metrics.json"), "r", encoding="utf-8"))
    ablation = {
        "pdf_corpus": {
            "documents_indexed": report["documents_indexed"],
            "chunks_indexed": report["chunks_indexed"],
        },
        "pytorch_nli_model": {
            "test_accuracy": nli_metrics["test_accuracy"],
            "test_macro_f1": nli_metrics["test_macro_f1"],
            "total_pairs": nli_metrics["total_pairs"],
        },
        "claim_diff_rag_benchmark": report,
    }
    with open(os.path.join(ROOT, "benchmarks", "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)

    md = (
        "# Temporal & Contradiction-Aware Claim-Diff RAG Benchmark\n\n"
        f"- **Ingested PDFs:** {report['documents_indexed']} multi-page PDFs ({report['chunks_indexed']} chunks)\n"
        f"- **PyTorch NLI Contradiction Model:** Test Accuracy = **{nli_metrics['test_accuracy']}**, Macro-F1 = **{nli_metrics['test_macro_f1']}** (1,500 pairs)\n"
        f"- **Authoritative Doc Recall:** **{report['authoritative_doc_recall_pct']}%**\n"
        f"- **Contradiction Detection Rate:** **{report['contradiction_detection_rate_pct']}%**\n"
        f"- **Distractor Rejection Rate:** **{report['distractor_rejection_rate_pct']}%**\n"
        f"- **Grounding Faithfulness:** **{report['grounding_faithfulness_pct']}%**\n"
    )
    with open(os.path.join(ROOT, "benchmarks", "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
