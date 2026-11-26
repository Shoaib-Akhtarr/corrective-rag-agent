import json
import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.graph.crag_workflow import compute_pair_features
from app.retrieval.hybrid_index import HybridDocumentIndex

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


def build_pairwise_relevance_dataset():
    """
    Constructs domain-agnostic pairwise query-chunk relevance examples grouped by doc_id
    so cross-validation strictly evaluates unseen documents without reading filenames.
    """
    idx = HybridDocumentIndex(qdrant_path=":memory:")
    chunks = idx.chunks
    queries = [
        ("vector storage engine HelixDB LegacyVault latency", {"01_Apex_Sovereign_Cloud_Policy_2025_Q1", "02_NovaVector_Acquisition_Addendum_2025_Q3", "06_BioNova_VALOR2_Clinical_Telemetry_2026_Q1"}),
        ("HSM encryption key rotation Zurich Quorum EMA DORA", {"01_Apex_Sovereign_Cloud_Policy_2025_Q1", "03_Zurich_HSM_Board_Resolution_2026_Q1", "04_EMA_DORA_Clinical_Telemetry_Ruling_2026_Q1"}),
        ("GPU unit threshold SLA penalty credit Meridian Vertex V-900", {"01_Apex_Sovereign_Cloud_Policy_2025_Q1", "05_Meridian_AI_SLA_Amendment_2026_Q1"}),
        ("Luminos Labs patents Apex Cloud acquisition legal liability", {"02_NovaVector_Acquisition_Addendum_2025_Q3"}),
        ("clinical trial biometric audit indexing sub-20.0ms", {"04_EMA_DORA_Clinical_Telemetry_Ruling_2026_Q1", "06_BioNova_VALOR2_Clinical_Telemetry_2026_Q1"}),
        ("winter running jackets retail North America revenue", {"07_DISTRACTOR_Apex_Sports_Apparel_Earnings"}),
        ("LNG container vessels Rotterdam maritime shipping", {"08_DISTRACTOR_Meridian_Maritime_Fleet_Update"}),
    ]

    X_rows, y_rows, groups = [], [], []
    for q_text, relevant_docs in queries:
        hits = {c["chunk_id"]: c for c in idx.search(q_text, top_k=len(chunks))}
        for c in chunks:
            doc_scored = hits.get(c["chunk_id"], c)
            feats = compute_pair_features(q_text, doc_scored)
            X_rows.append(feats)
            y_rows.append(1 if c["doc_id"] in relevant_docs else 0)
            groups.append(c["doc_id"])

    return np.stack(X_rows), np.array(y_rows, dtype=np.int64), np.array(groups)


def main():
    X, y, groups = build_pairwise_relevance_dataset()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42)),
    ])

    gkf = GroupKFold(n_splits=4)
    cv_preds = cross_val_predict(pipeline, X, y, cv=gkf, groups=groups)
    cv_acc = round(float(accuracy_score(y, cv_preds)), 4)
    cv_f1 = round(float(f1_score(y, cv_preds)), 4)

    pipeline.fit(X, y)
    model_path = os.path.join(ARTIFACTS_DIR, "distractor_grader.joblib")
    joblib.dump(pipeline, model_path)

    metrics = {
        "model": "StandardScaler + LogisticRegression Pairwise Query-Chunk Relevance Grader (7 features)",
        "total_query_chunk_pairs": int(len(y)),
        "relevant_pairs": int(y.sum()),
        "distractor_pairs": int((1 - y).sum()),
        "group_kfold_4fold_accuracy": cv_acc,
        "group_kfold_4fold_f1": cv_f1,
        "confusion_matrix": confusion_matrix(y, cv_preds).tolist(),
        "classification_report": classification_report(y, cv_preds, target_names=["irrelevant", "relevant"], output_dict=True),
        "artifact_path": "artifacts/distractor_grader.joblib",
    }
    with open(os.path.join(BENCH_DIR, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {model_path} | Unseen-Doc GroupKFold Acc={cv_acc}, F1={cv_f1}")


if __name__ == "__main__":
    main()
