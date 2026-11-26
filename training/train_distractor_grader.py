import json
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(ROOT, "data", "multihop_corpus.json")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    texts = []
    labels = []  # 1 = valid signal doc, 0 = adversarial distractor
    for doc in corpus["documents"]:
        clean_title = doc["title"].replace("DISTRACTOR: ", "")
        texts.append(f"{clean_title}. {doc['category']}. {doc['content']}")
        is_distractor = 1 if ("distractor:" in doc["title"].lower() or "unrelated to" in doc["content"].lower()) else 0
        labels.append(1 - is_distractor)

    y = np.array(labels)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english")),
        ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipeline, texts, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(pipeline, texts, y, cv=cv, scoring="f1")

    pipeline.fit(texts, y)
    preds = pipeline.predict(texts)

    model_path = os.path.join(ARTIFACTS_DIR, "distractor_grader.joblib")
    joblib.dump(pipeline, model_path)

    metrics = {
        "model": "TfidfVectorizer + LogisticRegression (balanced)",
        "total_samples": len(texts),
        "signal_docs": int(y.sum()),
        "distractor_docs": int((1 - y).sum()),
        "cv_5fold_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_5fold_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_5fold_f1_mean": round(float(cv_f1.mean()), 4),
        "train_accuracy": round(float(accuracy_score(y, preds)), 4),
        "train_f1": round(float(f1_score(y, preds)), 4),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds, target_names=["distractor", "signal"], output_dict=True),
        "artifact_path": "artifacts/distractor_grader.joblib",
    }
    with open(os.path.join(BENCH_DIR, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {model_path} | 5-Fold CV Acc={metrics['cv_5fold_accuracy_mean']:.4f}, F1={metrics['cv_5fold_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
