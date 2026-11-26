import json
import os
import random
import re
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)

NLI_LABELS = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]


class ContradictionNLIMLP(nn.Module):
    """
    Siamese + Cross-Interaction Neural NLI Classifier.
    Takes [|u - v|, u * v, structural_conflict_features] and predicts
    ENTAILMENT (0), CONTRADICTION (1), or NEUTRAL (2).
    """
    def __init__(self, tfidf_dim: int, struct_dim: int = 6):
        super().__init__()
        in_dim = tfidf_dim * 2 + struct_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


def extract_structural_features(a: str, b: str) -> np.ndarray:
    nums_a = set(re.findall(r"\d+(?:\.\d+)?", a))
    nums_b = set(re.findall(r"\d+(?:\.\d+)?", b))
    num_mismatch = 1.0 if (nums_a and nums_b and nums_a != nums_b) else 0.0
    num_match = 1.0 if (nums_a and nums_b and nums_a == nums_b) else 0.0

    neg_words = {"not", "never", "zero", "no", "excluded", "optional", "cannot", "prohibited"}
    toks_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    toks_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    neg_a = len(toks_a & neg_words) % 2
    neg_b = len(toks_b & neg_words) % 2
    polarity_flip = 1.0 if neg_a != neg_b else 0.0

    jaccard = len(toks_a & toks_b) / max(1, len(toks_a | toks_b))
    supersede_cue = 1.0 if any(w in b.lower() for w in ["supersed", "replac", "amend", "replaces"]) else 0.0
    entity_overlap = len((toks_a & toks_b) - neg_words) / max(1, min(len(toks_a), len(toks_b)))
    return np.array([num_mismatch, num_match, polarity_flip, jaccard, supersede_cue, entity_overlap], dtype=np.float32)


def build_nli_corpus(n_per_class: int = 500):
    random.seed(42)
    subjects = [
        "HelixDB", "LegacyVault-v2", "Zurich HSM Quorum", "Frankfurt Ops Committee",
        "Apex Cloud eu-central-2", "Meridian AI Labs SLA", "Dr. Elena Rostova",
        "BioNova VALOR-2", "Vertex V-900", "Aetheris Genomics"
    ]
    metrics = [
        ("default vector storage engine", ["LegacyVault-v2", "HelixDB", "CoreVector-v1"]),
        ("p99 query latency", ["45.0 milliseconds", "14.2 milliseconds", "11.4 milliseconds", "28.5 milliseconds"]),
        ("encryption key rotation authority", ["Frankfurt Ops Committee", "Zurich HSM Quorum", "Geneva Security Council"]),
        ("key rotation cycle", ["90 days", "30 days", "60 days"]),
        ("SLA penalty threshold", ["25,000 units", "10,000 units", "15,000 units"]),
        ("audit log threshold", ["20.0ms", "50.0ms", "15.0ms"]),
        ("non-EU semiconductor board seats", ["zero board seats", "two board seats"]),
    ]
    pairs = []
    # Contradictions (numeric, entity replacement, polarity flip, supersession)
    for _ in range(n_per_class):
        subj = random.choice(subjects)
        pred, vals = random.choice(metrics)
        v1, v2 = random.sample(vals, 2)
        c1 = f"{subj} {pred} is set to {v1} under policy."
        if random.random() < 0.5:
            c2 = f"Superseding prior policy, {subj} {pred} is updated to {v2}."
        else:
            c2 = f"{subj} {pred} is not {v1}; it requires {v2}."
        pairs.append((c1, c2, 1))

    # Entailments (paraphrase / consistent confirmation)
    for _ in range(n_per_class):
        subj = random.choice(subjects)
        pred, vals = random.choice(metrics)
        v = random.choice(vals)
        c1 = f"{subj} {pred} is {v} across production clusters."
        c2 = f"Verified audit confirms {subj} maintains {pred} at {v}."
        pairs.append((c1, c2, 0))

    # Neutrals (unrelated subjects / distractor topics)
    unrelated = [
        "Apex Sports Apparel reported $320M Q4 winter running gear sales.",
        "Meridian Maritime Group christened two LNG container ships in Rotterdam.",
        "Statkraft Nordic residential electricity tariffs rose 3% in Oslo.",
        "Alpine Ski Resorts recorded heavy February snowfall in Zermatt.",
        "Zurich Watchmakers Guild hosted 40 mechanical tourbillon pieces.",
    ]
    for _ in range(n_per_class):
        subj = random.choice(subjects)
        pred, vals = random.choice(metrics)
        v = random.choice(vals)
        c1 = f"{subj} {pred} is {v} in eu-central-2."
        c2 = random.choice(unrelated)
        pairs.append((c1, c2, 2))

    random.shuffle(pairs)
    return pairs


def featurize_pairs(pairs, vectorizer: TfidfVectorizer):
    a_texts = [p[0] for p in pairs]
    b_texts = [p[1] for p in pairs]
    u = vectorizer.transform(a_texts).toarray().astype(np.float32)
    v = vectorizer.transform(b_texts).toarray().astype(np.float32)
    diff = np.abs(u - v)
    prod = u * v
    struct = np.stack([extract_structural_features(a, b) for a, b in zip(a_texts, b_texts)], axis=0)
    return np.concatenate([diff, prod, struct], axis=1)


def main():
    torch.manual_seed(42)
    pairs = build_nli_corpus(n_per_class=500)  # 1,500 labeled pairs
    all_texts = [p[0] for p in pairs] + [p[1] for p in pairs]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=384, sublinear_tf=True)
    vectorizer.fit(all_texts)

    X = featurize_pairs(pairs, vectorizer)
    y = np.array([p[2] for p in pairs], dtype=np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )

    tfidf_dim = len(vectorizer.get_feature_names_out())
    model = ContradictionNLIMLP(tfidf_dim=tfidf_dim, struct_dim=6)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    best_val_acc = 0.0
    best_state = None
    loss_history = []

    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(torch.from_numpy(X_val)), dim=-1).numpy()
            val_acc = float(accuracy_score(y_val, val_preds))
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 15 == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "train_loss": round(total_loss / len(X_train), 4), "val_acc": round(val_acc, 4)})

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(torch.from_numpy(X_test)), dim=-1).numpy()

    test_acc = round(float(accuracy_score(y_test, test_preds)), 4)
    test_f1 = round(float(f1_score(y_test, test_preds, average="macro")), 4)

    joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, "nli_vectorizer.joblib"))
    torch.save({"state_dict": best_state, "tfidf_dim": tfidf_dim, "labels": NLI_LABELS}, os.path.join(ARTIFACTS_DIR, "contradiction_nli.pt"))

    metrics = {
        "model": "PyTorch Siamese+Cross-Interaction NLI MLP (256->128->3, AdamW + CosineAnnealingLR)",
        "total_pairs": len(pairs),
        "train_val_test_split": [len(X_train), len(X_val), len(X_test)],
        "epochs": 60,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "classification_report": classification_report(y_test, test_preds, target_names=NLI_LABELS, output_dict=True),
        "loss_history": loss_history,
    }
    with open(os.path.join(BENCH_DIR, "nli_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved contradiction_nli.pt | Test Acc={test_acc}, Macro-F1={test_f1}")


if __name__ == "__main__":
    main()
