import json
import os
import random
import re
import sys
import httpx
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)

NLI_LABELS = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]


class ContradictionNLIMLP(nn.Module):
    """
    Siamese + Cross-Interaction Neural NLI Classifier.
    Input: [|u - v|, u * v, 6 structural conflict features] -> [ENTAILMENT, CONTRADICTION, NEUTRAL].
    """

    def __init__(self, tfidf_dim: int, struct_dim: int = 6):
        super().__init__()
        in_dim = tfidf_dim * 2 + struct_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 320),
            nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(160, 3),
        )

    def forward(self, x):
        return self.net(x)


def extract_structural_features(a: str, b: str) -> np.ndarray:
    nums_a = set(re.findall(r"\d+(?:\.\d+)?", a))
    nums_b = set(re.findall(r"\d+(?:\.\d+)?", b))
    num_mismatch = 1.0 if (nums_a and nums_b and nums_a != nums_b) else 0.0
    num_match = 1.0 if (nums_a and nums_b and nums_a == nums_b) else 0.0

    neg_words = {"not", "never", "zero", "no", "excluded", "optional", "cannot", "prohibited", "without", "neither", "nor"}
    toks_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    toks_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    neg_a = len(toks_a & neg_words) % 2
    neg_b = len(toks_b & neg_words) % 2
    polarity_flip = 1.0 if neg_a != neg_b else 0.0

    jaccard = len(toks_a & toks_b) / max(1, len(toks_a | toks_b))
    amend_cue = 1.0 if any(w in (a + " " + b).lower() for w in ["supersed", "replac", "amend", "replaces", "except", "only"]) else 0.0
    content_overlap = len((toks_a & toks_b) - neg_words) / max(1, min(len(toks_a), len(toks_b)))
    return np.array([num_mismatch, num_match, polarity_flip, jaccard, amend_cue, content_overlap], dtype=np.float32)


def _extract_evidence_window(premise: str, hypothesis: str) -> str:
    """Selects top-2 premise sentences overlapping hypothesis to align contract length with sentence claims."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?;\n])\s+", premise) if len(s.strip()) > 20]
    if len(sents) <= 2:
        return premise[:600]
    h_toks = set(re.findall(r"[a-z0-9]{3,}", hypothesis.lower()))
    scored = []
    for s in sents:
        s_toks = set(re.findall(r"[a-z0-9]{3,}", s.lower()))
        scored.append((len(h_toks & s_toks), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(s for _, s in scored[:2])[:650]


def load_real_nli_datasets():
    """
    Loads real open NLI benchmarks with zero synthetic template generation:
      1. kiddothe2b/contract-nli (contractnli_a): label map {0:1 (CONTRADICTION), 1:0 (ENTAILMENT), 2:2 (NEUTRAL)}
      2. nyu-mll/multi_nli (validation_matched grouped by promptID): {0:0 (ENTAILMENT), 1:2 (NEUTRAL), 2:1 (CONTRADICTION)}
    Strict train / val / unseen test separation guarantees honest generalization metrics.
    """
    cnli_info = httpx.get("https://huggingface.co/api/datasets/kiddothe2b/contract-nli/parquet", timeout=15).json()["contractnli_a"]
    cnli_map = {0: 1, 1: 0, 2: 2}

    def _convert_cnli(df: pd.DataFrame):
        out = []
        for _, row in df.iterrows():
            p = _extract_evidence_window(str(row["premise"]), str(row["hypothesis"]))
            h = str(row["hypothesis"])
            out.append((p, h, cnli_map[int(row["label"])]))
        return out

    cnli_train = _convert_cnli(pd.read_parquet(cnli_info["train"][0]))
    cnli_val = _convert_cnli(pd.read_parquet(cnli_info["validation"][0]))
    cnli_test = _convert_cnli(pd.read_parquet(cnli_info["test"][0]))

    mnli_info = httpx.get("https://huggingface.co/api/datasets/nyu-mll/multi_nli/parquet", timeout=15).json()["default"]
    df_mnli = pd.read_parquet(mnli_info["validation_matched"][0])
    df_mnli = df_mnli[df_mnli["label"].isin([0, 1, 2])].reset_index(drop=True)
    mnli_map = {0: 0, 1: 2, 2: 1}

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trval_idx, te_idx = next(gss1.split(df_mnli, df_mnli["label"], groups=df_mnli["promptID"]))
    df_trval = df_mnli.iloc[trval_idx].reset_index(drop=True)
    df_te = df_mnli.iloc[te_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, va_idx = next(gss2.split(df_trval, df_trval["label"], groups=df_trval["promptID"]))

    def _convert_mnli(sub: pd.DataFrame):
        return [(str(r["premise"]), str(r["hypothesis"]), mnli_map[int(r["label"])]) for _, r in sub.iterrows()]

    train_pairs = cnli_train + _convert_mnli(df_trval.iloc[tr_idx])
    val_pairs = cnli_val + _convert_mnli(df_trval.iloc[va_idx])
    test_pairs = cnli_test + _convert_mnli(df_te)
    cnli_test_len = len(cnli_test)
    return train_pairs, val_pairs, test_pairs, cnli_test_len


def featurize_pairs(pairs, vectorizer: TfidfVectorizer) -> np.ndarray:
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
    random.seed(42)
    train_pairs, val_pairs, test_pairs, cnli_test_len = load_real_nli_datasets()

    all_train_texts = [p[0] for p in train_pairs] + [p[1] for p in train_pairs]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=768, sublinear_tf=True, min_df=2)
    vectorizer.fit(all_train_texts)

    X_train = featurize_pairs(train_pairs, vectorizer)
    y_train = np.array([p[2] for p in train_pairs], dtype=np.int64)
    X_val = featurize_pairs(val_pairs, vectorizer)
    y_val = np.array([p[2] for p in val_pairs], dtype=np.int64)
    X_test = featurize_pairs(test_pairs, vectorizer)
    y_test = np.array([p[2] for p in test_pairs], dtype=np.int64)

    # Inverse-frequency class weights for imbalanced legal/NLI splits
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=128,
        shuffle=True,
    )

    tfidf_dim = len(vectorizer.get_feature_names_out())
    model = ContradictionNLIMLP(tfidf_dim=tfidf_dim, struct_dim=6)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    best_val_f1 = 0.0
    best_state = None
    loss_history = []

    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by, weight=cls_w)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = torch.argmax(model(torch.from_numpy(X_val)), dim=-1).numpy()
            vf1 = float(f1_score(y_val, vp, average="macro"))
        if vf1 >= best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "train_loss": round(total_loss / len(X_train), 4), "val_macro_f1": round(vf1, 4)})

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(torch.from_numpy(X_test)), dim=-1).numpy()

    test_acc = round(float(accuracy_score(y_test, test_preds)), 4)
    test_f1 = round(float(f1_score(y_test, test_preds, average="macro")), 4)
    cnli_only_acc = round(float(accuracy_score(y_test[:cnli_test_len], test_preds[:cnli_test_len])), 4)
    cnli_only_f1 = round(float(f1_score(y_test[:cnli_test_len], test_preds[:cnli_test_len], average="macro")), 4)

    joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, "nli_vectorizer.joblib"))
    torch.save(
        {"state_dict": best_state, "tfidf_dim": tfidf_dim, "labels": NLI_LABELS},
        os.path.join(ARTIFACTS_DIR, "contradiction_nli.pt"),
    )

    metrics = {
        "model": "PyTorch Siamese+Cross-Interaction NLI MLP (320->160->3, AdamW + CosineAnnealingLR)",
        "datasets": ["kiddothe2b/contract-nli (contractnli_a)", "nyu-mll/multi_nli (promptID-disjoint split)"],
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "epochs": 25,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "contractnli_test_accuracy": cnli_only_acc,
        "contractnli_test_macro_f1": cnli_only_f1,
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "classification_report": classification_report(y_test, test_preds, target_names=NLI_LABELS, output_dict=True),
        "loss_history": loss_history,
    }
    with open(os.path.join(BENCH_DIR, "nli_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved contradiction_nli.pt | Test N={len(test_pairs)} Acc={test_acc}, Macro-F1={test_f1} (ContractNLI Acc={cnli_only_acc})")


if __name__ == "__main__":
    main()
