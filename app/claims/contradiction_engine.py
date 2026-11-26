import os
import re
from typing import List, Dict, Any
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from training.train_contradiction_nli import ContradictionNLIMLP, extract_structural_features, NLI_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


class ClaimDiffEngine:
    """
    Temporal & Contradiction-Aware Claim Resolution Engine.
    Runs trained PyTorch Siamese NLI across retrieved PDF claims, detects
    temporal/numeric/authority conflicts, and emits a Claim Lineage Diff.
    """
    def __init__(self):
        vec_path = os.path.join(ARTIFACTS_DIR, "nli_vectorizer.joblib")
        ckpt_path = os.path.join(ARTIFACTS_DIR, "contradiction_nli.pt")
        self.vectorizer = joblib.load(vec_path) if os.path.exists(vec_path) else None
        self.model = None
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model = ContradictionNLIMLP(tfidf_dim=ckpt["tfidf_dim"], struct_dim=6)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

    def predict_pair(self, claim_a: str, claim_b: str) -> Dict[str, Any]:
        if self.model is None or self.vectorizer is None:
            return {"label": "NEUTRAL", "contradiction_prob": 0.0}
        u = self.vectorizer.transform([claim_a]).toarray().astype(np.float32)
        v = self.vectorizer.transform([claim_b]).toarray().astype(np.float32)
        diff = np.abs(u - v)
        prod = u * v
        struct = extract_structural_features(claim_a, claim_b).reshape(1, -1)
        x = torch.from_numpy(np.concatenate([diff, prod, struct], axis=1))
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1)[0]
            idx = int(torch.argmax(probs).item())
        return {
            "label": NLI_LABELS[idx],
            "entailment_prob": round(float(probs[0].item()), 4),
            "contradiction_prob": round(float(probs[1].item()), 4),
            "neutral_prob": round(float(probs[2].item()), 4),
        }

    def extract_claims(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        claims = []
        for c in chunks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c["content"]) if len(s.strip()) > 35]
            for s in sentences[:4]:
                if s.startswith("Document Authority:") or s.startswith("Section ") or s.startswith("Article "):
                    continue
                claims.append({
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "page": c["page"],
                    "effective_date": c["effective_date"],
                    "authority_label": c["authority_label"],
                    "authority_tier": c["authority_tier"],
                    "text": s,
                })
        return claims

    def resolve_conflicts(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        claims = self.extract_claims(chunks)
        q_tokens = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower())}

        # Keep claims relevant to the query
        scored = []
        for cl in claims:
            c_tokens = set(re.findall(r"[a-z0-9]{3,}", cl["text"].lower()))
            overlap = len(q_tokens & c_tokens)
            if overlap > 0:
                scored.append((overlap, cl))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_claims = [c for _, c in scored[:10]] if scored else claims[:6]

        contradictions = []
        superseded_texts = set()

        for i in range(len(top_claims)):
            for j in range(i + 1, len(top_claims)):
                ca, cb = top_claims[i], top_claims[j]
                if ca["doc_id"] == cb["doc_id"]:
                    continue
                ta = set(re.findall(r"[a-z0-9]{4,}", ca["text"].lower()))
                tb = set(re.findall(r"[a-z0-9]{4,}", cb["text"].lower()))
                shared = len(ta & tb)
                if shared < 2:
                    continue

                nli = self.predict_pair(ca["text"], cb["text"])
                # Also flag explicit supersession / numeric-date conflict
                nums_a = set(re.findall(r"\d+(?:\.\d+)?", ca["text"]))
                nums_b = set(re.findall(r"\d+(?:\.\d+)?", cb["text"]))
                has_supersede_cue = any(w in (ca["text"] + " " + cb["text"]).lower() for w in ["supersed", "replac", "amend", "instead"])
                is_conflict = (
                    nli["label"] == "CONTRADICTION"
                    or (nli["contradiction_prob"] >= 0.40 and nums_a != nums_b)
                    or (has_supersede_cue and shared >= 3)
                )
                if is_conflict:
                    rank_a = (ca["authority_tier"], ca["effective_date"])
                    rank_b = (cb["authority_tier"], cb["effective_date"])
                    winner, loser = (ca, cb) if rank_a >= rank_b else (cb, ca)
                    superseded_texts.add(loser["text"])
                    contradictions.append({
                        "winning_claim": winner["text"],
                        "winning_source": f"{winner['filename']} p.{winner['page']} ({winner['authority_label']}, {winner['effective_date']})",
                        "superseded_claim": loser["text"],
                        "superseded_source": f"{loser['filename']} p.{loser['page']} ({loser['authority_label']}, {loser['effective_date']})",
                        "nli_contradiction_prob": nli["contradiction_prob"],
                        "resolution_rule": f"Authority tier ({winner['authority_tier']} vs {loser['authority_tier']}) + Effective Date ({winner['effective_date']} vs {loser['effective_date']})",
                    })

        active_claims = [c for c in top_claims if c["text"] not in superseded_texts]
        active_claims.sort(key=lambda c: (c["authority_tier"], c["effective_date"]), reverse=True)

        return {
            "claims_analyzed": len(top_claims),
            "contradictions_detected": len(contradictions),
            "claim_lineage_diff": contradictions,
            "authoritative_claims": active_claims[:5],
        }


claim_diff_engine = ClaimDiffEngine()
