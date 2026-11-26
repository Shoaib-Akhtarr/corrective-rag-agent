import os
import re
from typing import Any, Dict, List, Tuple, TypedDict
import joblib
import numpy as np
from langgraph.graph import END, StateGraph
from app.claims.contradiction_engine import claim_diff_engine
from app.llm.client import llm_gateway
from app.retrieval.hybrid_index import hybrid_index

_GRADER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts",
    "distractor_grader.joblib",
)
_ML_GRADER = joblib.load(_GRADER_PATH) if os.path.exists(_GRADER_PATH) else None


class CRAGState(TypedDict):
    query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    graded_docs: List[Dict[str, Any]]
    rejected_distractors: List[str]
    claim_diff: Dict[str, Any]
    hop_count: int
    max_hops: int
    revision_count: int
    draft_answer: str
    is_grounded: bool
    faithfulness_score: float
    unverified_numbers: List[str]
    llm_trace: Dict[str, Any]
    audit_trail: List[str]


_STOP_CAPS = {
    "Section", "Article", "Document", "Authority", "Effective", "Date", "The", "This",
    "That", "These", "Any", "All", "For", "Per", "What", "Who", "Does", "Which", "Where",
    "When", "How", "Upon", "Except", "During", "Following", "Under", "Page",
}


def extract_generic_entities(text: str, max_entities: int = 6) -> List[str]:
    """
    Domain-agnostic technical identifier and noun-phrase extractor used as fallback
    when no LLM endpoint is configured. Never relies on hardcoded corpus entities.
    """
    candidates: List[str] = []
    # 1. Hyphenated / alphanumeric / versioned codes (e.g., eu-central-2, LegacyVault-v2, V-900)
    for tok in re.findall(r"\b(?:[a-zA-Z]+(?:-[a-zA-Z0-9]+)*-\d+[a-zA-Z0-9-]*|[A-Z][a-zA-Z0-9]+-[a-zA-Z0-9-]+)\b", text):
        if tok not in candidates:
            candidates.append(tok)
    # 2. Multi-word capitalized noun phrases (e.g., Zurich HSM Quorum, Meridian AI Labs)
    for phrase in re.findall(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+(?:of|for|&)?\s*[A-Z][a-zA-Z0-9]+){1,3})\b", text):
        first = phrase.split()[0]
        if first in _STOP_CAPS:
            continue
        if phrase not in candidates and len(phrase) > 4:
            candidates.append(phrase)
    # 3. Domain acronyms >= 3 uppercase chars (e.g., DORA, HSM, EMA, GDPR)
    for acr in re.findall(r"\b[A-Z]{3,6}\b", text):
        if acr not in {"PDF", "USD", "REGULATORY", "INTERNAL", "MEMO"} and acr not in candidates:
            candidates.append(acr)
    return candidates[:max_entities]


_QUERY_STOP = {
    "what", "which", "when", "where", "who", "whom", "whose", "how", "why",
    "does", "did", "have", "has", "had", "was", "were", "are", "the", "and",
    "for", "that", "this", "with", "from", "into", "across", "earlier",
    "current", "under", "about", "triggers", "issued", "any", "all", "per", "via",
}


def compute_pair_features(query: str, doc: Dict[str, Any]) -> np.ndarray:
    """Computes domain-agnostic pairwise query-document relevance features."""
    q_toks = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    c_toks = set(re.findall(r"[a-z0-9]{3,}", doc.get("content", "").lower()))
    overlap_q = len(q_toks & c_toks) / max(1, len(q_toks))
    jaccard = len(q_toks & c_toks) / max(1, len(q_toks | c_toks))
    dense = float(doc.get("dense_score", 0.0))
    bm25 = min(float(doc.get("bm25_score", 0.0)) / 10.0, 1.5)
    rrf = float(doc.get("rrf_score", 0.0)) * 30.0

    q_ents = extract_generic_entities(query)
    ent_words = {w for e in q_ents for w in re.findall(r"[a-z0-9]{3,}", e.lower())}
    pred_toks = {t for t in q_toks if t not in ent_words and t not in _QUERY_STOP}
    pred_overlap = len(pred_toks & c_toks) / max(1, len(pred_toks)) if pred_toks else overlap_q
    shared_ents = len(set(q_ents) & set(extract_generic_entities(doc.get("content", ""))))
    return np.array([overlap_q, jaccard, dense, bm25, rrf, pred_overlap, min(shared_ents, 3) / 3.0], dtype=np.float32)


def _extract_numeric_tokens(text: str) -> set:
    return set(re.findall(r"(?:\$?\d[\d,]*(?:\.\d+)?%?)", text or ""))


class CorrectiveRAGAgent:
    """
    Temporal & Contradiction-Aware Corrective RAG built on LangGraph StateGraph.
    Pipeline: Qdrant+BM25 Hybrid Retrieve -> Pairwise ML Grader -> Multi-Hop Decomposition
    -> PyTorch NLI Claim-Diff -> Constrained LLM Synthesis & Self-Correction Loop.
    """

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.graph = self._build_graph()

    def grade_documents(self, query: str, docs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        graded: List[Dict[str, Any]] = []
        rejected: List[str] = []
        for doc in docs:
            feats = compute_pair_features(query, doc)
            q_overlap, _, dense_score, _, _, pred_overlap, _ = feats
            # Reject chunks that match zero query predicate words (entity-overlap trap / topical mismatch)
            if pred_overlap == 0.0 and q_overlap < 0.25:
                rejected.append(doc["doc_id"])
                continue
            if _ML_GRADER is not None:
                prob = float(_ML_GRADER.predict_proba(feats.reshape(1, -1))[0][1])
                keep = prob >= 0.50 or (pred_overlap >= 0.20 and dense_score >= 0.25)
            else:
                keep = pred_overlap >= 0.12 or (q_overlap >= 0.18 and dense_score >= 0.22)

            if keep:
                graded.append(doc)
            else:
                rejected.append(doc["doc_id"])
        return graded, rejected

    def decompose_and_hop(self, query: str, current_docs: List[Dict[str, Any]]) -> List[str]:
        context_snip = "\n".join(f"[{d['doc_id']} p.{d['page']}] {d['content'][:280]}" for d in current_docs[:4])
        res = llm_gateway.complete_json(
            system_prompt=(
                "Extract up to 4 concise follow-up search sub-queries or specific entity names needed "
                "for multi-hop verification against conflicting documents. Return JSON: {\"sub_queries\": [\"...\"]}."
            ),
            user_prompt=f"Question: {query}\nRetrieved snippets:\n{context_snip}",
            max_tokens=200,
        )
        if res["ok"] and isinstance(res["data"], dict):
            sqs = [str(s).strip() for s in res["data"].get("sub_queries", []) if str(s).strip()]
            if sqs:
                return sqs[:4]

        blob = query + " " + " ".join(d["content"] for d in current_docs[:4])
        return extract_generic_entities(blob, max_entities=5)

    def _fallback_synthesis(self, claim_diff: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
        auth_claims = claim_diff.get("authoritative_claims", [])
        if not auth_claims and not docs:
            return "Insufficient verified document evidence to answer safely."

        lines = ["### Authoritative Resolution (Post Temporal & Authority Claim-Diff)"]
        if auth_claims:
            for cl in auth_claims[:4]:
                lines.append(
                    f"- **[{cl['doc_id']} p.{cl['page']} | {cl['authority_label']} | {cl['effective_date']}]** {cl['text']}"
                )
        else:
            for d in docs[:2]:
                lines.append(
                    f"- **[{d['doc_id']} p.{d['page']} | {d['authority_label']} | {d['effective_date']}]** {d['content'][:260]}"
                )

        diffs = claim_diff.get("claim_lineage_diff", [])
        if diffs:
            lines.append("\n### Contradictions Resolved (Superseded Claims)")
            for diff in diffs:
                lines.append(
                    f"- **Superseded:** {diff['superseded_source']} -> *\"{diff['superseded_claim'][:120]}...\"* "
                    f"**Replaced by:** {diff['winning_source']} (NLI Contradiction Prob: {diff['nli_contradiction_prob']})"
                )
        return "\n".join(lines)

    def synthesize_grounded_answer(
        self,
        query: str,
        claim_diff: Dict[str, Any],
        docs: List[Dict[str, Any]],
        repair_hint: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        auth_claims = claim_diff.get("authoritative_claims", [])
        diffs = claim_diff.get("claim_lineage_diff", [])
        evidence_lines = [
            f"[{cl['doc_id']} p.{cl['page']} | {cl['authority_label']} | {cl['effective_date']}] {cl['text']}"
            for cl in auth_claims[:5]
        ]
        if not evidence_lines:
            evidence_lines = [
                f"[{d['doc_id']} p.{d['page']} | {d['authority_label']} | {d['effective_date']}] {d['content'][:320]}"
                for d in docs[:4]
            ]
        diff_lines = [
            f"SUPERSEDED ({d['superseded_source']}): {d['superseded_claim']} ==> REPLACED BY ({d['winning_source']}): {d['winning_claim']}"
            for d in diffs
        ]

        sys_prompt = (
            "You are an enterprise Forward-Deployed AI Reliability Assistant. Answer strictly using ONLY the verified "
            "authoritative claims below. Cite every factual claim using exact bracket format [doc_id p.X]. "
            "Explicitly mention any superseded claim if present. NEVER invent numbers, dates, or entities not in the claims."
        )
        user_prompt = (
            f"Question: {query}\n\n"
            f"Authoritative Surviving Claims:\n" + "\n".join(evidence_lines) + "\n\n"
            f"Resolved Contradictions:\n" + ("\n".join(diff_lines) if diff_lines else "None detected.")
        )
        if repair_hint:
            user_prompt += f"\n\nCORRECTION REQUIRED: {repair_hint}"

        llm_res = llm_gateway.complete(sys_prompt, user_prompt, temperature=0.05, max_tokens=550)
        if llm_res["ok"] and llm_res["text"].strip():
            return llm_res["text"].strip(), llm_res["trace"]

        return self._fallback_synthesis(claim_diff, docs), llm_res["trace"]

    def _verify_grounding(self, answer: str, docs: List[Dict[str, Any]]) -> Tuple[bool, float, List[str]]:
        clean_norm = (answer or "").replace("\u202f", " ").replace("\xa0", " ")
        valid_ids = {d["doc_id"] for d in docs}
        cited_ids = re.findall(r"[\[【]([^\s|\]]+)", clean_norm)
        citations_valid = len(cited_ids) > 0 and all(cid in valid_ids for cid in cited_ids)

        body_no_cites = re.sub(r"[\[【][^\]]+[\]】]", " ", clean_norm)
        # Exclude system-emitted Claim-Diff provenance tags & confidence scores before checking LLM hallucination
        body_no_cites = re.sub(r"\(NLI Contradiction Prob:\s*[\d.]+\)", " ", body_no_cites)
        body_no_cites = re.sub(r"\b\d{2}_[A-Za-z0-9_]+\.(?:pdf|md|txt)\b", " ", body_no_cites)
        body_no_cites = re.sub(r"\bp\.\d+\b", " ", body_no_cites)
        body_no_cites = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", body_no_cites)
        source_nums = set()
        for d in docs:
            source_nums.update(_extract_numeric_tokens(d.get("content", "")))
        ans_nums = _extract_numeric_tokens(body_no_cites)
        unverified_nums = sorted([n for n in ans_nums if n not in source_nums and n.lstrip("$%") not in {"1", "2", "3", "4"}])

        score = 1.0 if (citations_valid and not unverified_nums) else (0.5 if citations_valid else 0.0)
        return (citations_valid and len(unverified_nums) == 0), score, unverified_nums

    def _build_graph(self):
        wf = StateGraph(CRAGState)

        def node_retrieve(state: CRAGState) -> Dict[str, Any]:
            initial = hybrid_index.search(state["query"], top_k=6)
            graded, rejected = self.grade_documents(state["query"], initial)
            seen = {d["chunk_id"] for d in graded}
            msg = (
                f"Hop 1 (Qdrant {hybrid_index.qdrant_mode}/{hybrid_index.embedder.backend_name} + BM25 RRF): "
                f"kept {len(graded)}, rejected={sorted(set(rejected))}"
            )
            return {
                "retrieved_docs": initial,
                "graded_docs": graded,
                "rejected_distractors": sorted(set(rejected)),
                "hop_count": 1,
                "audit_trail": state["audit_trail"] + [msg],
            }

        def node_multihop(state: CRAGState) -> Dict[str, Any]:
            all_graded = list(state["graded_docs"])
            seen = {d["chunk_id"] for d in all_graded}
            rejected = set(state["rejected_distractors"])
            sub_queries = self.decompose_and_hop(state["query"], all_graded)
            for sq in sub_queries:
                hop_docs = hybrid_index.search(sq, top_k=3)
                g_hop, r_hop = self.grade_documents(state["query"], hop_docs)
                for d in g_hop:
                    if d["chunk_id"] not in seen:
                        all_graded.append(d)
                        seen.add(d["chunk_id"])
                rejected.update(r_hop)
            hop = state["hop_count"] + (1 if sub_queries else 0)
            msg = f"Hop {hop} (Entity/Sub-Query Expansion {sub_queries}): total verified chunks={len(all_graded)}"
            return {
                "sub_queries": sub_queries,
                "graded_docs": all_graded,
                "rejected_distractors": sorted(rejected),
                "hop_count": hop,
                "audit_trail": state["audit_trail"] + [msg],
            }

        def node_claim_diff(state: CRAGState) -> Dict[str, Any]:
            diff = claim_diff_engine.resolve_conflicts(state["query"], state["graded_docs"])
            msg = (
                f"Claim-Diff Engine (PyTorch NLI): analyzed {diff['claims_analyzed']} claims, "
                f"resolved {diff['contradictions_detected']} temporal/authority contradictions."
            )
            return {"claim_diff": diff, "audit_trail": state["audit_trail"] + [msg]}

        def node_synthesize_verify(state: CRAGState) -> Dict[str, Any]:
            rev = state.get("revision_count", 0)
            hint = ""
            if rev > 0 and state.get("unverified_numbers"):
                hint = (
                    f"Previous draft contained unverified numbers {state['unverified_numbers']} or missing [doc_id p.X] citations. "
                    "Remove unverified numbers and cite strictly [doc_id p.X]."
                )
            draft, trace = self.synthesize_grounded_answer(
                state["query"], state["claim_diff"], state["graded_docs"], repair_hint=hint
            )
            ok, score, bad_nums = self._verify_grounding(draft, state["graded_docs"])
            # Deterministic safe fallback if LLM failed verification on retry
            if not ok and rev >= 1:
                draft = self._fallback_synthesis(state["claim_diff"], state["graded_docs"])
                ok, score, bad_nums = self._verify_grounding(draft, state["graded_docs"])
                trace["fallback_used"] = True
            msg = (
                f"Synthesis & Grounding Gate (rev={rev}, provider={trace['provider']}, model={trace['model']}): "
                f"grounded={ok}, faithfulness={score}, tokens={trace['total_tokens']}"
            )
            return {
                "draft_answer": draft,
                "is_grounded": ok,
                "faithfulness_score": score,
                "unverified_numbers": bad_nums,
                "llm_trace": trace,
                "revision_count": rev + 1,
                "audit_trail": state["audit_trail"] + [msg],
            }

        def should_retry(state: CRAGState) -> str:
            if not state["is_grounded"] and state["revision_count"] < 2 and not state["llm_trace"].get("fallback_used"):
                return "synthesize"
            return END

        wf.add_node("retrieve", node_retrieve)
        wf.add_node("multihop", node_multihop)
        wf.add_node("claim_diff", node_claim_diff)
        wf.add_node("synthesize", node_synthesize_verify)
        wf.set_entry_point("retrieve")
        wf.add_edge("retrieve", "multihop")
        wf.add_edge("multihop", "claim_diff")
        wf.add_edge("claim_diff", "synthesize")
        wf.add_conditional_edges("synthesize", should_retry, {"synthesize": "synthesize", END: END})
        return wf.compile()

    def run(self, query: str) -> CRAGState:
        init_state: CRAGState = {
            "query": query,
            "sub_queries": [],
            "retrieved_docs": [],
            "graded_docs": [],
            "rejected_distractors": [],
            "claim_diff": {},
            "hop_count": 0,
            "max_hops": self.max_hops,
            "revision_count": 0,
            "draft_answer": "",
            "is_grounded": False,
            "faithfulness_score": 0.0,
            "unverified_numbers": [],
            "llm_trace": {},
            "audit_trail": [f"LangGraph CRAG started across {len(hybrid_index.chunks)} chunks."],
        }
        return self.graph.invoke(init_state)


crag_agent = CorrectiveRAGAgent()
