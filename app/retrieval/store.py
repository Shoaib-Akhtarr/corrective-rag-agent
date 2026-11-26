import json
import os
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class MultiHopVectorStore:
    """
    Offline MultiHop-RAG retrieval engine utilizing BM25Okapi sparse indexing
    and semantic keyword matching without requiring external internet search.
    """
    def __init__(self, corpus_path: str = None):
        if corpus_path is None:
            corpus_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "multihop_corpus.json"
            )
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.documents: List[Dict[str, Any]] = self.raw_data["documents"]
        self.benchmark_queries: List[Dict[str, Any]] = self.raw_data["benchmark_queries"]

        self.tokenized_corpus = [self._tokenize(d["title"] + " " + d["content"]) for d in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for doc, score in ranked:
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"],
                "retrieval_score": round(float(score), 4)
            })
        return results

store = MultiHopVectorStore()
