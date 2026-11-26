import os
import re
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from app.ingestion.doc_loader import load_directory, parse_document_bytes


class HybridDocumentIndex:
    """
    Hybrid BM25 + TF-IDF Cosine + Reciprocal Rank Fusion (RRF) index
    over real ingested PDF / Markdown / TXT document chunks.
    """
    def __init__(self, docs_dir: str = None):
        if docs_dir is None:
            docs_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "documents",
            )
        self.docs_dir = docs_dir
        self.chunks: List[Dict[str, Any]] = []
        self.bm25 = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.reload()

    def reload(self):
        self.chunks = load_directory(self.docs_dir)
        self._rebuild_index()

    def ingest_bytes(self, filename: str, raw_bytes: bytes) -> int:
        new_chunks = parse_document_bytes(filename, raw_bytes)
        if not new_chunks:
            return 0
        # Replace existing chunks from same filename if re-uploaded
        self.chunks = [c for c in self.chunks if c["filename"] != filename] + new_chunks
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self):
        if not self.chunks:
            self.bm25 = None
            self.vectorizer = None
            self.tfidf_matrix = None
            return
        tokenized = [re.findall(r"[a-z0-9.-]+", c["content"].lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform([c["content"] for c in self.chunks]).toarray()

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if not self.chunks or self.bm25 is None:
            return []
        tokens = re.findall(r"[a-z0-9.-]+", query.lower())
        bm25_scores = self.bm25.get_scores(tokens)
        q_vec = self.vectorizer.transform([query]).toarray()[0]
        dense_scores = self.tfidf_matrix @ q_vec

        bm25_rank = np.argsort(bm25_scores)[::-1]
        dense_rank = np.argsort(dense_scores)[::-1]

        rrf_k = 60.0
        rrf_scores = np.zeros(len(self.chunks), dtype=np.float64)
        for rank, idx in enumerate(bm25_rank):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_rank):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = dict(self.chunks[idx])
            item["rrf_score"] = round(float(rrf_scores[idx]), 5)
            item["bm25_score"] = round(float(bm25_scores[idx]), 4)
            item["dense_score"] = round(float(dense_scores[idx]), 4)
            results.append(item)
        return results


hybrid_index = HybridDocumentIndex()
