import os
import re
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.ingestion.doc_loader import load_directory, parse_document_bytes


_EMBEDDING_DIM = 384


class DenseEmbedder:
    """
    Loads local SentenceTransformer (all-MiniLM-L6-v2) when cached/available,
    or falls back to an L2-normalized 384-d TF-IDF+SVD semantic projection.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.st_model = None
        self.backend_name = "svd-384-fallback"
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=4096, sublinear_tf=True)
        self.svd: Optional[TruncatedSVD] = None
        self._try_load_st()

    def _try_load_st(self) -> None:
        if os.getenv("DISABLE_ST_EMBEDDER", "0") == "1":
            return
        try:
            from sentence_transformers import SentenceTransformer
            local_only = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"
            self.st_model = SentenceTransformer(self.model_name, local_files_only=local_only)
            self.backend_name = self.model_name
        except Exception:
            self.st_model = None

    def fit(self, texts: List[str]) -> None:
        if self.st_model is not None or not texts:
            return
        mat = self.tfidf.fit_transform(texts)
        n_comp = min(_EMBEDDING_DIM, max(1, min(mat.shape) - 1))
        if n_comp >= 2:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.svd.fit(mat)
        else:
            self.svd = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        if self.st_model is not None:
            vecs = self.st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)

        mat = self.tfidf.transform(texts)
        if self.svd is not None:
            proj = self.svd.transform(mat)
        else:
            proj = mat.toarray()
        out = np.zeros((len(texts), _EMBEDDING_DIM), dtype=np.float32)
        cols = min(_EMBEDDING_DIM, proj.shape[1])
        out[:, :cols] = proj[:, :cols]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class HybridDocumentIndex:
    """
    Production Hybrid Retrieval Engine backed by Qdrant (Dense Cosine + Payload Filtering)
    fused with BM25Okapi lexical scoring via Reciprocal Rank Fusion (RRF).
    """

    COLLECTION_NAME = "claim_diff_chunks"

    def __init__(self, docs_dir: Optional[str] = None, qdrant_path: Optional[str] = None):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.docs_dir = docs_dir or os.path.join(root, "documents")
        self.qdrant_path = qdrant_path or os.path.join(root, "artifacts", "qdrant_storage")
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_by_id: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.embedder = DenseEmbedder()
        self.client, self.qdrant_mode = self._init_qdrant_client()
        self.reload()

    def _init_qdrant_client(self):
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        if qdrant_url:
            return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=10), "remote-server"
        if self.qdrant_path == ":memory:":
            return QdrantClient(":memory:"), "in-memory"
        os.makedirs(self.qdrant_path, exist_ok=True)
        try:
            return QdrantClient(path=self.qdrant_path), "embedded-disk"
        except Exception:
            return QdrantClient(":memory:"), "in-memory-fallback"

    def reload(self) -> None:
        self.chunks = load_directory(self.docs_dir)
        self._rebuild_index()

    def ingest_bytes(self, filename: str, raw_bytes: bytes) -> int:
        new_chunks = parse_document_bytes(filename, raw_bytes)
        if not new_chunks:
            return 0
        self.chunks = [c for c in self.chunks if c["filename"] != filename] + new_chunks
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self) -> None:
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        if not self.chunks:
            self.bm25 = None
            return

        tokenized = [re.findall(r"[a-z0-9.-]+", c["content"].lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        texts = [c["content"] for c in self.chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)

        if self.client.collection_exists(self.COLLECTION_NAME):
            self.client.delete_collection(self.COLLECTION_NAME)
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=_EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
        )
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=vectors[idx].tolist(),
                payload=c,
            )
            for idx, c in enumerate(self.chunks)
        ]
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)

    def search(
        self,
        query: str,
        top_k: int = 6,
        min_authority_tier: Optional[int] = None,
        effective_after: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.chunks or self.bm25 is None:
            return []

        must_conditions: List[qmodels.Condition] = []
        if min_authority_tier is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="authority_tier",
                    range=qmodels.Range(gte=int(min_authority_tier)),
                )
            )
        if effective_after:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="effective_date",
                    range=qmodels.Range(gte=effective_after),
                )
            )
        q_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        q_vec = self.embedder.encode([query])[0].tolist()
        qdrant_hits = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=q_vec,
            query_filter=q_filter,
            limit=len(self.chunks),
            with_payload=True,
        ).points

        dense_scores_by_id: Dict[str, float] = {}
        dense_rank_by_id: Dict[str, int] = {}
        for rank, pt in enumerate(qdrant_hits):
            cid = pt.payload["chunk_id"]
            if cid not in self.chunk_by_id:
                continue
            dense_scores_by_id[cid] = float(pt.score)
            dense_rank_by_id[cid] = rank

        allowed_ids = set(dense_rank_by_id.keys())
        tokens = re.findall(r"[a-z0-9.-]+", query.lower())
        raw_bm25 = self.bm25.get_scores(tokens)
        bm25_pairs = [
            (self.chunks[i]["chunk_id"], float(raw_bm25[i]))
            for i in range(len(self.chunks))
            if not q_filter or self.chunks[i]["chunk_id"] in allowed_ids
        ]
        bm25_pairs.sort(key=lambda x: x[1], reverse=True)
        bm25_scores_by_id = dict(bm25_pairs)
        bm25_rank_by_id = {cid: r for r, (cid, _) in enumerate(bm25_pairs)}

        rrf_k = 60.0
        scored: List[tuple] = []
        for cid in allowed_ids:
            d_rank = dense_rank_by_id.get(cid, len(self.chunks))
            b_rank = bm25_rank_by_id.get(cid, len(self.chunks))
            rrf = (1.0 / (rrf_k + d_rank + 1)) + (1.0 / (rrf_k + b_rank + 1))
            scored.append((rrf, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rrf, cid in scored[:top_k]:
            item = dict(self.chunk_by_id[cid])
            item["rrf_score"] = round(float(rrf), 5)
            item["bm25_score"] = round(bm25_scores_by_id.get(cid, 0.0), 4)
            item["dense_score"] = round(dense_scores_by_id.get(cid, 0.0), 4)
            results.append(item)
        return results


hybrid_index = HybridDocumentIndex()
