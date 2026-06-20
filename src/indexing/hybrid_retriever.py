"""Hybrid retrieval: FAISS + BM25 → RRF fusion → BGE Reranker → Top-5.

"先广后精" two-stage pipeline:
  Stage 1: FAISS (semantic, Top-20) + BM25 (keyword, Top-20) → RRF → 40 candidates
  Stage 2: BGE-Reranker re-scores → Top-5 seeds for graph expansion
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

_RRF_K = 60  # Reciprocal Rank Fusion constant


def _query_tokens(text: str) -> list[str]:
    """Lightweight multilingual tokenizer for BM25 indexing/querying."""
    raw = str(text).lower()
    en_words = re.findall(r"[a-z0-9_]{2,}", raw)
    zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", raw)
    single_zh = re.findall(r"[\u4e00-\u9fff]", raw)
    return en_words + zh_terms + single_zh


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """Wraps rank_bm25.BM25Okapi with build / save / load / search."""

    def __init__(self) -> None:
        self._bm25: Any = None
        self._corpus: list[list[str]] = []
        self._documents: list[Document] = []

    def build(self, documents: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._documents = list(documents)
        self._corpus = [_query_tokens(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, k: int = 20) -> list[Document]:
        if self._bm25 is None:
            return []
        tokens = _query_tokens(query)
        if not tokens:
            return self._documents[:k]
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        return [self._documents[i] for i, _ in ranked[:k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"corpus": self._corpus, "documents": self._documents}, f
            )

    def load(self, path: Path) -> None:
        from rank_bm25 import BM25Okapi

        with open(path, "rb") as f:
            data = pickle.load(f)
        self._corpus = data["corpus"]
        self._documents = data["documents"]
        self._bm25 = BM25Okapi(self._corpus)

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None


# ---------------------------------------------------------------------------
# BGE Reranker (lazy-loaded via sentence_transformers.CrossEncoder)
# ---------------------------------------------------------------------------

class BGEReranker:
    """Lazy-loading BGE Reranker using sentence_transformers.CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder
            print(f"    加载 Reranker 模型: {self._model_name} ...")
            self._model = CrossEncoder(self._model_name)
            return True
        except Exception as exc:
            print(f"    Reranker 模型加载失败，将跳过精排步骤: {exc}")
            self._load_failed = True
            return False

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not documents:
            return []
        if not self._ensure_loaded():
            return documents[:top_k]
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(documents, scores), key=lambda x: float(x[1]), reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fusion(
    vector_results: list[Document],
    bm25_results: list[Document],
    k: int = _RRF_K,
    max_candidates: int = 40,
) -> list[Document]:
    """Merge FAISS + BM25 results via Reciprocal Rank Fusion.

    score = Σ 1 / (rank_in_list + k), merged by node_id, deduplicated.
    """
    fusion: dict[str, tuple[float, Document]] = {}

    for rank, doc in enumerate(vector_results):
        nid = str(doc.metadata.get("node_id", ""))
        if not nid:
            continue
        score = 1.0 / (rank + 1 + k)
        if nid in fusion:
            prev_score, _ = fusion[nid]
            fusion[nid] = (prev_score + score, doc)
        else:
            fusion[nid] = (score, doc)

    for rank, doc in enumerate(bm25_results):
        nid = str(doc.metadata.get("node_id", ""))
        if not nid:
            continue
        score = 1.0 / (rank + 1 + k)
        if nid in fusion:
            prev_score, _ = fusion[nid]
            fusion[nid] = (prev_score + score, doc)
        else:
            fusion[nid] = (score, doc)

    ranked = sorted(fusion.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:max_candidates]]


# ---------------------------------------------------------------------------
# Hybrid Retriever (orchestrator)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Orchestrates FAISS + BM25 → RRF → Reranker pipeline."""

    def __init__(
        self,
        faiss_store: Any,
        bm25_index: BM25Index | None,
        reranker: BGEReranker | None,
        *,
        enable_bm25: bool = True,
        enable_reranker: bool = True,
    ) -> None:
        self._faiss = faiss_store
        self._bm25 = bm25_index
        self._reranker = reranker
        self._enable_bm25 = enable_bm25
        self._enable_reranker = enable_reranker

    def retrieve(
        self,
        question: str,
        k_faiss: int = 20,
        k_bm25: int = 20,
        final_k: int = 5,
    ) -> list[Document]:
        # Stage 1: parallel retrieval
        vector_docs: list[Document] = []
        try:
            vector_docs = self._faiss.similarity_search(question, k=k_faiss)
        except Exception:
            pass

        bm25_docs: list[Document] = []
        if self._enable_bm25 and self._bm25 is not None and self._bm25.is_loaded:
            bm25_docs = self._bm25.search(question, k=k_bm25)

        # If BM25 is disabled or empty, fallback to vector-only
        if not bm25_docs:
            candidates = vector_docs[:final_k * 4]  # 20 candidates
        else:
            # RRF fusion
            candidates = rrf_fusion(vector_docs, bm25_docs)

        if not candidates:
            return []

        # Stage 2: rerank
        if self._enable_reranker and self._reranker is not None and len(candidates) > final_k:
            return self._reranker.rerank(question, candidates, top_k=final_k)

        return candidates[:final_k]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def load_bm25_index(index_dir: str | Path) -> BM25Index:
    """Load a persisted BM25 index from disk."""
    index_path = Path(index_dir) / "bm25_index.pkl"
    bm25 = BM25Index()
    if index_path.exists():
        bm25.load(index_path)
    return bm25


def create_hybrid_retriever(
    faiss_store: Any,
    bm25_dir: str | Path,
    reranker_model: str,
    *,
    enable_bm25: bool = True,
    enable_reranker: bool = True,
) -> HybridRetriever:
    bm25 = load_bm25_index(bm25_dir)
    reranker = BGEReranker(model_name=reranker_model) if enable_reranker else None
    return HybridRetriever(
        faiss_store=faiss_store,
        bm25_index=bm25,
        reranker=reranker,
        enable_bm25=enable_bm25,
        enable_reranker=enable_reranker,
    )
