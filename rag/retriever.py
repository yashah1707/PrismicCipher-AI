from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import ConfigDict, Field
from sentence_transformers import CrossEncoder

from config import ENSEMBLE_TOP_K, FAISS_FETCH_K, FINAL_CONTEXT_CHUNKS, RERANKER_MODEL
from rag.loader import sanitize_text


@lru_cache(maxsize=4)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Return a cached CrossEncoder instance for the given model name."""
    return CrossEncoder(model_name)


class HybridRerankRetriever(BaseRetriever):
    """Retrieve with FAISS MMR + BM25, then rerank candidates with a cross-encoder."""

    vectorstore: Any
    documents: list[Document] = Field(default_factory=list)
    top_k: int = ENSEMBLE_TOP_K
    final_k: int = FINAL_CONTEXT_CHUNKS
    faiss_fetch_k: int = FAISS_FETCH_K
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    reranker_model_name: str = RERANKER_MODEL

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query = sanitize_text(query)
        if not self.documents:
            return []

        faiss_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.top_k,
            fetch_k=self.faiss_fetch_k,
        )

        bm25 = BM25Retriever.from_documents(self.documents)
        bm25.k = self.top_k
        bm25_docs = bm25.invoke(query)

        candidates = self._merge_candidates(faiss_docs, bm25_docs)
        if not candidates:
            return []

        reranker = _get_cross_encoder(self.reranker_model_name)
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _score in ranked[: self.final_k]]

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self._get_relevant_documents(query)

    def _merge_candidates(
        self,
        faiss_docs: list[Document],
        bm25_docs: list[Document],
    ) -> list[Document]:
        scores: dict[str, float] = {}
        docs_by_key: dict[str, Document] = {}

        for rank, doc in enumerate(faiss_docs):
            key = _document_key(doc)
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (rank + 1)
            docs_by_key[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = _document_key(doc)
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (rank + 1)
            docs_by_key[key] = doc

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [docs_by_key[key] for key in ranked_keys[: self.top_k]]


def _document_key(document: Document) -> str:
    filename = document.metadata.get("filename", "")
    page = document.metadata.get("page", "")
    return f"{filename}:{page}:{hash(document.page_content)}"
