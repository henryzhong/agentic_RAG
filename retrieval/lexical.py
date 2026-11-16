"""BM25 first stage."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from agentic_rag_v2.config import BM25_K
from agentic_rag_v2.data import Chunk
from agentic_rag_v2.retrieval.text import tokenize


class LexicalIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.title + " " + c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int = BM25_K,
               exclude: set[str] | None = None) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        out: list[tuple[str, float]] = []
        for i in order:
            cid = self.chunks[i].chunk_id
            if exclude and cid in exclude:
                continue
            if scores[i] <= 0:
                break
            out.append((cid, float(scores[i])))
            if len(out) >= k:
                break
        return out
