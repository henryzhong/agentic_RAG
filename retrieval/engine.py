"""Hybrid retriever: BM25 seeds + exact title anchoring + entity-graph expansion.

This is the retrieval tool set the ReAct loop calls. Fusion weights are the ones
validated in the v1 harness (lexical 0.58 / graph 0.42, each max-normalised), so
v2 retrieval recall stays comparable to the published baseline.
"""
from __future__ import annotations

import math
from typing import Iterable

from agentic_rag_v2.config import BM25_K, GRAPH_FANOUT, GRAPH_HOPS
from agentic_rag_v2.data import Chunk
from agentic_rag_v2.retrieval.graph import EntityGraph
from agentic_rag_v2.retrieval.lexical import LexicalIndex
from agentic_rag_v2.retrieval.text import RELATION_WORDS, STOP, TOKEN_RE


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.lexical = LexicalIndex(chunks)
        self.graph = EntityGraph(chunks)

    # ------------------------------------------------------------------ tools
    def bm25_search(self, query: str, k: int = BM25_K,
                    exclude: set[str] | None = None) -> list[tuple[str, float]]:
        return self.lexical.search(query, k=k, exclude=exclude)

    def link_entities(self, query: str) -> list[tuple[str, float]]:
        """Entity linking: title spans first, then high-specificity n-grams.

        Relation words (child / spouse / left / year...) are never seeds; they
        describe the edge we are looking for, not a node.
        """
        qtoks = [t for t in TOKEN_RE.findall(query.lower()) if t not in STOP]
        if not qtoks:
            return []
        qtext = " ".join(qtoks)
        found: dict[str, float] = {}

        for e, cids in self.graph.entity_chunks.items():
            if len(e) < 5 or e in RELATION_WORDS:
                continue
            etoks = e.split()
            if len(etoks) == 1 and (etoks[0] in RELATION_WORDS or len(etoks[0]) < 4):
                continue
            if e in qtext:
                found[e] = max(found.get(e, 0.0),
                               2.0 + 0.35 * len(etoks) + 0.6 / math.sqrt(len(cids)))

        for n in range(min(6, len(qtoks)), 0, -1):
            for i in range(len(qtoks) - n + 1):
                span = " ".join(qtoks[i:i + n])
                if span in RELATION_WORDS or span in found:
                    continue
                cids = self.graph.entity_chunks.get(span)
                if not cids:
                    continue
                idf = math.log(1.0 + len(self.chunks) / (1.0 + len(cids)))
                if n == 1 and idf < 2.2:
                    continue
                found[span] = 0.9 + 0.30 * n + 0.25 * idf
        return sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))[:12]

    def expand(self, seeds: list[tuple[str, float]], hops: int = GRAPH_HOPS,
               exclude: set[str] | None = None):
        return self.graph.expand(seeds, hops=hops, fanout=GRAPH_FANOUT, exclude=exclude)

    def neighbours_of_chunks(self, chunk_ids: list[str], limit: int = 6):
        return self.graph.neighbours_of_chunks(chunk_ids, limit=limit)

    # ---------------------------------------------------------------- hybrid
    def title_anchor(self, query: str) -> dict[str, float]:
        """Exact-title anchoring boost.

        If the query names a document, that document must be seen. Token coverage
        (not equality) so "Kent County Cricket Club in June, 2011" still anchors
        "Kent County Cricket Club in 2011".
        """
        boosts: dict[str, float] = {}
        qtok_set = set(TOKEN_RE.findall(query.lower()))
        for e, cids in self.graph.entity_chunks.items():
            if len(e) < 6 or e in RELATION_WORDS:
                continue
            etoks = [t for t in e.split() if t not in STOP]
            if len(etoks) < 2:
                continue
            hit = sum(1 for t in etoks if t in qtok_set)
            coverage = hit / len(etoks)
            if coverage >= 0.7 and hit >= 2:
                idf = math.log(1.0 + len(self.chunks) / (1.0 + len(cids)))
                boost = 0.55 + 0.45 * coverage + 0.18 * idf
                for cid in sorted(cids)[:2]:
                    boosts[cid] = max(boosts.get(cid, 0.0), boost)
        return boosts

    def hybrid(self, query: str, k: int = BM25_K, use_graph: bool = True,
               hops: int = GRAPH_HOPS,
               exclude: set[str] | None = None) -> tuple[list[tuple[str, float]], dict]:
        """Return (ranked [(chunk_id, score)], trace)."""
        lex = self.bm25_search(query, k=max(k * 2, 12), exclude=exclude)
        lex_max = max((s for _, s in lex), default=1.0) or 1.0
        fused: dict[str, float] = {cid: 0.58 * (s / lex_max) for cid, s in lex}
        trace: dict = {"lexical": [(c, round(s, 4)) for c, s in lex[:k]],
                       "seeds": [], "expanded": [], "paths": [], "title_hits": []}

        anchors = self.title_anchor(query)
        for cid, boost in anchors.items():
            if exclude and cid in exclude:
                continue
            fused[cid] = fused.get(cid, 0.0) + boost
        trace["title_hits"] = sorted(anchors)[:6]

        if use_graph:
            seeds = self.link_entities(query)
            trace["seeds"] = [e for e, _ in seeds]
            if seeds:
                ranked, ent_scores, paths = self.expand(seeds, hops=hops, exclude=exclude)
                g_max = max((s for _, s in ranked), default=1.0) or 1.0
                for cid, s in ranked[: k * 3]:
                    fused[cid] = fused.get(cid, 0.0) + 0.42 * (s / g_max)
                trace["expanded"] = sorted(ent_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
                trace["paths"] = [paths[e] for e, _ in trace["expanded"][:6]]

        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        trace["fused"] = [(cid, round(s, 4)) for cid, s in ranked]
        return ranked, trace

    def shared_entities(self, chunk_ids: Iterable[str], limit: int = 8):
        return self.graph.shared_entities(list(chunk_ids), limit=limit)


def index_stats(retriever: HybridRetriever) -> dict:
    ents = retriever.graph.entity_count()
    return {
        "chunks": len(retriever.chunks),
        "entities": ents,
        "cooccur_edges": retriever.graph.edge_count(),
        "avg_entity_degree": round(
            sum(len(v) for v in retriever.graph._neighbours.values()) / max(ents, 1), 2),
    }
