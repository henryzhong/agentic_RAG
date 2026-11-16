"""Entity co-occurrence graph.

Two edge kinds, both deterministic:
  * mention   : chunk -- entity (which passage mentions which entity)
  * co-occur  : entity -- entity (they appear in the same passage)

Expansion is a bounded BFS with decay, and chunk scoring uses IDF weighting with
max-aggregation: a rare entity is diagnostic, and many weak entities must not
outvote one precise match (hub-entity pollution).
"""
from __future__ import annotations

import math
from collections import defaultdict

from agentic_rag_v2.config import GRAPH_FANOUT, GRAPH_HOPS
from agentic_rag_v2.data import Chunk
from agentic_rag_v2.retrieval.text import extract_entities


class EntityGraph:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.chunk_entities: dict[str, tuple[str, ...]] = {}
        self.entity_chunks: dict[str, list[str]] = defaultdict(list)
        self.cooccur: dict[tuple[str, str], int] = {}

        for chunk in chunks:
            ents = tuple(sorted(extract_entities(chunk.text, chunk.title)))
            self.chunk_entities[chunk.chunk_id] = ents
            for e in ents:
                self.entity_chunks[e].append(chunk.chunk_id)
            # sorted -> identical edge set in every process (PYTHONHASHSEED safe)
            for i in range(len(ents)):
                for j in range(i + 1, min(i + 6, len(ents))):
                    key = (ents[i], ents[j])
                    self.cooccur[key] = self.cooccur.get(key, 0) + 1

        self._neighbours: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (a, b), w in sorted(self.cooccur.items()):
            self._neighbours[a].append((b, w))
            self._neighbours[b].append((a, w))
        for e in self._neighbours:
            self._neighbours[e].sort(key=lambda kv: (-kv[1], kv[0]))

    # ------------------------------------------------------------------ graph
    def entity_count(self) -> int:
        return len(self.entity_chunks)

    def edge_count(self) -> int:
        return len(self.cooccur)

    def neighbours(self, entity: str, limit: int = GRAPH_FANOUT) -> list[tuple[str, int]]:
        return self._neighbours.get(entity, [])[:limit]

    def expand(self, seeds: list[tuple[str, float]], hops: int = GRAPH_HOPS,
               fanout: int = GRAPH_FANOUT,
               exclude: set[str] | None = None) -> tuple[list[tuple[str, float]], dict[str, float], dict[str, list[str]]]:
        """BFS over entities; returns (ranked chunks, entity scores, paths)."""
        ent_scores: dict[str, float] = {e: s for e, s in seeds}
        paths: dict[str, list[str]] = {e: [e] for e, _ in seeds}
        frontier = [e for e, _ in seeds]

        for hop in range(1, hops + 1):
            decay = 0.55 ** hop
            nxt: list[str] = []
            for e in frontier:
                for n, w in self.neighbours(e, fanout):
                    sc = ent_scores[e] * decay * (w / (w + 1.5))
                    if sc > ent_scores.get(n, 0.0):
                        ent_scores[n] = sc
                        paths[n] = paths[e] + [n]
                        nxt.append(n)
            frontier = nxt

        chunk_scores: dict[str, float] = defaultdict(float)
        for e, sc in ent_scores.items():
            cids = self.entity_chunks.get(e)
            if not cids:
                continue
            idf = math.log(1.0 + len(self.chunks) / (1.0 + len(cids)))
            spec = idf / math.sqrt(len(cids))
            for cid in cids:
                if exclude and cid in exclude:
                    continue
                chunk_scores[cid] = max(chunk_scores[cid], sc * spec)
        ranked = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked, ent_scores, paths

    def neighbours_of_chunks(self, chunk_ids: list[str],
                             limit: int = 6) -> list[tuple[str, float]]:
        """Chunks sharing at least one entity with the given evidence (one hop)."""
        seen = set(chunk_ids)
        scores: dict[str, float] = defaultdict(float)
        for cid in chunk_ids:
            for e in self.chunk_entities.get(cid, ()):
                cids = self.entity_chunks.get(e, ())
                if not cids:
                    continue
                idf = math.log(1.0 + len(self.chunks) / (1.0 + len(cids)))
                spec = idf / math.sqrt(len(cids))
                for other in cids:
                    if other in seen:
                        continue
                    scores[other] = max(scores[other], spec)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def shared_entities(self, chunk_ids: list[str], limit: int = 8) -> list[tuple[str, list[int]]]:
        """Bridge entities: surface forms occurring in more than one passage."""
        owners: dict[str, list[int]] = defaultdict(list)
        for i, cid in enumerate(chunk_ids, 1):
            for e in self.chunk_entities.get(cid, ()):
                owners[e].append(i)
        shared = [(e, ids) for e, ids in owners.items() if len(ids) >= 2 and len(e) >= 4]
        shared.sort(key=lambda kv: (-len(kv[1]), -len(kv[0])))
        return shared[:limit]
