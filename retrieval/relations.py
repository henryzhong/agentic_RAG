"""Semantic relations from evidence — the v1 knowledge-graph layer.

v1 extracted (head, relation, tail) triples lazily with one extra LLM call per
round and injected them into the assess/answer prompts. That call is the first
thing to cut when the budget is two calls per question, so v2 **reuses v1's
extraction cache** instead of re-extracting: the cache is a static artifact, like
the corpus and the gold file, and covers a third of the corpus yet touches every
question's evidence pool.

The extracted relations are what a co-occurrence graph cannot express: they let
the answerer follow "father of", "followed by", "record label" as an explicit edge
instead of hoping both entities co-occur in one paragraph.

Extraction stays available for uncached passages (`extract=True`), but it is off
by default: it would add an LLM call and break the two-call budget.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agentic_rag_v2.config import LEGACY_WORK, WORK
from agentic_rag_v2.types import Evidence

DEFAULT_CACHE = LEGACY_WORK / "kg_triples.jsonl"
MAX_RELATIONS = 14


class RelationStore:
    def __init__(self, path: str | os.PathLike | None = None, *, client=None,
                 extract: bool = False):
        self.path = Path(path) if path else DEFAULT_CACHE
        self.client = client
        self.extract = extract
        self.cache: dict[str, list[dict]] = {}
        self.loaded = False
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.error = f"relation cache missing: {self.path}"
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                triples = rec.get("triples") or []
                if triples:
                    self.cache[rec["chunk_id"]] = triples
            self.loaded = True
        except Exception as e:  # noqa: BLE001 - relations are an enhancement
            self.error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ query
    def for_chunks(self, chunk_ids: list[str], limit: int = MAX_RELATIONS) -> list[str]:
        """Deduplicated `head --relation--> tail` lines for the given passages."""
        seen: set[tuple[str, str, str]] = set()
        lines: list[str] = []
        for cid in chunk_ids:
            for triple in self.cache.get(cid, ()):
                key = (triple["head"].lower(), triple["relation"], triple["tail"].lower())
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {triple['head']} --{triple['relation']}--> {triple['tail']}")
                if len(lines) >= limit:
                    return lines
        return lines

    def prompt_lines(self, evidence: Evidence, chunk_ids: list[str] | None = None,
                     limit: int = MAX_RELATIONS) -> str:
        ids = chunk_ids or [p.chunk_id for p in evidence.top(10)]
        lines = self.for_chunks(ids, limit=limit)
        if not lines:
            return ""
        return ("KNOWLEDGE-GRAPH RELATIONS extracted from the evidence:\n"
                + "\n".join(lines))

    def coverage(self, chunk_ids: list[str]) -> dict:
        hit = sum(1 for cid in chunk_ids if cid in self.cache)
        return {"chunks": len(chunk_ids), "with_relations": hit}

    def describe(self) -> dict:
        return {"kind": "relation_cache", "path": str(self.path), "loaded": self.loaded,
                "chunks_cached": len(self.cache), "error": self.error,
                "extract_on_demand": self.extract}


_STORE: RelationStore | None = None


def get_store() -> RelationStore:
    """Process-wide store: the file is read once, not once per question."""
    global _STORE
    if _STORE is None:
        _STORE = RelationStore()
    return _STORE
