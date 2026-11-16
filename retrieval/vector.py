"""Dense (semantic) retrieval over the shared corpus.

The corpus is embedded once and cached, so query time is one encoder forward pass
plus an exact dot-product search: 8070 x 1024 float32 is 33 MB, and a full scan
costs well under a millisecond. No ANN index is warranted at this size, and exact
search keeps results deterministic and reproducible.

Cache key covers the model directory, the backend name, the prompt prefixes and a
fingerprint of the corpus, so any of those changing rebuilds instead of silently
serving stale vectors.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from agentic_rag_v2.config import EMBED_CACHE
from agentic_rag_v2.data import Chunk
from agentic_rag_v2.models.embedder import Embedder


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Content hash of the corpus, not of its shape.

    Hashing ids + text *lengths* (the first version) let an edit that kept the
    length — `alpha` -> `omega` — reuse stale vectors.
    """
    h = hashlib.sha1()
    for chunk in chunks:
        h.update(chunk.chunk_id.encode("utf-8"))
        h.update(b"\x1f")
        h.update(chunk.title.encode("utf-8"))
        h.update(b"\x1f")
        h.update(chunk.text.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def cache_slug(embedder: Embedder, chunks: Sequence[Chunk]) -> str:
    """Identifies (weights, corpus): any change rebuilds instead of serving stale data."""
    key = f"{embedder.identity()}|{corpus_fingerprint(chunks)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class VectorIndex:
    """Normalised passage vectors + their chunk ids, searched by cosine."""

    ids: list[str]
    vectors: np.ndarray                       # (N, dim) float32, L2-normalised
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query_vector: np.ndarray, k: int,
               exclude: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k by cosine similarity; `exclude` filters already-seen chunks."""
        if not self.ids or k <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return []
        scores = self.vectors @ (q / norm)
        if exclude:
            keep = np.fromiter((cid not in exclude for cid in self.ids), dtype=bool,
                               count=len(self.ids))
            scores = np.where(keep, scores, -np.inf)
        top = min(k, len(self.ids))
        order = np.argpartition(-scores, top - 1)[:top]
        ranked = sorted(order.tolist(), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in ranked if np.isfinite(scores[i])]

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Vectors in `.npz`, ids + manifest in a `.json` sidecar.

        Ids live in the JSON on purpose: keeping them as an object array in the npz
        forced `allow_pickle=True` on load, i.e. arbitrary code execution from a
        cache file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), vectors=self.vectors)
        meta = dict(self.meta)
        meta["ids"] = list(self.ids)
        meta["dim"] = self.dim
        meta["chunks"] = len(self.ids)
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False),
                                            encoding="utf-8")
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path)
        blob = np.load(path.with_suffix(".npz"), allow_pickle=False)
        vectors = np.asarray(blob["vectors"], dtype=np.float32)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        ids = [str(x) for x in meta.pop("ids", [])]
        if not ids:
            raise ValueError(f"index cache {path} has no ids sidecar; rebuild it")
        if len(ids) != len(vectors):
            raise ValueError(f"index cache {path} is inconsistent: "
                             f"{len(ids)} ids vs {len(vectors)} vectors")
        meta.setdefault("chunks", len(ids))
        meta.setdefault("dim", int(vectors.shape[1]) if vectors.size else 0)
        return cls(ids=ids, vectors=vectors, meta=meta)


def _cache_is_valid(index: "VectorIndex", embedder: Embedder, chunks: Sequence[Chunk],
                    fingerprint: str) -> tuple[bool, str]:
    """Validate a loaded cache against the current embedder and corpus.

    Three independent checks, because a cache file can be copied, renamed or left
    behind by a different model: manifest fingerprint, embedder identity, and the
    vector dimension when the embedder can state it.
    """
    if index.meta.get("corpus_fingerprint") != fingerprint:
        return False, "corpus fingerprint mismatch"
    if index.meta.get("embedder_identity") != embedder.identity():
        return False, "embedder identity mismatch"
    expected = getattr(embedder, "expected_dim", None)
    if expected and index.dim != expected:
        return False, f"dimension {index.dim} != expected {expected}"
    if index.meta.get("chunks") != len(chunks):
        return False, "chunk count mismatch"
    return True, "ok"


def build_index(chunks: Sequence[Chunk], embedder: Embedder, *,
                batch_size: int = 256, force: bool = False,
                cache_path: Path | None = None,
                progress: bool = False) -> tuple[VectorIndex, dict]:
    """Build (or load) the passage index for this corpus + embedder."""
    fingerprint = corpus_fingerprint(chunks)
    slug = cache_slug(embedder, chunks)
    path = cache_path or (EMBED_CACHE / slug)
    stats: dict = {"slug": slug, "path": str(path.with_suffix(".npz")), "cached": False,
                   "corpus_fingerprint": fingerprint}

    if not force and path.with_suffix(".npz").exists():
        try:
            cached = VectorIndex.load(path)
        except Exception as e:  # noqa: BLE001 - a broken cache is rebuilt, not fatal
            stats["cache_error"] = f"{type(e).__name__}: {e}"
        else:
            valid, why = _cache_is_valid(cached, embedder, chunks, fingerprint)
            if valid:
                stats.update({"cached": True, "vectors": len(cached), "dim": cached.dim})
                return cached, stats
            stats["cache_error"] = why

    t0 = time.perf_counter()
    vectors: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start:start + batch_size])
        vectors.append(embedder.embed_passages([_passage_text(c) for c in batch]))
        if progress:
            done = min(start + batch_size, len(chunks))
            print(f"  embedded {done}/{len(chunks)}", flush=True)
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 1024), dtype=np.float32)
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedder": embedder.describe(),
        "embedder_identity": embedder.identity(),
        "chunks": len(chunks),
        "dim": int(matrix.shape[1]) if len(matrix) else 0,
        "corpus_fingerprint": fingerprint,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    index = VectorIndex(ids=[c.chunk_id for c in chunks], vectors=matrix, meta=meta)
    saved = index.save(path)
    stats.update({"cached": False, "vectors": len(index), "dim": index.dim,
                  "seconds": meta["seconds"], "saved": str(saved)})
    return index, stats


def _passage_text(chunk: Chunk) -> str:
    return f"{chunk.title}. {chunk.text}" if chunk.title else chunk.text


class SemanticRetriever:
    """Bundles the embedder and the index behind the retrieval-tool interface."""

    name = "semantic"

    def __init__(self, index: VectorIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder

    def search(self, query: str, k: int,
               exclude: set[str] | None = None) -> list[tuple[str, float]]:
        vector = self.embedder.embed_queries([query])[0]
        return self.index.search(vector, k=k, exclude=exclude)

    def embed_many(self, queries: Iterable[str]) -> dict[str, np.ndarray]:
        """Batch-embed several queries with one forward pass.

        Keys are the *original* strings, so callers can look the vector up with the
        exact query they passed in (a stripped key would raise KeyError on any
        input with surrounding whitespace).
        """
        originals = [q for q in queries if q and q.strip()]
        unique = list(dict.fromkeys(q.strip() for q in originals))
        if not unique:
            return {}
        vectors = self.embedder.embed_queries(unique)
        by_text = dict(zip(unique, vectors))
        return {q: by_text[q.strip()] for q in originals}

    def describe(self) -> dict:
        return {"index": {k: self.index.meta.get(k) for k in
                          ("chunks", "dim", "built_at", "seconds", "corpus_fingerprint")},
                "vectors": len(self.index),
                "embedder": self.embedder.describe()}
