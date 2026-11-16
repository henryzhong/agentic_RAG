"""E5 embedders: one interface, two interchangeable backends.

    LocalE5Embedder   transformers + mean pooling, in-process (GPU or CPU)
    VLLMEmbedder      OpenAI-compatible /v1/embeddings (vllm/vllm-openai image)

Both apply the multilingual-e5 prefixes, which are part of the model contract:
text embedded without them scores systematically worse against the other side.

    query   -> "query: <question>"
    passage -> "passage: <text>"

Vectors come back L2-normalised float32, so cosine similarity is a dot product.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def describe(self) -> dict: ...

    def identity(self) -> str:
        """Stable string identifying *which weights* produce the vectors."""

    @property
    def expected_dim(self) -> int | None:
        """Known output dimension, or None when it is only discoverable at runtime."""


def format_query(text: str) -> str:
    return f"query: {text.strip()}"


def format_passage(text: str) -> str:
    return f"passage: {text.strip()}"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


class LocalE5Embedder:
    """Runs multilingual-e5-large in-process with mean pooling."""

    name = "e5-local"

    def __init__(self, model_dir: str | os.PathLike | None = None, *,
                 max_len: int | None = None, batch_size: int | None = None,
                 device: str | None = None):
        # local weights only: a hub lookup must never block a run
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModel, AutoTokenizer

        from agentic_rag_v2.config import EMBED_BATCH, EMBED_MAX_LEN, EMBED_DIM, e5_model_dir

        self._torch = torch
        self.model_dir = str(model_dir or e5_model_dir() or "")
        if not self.model_dir:
            raise FileNotFoundError("multilingual-e5-large not found; set RAGV2_E5_DIR")
        self.max_len = max_len or EMBED_MAX_LEN
        self.batch_size = batch_size or EMBED_BATCH
        self.dim = EMBED_DIM
        self._lock = threading.Lock()

        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_dir, dtype=dtype).to(self.device).eval()

    # ------------------------------------------------------------------ embed
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([format_query(t) for t in texts])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([format_passage(t) for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            with self._lock, torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                out.append(pooled.float().cpu().numpy())
        return _normalise(np.vstack(out))

    def describe(self) -> dict:
        return {"backend": "local", "kind": "e5", "name": self.name,
                "model_dir": self.model_dir, "dim": self.dim, "device": self.device,
                "max_len": self.max_len}

    def identity(self) -> str:
        return f"local|{self.model_dir}|{self.max_len}"

    @property
    def expected_dim(self) -> int | None:
        return self.dim


class VLLMEmbedder:
    """Calls a vLLM OpenAI-compatible embeddings server (see scripts/serve_vllm_embed.ps1)."""

    name = "e5-vllm"

    def __init__(self, endpoint: str | None = None, *, model: str | None = None,
                 batch_size: int = 64, timeout: float = 120.0):
        import httpx

        from agentic_rag_v2.config import EMBED_DIM, EMBED_ENDPOINT, EMBED_MODEL_NAME

        self.endpoint = (endpoint or EMBED_ENDPOINT).rstrip("/")
        self.model = model or EMBED_MODEL_NAME
        self.batch_size = batch_size
        self.dim = EMBED_DIM
        self._client = httpx.Client(timeout=timeout)
        self._lock = threading.Lock()
        self.last_error: str | None = None

    # ------------------------------------------------------------------ probe
    def probe(self) -> dict:
        """Cheap health check used by scripts and the run manifest."""
        try:
            r = self._client.get(f"{self.endpoint}/models")
            r.raise_for_status()
            served = [m.get("id") for m in (r.json().get("data") or [])]
            return {"ok": True, "served": served, "endpoint": self.endpoint}
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            return {"ok": False, "error": self.last_error, "endpoint": self.endpoint}

    # ------------------------------------------------------------------ embed
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([format_query(t) for t in texts])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([format_passage(t) for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            with self._lock:
                r = self._client.post(f"{self.endpoint}/embeddings", json=payload)
            r.raise_for_status()
            data = r.json().get("data") or []
            data.sort(key=lambda d: d.get("index", 0))
            vectors.extend(np.asarray(d["embedding"], dtype=np.float32) for d in data)
        matrix = np.vstack(vectors)
        self.dim = int(matrix.shape[1])
        self._dim_verified = self.dim
        return _normalise(matrix)

    def describe(self) -> dict:
        return {"backend": "vllm", "kind": "e5", "name": self.name,
                "endpoint": self.endpoint, "model": self.model, "dim": self.dim}

    def identity(self) -> str:
        # the endpoint matters: two servers can expose the same served name with
        # different weights, and a shared cache would silently mix their vectors
        return f"vllm|{self.endpoint}|{self.model}"

    @property
    def expected_dim(self) -> int | None:
        # vLLM reports the dimension only after the first call
        return getattr(self, "_dim_verified", None)


class CachedEmbedder:
    """Memoises query vectors: the planner, the judge and the answerer may ask
    about the same text, and a repeat must not cost another forward pass."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return getattr(self.inner, "dim", 0)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        keys = [t.strip() for t in texts]
        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if missing:
            fresh = self.inner.embed_queries(missing)
            with self._lock:
                for text, vec in zip(missing, fresh):
                    self._cache[text] = vec
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._cache[k] for k in keys])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed_passages(texts)

    def describe(self) -> dict:
        info = dict(self.inner.describe())
        info["cached_queries"] = len(self._cache)
        return info

    def identity(self) -> str:
        return self.inner.identity()

    @property
    def expected_dim(self) -> int | None:
        return self.inner.expected_dim


def build_embedder(backend: str = "local", **kwargs) -> Embedder:
    """Construct the configured backend, wrapped in the query cache."""
    backend = (backend or "local").lower()
    if backend == "vllm":
        embedder: Embedder = VLLMEmbedder(**kwargs)
    elif backend == "local":
        embedder = LocalE5Embedder(**kwargs)
    else:
        raise ValueError(f"unknown embedding backend: {backend}")
    return CachedEmbedder(embedder)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity helper for tests and diagnostics."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not math.isfinite(denom) or denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dump_json(path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
