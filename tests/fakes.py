"""Deterministic doubles for the LLM, Model A and the evidence judge."""
from __future__ import annotations

from typing import Callable

from agentic_rag_v2.types import Evidence, JudgeVerdict, RouteSignal

Reply = dict | str | Exception | Callable[[list[dict]], dict]


class ScriptedClient:
    """Maps a call `tag` to a canned JSON reply; records the call order."""

    def __init__(self, responses: dict[str, Reply], model: str = "scripted"):
        self.responses = responses
        self.calls: list[str] = []
        self.model = model

    def chat(self, messages, *, max_tokens: int = 1024, temperature: float | None = 0.0,
             tag: str = "") -> str:  # pragma: no cover - json_call covers the pipeline
        raise AssertionError("pipeline should use json_call")

    def json_call(self, messages, *, max_tokens: int = 1024, tag: str = "") -> dict:
        self.calls.append(tag)
        if tag not in self.responses:
            raise KeyError(f"no scripted reply for tag {tag!r}")
        reply = self.responses[tag]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(messages)
        if isinstance(reply, str):
            raise AssertionError("json_call expects a dict reply")
        return reply


class FakeAnalyzer:
    ok = True
    error = None
    checkpoint = "fake"

    def __init__(self, *, hop: str = "2", multihop: bool = True, prob: float = 0.85,
                 complexity: str = "medium", intent: str = "multi_hop"):
        self.hop = hop
        self.multihop = multihop
        self.prob = prob
        self.complexity = complexity
        self.intent = intent
        self.questions: list[str] = []

    def predict(self, question: str) -> RouteSignal:
        self.questions.append(question)
        return RouteSignal(available=True, intent=self.intent, intents={self.intent: 0.9},
                           multihop=self.multihop, multihop_prob=self.prob,
                           hop_bucket=self.hop, complexity=self.complexity)

    def require(self) -> None:
        pass


class BrokenAnalyzer:
    """Mimics a checkpoint that failed to load."""

    ok = False
    checkpoint_name = "broken"
    checkpoint = "<missing>/best.pt"
    error = "FileNotFoundError: <missing>/best.pt"
    calls = 0

    def predict(self, question: str) -> RouteSignal:  # pragma: no cover - never reached
        BrokenAnalyzer.calls += 1
        raise AssertionError("a broken analyzer must not be used")

    def require(self) -> None:
        raise RuntimeError(f"Model A checkpoint '{self.checkpoint_name}' could not be loaded")


class RecordingJudge:
    """Judge that replays a fixed verdict sequence and snapshots the evidence."""

    name = "recording_judge"
    llm_calls = 0

    def __init__(self, verdicts: list[dict] | None = None):
        self.verdicts = list(verdicts or [])
        self.seen: list[list[str]] = []

    def judge(self, question: str, evidence: Evidence, *,
              multihop: bool = False) -> JudgeVerdict:
        self.seen.append([p.chunk_id for p in evidence.ordered()])
        raw = self.verdicts.pop(0) if self.verdicts else {"sufficient": True}
        return JudgeVerdict(sufficient=bool(raw.get("sufficient")),
                            missing=str(raw.get("missing") or ""),
                            next_query=str(raw.get("next_query") or ""),
                            reason=str(raw.get("reason") or ""),
                            source="llm" if "sufficient" in raw else "stub")


class FakeReranker:
    """Cross-encoder double: passages containing `key` score high."""

    available = True
    kind = "fake"
    error = None

    def __init__(self, key: str = "second hop", high: float = 6.0, low: float = -6.0):
        self.key = key
        self.high = high
        self.low = low
        self.calls = 0

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        return [self.high if self.key in p else self.low for p in passages]

    def describe(self) -> dict:
        return {"available": True, "kind": "fake", "model_dir": "fake", "error": None}


class BrokenReranker(FakeReranker):
    available = False
    error = "not loaded"

    def rerank(self, query: str, passages: list[str]) -> None:  # type: ignore[override]
        return None


class FakeEmbedder:
    """Deterministic bag-of-words embedder: no model, no network, stable vectors."""

    name = "fake-e5"

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vector(self, text: str):
        import numpy as np

        tokens = text.lower().replace("query:", " ").replace("passage:", " ").split()
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            vec[hash(token) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm

    def embed_queries(self, texts):
        import numpy as np

        self.calls.append(list(texts))
        return np.vstack([self._vector(f"query: {t}") for t in texts]) if texts \
            else np.zeros((0, self.dim), dtype=np.float32)

    def embed_passages(self, texts):
        import numpy as np

        return np.vstack([self._vector(f"passage: {t}") for t in texts]) if texts \
            else np.zeros((0, self.dim), dtype=np.float32)

    def describe(self) -> dict:
        return {"backend": "fake", "kind": "e5", "name": self.name, "dim": self.dim,
                "model_dir": "fake"}

    def identity(self) -> str:
        # deliberately independent of `dim`, mirroring a served model whose
        # dimension is only discovered at runtime
        return f"fake|{self.name}"

    @property
    def expected_dim(self) -> int | None:
        return self.dim


class _FakeVectorIndex:
    """Stand-in for VectorIndex: 'vectors' are the query strings themselves."""

    meta: dict = {"chunks": 0, "dim": 0}

    def __init__(self, owner: "FakeSemantic"):
        self.owner = owner
        self.ids = list(owner.mapping.values())

    def search(self, vector, k: int, exclude=None):
        return self.owner._hits(str(vector), k, exclude)


class FakeSemantic:
    """Keyword-driven stand-in for SemanticRetriever (no model, no real index).

    `embed_many` returns the query text as the "vector", and the fake index maps
    keywords found in that text to chunk ids — so the batched-encode path used by
    `run_search` is exercised exactly like production.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.queries: list[str] = []
        self.index = _FakeVectorIndex(self)

    def _hits(self, query: str, k: int, exclude=None):
        hits = [(cid, 0.9 - 0.01 * i) for i, (kw, cid) in enumerate(self.mapping.items())
                if kw in query.lower() and (not exclude or cid not in exclude)]
        return hits[:k]

    def search(self, query: str, k: int, exclude=None):
        self.queries.append(query)
        return self._hits(query, k, exclude)

    def embed_many(self, queries):
        return {q: q for q in queries}

    def describe(self) -> dict:
        return {"index": {"chunks": len(self.mapping), "dim": 0},
                "vectors": len(self.mapping), "embedder": {"backend": "fake"}}
