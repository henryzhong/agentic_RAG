"""Typed contracts shared by every stage.

The pipeline is a chain of pure transformations between these objects, which is
what makes each stage independently testable: a test feeds a literal object and
asserts on the next one, without a graph engine in between.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SearchMode(str, Enum):
    """Search strategy chosen by the router stage."""

    KEEP = "keep"        # the question is already a good query
    REWRITE = "rewrite"  # search one or more rewritten / decomposed queries
    GRAPH = "graph"      # entity-graph first (bridge questions)

    @classmethod
    def coerce(cls, value: object, default: "SearchMode | None" = None) -> "SearchMode":
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return default or cls.KEEP


# --------------------------------------------------------------------- stage 1
@dataclass(slots=True)
class RouteSignal:
    """Model A output, normalised. `available=False` means the model was absent
    or failed; consumers must then fall back to neutral defaults."""

    available: bool = False
    intent: str = "unknown"
    intents: dict[str, float] = field(default_factory=dict)
    multihop: bool = False
    multihop_prob: float = 0.5
    hop_bucket: str = "2"
    complexity: str = "medium"
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "intent": self.intent,
            "intents": dict(self.intents),
            "multihop": self.multihop,
            "multihop_prob": self.multihop_prob,
            "hop_bucket": self.hop_bucket,
            "complexity": self.complexity,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    def describe(self) -> str:
        """One-line summary handed to the router LLM."""
        if not self.available:
            return f"unavailable ({self.error}) — assume a multi-hop bridge question"
        return (f"intent={self.intent} multihop_prob={self.multihop_prob:.2f} "
                f"hop={self.hop_bucket} complexity={self.complexity}")


@dataclass(slots=True)
class StageBudget:
    """Deterministic limits applied to the LLM's proposal."""

    rounds: int
    top_k: int
    answer_top_n: int
    source: str = "default"

    def to_dict(self) -> dict:
        return {"rounds": self.rounds, "top_k": self.top_k,
                "answer_top_n": self.answer_top_n, "source": self.source}


# --------------------------------------------------------------------- stage 2
@dataclass(slots=True)
class SearchPlan:
    """One planning call delivers route + queries + graph switch + budget."""

    mode: SearchMode
    queries: list[str] = field(default_factory=list)
    graph: bool = False
    rounds: int = 1
    top_k: int = 8
    reason: str = ""
    source: str = "llm"                 # llm | fallback | forced

    def search_queries(self, question: str) -> list[str]:
        """De-duplicated query list, order preserved."""
        out: list[str] = []
        for q in self.queries:
            q = (q or "").strip()
            if q and q not in out:
                out.append(q)
        if self.mode is SearchMode.KEEP:
            out = [question]
        return out or [question]

    def to_dict(self) -> dict:
        return {"mode": self.mode.value, "queries": list(self.queries),
                "graph": self.graph, "rounds": self.rounds, "top_k": self.top_k,
                "reason": self.reason, "source": self.source}


# --------------------------------------------------------------------- stage 3
@dataclass(slots=True)
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    query: str = ""
    source: str = "bm25"                # bm25 | graph | graph-hop

    def to_dict(self, with_text: bool = False) -> dict:
        out = {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
               "title": self.title, "score": round(self.score, 4),
               "query": self.query, "source": self.source}
        if with_text:
            out["text"] = self.text
        return out


@dataclass
class Evidence:
    """The evidence pool. Always iterated in `(-score, chunk_id)` order so two
    processes on the same input produce the same prompt."""

    passages: dict[str, Passage] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.passages)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self.passages

    def add(self, passage: Passage) -> bool:
        """Insert; keeps the higher-scoring duplicate. Returns True if added."""
        old = self.passages.get(passage.chunk_id)
        if old is not None and old.score >= passage.score:
            return False
        self.passages[passage.chunk_id] = passage
        return True

    def merge(self, passages: list[Passage], cap: int | None = None) -> int:
        added = sum(1 for p in passages if self.add(p))
        if cap is not None:
            self.cap(cap)
        return added

    def ordered(self) -> list[Passage]:
        return sorted(self.passages.values(), key=lambda p: (-p.score, p.chunk_id))

    def top(self, n: int | None = None) -> list[Passage]:
        items = self.ordered()
        return items if n is None else items[:n]

    def cap(self, n: int) -> None:
        if len(self.passages) <= n:
            return
        self.passages = {p.chunk_id: p for p in self.ordered()[:n]}

    def doc_ids(self) -> list[str]:
        return [p.doc_id for p in self.ordered()]

    def to_dict(self) -> list[dict]:
        return [p.to_dict() for p in self.ordered()]


@dataclass(slots=True)
class JudgeVerdict:
    """Local stopping decision for the adaptive loop (no LLM cost)."""

    sufficient: bool
    missing: str = ""
    next_query: str = ""
    reason: str = ""
    score: float = 0.0
    signals: dict = field(default_factory=dict)
    source: str = "coverage"            # coverage | llm | jev | stub

    def to_dict(self) -> dict:
        return {"sufficient": self.sufficient, "missing": self.missing,
                "next_query": self.next_query, "reason": self.reason,
                "score": round(self.score, 4), "signals": dict(self.signals),
                "source": self.source}


@dataclass
class SearchResult:
    evidence: Evidence
    rounds: int
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    queries: list[list[str]] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0


# --------------------------------------------------------------------- stage 4
@dataclass
class RerankResult:
    order: list[str]                    # chunk ids, best first
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    source: str = "local"               # local | llm | score
    llm_calls: int = 0


# --------------------------------------------------------------------- stage 5
@dataclass
class AnswerResult:
    answer: str = "UNKNOWN"
    confidence: float = 0.0
    sufficient: bool = True             # the answer model's own evidence verdict
    missing: str = ""
    facts: list[str] = field(default_factory=list)
    cited_docs: list[str] = field(default_factory=list)
    block_ids: dict[int, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {"answer": self.answer, "confidence": self.confidence,
                "sufficient": self.sufficient, "missing": self.missing,
                "facts": list(self.facts), "cited_docs": list(self.cited_docs),
                "block_n": len(self.block_ids), "error": self.error}


# --------------------------------------------------------------------- result
@dataclass
class PipelineResult:
    question: str
    answer: str
    cited_docs: list[str]
    evidence: Evidence
    route: RouteSignal
    budget: StageBudget
    plan: SearchPlan
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    answer_result: AnswerResult | None = None
    trace: list[dict] = field(default_factory=list)
    rounds: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0

    def to_record(self) -> dict:
        """Evaluation-facing view (no gold, no prompt text)."""
        return {
            "question": self.question,
            "pred": self.answer,
            "cited_docs": list(self.cited_docs),
            "retrieved_docs": self.evidence.doc_ids(),
            "n_evidence": len(self.evidence),
            "rounds": self.rounds,
            "llm_calls": self.llm_calls,
            "latency_ms": round(self.latency_ms, 1),
            "mode": self.plan.mode.value,
            "route": self.route.to_dict(),
            "plan": self.plan.to_dict(),
            "budget": self.budget.to_dict(),
            "verdicts": [v.to_dict() for v in self.verdicts],
            "ranking": list(self.ranking),
            "answer": self.answer_result.to_dict() if self.answer_result else None,
            "trace": self.trace,
        }
