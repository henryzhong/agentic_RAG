"""Stage 4 — rerank the evidence pool with a local cross-encoder.

The default is local (`strategy="local"`): a CPU cross-encoder scores every pooled
passage against the question in one batched pass (~2 ms per passage), so ordering
costs no API call and no network latency.

`strategy="llm"` keeps a DeepSeek ordering as an explicit ablation, and
`strategy="score"` returns retrieval order (the v1 baseline behaviour).
"""
from __future__ import annotations

import math

from agentic_rag_v2.llm import ChatClient
from agentic_rag_v2.models.reranker import LocalReranker
from agentic_rag_v2.prompts import RERANK_SYSTEM
from agentic_rag_v2.types import Evidence, RerankResult

RERANK_EXCERPT_CHARS = 400


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rerank(question: str, evidence: Evidence, *,
           reranker: LocalReranker | None = None,
           client: ChatClient | None = None,
           strategy: str = "local") -> RerankResult:
    ordered = evidence.ordered()
    fallback = [p.chunk_id for p in ordered]
    if len(ordered) <= 1 or strategy == "score":
        return RerankResult(order=fallback, source="score")

    if strategy == "local":
        return _local(question, ordered, reranker, fallback)
    if strategy == "llm":
        return _llm(question, ordered, client, fallback)
    raise ValueError(f"unknown rerank strategy: {strategy}")


def _local(question: str, ordered, reranker: LocalReranker | None,
           fallback: list[str]) -> RerankResult:
    scores = reranker.rerank(question, [p.text for p in ordered]) if reranker else None
    if not scores or len(scores) != len(ordered):
        return RerankResult(order=fallback, reason="local reranker unavailable",
                            source="score")
    ranked = sorted(zip(ordered, scores), key=lambda kv: (-kv[1], kv[0].chunk_id))
    return RerankResult(
        order=[p.chunk_id for p, _ in ranked],
        scores={p.chunk_id: round(_sigmoid(float(s)), 4) for p, s in ranked},
        reason="local cross-encoder", source="local")


def _llm(question: str, ordered, client: ChatClient | None,
         fallback: list[str]) -> RerankResult:
    if client is None:
        return RerankResult(order=fallback, source="score")
    listing = "\n\n".join(
        f"[{i}] ({p.title}) {p.text[:RERANK_EXCERPT_CHARS].replace(chr(10), ' ')}"
        for i, p in enumerate(ordered, 1))
    try:
        obj = client.json_call(
            [{"role": "system", "content": RERANK_SYSTEM},
             {"role": "user", "content": f"QUESTION:\n{question}\n\nPASSAGES:\n{listing}"}],
            max_tokens=600, tag="rerank")
    except Exception as exc:  # noqa: BLE001 - never lose evidence to a failed call
        return RerankResult(order=fallback, reason=f"rerank call failed: {exc}",
                            source="score")

    chosen: list[str] = []
    for raw in obj.get("ranking") or []:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(ordered):
            cid = ordered[idx - 1].chunk_id
            if cid not in chosen:
                chosen.append(cid)
    if not chosen:
        return RerankResult(order=fallback, reason="no usable ranking", source="score")
    chosen.extend(cid for cid in fallback if cid not in chosen)
    return RerankResult(order=chosen, reason=str(obj.get("reason") or ""),
                        source="llm", llm_calls=1)
