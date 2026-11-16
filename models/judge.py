"""Evidence judges — the stopping rule of the adaptive search loop.

Contract:  judge(question, evidence) -> JudgeVerdict

`CoverageJudge`  local and deterministic (default): are the question's content
                 words and named entities present in the pooled evidence? It costs
                 no API call, which is what keeps the loop at two network calls.
`LocalJudgeAdapter` plugs in any local callable — this is where a JEV-type
                 classifier goes: a model that predicts evidence sufficiency.
`LLMJudge`       DeepSeek-based judge, kept as an explicit ablation.
`StubJudge`      deterministic double for tests.
"""
from __future__ import annotations

import math
from typing import Callable, Protocol

from agentic_rag_v2.config import (COVERAGE_BEST_PASSAGE_THRESHOLD,
                                   COVERAGE_ENTITY_THRESHOLD, COVERAGE_TOKEN_THRESHOLD,
                                   JUDGE_DIGEST_CHARS, JUDGE_DIGEST_TOP)
from agentic_rag_v2.llm import ChatClient
from agentic_rag_v2.prompts import JUDGE_SYSTEM
from agentic_rag_v2.retrieval.text import RELATION_WORDS, tokenize
from agentic_rag_v2.types import Evidence, JudgeVerdict


class EvidenceJudge(Protocol):
    name: str
    llm_calls: int

    def judge(self, question: str, evidence: Evidence, *,
              multihop: bool = False) -> JudgeVerdict: ...


def evidence_digest(evidence: Evidence, top: int = JUDGE_DIGEST_TOP,
                    chars: int = JUDGE_DIGEST_CHARS) -> str:
    """Score-ordered digest (same view the answer prompt gets by default)."""
    lines = []
    for i, p in enumerate(evidence.top(top), 1):
        lines.append(f"[{i}] {p.title} :: {p.text[:chars].replace(chr(10), ' ')}")
    return "\n".join(lines)


class CoverageJudge:
    """Local coverage check: content-word + entity presence, no model call.

    Deliberately explainable: the verdict carries the numbers that produced it,
    so a wrong "still insufficient" is visible in the trace instead of hidden in
    a model.
    """

    name = "coverage_judge"
    llm_calls = 0

    def __init__(self, retriever=None, *, token_threshold: float = COVERAGE_TOKEN_THRESHOLD,
                 entity_threshold: float = COVERAGE_ENTITY_THRESHOLD,
                 best_passage_threshold: float = COVERAGE_BEST_PASSAGE_THRESHOLD):
        self.retriever = retriever
        self.token_threshold = token_threshold
        self.entity_threshold = entity_threshold
        self.best_passage_threshold = best_passage_threshold

    def judge(self, question: str, evidence: Evidence, *,
              multihop: bool = False) -> JudgeVerdict:
        q_tokens = [t for t in tokenize(question) if t not in RELATION_WORDS]
        if not q_tokens:
            q_tokens = tokenize(question)
        if not q_tokens or not evidence.passages:
            return JudgeVerdict(sufficient=False, next_query="", missing=question,
                                reason="no content tokens or no evidence",
                                score=0.0, source="coverage")

        q_set = set(q_tokens)
        pooled: set[str] = set()
        best_passage_cov = 0.0
        top = evidence.top(3)
        top_tokens: set[str] = set()
        for passage in evidence.ordered():
            tokens = set(tokenize(f"{passage.title} {passage.text}"))
            pooled |= tokens
            best_passage_cov = max(best_passage_cov, len(q_set & tokens) / len(q_set))
        for passage in top:
            top_tokens |= set(tokenize(f"{passage.title} {passage.text}"))

        missing_tokens = [t for t in q_tokens if t not in pooled]
        token_cov = (len(q_set) - len(missing_tokens)) / len(q_set)

        seeds: list[str] = []
        entity_cov = 1.0
        bridge = ""
        bridge_resolved = True
        if self.retriever is not None:
            seeds = [e for e, _ in self.retriever.link_entities(question)[:5]]
            if seeds:
                present = 0
                for seed in seeds:
                    chunk_ids = self.retriever.graph.entity_chunks.get(seed) or []
                    if any(cid in evidence for cid in chunk_ids):
                        present += 1
                entity_cov = present / len(seeds)
            if multihop:
                bridge_resolved, bridge = self._bridge_status(question, evidence)

        relation_cov = self._relation_coverage(question, evidence, seeds)

        score = 0.45 * token_cov + 0.25 * best_passage_cov + 0.15 * entity_cov + 0.15 * relation_cov
        sufficient = (token_cov >= self.token_threshold
                      and best_passage_cov >= self.best_passage_threshold
                      and entity_cov >= self.entity_threshold
                      and relation_cov >= 1.0)
        if multihop and not bridge_resolved:
            sufficient = False

        next_query = ""
        if not sufficient:
            titles = [p.title for p in evidence.top(2) if p.title]
            # tokens of the question that the strongest passage does NOT contain:
            # these describe the still-missing relation, which is what the second
            # hop has to supply
            relation_gap = [t for t in tokenize(question) if t not in top_tokens][:3]
            parts = [p for p in [bridge or (titles[0] if titles else ""),
                                 *relation_gap] if p]
            next_query = " ".join(parts).strip() or " ".join(titles + missing_tokens[:3]).strip()

        return JudgeVerdict(
            sufficient=sufficient,
            missing=" ".join(missing_tokens[:6]),
            next_query=next_query,
            reason=(f"coverage token={token_cov:.2f} best_passage={best_passage_cov:.2f} "
                    f"entity={entity_cov:.2f} relation={relation_cov:.2f} "
                    f"bridge={bridge or '-'}{'' if bridge_resolved else ' (unresolved)'}"),
            score=score,
            signals={"token_coverage": round(token_cov, 4),
                     "best_passage_coverage": round(best_passage_cov, 4),
                     "entity_coverage": round(entity_cov, 4),
                     "relation_coverage": round(relation_cov, 4),
                     "missing_tokens": missing_tokens[:8],
                     "bridge_entity": bridge,
                     "bridge_resolved": bridge_resolved,
                     "seeds": seeds},
            source="coverage",
        )

    @staticmethod
    def _relation_coverage(question: str, evidence: Evidence, seeds: list[str]) -> float:
        """Is the *asked relation* stated next to a question entity?

        Word coverage alone cannot tell "both hops' entities appear somewhere in
        the pool" from "the chain is established", so the judge used to declare
        victory after the first hop. Requiring one passage to mention a question
        entity AND the question's relation words is what v1's LLM assess did
        implicitly, and it is the whole difference between stopping at round 1
        and going after the second hop.
        """
        rel_tokens = [t for t in tokenize(question) if t in RELATION_WORDS]
        if not rel_tokens:
            return 1.0
        need = max(1, math.ceil(0.6 * len(rel_tokens)))
        best = 0
        for passage in evidence.ordered():
            tokens = set(tokenize(f"{passage.title} {passage.text}"))
            mentions_seed = any(
                any(word in tokens for word in seed.split()) for seed in seeds
            ) if seeds else True
            if mentions_seed:
                best = max(best, sum(1 for t in rel_tokens if t in tokens))
        return min(1.0, best / need)

    def _bridge_status(self, question: str, evidence: Evidence) -> tuple[bool, str]:
        """Did the chain reach the second hop?

        Candidate bridges are entities of the strongest passages that the question
        does not name (e.g. "styx" for 'the band that recorded Renegade'). The chain
        counts as resolved when a *different* pooled passage is titled after that
        entity — that is the document holding the second-hop fact. A looser
        fallback accepts an entity mentioned by two documents in the pool.
        """
        q_text = " ".join(tokenize(question))
        by_id = getattr(self.retriever, "by_id", {})
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for passage in evidence.top(3):
            for ent in self.retriever.graph.chunk_entities.get(passage.chunk_id, ()):
                if len(ent) < 4 or ent in RELATION_WORDS or ent in q_text or ent in seen:
                    continue
                seen.add(ent)
                candidates.append((ent, passage.chunk_id))
        # most useful first: an entity whose own document is still missing from the
        # pool is exactly the second hop we have not fetched yet
        candidates.sort(key=lambda kv: (0 if self._bridge_doc_missing(kv[0], evidence) else 1,
                                        -len(kv[0].split()), -len(kv[0])))
        candidates = candidates[:8]

        for ent, src in candidates:
            for passage in evidence.ordered():
                if passage.chunk_id != src and ent in passage.title.lower():
                    return True, ent
        for ent, _src in candidates:
            chunk_ids = self.retriever.graph.entity_chunks.get(ent) or []
            docs = {by_id[c].doc_id for c in chunk_ids if c in evidence and c in by_id}
            if len(docs) >= 2:
                return True, ent
        return (not candidates), (candidates[0][0] if candidates else "")

    def _bridge_doc_missing(self, entity: str, evidence: Evidence) -> bool:
        """Is there a document about this entity that the pool does not contain?"""
        by_id = getattr(self.retriever, "by_id", {})
        titled = [cid for cid in (self.retriever.graph.entity_chunks.get(entity) or [])[:30]
                  if cid in by_id and entity in by_id[cid].title.lower()]
        return bool(titled) and any(cid not in evidence for cid in titled)


class LLMJudge:
    """LLM evidence controller (ablation: costs one extra network call)."""

    name = "llm_judge"

    def __init__(self, client: ChatClient, *, tag: str = "judge"):
        self.client = client
        self.tag = tag
        self.llm_calls = 0

    def judge(self, question: str, evidence: Evidence, *,
              multihop: bool = False) -> JudgeVerdict:
        prompt = f"QUESTION:\n{question}\n\nEVIDENCE SO FAR:\n{evidence_digest(evidence)}"
        try:
            obj = self.client.json_call(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=400, tag=self.tag)
        except Exception as e:  # noqa: BLE001 - a failed judge must not lose the run
            return JudgeVerdict(sufficient=True, score=0.0,
                                reason=f"judge error: {type(e).__name__}: {e}",
                                source="error")
        self.llm_calls += 1
        return JudgeVerdict(
            sufficient=bool(obj.get("sufficient")),
            missing=str(obj.get("missing") or ""),
            next_query=str(obj.get("next_query") or "").strip(),
            reason=str(obj.get("reason") or ""),
            source="llm",
        )


class LocalJudgeAdapter:
    """Adapter for a local classifier (the planned JEV-type judge).

    The callable receives (question, evidence) and returns a dict with any of
    `sufficient`, `missing`, `next_query`, `reason`, `score`. No LLM call is made,
    so `llm_calls` stays 0.
    """

    name = "local_judge"

    def __init__(self, fn: Callable[[str, Evidence], dict], *, tag: str = "jev"):
        self.fn = fn
        self.tag = tag
        self.llm_calls = 0

    def judge(self, question: str, evidence: Evidence, *,
              multihop: bool = False) -> JudgeVerdict:
        raw = self.fn(question, evidence) or {}
        return JudgeVerdict(
            sufficient=bool(raw.get("sufficient")),
            missing=str(raw.get("missing") or ""),
            next_query=str(raw.get("next_query") or "").strip(),
            reason=str(raw.get("reason") or ""),
            score=float(raw.get("score") or 0.0),
            source="jev",
        )


class StubJudge:
    """Always 'insufficient, no follow-up' — ends the loop after one round."""

    name = "stub_judge"
    llm_calls = 0

    def __init__(self, sufficient: bool = False, next_query: str = ""):
        self._sufficient = sufficient
        self._next_query = next_query

    def judge(self, question: str, evidence: Evidence, *,
              multihop: bool = False) -> JudgeVerdict:
        return JudgeVerdict(sufficient=self._sufficient, next_query=self._next_query,
                            reason="stub", source="stub")
