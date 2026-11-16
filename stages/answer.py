"""Stage 5 — answer: one DeepSeek call delivers sufficiency + answer + citations.

The model both judges the evidence ("sufficient") and answers from it, so the
pipeline keeps a fixed budget of two network calls per question. The citation
organisation is the mapping from the numbers it returns to the document ids the
evaluator scores.

v3 adds an answer-side repair pass (docs/PLAN_2026-11-16.md §1): the model
nominates alternative surface forms and a deterministic verifier keeps only a
form that appears verbatim in the evidence it was given. Nothing here looks at
gold data — the verifier only checks the prompt the model itself saw.
"""
from __future__ import annotations

import re

from agentic_rag_v2.config import ANSWER_MAX_CHARS, ANSWER_TOP_N
from agentic_rag_v2.llm import ChatClient
from agentic_rag_v2.prompts import (ANSWER_SYSTEM, ANSWER_SYSTEM_V1,
                                    ANSWER_SYSTEM_V3, COMPARE_BLOCK, DIRECT_SYSTEM)
from agentic_rag_v2.types import AnswerResult, Evidence, SearchMode

MAX_CHUNKS_PER_DOC = 2      # no single document may crowd out the others
QUOTES = "\"'`\u2018\u2019\u201c\u201d"
TRAILING = " \t\r\n.,;:!?"


def normalize_answer(answer: str) -> str:
    """Local cleanup of an extracted span (no gold knowledge involved).

    Collapses date ranges to their end date, strips surrounding quotes, a leading
    article/preposition and trailing punctuation — the near-miss bucket found in
    v1 error analysis.
    """
    a = re.sub(r"\s+", " ", (answer or "").strip())
    a = a.strip(QUOTES).strip()
    a = a.rstrip(TRAILING).strip(QUOTES).strip()
    m = re.match(r"^(\d{1,2})\s*[\u2013\u2014-]\s*(\d{1,2})\s+([A-Za-z]+\s+\d{4})$", a)
    if m:
        a = f"{m.group(2)} {m.group(3)}"
    return re.sub(r"^(the|a|an|on|in|at)\s+", "", a, flags=re.I).strip()


def _flat(text: str) -> str:
    """Matching form: lowercase, punctuation to spaces, single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def appears_verbatim(span: str, evidence_text: str) -> bool:
    needle = _flat(span)
    return bool(needle) and needle in _flat(evidence_text)


def verify_answer(primary: str, candidates: list[str], evidence_text: str,
                  ) -> tuple[str, str]:
    """Keep a span the evidence actually contains; returns (answer, source).

    v2 trusted a single string, so a composed surface form the passage never
    states (e.g. "Musashino, Tokyo, Japan") could not be repaired. The model now
    lists alternatives and we take the longest one that is literally present.
    """
    primary = (primary or "").strip()
    usable = [c.strip() for c in candidates if c and c.strip()]
    if primary and primary.upper() != "UNKNOWN" and appears_verbatim(primary, evidence_text):
        return primary, "primary"
    verified = [c for c in usable if appears_verbatim(c, evidence_text)]
    if verified:
        return max(verified, key=len), "candidate"
    if primary and primary.upper() != "UNKNOWN":
        return primary, "fallback"
    if usable:
        return max(usable, key=len), "candidate"
    return "UNKNOWN", "fallback"


def build_evidence_block(evidence: Evidence, order: list[str] | None = None,
                         *, top_n: int = ANSWER_TOP_N,
                         max_chars: int = ANSWER_MAX_CHARS) -> tuple[str, dict[int, str]]:
    """Render the numbered evidence block; returns (block, {number: chunk_id})."""
    items = [evidence.passages[cid] for cid in order if cid in evidence.passages] \
        if order else evidence.ordered()

    lines: list[str] = []
    num_map: dict[int, str] = {}
    per_doc: dict[str, int] = {}
    used = 0
    for passage in items:
        if len(num_map) >= top_n:
            break
        if per_doc.get(passage.doc_id, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        text = passage.text.replace("\n", " ")
        if used and used + len(text) > max_chars:
            break
        used += len(text)
        per_doc[passage.doc_id] = per_doc.get(passage.doc_id, 0) + 1
        number = len(num_map) + 1
        num_map[number] = passage.chunk_id
        lines.append(f"[{number}] ({passage.title}) {text}")
    return "\n\n".join(lines), num_map


def bridge_lines(retriever, evidence: Evidence, block_ids: list[str], limit: int = 8) -> str:
    """Bridge entities shared by the passages shown to the answer model.

    Numbering matches the answer block, so the model can chain "passage 1 <-> 3"
    instead of guessing which entities are linked (v1 measured this as part of
    its best configuration).
    """
    if retriever is None or len(block_ids) < 2:
        return ""
    shared = retriever.graph.shared_entities(block_ids, limit=limit)
    if not shared:
        return ""
    lines = [f"- '{e}' links passages {', '.join(map(str, ids))}" for e, ids in shared[:limit]]
    return "GRAPH LINKS (entities shared across passages):\n" + "\n".join(lines)


def answer(question: str, evidence: Evidence, client: ChatClient,
           *, order: list[str] | None = None, top_n: int = ANSWER_TOP_N,
           max_chars: int = ANSWER_MAX_CHARS, retriever=None, relations=None,
           prompt: str = "v3", mode: SearchMode | None = None,
           tag: str = "answer") -> AnswerResult:
    block, num_map = build_evidence_block(evidence, order, top_n=top_n, max_chars=max_chars)
    if not block:
        return AnswerResult(answer="UNKNOWN", sufficient=False,
                            missing="no evidence retrieved", block_ids={})

    block_ids = [num_map[i] for i in sorted(num_map)]
    links = bridge_lines(retriever, evidence, block_ids[:8])
    prompt_text = f"QUESTION:\n{question}\n\nEVIDENCE:\n{block}"
    if links:
        prompt_text += f"\n\n{links}\nUse these links to decide which passages form the chain."
    if relations is not None:
        rel_lines = relations.prompt_lines(evidence, block_ids[:10])
        if rel_lines:
            prompt_text += f"\n\n{rel_lines}"
    system = _system_prompt(prompt)
    if mode is SearchMode.COMPARE and prompt != "v1":
        system = f"{system}\n\n{COMPARE_BLOCK}"

    try:
        obj = client.json_call(
            [{"role": "system", "content": system},
             {"role": "user", "content": prompt_text}],
            max_tokens=800, tag=tag)
    except Exception as e:  # noqa: BLE001 - a failed answer must still be scored
        return AnswerResult(answer="UNKNOWN", sufficient=False, block_ids=num_map,
                            error=f"{type(e).__name__}: {e}")

    candidates = [normalize_answer(str(c)) for c in (obj.get("answer_candidates") or [])][:3]
    candidates = [c for c in candidates if c and c.upper() != "UNKNOWN"]
    primary = normalize_answer(str(obj.get("answer") or "UNKNOWN")) or "UNKNOWN"
    text, source = verify_answer(primary, candidates,
                                 " ".join(p.text for p in evidence.ordered()))

    cited: list[str] = []
    for raw in obj.get("evidence_ids") or []:
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        cid = num_map.get(number)
        if cid is None:
            continue
        doc_id = evidence.passages[cid].doc_id
        if doc_id not in cited:
            cited.append(doc_id)

    try:
        confidence = float(obj.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    sufficient = bool(obj.get("sufficient", text.upper() != "UNKNOWN"))
    return AnswerResult(
        answer=text,
        confidence=round(confidence, 4),
        sufficient=sufficient,
        missing=str(obj.get("missing") or ""),
        facts=[str(f) for f in (obj.get("relevant_facts") or [])][:6],
        candidates=candidates,
        answer_source=source,
        cited_docs=cited,
        block_ids=num_map,
    )


def _system_prompt(prompt: str) -> str:
    if prompt == "v1":
        return ANSWER_SYSTEM_V1
    if prompt == "v2":
        return ANSWER_SYSTEM
    return ANSWER_SYSTEM_V3


def answer_direct(question: str, client: ChatClient, *, tag: str = "direct") -> AnswerResult:
    """No-retrieval lower bound, same output contract as the full pipeline."""
    try:
        obj = client.json_call(
            [{"role": "system", "content": DIRECT_SYSTEM},
             {"role": "user", "content": question}], max_tokens=300, tag=tag)
        text = normalize_answer(str(obj.get("answer") or "UNKNOWN")) or "UNKNOWN"
        return AnswerResult(answer=text, sufficient=False)
    except Exception as e:  # noqa: BLE001
        return AnswerResult(answer="UNKNOWN", sufficient=False,
                            error=f"{type(e).__name__}: {e}")
