"""Answer / evidence metrics (SQuAD-style normalisation).

Kept identical to the v1 harness so v2 numbers are directly comparable to the
published baseline.
"""
from __future__ import annotations

import re
import string
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = str(s or "").lower()
    s = s.translate(PUNCT_TABLE)
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def best_over_refs(pred: str, refs: list[str]) -> tuple[float, float]:
    if not refs:
        return 0.0, 0.0
    return max(em(pred, r) for r in refs), max(f1(pred, r) for r in refs)


def containment(pred: str, gold: str) -> float:
    """Lenient production check: is the gold surface form inside the answer?"""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not g:
        return 0.0
    return float(g in p)


def answer_metrics(pred: str, gold: str, aliases: list[str] | None = None) -> dict:
    refs = [gold] + list(aliases or [])
    e, f = best_over_refs(pred, refs)
    return {"em": e, "f1": f, "containment": max(containment(pred, r) for r in refs)}


def evidence_recall(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0
    got = set(retrieved_doc_ids)
    return len(got & set(gold_doc_ids)) / len(set(gold_doc_ids))


def evidence_precision(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not retrieved_doc_ids:
        return 0.0
    got, gold = set(retrieved_doc_ids), set(gold_doc_ids)
    return len(got & gold) / len(got)


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = ["em", "f1", "containment", "evidence_recall_final", "evidence_recall_retrieved"]
    out: dict = {"n": len(records)}
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    for k in ["rounds", "llm_calls", "latency_ms"]:
        vals = [r.get(k, 0) for r in records]
        if vals:
            out[f"avg_{k}"] = round(sum(vals) / len(vals), 1)
    out["errors"] = sum(1 for r in records if r.get("error"))
    return out
