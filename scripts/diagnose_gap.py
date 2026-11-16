"""Decompose the EM gap between two systems into its causes.

    python agentic_rag_v2/scripts/diagnose_gap.py --base results/.../a.jsonl --vs results/.../b.jsonl

A raw EM difference says nothing about *why*. This splits every question into
three buckets, which is what decides where to spend effort:

    perfect recall + right answer     -> retrieval and answer both worked
    perfect recall + wrong answer     -> retrieval fine, the ANSWERER lost it
    partial recall                    -> retrieval never had the evidence

It also reports UNKNOWN rates, rounds used, evidence-pool size and the answer
block size, so "the other pipeline just retrieves more" is a measurement instead
of a guess.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return {rec["qid"]: rec for rec in
            (json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip())}


def block_stats(record: dict) -> tuple[int | None, int | None]:
    """(block passages, distinct docs in block) for either pipeline's schema."""
    for entry in record.get("trace") or []:
        if entry.get("node") == "answer":
            return entry.get("block_n"), len(entry.get("block_docs") or [])
    answer = record.get("answer") or {}
    if "block_n" in answer:
        return answer.get("block_n"), None
    return None, None


def buckets(records: dict[str, dict]) -> dict:
    perfect = [r for r in records.values() if r["evidence_recall_retrieved"] >= 0.999]
    partial = [r for r in records.values() if r["evidence_recall_retrieved"] < 0.999]
    em_perfect = sum(r["em"] for r in perfect) / len(perfect) if perfect else 0.0
    return {
        "n": len(records),
        "em": round(sum(r["em"] for r in records.values()) / max(len(records), 1), 4),
        "f1": round(sum(r["f1"] for r in records.values()) / max(len(records), 1), 4),
        "unknown": sum(1 for r in records.values() if str(r["pred"]).strip().upper() == "UNKNOWN"),
        "recall": round(sum(r["evidence_recall_retrieved"] for r in records.values()) / max(len(records), 1), 4),
        "perfect_recall": len(perfect),
        "partial_recall": len(partial),
        "em_when_perfect": round(em_perfect, 4),
        "lost_by_retrieval": len(partial),
        "lost_by_answer": sum(1 for r in perfect if r["em"] < 1.0),
        "calls": round(sum(r.get("llm_calls", 0) for r in records.values()) / max(len(records), 1), 2),
        "latency_ms": round(sum(r.get("latency_ms", 0) for r in records.values()) / max(len(records), 1), 1),
        "rounds": dict(sorted(Counter(r.get("rounds", 0) for r in records.values()).items())),
        "pool_median": statistics.median([r.get("n_evidence", 0) for r in records.values()]),
        "block_median": statistics.median([b for b, _ in map(block_stats, records.values())
                                           if b is not None] or [0]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base system .jsonl")
    ap.add_argument("--vs", required=True, help="path to comparison system .jsonl")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--vs-name", default="vs")
    ap.add_argument("--show", type=int, default=12, help="how many answer-side losses to list")
    args = ap.parse_args(argv)

    base, vs = load(args.base), load(args.vs)
    common = sorted(set(base) & set(vs))
    base = {q: base[q] for q in common}
    vs = {q: vs[q] for q in common}

    print(f"paired on {len(common)} questions\n")
    rows = [(args.base_name, buckets(base)), (args.vs_name, buckets(vs))]
    keys = ["em", "f1", "unknown", "recall", "perfect_recall", "em_when_perfect",
            "lost_by_retrieval", "lost_by_answer", "calls", "latency_ms", "pool_median", "block_median"]
    print(f"{'metric':<20}" + "".join(f"{name:>16}" for name, _ in rows))
    for key in keys:
        print(f"{key:<20}" + "".join(f"{str(b[key]):>16}" for _, b in rows))
    for name, b in rows:
        print(f"{name} rounds: {b['rounds']}")

    wins = sum(1 for q in common if vs[q]["em"] > base[q]["em"])
    losses = sum(1 for q in common if vs[q]["em"] < base[q]["em"])
    print(f"\npaired: +{wins}/-{losses}")

    answer_side = [q for q in common
                   if base[q]["em"] > vs[q]["em"] and vs[q]["evidence_recall_retrieved"] >= 0.999]
    print(f"\nlost by the answerer while all gold docs were retrieved: {len(answer_side)}")
    for q in answer_side[: args.show]:
        print(f"  {q:<8} {args.base_name}={base[q]['pred'][:30]!r:<32} "
              f"{args.vs_name}={vs[q]['pred'][:30]!r:<32} gold={base[q]['gold'][:26]!r}")

    retrieval_side = [q for q in common
                      if base[q]["em"] > vs[q]["em"] and vs[q]["evidence_recall_retrieved"] < 0.999]
    print(f"\nlost because the evidence was never retrieved: {len(retrieval_side)}")
    for q in retrieval_side[: args.show]:
        print(f"  {q:<8} recall {vs[q]['evidence_recall_retrieved']:.2f} -> "
              f"{base[q]['evidence_recall_retrieved']:.2f}   pred={vs[q]['pred'][:28]!r} "
              f"gold={vs[q]['gold'][:24]!r}  rounds={vs[q].get('rounds')}")


if __name__ == "__main__":
    main()
