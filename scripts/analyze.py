"""Paired analysis of a v2 evaluation batch.

    python agentic_rag_v2/scripts/analyze.py --tag v2core
    python agentic_rag_v2/scripts/analyze.py --tag v2core --base v2_full --vs v2_no_rerank --diffs

Reports EM/F1, per-question wins/losses against the base system, an exact
McNemar (binomial) p-value, and the ranking quality of the evidence order
(gold-document recall@k), which is answer-model independent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_v2.config import RESULTS, corpus_path, gold_path  # noqa: E402
from agentic_rag_v2.eval.metrics import summarize  # noqa: E402


def load_system(tag: str, name: str) -> dict[str, dict]:
    path = RESULTS / tag / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return {rec["qid"]: rec for rec in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def doc_recall(record: dict, gold_docs: set[str], doc_of: dict[str, str], k: int | None = None) -> float:
    order = record.get("ranking") or record.get("retrieved_docs") or []
    if k is not None:
        order = order[:k]
    docs = {doc_of.get(c) for c in order} - {None}
    return len(docs & gold_docs) / len(gold_docs) if gold_docs else 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="v2_full")
    ap.add_argument("--vs", default="", help="comma separated systems; default = all in the batch")
    ap.add_argument("--diffs", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="cutoff for ranking recall")
    args = ap.parse_args(argv)

    from agentic_rag_v2.data import load_chunks

    gold = json.loads(gold_path().read_text(encoding="utf-8"))
    doc_of = {c.chunk_id: c.doc_id for c in load_chunks()}
    summary = json.loads((RESULTS / args.tag / "summary.json").read_text(encoding="utf-8"))
    names = [s["system"] for s in summary["summary"]]
    systems = [s.strip() for s in args.vs.split(",") if s.strip()] or names

    base = load_system(args.tag, args.base) if args.base in names else None
    print(f"batch={args.tag}  base={args.base}  n={len(base) if base else '-'}")
    header = f"{'system':<16}{'EM':>7}{'F1':>7}{'rec_fin':>9}{'rec_ret':>9}{f'rank@{args.k}':>9}{'calls':>7}{'lat_ms':>9}  paired vs base"
    print(header)
    print("-" * len(header))

    for name in systems:
        records = load_system(args.tag, name)
        stats = summarize(list(records.values()))
        rank = sum(doc_recall(r, set(gold[q]["gold_doc_ids"]), doc_of, args.k)
                   for q, r in records.items()) / max(len(records), 1)
        line = (f"{name:<16}{stats.get('em', 0):>7.3f}{stats.get('f1', 0):>7.3f}"
                f"{stats.get('evidence_recall_final', 0):>9.3f}"
                f"{stats.get('evidence_recall_retrieved', 0):>9.3f}"
                f"{rank:>9.3f}{stats.get('avg_llm_calls', 0):>7.1f}"
                f"{stats.get('avg_latency_ms', 0):>9.0f}")
        if base and name != args.base:
            wins = sum(1 for q in base if records[q]["em"] > base[q]["em"])
            losses = sum(1 for q in base if records[q]["em"] < base[q]["em"])
            line += f"  +{wins}/-{losses}  p={exact_mcnemar(wins, losses):.3f}"
        print(line)

    if args.diffs and base:
        for name in systems:
            if name == args.base:
                continue
            records = load_system(args.tag, name)
            changed = [(q, base[q], records[q]) for q in base if base[q]["em"] != records[q]["em"]]
            if not changed:
                continue
            print(f"\n--- {args.base} vs {name}: {len(changed)} changed answers")
            for qid, a, b in sorted(changed):
                arrow = "WIN " if b["em"] > a["em"] else "LOSS"
                print(f"{arrow} {qid:8} {a['pred'][:34]!r:36} -> {b['pred'][:34]!r:36} "
                      f"gold={a['gold'][:24]!r}")


if __name__ == "__main__":
    main()
