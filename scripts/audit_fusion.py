"""Measure what the duplicate-vote fusion bug actually did to the ranking.

    python agentic_rag_v2/scripts/audit_fusion.py \
        --records agentic_rag_v2/results/audit_rerun/v2_full.jsonl

Before the fix, `_fuse` added `ROUTE_WEIGHTS[route] * normalised[cid]` once per
(query, chunk) hit, so a chunk that three of the planner's queries found was
counted three times **by the same route** and outranked chunks that a different
route found once. This script replays each stored plan's queries through the real
routes, then compares the old accumulator with the current one.

The old accumulator is a faithful re-implementation of the pre-fix loop (iterate
the raw hit list, not the per-chunk reduction); it is kept here only so the
documented impact numbers can be regenerated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_v2 import config                                   # noqa: E402
from agentic_rag_v2.data import load_chunks                          # noqa: E402
from agentic_rag_v2.retrieval import HybridRetriever                 # noqa: E402
from agentic_rag_v2.retrieval.vector import SemanticRetriever, build_index  # noqa: E402
from agentic_rag_v2.models.embedder import build_embedder            # noqa: E402
from agentic_rag_v2.stages.search import (_fuse, _normalise, _route_tasks,  # noqa: E402
                                          _run_tasks, ROUTE_WEIGHTS,
                                          TITLE_BOOST_IS_ABSOLUTE)
from agentic_rag_v2.types import Evidence, SearchMode, SearchPlan     # noqa: E402


def old_fuse(route_results: dict, exclude: set[str]) -> dict[str, float]:
    """Pre-fix loop: one vote per (query, chunk) hit instead of per chunk."""
    fused: dict[str, float] = {}
    for route, hits in route_results.items():
        if not hits:
            continue
        normalised = _normalise(route, hits)
        for _query, cid, _score in hits:
            if cid in exclude:
                continue
            contribution = (normalised[cid] if (TITLE_BOOST_IS_ABSOLUTE and route == "title")
                            else ROUTE_WEIGHTS[route] * normalised[cid])
            fused[cid] = fused.get(cid, 0.0) + contribution
    return fused


def _top(scores: dict[str, float], k: int) -> list[str]:
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(ROOT / "agentic_rag_v2" / "results"
                                            / "audit_rerun" / "v2_full.jsonl"))
    ap.add_argument("--backend", default=config.EMBED_BACKEND)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    chunks = load_chunks()
    retriever = HybridRetriever(chunks)
    embedder = build_embedder(args.backend)
    index, stats = build_index(chunks, embedder)
    semantic = SemanticRetriever(index, embedder)
    print(f"[semantic:{args.backend}] cached={stats['cached']} vectors={stats['vectors']}",
          flush=True)

    records = [json.loads(line) for line in
               Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    totals = {"questions": 0, "dup_questions": 0, "dup_pairs": 0, "hit_pairs": 0,
              "top1_changed": 0, "topk_sets_differ": 0, "order_changed": 0,
              "pool_changed": 0}
    examples: list[dict] = []

    for rec in records:
        plan_dict = rec.get("plan") or {}
        queries = plan_dict.get("queries") or [rec["question"]]
        plan = SearchPlan(mode=SearchMode.coerce(plan_dict.get("mode"), default=SearchMode.KEEP),
                          queries=list(queries), graph=bool(plan_dict.get("graph")),
                          rounds=int(plan_dict.get("rounds") or 1),
                          top_k=int(plan_dict.get("top_k") or config.DEFAULT_TOP_K))
        tasks = _route_tasks(queries, plan.graph, Evidence(), True)
        vectors = semantic.embed_many(queries)
        route_results = _run_tasks(tasks, retriever, plan, Evidence(), True,
                                   semantic=semantic, dense_vectors=vectors)

        dup_pairs = 0
        hit_pairs = 0
        for route, hits in route_results.items():
            counts: dict[str, int] = {}
            for _q, cid, _s in hits:
                counts[cid] = counts.get(cid, 0) + 1
            hit_pairs += len(hits)
            dup_pairs += sum(n - 1 for n in counts.values() if n > 1)

        new_scores, _ = _fuse(route_results, exclude=set())
        old_scores = old_fuse(route_results, exclude=set())
        k = plan.top_k * 2
        new_top, old_top = _top(new_scores, k), _top(old_scores, k)

        totals["questions"] += 1
        totals["hit_pairs"] += hit_pairs
        totals["dup_pairs"] += dup_pairs
        if dup_pairs:
            totals["dup_questions"] += 1
        if old_top[:1] != new_top[:1]:
            totals["top1_changed"] += 1
        if set(old_top) != set(new_top):
            totals["topk_sets_differ"] += 1
        if old_top != new_top:
            totals["order_changed"] += 1
        if set(old_scores) != set(new_scores):
            totals["pool_changed"] += 1
        if dup_pairs and len(examples) < 5:
            examples.append({
                "qid": rec["qid"], "queries": queries,
                "duplicate_votes": dup_pairs, "hit_pairs": hit_pairs,
                "top1_before": old_top[:1], "top1_after": new_top[:1],
            })

    n = totals["questions"] or 1
    print(json.dumps({
        **totals,
        "dup_questions_pct": round(100 * totals["dup_questions"] / n, 1),
        "dup_share_of_hits_pct": round(100 * totals["dup_pairs"] / max(1, totals["hit_pairs"]), 1),
        "top1_changed_pct": round(100 * totals["top1_changed"] / n, 1),
        "topk_sets_differ_pct": round(100 * totals["topk_sets_differ"] / n, 1),
        "order_changed_pct": round(100 * totals["order_changed"] / n, 1),
    }, indent=1, ensure_ascii=False))
    print(json.dumps(examples, indent=1, ensure_ascii=False))
    out = ROOT / "agentic_rag_v2" / "work" / "fusion_audit.json"
    out.write_text(json.dumps({"totals": totals, "examples": examples},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
