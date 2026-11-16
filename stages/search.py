"""Stage 3 — search: multi-route parallel retrieval with controlled adaptivity.

One round:
    act      run every (query, route) task in parallel
             routes: bm25 | dense (E5) | title-anchor | graph (GraphRAG) | graph-hop
    observe  normalise each route, fuse, add the new passages to the pool, cap it
    judge    LOCAL coverage judge (no API call) decides whether another round can
             still add something; only then does round 2 run

The loop therefore never buys extra LLM calls: a simple question finishes in one
round, a hard one spends CPU on retrieval, and the two-model budget holds.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from agentic_rag_v2.config import DENSE_K_MULTIPLIER, GRAPH_HOPS, POOL_CAP
from agentic_rag_v2.types import (Evidence, JudgeVerdict, Passage, SearchPlan,
                                  SearchMode, SearchResult)

# fusion weights: identical to the v1 harness for the routes it had
ROUTE_WEIGHTS = {"bm25": 0.58, "dense": 0.60, "graph": 0.42, "hop": 0.85}
TITLE_BOOST_IS_ABSOLUTE = True      # title anchoring already produces a boost
# max: divide by the route's best score (original behaviour); minmax: rescale the
# candidate window to [0,1]. Dense cosine scores are compressed into a narrow band
# (~0.65-0.85), so max-scaling would make semantic evidence a near-constant vote.
ROUTE_NORM = {"dense": "minmax"}
MAX_ROUTE_WORKERS = 4


def run_search(question: str, plan: SearchPlan, retriever, judge=None,
               *, pool_cap: int = POOL_CAP, parallel: bool = True,
               semantic=None) -> SearchResult:
    evidence = Evidence()
    verdicts: list[JudgeVerdict] = []
    round_trace: list[dict] = []
    queries_trace: list[list[str]] = []
    llm_calls = 0

    queries = plan.search_queries(question)
    use_graph = plan.graph
    total_rounds = max(1, plan.rounds)

    for round_idx in range(total_rounds):
        queries_trace.append(list(queries))
        prev_size = len(evidence)
        tasks = _route_tasks(queries, use_graph, evidence, semantic is not None)
        # one batched encode per round instead of one forward pass per route task
        dense_vectors = semantic.embed_many(queries) if (semantic and any(
            route == "dense" for _, route in tasks)) else {}
        started = time.perf_counter()
        route_results = _run_tasks(tasks, retriever, plan, evidence, parallel,
                                   semantic=semantic, dense_vectors=dense_vectors)
        search_ms = (time.perf_counter() - started) * 1000

        seen = set(evidence.passages)
        fused, per_route = _fuse(route_results, exclude=seen)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: plan.top_k * 2]
        added = 0
        for cid, score in ranked:
            chunk = retriever.by_id[cid]
            source = per_route[cid]["routes"][0] if per_route[cid]["routes"] else "bm25"
            added += int(evidence.add(Passage(
                chunk_id=cid, doc_id=chunk.doc_id, title=chunk.title, text=chunk.text,
                score=round(score, 4), query=per_route[cid]["query"], source=source)))
        evidence.cap(pool_cap)

        entry = {
            "round": round_idx + 1,
            "queries": list(queries),
            "routes": {name: len(hits) for name, hits in route_results.items()},
            "added": added,
            "n_evidence": len(evidence),
            "search_ms": round(search_ms, 1),
        }

        if round_idx + 1 >= total_rounds or judge is None:
            round_trace.append(entry)
            break

        verdict = judge.judge(question, evidence,
                              multihop=use_graph or total_rounds > 1)
        llm_calls += 1 if verdict.source == "llm" else 0
        verdicts.append(verdict)
        entry["verdict"] = verdict.to_dict()
        round_trace.append(entry)

        if verdict.sufficient or not verdict.next_query:
            break
        if len(evidence) <= prev_size and added == 0:
            break                       # an extra round cannot add anything new
        queries = [verdict.next_query]
        use_graph = True                # follow-up rounds always get the graph route

    return SearchResult(evidence=evidence, rounds=len(round_trace), verdicts=verdicts,
                        queries=queries_trace, trace=round_trace, llm_calls=llm_calls)


# --------------------------------------------------------------------- routes
def _route_tasks(queries: list[str], use_graph: bool, evidence: Evidence,
                 use_dense: bool = False) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for q in queries:
        tasks.append((q, "bm25"))
        if use_dense:
            tasks.append((q, "dense"))
        tasks.append((q, "title"))
        if use_graph:
            tasks.append((q, "graph"))
    if use_graph and evidence.passages:
        tasks.append(("", "hop"))
    return tasks


def _run_tasks(tasks: list[tuple[str, str]], retriever, plan: SearchPlan,
               evidence: Evidence, parallel: bool, *,
               semantic=None,
               dense_vectors: dict | None = None) -> dict[str, list[tuple[str, str, float]]]:
    """Execute every route; returns route -> [(query, chunk_id, raw score)]."""
    grouped: dict[str, list[tuple[str, str, float]]] = {name: [] for _, name in tasks}
    if not tasks:
        return grouped

    def run(task: tuple[str, str]) -> tuple[str, str, list[tuple[str, float]]]:
        query, route = task
        return route, query, _run_route(query, route, retriever, plan, evidence,
                                        semantic=semantic, dense_vectors=dense_vectors)

    if parallel and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_ROUTE_WORKERS, len(tasks))) as pool:
            results = list(pool.map(run, tasks))
    else:
        results = [run(task) for task in tasks]

    for route, query, hits in results:
        grouped[route].extend((query, cid, score) for cid, score in hits)
    return grouped


def _run_route(query: str, route: str, retriever, plan: SearchPlan,
               evidence: Evidence, *, semantic=None,
               dense_vectors: dict | None = None) -> list[tuple[str, float]]:
    exclude = set(evidence.passages)
    if route == "bm25":
        return retriever.bm25_search(query, k=plan.top_k * 2, exclude=exclude)
    if route == "dense":
        if semantic is None:
            return []
        vector = (dense_vectors or {}).get(query)
        if vector is None:
            return semantic.search(query, k=plan.top_k * DENSE_K_MULTIPLIER,
                                   exclude=exclude)
        return semantic.index.search(vector, k=plan.top_k * DENSE_K_MULTIPLIER,
                                     exclude=exclude)
    if route == "title":
        return [(cid, boost) for cid, boost in retriever.title_anchor(query).items()
                if cid not in exclude]
    if route == "graph":
        seeds = retriever.link_entities(query)
        if not seeds:
            return []
        ranked, _, _ = retriever.expand(seeds, hops=GRAPH_HOPS, exclude=exclude)
        return ranked[: plan.top_k * 3]
    if route == "hop":
        seeds = evidence.top(5)
        return retriever.neighbours_of_chunks([p.chunk_id for p in seeds],
                                              limit=max(4, plan.top_k // 2))
    raise ValueError(f"unknown route: {route}")


def _normalise(route: str, hits: list[tuple[str, str, float]]) -> dict[str, float]:
    """Rescale one route's raw scores into comparable units (route, not global).

    `hits` may contain the same chunk several times when the plan issued more than
    one query; the caller reduces those to one score per chunk first.
    """
    scores = [s for _, _, s in hits]
    if ROUTE_NORM.get(route) == "minmax":
        low, high = min(scores), max(scores)
        span = (high - low) or 1.0
        return {cid: (s - low) / span for _, cid, s in hits}
    peak = max(scores, default=1.0) or 1.0
    return {cid: s / peak for _, cid, s in hits}


def _best_per_chunk(hits: list[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    """Collapse (query, chunk, score) hits to one entry per chunk: the best score.

    A route gets ONE vote per chunk no matter how many of the plan's queries found
    it. Without this reduction the fusion loop added the route weight once per
    (query, chunk) pair, so a chunk found by 3 queries of the same retriever
    outvoted the other routes purely because the planner wrote more queries.
    """
    best: dict[str, tuple[str, float]] = {}
    for query, cid, score in hits:
        current = best.get(cid)
        if current is None or score > current[1]:
            best[cid] = (query, score)
    return best


def _fuse(route_results: dict[str, list[tuple[str, str, float]]],
          exclude: set[str]) -> tuple[dict[str, float], dict[str, dict]]:
    """Normalise each route, apply its weight once per chunk, then sum routes."""
    fused: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for route, hits in route_results.items():
        if not hits:
            continue
        per_chunk = _best_per_chunk(hits)
        normalised = _normalise(route, [(q, cid, s) for cid, (q, s) in per_chunk.items()])
        for cid, (query, _score) in per_chunk.items():
            if cid in exclude:
                continue
            contribution = normalised[cid] if (TITLE_BOOST_IS_ABSOLUTE and route == "title") \
                else ROUTE_WEIGHTS[route] * normalised[cid]
            fused[cid] = fused.get(cid, 0.0) + contribution
            entry = meta.setdefault(cid, {"routes": [], "query": ""})
            entry["routes"].append(route)
            if not entry["query"] and query:
                entry["query"] = query
    for entry in meta.values():
        entry["routes"].sort(key=lambda r: -ROUTE_WEIGHTS.get(r, 1.0))
    return fused, meta
