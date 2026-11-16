"""Corpus and benchmark loading.

The corpus is reused from the v1 harness on purpose: the same 8070 shared chunks
and the same gold file keep v2 comparable to the published v1 numbers, and the
anti-cheating contract (no question ids, no gold fields inside retrieval) holds
by construction because retrieval only ever sees `Chunk` objects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentic_rag_v2.config import BENCH, corpus_path, gold_path

DATASETS = {
    "hotpotqa_50": BENCH / "hotpotqa_50.jsonl",
    "musique_50": BENCH / "musique_50.jsonl",
}


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def load_chunks(path: Path | None = None) -> list[Chunk]:
    p = path or corpus_path()
    if not p.exists():
        raise FileNotFoundError(
            f"corpus not found at {p}; build it with agentic_rag/corpus.py "
            "or set RAGV2_DATA_ROOT")
    chunks: list[Chunk] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        chunks.append(Chunk(chunk_id=rec["chunk_id"], doc_id=rec["doc_id"],
                            title=rec["title"], text=rec["text"]))
    return chunks


def load_gold(path: Path | None = None) -> dict[str, dict]:
    p = path or gold_path()
    return json.loads(p.read_text(encoding="utf-8"))


def load_benchmark(dataset: str, n: int | None = None) -> list[dict]:
    path = DATASETS[dataset]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[:n] if n else rows


def load_questions(datasets: list[str], n: int | None = None,
                   qids: list[str] | None = None) -> list[dict]:
    """Questions from the sampled benchmark files, optionally filtered by id."""
    questions: list[dict] = []
    for name in datasets:
        questions.extend(load_benchmark(name, n))
    if qids:
        keep = set(qids)
        found = {q["id"] for q in questions}
        missing = keep - found
        if missing:
            raise ValueError(f"question ids not found: {sorted(missing)}")
        questions = [q for q in questions if q["id"] in keep]
    return questions
