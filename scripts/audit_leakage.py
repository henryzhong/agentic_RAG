"""Audit overlap between Model A's training data and the evaluation questions.

    python agentic_rag_v2/scripts/audit_leakage.py

Checks, for every split file under air/data/processed{,_v2,_clean}:
  * exact question overlap (build_dataset.norm_question) with the 100 eval questions
  * which eval questions appear in Model A's *train* file specifically

The eval set is benchmark/hotpotqa_50.jsonl + benchmark/musique_50.jsonl; both were
sampled from the same raw dev files that build_dataset.py reads, so this is the
check that decides whether the reported Model A accuracy means anything.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH = ROOT / "benchmark"
PROCESSED = (ROOT / "air" / "data" / "processed", ROOT / "air" / "data" / "processed_v2",
             ROOT / "air" / "data" / "processed_clean")
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "ood_test.jsonl",
               "hard_negatives.jsonl", "decomposition_train.jsonl",
               "decomposition_test.jsonl", "decomposition_validation.jsonl")


def norm_question(q: str) -> str:
    """Same normalisation as air/src/common.py::norm_question."""
    import re

    q = (q or "").lower().strip()
    q = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def load_eval() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("hotpotqa_50", "musique_50"):
        for line in (BENCH / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def load_split(path: Path) -> tuple[set[str], set[str]]:
    """Returns (questions, sample_ids) in the file; tolerates several schemas."""
    questions: set[str] = set()
    ids: set[str] = set()
    if not path.exists():
        return questions, ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        q = rec.get("question") or (rec.get("input") or {}).get("question") or ""
        if q:
            questions.add(norm_question(q))
        sid = rec.get("sample_id") or rec.get("id")
        if sid:
            ids.add(str(sid))
    return questions, ids


def main() -> None:
    evals = load_eval()
    eval_q = {qid: norm_question(rec["question"]) for qid, rec in evals.items()}
    print(f"eval questions: {len(eval_q)}")

    report: dict[str, dict] = {}
    for root in PROCESSED:
        if not root.exists():
            continue
        for name in SPLIT_FILES:
            path = root / name
            questions, ids = load_split(path)
            if not questions:
                continue
            hit = {qid for qid, q in eval_q.items() if q in questions}
            report[f"{root.name}/{name}"] = {
                "n_pretrain_samples": len(questions),
                "overlap": len(hit),
                "pct_of_eval": round(100 * len(hit) / len(eval_q), 1),
                "examples": sorted(hit)[:5],
            }

    width = max(len(k) for k in report) + 2
    print(f"\n{'split file':<{width}}{'样本':>8}{'与评测重叠':>12}{'占评测%':>10}")
    for key, row in report.items():
        print(f"{key:<{width}}{row['n_pretrain_samples']:>8}{row['overlap']:>12}"
              f"{row['pct_of_eval']:>9}%")

    train_keys = [k for k in report if k.endswith("train.jsonl")]
    print("\n--- Model A 实际训练文件（train.jsonl）里的评测题 ---")
    for key in train_keys:
        row = report[key]
        print(f"{key}: {row['overlap']}/{len(eval_q)} ({row['pct_of_eval']}%) "
              f"例: {row['examples']}")

    # per-dataset slice
    for ds in ("hotpotqa", "musique"):
        qids = [qid for qid, rec in evals.items() if rec.get("source") == ds]
        line = [f"{ds}: {len(qids)} 题"]
        for key in train_keys:
            questions, _ = load_split(PROJECT(key))
            hit = sum(1 for qid in qids if eval_q[qid] in questions)
            line.append(f"{key.split('/')[0]}={hit}/{len(qids)}")
        print("  " + "   ".join(line))

    out = ROOT / "agentic_rag_v2" / "work" / "leakage_audit.json"
    out.write_text(json.dumps({"eval_n": len(eval_q), "report": report},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", out)


def PROJECT(key: str) -> Path:
    root_name, _, file_name = key.partition("/")
    for root in PROCESSED:
        if root.name == root_name:
            return root / file_name
    return Path(key)


if __name__ == "__main__":
    main()
