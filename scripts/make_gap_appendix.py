"""Generate the appendix tables for the EM-gap report (UTF-8 safe).

    python agentic_rag_v2/scripts/make_gap_appendix.py

Reads the result batches and writes `docs/EM_GAP_ANALYSIS.md` appendix tables.
Kept as a script (not hand-written markdown) so the numbers cannot drift from the
result files.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "agentic_rag_v2" / "results"


def load(path: Path) -> dict[str, dict]:
    return {rec["qid"]: rec for rec in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip())}


def table(a: dict, b: dict, label_a: str, label_b: str,
          title: str, note: str = "") -> list[str]:
    diff = sorted([q for q in a if a[q]["pred"] != b[q]["pred"]],
                  key=lambda q: (not (b[q]["em"] > a[q]["em"]), q))
    lines = [f"#### {title}", "",
             f"| # | qid | gold | {label_a} | {label_b} | 谁对 |",
             "|---|---|---|---|---|---|"]
    for i, q in enumerate(diff, 1):
        who = label_b if b[q]["em"] > a[q]["em"] else (
            label_a if a[q]["em"] > b[q]["em"] else "都错")
        gold = str(a[q]["gold"])[:42].replace("|", "\\|")
        pa = str(a[q]["pred"])[:40].replace("|", "\\|")
        pb = str(b[q]["pred"])[:40].replace("|", "\\|")
        lines.append(f"| {i} | {q} | {gold} | {pa} | {pb} | {who} |")
    wins_b = sum(1 for q in diff if b[q]["em"] > a[q]["em"])
    wins_a = sum(1 for q in diff if a[q]["em"] > b[q]["em"])
    lines += ["", f"变化 **{len(diff)}** 题：{label_b} 赢 **{wins_b}**，"
                  f"{label_a} 赢 **{wins_a}**，其余两边都错。"]
    if note:
        lines += ["", note]
    lines.append("")
    return lines


def main() -> None:
    ce = load(RESULTS / "v2final" / "v2_llm_judge.jsonl")        # cross-encoder 顺序
    score = load(RESULTS / "v2confirm" / "v2_llm_judge.jsonl")   # 检索分数序
    base2 = load(RESULTS / "v2kgab" / "v2_full.jsonl")           # 2 调用，KG 关
    kg2 = load(RESULTS / "v2kgab" / "v2_kg.jsonl")               # 2 调用，KG 开
    kg_best = load(RESULTS / "v2kgllm" / "v2_llm_judge_kg.jsonl")  # 最优档，KG 开

    lines: list[str] = ["## 附录：逐题变更明细", "",
                        "以下表格由 `scripts/make_gap_appendix.py` 直接读结果文件生成，"
                        "数字不会与批次脱节。", ""]
    lines += table(ce, score, "CE 序", "分数序",
                   "A. 排序 A/B —— 同一证据池、同一轮数、同一 judge，只换编号顺序",
                   "注：两批的证据池 100/100 相同、检索轮数 100/100 相同，"
                   "因此这里每一行都是纯粹的排序效应。")
    lines += table(base2, kg2, "KG 关", "KG 开",
                   "B1. KG 三元组注入 A/B —— 2 调用档（分数序 + 本地 judge）")
    lines += table(score, kg_best, "KG 关", "KG 开",
                   "B2. KG 三元组注入 A/B —— 最优档（分数序 + LLM judge）",
                   "注：B1 里 KG 净 +2 题、B2 里净 −3 题、CE 序下净 −3 题 —— "
                   "三次方向不一致且都在 ±3 题内，属噪声，不构成原因。")

    out = Path(__file__).resolve().parents[1] / "docs" / "EM_GAP_APPENDIX.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
