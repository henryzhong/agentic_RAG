"""Render the v2 architecture diagram (PNG + SVG).

    python agentic_rag_v2/docs/make_architecture.py

Generated from code so it stays in step with the pipeline: stage names, the
per-stage cost column and the ablation switches match `pipeline.PipelineOptions`.
Layout is a single vertical flow with three columns — stage labels, the flow, and
the network-cost / shared-asset rail — so nothing overlaps.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "local": "#DCEAF7",      # local computation, no API
    "llm": "#FBE3D6",        # paid network call
    "data": "#E7F0E3",       # data / intermediate artefact
    "judge": "#F2E4F6",      # control decision
    "outline": "#4A5568",
}

MAIN_X, MAIN_W = 0.135, 0.575      # flow column
COST_X = 0.755                      # network-call column
RAIL_X, RAIL_W = 0.800, 0.195       # right rail


def box(ax, x, y, w, h, text, *, kind="local", fontsize=10, weight="normal",
        linestyle="-", alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.3, edgecolor=COLORS["outline"], facecolor=COLORS[kind],
        linestyle=linestyle, alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=3, linespacing=1.55)


def arrow(ax, start, end, *, color=COLORS["outline"], lw=1.4, rad=0.0,
          linestyle="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", linestyle=linestyle,
        shrinkA=1, shrinkB=1, zorder=1))


def label(ax, x, y, text, *, fontsize=8.6, color="#4A5568", weight="normal"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha="center", va="center",
            weight=weight, zorder=4, linespacing=1.4)


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.6, 11.6), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.075, 1)
    ax.axis("off")

    ax.text(0.5, 0.977, "Agentic RAG v2 架构", ha="center", va="center",
            fontsize=21, weight="bold", color="#1A202C")
    ax.text(0.5, 0.947,
            "两次 LLM 调用 + 本地智能决策 + 受控自适应检索      "
            "实测 100 题：2.0 调用 EM 0.48 ｜ 3.4 调用 EM 0.62（v1 参照 5.9 调用 0.59）",
            ha="center", va="center", fontsize=10.4, color="#4A5568")

    label(ax, 0.045, 0.905, "阶段", fontsize=9.6, weight="bold", color="#2D3748")
    label(ax, COST_X, 0.905, "网络调用/题", fontsize=9.6, weight="bold", color="#2D3748")
    label(ax, RAIL_X + RAIL_W / 2, 0.905, "开关 / 资产", fontsize=9.6, weight="bold",
          color="#2D3748")

    # ---------------------------------------------------------------- question
    box(ax, 0.270, 0.850, 0.330, 0.048, "Question（单跳 / 多跳混合）", kind="data",
        fontsize=11, weight="bold")
    arrow(ax, (0.435, 0.850), (0.435, 0.834))

    # ------------------------------------------------------------------ stage 1
    label(ax, 0.045, 0.790, "① analyze", fontsize=9.6, weight="bold")
    box(ax, MAIN_X, 0.746, MAIN_W, 0.086,
        "Model A（本地多任务分类器 DistilBERT，air/checkpoints/model_a/E5）\n"
        "intent ／ multihop_prob ／ hop 桶 ／ complexity  →  RouteSignal\n"
        "确定性预算：rounds、top_k（只降不升，仅\"确信单跳\"才减预算）",
        kind="local", fontsize=9.6)
    label(ax, COST_X, 0.789, "0", fontsize=12, weight="bold", color="#2B6CB0")
    arrow(ax, (0.435, 0.746), (0.435, 0.730))

    # ------------------------------------------------------------------ stage 2
    label(ax, 0.045, 0.679, "② plan", fontsize=9.6, weight="bold")
    box(ax, MAIN_X, 0.630, MAIN_W, 0.096,
        "DeepSeek #1 —— 一次调用完成「路由 + 改写 + 搜索计划」\n"
        "keep（原问题即好 query） ｜ rewrite（改写/拆解 1–3 条） ｜ graph（按需 GraphRAG）\n"
        "输出 queries ／ rounds ／ top_k；代码按 Model A 预算钳制，非法输出回落启发式",
        kind="llm", fontsize=9.6)
    label(ax, COST_X, 0.678, "1", fontsize=12, weight="bold", color="#C05621")

    for x_from, x_to in ((0.230, 0.200), (0.330, 0.380), (0.430, 0.560), (0.530, 0.690)):
        arrow(ax, (x_from, 0.630), (x_to, 0.590))

    # ------------------------------------------------------------------ stage 3
    label(ax, 0.045, 0.556, "③ search", fontsize=9.6, weight="bold")
    label(ax, 0.045, 0.527, "（并行）", fontsize=8.4)

    routes = [
        (0.135, 0.145, "BM25\n词面召回"),
        (0.290, 0.165, "E5 dense\nmultilingual-e5-large\n(local ／ vLLM)"),
        (0.465, 0.150, "标题锚定\nexact title anchor"),
        (0.625, 0.120, "实体图 GraphRAG\n共现图 BFS（按需）"),
    ]
    for x, w, text in routes:
        box(ax, x, 0.512, w, 0.074, text, kind="local", fontsize=8.8)

    box(ax, 0.625, 0.442, 0.120, 0.050, "图跳 graph-hop\n证据池内桥接实体",
        kind="local", fontsize=8.2, linestyle="--")

    box(ax, MAIN_X, 0.360, MAIN_W, 0.062,
        "融合 Fusion：各路按自身归一（dense 用 min-max，其余 max）× 权重求和\n"
        "bm25 0.58 ｜ dense 0.60 ｜ graph 0.42 ｜ hop 0.85",
        kind="data", fontsize=9.4)
    for x, w, _ in routes:
        arrow(ax, (x + w / 2, 0.512), (x + w / 2, 0.424), color="#718096")
    arrow(ax, (0.685, 0.442), (0.640, 0.424), color="#718096")

    arrow(ax, (0.435, 0.360), (0.435, 0.344))
    box(ax, MAIN_X, 0.288, MAIN_W, 0.054,
        "证据池 Evidence（硬上限 30 段，(-score, chunk_id) 确定性排序）",
        kind="data", fontsize=9.8, weight="bold")
    label(ax, COST_X, 0.470, "0", fontsize=12, weight="bold", color="#2B6CB0")

    arrow(ax, (0.435, 0.288), (0.435, 0.272))
    box(ax, MAIN_X, 0.176, MAIN_W, 0.094,
        "Local Judge（零调用）：覆盖度 ＋ 关系覆盖 ＋ 桥接落地 ＋ 多跳需 ≥2 文档\n"
        "关系覆盖 ＝「被问的关系词」必须与问题实体同段出现\n"
        "否则只找到第一跳就会误判为\"证据充分\"（这是旧版早停的根因）",
        kind="judge", fontsize=9.3)
    label(ax, COST_X, 0.223, "0", fontsize=12, weight="bold", color="#2B6CB0")

    arrow(ax, (0.710, 0.223), (0.770, 0.223))
    arrow(ax, (0.770, 0.223), (0.770, 0.512), color="#975A16", lw=1.7)
    arrow(ax, (0.770, 0.512), (0.748, 0.525), color="#975A16", lw=1.7)
    label(ax, 0.870, 0.372, "证据不足 → 追加一轮\n（桥接实体 ＋ 缺失关系词）",
          color="#975A16", fontsize=8.2)

    arrow(ax, (0.435, 0.176), (0.435, 0.160))
    label(ax, 0.510, 0.168, "证据充分", fontsize=8.8, color="#276749")

    # ------------------------------------------------------------------ stage 4
    label(ax, 0.045, 0.115, "④ rerank", fontsize=9.6, weight="bold")
    box(ax, MAIN_X, 0.068, MAIN_W, 0.090,
        "默认：检索分数序（实测最优）\n"
        "可选：本地 cross-encoder 重排 —— 提高 gold 文档 rank@8，\n"
        "但多跳答案选择 −7pt（它偏好\"像问题\"的段落，而那正是干扰项）",
        kind="data", fontsize=9.3)
    label(ax, COST_X, 0.113, "0", fontsize=12, weight="bold", color="#2B6CB0")
    arrow(ax, (0.435, 0.068), (0.435, 0.053))

    # ------------------------------------------------------------------ stage 5
    label(ax, 0.045, -0.008, "⑤ answer", fontsize=9.6, weight="bold")
    box(ax, MAIN_X, -0.042, MAIN_W, 0.093,
        "DeepSeek #2 —— 一次调用完成「充分性判断 + 答案生成 + 引用组织」\n"
        "输出 sufficient ／ 最短逐字 span ／ evidence_ids ／ 置信度\n"
        "禁止类别词上推：\"right-hand batsman\" 应答 \"right-hand\"",
        kind="llm", fontsize=9.6)
    label(ax, COST_X, -0.008, "1", fontsize=12, weight="bold", color="#C05621")

    # -------------------------------------------------------------- right rail
    box(ax, RAIL_X, 0.700, RAIL_W, 0.180,
        "PipelineOptions（消融＝配置）\n\n"
        "use_dense       语义召回开关\n"
        "judge_kind      coverage ｜ llm\n"
        "rerank          score ｜ local ｜ llm\n"
        "use_relations   KG 三元组注入\n"
        "answer_prompt   v2 ｜ v1\n"
        "max_rounds      自适应轮数上限",
        kind="judge", fontsize=8.5)

    box(ax, RAIL_X, 0.410, RAIL_W, 0.260,
        "共享资产（与 v1 同一份）\n\n"
        "语料         8,070 段共享池\n"
        "gold.json    仅评测侧读取\n"
        "benchmark    100 题（50+50）\n"
        "Model A      多任务分类权重\n"
        "E5 索引      8,070×1024（27MB）\n"
        "KG 缓存      2,779 段三元组\n"
        "reranker     ONNX cross-encoder",
        kind="data", fontsize=8.5)

    box(ax, RAIL_X, 0.176, RAIL_W, 0.150,
        "输出\n\n"
        "Answer ＋ citations\n"
        "＋ trace：每段耗时 / 路由 /\n"
        "判定信号 / 调用数",
        kind="data", fontsize=8.5)

    # ------------------------------------------------------------------ legend
    legend_y = -0.062
    items = [("local", "本地计算（无 API）"), ("llm", "付费 LLM 调用"),
             ("data", "数据 / 中间产物"), ("judge", "控制决策")]
    x = 0.135
    for kind, text in items:
        ax.add_patch(plt.Rectangle((x, legend_y), 0.015, 0.013, facecolor=COLORS[kind],
                                   edgecolor=COLORS["outline"], linewidth=1.1,
                                   clip_on=False))
        ax.text(x + 0.020, legend_y + 0.0065, text, fontsize=8.5, va="center",
                clip_on=False)
        x += 0.160

    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"architecture.{suffix}", bbox_inches="tight",
                    facecolor="white", pad_inches=0.22)
    plt.close(fig)
    print("wrote", HERE / "architecture.png", "and architecture.svg")


if __name__ == "__main__":
    main()
