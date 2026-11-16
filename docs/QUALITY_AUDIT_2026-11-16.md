# Agentic RAG v2 质量审计报告

> **编号**：`RAGV2-AUDIT-2026-11-16-001`
> **日期**：2026-11-16
> **对象**：`agentic_rag_v2`（两次 LLM + 本地智能决策 + 受控自适应检索）
> **范围**：用户提出的 6 项整治 —— ① Model A 训练/测试泄露 ② Model A 是否全链路无条件使用
> ③ LLM plan 与 LLM judge 的输出契约 ④ 向量缓存缺陷 ⑤ 融合层重复计票 ⑥ 真实重跑
> **结论**：6 项全部核实并修复，其中 ①④⑤ 是**真实缺陷**（会静默产生错误结果），
> ②③ 是**设计/文档缺陷**（不会算错，但会让结论不可信），⑥ 已用修复后的代码 + 去泄露权重真实重跑。

---

## 0. 摘要

| # | 问题 | 判定 | 关键证据 | 处置 |
|---|---|---|---|---|
| ① | Model A 训练数据与评测集重叠 | **存在严重泄露** | 100 道评测题中 **83 道**在 `processed/train.jsonl` 里（HotpotQA 37/50、MuSiQue 46/50），另有 8 道在 validation、9 道在 test | 重新切分（先扣评测题再切 80/10/10）→ `processed_clean/`，重训 **E6**，用 holdout 诚实评测 |
| ② | Model A 是否所有情况都用到 | **之前有开关，已删** | `PipelineOptions.use_model_a`、`DisabledAnalyzer`、`v2_no_model_a` 臂；默认权重还是含泄露的 E5 | 删开关/删臂/删降级路径；构造期 `require()` 硬校验；默认 **E6**；基线臂也必须过第一段 |
| ③ | plan / judge 的输出是什么 | **之前没有文档** | schema 只存在于 prompt 字符串和 `coerce()` 里 | 本文 §3 给出字段、钳制规则、失败回落、四段真实 trace |
| ④ | 向量缓存有问题 | **5 个真实缺陷** | 旧缓存用 id+文本长度做指纹、ids 用 pickle 存、slug 不含 endpoint、加载不校验、query key 未归一 | 内容哈希 + JSON sidecar + identity 校验 + 4 项加载校验；旧缓存全部被拒并重建 |
| ⑤ | 融合层重复计票 | **真实缺陷，影响排序** | **94/100** 题存在同一路由对同一 chunk 重复投票：2,911/12,391 个 hit 是重复（23.5%），top-1 变化 19/100、top-2k 集合变化 69/100、顺序变化 87/100 | `_best_per_chunk`：每路由每 chunk 只投一票 |
| ⑥ | 是否真实重跑 | **之前混用旧批次** | 结果目录里既有旧代码批次，也有带 LLM 响应缓存的批次 | `--no-cache` 真实重跑（E6 + 修复后代码），新批次 `audit_rerun` |

---

## 1. 问题①：Model A 的训练数据与评测集泄露

### 1.1 现象

Model A（4 任务分类器：intent / complexity / multihop / hop）在早期报告里几乎满分
（hop/complexity/multihop = 1.000、intent F1 = 0.975）。这个数字好得不像迁移学习的结果，
需要先排除"它见过这些题"。

### 1.2 证据

[`scripts/audit_leakage.py`](../scripts/audit_leakage.py) 对 `air/data/processed*` 下每个切分文件
做问题级比对（`common.norm_question` 归一后精确匹配）：

| 切分文件 | 样本数 | 与 100 道评测题重叠 | 占评测集 |
|---|---|---|---|
| `processed/train.jsonl` | 15,184 | **83** | **83.0%** |
| `processed/validation.jsonl` | 1,856 | 8 | 8.0% |
| `processed/test.jsonl` | 1,952 | 9 | 9.0% |
| `processed/decomposition_train.jsonl` | 4,648 | 58 | 58.0% |
| `processed/ood_test.jsonl` | 199 | 0 | 0.0% |
| `processed/hard_negatives.jsonl` | 272 | 0 | 0.0% |

按数据集切分：HotpotQA **37/50**、MuSiQue **46/50** 落在 `train.jsonl` 里。

**根因**：[`air/src/build_dataset.py`](../../air/src/build_dataset.py) 从
`hotpot_dev_distractor_v1.json` 与 `musique_ans_v1.0_dev.jsonl` 抽样，
而 `benchmark/hotpotqa_50.jsonl`、`benchmark/musique_50.jsonl` **也是从同两个 dev 文件抽的**，
之后用 `stable_bucket(sample_id, "air-split-v1")` 做 80/10/10 切分。
两条独立流水线处理同一批样本 → 评测题自然落在训练集里。

### 1.3 修复：先扣题，再切分

[`scripts/make_clean_split.py`](../scripts/make_clean_split.py)：

1. 复制 `air/data/processed/` → `air/data/processed_clean/`；
2. 对 `train / validation / test / hard_negatives / ood_test` 逐行判断：问题与任一评测题
   **归一后精确相同**，或 **token Jaccard ≥ 0.85**（近重复），即丢弃；
3. 被丢掉的样本带标签写入 `eval_holdout.jsonl`（100 条）→ 变成"真正没见过的测试集"。

| 文件 | 原 | 清洗后 | 丢弃 |
|---|---|---|---|
| `train.jsonl` | 15,184 | 15,093 | 91 |
| `validation.jsonl` | 1,856 | 1,846 | 10 |
| `test.jsonl` | 1,952 | 1,942 | 10 |

（丢弃 111 行 > 100 题，因为少数题在同一切分里出现多次。）

**验证**：重跑 `audit_leakage.py`（现已把 `processed_clean` 纳入扫描）：

```
processed_clean/train.jsonl        15093           0      0.0%
processed_clean/validation.jsonl    1846           0      0.0%
processed_clean/test.jsonl          1942           0      0.0%
processed_clean/decomposition_train.jsonl  4648   58     58.0%
```

`decomposition_train.jsonl` 仍然重叠 58%，但 **Model A 不读它**：
[`air/src/model_a.py::run_experiment`](../../air/src/model_a.py) 只读
`train.jsonl` + `validation.jsonl`（+ `use_hard=True` 时的 `hard_negatives` train 部分）。
这条约束已写进 `make_clean_split.py` 的注释，避免以后有人"顺手"把分解任务也加进训练。

### 1.4 重训与诚实评测

- 给 `air/src/model_a.py` 增加可选参数 `data_dir`（默认行为不变），新增配置 **E6**
  = E5 的全部超参（`tasks=[intent, complexity, multihop, hop]`, 3 epochs, lr 2e-5, bs 32, hard negatives）
  但 `data_dir=processed_clean`。训练 53s，`val_score = 0.9543`。
- [`scripts/eval_model_a_holdout.py`](../scripts/eval_model_a_holdout.py) 在同一份
  `eval_holdout.jsonl`（100 题标签）上对比：

| 权重 | 见过这 100 题 | hop | complexity | multihop | intent F1 | mean |
|---|---|---|---|---|---|---|
| E5（含泄露） | 是（83/100 在训练集） | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 |
| **E6（去泄露）** | 否（0/100） | 0.990 | 0.990 | 1.000 | 0.955 | **0.9933** |

### 1.5 结论（重要，不要误读）

1. **泄露是真实存在的**：之前"Model A 满分"的表述无效，必须换成"在去泄露切分上测得"。
2. **但泄露不是 Model A 好用的原因**：去掉泄露只掉 1pt hop/complexity、2.5pt intent F1。
   这个任务（4 分类 + 二分类，输入是问题文本）对 `distilbert-base-uncased` 已经饱和，
   所以"Model A 的路由信号可信"这个架构结论**在干净数据上依然成立**。
3. **端到端影响可忽略**：在 `v2ab` 批次里 `v2_full` 与 `v2_no_model_a` 逐题完全一致
   （0.45 / 0.5846 / 0.8083 三个指标一模一样），Model A 对最终 EM 本来就是中性的。
4. **新规矩**：新增任何评测集，先跑 `audit_leakage.py`；新增训练切分，必须从**所有** split 里
   扣掉评测题 id；今后报告 Model A 指标必须写清 checkpoint 名（E6 起）与切分来源。

复现：

```powershell
& $PY agentic_rag_v2/scripts/audit_leakage.py
& $PY agentic_rag_v2/scripts/make_clean_split.py            # 加 --dry-run 只看统计
& $PY air/src/model_a.py --exp E6
& $PY agentic_rag_v2/scripts/eval_model_a_holdout.py --checkpoints E5,E6
```

---

## 2. 问题②：Model A 必须无条件使用（已删掉所有开关）

### 2.1 现象

修复前流水线里存在三条"绕过第一段"的路：

- `PipelineOptions.use_model_a: bool = True`（配置开关）；
- `DisabledAnalyzer`（返回 `RouteSignal(available=False)`，`budget_for` 退化成默认预算）；
- 评测臂 `v2_no_model_a`（`PipelineOptions(use_model_a=False)`）。

同时 `ModelAAnalyzer` 的默认 checkpoint 是 **E5**（即 1.3 节里含泄露的那份）。

后果不是"算错"，而是**结论不可比较**：一旦有人拿 `v2_no_model_a` 的 run 当基线，
或者 checkpoint 加载失败被静默降级，结果与 `v2_full` 之间差的就不只是"第一段"，
而没人能从 manifest 里看出来。

### 2.2 修复

| 改动 | 文件 |
|---|---|
| 删除 `use_model_a` 字段（`PipelineOptions` 里不再有它） | [pipeline.py](../pipeline.py) |
| 删除 `DisabledAnalyzer` | [models/model_a.py](../models/model_a.py) |
| 删除 `v2_no_model_a` 臂（`SYSTEMS` / `SYSTEM_ORDER`） | [eval/runner.py](../eval/runner.py) |
| `AgenticRAG.__init__`：`analyzer = analyzer or ModelAAnalyzer(); analyzer.require()` | [pipeline.py](../pipeline.py) |
| `run_suite`：无条件构建 + `require()`（manifest 记录 `checkpoint`） | [eval/runner.py](../eval/runner.py) |
| `answer_baseline` 也先跑 `analyze()`（无检索下界同样过第一段） | [pipeline.py](../pipeline.py) |
| 默认 checkpoint `E5 → E6`（可用 `RAGV2_MODEL_A` 覆盖） | [config.py](../config.py) |

`require()` 是硬校验：加载失败 → `RuntimeError`，run 直接起不来。
**宁可起不来，也不要静默降级**。

### 2.3 行为变化（明确记录）

- checkpoint 缺失/损坏时，过去是"跑完但路由中性"，现在是**启动即失败**。
- `v2_direct`（无检索下界）的记录里 `route.available` 从 `False` 变成真实信号，
  `trace` 从 `["direct"]` 变成 `["analyze", "direct"]`。历史 `v2ab` 批次里的
  `v2_direct.route.available=false` 与现在不可比，属预期。
- `v2_no_model_a` 的历史结果（`v2ab`：EM 0.45，与 `v2_full` 逐题一致）保留为历史证据，
  但代码已删，不再可复现。

### 2.4 测试

```
test_model_a_cannot_be_switched_off            # PipelineOptions 里不存在 use_model_a
test_broken_model_a_blocks_construction        # 坏权重 -> RuntimeError，且从不回落
test_every_question_passes_through_model_a     # answer() 与 answer_baseline() 都调用 predict()
```

---

## 3. 问题③：LLM plan 与 LLM judge 到底输出什么

两条（可选第三条）LLM 调用的**输出契约**如下。所有输出都只是"提案"，
最终值由 `coerce()` 按 Model A 的预算钳制 —— **模型提议，代码裁决**。

### 3.1 第 1 次调用：plan（`stages/plan.py::make_plan`）

**输入**（user message，三段）：

```
QUESTION:
<原问题>

ROUTE SIGNAL (small model):
<Model A 的一行摘要：intent / multihop_prob / hop_bucket / complexity>

BUDGET (hard caps enforced downstream):
rounds<=2, top_k<=8
```

system prompt 里给出路由定义（`keep` / `rewrite` / `graph`）与下面的 JSON schema。

**输出 schema**（`{"mode", "queries", "graph", "rounds", "top_k", "reason"}`）：

| 字段 | 类型 | 含义 | `coerce()` 的钳制规则 |
|---|---|---|---|
| `mode` | `keep｜rewrite｜graph` | 路由：原样搜 / 改写后搜 / 走实体图 | 非法值 → `keep`；`keep` → `queries` 强制回原问题、`graph` 强制 `False` |
| `queries` | `list[str]`（提议 1-3） | 自包含检索查询，第一条为主查询 | 去空、去重、去等于原问题的项；上限 `PLAN_MAX_QUERIES`；`rewrite` 下为空则回退原问题 |
| `graph` | `bool` | 是否开启实体共现图路由 | `mode=graph` → 强制 `True`；`mode=keep` → 强制 `False`（不允许"keep 但开图"） |
| `rounds` | `int` | 期望检索轮数 | 非 graph：`clamp(提议, 1, min(预算, 4))`；graph：先 `max(提议, 预算)` 再 `clamp(1, min(预算, 4))`，因为该值已 ≥ 预算，**最终恒等于 `min(预算, MAX_ROUNDS)`** —— 图路线的轮数由 Model A 的预算决定，LLM 既不能加大也不能压缩 |
| `top_k` | `int` | 每路线取回数 | `clamp(MIN_TOP_K=4, min(预算+4, MAX_TOP_K=16))` |
| `reason` | `str` | 一句话理由（进 trace，便于审计） | 原样保留，非字符串转 `str` |

失败回落：`json_call` 抛异常 → `heuristic()`（多跳→`graph`，否则 `keep`）+ `source="fallback"`
+ `reason` 写成错误信息；`use_planner_llm=False` 走同一函数（`source="fallback"`）；
`force_mode` 直接覆盖 `mode`（`source="forced"`，`keep` 时查询锁死原问题）。

**真实样例**（`audit_rerun` / `hp-005`，`source="llm"`）：

```json
{
 "question": "What is the name of the number-one single featuring the winner of Celebrity Apprentice 3?",
 "route": {"intent": "multi_hop_qa", "multihop_prob": 0.9998, "hop_bucket": "2",
           "complexity": "medium", "latency_ms": 7.42},
 "budget": {"rounds": 2, "top_k": 8, "answer_top_n": 30, "source": "model_a"},
 "plan": {
  "mode": "graph",
  "queries": ["Celebrity Apprentice 3 winner",
              "number-one single by Celebrity Apprentice 3 winner"],
  "graph": true, "rounds": 2, "top_k": 8,
  "reason": "The answer requires chaining the winner of Celebrity Apprentice 3 to that person's number-one single, so graph retrieval with two rounds is needed.",
  "source": "llm"
 }
}
```

### 3.2 第 2 段（本地，默认）：CoverageJudge

不是 LLM，但**同样是"判定输出"**，且它决定循环是否继续，因此必须写清：
`judge(question, evidence, multihop) -> JudgeVerdict`。

| 字段 | 含义 |
|---|---|
| `sufficient` | 四信号同时达标 + （多跳时）桥接已打通 |
| `missing` | 问题里仍未出现在证据池的词（前 6 个，`missing_tokens`） |
| `next_query` | 下一轮查询：`桥接实体` + `最强段落里缺失的问题词`（最多 3 个）；确定性拼接 |
| `reason` | `coverage token=… best_passage=… entity=… relation=… bridge=…` |
| `score` | `0.45·token + 0.25·best_passage + 0.15·entity + 0.15·relation`（仅记录，不参与判定） |
| `signals` | `token_coverage / best_passage_coverage / entity_coverage / relation_coverage / missing_tokens / bridge_entity / bridge_resolved / seeds` |
| `source` | `coverage` |

**真实样例 A（判为不足 → 追加一轮，`hp-008`）**：

```json
{"sufficient": false,
 "missing": "going headquartered",
 "next_query": "new testament going six working",
 "reason": "coverage token=0.78 best_passage=0.56 entity=0.80 relation=0.67 bridge=new testament (unresolved)",
 "score": 0.7089,
 "signals": {"token_coverage": 0.7778, "best_passage_coverage": 0.5556,
             "entity_coverage": 0.8, "relation_coverage": 0.6667,
             "missing_tokens": ["going", "headquartered"],
             "bridge_entity": "new testament", "bridge_resolved": false,
             "seeds": ["john burges", "s work", "burges", "compa", "quarter"]},
 "source": "coverage"}
```

**真实样例 B（判为充分 → 早停，`hp-001`）**：

```json
{"sufficient": true, "missing": "batsmans", "next_query": "",
 "reason": "coverage token=0.88 best_passage=0.88 entity=1.00 relation=1.00 bridge=england",
 "score": 0.9125,
 "signals": {"token_coverage": 0.875, "best_passage_coverage": 0.875,
             "entity_coverage": 1.0, "relation_coverage": 1.0,
             "missing_tokens": ["batsmans"], "bridge_entity": "england",
             "bridge_resolved": true},
 "source": "coverage"}
```

### 3.3 可选第 3 次调用：LLMJudge（`judge_kind="llm"`，即 `v2_llm_judge` 臂）

**输入**：`QUESTION` + 证据摘要（`evidence_digest`：按分数序前 `JUDGE_DIGEST_TOP` 段的
`[i] 标题 :: 正文前 JUDGE_DIGEST_CHARS 字`）。

**输出 schema**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `sufficient` | `bool` | 证据是否足以给出可验证的短答案（prompt 明确"严格"，证据没写出答案就算不足） |
| `missing` | `str` | 还缺的具体事实/实体 |
| `next_query` | `str` | 一条自包含的补充查询；为空 = 不要再查了（循环结束） |
| `reason` | `str` | 一句话理由 |

失败回落：调用异常 → `sufficient=True`,`source="error"`（判定失败不能把整个 run 丢掉，
只能退化成"就用手上的证据回答"），`llm_calls` 不增加。

**真实样例**（`audit_rerun` / `hp-005`，判不足但最终答对）：

```json
{"sufficient": false,
 "missing": "The name of the winner of Celebrity Apprentice 3 (season 9) and the title of that winner's number-one single",
 "next_query": "Celebrity Apprentice 3 winner Bret Michaels number one single",
 "reason": "Evidence [6] identifies the season but truncates the winner's name, and no evidence states the title of any number-one single by that winner.",
 "score": 0.0, "signals": {}, "source": "llm"}
```

### 3.4 循环控制（谁有权改变执行）

```
for round in 1..plan.rounds:
    多路检索 → 融合 → 取 top_k*2 入池 → pool_cap 截断
    if 最后一轮 or 无 judge: break
    verdict = judge(...)
    if verdict.sufficient or not verdict.next_query: break
    if 本轮没有任何新增证据（added == 0）: break     # 追加一轮不可能带来新东西
    queries = [verdict.next_query]; use_graph = True  # 后续轮一律开图
```

即：**LLM 只能"要求继续查"，不能"要求多花钱"**（轮数被预算硬钳制），
也**不能"提前结束到没有证据"**（最后一轮无论如何都会跑完）。

---

## 4. 问题④：向量缓存的 5 个缺陷

语义检索（multilingual-e5-large）把 8,070 段语料编码一次后落盘复用：`work/embeddings/<slug>.npz`。
审计发现 5 个缺陷，全部属于"看起来命中缓存、其实是错的向量"这一类静默错误。

| # | 缺陷 | 后果 | 修复 | 守护测试 |
|---|---|---|---|---|
| 1 | 语料指纹 = `chunk_id + 文本长度` | 等长改写（`alpha` → `omega`）不触发重建，检索用的是旧文本的向量 | `corpus_fingerprint()` 改为 **id + title + text 内容哈希** | `test_corpus_fingerprint_detects_same_length_edits` |
| 2 | chunk ids 存进 `.npz` 的 object 数组 | 加载必须 `allow_pickle=True` → **缓存文件可执行任意代码** | ids 移到 `.json` sidecar，`np.load(..., allow_pickle=False)` | `test_index_rejects_a_cache_without_its_sidecar` |
| 3 | cache slug 只由「模型目录 + 语料」决定 | 换 vLLM endpoint（另一台机器/另一种精度）仍复用同一份向量 | slug 用 `embedder.identity()`，vLLM 后端把 **endpoint 纳入身份** | `test_cache_slug_changes_with_the_vllm_endpoint`、`test_cache_slug_changes_with_corpus` |
| 4 | 加载后不校验 | 复制/改名/换模型留下的缓存文件会被直接使用 | 4 项校验：`corpus_fingerprint`、`embedder_identity`、向量维度（`embedder.expected_dim`）、chunk 数；任一不符 → 记 `cache_error` 并重建 | `test_index_rejects_a_cache_built_for_another_corpus`、`test_build_index_reuses_the_cache` |
| 5 | `CachedEmbedder` 的 key 未归一 | 带首尾空白的 query 查不到向量（`KeyError` 或静默 miss） | key 统一 `strip()`；`embed_many` 用**原始字符串**返回，避免调用方回查失败 | `test_embed_many_keys_are_the_original_queries`、`test_cached_embedder_avoids_repeated_forward_passes` |

**实测证据**（修复后扫描磁盘上的全部缓存）：

```
87b306ccca22d882  (True, 'ok')                      <- 新格式，命中
95c268a8ebc37b01  REJECTED ValueError: index cache ...npz has no ids sidecar; rebuild it
d0453e7e137df615  REJECTED ValueError: index cache ...npz has no ids sidecar; rebuild it
```

两个旧文件（12:05 的 vLLM 版、12:22 的 local 版）都被判为 stale 并已删除重建：

| 项 | 旧缓存 | 新缓存 |
|---|---|---|
| slug | `95c268a8…` / `d0453e7e…` | `87b306ccca22d882` |
| corpus_fingerprint | `188ee3cea8260a16`（长度指纹） | `a97d3b0020da8e6b`（内容指纹） |
| 结构 | `.npz`（含 pickle 的 ids） | `.npz`（仅 float32 矩阵）+ `.json`（ids + manifest） |
| 命中耗时 | — | **0.12s**（首次重建 22.5s） |

复现：

```powershell
& $PY agentic_rag_v2/scripts/build_embeddings.py --backend local --force
& $PY -c "from agentic_rag_v2.retrieval.vector import build_index; from agentic_rag_v2.models.embedder import build_embedder; from agentic_rag_v2.data import load_chunks; print(build_index(load_chunks(), build_embedder('local'))[1])"
```

（第二条命令应打印 `"cached": true`；把语料改一个字再跑，会打印 `cache_error: corpus fingerprint mismatch` 并重建。）

> 顺带修掉一个"文档与现实不一致"的问题：README 说 G 盘的模型"默认就会找到"，
> 但 `config.e5_model_dir()` 只认 `RAGV2_E5_DIR` 与仓库内 `models/`，在干净的进程里
> `build_embeddings.py --verify` 会直接报 `multilingual-e5-large not found`。
> 已把 `RAGV2_E5_DIR_VLLM` 指定的模型目录 加入候选目录，现在无需任何环境变量。

---

## 5. 问题⑤：融合层重复计票

### 5.1 现象（旧实现）

`_fuse` 的意图是"每个路由给每个 chunk 投一票，按路由权重加权求和"。旧代码是这样做的：

```python
normalised = _normalise(route, hits)          # dict: cid -> 归一化分数
for _query, cid, _score in hits:              # 遍历的是**原始 hit 列表**
    fused[cid] = fused.get(cid, 0.0) + ROUTE_WEIGHTS[route] * normalised[cid]
```

问题：`hits` 里有 **同一 chunk 被同一路由的多个查询命中** 的重复项
（`_route_tasks` 会为 plan 的每条 query 各生成一个 `bm25`/`dense`/`title` 任务）。
于是这条 chunk 会被**加多次权重**：结构化上不再是"投票"，而是"谁被越多条查询命中谁越赢"。

一个具体后果：planner 写了 3 条 query 的题，`bm25` 路由会重复计票，把另一条路由
（例如 `graph`）命中的第二跳文档压下去 —— 而"写几条 query"是 LLM 的自由度，不是相关性。

### 5.2 实测影响（[`scripts/audit_fusion.py`](../scripts/audit_fusion.py)）

把 `audit_rerun/v2_full` 里 100 道题的**真实 plan 查询**重放一遍真实路由，
再分别用旧累加器与新累加器融合，比较结果：

| 指标 | 数值 |
|---|---|
| 存在重复投票的题 | **94 / 100（94.0%）** |
| 重复 hit 对 / 全部 hit 对 | **2,911 / 12,391（23.5%）** |
| 融合后 top-1 变化 | **19 / 100（19.0%）** |
| 入池集合（top-2k）变化 | **69 / 100（69.0%）** |
| 池内顺序变化 | **87 / 100（87.0%）** |
| 候选池成员变化（融合出的所有 key） | 0 / 100 |

例（`hp-002`，2 条 query，119 个 hit 中 31 个重复）：修复前 top-1 是 `a87ed387af180d#c0`，
修复后是 `fee767ad018d22#c0`。

### 5.3 修复

[`stages/search.py::_best_per_chunk`](../stages/search.py)：先把
`[(query, chunk, score)]` 收敛成 `{chunk: (query, best_score)}`（同一 chunk 取**最高**分），
再做路由内归一、再按路由权重求和。即：

```
每个 (路由, chunk) 恰好一票，票面 = 路由权重 × 该路由对该 chunk 的最高归一化分数
最终得分 = Σ_路由 票面
```

### 5.4 修复后的端到端影响（诚实记录）

新批次（§6）`v2_full` 的 EM 是 0.48，与修复前的旧批次数值相同；但**旧批次开着 LLM 响应缓存、
用的是 E5 权重**，所以两者不是严格对照，不能据此说"这个修复对 EM 无影响"。
可以确定的只有一点：被改动的是 19% 题的 top-1 与 69% 题的入池集合、87% 题的池内顺序，
而这批题上"顺序变化 → 答案变化"的比例本来就低
（参见 [EM 差距归因报告 §4.4](EM_GAP_ANALYSIS.md)：顺序影响的是"把哪一段当成答案"，
而不是"池子里有没有那一段"）。

但这层的正确性是**其他结论的前提**：路由权重（`bm25 .58 / dense .60 / graph .42 / hop .85`）、
"dense 比 bm25 强 6pt"这类结论都必须建立在"一 chunk 一票"上，否则调出来的权重是在拟合
"planner 写几条 query"这个无关变量。

守护测试：`test_fusion_takes_the_best_score_per_chunk_within_a_route`、`test_fusion_excludes_seen_chunks`。

---

## 6. 问题⑥：真实重跑

### 6.1 为什么需要重跑

之前的结果目录里混着三种不可比的批次：旧代码（修复前）、E5 权重（含泄露）、
以及开着 LLM 响应缓存（`ArkClient(use_cache=True)`）的批次。**任何一条都会让"结论"和"代码"脱钩。**

### 6.2 本次重跑条件（全部满足才记录）

| 项 | 值 |
|---|---|
| 代码 | 修复 ①②④⑤ 之后的工作树 |
| Model A | **E6**（去泄露，见 §1.4） |
| LLM | `deepseek-v4.1-flash` |
| LLM 响应缓存 | **关闭**（`--no-cache`） |
| 语料 / benchmark | 8,070 段共享语料；HotpotQA 50 + MuSiQue 50 |
| 语义索引 | local E5，缓存命中（`slug=87b306ccca22d882`） |
| 批次 tag | `audit_rerun` |

### 6.3 结果

| 系统 | EM | F1 | containment | 证据召回(检索) | 证据召回(引用) | 轮数 | LLM 调用 | 延迟 | wall |
|---|---|---|---|---|---|---|---|---|---|
| `v2_full`（2 调用档） | 0.48 | 0.6053 | 0.54 | 0.8467 | 0.7608 | 2.1 | 2.0 | 4.45s | 113s |
| `v2_llm_judge`（LLM 判定档） | **0.59** | 0.6963 | 0.67 | **0.9275** | 0.8325 | 2.0 | 3.4 | 6.46s | 165s |
| `v2_no_dense`（关语义召回） | 0.44 | 0.5500 | 0.49 | 0.7850 | 0.7208 | 2.1 | 2.0 | 3.96s | 101s |

结论：

1. **语义召回仍然是最稳的一块**：关掉 dense → EM −4pt、检索召回 −6pt（0.8467 → 0.7850）。
2. **本地 judge 仍是 2 调用档的真瓶颈**：换成 LLM 判定 → EM +11pt、召回 +8pt，代价是调用 2.0 → 3.4。
3. **与旧批次并排**（同口径历史值）：`v2_full` 0.48 → 0.48（一致）；`v2_llm_judge` 0.62 → 0.59
   （E6 + 无缓存重跑，±3pt 落在 100 题 ±9pt 的置信区间内，不做结论性解读）。
4. **轮数分布**（`v2_full`）：1 轮 29 题、2 轮 50 题、4 轮 21 题（4 轮即 `MAX_ROUNDS` 上限）。

配对显著性（McNemar，`scripts/analyze.py`）：

```
batch=audit_rerun  base=v2_full  n=100
v2_llm_judge  EM 0.590  F1 0.696  rec_fin 0.833  rec_ret 0.927  rank@8 0.810  calls 3.4  +14/-3  p=0.013
```

即"LLM 判定 vs 本地判定"在本批次是 **+14 胜 / −3 负，p = 0.013**，不是噪声。

### 6.4 复现命令

```powershell
$PY = "<PYTHON_EXE>"

# 真实重跑（关 LLM 缓存）
& $PY agentic_rag_v2/scripts/run_eval.py `
    --systems v2_full,v2_llm_judge,v2_no_dense --n 100 --workers 4 `
    --tag audit_rerun --no-cache

# 配对分析
& $PY agentic_rag_v2/scripts/analyze.py --tag audit_rerun --base v2_full --vs v2_llm_judge --diffs
```

原始产物：`results/audit_rerun/{summary.json,v2_full.jsonl,v2_llm_judge.jsonl,v2_no_dense.jsonl}`，
每条记录含完整 `trace`（analyze/plan/search/rerank/answer + 逐轮 verdict + 路由计数 + 耗时）。

---

## 7. 变更清单

| 文件 | 变更 |
|---|---|
| [config.py](../config.py) | 新增 `MODEL_A_CHECKPOINT`（默认 `E6`，`RAGV2_MODEL_A` 可覆盖） |
| [models/model_a.py](../models/model_a.py) | `checkpoint_name` 记录；`require()` 硬校验；删除 `DisabledAnalyzer` |
| [pipeline.py](../pipeline.py) | 删除 `use_model_a`；构造期 `require()`；`answer_baseline` 也过 Model A；manifest 记录 checkpoint 名与路径 |
| [eval/runner.py](../eval/runner.py) | 删除 `v2_no_model_a` 臂；无条件构建 Model A + `require()`；manifest 增加 checkpoint；补上 `v2_no_rel` 到 `SYSTEM_ORDER`（此前 `SYSTEMS` 有 19 个臂而顺序表只有 18 个，错误提示里看不到它） |
| [stages/analyze.py](../stages/analyze.py) | docstring 更正（不再是"缺模型就降级"） |
| [stages/search.py](../stages/search.py) | `_best_per_chunk`：每路由每 chunk 一票 |
| [retrieval/vector.py](../retrieval/vector.py) | 内容指纹、JSON sidecar、`allow_pickle=False`、4 项加载校验 |
| [config.py](../config.py) | `E5_CANDIDATE_DIRS` 补上 `RAGV2_E5_DIR_VLLM` 指定的模型目录：此前只有 README 写了这个路径，`config.e5_model_dir()` 不认识它，导致 `build_embeddings.py` 在干净进程里报 "multilingual-e5-large not found" |
| [models/embedder.py](../models/embedder.py) | `identity()` / `expected_dim`；`CachedEmbedder` key 归一 |
| [tests/fakes.py](../tests/fakes.py) | `FakeAnalyzer.require()`；新增 `BrokenAnalyzer` |
| [tests/test_pipeline.py](../tests/test_pipeline.py) | 3 个新测试取代 `use_model_a=False` 用例 |
| [scripts/audit_leakage.py](../scripts/audit_leakage.py) | 扫描范围加入 `processed_clean` |
| [scripts/make_clean_split.py](../scripts/make_clean_split.py) | 注释明确"分解文件不参与 Model A 训练"及其 58% 重叠 |
| [scripts/audit_fusion.py](../scripts/audit_fusion.py) | **新增**：量化重复计票的影响 |
| [air/src/model_a.py](../../air/src/model_a.py) | `data_dir` 参数 + `E6` 配置（默认行为不变） |
| [README.md](../README.md) | `v2_no_model_a` 移除、Model A 无条件说明、缓存格式说明、本文链接 |

测试状态：`pytest agentic_rag_v2/tests -q` → **107 passed, 4 skipped**。

---

## 8. 仍未解决 / 下一步

1. **本地 judge 是真瓶颈**（§6.3-2）：2 调用档想追到 0.59，唯一有效方向是训练过的
   本地判定模型（`LocalJudgeAdapter` 已留接口，计划接 JEV 类模型）。
2. **答案边界**：`hp-001` 这类 "right-hand batsman" vs "right-hand" 的近失
   （11 次 EM 里占相当比例）属于 span 边界问题，与检索无关。
3. **轮数上限 4 触顶 21/100**：说明多跳题的证据链常常需要 3 跳以上，
   后续应评估"加深图展开"而非"增加 LLM 轮次"。
4. `decomposition_*` 与 `gen_train_*` 仍是评测同源数据，虽然 Model A 不读，
   但**任何将来要用它们的实验都必须先过 `audit_leakage.py`**。
