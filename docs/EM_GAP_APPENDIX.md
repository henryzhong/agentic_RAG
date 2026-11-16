## 附录：逐题变更明细

以下表格由 `scripts/make_gap_appendix.py` 直接读结果文件生成，数字不会与批次脱节。

#### A. 排序 A/B —— 同一证据池、同一轮数、同一 judge，只换编号顺序

| # | qid | gold | CE 序 | 分数序 | 谁对 |
|---|---|---|---|---|---|
| 1 | hp-004 | Championship | Football League Championship | Championship | 分数序 |
| 2 | hp-028 | Dutch | Netherlands | Dutch | 分数序 |
| 3 | hp-047 | cocktails | cocktail | cocktails | 分数序 |
| 4 | mq-010 | The Hateful Eight | Deepwater Horizon | Hateful Eight | 分数序 |
| 5 | mq-019 | 323–272 BC | 280 BC and 279 BC | 323–272 BC | 分数序 |
| 6 | mq-026 | Denver | South Park, Colorado | Denver, Colorado | 分数序 |
| 7 | mq-039 | the Politburo | to the Politburo | Politburo | 分数序 |
| 8 | mq-043 | Richland County | Mecklenburg County | Richland County | 分数序 |
| 9 | hp-013 | Macau, China | Nossa Senhora de Fátima, Macau | Nossa Senhora de Fátima, Macau, China | 都错 |
| 10 | hp-022 | Delilah Luke | Delilah Rene | Delilah | 都错 |
| 11 | hp-024 | Mark S. Hoplamazian | UNKNOWN | Mark Hoplamazian | 都错 |
| 12 | mq-001 | Saint Sava | Mita Reljin | UNKNOWN | 都错 |
| 13 | mq-006 | Clifton College | UNKNOWN | Northwestern University | 都错 |
| 14 | mq-012 | Food Safety Modernization Act (FSMA) | UNKNOWN | danger zone | 都错 |
| 15 | mq-013 | Matthew Lawrence | UNKNOWN | Douglas Lawrence Osowski | 都错 |
| 16 | mq-017 | Matt Damon | UNKNOWN | Jason Bourne | 都错 |
| 17 | mq-020 | Jennifer Connelly | UNKNOWN | Kirsten Dunst | 都错 |
| 18 | mq-024 | 2 million | 458 million | 412 million people | 都错 |
| 19 | mq-032 | January 2015 | January 3, 2015 | January 3, 2017 | 都错 |
| 20 | mq-037 | Lawrence Hilton - Jacobs | Joe Jackson | UNKNOWN | 都错 |
| 21 | mq-044 | University of South Carolina | University of South Carolina | 10 | CE 序 |
| 22 | mq-045 | Presque Isle County | Fayette County | UNKNOWN | 都错 |

变化 **22** 题：分数序 赢 **8**，CE 序 赢 **1**，其余两边都错。

注：两批的证据池 100/100 相同、检索轮数 100/100 相同，因此这里每一行都是纯粹的排序效应。

#### B1. KG 三元组注入 A/B —— 2 调用档（分数序 + 本地 judge）

| # | qid | gold | KG 关 | KG 开 | 谁对 |
|---|---|---|---|---|---|
| 1 | hp-037 | 10-metre platform event | 10-metre platform | 10-metre platform event | KG 开 |
| 2 | mq-039 | the Politburo | to the Politburo | Politburo | KG 开 |
| 3 | mq-044 | University of South Carolina | 51 | University of South Carolina | KG 开 |
| 4 | hp-004 | Championship | Premier League | Football League Championship | 都错 |
| 5 | hp-013 | Macau, China | Nossa Senhora de Fátima, Macau, China | Nossa Senhora de Fátima, Macau | 都错 |
| 6 | hp-022 | Delilah Luke | Delilah | Delilah Rene | 都错 |
| 7 | mq-005 | names were replaced by new Russian names | After the expulsion of the German popula | ethnic Russians, Belarusians, and Ukrain | 都错 |
| 8 | mq-006 | Clifton College | UNKNOWN | Western Australian Academy of Performing | 都错 |
| 9 | mq-007 | Mr. Lawrence | Mr. Lawrence | Tom Kenny | KG 关 |
| 10 | mq-010 | The Hateful Eight | Lord of the Rings: The Fellowship of the | UNKNOWN | 都错 |
| 11 | mq-012 | Food Safety Modernization Act (FSMA) | UNKNOWN | Food Safety Modernization Act | 都错 |
| 12 | mq-020 | Jennifer Connelly | UNKNOWN | Kirsten Dunst | 都错 |
| 13 | mq-045 | Presque Isle County | Georgia | UNKNOWN | 都错 |

变化 **13** 题：KG 开 赢 **3**，KG 关 赢 **1**，其余两边都错。

#### B2. KG 三元组注入 A/B —— 最优档（分数序 + LLM judge）

| # | qid | gold | KG 关 | KG 开 | 谁对 |
|---|---|---|---|---|---|
| 1 | hp-004 | Championship | Championship | Football League Championship | KG 关 |
| 2 | hp-013 | Macau, China | Nossa Senhora de Fátima, Macau, China | Nossa Senhora de Fátima, Macau | 都错 |
| 3 | hp-022 | Delilah Luke | Delilah | Delilah Rene | 都错 |
| 4 | hp-050 | Art Deco-style skyscraper | Art Deco-style | Art Deco | 都错 |
| 5 | mq-001 | Saint Sava | UNKNOWN | Milena Reljin | 都错 |
| 6 | mq-007 | Mr. Lawrence | Lori Alan | Tom Kenny | 都错 |
| 7 | mq-012 | Food Safety Modernization Act (FSMA) | danger zone | Food Safety Modernization Act | 都错 |
| 8 | mq-013 | Matthew Lawrence | Douglas Lawrence Osowski | UNKNOWN | 都错 |
| 9 | mq-025 | Maria Bello | Maria Bello | Salma Hayek | KG 关 |
| 10 | mq-026 | Denver | Denver, Colorado | Lami, Fiji | KG 关 |
| 11 | mq-037 | Lawrence Hilton - Jacobs | UNKNOWN | Joe Jackson | 都错 |

变化 **11** 题：KG 开 赢 **0**，KG 关 赢 **3**，其余两边都错。

注：B1 里 KG 净 +2 题、B2 里净 −3 题、CE 序下净 −3 题 —— 三次方向不一致且都在 ±3 题内，属噪声，不构成原因。
