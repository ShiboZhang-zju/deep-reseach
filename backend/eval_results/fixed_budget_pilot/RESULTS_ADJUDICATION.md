# 固定预算 pilot —— 主指标裁决报告（2026-11-02）

> 生成：`python scripts/adjudicate_final.py`（只读 verdicts/mapping）
> 原始数据：`adjudication_input.jsonl`（91 单元）、`verdicts.jsonl`（pass 1）、
> `verdicts.pass2.jsonl`（检索修复重评 6 单元）、`review_mapping.json`（已解封）

---

## 0. 一句话结论

**按预注册 §7 规则，本轮判定为 NO-GO；但该判定不具备决策效力。**
整个 -4.0pp 差异由一个单元（`fbp-089`）承载，而该单元的两次检索都失败、
评审员自己声明"搜索结果无用"并把置信度降到 low。真正被这轮实验证实的
不是 v7.2 或 v6 的优劣，而是**评审工具的检索环节对小众主题失效**——
这必须先修，否则任何臂间比较都无意义。

---

## 1. 主指标（三种口径并列，不择一）

| 口径 | 臂 A（v7.2） | 臂 B（v6） | delta | §7 分支 |
|---|---:|---:|---:|---|
| **V1 原样（预注册）** | T=**0.0%**（n=12） | T=**4.0%**（n=25） | **−4.0pp** | **NO-GO**（T_A < T_B） |
| **V2 剔除检索失败单元** | 0.0%（n=12） | 0.0%（n=19） | **+0.0pp** | **TIE**（无可分辨证据） |
| **V3 用修复后重评替换** | 0.0%（n=12） | 4.0%（n=25） | −4.0pp | NO-GO（但关键单元 low 置信） |

T = NONE / (NONE + PARTIAL + FULL)，沿用 v7 replay pilot 的判定口径。

**部分分敏感性**（同一批数据，换计分方式）：

| 计分 | A | B | delta |
|---|---:|---:|---:|
| PARTIAL=0.0（预注册） | 0.0% | 4.0% | −4.0pp |
| PARTIAL=0.5 | 45.8% | 52.0% | −6.2pp |
| PARTIAL=0.75 | 68.8% | 76.0% | −7.3pp |

三种计分下 delta 都在 −4 ~ −7pp，**方向一致（A 略低或持平）**，但幅度全部落在
§7 的 ±10pp 平局带内——除 V1 因 T_A=0 的边界效应被规则读成 NO-GO。

## 2. 判决分布：几乎无方差

| 臂 | n | NONE | PARTIAL | FULL | 熵 |
|---|---:|---:|---:|---:|---:|
| A（v7.2） | 12 | 0 | **11** | 1 | 0.41 bits |
| B（v6） | 25 | **1** | **24** | 0 | 0.24 bits |

- 两臂都 ~95% PARTIAL → 预注册指标（NONE 计数）几乎没有区分度。
- 对照 v7 replay pilot：NONE 6.7% / PARTIAL 70.0% / FULL 23.3%。
  本轮 FULL 只有 2.7%（pilot 23.3%），PARTIAL 从 70% 涨到 94.6%。

**PARTIAL 不是兜底档（已核查）**：35 个 PARTIAL 的 `residual_novelty` 全部各不相同
（35/35 distinct，长度 394–907 字符，无短于 40 字符的），notes 两两相似度仅 0.06，
`key_prior_work` 普遍非空。所以判决内容是真实的，问题在于**判定门槛的合取偏好**：
评审员只要"三个要素没有同时被覆盖"就判 PARTIAL，而 pilot 会判 FULL。
36/37 的 residual 或 notes 都出现合取语言（and / both / combination of）。

## 3. 关键单元 `fbp-089`（承载全部差异）

| 项 | pass 1 | pass 2（检索修复） |
|---|---|---|
| 主题 / 臂 | pe2e-015 (MoE routing) / **B** | 同 |
| 判决 | **NONE**, medium | **NONE**, **low** |
| search_useful | `true`（但正文承认偏离） | **`false`** |
| key_prior_work | `[]` **空** | `[]` **空** |
| 检索结果 | 12 篇全是统计因果中介 / 医疗 GNN，**零篇 MoE 电路发现** | 12 篇全是 "…Inference" 泛化论文，**零篇相关** |

pass-1 自述："Search was largely off-target … absent are the canonical
circuit-discovery papers (activation/path patching, IOI, hydra-effect)
that would be the true nearest neighbours."
pass-2 自述："Search results are essentially useless for this claim …
Recommend re-running retrieval with domain-specific queries."

**剔除该单元：A=0.0%、B=0.0%、delta=+0.0pp（TIE）。**

## 4. 检索修复为什么也失败

pass-2 改用**数据库中真实近邻论文标题**构造查询（本应是正确做法，也是
killer 检索该用的同一修法），但近邻标题本身是
"Context-Adaptive Inference: A Unified Statistical and Foundation-Model View"
"Routers Learn the Geometry of Their Experts"——去副标题后只剩泛化词
（inference / geometry），检索器按词面匹配回一堆无关论文。

**根因**：该主题的近邻论文本身偏理论/统计，标题承载的领域词不足；
纯词面检索（S2/OpenAlex/arXiv 的 keyword search）无法表达
"MoE router causal intervention / hydra effect" 这类组合语义。
这是 README 瓶颈 #4（39% 候选零词面重合）在**评审侧**的同一表现。

## 5. 6 个检索失败单元（全部在臂 B）

`fbp-037 / 058 / 071 / 085 / 087 / 089`：pass-1 有 5 个 `search_useful=false`，
`fbp-089` 布尔为 true 但正文承认失败。pass-2 重评后 5 个维持 PARTIAL、
`fbp-089` 维持 NONE 但降为 low。

**注意这本身是一个混淆**：检索失败集中在臂 B。若检索能力对两臂不等价，
主指标就不是在测"机会质量"，而是在测"主题难度 + 检索鲁棒性"的混合。

## 6. Idea 层（次指标）

| 判决 | 臂 A（n=40） | 臂 B（n=14） |
|---|---:|---:|
| fund | **0** | **2** |
| maybe | 40 | 12 |
| no | 0 | 0 |

与次指标报告的表面结论（A 9 个 executable vs B 2 个）**完全相反**：
管线自己的门禁认为 A 更好，独立评审认为 B 的 Idea 更值得做。
但 A 的 40 个 maybe 里有大量 `research_direction_only`（无 method_sketch 的方向性想法），
与 B 的可执行 Idea 不可直接比较——**这个对比本身也需要分层后才成立**。

## 7. 管线自评 vs 独立评审（校准问题）

| 管线审计结论 | 独立评审 | 单元数 |
|---|---|---:|
| confirmed | PARTIAL | **29** |
| confirmed | NONE | 1 |
| confirmed | FULL | 1 |
| uncertain | PARTIAL | 6 |

**31 个"confirmed"中只有 1 个被独立评审确认完全被覆盖（FULL）**，其余 29 个降级为
PARTIAL。这说明管线的 `confirmed` 语义是"近邻未覆盖"，而非"新颖"——
与代码注释里的自我声明一致，但**从未被这样的数据量化过**。

## 8. 这轮实验到底建立了什么

**成立的结论：**

1. 实验基础设施可用：12 次运行、stamp 正确、无工程故障、盲评 91 单元完成、
   mapping 可解封、判定规则可机械执行。
2. 管线自评与独立评审存在系统性差距（confirmed→PARTIAL 29/31），
   这个差距现在被量化了。
3. 两臂的 gap 判决分布都极度集中于 PARTIAL，预注册的 NONE 计数指标
   **在 6 主题规模上不具备区分力**（熵 0.24–0.41 bits）。

**不成立的结论（不得引用）：**

1. ❌ "v6 优于 v7.2"——差异来自一个检索失败的单元，剔除后为 0。
2. ❌ "v7.2 优于 v6"——次指标的表面优势（9 vs 2 executable）被审计量混淆
   （A 9/12，B 2/25），且与独立评审方向相反（fund: A=0, B=2）。
3. ❌ "本体论无效"——本轮没有能力测出本体论效应。

## 9. 必须先修的三件事（按阻塞顺序）

1. **评审检索（阻塞性）**：keyword search 对小众组合语义失效。
   候选方案：① 用 embedding 检索（README 瓶颈 #4 建议的同一改动，可复用
   `embedding_service`）；② S2 的 `paper/search/match` 用摘要片段而非标题；
   ③ 引入 OpenAlex 的概念（concept）过滤。**先修这个，否则重跑也是同样结果。**
2. **判定门槛的合取偏好**：给 rubric 增加"若 prior work 覆盖了 delta 的各要素
   但未覆盖其组合，判 PARTIAL 还是 FULL"的明确规则，并在 pilot 数据上回归校准
   （用 pilot 的 21 个 PARTIAL / 7 个 FULL 做对照）。
3. **检索失败与臂的混淆**：6/6 检索失败全在臂 B。需检查这是偶然还是
   臂 B 的主题/证据形态所致；若是后者，主指标必须在**同检索条件下**比较。

## 10. 人工抽查（§8.2，未完成）

抽样 18/91（20%，按 unit_type × arm 分层，种子 20261102）：

```
fbp-007 fbp-014 fbp-015 fbp-023 fbp-028 fbp-029 fbp-030 fbp-039
fbp-053 fbp-054 fbp-055 fbp-070 fbp-075 fbp-077 fbp-079 fbp-085
fbp-087 fbp-091
```

**未执行**——这一项必须由人完成。规则：一致率 ≥70% → LLM 判决可用；
<70% → LLM 判决作废，转全量人工。

考虑到 §3/§4 的检索失败，**我的建议是先修检索、再抽查**：
现在抽查等于让人去判"检索没找到证据"的单元，会重复同一个盲点。
