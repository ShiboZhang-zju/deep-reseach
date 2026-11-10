# 固定预算 pilot —— 最终裁决（语义召回修正后，2026-11-02）

> 本文件**取代** `RESULTS_ADJUDICATION.md` 作为主指标结论。旧文件保留作审计轨迹，
> 它记录的 NO-GO 已被证明是**测量故障**的产物，不是实验结论。
>
> 仪器版本：pass 3（语义召回）。改动只涉及**检索**，判定 rubric 与 §7 规则**逐字未改**。

---

## 0. 结论

**按 §7 冻结规则：T_A = T_B = 0.0%，delta = 0.0pp → TIE，§7.3 默认判 NO-GO。**
即：本轮实验**没有发现** v7.2 比 v6 产出更多可发表机会的证据。

但必须同时说明：**这个 TIE 是"无区分力"，不是"已证明等价"**——两臂的 T 都是 0.0%，
指标没有余量去区分它们。两臂各自的 12 / 25 个 gap 中，**没有任何一个**被独立评审
判为"可发表空白"。

---

## 1. 主指标

| 项 | 臂 A（v7.2） | 臂 B（v6） |
|---|---:|---:|
| gap 单元数 | 12 | 25 |
| NONE（可发表空白） | **0** | **0** |
| PARTIAL（需收窄） | 11 | 22 |
| FULL（已被覆盖） | 1 | 3 |
| **T = NONE/(NONE+PARTIAL+FULL)** | **0.0%** | **0.0%** |
| FULL 率（被覆盖） | 8.3% | 12.0% |
| 评审置信度 | high 1 / medium 11 | high 3 / medium 22 |

**逐主题配对：**

| topic | A n | A NONE | A FULL | B n | B NONE | B FULL |
|---|---:|---:|---:|---:|---:|---:|
| pe2e-001 | 2 | 0 | 0 | 4 | 0 | 0 |
| pe2e-007 | 2 | 0 | 0 | 6 | 0 | **2** |
| pe2e-013 | 2 | 0 | 0 | 3 | 0 | 0 |
| pe2e-015 | 2 | 0 | **1** | 5 | 0 | 0 |
| pe2e-019 | 2 | 0 | 0 | 2 | 0 | 0 |
| pe2e-024 | 2 | 0 | 0 | 5 | 0 | **1** |

六个主题上一臂都没有 NONE，配对差值全为 0。

## 2. Idea 层

| 判决 | 臂 A（n=40） | 臂 B（n=14） |
|---|---:|---:|
| fund | **0** | **3** |
| maybe | 40 | 11 |
| no | 0 | 0 |

臂 B 有 3 个 Idea 被评为"值得做"，臂 A 是 0。**方向与管线自评相反**
（管线给出的 executable 是 A=9 / B=2）。

⚠️ 但 A 的 40 个单元里绝大多数是 `research_direction_only`（无 method_sketch 的方向性
想法），与 B 的可执行 Idea 不同质，**这个对比需要先分层才成立**。

## 3. 仪器为什么换了一版（这是全部争议的核心）

### pass 1（关键词检索）的判决不可用

| 证据 | 数值 |
|---|---|
| 评审员自述"检索无用"的单元 | **6/37** |
| 承载全部 -4.0pp 差异的单元 | **1 个**（`fbp-089`） |
| `fbp-089` 检索回的 12 篇里与其主题相关的 | **0 篇**（全是统计因果中介/医疗 GNN） |
| `fbp-089` 的 `key_prior_work` | `[]` 空 |
| 剔除 `fbp-089` 后的 delta | **0.0pp** |

`fbp-089` 自述原文：*"Search was largely off-target … absent are the canonical
circuit-discovery papers (activation/path patching, IOI, hydra-effect) that
would be the true nearest neighbours."*

**一个自身承认检索失败的单元，不能承载整个实验的判决。**

### pass 2（用近邻标题重建查询）也失败

去副标题后只剩泛化词（"Context-Adaptive Inference"），检索回一堆只匹配
"inference" 的论文。评审员改口 *"essentially useless"*，置信度降到 low。

### pass 3（语义召回）的阈值是**测出来的**

`scripts/_calibrate_sim.py`，37 个真实 gap：

| 分布 | n | median | p10 | max |
|---|---:|---:|---:|---:|
| claim vs 它自己的审计近邻（**应保留**） | 111 | **0.507** | 0.376 | 0.715 |
| claim vs `fbp-089` 的已知噪声池（**应滤除**） | 12 | 0.332 | 0.264 | **0.408** |

阈值 **0.41**：保留 **82%** 真实近邻，滤除 **100%** 已知噪声。两个分布尾部有重叠，
所以这是**软过滤**，`kept`/`filtered` 计数强制写进每份判决并透露给评审员。

**修复效果（硬证据）：**

| 指标 | pass 1 | pass 3 |
|---|---:|---:|
| 评审自述检索无用 | 6/37 | **0/37** |
| `fbp-089` 判决 | NONE | **PARTIAL** |
| `fbp-089` 检索回的前 12 篇 | 0 篇相关 | **12 篇全部相关**（sim 0.584–0.660） |
| FULL 数 | 1 | **4**（全部 high 置信） |

`fbp-089` 修复后检索到的第一篇正是
*"From Observation to Intervention: A Causal Audit of Expert Importance in
Mixture-of-Experts"*——它的**直接杀手文献**，pass 1 完全没找到。

## 4. 管线自评 vs 独立评审（校准）

| 管线审计结论 | 独立评审 | 单元数 |
|---|---|---:|
| confirmed | PARTIAL | **28** |
| uncertain | PARTIAL | 5 |

**33 个"confirmed"里 0 个被判 NONE、0 个被判 FULL**——在这个样本上，
管线说的 `confirmed` 与"存在可发表空白"几乎无关。这是本轮最稳的结论，
因为它不依赖两臂对比。

（pass 1 有 1 个 confirmed→FULL、1 个 confirmed→NONE，pass 3 两者都消失了，
进一步说明那两个是检索噪声。）

## 5. 明确不能引用的结论

1. ❌ **"v7.2 优于 v6"** —— T 都是 0.0%，delta 0.0pp；Idea 层方向相反。
2. ❌ **"v6 优于 v7.2"** —— FULL 3 vs 1、Idea fund 3 vs 0 是唯一有利信号，
   但 n=25 vs 12，且臂 A 每主题只审 2 个 gap、臂 B 审 4–6 个，
   **审计量不对称**使单位比较失效。
3. ❌ **"两臂等价"** —— T=0 的平局是无区分力，不是等价性证明。
4. ❌ **"本体论无效"** —— 本轮设计无法分离本体论效应与审计量效应。

## 6. 设计缺陷（下一轮必须先修）

| 缺陷 | 影响 | 修法 |
|---|---|---|
| **审计量不对称**（A 每主题 2 个 gap，B 4–6 个） | 所有跨臂比率不可比 | 固定"每臂每主题审计 N 个"或按已审数量归一 |
| **T 指标无余量**（两臂都 0 NONE） | 无法检测差异 | 改用序数量表：FULL/PARTIAL/NONE 加权，或直接比 FULL 率 |
| **Idea 层不同质**（方向性 vs 可执行） | fund 率不可比 | 分层报告，或只比同 tier 的 Idea |
| **killer 检索仍走 topic 锚点** | 3 次执行 0 命中 | 锚点改为 killer 描述（见 §7） |

## 7. 仍未接入语义召回的地方

`search_papers.py::_prefilter_by_similarity` 已存在，但它用 **topic** 作锚点，
所以 killer 检索（查询是 killer 描述）仍会被按 topic 过滤，收窄不到 killer。
本轮**未改**这一点（改它会动到所有任务的检索主路径，风险高且需全量回归）。
建议的下一步：给 `_prefilter_by_similarity` 增加可选 `anchor_text` 参数，
killer 路径传 `closest_killer_work`。

## 8. 人工抽查（§8.2，仍未执行）

抽样 18/91（20%，分层，种子 20261102）：

```
fbp-007 fbp-014 fbp-015 fbp-023 fbp-028 fbp-029 fbp-030 fbp-039
fbp-053 fbp-054 fbp-055 fbp-070 fbp-075 fbp-077 fbp-079 fbp-085
fbp-087 fbp-091
```

现在**可以**抽查了（检索已修，不再让人重复同一个盲点）。规则：一致率 ≥70% →
LLM 判决可用；<70% → 作废转人工。
