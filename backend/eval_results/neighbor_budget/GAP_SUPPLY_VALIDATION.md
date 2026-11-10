# Gap 供给修复 — 样本外验证结果

> 目的：检验 2026-11-06 的挖掘 prompt 改动（禁止配置型/合取型 delta、要求单一机制断言）是否能提高 `NONE` 率。
> 结论：**形态确实改了，但 `NONE` 率没有改变（仍为 0）**。诊断因此前进了一层，并纠正了我上一轮的一个错误结论。

---

## 1. 样本外设计

用 `eval/production_e2e/topics.jsonl` 中**未参与规则拟合**的主题：

| 主题 | 层 | task |
|---|---|---|
| pe2e-003 灾难性遗忘 / PEFT | narrow_mature | `c5fcae2d…` |
| pe2e-008 自一致性幻觉检测 | narrow_mature | `bfc0c0d5…` |

原计划第二个主题是 pe2e-016，因**证据抽取阶段失败**（66 篇论文 / 0 证据，`identical_error_streak`）而替换。该失败发生在挖掘之前，与本次 prompt 改动无关；同批 pe2e-003 抽取正常（334 篇 / 253 证据）。

**必须先做的一件事**：服务进程是 11-03 启动的，而 prompt 于 11-06 15:32 修改。若不重启，跑出来仍是旧 prompt 的结果——那会让整个验证变成自我欺骗。已确认两个服务按新代码重启（`/experiment-config` 核对 neighbors 3/8）。

---

## 2. 结果一：delta 形态确实改变了

| 特征 | 旧 prompt（n=116） | 新 prompt（n=24） |
|---|---:|---:|
| 含数字常量 | 86 (74%) | **0 (0%)** |
| 含 `>= / at least / X%` 配方语 | 69 (59%) | **0 (0%)** |
| 并列 ≥3 个维度名词 | 28 (24%) | 3 (12%) |
| 合取检测器触发 | 7 (6%) | 0 (0%) |
| 平均词数 | 39.4 | 45.9 |

新 delta 全是单一机制断言，例如：

> "The retention advantage attributed to PEFT over full fine-tuning is not a stable property of the method but a small-model artifact: as base-model capacity grows, the gap narrows because the interference that PEFT avoids is absorbed by redundant capacity…"

> "Subspace alignment is not the mediator of forgetting robustness it is assumed to be: the negative alignment-forgetting correlation is an artifact of the low-rank approximation regime…"

**prompt 改动按预期生效了。**

---

## 3. 结果二：`NONE` 率没有变化

盲评 6 个已审计 gap（`validation_verdicts.jsonl`，泄漏检查 CLEAN）：

| 集合 | gap 数 | NONE | PARTIAL | FULL | **T** |
|---|---:|---:|---:|---:|---:|
| 主实验（旧 prompt） | 28 | 0 | 26 | 2 | **0.0%** |
| 样本外（新 prompt） | 6 | 0 | 6 | 0 | **0.0%** |

**形态改了，判定没变。** 这否证了"形态是 `NONE=0` 的唯一原因"这一假设。

评审理由仍然是同一种推理（真实原文）：

> "The retrieved set confirms **the ingredients exist** but does not show anyone applying spectral centrality / rank-aggregation to the self-consistency comparison graph…" —— 判 PARTIAL

> "The retrieved results confirm **the ingredients** of the claimed delta but none measure the specific regime boundary…" —— 判 PARTIAL

即：**即使 delta 已经是单一机制主张，评审仍然用"相邻工作存在但没做这个具体的事"来答 PARTIAL。**

---

## 4. 决定性诊断：评审器能不能给出 NONE？

不能从真实数据推断（可能这些 gap 本来就没有一个够格）。所以做了一个校准探针，喂入两个 gap：

| 探针 | 主张 | 判定 |
|---|---|---|
| `calib-001` | "Transformer 推理行为在**月球重力**（1/6 g）下是否改变" | **NONE**（confidence=high） |
| `calib-002` | "训练脚本的**编程语言**是否影响学到的权重（算术与随机种子固定）" | **PARTIAL** |

`calib-001` 的评审原话：

> "technically unfilled — hence NONE by the letter of the rubric. But this is a **vacuous gap**: it is not a meaningful open problem…"

**结论：评审器能输出 NONE，但只有在该领域完全没有相邻工作时才会。** 只要主题有 300+ 篇相关论文，NONE 就不可达。

### 4.1 机制：rubric 的 PARTIAL 是吸收一切的兜底

```
- NONE:    no prior work covers the claimed delta.
- PARTIAL: prior work covers PART of the delta, **covers it in a different
           setting**, or the delta is a re-parameterisation of known work.
- FULL:    prior work already establishes the claimed delta.
```

中间那句 "**covers it in a different setting**" 由任何相邻论文满足。真实主题永远有相邻论文 → 永远是 PARTIAL。

量化（32 条 PARTIAL）：

| 理由 | 占比 |
|---|---:|
| 部分覆盖了主张 | 50% |
| **有相邻工作（不同设定）** | **50%** |
| 成分覆盖但组合没做（严格匹配） | 8% |
| 不是那个具体 delta | 6% |

---

## 5. 我上一轮的一个错误，必须纠正

上一轮 `GAP_SUPPLY_FIX.md` 报告："**46%（12/26）的 PARTIAL 理由是'成分覆盖但组合没做'**"，并据此把合取型定为主因。

**这个数字是被宽正则夸大的。** 我用的模式是 `combin\w*|conjunct\w*|all (three|four|of)|no single|together|jointly|both`，它命中的很多只是句子里出现了 "both"、"together"、"combines"，并非该论证。严格匹配（`individual ingredients` / `but not the full combination`）只有 **2/26 = 8%**。

影响与不影响的边界：

- **不受影响**：prompt 改动本身是对的。§2 证明要模型写单一机制断言确实可行，配置型/合取型确实被消除了。而且这类形态即使在本次不是主因，也仍然是结构上的死路（配置型不可能 NONE 这一点独立成立）。
- **受影响**：`GAP_SUPPLY_FIX.md` 里"46% 是主因"这一归因**错了**，以及由它导出的"修形态就能让 NONE 出现"这一预期**已被证伪**（§3）。

已在 `GAP_SUPPLY_FIX.md` 顶部加更正标注。

---

## 6. 真正的瓶颈在哪

按证据排列：

1. **rubric 的兜底条款**（已证）：`PARTIAL` 包含"不同设定"，使 NONE 在文献密集领域不可达。这是**仪器问题**，不是 gap 问题。
2. **NONE 的定义过于严格**：要求 "no prior work covers the claimed delta"——在一个成熟主题上，字面意义的"完全没有相关工作"几乎不存在。
3. **文献密度效应**：探针显示只有"完全无相邻工作"的荒谬主张才拿 NONE。

**这三条都不是靠改 prompt 能解决的。** 上一轮把指标问题误判成供给问题，根因是我依赖了一个未校验的正则统计，而没有先做校准探针。**正确顺序应该是：先校准仪器，再诊断数据。**

---

## 7. 下一步（按优先级）

1. **改 rubric 的 NONE/PARTIAL 边界**（必须先做）：把 "covers it in a different setting" 从 PARTIAL 移除或强约束为"覆盖了同一个机制/同一层因果"，否则任何真实主题的 T 都恒为 0。
2. **改主指标**：`T = NONE/(...)` 在有文献的领域结构性为 0。改用「同一 delta 下，评审认为"residual novelty 是否值得做"」或分级 residual 宽度，而不是二分 NONE。
3. **改完后必须重跑校准探针**：至少包含一个"应该判 NONE 的真实 gap"（由人工先判定），用于验证新 rubric 不是同样不可达。

**在 1、2 完成前，不应再跑任何以 `T` 为主指标的实验**，包括 no-audit 消融——它会重演 v7.2 的 TIE。

---

## 8. 产物

- `scripts/_nb_export_validation.py`：样本外导出（复用主实验盲评纪律，独立文件）
- `scripts/_nb_adjudicate.py`：改为 `NB_IN_PATH` / `NB_OUT_PATH` 可覆盖，并修正默认参数在 def 时绑定导致 env 失效的问题
- `eval_results/neighbor_budget/validation_units.jsonl` + `validation_key.json`（SEALED）
- `eval_results/neighbor_budget/validation_verdicts.jsonl`（6 条）
- `eval_results/neighbor_budget/calibration_probes.jsonl` + `calibration_verdicts.jsonl`（关键证据）
