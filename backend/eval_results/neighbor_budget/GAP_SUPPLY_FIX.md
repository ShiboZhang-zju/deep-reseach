# Gap 供给修复（2026-11-06）

> ## ⚠️ 更正（2026-11-09，见 `GAP_SUPPLY_VALIDATION.md`）
>
> **本文 §1.2 报告的"46%（12/26）PARTIAL 理由是'成分覆盖但组合没做'"是错的。**
> 该数字来自一个过宽的正则（`combin\w*|conjunct\w*|together|jointly|both|...`），
> 误把句子里出现 "both"、"together"、"combines" 当作该论证。严格匹配只有 **2/26 = 8%**。
>
> 真正的主流理由（32 条 PARTIAL 的统计）是：
> **"部分覆盖了主张" 50%** 与 **"存在相邻工作（不同设定）" 50%** ——
> 后者正是 rubric 中 PARTIAL 的兜底条款。
>
> **受影响的结论**：由"46% 是主因"导出的预期——"修掉合取/配置形态就能让 NONE 出现"——
> **已被样本外验证证伪**：新 prompt 下 delta 形态确实全部改为单一机制断言
> （数字 74%→0%，配方语 59%→0%），但盲评 6/6 仍是 PARTIAL，`NONE` 仍为 0。
>
> **不受影响的结论**：prompt 改动本身有效且应予保留（配置型/合取型确实被消除）；
> 且配置型"结构上不可能判 NONE"这一推理独立成立。
>
> **根本原因**：`NONE` 不可达是**评测仪器（rubric）问题**，不是 gap 供给问题。
> 校准探针证明：评审器能输出 NONE，但仅当该领域**完全没有相邻工作**时
> （月球重力探针→NONE；编程语言探针→PARTIAL）。
> 详见 `GAP_SUPPLY_VALIDATION.md` §4。

> 起因：近邻预算实验的独立盲评给出 `NONE=0 / PARTIAL=26 / FULL=2`（28 个 gap），
> 主指标 `T = NONE/(...)` 在两臂都是 0.0%，**没有余量**（`RESULTS.md` §2.1）。
> 上一轮已确认"换指标救不回来"（`METRIC_REDESIGN.md` §4），所以必须修供给。

---

## 1. 先诊断，不猜

排除掉的假设：

| 假设 | 检验 | 结论 |
|---|---|---|
| gap 类型太单一 | 116 个 gap 覆盖 7 类（boundary 25 / cross_domain 19 / missing_eval 16 / scaling 16 / untested 14 / contradiction 13 / mechanism 13） | **排除**，分布均衡 |
| 证据来源单一 | limitation 150 / negative_result 96 / comparison 94 / method 74 / result 49 / metric 44 / problem 43 / future_work 39 / dataset 34 | **排除**，来源多样 |
| 检索不足 | 3 vs 8 近邻对照 `ΔT = +0.0pp` | **排除**，加近邻无效 |

真正的原因在**`claimed_delta` 的表述形态**上。分两步查实：

### 1.1 第一步：形态统计（116 个真实 gap）

| 特征 | 占比 |
|---|---:|
| 含数字常量 | 74% |
| 含 `>=` / `at least` / `N datasets` | 47% |
| 含 ≥3 个并列子句 | 49% |

### 1.2 第二步：读 PARTIAL 的评审理由（**已更正，见顶部**）

初版这里报的是"46% = 成分覆盖但组合未做"，那个数字是**错的**。错误出在方法上：当时用的是宽正则（`combin\w*|conjunct\w*|together|jointly|both|...`），句子里出现 "both"/"together"/"combines" 就被计入，而那不是该论证；且各条并非互斥分类，被当成互斥来加总。

严格匹配（`individual ingredients` / `but not the full combination`）后，真实分布（32 条 PARTIAL，含样本外 6 条）：

| 评审给 PARTIAL 的理由 | 占比 |
|---|---:|
| 部分覆盖了主张 | 50% |
| **存在相邻工作（不同设定）** | **50%** |
| 成分覆盖但组合没做（严格匹配） | **8%** |
| 不是那个具体 delta | 6% |

即：主流理由是"**相邻工作存在**"——而这正是 rubric 里 PARTIAL 的兜底条款。

评审原话（真实记录，这条确实存在，只是不占 46%）：

> "The retrieved literature substantially covers **the individual ingredients** of the claimed delta **but not the full combination**."

---

## 2. 根因：两种 gap 形态在结构上不可能被判为 NONE

| 形态 | 例子 | 为什么必然是 PARTIAL |
|---|---|---|
| **配置型** | "在 ≤1B backbone 上、在 NarrativeQA 上、在 >30% vs <10% 压缩率下" | 主题已被覆盖，只是某个格子没人报 |
| **合取型** | "首次同时做 A + B + C" / "(i) 且 (ii) 且 (iii)" | 每个成分都有人发过，答案永远是"成分有、组合无" |

这两种形态下，评审的诚实回答只能是 PARTIAL：
- 不可能是 NONE（主题确实存在）
- 不可能是 FULL（没人报过那个格子/那个组合）

**这是表述形态的性质，不是检索的不足。** 所以加近邻无效（实测 +0.0pp），换指标也无效。

而且原 prompt 在**鼓励**这种形态：

```
- state what existing work already covers and the smallest missing capability;
- quantifiers in testable_hypothesis and claimed_delta must be concrete numbers
  (">10% label noise", "<3B parameters").
```

第二行直接要求写具体数字，等于把 delta 推向配置型。

---

## 3. 改动

### 3.1 主修复：挖掘 prompt 的结构性约束（`_GAP_MINING_SYSTEM`）

新增三条：

1. **明确禁止两种形态**，并说明为什么（评审只能答 PARTIAL），要求把配置/组合写进 `missing_capability` 后**丢弃候选**，而不是包装成 capability。
2. **要求 delta 是单一机制断言**："X causes Y through Z" / "Z is not the mediator it is assumed to be" / "the effect reverses because W"。mechanism、因果归因、被反驳的假设——**一个**，不能是并列。
3. **自检**：
   - 删掉所有数字、数据集名、模型尺寸后，delta 仍须断言一个机制；
   - 删掉任意一个子句后，delta 必须变为假或无趣（即"存活下来"说明它是合取）。

这是**真正的修复**：改的是让模型产出什么，而不是事后过滤。

### 3.2 辅助：合取检测器 + 门（**默认关闭**）

`_is_conjunction_only_delta` + `gap_mining_conjunction_gate`。

**默认关闭**，理由与代码库中已有的 `operationalization` 门完全一致（该门注释自己写明：规则从同一批样本拟合而来，启用需样本外验证）。本规则同样是在它要评分的 28 个单元上拟合的，现在启用就是重犯 v7.3 的错。

实测能力，如实记录：

| 指标 | 值 |
|---|---|
| 触发率 | 7/116 = **6.0%**（低召回） |
| 触发样本的准确性 | 手工核对 7/7 确为合取形态（**高精确**） |

第一次实现（剥离数字后判断残留内容）**触发 0/116**，说明测错了形态，已删除不保留。

---

## 4. 修复过程中真实犯的两个错（都加了测试）

**错误 1：prompt 里的字面花括号导致挖掘崩溃。**
我最初把占位写成 `{i}` / `{ii}`，而 `_GAP_MINING_SYSTEM` 会经过 `str.format()`，直接 `KeyError: 'i'`，**每个 lens 调用都会挂**。全量回归立刻抓到（5 个 v7.2 测试失败）。
→ 已改为 `(1) and (2) and (3)`，并新增 `test_mining_prompt_formats_without_stray_braces`，用运行时同样的 `.format()` 调用验证。

**错误 2：正则排除句点，导致含小数点的正常 delta 永不触发。**
`[^.;!?]` 把 "2." 中的句点也排除了。
→ 已改为允许非句末句点，并新增测试固定行为。

---

## 5. 必须明确：这不能证明修复有效

**本次改动没有任何证据表明它会提高 NONE 率。** 要证明，需要：
1. 用新 prompt 重跑一批主题（样本外）；
2. 独立盲评；
3. 观察 NONE 是否出现非零。

在那之前，只能说：**造成了 `NONE=0` 的那个结构性原因已经被识别，并针对它做了改动**。这正是本项目预注册纪律要求的表述方式——不做未验证的机制声称。

---

## 6. 与指标改造的关系

顺序不可颠倒（`METRIC_REDESIGN.md` §5）：

1. **改度量**：已完成。主指标改为三层（机制层同口径边际增益 / 判定层有序分布 / 质量层 fund 计数），`T` 降为描述量。
2. **修供给**：本次。目标是让判定层出现 NONE，使量表重新有余量。
3. **再做 no-audit 消融**：只有在第 2 步成功后才有意义。

若第 2 步后 NONE 仍为 0，则说明问题不在 delta 形态而在更上游（例如主题本身已被充分研究、或证据抽取没有提供"没人做过"的材料），需要重新诊断而不是继续调 prompt。

---

## 7. 产物

- `app/agent/steps/mine_gaps.py`：`_GAP_MINING_SYSTEM` 三条新约束；`_is_conjunction_only_delta`；`_settings_conjunction_gate`；拒绝码 `CONJUNCTION_ONLY_DELTA`
- `app/config.py`：`gap_mining_conjunction_gate = False`（含完整诊断注释）
- `tests/test_gap_supply_delta_shape.py`：16 项（诊断数字、prompt 契约、format 安全、检测器行为、门默认关闭）
- 全量测试：**783 passed**（原 767 + 16）
