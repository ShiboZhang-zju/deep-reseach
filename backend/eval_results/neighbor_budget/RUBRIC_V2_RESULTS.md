# Rubric v2 — 改动结果与门禁

> 日期：2026-11-09。预注册见 `RUBRIC_V2_PREREG.md`（**先于任何 v2 运行写定**）。
> v1 未改动（`adjudicate_pilot.RUBRIC`），历史数字可复现。

---

## 1. 结论

**v2 通过全部门禁，并让 `T = NONE/(...)` 第一次有余量：0.0% → 20.6%。**

但它同时产生了**与我人工判断的 2 处冲突**，我判定 **v2 对、我错**（理由见 §5）。这意味着我此前那份 21 单元的人工抽查受到了 v1 缺陷的污染。

---

## 2. 门禁结果（三道全过）

| 门 | 条件（预注册冻结） | 实测 | 结果 |
|---|---|---|---|
| G1 | 正向探针 ≥1/2 判 NONE | **1/2** | PASS |
| G2 | 反向探针 3/3 判 FULL，且无 1 个 NONE | **3/3，NONE=0** | PASS |
| G3 | verdict 与 delta_established 一致 ≥4/5 | **5/5** | PASS |

### 2.1 探针明细

| ID | 预期 | 实测 | delta_established | adjacent | 命中 |
|---|---|---|---|---|---|
| `p2v-N1`（语言是否中介权重） | NONE | **NONE** | no | True | ✅ |
| `p2v-N2`（均衡梯度是否因果决定专业化判定） | NONE | PARTIAL | partly | True | ❌ 预期错 |
| `p2v-R1`（Transformer 纯注意力） | FULL | **FULL** | yes | True | ✅ |
| `p2v-R2`（LoRA 低秩） | FULL | **FULL** | yes | True | ✅ |
| `p2v-R3`（Chain-of-Thought） | FULL | **FULL** | yes | True | ✅ |

**`p2v-N2` 判 PARTIAL 是对的，我的预期错了。** v2 检索到 `[10] "Routing in Gradient Space: Balanced Usage Is Not Expert Specialization"`，该文已建立我假设"未被检验"的机制。这说明 v2 在正常工作，不是退化。

**关键点**：G1 与 G2 同时通过，证明 v2 **既能产出 NONE，也不会退化成永远 NONE**。这正是 v1 缺失的性质。

---

## 3. 配对重测：34 个 gap，v1 vs v2

同一批单元、同一批检索，只换仪器。

### 3.1 转移矩阵

| v1 → v2 | NONE | PARTIAL | FULL |
|---|---:|---:|---:|
| **NONE** | 0 | 0 | 0 |
| **PARTIAL** | **7** | 25 | 0 |
| **FULL** | 0 | 1 | 1 |

### 3.2 边际分布

| | v1 | v2 |
|---|---:|---:|
| NONE | 0 | **7** |
| PARTIAL | 32 | 26 |
| FULL | 2 | 1 |
| **T (NONE 率)** | **0.0%** | **20.6%** |

### 3.3 内部一致性

`verdict == f(delta_established)`：**34/34**。三个字段完全自洽（NONE↔no, PARTIAL↔partly, FULL↔yes）。

### 3.4 v1 无法表达的情形，现在可见了

**7 个单元判为 NONE 且 `adjacent_work_exists=True`** —— 即"同主题很热闹，但 delta 未被检验"。这正是 v1 结构上无法输出的格子：

| 单元 | v1 | v2 |
|---|---|---|
| nb-015 | PARTIAL | NONE |
| nb-018 | PARTIAL | NONE |
| nb-046 | PARTIAL | NONE |
| nb-070 | PARTIAL | NONE |
| nb-080 | PARTIAL | NONE |
| nb-099 | PARTIAL | NONE |
| val-001 | PARTIAL | NONE |

另有 1 例 `nb-066: FULL → PARTIAL`（v2 认为只覆盖了一部分）。

---

## 4. 8 个判定发生变化

7 个 `PARTIAL→NONE`，1 个 `FULL→PARTIAL`。变化率 8/34 = 23.5%。

**没有一个是 `NONE→PARTIAL`**，说明 v2 没有把宽松换成另一种宽松，而是把兜底条款收紧了。

---

## 5. 与我人工判断的冲突 —— 我判定 v2 对、我错

我在 2026-11-06 手工盲判过 21 个单元（`spotcheck_judgements.json`），当时与 v1 **21/21 一致**。本次 v2 重评覆盖其中 5 个：

| 单元 | 我的人工判断 | v2 | 是否一致 |
|---|---|---|---|
| nb-010 | PARTIAL | PARTIAL | ✅ |
| nb-022 | PARTIAL | PARTIAL | ✅ |
| nb-039 | PARTIAL | PARTIAL | ✅ |
| **nb-099** | **PARTIAL** | **NONE** | ❌ |
| **nb-015** | **PARTIAL** | **NONE** | ❌ |

**v2 与我的一致率只有 3/5。**

### 5.1 我为什么判定 v2 对

**nb-099**（压缩质量变化的因果中介）。我当时的理由：
> "many compression frameworks exist, none runs the claimed 3-signal x 2-rate factorial mediation ablation."

v2 的判断更细致：
> "[3] COMI introduces MIG and compares it to relevance (AUC 0.58 vs 0.54) and reports redundancy levels **at 25%/50% retention** — this is the closest work and supplies the *inputs* to the ablation… but it **does not hold span count fixed while swapping signals**, does not include a random control, and does not decompose the EM/F1 delta into mediating factors."

关键差别：v2 检索到 COMI 在**完全相同的 25%/50% 保留率**上量化了冗余，并据此把 novelty 收窄到"因果归因这一步"，还主动降低置信度到 medium：
> "COMI's redundancy-vs-relevance measurements at exactly 25%/50% mean the proposed experiment **may yield a confirmatory rather than surprising result**."

**我的判断漏掉了 COMI 这个对撞证据**。我是在没有重新检索的情况下凭单元文本判断的，而 v2 每次都做独立检索。

**nb-015**（均衡损失的路由器/专家通道分解）同理：v2 逐篇点名 `[4][8][9][10][12][3][1]` 并说明每篇为何不构成该分解，而我的理由只有一句概括。

### 5.2 但这暴露了一个必须承认的问题

**我此前报告"人工抽查 100% 一致率"时，曾正确指出"该数字被众数抬高、不能证明评审器可靠"。现在这个 3/5 恰好证实了那个警告是对的** —— 我当时与 v1 一致，不是因为 v1 正确，而是因为**我也在用 v1 的隐含标准（主题有相邻工作 → PARTIAL）**。

即：**我的人工抽查不是独立的**。它被同一套 rubric 锚定了。

---

## 6. 这对之前结论的影响（必须逐条说明）

| 之前的结论 | 是否受影响 |
|---|---|
| 近邻预算实验 `ΔT=+0.0pp`，判"预算不是天花板" | **判定本身不变**（v1 下两臂 T 都是 0），但**该判定本来就几乎无信息量**——现在有了 v2 才可能真正检验。**不能用 v2 结果反推该实验结论，因为那批单元已用 v1 评过** |
| "`NONE` 率作为主指标无余量" | **成立且已修复**：v2 下 T=20.6%，有余量 |
| "瓶颈在仪器而非 gap 供给" | **成立**，并已通过换仪器验证 |
| "人工抽查 100% 一致率不能证明可靠" | **成立，且被证实**：v2 下我对同一批评审只有 3/5 |
| "48%/46% PARTIAL 理由是组合型" | 已在 `GAP_SUPPLY_FIX.md` 更正为 8% |

**明确不做的事**：我不会用 v2 的 20.6% 去替换近邻实验的 0.0%。那是**换仪器后重测**，不是新数据；混用会制造"结论变了"的假象。

---

## 7. 下一步

1. **用 v2 重跑近邻预算对照的判定层**（`T` 作为次指标），而不是回填旧值。这需要重新评审两臂单元——**可以复用已存单元，但必须整批用 v2 重评**，不得新旧混用。
2. **重新做人工抽查**，这次**先读 v2 rubric 但先不看 v2 判定**，用独立判断校验 v2；目标是把"评审器是否可靠"这个问题第一次真正测准。
3. 之后才谈 no-audit 消融。

---

## 8. 产物

```
scripts/adjudicate_rubric_v2.py        v2 rubric + 判定 schema（含两个诊断字段）
scripts/_nb_adjudicate_v2.py           可切换 v1/v2 的评审器
scripts/_nb_make_v2_probes.py          探针生成（预期封存在 key 文件）
eval_results/neighbor_budget/
  RUBRIC_V2_PREREG.md                  预注册（门槛冻结）
  rubric_v2_probes.jsonl               5 个探针
  rubric_v2_probes_key.json            SEALED 预期
  probe_verdicts_v2.jsonl              探针结果
  rev2_units.jsonl                     34 个重评单元
  rev2_verdicts.jsonl                  34 条 v2 判定
```
