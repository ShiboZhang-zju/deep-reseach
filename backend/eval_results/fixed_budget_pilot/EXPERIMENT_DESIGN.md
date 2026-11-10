# v7.2 固定预算生产价值实验 — 运行方案（预注册，2026-10-31）

> 状态：**完成（2026-11-02）**。12 次运行（6 主题 × 2 臂）全部完成且完整性核查通过；91 个单元完成三轮评定：pass 1 关键词检索 → pass 2 标题重建查询（失败）→ **pass 3 语义召回（最终仪器）**。
> **最终结论见 `RESULTS_FINAL.md`**（取代 `RESULTS_ADJUDICATION.md`；后者保留为审计轨迹，其 NO-GO 已证明是测量故障的产物）。次指标见 `RESULTS_SECONDARY.md`。
> 判定规则（§7）在看到任何结果前写死，**三轮评定中逐字未改**；运行矩阵见 `run_matrix.json`。
> **最终裁决：T_A = T_B = 0.0%，delta = 0.0pp → TIE，§7.3 默认 NO-GO。** 但这是"无区分力"而非"已证明等价"——两臂各自 12/25 个 gap 中**没有一个**被判为可发表空白。真正稳的结论是：管线自评的 `confirmed` 与独立评审的"存在空白"几乎无关（33 个 confirmed → 0 个 NONE）。§8.2 人工抽查（18/91 已抽样）**仍未执行**，现在可以做了。

---

## 1. 研究问题与假设

**问题**：在相同的固定预算、相同的（已调试的）管线其余部分下，v7.2 的七类机会本体论 + 准入收紧，相对 v6 的两类挖掘，是否产出**更可信**的研究机会与 Idea？

**为什么现在做**：v7 replay pilot（full-jury 口径）NO-GO 后，按预注册规则先修准入（v7.2，已落地并有回归）。本实验是被指定的下一步。探针 A–E 已证明当前栈端到端可用（E 产出 executable_candidate），仪器校准完成。

**假设（可证伪）**：
- H1（v7.2 价值假设）：v7.2 臂的盲评机会真率（true-rate）≥ v6 臂。
- H0：两臂无差，或 v6 臂更高（本体论未兑现价值）。

**明确不在本实验回答的**：审计机制的独立贡献（V2-no-audit 消融，等本实验定标后另行设计）。

## 2. 设计总览

**配对两臂设计**：6 主题 × 2 臂 = 12 次全管线运行。同主题配对控制主题难度；两臂**仅挖掘模式不同**，其余全部相同。

| | 臂 A（v7.2） | 臂 B（v6 挖掘模式） |
|---|---|---|
| 挖掘类型 | 7 类 lens fan-out | 2 类（boundary_gap, missing_evaluation） |
| 准入契约 | v7.2（先例检索 + PERFORMANCE_TARGET_DELTA） | v6（问题级阻断 + 全局 limitation 要求） |
| policy version 戳 | `opportunity-ontology-v7.2` | `evidence-admission-v6-exp`（区别于历史 v6，行可分离） |
| 审计/预算/干预/Idea | **完全相同**（当前栈 + 当前 .env） | 同左 |
| 已修复 bug（novelty 短路、winner 排序等） | 含 | **含**（这是设计要点：两臂共享调试成果，归因只落在挖掘策略上） |

**为什么不用 git checkout 跑完整 v6 栈**：v6 时代栈自带当时的 bug（novelty 闸门短路、Top-1 审计裁剪），会让 v6 臂因与本体论无关的原因表现更差——对照不干净。v6 挖掘模式重建在当前栈上，把"挖掘策略"变成唯一自变量。

## 3. 工程前置项（Phase 0，执行前必须完成）

### 3.1 `GAP_MINING_LEGACY_V6_MODE` 开关

config 增加 `gap_mining_legacy_v6_mode: bool = False`。开启时 `mine_gaps.py`：

1. **契约集**：只用 2 个 lens（boundary_gap、missing_evaluation），prompt 用 v6 原文（已在 git `686f6d1` 快照中核对，含"两类菜单 + 全局 limitation/negative_result 硬要求 + 占位符禁令"）；
2. **问题级阻断恢复**：`NO_LIMITATION_SIGNAL`（问题无 limitation 证据→整题不可挖）与 `UNRESOLVED_VERIFIED_CONTRADICTION`（矛盾未解决→整题不可挖）回到准入；
3. **v7.2 闸门跳过**：先例检索与 PERFORMANCE_TARGET_DELTA 在此模式下不执行（它们是 v7.2 假设的一部分，不能泄入对照臂）；
4. **policy version**：写 `evidence-admission-v6-exp`，与历史 281 条 `evidence-admission-v6` 行区分；
5. 其余一切（语义去重、top-K 竞争、饥饿加权、审计、O2、干预、Idea）不动。

**验收**：
- 单测：模式开启时 lens fan-out 只有 2 个、问题级阻断按 v6 规则触发、policy version 戳正确、v7.2 闸门未执行；
- 对照测试：用探针 E 的证据池分别在两模式下跑 mining，确认产出形态与各自契约一致；
- 工作量估计：~200 行（含测试），半天。

### 3.2 运行基建

- 两个服务：8001（臂 A，当前 .env）、8002（臂 B，`GAP_MINING_LEGACY_V6_MODE=true`），同一 DB（WAL）；
- 复用 `_probe_monitor.py` 系列监控，参数化任务 ID。

## 4. 主题清单（与 replay pilot 完全同题，保证可比）

| ID | 分层 | 主题 | 预期主打 lens |
|---|---|---|---|
| pe2e-001 | narrow_mature | Context compression for retrieval-augmented generation with long documents | boundary / missing_evaluation |
| pe2e-007 | narrow_mature | Position bias (lost-in-the-middle) in long-context language models | boundary / contradiction |
| pe2e-013 | emerging_sparse | Test-time training for adaptive inference in language models | scaling_regime（v7.2 闸门主战场） |
| pe2e-015 | emerging_sparse | Mechanistic interpretability of mixture-of-experts routing decisions | mechanism_gap |
| pe2e-019 | broad_ambiguous | How to improve the scientific discovery capability of large language models | cross_domain / untested_assumption |
| pe2e-024 | broad_ambiguous | General-purpose uncertainty quantification for deep learning | missing_evaluation / scaling |

分层沿用 pilot（narrow_mature ×2 / emerging_sparse ×2 / broad_ambiguous ×2），每层两题。

**注意**：这些主题的 v6 时代 funnel 数据已在库（`replay_topics.jsonl`），可作背景参照，但**不作对照**——-era 端点/检索状态不同。对照只在本实验的两臂之间。

## 5. 运行矩阵与顺序

12 次运行，按主题配对、臂序随机（抛硬币定，已预先生成并记录于 §10 mapping 文件，防事后挑选）：

```
run 1: pe2e-001 臂序 B→A     run 2: pe2e-007 臂序 A→B
run 3: pe2e-013 臂序 A→B     run 4: pe2e-015 臂序 B→A
run 5: pe2e-019 臂序 A→B     run 6: pe2e-024 臂序 B→A
```

同主题两次运行**背靠背**（间隔 < 5 分钟），跨主题串行。臂序 B 先行的主题中，臂 A 受 PDF 缓存/去重惠及的程度被臂序随机化抵消。

**预算配置（两臂相同，即当前 .env）**：budgeted 审计、Top-2、remediation 全局上限 3、agent 超时 4h。运行中不改任何配置。

**成本估算**：12 × 60–120 min ≈ **12–24 小时**端点占用；建议分两批（各 6 次）跑，批间人工核对中间产物。

## 6. 指标

### 主指标（预注册）
**盲评机会真率**：对每臂中**进入过审计**的 gap（status ∈ surviving / confirmed_runner_up / inconclusive / rejected-by-audit），按 pilot 裁决口径判 `NONE（真机会）/ PARTIAL / FULL（假开放）`：
```
true-rate = NONE / (NONE + PARTIAL + FULL)
```
两臂各自计算，配对比较（按主题）。

### 次指标
- executable_candidate 数（按主题配对）
- 弃权率（abstained 任务数 / 6）
- 机会类型覆盖（v7.2 臂的实际多样性 vs v6 臂必然的 2 类）
- 成本：每任务 wall-clock、入库论文数、LLM 调用数
- **Idea 层**：对 executable / conditional_review Idea 判"是否值得做"（fund / maybe / no）

### 搭车数据（零额外成本）
- killer 检索命中率（两臂累计；v7.2 臂专属机制）
- 饥饿机制触发情况
- **两臂分数分布**：若 v6 模式臂的 Idea 分数也 ~0.65，则证明 rubric 整体变严而非 v7.2 Idea 变弱——直接喂给 P4 go 线决策

## 7. 预注册判定规则（冻结）

设 v7.2 臂 true-rate = T_A，v6 臂 = T_B：

1. **GO 全量实验**（24 主题、含 V2-no-audit 消融设计）：`T_A ≥ T_B + 10pp` **且** v7.2 executable 数 ≥ v6。
2. **NO-GO / 回滚评估**：`T_A < T_B`。行动：按 gap 类型分层归因——哪些类型拖累、哪些兑现，只对拖累类型收紧准入（v7.3），不动整个本体论。
3. **平局**（|T_A − T_B| < 10pp）：看次指标 tie-break：executable 数 → 类型覆盖 → 成本。若 v7.2 无任何次指标优势，按 NO-GO 处理（更贵且不更好）。
4. **样本量声明**：6 主题是 **pilot，只定向不检验**。任何结论都表述为"方向性证据"，显著性留给全量实验。
5. **中断规则**：若前 2 个主题（4 次运行）中出现**工程性故障**（非弃权类终态，如 failed / 超时 / phase 卡死），暂停并先修——故障运行不计入指标，修复后重跑该配对。

## 8. 盲评协议

### 8.1 匿名化导出

每臂运行结束后，从 DB 导出裁决单元到统一模板（`adjudication_input.jsonl`）：

- **gap 单元**：`{anon_id, topic_stratum, gap_type, claimed_delta, observed_problem, missing_capability, evidence_claims[]（论文标题+span 摘要）, audit_result, novelty_confidence}`
- **idea 单元**：`{anon_id, topic_stratum, title, motivation, experiment_sketch, final_score, decision}`

**剔除一切臂标识**：无 task_id、无 policy version、无时间戳、无运行顺序。gap_type 字段保留（v6 臂只可能有 2 类——这是协议已知的泄露点，**必须接受并披露**：评审者可能从类型推断臂。缓解：评审按单元独立打分，不做臂间比较；类型泄露影响的是"知道它来自哪臂"而非"该机会是否为真"）。

乱序后写入，`review_mapping.json`（anon_id → 真实 id + 臂）单独存放，评审期间不打开。

### 8.2 评审

1. **主评（LLM 联网）**：沿用 pilot pass2 协议——对每个单元联网检索 prior art，判 NONE / PARTIAL / FULL（gap）或 fund / maybe / no（idea），notes 逐行标注检索证据。一次过，不迭代。
2. **人工校准抽查（必须）**：pilot 的已知局限是"同一 LLM 评审，非独立人工"。本实验要求**用户**对 ≥20% 单元（约 8–10 个 gap + 4–5 个 idea，分层抽样）独立盲判，与 LLM 主评对照：
   - 一致率 ≥ 70%：LLM 主评可用，全文引用时披露；
   - 一致率 < 70%：LLM 主评作废，全量人工判（或扩抽样再定）。

### 8.3 已知局限（引用结果时必须披露）

- 评审者可能从 gap_type 推断臂（见 8.1）；
- LLM 主评与被评管线可能共享系统性偏见（同源模型）——人工抽查是对此的唯一制衡；
- 6 主题 pilot 无显著性。

## 9. 执行清单（按序）

| # | 事项 | 产出 | 预计 |
|---|---|---|---|
| 0 | 实现 `GAP_MINING_LEGACY_V6_MODE`（§3.1）+ 回归测试 | 代码 + 测试 | 半天 |
| 1 | 全量测试回归 | 710+ passed | 20 min |
| 2 | 起 8002（v6 模式）服务，冒烟一个短任务确认模式生效 | trace 验证 | 30 min |
| 3 | 批 1：run 1–6（pe2e-001/007/013 背靠背） | 6 次运行 | ~6–12 h |
| 4 | 批 1 人工核对（终态分布、无工程故障） | 核对记录 | 30 min |
| 5 | 批 2：run 7–12（pe2e-015/019/024） | 6 次运行 | ~6–12 h |
| 6 | 匿名化导出 + 乱序 + mapping 封存 | adjudication_input.jsonl | 1 h |
| 7 | LLM 联网主评 | verdicts.jsonl | 2–3 h |
| 8 | 用户人工抽查 ≥20% | 一致率报告 | 1–2 h（用户） |
| 9 | 解封 mapping，按 §7 规则出结论 | summary.md | 1 h |

## 10. 文件与产物

```
eval_results/fixed_budget_pilot/
├── EXPERIMENT_DESIGN.md        # 本文件（预注册，冻结）
├── run_matrix.json             # 12 次运行的 topic × arm × order × task_id
├── adjudication_input.jsonl    # 匿名化裁决单元（评审者只见此文件）
├── review_mapping.json         # anon_id ↔ 真实 id/臂（评审后解封）
├── verdicts.jsonl              # LLM 主评结果
├── human_spotcheck.jsonl       # 用户人工抽查
└── summary.md                  # 最终结论（按 §7 规则）
```

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| v6 模式重建引入行为偏差（不是真 v6） | 契约文本从 git 快照逐字复制；单测锁定 lens 数/阻断规则/版本戳 |
| 同主题第二次运行吃缓存便宜 | 臂序随机化；成本只作次指标并披露 |
| LLM 端点在两臂间漂移 | 背靠背运行（间隔 < 5 min）；跨主题串行分散时段 |
| 评审从类型猜臂 | 单元独立打分 + 披露；人工抽查制衡 |
| 中途发现新管线 bug | §7.5 中断规则：故障配对修复后重跑，不污染样本 |
| 全部弃权（两臂都无 executable） | true-rate 主指标不依赖 executable；Idea 层指标降级为 conditional_review 判 fund/maybe/no |
