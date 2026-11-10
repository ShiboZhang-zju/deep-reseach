# 近邻预算对照 — 预注册方案（2026-11-03）

> 状态：**设计完成，未运行**。判定规则在看到结果之前写定，不得事后修改。
> 上游：`../no_audit_ablation/EXPERIMENT_DESIGN.md` §7.0 明确要求本实验**先于或并行于**
> V2-no-audit 消融执行——否则消融的阴性结果不可解释。
> 诊断依据：`../killer_funnel_2026-11-02.json`。

---

## 1. 要回答的问题

**审计判新颖性时只比对 3 篇近邻。这个预算本身是否就是"confirmed 不蕴含新颖"的成因？**

这是比"审计有没有用"更基础的问题。若预算就是天花板，那么在 3 篇下测出的任何"审计无贡献"都是**预算造成的假结论**，而不是关于机制的结论。

## 2. 已知的剂量响应（本实验的全部依据）

用生产配置（`text-embedding-v4`，阈值 0.45）对 37 个受评 gap 做语料库召回，取独立评审**点名**的 killer 论文（库内可定位者，n=71），看它落在召回排序的第几位：

| k | 命中 killer | 累计 | 相对 k−1 的边际 |
|---:|---:|---:|---:|
| 1 | 10 | 14.1% | — |
| 2 | 20 | 28.2% | +10 |
| **3（生产默认）** | **26** | **36.6%** | +6 |
| 5 | 38 | 53.5% | **+12（最大跃升）** |
| 6 | 40 | 56.3% | +2 |
| **8（本实验处理组）** | **43** | **60.6%** | +3 |
| 10 | 45 | 63.4% | +2 |
| 15 | 49 | 69.0% | +4 |
| 20 | 54 | 76.1% | +5 |

**为什么取 8，不取 5 或 10**：

- 生产预算 3 只能看到 36.6% 的杀手文献——**约三分之二不可见**，这是"confirmed→0 NONE"最直接的候选解释。
- k=8 达到 60.6%，相对 k=3 提升 **+24.0pp**，是能观测到效果的最小剂量。
- 曲线在 k=8 之后明显趋平（8→10 仅 +2.8pp），取 10 会让成本上升而剂量几乎不变。
- k=5 的边际收益最大（+12 篇），但它只覆盖 53.5%，若 k=5 与 k=3 无差异，会无法区分"预算不是天花板"与"剂量不足"。k=8 保留了更大的效应检出空间。

## 3. 两臂

| | 臂 C（control） | 臂 D（treated） |
|---|---|---|
| `budgeted_audit_neighbors` | **3** | **8** |
| `audit_neighbors_override` | 0（不覆盖） | **8** |
| 其余全部预算 | 相同 | 相同（查询 4 / 候选 10 / 全文 2 / killer 查询 3 / 源相同） |
| 供给侧修复 | 生效 | 生效 |
| 相关性下限 | 0.3 | **0.3（必须相同）** |

**单变量保证**：只有 `neighbors` 变化。这一点由 `tests/test_neighbor_budget_override.py::test_override_raises_only_the_neighbour_count` 直接断言（除 `neighbors` 外 `_audit_budget()` 逐键相等）。

**相关性下限必须两臂相同**：`gap_relevance_min_neighbor=0.3` 会把低于下限的近邻判定降级为 UNCERTAIN。若只在一臂启用更强的下限，比较就同时动了两变量（数量 + 相关性策略）。

## 4. 成本非对称的诚实说明

近邻数**不是**线性成本，且这一点对比较很重要：

- 审计判定是**一次** `llm.chat_json` 调用，所有近邻拼进同一个 prompt（`_audit_user.neighbors=_format_neighbors(db, neighbors)`）。k=8 只是让这一个 prompt 变长，**不增加调用次数、不增加超时暴露**。
- 全文抽取由 `budgeted_audit_fulltext_max_papers=2` 独立封顶，**不随近邻数增长**。
- 因此两臂的主要成本差异是 prompt token（可测），而非调用数或 PDF 下载量。

**仍需记录的**：单 gap 审计墙钟（`GAP_AUDIT_TIMEOUT_SECONDS=600` 是硬罩）与 `phase_runs.duration_ms`。若臂 D 出现超时，必须单独报告并剔除，不能计为"更多近邻无用"。

## 5. 主题

沿用固定预算实验的 6 主题（`pe2e-001/007/013/015/019/024`），6 × 2 = 12 次运行，背靠背配对，臂序随机。

## 6. 指标

### 6.1 主指标

```
T_arm = NONE / (NONE + PARTIAL + FULL)      # 独立评审，pass-3 语义召回仪器
ΔT = T_D − T_C
```

### 6.2 机制读数（本实验真正的产出）

| 指标 | 为什么需要 |
|---|---|
| **killer 覆盖增量** | 用 `killer_funnel_audit.py` 复测：臂 D 的近邻里出现的评审点名 killer 数是否显著上升。**这是处理是否真的生效的直接证据**，不依赖任何评审判断 |
| `gap_audit_neighbor_budget` trace：`effective_limit` / `cap_bound` / `selected_below_floor` | `cap_bound=false` 说明该 gap 的池子本来就小于 8，该样本对处理**无剂量**，必须单列 |
| `novelty_confidence` 分布（两臂） | 若臂 D 系统性更低，说明"看到更多→更不自信" |
| `gap_audit_low_novelty_downgrade` 计数 | **混杂因素**：`_LOW_NOVELTY_CONFIRMED=0.4` 会把低自信的 confirmed 降级为 uncertain，使两臂进入下游的 gap 集合不同 |
| 单 gap 墙钟 / 超时数 | 见 §4 |

### 6.3 已知混杂因素（必须在结论里量化，不能只声明）

**近邻越多 → 越可能发现覆盖 → `novelty_confidence` 越低 → 越容易触发 `confirmed→uncertain` 降级。** 这会改变两臂的下游 gap 集合，使 `T` 的分母不同。

处置：结论必须同时报告"降级前"与"降级后"两套 `T`。若两种口径给出相反的 `ΔT`，则该实验**不成立**，需先修正降级策略再重测。

## 7. 预注册判定规则（冻结）

1. **预算不是天花板（NO-GO for the budget hypothesis）**：`ΔT < +10pp` **且** killer 覆盖增量不显著（臂 D 近邻内 killer 命中数未显著高于臂 C）。
   → 行动：预算不是约束。**此时 V2-no-audit 消融在 k=3 下的结论才可用于"删机制"决策**（解锁上游 §7.0）。
2. **预算是天花板（GO）**：`ΔT ≥ +10pp` 或 killer 覆盖显著上升。
   → 行动：审计的独立性**必须在 k=8 下重测**；生产默认应改为 8；消融实验改在 k=8 下跑。
3. **中间态**：killer 覆盖显著上升但 `ΔT < +10pp`。
   → 行动：处理生效但没转化为可信度。这是**最有价值的结果**——说明瓶颈已从"没看到"移到"看到了但判不出来"（即判断层），下一步应查原子 claim 判定与 PARTIAL 语义，而不是继续加预算。
4. **无剂量样本剔除**：`cap_bound=false` 的 gap 单列为"未受处理"，不计入主指标；同时报告其占比。若占比 > 40%，该实验对主指标**无检验力**，只能报机制读数。
5. **超时处理**：臂 D 超时的 gap 单独报告，不计入 `T`，但必须在结论中出现其数量——把超时当失败会系统性低估处理组。
6. **样本量声明**：6 主题是定位实验。所有结论表述为方向性证据。

## 8. 运行

### 8.0 运行中修订（2026-11-03，均已在看到主指标前记录）

三处修订，全部源于工程性故障或剂量未交付，**不涉及任何判定规则**（§7 仍未改动）。

**(a) 并发改串行。** 原计划两臂背靠背并发。实测：两任务同秒启动后，臂 C 的 `extract_evidence_round_1` 卡在 `running` 8 分钟以上（臂 D 同时长下 5 分钟即完成），导致覆盖快照从未生成，readiness 重试 12 次后以 `no_high_importance_question_has_coverage_snapshot` 失败。两个任务各自 `EVIDENCE_EXTRACT_CONCURRENCY=4`，共享 SQLite 与 PDF/embedding 并发池，属资源争用。**改为同主题内顺序执行**（`arm_order` 仍随机），代价是墙钟约翻倍。这不引入新的混杂：近邻数仍是唯一配置差异，而并发争用不是假设的一部分。

**(b) 加入 `RECOVER_TASKS_ON_STARTUP=false`（必须）。** `recover_interrupted_tasks()` 在**每个**服务启动时把处于中断状态（含 `searching`/`extracting_evidence`）的**所有**任务重置为 `pending`，不区分归属。共享 DB 的双服务实验中，重启任一臂会静默中断另一臂正在跑的任务，并把其 phase 标为 failed；任务随后可能被**另一个臂的配置**接管续跑。这是能直接作废整个实验的静默污染，故实验服务一律关闭自动恢复，改用协议 §7.5 的人工中断处理。

**(c) 修复"召回进池但选不上近邻"。** 第一次真实审计 trace 暴露：`limit=8 selected=2 cap_bound=False pool=10`。根因：`select_gap_specific_neighbors` 的排序只覆盖有对抗检索映射的论文，其 fallback 只走 `collect_same_question_neighbors`（gap **自身问题**的证据链接）。语料库召回命中的论文因此能进入 `candidate_paper_ids`（计入准入、使 `candidate_pool_size` 看起来正常）却**永远无法被选为近邻**。

  这正是 §3.2 所述 67.1% 损失桶的平移：召回把 killer 从"结构上不可达"变成"进了池但不参与比对"。修复后同一 gap 在真实数据上从 2 篇提升到 8 篇、5 篇提升到 8 篇；回归测试 `tests/test_neighbor_selection_from_recall.py`（4 项，修复前 4 项全部失败、修复后全部通过）。

  **披露**：修复后首批有效运行的 trace 为臂 C `3/3 cap_bound=True`、臂 D `8/8 cap_bound=True`——剂量在**两臂**都真实交付，这是本实验有效性的前提。

以上修订期间作废的运行全部记录在 `run_matrix.json` 的 `aborted_runs`，`counted_in_metrics=false`。

```bash
cd backend

# 臂 C：control（默认 3）
GAP_AUDIT_BUDGETED=true AUDIT_NEIGHBORS_OVERRIDE=0 \
  python -m uvicorn app.main:app --port 8001

# 臂 D：treated（8）
GAP_AUDIT_BUDGETED=true AUDIT_NEIGHBORS_OVERRIDE=8 \
  python -m uvicorn app.main:app --port 8002

# 复测 killer 覆盖（处理是否生效的直接读数）
python scripts/killer_funnel_audit.py --dir eval_results/neighbor_budget/<arm> \
  --out eval_results/neighbor_budget/<arm>/funnel.json
```

产物目录 `eval_results/neighbor_budget/`：

```
EXPERIMENT_DESIGN.md      # 本文件（预注册，冻结）
armC_control/  armD_treated/   # 各自的 run 输出 + funnel.json
adjudication_input.jsonl  review_mapping.json  verdicts.jsonl
human_spotcheck.jsonl     RESULTS.md
```

## 9. 不做什么

- **不改相关性下限**（两臂都 0.3）：那会让单变量失效。
- **不改全文抽取上限**（两臂都 2）：它不是本次假设的一部分；若怀疑它是瓶颈，应另立实验。
- **不同时跑 no-audit 消融**：两者共享主题与评审仪器，混跑会污染盲评与预算记账。先本实验，再消融。
- **不用本实验的评审结论去推翻固定预算实验的 NO-GO**：口径不同，不得合并。
