# Deep Research

**Evidence-grounded Research Gap Discovery and Idea Validation System（V2）**

输入一个研究方向，系统自动完成：研究契约构建 → 研究问题分解 → 多轮多源检索 → 证据抽取 → 覆盖度评估 → 研究缺口挖掘与对抗审计 → 干预方案生成 → 最小实验设计，最终产出一份可回溯到具体论文证据的研究报告与（可能的）可执行 Idea。

系统要解决的问题是**长链路上的 LLM 幻觉**：不再"检索完让 LLM 凭空编 Idea"，而是要求每个研究缺口、每个 Idea 都能沿 `Idea → Intervention → Gap → Evidence → 论文原文 span` 这条链路逐级回溯，并由独立于生成方的审计闸门验证。系统允许并接受**零可信 Idea** 的诚实结论——证据不足时它弃权（`abstained`），而不是编一个看起来合理的想法。任何终止路径都至少交付一份领域态势简报（Landscape Brief）。

---

## 0. 预期目标与当前进展

### 0.1 核心主张（可证伪）

> **在同样的文献供给下，`Evidence → Gap Audit → Intervention → Experiment` 全链路，相对"直接生成"与"检索后生成"两个简单方案，更能产出可信科研 Idea，也更能拒绝假 Idea。**

这句话被拆成两个必须同时成立的 headline 指标（双指标是为了堵住"全部弃权"这条作弊路径——只压 false-open 率的话，一个永远弃权的系统会拿满分）：

| 指标 | 定义 | 期望方向 |
|------|------|----------|
| **False-open-gap rate** | 被独立验证已有 prior art 覆盖的 claimed gap / 全部声称开放的 gap | ↓ |
| **Credible Idea Yield** | 至少产出 1 个通过独立验证的可信 Idea 的 topic / 全部 topic | ↑ |

配套的**预注册裁决规则**（写在跑之前，防止看到数据再定方向）：增益主要来自 gap 正确性 → 加强 NPA/Audit；novelty 仍差 → 查 retrieval；novelty 好但 feasibility 差 → 优化 Intervention/Experiment；**与 Retrieval+LLM 基线无显著差异 → 删机制，而不是继续往系统上加东西**。

### 0.2 分阶段目标与状态

| 阶段 | 目标 | 状态 |
|------|------|------|
| P0 工程化基础 | 可中断、可续跑、可观测 | ✅ 完成 |
| Pipeline V2 重构 | 从"检索后编 Idea"转为证据驱动，六层结构化落库 | ✅ 完成 |
| 链路优化（O 系列） | 闸门失败即补检索而非终止；分级产出；全文 RAG | ✅ 完成 |
| 证据供给与运行时稳定性 | 让准入所需的证据类型真的能被抽到（chunk 按 section 轮转、关系语义统一、上下文预算） | ✅ 完成 |
| 机会发现广度（v7 ontology） | 不再只认"作者自认的 limitation"，7 类 opportunity 各有准入契约 | ⚠️ pilot 复核后 NO-GO（[3] 不成立；LLM 评审，非独立人工） |
| 系统价值验证（production_e2e_v1） | 三系统对照，回答"复杂 pipeline 到底有没有价值" | ✅ 24 主题跑完并完成盲评 |
| v7 生产价值对照 | 固定预算下 v6 policy vs v7 policy 的产出价值 | ⬜ 未开始（明确推迟，不与 full-jury 漏斗混比） |

### 0.3 当前进展（截至 2026-10-28）

**已拿到的第一组真实 E2E 数据**（`backend/eval_results/pe2e_v1_final/summary.md`，24 主题，人工盲评 24/24、23/23、22/22 全部判定完毕）：

| System | False-open-gap ↓ | Credible Idea Yield ↑ | Novelty ↑ | Feasibility ↑ | Abstention |
|---|---:|---:|---:|---:|---:|
| Direct LLM | 25.0% | 45.8% | 1.75 | 4.38 | 0.0% |
| Retrieval + LLM | 60.9% | 34.8% | 1.44 | 4.70 | 0.0% |
| **Full V2** | **18.2%** | **81.8%** | **2.27** | 4.59 | 4.3% |

两个 headline 都朝预期方向走，且 Full V2 是唯一会弃权的系统（4.3%）。**但这只是系统级对比，不是 Gap Audit 的因果消融**——V2 同时拥有全文 chunk、Evidence Unit、多轮检索、best-of-N 候选，赢了不能证明"审计单独导致了这个结果"。机制归因需要 V2-no-audit 之类的消融，尚未做。

**v7 机会本体论已实现并跑完 pilot**（`backend/eval_results/v7_replay_pilot/replay_report.json`，6 主题、冻结证据池、无重检索）：

- 召回面：v7 产出 35 个候选，其中 **30 个在 v6 中无对应物**（v7-only）。
- 跨类型聚簇后 30 → 29 簇（压缩比 1.03），说明新增量不是同一发现的措辞变体。
- 成本：mining 42 次 LLM 调用 / 1068s，审计 1819s。
- **q2/q3 已裁决（2026-10-29，第二轮联网复核后）**：q2 边际精度 **6.7%**（2 true / 21 partial / 7 false-open）；[1][2] MET，**[3] NOT MET** → 按预注册规则 **NO-GO**。第一轮的名义 GO 依赖的 v7r-005 经检索发现已被 arXiv 2607.28308（2026-07-30，早于预注册日）部分覆盖，改判 PARTIAL；剩下的 2 个 true 都是无机制的性能目标，且没有关联 Idea。判定由同一 LLM 评审完成（非独立人工、非盲）。`scaling_regime` 假阳性最高（4/7）。详见 `eval_results/v7_replay_pilot/adjudication_summary.md`。

- **探针 C**（`1caa24c2`，同主题，第三/四轮修复后）：**116 分钟，终态 `abstained`**。P5/P2-①/增量覆盖度/Top-2/P4 逐项生效（增量覆盖度 r7 在 1058 条证据池上仅 106s；P2-① 计时立刻定位"单篇卡死 390s 决定整体上限"）；但暴露一个严重缺陷：**winner 选择只按 cheap rank 排序，让 novelty=0.9 的 gap 输给 novelty=0.42 的 gap（0.7335 vs 0.7567，差 0.023）**，导致全部 Idea 封顶 tier B、最终 abstain。已修（排序键改为 `(clears_novelty_gate, cheap_rank, novelty_confidence)`）+ 2 项回归。P1 killer 检索执行 2 次、命中 0，机制正确但收益待更多样本。完整对照：同一报告 §10。

- **探针 D/E**（`0e695ea5` / `457dac9a`，同主题，2026-10-31）：D 确认 winner 修复无回归，但暴露**更基础的 bug**——审计 `remaining_delta` 为空时，`_evaluate_hard_gates` 整个 novelty 分支被跳过，同一审计里的 `novelty_confidence=0.9` 被完全忽略，全部干预封顶 tier B。已修 + 2 项回归。E（加载修复后）首次跑出 **`executable_candidate`（tier A）**，终态 `waiting_for_user_review`，验证通过。E 同时暴露一个**政策问题（未决）**：该 executable Idea 的 LLM 评分仅 0.4681（go 线 0.70），trace 如实打了 `GATE_PASS_SCORE_BELOW_LINE`——硬门禁与 rubric 两轴对同一 Idea 给出相反结论。详见报告 §11。

**Idea 分数阈值全库校准分析（2026-10-31，595 个带分数 Idea / 94 个任务）**：`0.70` go 线原本硬编码在三处，且**只对历史分布有效、对 v7.2 不适用**——历史（v3–v7，n=579）median **0.711**、61.3% 过线；v7.2（n=16）median **0.654**、12.5% 过线，4 个 `executable_candidate` 全部 0 个过线。两轴不一致是常态（门禁 PASS+分数低 59 个，门禁 BLOCK+分数高 224 个），且分数落在离散网格上（0.7120×26、0.7405×23…），0.70 无特殊意义。已把阈值统一为 `IDEA_SCORE_GO_LINE` / `IDEA_SCORE_REVISE_LINE`（数值故意维持 0.70，下调属政策决策）。只读分析工具：`backend/scripts/idea_score_distribution.py`，存档 `eval_results/idea_score_distribution_2026-10-31.json`。

**v7.2 固定预算生产价值实验（瓶颈 #7）——运行 + 盲评完成（2026-11-02）**：Phase 0 用 `GAP_MINING_LEGACY_V6_MODE` 在当前已调试栈重建 v6 两类挖掘（717 tests passed）。12 次运行（6 主题 × 2 臂）全部完成且完整性核查通过；91 个单元完成联网主评（pass 1），6 个检索失败单元做了修复重评（pass 2）。

**裁决结果：T_A = T_B = 0.0%，delta = 0.0pp → TIE，按 §7.3 默认判 NO-GO**（无 v7.2 优势证据）。但这 TIE 是"**无区分力**"而非"已证明等价"：两臂各自 12/25 个 gap 中**没有一个**被独立评审判为可发表空白，指标没有余量去区分。

判定过程有三轮，全部披露：pass 1 关键词检索给出 −4.0pp NO-GO，但整个差异由**一个检索失败的单元**（`fbp-089`，12 篇结果零篇相关）承载，剔除后为 0；pass 2 用近邻标题重建查询也失败；**pass 3 改用语义召回**（关键词过检索 + embedding 过滤，阈值 0.41 由 37 个真实 gap 实测标定：保留 82% 真近邻、滤除 100% 已知噪声），评审自述"检索无用"从 **6/37 降到 0/37**，`fbp-089` 的 NONE 消失，且修复后检索到的第一篇正是它的直接杀手文献。**§7 规则三轮逐字未改。**

**本轮真正建立的结论**：管线自评的 `confirmed` 与独立评审的"存在可发表空白"几乎无关——**33 个 `confirmed` → 0 个 NONE**（28 个 PARTIAL + 5 个 uncertain→PARTIAL）。这条不依赖两臂对比，是最稳的结论。

**不得引用的结论**：❌ "v7.2 优于 v6"（T 相同）❌ "v6 优于 v7.2"（FULL 3 vs 1、Idea fund 3 vs 0 是唯一信号，但 n=25 vs 12 且**审计量不对称**：A 每主题 2 个 gap、B 4–6 个）❌ "两臂等价"（T=0 是无力检测）❌ "本体论无效"（无法分离本体论与审计量效应）。

**下一轮必须先修的设计缺陷**：审计量不对称（A 2 个 vs B 4–6 个）、T 指标无余量（两臂都 0 NONE，应改序数量表或比 FULL 率）、Idea 层不同质（方向性 vs 可执行需分层）。

存档：**`backend/eval_results/fixed_budget_pilot/RESULTS_FINAL.md`（最终结论）**、`RESULTS_ADJUDICATION.md`（pass-1 审计轨迹，保留但已被取代）、`RESULTS_SECONDARY.md`、`EXPERIMENT_DESIGN.md`、`run_matrix.json`。§8.2 人工抽查（18/91 已抽样）**仍未执行**，检索已修、现在可做。**另一待修项**：饥饿机制在臂 A 未达目标（`candidate` 占比 80.3% vs <50%）；killer 检索锚点仍是 topic 而非 killer 描述（见 `RESULTS_FINAL.md` §7）。

**工程状态**：`717` 个测试全部通过（2026-10-31）；Alembic 迁移到 `0034`；证据抽取已改为无条件覆盖全部候选论文（不再有每轮篇数上限），靠共享并发池 + PDF 预热把成本压回来。

**killer 漏斗诊断 —— 33 confirmed → 0 NONE 的归因（2026-11-02）**：`scripts/killer_funnel_audit.py`（只读，存档 `eval_results/killer_funnel_2026-11-02.json`）把"管线 confirmed 与真实可发表性无关"归因到了**供给**而非判断。对 37 个受评 gap，取独立评审点名的 killer 论文并定位到任务语料库（标题 Jaccard ≥0.90）：

| 漏斗层 | n | 占 82 篇库内 killer |
|---|---:|---:|
| 被选为审计近邻 | 4 | 4.9% |
| 过相关性筛选但未入选 | 6 | 7.3% |
| 进了审计池、被 10 篇上限截断（**未做相关性筛选**） | 17 | 20.7% |
| **在语料库中，从未进入任何审计池** | 55 | **67.1%** |

配套：**每次审计近邻数恒为 3**（37/37 等于 budgeted 上限）；审计自评 `novelty_confidence` 对"被覆盖/未覆盖"**几乎无区分力**（AUC(PARTIAL>FULL)=**0.610**）。根因是候选池 = 该 gap 对抗检索 ∪ `collect_same_question_neighbors`（只走该 gap 自身问题的证据链接）——从未关联到这些问题的库内 killer **结构上不可达**；再叠加"先按 `TaskPaper.final_score`（任务级相关性）截断到 10 篇、后做 gap 相关性筛选"的顺序错误。

**已落地的两处供给修复**（`AUDIT_CORPUS_RECALL_ENABLED` / `AUDIT_SCREEN_BEFORE_CAP`，默认开）：① 语料库语义召回并入审计池（`steps/corpus_recall.py`，阈值 0.45，top_k 15，embedding 不可用 fail-open）；② 先按 gap 相关性筛选再截断，截断依据从任务级分数改为 gap 相关性。**实测上限（真实库 + 真实 embedding）**：库内 killer 召回 68.4%，但在全排序里进入 Top-3 的只有 **36.6%**、Top-15 之外仍占 31.0%——召回解决了"不可达"，**没有**解决"审计仍只看 3 篇"。

**V2-no-audit 消融已实现、未运行**（`GAP_AUDIT_DISABLED_ABLATION`，`steps/audit_gaps.py::_ablate_audit`）：不检索、不抽近邻、不判新颖性，gap 直接 `surviving` 并打独立策略戳 `gap-search-ablation-none-v1` + `ABLATION_NO_AUDIT`。关键设计（均有回归）：策略戳必须与真实审计区分（同时它参与 `input_version` 哈希与 `_latest_decided_audit` 过滤）；开关在**调用时**解析而非 import 时常量（首版写错，被测试抓到）；干预 novelty 闸门在该臂**显式 PASS** 而非留 UNKNOWN（否则消融臂 executable 恒为 0，差异变成记账差异），且该 PASS **只作用于消融臂自己的记录**——真实 `closed` 结论不被开关抹除。预注册方案见 `eval_results/no_audit_ablation/EXPERIMENT_DESIGN.md`。

**该方案的一条硬约束（§7.0）**：在 `neighbors=3` 下得到的阴性 `ΔT` **不得**用于"删审计"决策——必须先有 `neighbors=3 vs 8` 的固定预算对照，否则"审计无用"会是预算造成的假结论。

**近邻预算对照 —— 已预注册、开关已实现、未运行（2026-11-03）**：`AUDIT_NEIGHBORS_OVERRIDE`（`config.py`）+ `effective_neighbor_budget()`（`audit_gaps.py`）+ `gap_audit_neighbor_budget` trace，方案见 `eval_results/neighbor_budget/EXPERIMENT_DESIGN.md`。

剂量响应（37 个受评 gap、生产 embedding 实测，库内 killer n=71 在召回排序中的位置）：

| k | 1 | 2 | **3（默认）** | 5 | 6 | **8（处理组）** | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 累计命中 | 14.1% | 28.2% | **36.6%** | 53.5% | 56.3% | **60.6%** | 63.4% | 69.0% | 76.1% |

**生产预算 3 只能看到约 1/3 的杀手文献**，这是"confirmed → 0 NONE"最直接的候选解释。取 8 而非 10：k=3→8 提升 +24.0pp，而 8→10 仅 +2.8pp（曲线在 8 之后趋平）。

单变量保证：`test_override_raises_only_the_neighbour_count` 断言除 `neighbors` 外 `_audit_budget()` 逐键相等（查询 4 / 候选 10 / 全文 2 / killer 查询 3 / 源 全不变）。相关性下限两臂**必须同为 0.3**——既有的 `_mechanical_novelty_score` 已把低于下限的近邻判定降级为 UNCERTAIN，所以"更多近邻只是更多噪声"这一质疑已有既有防线覆盖，不需要（也不应该）在实验里再加一层。

**成本非对称的实测结论**：审计判定是**一次** `llm.chat_json` 调用、所有近邻拼进同一 prompt，所以 k=8 只让 prompt 变长，**不增加调用次数与超时暴露**；全文抽取由 `budgeted_audit_fulltext_max_papers=2` 独立封顶，不随近邻数增长。

**已识别的混杂因素（写入协议 §6.3，必须量化而非声明）**：近邻越多 → 越可能发现覆盖 → `novelty_confidence` 越低 → 越易触发 `_LOW_NOVELTY_CONFIRMED=0.4` 的 confirmed→uncertain 降级（trace `gap_audit_low_novelty_downgrade`），使两臂进入下游的 gap 集合不同。协议要求同时报告降级前后的两套 `T`；若两套口径给出相反的 `ΔT`，该实验不成立。

**近邻预算实验已开跑，并在第一对就抓到两个会作废实验的缺陷（2026-11-03）**

第一次真实审计 trace 直接暴露处理没生效：`limit=8 selected=2 cap_bound=False pool=10`——配置 8、池子 10、预算没用满，却只选 2 篇。若未发现，整个实验会得出"更多近邻无用"的假结论。

1. **召回进池但选不上近邻（代码缺陷，已修）**：`select_gap_specific_neighbors` 的排序只覆盖有对抗检索映射的论文，fallback 只走 `collect_same_question_neighbors`（gap 自身问题的证据链接）。语料库召回命中的论文能进 `candidate_paper_ids`（计入准入、`candidate_pool_size` 看起来正常）却**永远无法被选为近邻**——这是 67.1% 损失桶的平移：trace 变了，审计读到的内容没变。同一批真实 gap 上修复后 2→8、5→8；回归 `test_neighbor_selection_from_recall.py` 修复前 4 项全失败、修复后全通过。
2. **跨臂任务接管（可静默作废实验，已加开关）**：`recover_interrupted_tasks()` 在每个服务启动时把所有中断状态（含 `searching`）任务重置为 pending，**不区分归属**。双服务共享 DB 时，重启任一臂会中断另一臂在跑的任务并标 failed，之后该任务可能被**另一个臂的配置**接管续跑。实验服务一律 `RECOVER_TASKS_ON_STARTUP=false`（启动日志显式确认）。

另：**并发改串行**。两任务同秒启动时臂 C 的 `extract_evidence_round_1` 卡 running 8 分钟以上（臂 D 5 分钟完成），覆盖快照未生成、readiness 重试 12 次失败。两任务各 `EVIDENCE_EXTRACT_CONCURRENCY=4` 共享 PDF/embedding 池，属资源争用；改串行后同主题臂 C 一次通过。

**修复后的剂量交付已验证**（pe2e-001）：臂 C `3/3 cap_bound=True`、臂 D `8/8 cap_bound=True`——单变量剂量在两臂都真实交付。

**首批机制读数（n=1 对，仅方向性）**：臂 C novelty 0.94（3 篇）、臂 D 0.74/0.78（8 篇），方向与预判混杂因素一致（看到更多→更不自信），但均远高于 0.4 降级线，本对未触发降级。臂 D 出现 `below_floor=2`，证实扩大近邻确实拉进低相关论文，而既有 `gap_relevance_min_neighbor=0.3` 已在拦截——这也说明原计划的"噪声防护新开关"是重复实现，已删除，改为只把该信号记入 trace。

中期结果见 `eval_results/neighbor_budget/RESULTS_PARTIAL.md`（明确列出**不能**下的结论）。

**近邻预算对照最终结果（2026-11-06 完成，5 对有效 + 1 对作废）**

| 臂 | gap | NONE | PARTIAL | FULL | T |
|---|---:|---:|---:|---:|---:|
| C（3 篇） | 16 | 0 | 14 | 2 | **0.0%** |
| D（8 篇） | 12 | 0 | 12 | 0 | **0.0%** |

`ΔT = +0.0pp` → 按预注册规则 1 判 **预算不是天花板**。**但这条判定几乎不含信息**：两臂 NONE 都是 0，指标撞在 0% 地板上没有余量，`ΔT=0` 是"两边都测不出"而非"两边相等"——与 v7.2 的 TIE 是同一类缺陷。

**Idea 层更强的阴性信号**：C 48 个 / D 30 个 Idea，被独立评审判为 `fund` 的**两臂均为 0**。

**人工抽查 100% 一致率是假象**：gap 判定 92.9% 是 PARTIAL、idea 89.7% 是 maybe，人工样本也全落众数标签；gap/idea 分层 kappa 因零方差无法计算（期望一致率已达 100%）。抽查**没有发现分歧，但也没有证明评审器可靠**——它从未被要求区分两种判定。

**评审仪器健康度可测**：`search_useful=false` 仅 1.9%（pilot pass 1 为 16.2%），零命中单元 0。此前我用参数名错误的坏仪器跑过 2 个单元（检索全失败 `hits=0`），它把 PARTIAL 误判成 **NONE**——坏仪器会**高估**真空白；这 2 条已归档不计入。

**混杂因素预注册写错、已更正**：`audit_low_novelty_provisional_survive=True` 时低 novelty **不降级**，而是暂存并封顶 tier B；两臂 `low_novelty_downgrade` trace 均为 0。实测 C 臂 median novelty 0.700 / D 臂 0.662——**扩大近邻并未系统性压低自评**。

**不会说的三件事**：8 篇优于 3 篇（无余量）；审计机制应删除（只测了一个参数，判删除需在有余量指标上重做消融）；抽查支持评审器可靠（一致率被众数抬高）。

**本次真正产出**：①修复了一个在修复前影响**所有**任务的缺陷（召回进池但选不上近邻）；②确认 `NONE` 率作为主指标在 gap 供给不足时恒为 0，**必须改用有余量的量表**——这是下一轮要改的，而不是继续加机制。

详见 `eval_results/neighbor_budget/RESULTS.md`。

**指标改造（`METRIC_REDESIGN.md`，在现有数据上实测，未重跑管线）**

候选 1「claim 平均覆盖率」**否决**：C 0.119 > D 0.084 看似处理组更差，实为算术必然——覆盖表是近邻×claim 完整笛卡尔积（D 臂 `judgements/claim=8.00`、C 臂 `=3.00`，逐对精确），多加的 5 篇为低相关论文；实测高相关半区覆盖率 0.311 vs 低相关半区 0.119。**凡以"每近邻判定"为分母的指标，跨近邻预算比较都不可用。**

候选 2「每 claim 最大覆盖」通过，但**必须同口径截断**：

| 比较 | 值 |
|---|---:|
| C（3 篇） | 0.437 |
| D（**前 3 篇**） | 0.440 |
| D（全 8 篇） | 0.583 |
| `D(前3) − C` | **+0.003** |
| 第 4–8 篇边际 | **+0.143** |

**+0.003 是本次最有价值的数字之一**：同等近邻预算下两臂完全相等，证明实验操作干净、差异来自近邻数量本身（截断后出现一对 −0.083，也证明该指标非单调取巧）。**+0.143 说明额外 5 篇近邻确实带来覆盖发现，但未转化为任何被独立评审认可的可发表空白**——即协议 §7 规则 3 预测的**"看到了，但判不出来"**。

新主指标分三层报告（机制层同口径边际增益 / 判定层有序分布 / 质量层 fund 计数），`T = NONE/(...)` **作为主指标废弃**。改造顺序不可颠倒：**先改度量 → 再修 gap 供给 → 最后做 no-audit 消融**。在无余量指标上做消融，得到的 TIE 无法区分"机制无用"与"仪器测不出"。

**Gap 供给修复（`GAP_SUPPLY_FIX.md`，2026-11-06）**

先排除三个假设：gap 类型分布均衡（7 类、116 个）、证据来源多样（9 类）、加近邻无效（`ΔT=+0.0pp`）。真正原因在 `claimed_delta` 的**表述形态**——**直接读 26 条 PARTIAL 的评审理由**（比正则可靠）：**46% 是"各成分被覆盖、但组合未被做"**；49% 的 delta 并列 ≥3 个子句。

两种形态在结构上**不可能被判为 NONE**：

| 形态 | 例子 | 为什么必然 PARTIAL |
|---|---|---|
| 配置型 | "在 ≤1B / NarrativeQA / >30% vs <10% 下" | 主题已覆盖，只是某格子没报 |
| 合取型 | "首次同时做 A+B+C" | 每成分都发过，答案永远是"成分有、组合无" |

而原 prompt 第 881 行**直接要求写具体数字**，等于在鼓励配置型。

**主修复**：`_GAP_MINING_SYSTEM` 禁止两种形态、要求 delta 是**单一机制断言**、并加入自检（删掉数字后仍须断言机制；删掉任一子句后 delta 必须变假）。辅助的合取检测器**默认关闭**——它在要评分的 28 个单元上拟合，启用即重犯 v7.3 的错；实测触发 6.0%（低召回）、7/7 精确。第一次实现触发 0/116，测错形态，已删除。

**修复中真实犯的两个错，都已加测试**：①prompt 里写 `{i}`/`{ii}` 占位，而该串经过 `str.format()`，直接 `KeyError: 'i'` 让**每个 lens 调用崩溃**（全量回归抓到）；②正则 `[^.;!?]` 排除句点，导致含小数的 delta 永不触发。

**必须明确**：本次改动**没有任何证据表明会提高 NONE 率**。要证明需样本外重跑 + 盲评 + 观察 NONE 是否非零。当前只能说：造成 `NONE=0` 的结构性原因已被识别并针对性改动。

**样本外验证结果：形态改了，判定没改（`GAP_SUPPLY_VALIDATION.md`，2026-11-09）**

用 `production_e2e` 中**未参与规则拟合**的主题（pe2e-003 / pe2e-008）重跑。先修了一个会让验证失去意义的问题：服务是 11-03 起的，prompt 是 11-06 改的，**不重启就仍在跑旧 prompt**。

| 特征 | 旧 prompt（116） | 新 prompt（24） |
|---|---:|---:|
| 含数字 | 74% | **0%** |
| 含配方语（`>= / at least / X%`） | 59% | **0%** |

新 delta 全是单一机制断言（"X 不是 Y 的属性，而是由 Z 决定"）。**但盲评 6/6 仍是 PARTIAL，`NONE` 仍为 0。**

**决定性诊断：校准探针。** 喂入两个 gap：

| 探针 | 判定 |
|---|---|
| "Transformer 推理在**月球重力**下是否改变" | **NONE** |
| "训练脚本的**编程语言**是否影响权重" | PARTIAL |

评审器**能**给 NONE——但只有在该领域**完全没有相邻工作时**。原因在 rubric 本身：`PARTIAL` 定义为"覆盖了一部分 delta、**在不同设定下覆盖**、或是已知工作的重新参数化"，中间那句被任何相邻论文满足，而真实主题永远有 300+ 篇相邻论文。

**纠正我上一轮的一个错误**：先前报告的"46% 的 PARTIAL 理由是成分覆盖但组合没做"**是错的**，来自一个宽正则（把句中 "both"/"together"/"combines" 都计入）且把非互斥类别加总。严格匹配合计仅 **8%**。真实主流理由是"部分覆盖"(50%) 与"存在相邻工作"(50%)。已在代码注释、config、测试、文档四处的原位置加更正标注——**prompt 改动仍然正确保留**（配置型/合取型确实被消除），但它**不是** `NONE=0` 的解药。

**真正的瓶颈是评测仪器，不是 gap 供给**。下一步必须先改 rubric 的 NONE/PARTIAL 边界并重跑校准探针，否则任何以 `T` 为主指标的实验（含 no-audit 消融）都会重演 v7.2 的 TIE。

**785 个测试全部通过**（原 767 + 18）。

---

## Rubric v2：仪器修复（`RUBRIC_V2_RESULTS.md`，2026-11-09）

诊断已明确——`NONE` 不可达是 **rubric 缺陷**，不是 gap 供给问题。v1 的 PARTIAL 含一句

> "prior work covers PART of the delta, **covers it in a different setting**"

中间这句由任何相邻论文满足，而真实主题永远有数百篇相邻论文 → `PARTIAL` 吸收一切，`NONE` 要求文献真空。

**v2 三处改动**：①判断单位是 **delta（具体断言）而非 topic**，且必须先在 `delta_stated` 复述；②"不同设定"不再足以构成 PARTIAL；③新增"相邻 ≠ 覆盖"，显式映射到 NONE。另加 `delta_established` / `adjacent_work_exists` 两个诊断字段使改动可审计。v1 原样保留，历史数字可复现。

**预注册在先**（`RUBRIC_V2_PREREG.md`），门槛冻结，且**同时防范反向退化**（用"永远 NONE"替换"永远 PARTIAL"同样无信息量）：

| 门 | 条件 | 实测 |
|---|---|---|
| G1 | 正向探针 ≥1/2 判 NONE | **1/2** PASS |
| G2 | 反向探针 3/3 判 FULL 且无 NONE | **3/3** PASS |
| G3 | verdict 与 delta_established 一致 ≥4/5 | **5/5** PASS |

反向探针用经典论文（Transformer / LoRA / CoT）验证不退化——**3/3 判 FULL**。正向探针 `p2v-N2` 我预期 NONE 但实得 PARTIAL，**是我预期错了**（v2 检索到 `[10] "Balanced Usage Is Not Expert Specialization"` 已建立该机制）。

**配对重测 34 个 gap（同批单元、同批检索，只换仪器）**：

| | v1 | v2 |
|---|---:|---:|
| NONE | 0 | **7** |
| PARTIAL | 32 | 26 |
| FULL | 2 | 1 |
| **T** | **0.0%** | **20.6%** |

内部一致性 34/34；**7 个单元是"相邻工作存在但 delta 未被检验 → NONE"**，正是 v1 结构上无法输出的格子；变化全部是收紧（7 个 `PARTIAL→NONE`、1 个 `FULL→PARTIAL`），**无一个 `NONE→PARTIAL`**。

**必须记录的一个自我纠正**：v2 把我人工判过 5 个单元中的 2 个（`nb-015`、`nb-099`）从 PARTIAL 翻成 NONE，与我只 **3/5** 一致。判 **v2 对、我错**——v2 逐篇点名了反驳证据（如 COMI 在**完全相同的 25%/50% 保留率**上已量化冗余，还主动降置信度到 medium），而我的理由只有一句概括，且是在不重新检索的情况下凭单元文本判断的。

**这证实了我此前的警告**：那份"21/21 一致率"不是可靠性证据——**我当时也在用 v1 的隐含标准**（主题有相邻工作 → PARTIAL）。我的人工抽查被同一套 rubric 锚定了。

**明确不做**：不用 v2 的 20.6% 替换近邻实验的 0.0%。那是换仪器后重测，不是新数据。

**798 个测试全部通过**（原 785 + 13）。

---

## 独立复核与 v2.1（`RUBRIC_V2_INDEPENDENT_CHECK.md` / `RUBRIC_V21_FINAL.md`，2026-11-09）

**第一次真正独立地校验评审器，结果是 kappa = 0.000（3/14 一致，等于随机）。** 但这个结果主要暴露了**我的验证设计有缺陷**，而且拆开后双方各有错。

### 我的两个设计错误

1. **抽样结构性偏差**：我为避污染排除"已看过判定的单元"，而**被排除的恰好包含 v2 全部 7 个 NONE**（它们之所以被查看，正因为它们翻转过）。于是我把仪器表现最好的样本系统性剔除了 → 样本内 0/14 判 NONE 部分是假象。
2. **证据基础不对等**：我只依据管线提供的 ≤6 条证据，v2 每次自己跑文献检索（引 6–13 篇）。我的"没有证据检验 X"实际是"**管线抽取的证据里没有**"，不是"文献里没有"。**11 个分歧中 4 个（`nb-002/012/045/062`）因此是我错、v2 对**——如 `nb-012`，v2 找到 `[9] "Dense Contexts Are Hard Contexts"` **在固定 12k token 下**直接检验了密度效应。

### v2 的残余问题（已修）

**11 个分歧中 7 个，v2 只引用相邻工作、没有直接检验，却仍判 PARTIAL。** 自动扫描确认 **7/7 都靠"否定部分被削弱"或"检验了部分"来支撑**。以 `nb-065` 为例，v2 自己的笔记写"Search results do **NOT** contain a study that measures this correlation coefficient"，紧接着仍是 PARTIAL。

**v2.1 只加一条规则**：`NO CREDIT FOR WEAKENING AN ABSENCE CLAIM`——若 delta 实质是"A 比 B 好 X"，只有报告了 A vs B 的比较才算覆盖；相邻工作削弱"文献中尚无此比较"这一否定陈述**不构成**部分确立。附操作性自检：**若你正在写"没有检索工作直接测量 X，但有相关工作"，那就是 NONE**。

| | v1 | v2 | **v2.1** |
|---|---:|---:|---:|
| G1 正向探针判 NONE | — | 1/2 | **2/2** |
| G2 反向探针判 FULL | — | 3/3 | **3/3** |
| `T`（34 单元） | 0.0% | 20.6% | **55.9%** |
| 与人判一致率（14） | 3/5（被锚定） | 21% | **57%** |

v2→v2.1：12 个 `PARTIAL→NONE`，**无一个反向**；此前标记的 7 个残余兜底单元中 **4 个被修正**。

### 我停在这里，不再调参

`T=55.9%` 偏高，且有 1 例改差（`val-005`，证据确实记录了单闭环零分失效这一必要子命题，我认为 PARTIAL 更对）。**我可以再改一版让一致率更高，但不会**：①我自己的判断有 4/11 是错的，朝它调参是向有已知错误的基准拟合；②这正是我批评 v1 的方式——反复调 rubric 直到它与我一致，只是把锚点从 v1 换成"我"；③55.9% 是好是坏**无法从这 34 个单元判定**，需要新数据而非更好的 rubric。

**`T=55.9%` 只能作为"指标终于有余量"的证据，不能作为质量结论。** 下一步需要真正的外部独立标注（给予同等检索条件）。

**804 个测试全部通过**（原 798 + 6）。

**gap 供给侧候选闸门（默认关闭）**：`GAP_MINING_OPERATIONALIZATION_GATE` 拒绝"测量动词开头 + 多交付物合取 + 无机制词汇"的 `claimed_delta`（26/37 的 delta 是这种未执行测量模板）。在 37 单元上触发 5 次（3 FULL/2 PARTIAL，60% 假开放 vs 10.8% 基线，不漏 NONE），但**这 5 个单元就是推导出该规则的同一批单元**——事后拟合，故默认关闭，开启需 24 主题池的样本外检验。

**工程状态更新**：`767` 个测试全部通过（2026-11-03，原 728 + 新增 39）。

**v7.2 全链路真实探针（2026-10-30，两轮）**：
- **探针 A**（`df744606`，test-time compute scaling for small LMs）：119 分钟，产出 2 个 `executable_candidate` Idea + 2 份实验计划；v7.2 两道新闸门零误杀。据实测落地优化：审计 Top-1→Top-2、覆盖度增量匹配。
- **探针 B**（`0ae1a146`，long-context agent memory compression，全部修复加载后）：**67 分钟（−44%）**，产出 1 个 `executable_candidate` + 12 个方向性 Idea，3 个确认型 gap（A 只有 1 个）。逐项验收全部通过；同时暴露并修复两个新问题：① 增量覆盖度的水位线误用 `coverage_records.created_at`（该表 upsert 且不更新时间戳，导致 r6 把 r2 证据重匹一遍，`reused=1804` 与 r2 完全相同）——改用 `phase_runs.completed_at`；② P1 killer 检索缺"未命中"trace（无法区分守卫跳过 vs 检索无果）——已补 `gap_audit_uncertain_killer_search`。完整对照：`backend/eval_results/v72_pipeline_probe_report.md` §8。

---

## 1. 整体流程

### 1.1 主链路（Pipeline V2）

```mermaid
flowchart TD
    START([POST /api/tasks/{id}/start]) --> CLAR[clarify_topic<br/>方向是否明确?]
    CLAR -- 不明确 --> WAIT1[waiting_for_clarification] --> CLAR
    CLAR -- 明确 --> CONTRACT[build_contract<br/>结构化方向 + 资源约束]
    CONTRACT --> DECOMP[decompose_research_space<br/>5-12 个可检索问题]

    DECOMP --> SRCH[_run_search_loop<br/>证据积累循环]
    SRCH --> QGEN[generate_queries] --> SEARCH[search_papers<br/>多源检索+去重+相似度预过滤]
    SEARCH --> SCORE[score_papers<br/>含批内校准]
    SCORE --> EXTR[extract_evidence<br/>section 轮转预算]
    EXTR --> COV[update_coverage<br/>问题-证据覆盖矩阵]
    COV --> STOP{should_stop?}
    STOP -- 否 --> QGEN
    STOP -- 是 --> READY{readiness_gate}

    READY -- 不通过 --> BRIEF1[targeted_research<br/>O2 定向补检索]
    BRIEF1 -- 预算耗尽 --> END1[generate_landscape_brief<br/>more_research_required]
    BRIEF1 -- 有预算 --> SRCH
    READY -- 通过 --> PIPE[_run_opportunity_pipeline]

    PIPE --> MINE[mine_gaps<br/>证据注入有界]
    MINE --> AUDIT[audit_gaps<br/>对抗审计 + killer 检索]
    AUDIT --> SURV{有 surviving gap?}
    SURV -- 否 --> NARROW{narrow_gaps<br/>可收窄?}
    NARROW -- 是 --> AUDIT
    NARROW -- 否 --> BRIEF2[targeted_research] --> MINE
    BRIEF2 -. 预算耗尽 .-> END2[generate_landscape_brief<br/>more_research_required]

    SURV -- 是 --> PHEN[generate_phenomenon_plans]
    PHEN --> INTERV[generate_interventions<br/>三道硬闸门 + A/B/C 分档]
    INTERV --> PASS{有通过闸门的方案?}
    PASS -- 否 --> BRIEF2
    PASS -- 是 --> MINE2[generate_minimal_experiments<br/>hypothesis 聚簇 + 质量门禁]

    MINE2 --> IDEAS[_run_ideas_loop<br/>idea 级 novelty 快检 + 四级定级]
    IDEAS --> RAG[_run_rag_indexing] --> WIKI[_run_wiki_ingest]
    WIKI --> LAND[generate_landscape_brief] --> DONE([waiting_for_user_review / abstained])
```

> 图中关键机制：`_run_search_loop`（`runner.py:1712`）负责证据积累；`_run_opportunity_pipeline`（`runner.py:904`）负责机会流水线；`targeted_research` 是贯穿全链路的"失败即补检索"回环（O2），让闸门失败时自动补检索重试而非直接终止。

### 1.2 阶段清单

`runner.py` 的 `run_task()`（`runner.py:376`）按下列阶段编排，每个阶段都经 `phase_runs` 表做幂等把关（`phase_name` + `input_version` + `status`）：

| 阶段 | 模块 | 产出 |
|------|------|------|
| 方向澄清 | `steps/clarify_topic.py` | 是否明确 + 澄清问题 |
| 研究契约 | `steps/build_contract.py` | `research_contracts`（按输入哈希幂等） |
| 问题分解 | `steps/decompose_research_space.py` | `research_questions` |
| 检索循环 | `steps/generate_queries.py` → `search_papers.py` → `score_papers.py` | `search_query_records` / `search_query_papers` / `task_papers` |
| 轮次小结 | `steps/summarize_round.py` | 本轮摘要 + 知识缺口 |
| 证据抽取 | `steps/extract_evidence.py` | `evidence_units`（含原文 span 定位）+ `task_papers.extraction_status` |
| 覆盖度 | `steps/update_coverage.py` | `coverage_records` + `question_evidence_links` |
| 就绪判定 | `steps/readiness_gate.py` | 是否进入机会流水线 |
| 缺口挖掘 | `steps/mine_gaps.py` | `gap_candidates`（7 类 opportunity，per-lens fan-out） |
| 缺口相关性筛选 | `steps/gap_relevance.py` | 审计召回论文的 prior-art 相关性排序（`final_score` 为 NULL 时的近邻选择依据） |
| 对抗审计 | `steps/audit_gaps.py` | `gap_audits` + `neighbor_comparisons` |
| 缺口收窄 | `steps/narrow_gaps.py` | 改写 `claimed_delta` 后重审 |
| 定向补检索 | `steps/targeted_research.py` | 按失败原因补检索（O2） |
| 现象计划 | `steps/generate_phenomenon_plans.py` | `gap_phenomenon_plans` |
| 干预方案 | `steps/generate_interventions.py` | `intervention_candidates` + 闸门结果 |
| 最小实验 | `steps/generate_minimal_experiments.py` | `experiment_plans` + `research_ideas` |
| Idea 定级 | `runner.py::_run_ideas_loop` | 四级 decision + novelty 快检 |
| 论文深度分析 | `steps/analyze_papers.py` | 高优先级论文的结构化分析（`paper_analyses`） |
| 论文聚簇 | `steps/build_clusters.py` | 主题聚类（wiki 优先，LLM 兜底） |
| RAG 索引 | `services/rag_service.py` | `paper_chunks` + ChromaDB |
| Wiki 编译 | `services/wiki_service.py` | `wiki_pages` |
| 研究报告 | `steps/generate_report.py` | STORM 式两步报告 |
| 态势简报 | `steps/generate_landscape_brief.py` | 保底交付物（所有终止路径） |

---

## 2. 快速开始

```bash
# 1. 配置
cp .env.example .env          # 至少填好 LLM 端点与 Embedding
cd backend && pip install -r requirements.txt

# 2. 建库（推荐用 bootstrap 脚本，可处理全新库与遗留库）
python scripts/bootstrap_db.py
# 等价于： python -m alembic upgrade head

# 3. 启动后端
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. 前端（另开终端）
cd frontend && npm install && npm run dev    # http://localhost:3000，/api 代理到 8000

# 5. 测试
cd backend && python -m pytest -q
```

发起一次研究：`POST /api/tasks`（body 含研究方向）→ `POST /api/tasks/{id}/start`。任务终态为 `more_research_required` / `insufficient_evidence` 时可直接再次 `start` 续跑，已入库的论文、证据、覆盖度会被复用（阶段幂等）。

> 单次完整运行约 20–60 分钟，耗时主体是证据抽取（实测约 50s/篇）与外部检索网络往返，而非 LLM 推理（单次调用仅 1–2 秒）。进度增量提交，超时或中断都可续跑。

---

## 3. 技术栈

| 层 | 选型 |
|----|------|
| 后端框架 | FastAPI 0.115 + Python 3.11+ |
| 数据库 | SQLite（WAL）+ Alembic 迁移（`0000` → `0034`） |
| ORM | SQLAlchemy 2.0 |
| LLM | OpenAI 兼容端点，主/备双 provider + 熔断降级；上下文预算 131072（刻意低于实测窗口） |
| 论文源 | Semantic Scholar / arXiv / OpenAlex / Crossref / Unpaywall / CORE / IEEE / 本地语料库 |
| 全文与向量 | PyMuPDF + pdfplumber 解析、ChromaDB 0.5 + 可插拔 Embedding（DashScope `text-embedding-v4` / 1024 维） |
| 前端 | Vite 5 + React 18 + TypeScript 5 + Tailwind 3 |
| Agent 架构 | 轻量自研 Loop（不依赖 LangGraph） |

---

## 4. 两条核心约束

### 4.1 不允许 auto-promote：一切结论可回溯

- **Evidence 可回溯**：`evidence_units` 存 `normalized_claim` + 原文 span + chunk hash + 页码 + `verification_status`，span 定位校验是硬保障，模型不能推断文本未陈述的内容。
- **Gap 可回溯**：`gap_evidence_links` 只允许引用 prompt 实际展示过的证据 ID——否则模型引用一个从未展示的 ID，与凭空编造无法区分。
- **Idea 可回溯**：`motivation` 逐字引用 gap 的 top-2 证据 claim，可独立核对。
- **评分只排序，不定级**：Idea 的四级分类（`executable_candidate` / `conditional_review` / `research_direction_only` / `rejected`）由代码门禁决定，`final_score` 高低不能越过硬门禁。

### 4.2 Generator 与 Auditor 隔离

Gap 由 `mine_gaps` 生成，由 `audit_gaps` 独立审计（独立检索对抗性近邻论文、独立 prompt、独立的 `GAP_SEARCH_POLICY_VERSION`）。审计方的职责是**尽力证伪**这个缺口——找到已经覆盖该 claim 的 killer paper 就关闭或收窄它。Idea 生成后还有一道独立的 idea 级 novelty 快检（把方法投射到相邻领域找是否已有人实现）。

---

## 5. Gap 生命周期

```
candidate ──▶ auditing ──▶ confirmed ──▶ surviving        # 通过对抗审计，进入下游
                  │
                  ├──▶ partially_closed ──▶ 收窄后重审 ──▶ auditing
                  ├──▶ closed ──▶ rejected                 # 已被现有工作覆盖
                  └──▶ uncertain ──▶ rejected              # novelty_undecidable（不可判定）
```

**Opportunity 类型与准入契约（v7，`GAP_MINING_POLICY_VERSION = opportunity-ontology-v7`）**：挖掘不再共用一条"≥2 篇论文 + 局限信号"的统一准入规则——那会天然偏向"作者自己承认的 limitation"，而未验证假设、机制不明、尺度盲区、跨域迁移和结果冲突这些同样高价值的科研机会根本不会以 limitation 的形式出现。v7 定义 7 类 opportunity，每类有自己的发现指令与证据准入契约，按 lens fan-out 逐类生成（每类一次 LLM 调用、每 lens ≤2 候选、类型不匹配直接拒 `TYPE_LENS_MISMATCH`；审计成本仍由 top-K 竞争裁剪，不随候选数放大）：

| 类型 | 发现模式 | 准入契约（通用底线：引用证据须来自 ≥2 篇独立论文） |
|------|----------|------------------------------------------------------|
| `boundary_gap` | 方法在特定条件失效 | 需引用 limitation / negative_result 证据；v7.2 起 claimed_delta 以性能动词开头（"achieves >10%..."）且无边界语义的纯性能目标直接拒 `PERFORMANCE_TARGET_DELTA`（pilot 两个人工判真幸存者都是该形状，评审标注 target_only） |
| `missing_evaluation` | 某关键维度没评测 | 需引用 limitation / negative_result / future_work 证据 |
| `contradiction` | 两组可信结果冲突 | 需引用冲突双方 ≥2 条证据（"Verified conflict" 单元进 prompt 池） |
| `untested_assumption` | 多篇论文共享未经验证的假设 | prevalence 门槛：引用须来自 ≥3 篇独立论文（`ASSUMPTION_NEEDS_BROADER_PREVALENCE`） |
| `mechanism_gap` | 现象存在但原因不明 | 需引用 result / negative_result / comparison / limitation（现象必须被记录） |
| `scaling_regime` | 某一尺度区间没人研究 | 同上另加 dataset（已测范围必须被记录，未测区间须点名）；v7.2 起准入附带 claim 级先例检索——对抗查询→外部检索→保守判定，命中已在声称"未测范围"内做过测量的论文即拒 `SCALING_RANGE_ALREADY_MEASURED`（pilot 7 条中 4 条 false-open 的根因修复；检索基础设施故障只记 trace 不拒，与 idea 级 novelty 快检同一 fail-open 契约，`GAP_MINING_PRECEDENT_CHECK_ENABLED` 可关） |
| `cross_domain_transfer` | 邻域成熟机制未迁移 | 两侧证据齐备：`source_domain_evidence_ids` 与 `target_evidence_ids` 均非空、都在引用集内，且 source→target 映射须显式（`TRANSFER_NEEDS_BOTH_DOMAIN_EVIDENCE`） |

**矛盾不再阻塞挖掘**：v6 及以前，问题挂上 verified contradiction 就整体不可准入（`UNRESOLVED_VERIFIED_CONTRADICTION`），候选自带矛盾证据也被拒（`UNEXPECTED_CONTRADICTING_EVIDENCE`）——最有价值的科研问题形态（"论文 A 说有效、论文 B 说无效，什么变量导致结论翻转？"）在抵达 `generate_interventions` 的 `contradiction_resolver` lens 之前就被上游杀掉了。v7 起矛盾证据带 "Verified conflict" 标注进入 prompt 池，成为 `contradiction` 类 opportunity 的原料；问题级独立论文底线改为 supporting ∪ contradicting 的论文数（1 篇支持 + 1 篇反驳本身就是真实的跨论文冲突）。证据选择也改为按 evidence_type 轮转交错，避免 abundant 的 limitation 证据把 method/result 证据挤出 prompt（assumption / mechanism lens 依赖后者）。

关键语义（每一条都由实测缺陷驱动）：

- **收窄**（`steps/narrow_gaps.py`）：采用审计给出的 `remaining_delta` 作为新 `claimed_delta`，把近邻已覆盖内容并入 `existing_coverage`，回到 auditing 重新争取 confirmed。纯机械操作，不调用 LLM；无具体 delta 则不收窄；单 Gap 上限 2 次（全局 `MAX_NARROWING_PASSES_TOTAL=3`）。
- **只重审有新输入的 Gap**：收窄只改写单个 Gap 的主张，因此下一轮只审计被改写的那些。重审一个输入未变的 Gap 会用同样的对抗 query 得到同样的结论，纯属浪费预算。
- **不可判定终结**：若上轮已判定审计要求 `more_search`，而本轮 claim 与近邻集合完全未变，说明这个诉求无法被满足（例如外部源限流拿不到新对照），该 Gap 直接关闭为不可判定（`audit_result` 保持 `uncertain`，`rejection_reason = novelty_undecidable`）。它作为"未获证实"被如实记录，而不是静默丢弃。
- **surviving 是 contract 级终态**：survivor 查询不限定本轮 gap 列表——早前轮次确认的 Gap 在后续轮次中依然有效。mining 产出为空（新候选全被 `DUPLICATE_GAP` 拦截）时，已确认的 survivor 仍继续流向下游。
- **remediation 后复审而非重新挖掘**：补检索加入新证据会让 evidence fingerprint 变化、mining 幂等失效。此时直接重新 mining 会生成与旧 Gap 语义重复的新候选（措辞变体）并重复消耗审计预算。因此 remediation 轮优先把上一轮未决 Gap（`auditing` / `audited` / `inconclusive`）置回 `candidate` 直接复审，只有 Gap 池全部闭合后才重新 mining。
- **语义去重**：新候选与已有 Gap 的嵌入余弦相似度 ≥ 0.78 判 `DUPLICATE_GAP`，同时 mining prompt 注入已有 `claimed_delta` 列表。

---

## 6. 对抗审计（`steps/audit_gaps.py`）

单 Gap 审计的完整流程：生成对抗查询家族 → 外部检索 → 相关性筛选 → 近邻全文证据抽取 → 判决。全流程由单个 `GAP_AUDIT_TIMEOUT_SECONDS` 超时罩住，内含两处 commit 检查点（搜索+评分后、相关性筛选后），超时回滚不会丢弃已完成的工作。

**查询家族三级兜底**：查询家族的变体验证失败不再直接丢弃家族——① raw 变体 ≥2 时降级接受（`LOW_CONFIDENCE_VARIANTS`）；② raw 变体为空但结构化 intent 完整时，由 `_synthesize_family_variants` 从 intent 的结构化字段合成查询（`SYNTHESIZED_VARIANTS`；根因：pydantic `default_factory=list` 绕过 `min_length` 校验，LLM 返回空变体列表曾导致 4/5 家族被丢弃、准入必然失败）；③ 都不行才丢弃。

**审计模式**（由 `settings.gap_audit_budgeted` / `gap_audit_progressive` 控制，对应 `GAP_SEARCH_POLICY_VERSION`）：

| 模式 | 版本 | 说明 |
|------|------|------|
| Budgeted Falsification | `gap-search-budgeted-v16` | 每个昂贵维度硬上限：4 query × 3 源、Top-10 候选、Top-3 近邻、≤2 篇全文（可疑 killer 优先）、killer 检索 ≤3 query、Top-2 gap competition（2026-10-30 由 Top-1 上调：真实探针 df744606 显示 Top-1 把 7-lens fan-out 剪到每轮只审 1 个 gap、21 个准入候选仅 3 个被审且全部 Idea 挂在单一幸存 gap 上，而单次审计仅 ~250s，裁剪省的成本相对 119 分钟全程可忽略） |
| 渐进式对抗审计 | `gap-search-admission-v15` | 增量重审，只审有新输入的 Gap |
| 传统模式 | `gap-search-admission-v14` | 保留 v14 参数不变 |

**uncertain 判决的 killer 定向检索**（2026-10-30，探针 df744606）：uncertain/more_search 判决在进入 O2 补检索回环**之前**，先用审计自己生成的 killer 描述 + `killer_query_terms` 做一次定向检索（复用 confirmed 路径的 `_run_killer_search` 机制，含术语扩展与嵌入命中判定）——命中 → 直接 `closed/reject`（`UNCERTAIN_KILLER_FOUND`），省掉整轮 remediation；不命中 → 判决保持 uncertain 照常走 O2。动机：探针的两次 uncertain 审计都写着"direct-hit prior work was not retrieved"，而审计自己已命名了要找的 killer，每个 uncertain 却买到了一轮 ~20 分钟的 O2。confirmed 路径本轮已跑过 killer 检索则不重复（record 带 `skipped` 标记区分"跑过"与"守卫早退"）；`perform_search=False` 的回放模式不触发；检索基础设施故障 fail-open（不算命中）。

**检索准入的三条铁律**（每条都由实测缺陷驱动）：

1. 「入选资格」与「执行时机」必须分开：增强层无条件运行，只做 upgrade-in-place；用 `if len(hits) < limit` 守卫会使其永不执行。
2. 层间 `n_hits` 量纲不同，禁止跨层比较。
3. 多源截断必须 round-robin；增强层 SQL 必须 `WHERE p.id IN (候选集)`。

反面范式（勿再犯）：OR 匹配 + 引用排序（任意词都能入选）；命中计数 + 阈值允许全文层自由入选（长综述霸榜）。核心判据是**不同 query 必须返回不同论文**，有回归测试保护。

---

## 7. 证据供给与覆盖度语义

Gap 能否挖出来，取决于抽取阶段是否供给了准入所需的证据类型。

- **抽取覆盖全部候选，没有每轮篇数上限**：旧实现有 `EVIDENCE_MAX_PAPERS`（默认 30，`.env.example` 里甚至是 15）的每轮上限，再叠一层优先级配额。146 个候选配 15 篇上限时，五轮任务最多只能评估 75 篇——语料在构造上就不可达，且"哪些论文被读到"取决于恰好跑了几轮。现在 high/medium 候选全部抽取，成本靠单一共享并发池（`EVIDENCE_EXTRACT_CONCURRENCY`）与 PDF 预热压回来；已存证据的论文会被廉价跳过，因此后续轮次是**累积覆盖**而不是重掷同一个截断切片。
- **chunk 预算按 section 轮转**：每篇论文 6 个 chunk（`EVIDENCE_MAX_CHUNKS_PER_PAPER`）的预算在 section 之间轮转分配（conclusion 优先级最高），而不是按 section 全局排序后截断。PDF 解析把 Discussion / Limitations / Threats to Validity / Future Work 都归入 `conclusion`，而 Gap 准入要求每个研究问题至少有一条局限性信号；按全局优先级排序时，长 method 段会吃满整篇论文的预算，实测导致 conclusion 段一条证据都进不了模型。
- **section 提示**：conclusion 段要求把边界、失败条件、未测设定标为 `limitation` / `negative_result`；method / experiment 段要求记录实际评估的设定、数据集、基线，以及**文本中明确写出**的限制，禁止推断文本未陈述的内容。
- **supporting 语义单一来源**：`app/agent/evidence_relations.py` 统一定义哪些 `relation_type` 算证实（`supports`、`partially_answers`）、哪些算反证，以及 0.5 的最低相关性门槛（`MIN_RELATION_RELEVANCE`），覆盖度与 Gap 挖掘共用。此前两者定义不一致（覆盖度只认 `supports`，把 `partially_answers` 当背景），而匹配 prompt 恰恰把 `partially_answers` 定义为"部分回答/指出局限"、兜底启发式又把每条 `limitation` 映射到它——实测同一批证据在覆盖度侧算 0.15、在挖掘侧却可准入。
- **兜底启发式不再把实测对比降级**：LLM 匹配失败时走 `_determine_relation`，`comparison` 曾映射到 `background`，等于把每一次"对命名基线的实测对比"——恰恰是抽取 prompt 称为最强佐证的证据——踢出 supporting。现在映射到 `partially_answers`：与命名基线的对比是对"X 表现如何"的部分回答，不是背景。
- **覆盖度口径**：覆盖度 = 贡献 supporting 证据的**不同论文数**，不是证据条数。
- **增量匹配（2026-10-30，探针 df744606）**：覆盖度成本曾随证据池单调上涨（实测 246→398→509→667s）——每轮对全量问题×全量证据重做 LLM 匹配。由于证据行不可变、问题文本在 decompose 后不再变化，(question, evidence) 判定是稳定的：现在只把**上一覆盖快照之后新建的证据**送 LLM，已判定 pair（带 link 的）原样重放。`COVERAGE_INCREMENTAL_MATCHING=false` 可恢复全量重匹配（更换匹配 prompt 语义后需要）。
- **抽取状态显式落库**：`task_papers.extraction_status`（`full_text` / `abstract_only` / `not_assessed`）+ `extraction_gap_reason`（`pdf_not_retrieved` / `pdf_unavailable` / `pdf_parse_failed`）区分"这篇论文测过、结论为负"与"这篇论文根本没读到"。没有这个区分，`NO_LIMITATION_SIGNAL` 这类拒因无法归因到"论文确实没提局限"还是"我们没读到 conclusion 段"。

---

## 8. 分级产出与质量门禁

### 8.1 Confidence Tier（不应"闸门不过即丢弃"）

| 档位 | 含义 | 进入下游 |
|------|------|----------|
| A | 通过全部硬闸门，且有可定位的全文证据支撑 | 是 |
| B | 无闸门硬失败，但存在待确认项（某闸门 UNKNOWN/WARN，或缺口仅摘要级证据） | 是 |
| C | 存在闸门硬失败，作为推测性方向保留 | 否（仅供参考） |

> **可行性闸门放宽**：方案顺口提到"训练/微调"不再一票否决；仅当核心方案确实依赖训练且 Contract 声明无 GPU 时才判 FAIL，否则降级为 WARN。

### 8.2 Idea 四级分类（代码门禁定级）

```
executable_candidate    # 硬门禁全过：所属干预 tier=A 且实验计划通过质量校验
conditional_review      # tier B/C，或实验计划被拒（含反馈重试后仍失败）、评分失败
research_direction_only # 仅方向性产出，无可执行实验载体
rejected                # 实验计划被质量门禁拒绝（trace 记录 reason_codes + 计划摘要）
```

系统允许产出 0 个 `executable_candidate`。

### 8.3 Hypothesis Cluster 与实验一致性门禁

（`EXPERIMENT_GENERATION_POLICY_VERSION = experiment-consistency-v13`）

- **聚簇**：同一 Gap 的多个 passed intervention 在生成实验计划前先做一次 LLM 结构化聚簇（`_cluster_interventions_by_hypothesis`），判定标准是"检验的实验假设是否相同"（测什么）而非"机制是否相似"（怎么做）。同簇只产出 1 个 Idea + 1 个实验计划，其余 intervention 作为机制变体并入 baselines 消融臂（记录 `variant_intervention_ids`）。LLM 聚簇失败 / 任一簇 confidence < 0.7 / 出现未知 id 时保守降级为不合并。
- **统计方法门禁**：比例型配对指标（pass rate / regression rate / accuracy 等）配 plain t-test 且未列非参数替代（McNemar / Wilcoxon / permutation / bootstrap）即拒（`STATISTICAL_TEST_MISMATCH`）。
- **模型范围门禁**：从 gap 的 target_setting 提取参数上限（`<N b` 符号或 "under/below/smaller than N b" 文字）或 SLM 关键词（默认 7B cap），`model_spec` 中数值越界即拒（`MODEL_SCOPE_CONFLICT`）。
- **反馈重试**：被质量门禁拒绝后带 `reason_codes` 回传重写一次（场景原子为字面子串匹配，如 "verification" 不满足原子 "verifier"）；仍失败才判 rejected。生成 prompt 已同步前置统计方法与模型范围规则，让首次生成即合规。

### 8.4 Idea 级 Novelty 快检（P2-C）

Gap 审计验证的是 **GAP** 的新颖性（机制缺失），不是**具体方法**的新颖性（可能已有人造出了这个技术）。因此每个 cluster 的 idea 在定级后执行独立检索：LLM 生成 3 条对抗查询（exact-method / method-neighbour / **adjacent-domain**——把方法投射到最可能率先发表它的相邻领域，如 correction filtering → program repair patch filtering），按 rank 取 top-5 论文，LLM 保守判定"是否直接实现了同一机制 + 同一目的"。命中 → 降级 `conditional_review` + `METHOD_ALREADY_PUBLISHED` + 命中论文并入 `related_paper_ids`。

**基础设施失败（LLM / 检索异常）只记 trace 不降级**——外部源波动不是关于 idea 的证据。

### 8.5 Idea 元数据差异化（P2-B）

title 由模型按具体机制命名，prompt 禁止 + 代码 strip 双保险杜绝 "Minimal Experiment:" 前缀；`method_sketch` 使用 idea 视角的假设-方法-预期结果概述（不再逐字复制 intervention 工程描述）；阶段重跑时上一代 active idea 全部软删除（`superseded`），任务始终只呈现当前规则下的一代产物——重跑产出 0 个新 idea 时如实显示 0。

---

## 9. LLM 请求预算与上下文管理

本地部署的上下文窗口是硬约束（当前 DeepSeek-V4.1-Flash 实测 `max_model_len=557056`，配置里只声明 131072），且后端在两种情况下都只回一个笼统的 400：输入本身过长，或 `输入 + max_tokens` 超过窗口。以下机制保证 prompt 规模与轮次无关、且失败可归因：

- **输入准入**：发送前按字符估算 prompt 规模，超过 `上下文 − 预留输出 − 512` 即拒绝并抛 `LLMContextOverflow`，而不是等一个无法归因的 400。
- **估算自校准**：字符估算没有单一正确除数（UUID 密集文本实测约 1.8 字符/token，散文约 4），因此用后端返回的 `usage.prompt_tokens` 持续校准比值，取最坏近期观测并缓慢衰减；后端拒绝时还会从报错里的实测 prompt 规模反向校准。
- **输出额度**：`LLM_MAX_OUTPUT_TOKENS` 是准入时预留的**下限**而非上限。短 prompt 会按「上下文 − prompt 上界」申请更大输出额度（报告类输出远超 4k tokens），该上界按 `估算 × 1.5 + 2048` 计算——因为后端拒绝 `输入 + max_tokens > 上下文`，用估算值直接计算会在估算偏低时越界。
- **额度自愈**：若因 `输入 + max_tokens` 超限被拒，先用预留下限重发一次（额度是本系统自己选的，不该让某个阶段为此失败），prompt 本身过大才上报。
- **注入有界**：凡是"把某张表的全部行拼进 prompt"的步骤都必须有界。Gap 挖掘按每问题 8 条 / 每问题内每论文 2 条 / 全局 40 条三层上限选取证据，并在问题之间轮转，使全局上限截断长尾而不是让整个问题从 prompt 中消失；校验引用时只认实际注入的证据 ID。
- **供应商熔断**：`FallbackLLMProvider` 包装多个 provider，仅 transport / 服务错误时切换（`LLMBudgetExceeded` / `LLMContextOverflow` 等确定性错误直接抛出）。单个 provider 连续 2 次失败进入冷却，30s 起步指数退避、上限 300s。

---

## 10. 失败处理策略

| 场景 | 处理 |
|------|------|
| 单个数据源失败 | 记录错误并跳过该源，继续其他数据源 |
| 数据源返回 429 | 交给源级 cooldown 接管，不做短间隔退避重试（限流通常是出口 IP 级，重试只会加剧） |
| PDF 下载失败 | 按**原因**而非一律重试：429 / 5xx / 超时 / 连接错误 → transient，退避重试（3 次，2s/8s）；403 / 401 → paywall 或 bot 拦截，永久失败不重试；404 / 无 PDF 链接 → 永久失败。实测依据：从未缓存的 arXiv 论文，单纯重试可恢复 40/40、再 30/30，说明失败是限流窗口而非论文属性；而重试付费墙只是每篇每轮白等一分钟 |
| PDF 预热失败 | 不阻断抽取，按 `extraction_status` / `extraction_gap_reason` 如实记录，抽取用磁盘上已有的语料继续 |
| 全部数据源失败 | 本轮重试一次 |
| 重试后仍失败 | 任务标记 `failed`，等待用户重启或修改配置 |
| LLM 调用失败（可重试） | 重试 3 次（指数退避） |
| LLM 主模型故障 | `FallbackLLMProvider` 自动切备用模型；确定性错误不降级 |
| LLM 返回非法 JSON | 追加一次纠正轮次；只回显 400 字符摘录，保证重试请求不大于首轮 |
| LLM prompt 超出上下文 | 抛 `LLMContextOverflow`（与"服务故障"区分），由调用方缩小输入 |
| 检索循环失败但已完成 ≥1 轮 | 降级 `more_research_required`，保留已完成轮次的检索/证据/覆盖度，照常产出 Landscape Brief |
| Gap 挖掘阶段失败 | 降级 `more_research_required` 并照常产出 Brief，已完成的检索/证据/覆盖度全部保留、可续跑 |
| 单个 Gap 的审计决策非法 | 只废弃该条决策，其余 Gap 结论保留（失败粒度与输出粒度对齐） |
| SQLite 写锁冲突 | 一律走 `app/db/lock_retry.py` 的 `retry_on_locked(_async)` / `flush_with_retry` / `commit_with_retry`（先 rollback 再重试），禁止打点式逐处添加 |

---

## 11. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_ROUNDS` | 5 | 检索循环最大轮数 |
| `QUERIES_PER_ROUND` | 5 | 每轮生成 query 数 |
| `PAPERS_PER_SOURCE_PER_QUERY` | 15 | 每源每 query 返回论文数 |
| `HIGH_PRIORITY_TARGET` | 15 | 高优先级论文目标数量 |
| `EVIDENCE_EXTRACT_CONCURRENCY` | 4 | 同时在抽取的论文数（单一共享池，无批间屏障） |
| `EVIDENCE_MAX_CHUNKS_PER_PAPER` | 6 | 每篇论文送入 LLM 的 chunk 数，按 section 轮转 |
| `EVIDENCE_CHUNK_CONCURRENCY` | 4 | 单篇论文内并发抽取的 chunk 数 |
| `ENABLE_PDF_PREWARM` | true | 抽取前先把本轮 PDF 全部取回，把"能否拿到语料"与"语料说了什么"解耦 |
| `PDF_PREWARM_CONCURRENCY` | 3 | 预热下载并发（刻意低于抽取并发，arXiv 在 ~3 req/s 以上会限流） |
| `LLM_CONTEXT_TOKENS` | 131072 | 必须不超过实际模型窗口（当前 DeepSeek-V4.1-Flash 实测 557056，刻意低估） |
| `LLM_MAX_OUTPUT_TOKENS` | 8192 | 准入时预留的输出下限，不是上限 |
| `MAX_LLM_CALLS_PER_TASK` | 800 | 任务级 LLM 调用预算（审计/聚簇/抽取共享） |
| `AGENT_TIMEOUT_SECONDS` | 3600 | 任务级超时 |
| `GAP_AUDIT_TIMEOUT_SECONDS` | 600 | 单 Gap 审计全流程超时（600s 会掐断走到近邻抽取阶段的 Gap） |
| `GAP_AUDIT_MAX_QUERIES` | 12 | 每 Gap 对抗查询上限 |
| `AUDIT_NEIGHBOR_EVIDENCE_MAX_PAPERS` | 5 | 近邻全文证据抽取篇数上限 |
| `MAX_REMEDIATION_ROUNDS_TOTAL` | 3 | O2 定向补检索全局轮数上限 |
| `MAX_NARROWING_PASSES_TOTAL` | 3 | Gap 收窄总次数上限（单 Gap 上限 2 次） |
| `GAP_MINING_PRECEDENT_CHECK_ENABLED` | true | v7.2：scaling_regime 准入的 claim 级先例检索开关 |
| `GAP_MINING_PRECEDENT_CHECK_TOP_K` | 5 | 先例检索每候选送判的论文数上限 |
| `MAX_GAPS_TO_DEEP_AUDIT` | 2 | budgeted 模式每轮深审 gap 数（2026-10-30 由 1 上调，见第 6 节） |
| `GAP_STARVATION_DEFERRAL_LIMIT` | 3 | 候选被 deferred 达此次数即转 rejected；0 关闭退休（饥饿加权仍生效） |
| `IDEA_SCORE_GO_LINE` | 0.70 | Idea 判定 go 的分数线（2026-10-31 由硬编码改为配置；该线只校准于 v3–v7 分布，v7.2 median 0.654） |
| `IDEA_SCORE_REVISE_LINE` | 0.50 | Idea 判定 revise 的下限；低于此判 reject |
| `COVERAGE_INCREMENTAL_MATCHING` | true | 覆盖度增量匹配：只 LLM 判定新证据，已判定 pair 重放 |
| `SEARCH_PREFILTER_MIN_SIMILARITY` | 0.45 | 入库前主题相似度过滤阈值（0 关闭） |
| `SCORE_CALIBRATION_MIN_BATCH` / `_STRENGTH` | 5 / 0.15 | 评分批内校准最小批量与强度 |
| `MAX_CONCURRENT_AGENTS` | 2 | 并发任务上限（防 SQLite 写冲突） |
| `ENABLE_RAG_INDEXING` | true | 关闭后仍会抽取证据，只是不建全文索引 |

**终止条件**（`agent/policy.py::should_stop`，满足任一即停止）：当前轮数 ≥ `max_rounds`；高优先级论文数 ≥ `HIGH_PRIORITY_TARGET`；当前轮数 ≥ 2 且最近一轮新增高优先级论文数为 0；连续 2 轮新增论文重复率 > 0.75；用户手动停止。

**就绪门禁判据**（`steps/readiness_gate.py`）：高重要性问题的 covered 判据是 `distinct_supporting_papers >= 1`（或 `coverage_score >= MIN_HIGH_IMPORTANCE_COVERAGE = 0.3`），不是 `coverage_score > 0`。旧判据太弱——评分下限本身就高于 0，任何一条弱相关链接都能让问题"看起来被覆盖"，门禁形同虚设。另有 `MIN_HIGH_IMPORTANCE_SNAPSHOT_RATIO = 0.6` 要求高重要性问题中至少 60% 有覆盖度快照。字段名 `high_importance_covered_count` 是历史遗留，实际含义是"有快照的高重要性问题数"，不是质量门槛。

---

## 12. 评分公式

### 论文评分

```
paper_score = 0.25×relevance + 0.30×authority + 0.15×recency + 0.15×novelty + 0.15×idea_potential
```

- **authority 调整**：缺失元数据（无引用+无年份）打 0.7 折；顶会/顶刊（ICML / NeurIPS / ICLR / CVPR / ACL 等）加 0.1（上限 1.0）。
- **跨论文校准**：单轮评分论文数 ≥ `SCORE_CALIBRATION_MIN_BATCH`(5) 时，对 `final_score` 做批内校准 `adjusted = s + 0.15 × (s − batch_mean)`，把趋同的分数拉开区分度（历史上独立评分区分度仅约 0.05）。批均值保持稳定，因此下方阈值仍适用。

| 分数 | 优先级 |
|------|--------|
| ≥ 0.75 | high |
| 0.5 – 0.75 | medium |
| < 0.5 | low |

### Idea 评分

```
idea_score  = 0.20×novelty + 0.20×feasibility + 0.20×significance + 0.20×evidence_support
            + 0.10×differentiation + 0.05×experimentability + 0.05×potential_impact
final_score = idea_score − 0.08×risk
```

> 该 8 维打分与加权公式由 V1（`generate_ideas.py::_score_idea`）与 V2（`generate_minimal_experiments.py` 复用同一函数）共用。**评分只用于排序，不作为放行门禁**——放行由第 8 节的代码门禁决定。
>
> V1 链路另有验证惩罚：编造基线 −0.15/个（上限 −0.3），编造数据集 −0.05/个，指标-假设不匹配 −0.05/个，非新颖 −0.1。

### 去重策略（优先级从高到低）

DOI → arXiv ID → Semantic Scholar ID → OpenAlex ID → 标题 hash（normalized_title 的 SHA-256）→ 标题相似度（> 0.95 视为重复）。

---

## 13. 项目结构

```
deep research/
├── README.md                      # 主设计文档（本文件）
├── .env / .env.example
├── docs/                          # 设计文档与决策记录，索引见 docs/INDEX.md
│
├── backend/
│   ├── requirements.txt
│   ├── pytest.ini                 # asyncio_mode=auto, testpaths=tests
│   ├── alembic.ini
│   ├── alembic_migrations/
│   │   └── versions/              # 0000_initial_schema → 0034_task_paper_extraction_status
│   ├── scripts/
│   │   ├── bootstrap_db.py        # 建库/升级（处理全新库与遗留库，含备份+校验）
│   │   ├── extraction_audit.py    # 证据供给漏斗只读诊断（extraction_method 全史口径 / 双管线 section 对账 / 拒因）
│   │   └── killer_search_audit.py # killer 定向检索命中率统计（uncertain 路径执行数 / 命中率 / 跳过原因）
│   │
│   ├── app/
│   │   ├── main.py                # FastAPI 入口 + 路由挂载 + 中断任务恢复
│   │   ├── config.py              # Pydantic Settings（全部可配参数）
│   │   │
│   │   ├── agent/
│   │   │   ├── runner.py          # Agent Loop 主调度（run_task / _run_search_loop / _run_opportunity_pipeline）
│   │   │   ├── state.py           # ResearchState
│   │   │   ├── policy.py          # should_stop / early_termination_check
│   │   │   ├── prompts.py         # 全部 LLM prompt 模板
│   │   │   ├── evidence_relations.py  # supporting/反证关系与最低相关性的单一定义
│   │   │   └── steps/             # 24 个独立可单测的步骤模块（见 1.2 阶段清单）
│   │   │
│   │   ├── api/
│   │   │   ├── auth.py            # API Key 鉴权
│   │   │   ├── deps.py
│   │   │   └── routes/            # tasks / contracts / papers / reports / ideas /
│   │   │                          # experiments / gaps / interventions / events / traces
│   │   │
│   │   ├── db/
│   │   │   ├── session.py         # engine + SessionLocal
│   │   │   ├── models.py          # 全部 ORM 模型
│   │   │   ├── lock_retry.py      # SQLite 写锁重试契约
│   │   │   └── repositories/      # task / paper / gap / intervention / phase / search_query
│   │   │
│   │   ├── llm/
│   │   │   ├── base.py            # LLMProvider 抽象 + LLMBudgetExceeded / LLMContextOverflow
│   │   │   ├── venus_provider.py  # 默认 provider（OpenAI 兼容，含 schema 回显解包）
│   │   │   ├── openai_provider.py
│   │   │   ├── fallback_provider.py  # 主备降级 + 供应商熔断
│   │   │   └── factory.py
│   │   │
│   │   ├── paper_sources/         # arxiv / semantic_scholar / openalex / crossref /
│   │   │                          # unpaywall / core / ieee / local_corpus / citation_snowball
│   │   │
│   │   ├── schemas/schemas.py     # Pydantic 请求/响应模型
│   │   └── services/
│   │       ├── search_service.py      # 多源检索调度
│   │       ├── scoring_service.py     # 论文评分
│   │       ├── phase_service.py       # 阶段幂等（phase_runs）
│   │       ├── rate_limiter.py        # TokenBucket 限流
│   │       ├── embedding_service.py   # 可插拔 Embedding 后端
│   │       ├── rag_service.py         # 全文 RAG（PDF → chunk → ChromaDB）
│   │       ├── wiki_service.py        # LLM Wiki 增量编译
│   │       ├── metadata_enrichment.py # 元数据富化（含唯一索引守卫）
│   │       └── event_service.py       # SSE 事件推送
│   │
│   ├── eval/                      # 评测主线
│   │   ├── run_eval.py            # CLI 入口 + adapters
│   │   ├── common.py / config.py
│   │   ├── benchmarks/            # 外部 benchmark 仓库（ResearchBench / RINoBench，含各自 .git）
│   │   ├── internal/              # 内部回归集
│   │   ├── researchbench/         # ResearchBench adapter
│   │   ├── rinobench/             # RINoBench adapter
│   │   └── production_e2e/        # 三系统对照实验 + v7 replay（见第 18 节）
│   │
│   └── tests/                     # pytest 测试集（含 tests/eval/）
│
└── frontend/
    ├── package.json / vite.config.ts / tailwind.config.js / tsconfig.json
    └── src/
        ├── main.tsx / App.tsx / index.css
        ├── api/client.ts          # API 封装
        ├── types/index.ts         # TS 类型 + 状态标签映射
        ├── hooks/useTaskEvents.ts # SSE 订阅（含指数退避重连）
        ├── utils/time.ts
        └── components/            # TaskList / TaskDetail（10 个 Tab）/ ClarifyDialog /
                                   # StatusBadge / EventLog / ReportView / PapersView /
                                   # WikiView / CoverageView / GapsView /
                                   # InterventionsView / IdeasView / ExperimentsView / TracesView
```

---

## 14. 数据库设计

### 14.1 Pipeline V2 表一览

V2 的关键状态全部结构化落库，而不是塞进 `state_json`，这样每个结论都能回溯到证据、且续跑幂等：

| 表 | 作用 |
|----|------|
| `research_contracts` | 结构化研究方向 + 资源约束；按输入哈希幂等，改写方向会 supersede 旧版本 |
| `research_questions` | 由 Contract 分解出的 5-12 个可检索问题（含 importance / status） |
| `evidence_units` | 可回溯的证据单元：normalized_claim + 原文 span + chunk hash + 页码 + verification_status |
| `question_evidence_links` | 问题-证据关系（relation_type + relevance_score） |
| `coverage_records` | 每个问题的覆盖度快照（按贡献 supporting 证据的**不同论文数**驱动） |
| `gap_candidates` | 研究缺口候选（claimed_delta / existing_coverage / status / mining_policy_version） |
| `gap_evidence_links` | 缺口引用的证据（仅允许引用 prompt 实际展示过的证据） |
| `gap_audits` | 对抗审计记录（近邻、审计结论、recommended_action、audited_claimed_delta） |
| `neighbor_comparisons` | 缺口与近邻论文的逐篇对比，按 (gap_id, paper_id) upsert |
| `gap_phenomenon_plans` | 缺口对应的现象验证计划 |
| `intervention_candidates` | 干预方案 + 三道硬闸门结果 + confidence_tier |
| `search_query_records` / `search_query_papers` | 每条检索 query 的执行记录与命中论文，审计的检索准入依赖它 |
| `phase_runs` | 阶段级幂等与重试记录（phase_name + input_version + status） |
| `paper_analyses` / `paper_chunks` / `paper_roles` / `paper_citations` | PDF 全文分析、RAG 分块、论文角色、引用关系 |
| `wiki_pages` | LLM Wiki 增量编译的 Markdown 页面 |

> **`phase_runs.input_version` 契约**：必须包含「该阶段自身的策略版本 + 被判定对象的当前内容」。只写前者会导致规则改了却跳过阶段，只写后者会导致改了规则仍复用旧结论。Content 型输入的版本用内容哈希，而非数量。

### 14.2 基础表

| 表 | 关键字段 |
|----|----------|
| `research_tasks` | `id` / `user_input` / `normalized_topic` / `status` / `current_round` / `max_rounds` / `stop_reason` / `state_json` |
| `papers` | 全局论文表：`title` / `abstract` / `authors_json` / `year` / `venue` / `doi` / `arxiv_id` / `semantic_scholar_id` / `openalex_id` / `url` / `pdf_url` / `citation_count` / `sources_json` / `raw_json` / `normalized_title` / `title_hash` / `is_oa` |
| `task_papers` | 任务-论文关联与评分：`discovered_round` / `relevance_score` / `authority_score` / `recency_score` / `novelty_score` / `idea_potential_score` / `final_score` / `priority` / `reason` / `summary` / `extraction_status` / `extraction_gap_reason` |
| `research_rounds` | 检索轮次：`round_number` / `queries_json` / `papers_found` / `new_papers` / `duplicate_rate` / `summary` / `knowledge_gaps_json` |
| `reports` | 研究报告：`content_markdown` / `content_json` |
| `research_ideas` | Idea：8 维评分 + `decision`（四级） + `related_paper_ids_json` + `user_selected` + `score_status` / `score_error` / `quality_reason_codes_json` / `variant_intervention_ids_json` / `idea_status`（active / superseded） |
| `experiment_plans` | 实验方案：`hypothesis` / `dataset` / `dataset_provenance` / `model_spec` / `oracle` / `statistical_analysis` / `resource_budget` / `scenario_atoms_json` / `baselines` / `metrics` / `steps_markdown` / `steps_json` / `risks` |
| `agent_traces` | 执行轨迹：`step_name` / `step_type` / `round_number` / `input_json` / `output_json` / `llm_tokens_used` / `duration_ms` |
| `user_feedbacks` | 用户反馈：`feedback_type` / `content` / `selected_idea_ids_json` / `need_more_research` |

### 14.3 任务状态枚举

```
pending                       # 已创建，未启动
clarifying                    # 正在分析方向是否明确
waiting_for_clarification     # 等待用户回答澄清问题（前端需等待）
building_contract             # 构建 Research Contract
decomposing                   # 分解 Research Questions
searching                     # 检索循环中（含 O2 定向补检索）
summarizing                   # 生成本轮摘要
extracting_evidence           # 抽取 Evidence Units
updating_coverage             # 更新问题-证据覆盖矩阵
analyzing_papers              # PDF 全文分析
mining_gaps                   # 挖掘证据支撑的研究缺口
auditing_gaps                 # 对研究缺口做对抗审计
synthesizing_ideas            # 生成干预方案
generating_ideas              # 生成 Idea
judging_ideas                 # Idea 深度评估
generating_experiment         # 生成最小实验方案
reporting                     # 生成研究报告
waiting_for_user_review       # 等待用户查看报告/缺口/创意并反馈（前端需等待）
more_research_required        # 证据不足/无存活缺口，已产出 Landscape Brief
insufficient_evidence         # 证据不足以支撑任何缺口
abstained                     # 系统主动弃权，未产出可信 Idea
done                          # 完成
stopped                       # 用户手动停止
failed                        # 执行失败
```

### 14.4 唯一约束与索引

```sql
-- papers: 部分唯一索引（仅对非 NULL 值约束）
CREATE UNIQUE INDEX idx_papers_doi      ON papers(doi)                 WHERE doi IS NOT NULL;
CREATE UNIQUE INDEX idx_papers_arxiv    ON papers(arxiv_id)            WHERE arxiv_id IS NOT NULL;
CREATE UNIQUE INDEX idx_papers_s2       ON papers(semantic_scholar_id) WHERE semantic_scholar_id IS NOT NULL;
CREATE UNIQUE INDEX idx_papers_openalex ON papers(openalex_id)         WHERE openalex_id IS NOT NULL;

CREATE INDEX idx_papers_title_hash ON papers(title_hash);
CREATE INDEX idx_papers_year       ON papers(year);
CREATE INDEX idx_papers_citation   ON papers(citation_count);

CREATE UNIQUE INDEX idx_task_papers ON task_papers(task_id, paper_id);
CREATE UNIQUE INDEX idx_rounds      ON research_rounds(task_id, round_number);
```

> **enrichment 唯一索引守卫**（`services/metadata_enrichment.py`）：同一物理论文可能存在两行（如 OpenAlex 行与 S2 行，检索去重未合并时），S2 enrichment 会把 A 行的 DOI"补"给 B 行。pipeline 模式（共享 agent session）下该改动悬在事务中不提交，后续 flush 时触发 `IntegrityError(papers.doi)` 毒化整个任务。因此 enrichment 写入前对 4 个唯一索引字段逐一查重——值已被其他行占用则丢弃该字段；写区段经 `write_lock` 串行化 + `no_autoflush` 防止并发悬置改动互相不可见；失败分支统一 rollback + expire 解毒。

### 14.5 SQLite 配置

WAL 模式支持并发读写（后台 Agent 写入 + 前端查询 + SSE 读取）：

```python
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
```

> 所有写入路径统一走 `app/db/lock_retry.py`：`retry_on_locked(_async)` / `flush_with_retry` / `commit_with_retry`（先 rollback 再重试）。退避 SYNC=(0.15..15.0)s / ASYNC=(0.2..20.0)s 带 0–25% 抖动，事件循环上必须用 `asyncio.sleep`。新增写入点**禁止打点式逐处添加**，一律走上述封装。

---

## 15. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/tasks` | 创建研究任务 |
| GET | `/api/tasks` | 任务列表 |
| GET | `/api/tasks/{id}` | 任务详情（含 `state_json`） |
| POST | `/api/tasks/{id}/start` | 启动 / 续跑 Agent（重置 LLM 预算） |
| POST | `/api/tasks/{id}/stop` | 手动停止 |
| POST | `/api/tasks/{id}/clarify` | 提交澄清回答 |
| POST | `/api/tasks/{id}/feedback` | 提交用户反馈（关键词/约束） |
| GET | `/api/tasks/{id}/papers` | 论文列表（分页、优先级筛选） |
| GET | `/api/tasks/{id}/rounds` | 检索轮次记录 |
| GET | `/api/tasks/{id}/report` | 研究报告 |
| GET | `/api/tasks/{id}/ideas` | Research Ideas（`include_superseded=true` 看历史代） |
| POST | `/api/tasks/{id}/ideas/select` | 用户选择感兴趣的 Ideas |
| POST | `/api/tasks/{id}/ideas/judge` | 对选中的 Ideas 做深度质量评估 |
| POST/GET | `/api/tasks/{id}/experiments` | 生成 / 获取实验方案 |
| GET | `/api/tasks/{id}/experiments/{plan_id}/export` | 导出实验方案（Markdown / JSON） |
| GET | `/api/tasks/{id}/traces` | Agent 执行轨迹 |
| GET | `/api/tasks/{id}/events` | SSE 实时进度推送 |
| GET | `/api/contracts/...` | Contract / Questions / Evidence / Coverage / Phase 只读 API |
| GET | `/api/gaps/...` | Gap + 审计 + 近邻对比只读 API |
| GET | `/api/interventions/...` | 干预方案只读 API |

> `POST/PUT/DELETE /api/tasks` 在设置 `API_KEY` 后需携带 `X-API-Key` 头；留空则禁用（本地开发）。

---

## 16. 前端

单页应用，`App.tsx` 加载任务列表并每 5 秒轮询当前任务；`TaskDetail.tsx` 提供 10 个 Tab：

```
┌──────────────────────────────────────────────────────────────┐
│  任务状态: 阶段 | 轮次 n/5 | 论文数 | 高优数                    │
├────────────┬─────────────────────────────────┬───────────────┤
│ Agent 轨迹  │  概览 报告 论文 知识库 覆盖度     │  用户反馈区    │
│ (EventLog) │  缺口 干预 创意 实验 轨迹         │  补充关键词     │
│            │                                 │  选择 Idea     │
│            │         Tab 内容区               │  继续检索       │
│            │                                 │  生成实验方案   │
│            │                                 │  导出报告       │
└────────────┴─────────────────────────────────┴───────────────┘
```

进度走 SSE（`useTaskEvents.ts`，含指数退避重连与终态停止），而非轮询。

---

## 17. 环境变量

```env
# ===== LLM Provider（OpenAI 兼容协议；provider 名沿用 venus，端点由下面几项决定）=====
LLM_PROVIDER=venus
ENV_VENUS_OPENAPI_SECRET_ID=            # 走本地端点时可留空
VENUS_LLM_PROXY_URL=http://<host>:8080/openapi
VENUS_LLM_MODEL=<primary-model>
# 备用模型：主模型 transport/服务错误时自动降级；model 留空则禁用降级
FALLBACK_VENUS_LLM_PROXY_URL=http://<host2>:8080/openapi
FALLBACK_VENUS_LLM_MODEL=<fallback-model>

# ===== 上下文预算（不得超过实际模型窗口，否则超长 prompt 只会得到笼统的 400）=====
LLM_CONTEXT_TOKENS=131072
LLM_MAX_OUTPUT_TOKENS=8192              # 准入时预留的输出下限，不是上限

# ===== OpenAI（备用 provider）=====
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# ===== 论文源 API Keys（全部可选，无 key 时走免费额度）=====
# Semantic Scholar：留空即无 key 模式（限速 20/min，易触发 429）；有 key 后 5000/min
SEMANTIC_SCHOLAR_API_KEY=
# OpenAlex / Crossref：填邮箱即进入 polite pool（无需审批，即时生效）
OPENALEX_EMAIL=your@email.com
OPENALEX_API_KEY=
CROSSREF_EMAIL=your@email.com
IEEE_API_KEY=                            # 需 IEEE 订购，可留空跳过
CORE_API_KEY=

# ===== Embedding 后端（可插拔，走 OpenAI 兼容 API 绕开 Windows PyTorch segfault）=====
EMBEDDING_BACKEND=api                    # api | local
EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
EMBEDDING_KEY=                           # 独立 embedding key，留空则回退到 openai/venus
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024                       # 必须与所选模型一致

# ===== Agent 参数 =====
MAX_ROUNDS=5
QUERIES_PER_ROUND=5
PAPERS_PER_SOURCE_PER_QUERY=15
HIGH_PRIORITY_TARGET=15
MAX_CONCURRENT_AGENTS=2
AGENT_TIMEOUT_SECONDS=3600
MAX_LLM_CALLS_PER_TASK=800               # 任务级 LLM 调用预算（审计/聚簇/抽取共享）
MAX_LLM_TOKENS_PER_TASK=0                # 0 = 不限

# ===== 检索循环终止 =====
STOP_AFTER_EMPTY_ROUNDS=2
MAX_DUPLICATE_RATE=0.75

# ===== Gap 审计 =====
GAP_AUDIT_TIMEOUT_SECONDS=600            # 单 Gap 全流程一个超时罩
GAP_AUDIT_MAX_QUERIES=12
GAP_AUDIT_MAX_CANDIDATE_PAPERS=20
AUDIT_NEIGHBOR_EVIDENCE_MAX_PAPERS=5
GAP_AUDIT_BUDGETED=false                 # 开启 Budgeted Falsification (v16)
GAP_AUDIT_PROGRESSIVE=false              # 开启渐进式对抗审计 (v15)

# ===== 证据抽取 =====
# 抽取覆盖全部候选论文（high/medium），没有每轮篇数上限；成本由并发与预热控制
EVIDENCE_EXTRACT_CONCURRENCY=4           # 同时在抽取的论文数（单一共享池）
EVIDENCE_MAX_CHUNKS_PER_PAPER=6          # 按 section 轮转分配
EVIDENCE_CHUNK_CONCURRENCY=4             # 单篇内并发 chunk 数
EVIDENCE_PAPER_TIMEOUT_S=300             # 单篇墙钟超时，防一篇卡死整轮
EVIDENCE_MIN_PAPERS_PER_QUESTION=3
ENABLE_RAG_INDEXING=true                 # 关闭后仍会抽取证据

# ===== PDF 预热 =====
ENABLE_PDF_PREWARM=true                  # 抽取前先取回本轮全部 PDF
PDF_PREWARM_CONCURRENCY=3                # 刻意低于抽取并发，避免触发 arXiv 限流
PDF_PREWARM_TIMEOUT_PADDING_S=60         # 在重试预算之上额外留的余量

# ===== 链路优化 =====
MAX_REMEDIATION_ATTEMPTS=2               # 每个失败原因的定向补检索上限
MAX_REMEDIATION_ROUNDS_TOTAL=3           # O2 全局定向补检索轮数上限
MAX_NARROWING_PASSES_TOTAL=3             # 收窄总次数上限
GAP_MINING_PRECEDENT_CHECK_ENABLED=true  # v7.2: scaling_regime claim 级先例检索
GAP_MINING_PRECEDENT_CHECK_TOP_K=5       # 先例检索每候选送判论文数上限
GAP_STARVATION_DEFERRAL_LIMIT=3          # deferred 达此次数转 rejected；0 关闭
SEARCH_PREFILTER_MIN_SIMILARITY=0.45     # 入库前主题相似度过滤阈值（0 关闭）
SCORE_CALIBRATION_MIN_BATCH=5
SCORE_CALIBRATION_STRENGTH=0.15

# ===== 数据库 =====
DATABASE_URL=sqlite:///./deep_research.db

# ===== 服务 =====
HOST=0.0.0.0
PORT=8000

# ===== 认证（留空则禁用；生产环境务必设置）=====
API_KEY=
```

> 改动 `.env` 后必须重启 uvicorn（配置在进程启动时读入）。数据库结构变更后执行 `python scripts/bootstrap_db.py` 或 `python -m alembic upgrade head`。

---

## 18. 评测体系

评测代码在 `backend/eval/`，产出目录 `backend/eval_results/`。跨版本比较必须使用 real-response 口径。

**产物契约**：`attempts.jsonl` 记录每次尝试（含成本），`predictions.jsonl` 只记录唯一成功结果；resume 时跳过最新 attempt 已 ok 的样本。judge 冻结为 `eval-judge-v1`。

**Benchmark 适配**：`researchbench/adapter.py`（Retrieval / Generation / Ranking 三类任务）、`rinobench/adapter.py`（Novelty 二分类）。

**已冻结结论**：

| 版本 | 结论 |
|------|------|
| V1 baseline | RB Retrieval r1 0.7173 → r2 0.3122；Generation 0.40；Ranking 0.14（discrimination failure）；RINo Macro-F1 16.3% |
| **V2 criterion-first ranking** | 0.14 → **0.76/0.81** official（clean 0.711 real）。成功结构原因=强制二选一。**已冻结采用** |
| V3 criterion-first novelty | RINo 16.3% → 18.3/19.9%，calibration tradeoff（LowNovelty FP 降但 HighNovelty FN 翻倍）；根因 COVERAGE_COLLAPSE 在 criterion 提取层。存档 |
| V4a forced binary coverage | 干净负结果（CoverageDiscrimination 零分离），失败在 closest work 语义对齐子层。存档 |

**版本定格**：Ranking = V2；Novelty = V1。

### production_e2e_v1：三系统对照实验（当前主线）

`backend/eval/production_e2e/` 实现 24 主题的三系统对照（`direct_llm` / `retrieval_llm` / `full_v2`，均通过 HTTP 驱动后端），双 headline 指标是 **false-open-gap 率 ↓** 与 **Credible Idea Yield ↑**，核心主张是"拒绝更多假 Idea"。

```
topics.jsonl                # 24 主题冻结（12 narrow_mature / 6 emerging_sparse / 6 broad_ambiguous）
schema.py                   # 统一 Idea 输出契约（FinalIdea + E2EDecision）
backfill_pdf_chunks.py      # 回填 PDF chunk 语料
run_full_v2.py              # full_v2 系统驱动
baseline_direct.py          # direct_llm 基线
baseline_retrieval.py       # retrieval_llm 基线
super_audit.py              # 独立审计模板：为每个 submission 生成候选 killer 论文
rerun_infra_topics.py       # 按 final_status 重跑基础设施失败主题
make_review_sample.py       # 生成人工盲评样本
evaluate.py                 # 汇总指标（false-open-gap / Credible Idea Yield）
enrich_audit_snippets.py    # 审计 snippet 事后富化
opportunity_replay.py       # v7 冻结证据池 replay（v6 vs v7 机会发现对照）
prune_replay_tasks.py       # 清理 replay 写入的临时任务
probe_replay_deps.py        # replay 依赖自检（只读）
v15_replay_protocol.md      # v14→v15 渐进式审计 paired evaluation 预注册协议
```

**v1 结果（24 主题，人工盲评已全部完成）**：

| System | False-open-gap ↓ | Credible Idea Yield ↑ | Novelty ↑ | Feasibility ↑ | Abstention | Cost |
|---|---:|---:|---:|---:|---:|---|
| Direct LLM | 25.0% | 45.8% | 1.75 | 4.38 | 0.0% | 24 calls/29k tok |
| Retrieval + LLM | 60.9% | 34.8% | 1.44 | 4.70 | 0.0% | 23 calls/111k tok |
| Full V2 | 18.2% | 81.8% | 2.27 | 4.59 | 4.3% | 23.5h wall; 5138 papers |

读法（三条限制，缺一条就会过度解读）：

1. **这是系统级对比，不是因果消融**。V2 同时拥有全文 chunk、Evidence Unit、多轮检索、best-of-N 候选，赢了只证明 `Full V2 bundle > simple Retrieval+LLM`，不证明 `Gap Audit alone causes it`。机制归因需要 V2-no-audit 之类的消融，尚未做。
2. **Retrieval+LLM 的 false-open 率反而高于 Direct LLM**（60.9% vs 25.0%）。这不是噪声：给了文献之后，模型更倾向于把"读到的相关工作"包装成 gap，而 Direct LLM 因为不知道 prior art 反而更常给出宽泛但不易被证伪的想法。这条正是"检索本身不产生可信 Idea"的直接证据。
3. **成本口径不可合并**：baseline 是单次调用（calls/tokens），Full V2 是全链路（wall-clock + 论文数），分列呈现，不合成单一数字。

### v7 机会本体论 replay pilot（冻结证据池）

`opportunity_replay.py` 回答一个被冻结的因果问题：**给定完全相同的知识输入**（同 topic / contract / research questions / 检索到的论文 / evidence units / question-evidence links），v7 的 7 类 opportunity 本体论是否比 v6 发现更多真实机会，且不显著增加假 gap。不重检索、不重抽取、不重分解——冻结池是 v6 任务的**终态**证据（比 v6 任何单轮 mining 看到的都多），因此对两个 miner 同样宽松。

6 主题 pilot 结果（`eval_results/v7_replay_pilot/replay_report.json`）：

| 问题 | 结果 |
|------|------|
| q1 v7-only 召回 | 35 个候选，**30 个 v6 无对应物**（仅证明搜索空间变宽，本身不证明更好） |
| q2 边际机会精度 | **6.7%（2/30 NONE），PARTIAL 21，FULL 7（23.3%）——LLM 评审两轮（第二轮联网复核 12 行），非独立人工盲评**（见下文来源说明） |
| q3 边际可信 Idea 产出 | **NOT MET**：2 个 true gap（v7r-015 / v7r-023，均为 `target_only`）没有关联 Idea。第一轮靠 v7r-005（gap `ea4b38e0`，4 个 executable_candidate）达成，复核后改判 PARTIAL |
| q4 分类型漏斗 | 7 类全部有产出；`scaling_regime` 新增最多（7），`boundary_gap`/`mechanism_gap`/`cross_domain_transfer` 各有 4-5 个但审计未覆盖到 |
| q5 边际成本 | mining 42 次 LLM 调用 / 1068s；审计 1819s |

跨类型聚簇 30 → 29 簇（压缩比 1.03，cosine ≥0.78 union-find），说明新增量不是同一发现的措辞变体。

**预注册的 GO/NO-GO 门槛**：≥1 个 v7-only 机会被人工判为真 **且** 能产出可信 Idea **且** 新增机会不是以假 gap 为主。6 主题只决定 GO/NO-GO，不做 per-type 删除。

**裁决（2026-10-29，第二轮，`eval_results/v7_replay_pilot/adjudication_summary.md`）**：第一轮（无检索 LLM 预填）得出名义 GO（4 true / 18 partial / 8 FULL）。随后对 12 行（4 NONE + 8 FULL）做了联网复核，3 行改判：v7r-005 NONE→PARTIAL（arXiv 2607.28308 *Coherent Overlap in Sparse MoE Routing* 已用干预检验后续专家的贡献）、v7r-019 NONE→PARTIAL（2610.05076 TTT 失败的因果分解）、v7r-021 FULL→PARTIAL。复核后 **2 true / 21 partial / 7 FULL**，[1][2] MET、**[3] NOT MET → NO-GO**。敏感性：即使排除晚于预注册日的 2610.05076，[3] 仍不成立（v7r-019 没有关联 Idea）。偏差：两轮都是同一 LLM 评审，非独立人工、非盲。观察：mechanism_gap 的 Idea 产出集中在 PARTIAL gap（v7r-003/012/005），v7 的价值可能在"收窄残余"上，但需要另行预注册，不能在本 pilot 里事后改口径。

盲评工具链（可复现）：

```bash
cd backend
python -m eval.production_e2e.v7_blind_review sheet --pilot-dir eval_results/v7_replay_pilot  # 生成/重建盲评表
# 人工填 human_verdicts.jsonl（review_id → FULL / PARTIAL / NONE）
python -m eval.production_e2e.v7_blind_review score --pilot-dir eval_results/v7_replay_pilot  # 聚合 q2/q4 + 三条件（含 idea 关联）
```

**判定来源（2026-10-29）**：`human_verdicts.jsonl` 的判定由 **LLM 代判**，经用户批准合并。第一轮仅凭模型自身文献知识、未联网检索，原始预判另存于 `human_verdicts.llm_prefill.jsonl`。第一轮计分：NONE 4 / PARTIAL 18 / FULL 8 → 名义 GO（备份见 `human_verdicts.pass1_backup.jsonl`）。**第二轮（同日）**对 4 NONE + 8 FULL 联网复核，notes 前缀改为 `[LLM pass2 web-verified ...; same LLM reviewer, NOT independent human]`，每行写明核对到的 arXiv 编号；计分变为 NONE 2 / PARTIAL 21 / FULL 7 → **NO-GO**（[3] NOT MET；idea 关联由 `v7_blind_review score` 实时查库计算）。解读限制：① 剩下的 2 个 NONE（v7r-015、v7r-023）都是无机制的纯性能目标，后者近乎不可行；② v7r-008、v7r-028 判定前接触过 `mech_override_review.llm_verdicts.jsonl` 的同主题条目（标 `possible_contamination`，均为 PARTIAL，不影响门槛）；③ `scaling_regime` 7 条中 4 条 FULL，是 v7 新增最多、也最容易假开放的类型；④ 18 个未复核的 PARTIAL 若检索，可能有部分升为 FULL，但不影响 [3]；⑤ 第二轮非盲，且复核方向对 GO 不利。此结果**不等同于预注册要求的独立人工盲评**。

**明确不做的事**：不与 v6 的 budgeted-production 终态做 headline 对比（`v6_totals_context` 只作上下文）。full-jury 漏斗 vs 固定预算漏斗不是同口径，混比会得出"v7 产出更多"这种无意义的结论。v7 的生产价值需要单独的固定预算实验。

**统计口径铁律**（`.codebuddy/memory` 有完整记录，此处摘录最易踩的几条）：

- **未判定 ≠ 负面结果**：`human_verdicts.jsonl` 每 submission 一行（`verdict: ""`＝未判定）。`verdict_map.get(...)` 对每个主题都查得到，空 verdict 会被当成"非 false_open"与"不可信"。**只有 `verdict` 非空才进分母**。
- **Credible Idea Yield 分母 = 已判定主题数**，不是全部主题——否则判得越少分越低。危害实例：9/24 判定时表出 false-open 4.2%（分母 24）；按已判定口径应为 33.3%。这个差别会直接改变"V2 是否有优势"的决策。
- **必须打印 `review_coverage`**（topics_total / judged / complete）与表头 `Human review: N/M judged <-- INCOMPLETE`，让部分评审一眼可见；全未判定必须显示 `pending`（`rate: None`）而非 0%。
- **批量运行必须 `--concurrency 1`**：并发 3 会触发 168 次 `database is locked` 并导致事件循环僵死。
- **driver 超时 ≠ 任务失败**（查 `ResearchTask.status`）；**`parse_status=ok` ≠ pipeline 成功**（按 `final_status` 判定）。
- **统计必须按 `decision` 字段，不按 `final_status`**（终态映射会把有 idea 的任务标为 `abstained`）。
- **verdict 字段必须对齐 `evaluate.py`**：`verdict`（"false_open" | "true_gap" | ""）、`idea_credible`（bool）、novelty/feasibility（1-5）。
- **盲协议泄漏**有两个路径都要堵：source 回退当 venue、venue 本身存源名（`_SOURCE_NAMES` 白名单）。盲评表片头由单一函数 `blind_sheet_header()` 生成。
- **召回优先于精度**：候选论文相关性是一处未解限制（39% 候选与 claim 零词面重合），但词面重合作过滤判据两版实测均被否决（LLM 领域泛词是区分性词）；误删 killer 会让假 gap 不可见。处置方式是把这个噪声率连同指引内联进判定材料，而不是加过滤器。

---

## 19. 开发历程与当前状态

### 已实现的七个阶段

**P0 工程化基础**：拆分 `runner.py`（1573 行 → 编排逻辑 + 独立 steps 模块）；任务恢复（进程重启扫描中间态任务标记 `failed`）；SSE 队列限界（`maxsize=200`，终态后 10 秒清理）；API Key 认证。

**Pipeline V2 重构**：从"检索后让 LLM 编 Idea"转为证据驱动，落 Research Contract / Evidence Unit / Coverage Matrix / Gap Mining + 对抗审计 / Intervention + Minimal Experiment / Landscape Brief 六个结构化层。

**链路优化（O 系列）**：O2 定向补检索回环（闸门失败按原因补检索而非终止，双重预算防死循环）；O1 分级产出 A/B/C；O5a 可插拔 Embedding + 全文 RAG；O7 检索预过滤；评分跨论文校准；前端 V2 适配。

**运行时稳定性与证据供给**：上下文预算体系（准入/自校准/额度自愈/失败分类）；Gap 挖掘注入有界；失败粒度对齐；审计不再空转（只重审被收窄的 Gap + 不可判定终结）；chunk 预算按 section 轮转；关系语义统一到 `evidence_relations.py`。

**Gap 审计收敛与 Idea 生成门禁**：审计超时结构（两处 commit 检查点）；查询家族三级兜底；去重前移（0.85→0.78 + prompt 注入已有 claimed_delta）；surviving 语义修正为 contract 级终态；enrichment 唯一索引守卫；实验计划反馈重试；hypothesis cluster 门禁；idea 级 novelty 快检；idea 元数据差异化；实验计划一致性门禁；LLM 供应商熔断。

**证据供给漏斗修复（2026-09）**：抽取改为无条件覆盖全部候选论文（删掉每轮篇数上限与优先级配额——146 个候选配 15 篇上限时，五轮任务最多只能评估 75 篇，语料在构造上就不可达）；单一共享并发池取代批间屏障（旧代码每批新建 `Semaphore(batch_size)` 并 await 整批，耗时按 `ceil(N/batch)×单篇` 增长）；PDF 预热把"能否拿到语料"与"语料说了什么"解耦；PDF 下载按原因分类重试（transient 重试 / paywall 不重试）；`task_papers.extraction_status` 区分"测过为负"与"没读到"；兜底启发式不再把 `comparison` 降级为 background。

**机会发现广度（v7 ontology）**：7 类 opportunity 各有独立发现指令与准入契约，按 lens fan-out 逐类生成；矛盾不再阻塞挖掘，反而成为 `contradiction` 类的原料；证据选择按 evidence_type 轮转交错，避免 abundant 的 limitation 证据挤掉 method/result 证据。配套 `opportunity_replay.py` 冻结证据池 replay 基建，使"本体论改动是否真的发现更多真机会"可被单独测量。

### 已知瓶颈（下一步方向）

1. **limitation 证据供给已实测为健康，历史拒因属 v6 时代**：`NO_LIMITATION_SIGNAL`（395 次）与 `UNRESOLVED_VERIFIED_CONTRADICTION`（382 次）只在 pre-v7 policy 下存在（v7 起限制性要求改为 per-type 准入契约），而库中 `gap_candidates` 绝大多数仍是 v3–v6 产物（v7.1 仅 35 条）。对账版审计（`python scripts/extraction_audit.py`，2026-10-29，`backend/eval_results/extraction_audit_2026-10-29.json`）给出全史口径：证据单元 82.5% 来自 `pdf_fulltext`（24055/29143，覆盖 1792 篇论文），**123/125 个挖过缺口的任务都有 conclusion-section 证据**——抽取层的语料可达性不是瓶颈。两个度量口径的坑已修入脚本：① `task_papers.extraction_status` 只覆盖 0034 迁移之后的任务（历史任务全 NULL，不可作漏斗）；② `paper_chunks.section` 是 RAG 管线的分类器（默认 `unknown`），与 evidence 的 section（抽取管线，默认 `introduction`）是两套体系——曾据 RAG 视角误判"仅 25.5% full-text 论文有 conclusion chunk"，实际 evidence 视角覆盖率 100%。v7 下 limitation 类机会的真实供给情况待 replay 盲评（q2/q3）裁决。
2. **耗时统计**：阶段级耗时已由 `phase_service.execute_phase` 统一记录（`phase_duration` trace + `phase_runs`）；2026-10-29 起 O2 定向补检索内的抽取/覆盖度也走同一包装。**2026-10-30 起阶段内部也细分了**（`app/services/timing.py` 的 `StageTimings`，写入各 step trace 的 `stage_timings`）：抽取到每篇论文（`slowest_papers_s` + prewarm/download 阶段）、mining 到每个 lens 的 LLM 调用与去重嵌入、audit 到每个 gap、experiments 到 cluster/triage/打分/novelty/多样性闸门。旧的"只有阶段级"黑盒已消除。
3. **外部检索源在部分出口 IP 上被整体限流**（OpenAlex 对匿名请求、UA 带 mailto、mailto 查询参数三种方式一律 429），审计的近邻只能靠已入库论文补足。
4. **候选论文相关性**：39% 候选与 claim 零词面重合，属未解限制（详见第 18 节）。
5. **v7 pilot 复核后 NO-GO，准入修复已落地（v7.2，2026-10-30）**：第二轮联网复核后为 2 true / 21 partial / 7 FULL，[3] NOT MET。判定由同一 LLM 评审完成，非独立人工。按预注册规则不加机制、先修准入，两项均已实现并有回归测试（`tests/test_v72_admission_fixes.py`）：① `scaling_regime` 准入附带 claim 级先例检索（`SCALING_RANGE_ALREADY_MEASURED`，4/7 FULL 的根因修复，infra 故障 fail-open）；② `boundary_gap` 拒绝纯性能目标型 claimed_delta（`PERFORMANCE_TARGET_DELTA`，正则只掐性能动词开头且豁免含边界语义的 delta）。`GAP_MINING_POLICY_VERSION` 已 bump 至 `opportunity-ontology-v7.2`。如需推翻 NO-GO，唯一合规路径是独立人工复核 v7r-005；"PARTIAL 残余"价值（mechanism_gap 的 Idea 都挂在 PARTIAL gap 上）需另行预注册。v7.2 的效果验证属于固定预算实验（瓶颈 #7），不在 full-jury replay 口径下做。完整存档：`eval_results/v7_replay_pilot/adjudication_summary.md`。
6. **机制归因缺失**：production_e2e_v1 只证明 `Full V2 bundle > simple baselines`，没有 V2-no-audit 消融，因此"审计到底贡献了多少"仍未知。按预注册规则，这属于下一轮实验而非当前结论。
7. **v7 生产价值未测**：full-jury 漏斗与固定预算漏斗不同口径，v7 在真实预算约束下的产出价值需要单独实验。

### 2026-10-30 第四轮修复（探针 B 驱动）

| 项 | 改动 | 验收 |
|---|---|---|
| **deferred 候选饥饿** | `_cheap_gap_rank_score` 增加 deferred 轮数加权（+0.03/轮，上限 +0.09）；`gap_starvation_deferral_limit`（默认 3）到期转 `rejected` + `gap_starvation_retire` trace | `tests/test_gap_competition.py` 4 项；下次探针 candidate 占比应 <50% |
| **阶段内部计时** | 新增 `app/services/timing.py::StageTimings`；接入 extraction（每篇）、mining（每 lens + dedup + 先例检索）、audit（每 gap）、experiments（cluster/triage/scoring/novelty/diversity） | `tests/test_stage_timings.py` 6 项 |
| **killer 检索命中率** | 新增 `scripts/killer_search_audit.py`（只读）：聚合 uncertain 路径的执行数/命中数/跳过原因/降级数 | 探针 C 实测 2 次执行、**0 命中**；需再跑 2-3 个任务，持续 <10% 则降级为可选开关 |
| **winner 选择尊重审计 novelty** | `_resolve_gap_competition_winner` 排序键改为 `(clears_novelty_gate, cheap_rank, novelty_confidence)`；trace 增 `novelty_gate_threshold` + 每候选 `clears_novelty_gate` | `tests/test_gap_competition.py` +2（0.9 vs 0.42 场景 / 不越权）；根因见探针 C |
| **Idea 双轴标注** | `idea_quality_gate` trace 增加 `final_score` / `score_go_line` / `axes`；分数与门禁不一致时打 `GATE_PASS_SCORE_BELOW_LINE` / `GATE_BLOCK_SCORE_ABOVE_LINE` | 消除"`executable_candidate` 却 0.697 分"的读法歧义 |

### 当前非目标

1. 自动运行真实实验（只生成实验方案文档）
2. 多用户权限系统 / PostgreSQL 迁移（SQLite WAL 足够单用户场景）
3. Google Scholar 抓取（无官方稳定 API）
4. 引用网络的递归深度扩展（当前只做一跳引用关系，citation snowball 实测不可行已默认关闭）
5. LangGraph / AutoGen 等重型 Agent 框架

---

## 20. 相关文档

| 文档 | 性质 |
|------|------|
| [docs/INDEX.md](docs/INDEX.md) | 文档索引与整理处置记录 |
| [docs/refactor/01_target_architecture.md](docs/refactor/01_target_architecture.md) | V2 目标架构蓝图（目标态，实际实现以 README / 代码为准） |
| [docs/10-rag-fulltext-pipeline.md](docs/10-rag-fulltext-pipeline.md) | RAG 全文检索管线设计 + ChromaDB 选型决策 |
| [docs/11-llm-wiki.md](docs/11-llm-wiki.md) | LLM Wiki 知识编译（放弃 GraphRAG 的决策对比） |
| [docs/12-literature-map.md](docs/12-literature-map.md) | 文献地图设计（roadmap，未实现） |
| [docs/15_从研究方向到Idea的难点分析与优化建议.md](docs/15_从研究方向到Idea的难点分析与优化建议.md) | V2 长链路 7 道硬闸门的系统性诊断 |
| [docs/eval_report_v2_v3_2026-09-01.md](docs/eval_report_v2_v3_2026-09-01.md) | V2/V3 Novelty 评测汇报 |
| [backend/eval/production_e2e/README.md](backend/eval/production_e2e/README.md) | 三系统对照实验设计（v1 冻结：三系统定义、双 headline、盲评协议、分层、裁决规则） |
| [backend/eval/production_e2e/v15_replay_protocol.md](backend/eval/production_e2e/v15_replay_protocol.md) | v14→v15 渐进式审计 paired evaluation 预注册协议（指标注册表 + 安全指标 + 决策规则） |
