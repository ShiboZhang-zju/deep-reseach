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
| 机会发现广度（v7 ontology） | 不再只认"作者自认的 limitation"，7 类 opportunity 各有准入契约 | ✅ 实现完成，pilot 已跑 |
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
- **q2（边际机会精度）与 q3（边际可信 Idea 产出）仍是 PENDING**——需要人工盲评 `v7_only_opportunities.jsonl` 的 prior-art 覆盖情况。pilot 的 GO/NO-GO 门槛（≥1 个 v7-only 机会被人工判为真、且能产出可信 Idea、且新增机会不是以假 gap 为主）**尚未裁决**。

**工程状态**：`634` 个测试全部收集通过；Alembic 迁移到 `0034`；证据抽取已改为无条件覆盖全部候选论文（不再有每轮篇数上限），靠共享并发池 + PDF 预热把成本压回来。

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
| `boundary_gap` | 方法在特定条件失效 | 需引用 limitation / negative_result 证据 |
| `missing_evaluation` | 某关键维度没评测 | 需引用 limitation / negative_result / future_work 证据 |
| `contradiction` | 两组可信结果冲突 | 需引用冲突双方 ≥2 条证据（"Verified conflict" 单元进 prompt 池） |
| `untested_assumption` | 多篇论文共享未经验证的假设 | prevalence 门槛：引用须来自 ≥3 篇独立论文（`ASSUMPTION_NEEDS_BROADER_PREVALENCE`） |
| `mechanism_gap` | 现象存在但原因不明 | 需引用 result / negative_result / comparison / limitation（现象必须被记录） |
| `scaling_regime` | 某一尺度区间没人研究 | 同上另加 dataset（已测范围必须被记录，未测区间须点名） |
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
| Budgeted Falsification | `gap-search-budgeted-v16` | 每个昂贵维度硬上限：4 query × 3 源、Top-10 候选、Top-3 近邻、≤2 篇全文（可疑 killer 优先）、killer 检索 ≤3 query、Top-1 gap competition |
| 渐进式对抗审计 | `gap-search-admission-v15` | 增量重审，只审有新输入的 Gap |
| 传统模式 | `gap-search-admission-v14` | 保留 v14 参数不变 |

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
│   │   └── bootstrap_db.py        # 建库/升级（处理全新库与遗留库，含备份+校验）
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
| q2 边际机会精度 | **PENDING**（需人工盲评 prior-art 覆盖） |
| q3 边际可信 Idea 产出 | **PENDING** |
| q4 分类型漏斗 | 7 类全部有产出；`scaling_regime` 新增最多（7），`boundary_gap`/`mechanism_gap`/`cross_domain_transfer` 各有 4-5 个但审计未覆盖到 |
| q5 边际成本 | mining 42 次 LLM 调用 / 1068s；审计 1819s |

跨类型聚簇 30 → 29 簇（压缩比 1.03，cosine ≥0.78 union-find），说明新增量不是同一发现的措辞变体。

**预注册的 GO/NO-GO 门槛**：≥1 个 v7-only 机会被人工判为真 **且** 能产出可信 Idea **且** 新增机会不是以假 gap 为主。6 主题只决定 GO/NO-GO，不做 per-type 删除。**该门槛尚未裁决**——q2/q3 的人工列还空着。

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

1. **limitation 证据仍不足**：`NO_LIMITATION_SIGNAL` 是 Gap 准入的首要拒因。根因不在提示词，而在多数论文没解析出 conclusion 段（PDF 下载/解析失败后走 abstract 兜底）。PDF 预热与分类重试已缓解"拿不到 PDF"，但**section 识别率本身尚未测量**——`extraction_status` 现在能给出 `abstract_only` 占比，这是下一步的入口。
2. **耗时统计有盲区**：`agent_traces.duration_ms` 恒为 0，且定向补检索内的抽取不建 `phase_runs` 记录。
3. **外部检索源在部分出口 IP 上被整体限流**（OpenAlex 对匿名请求、UA 带 mailto、mailto 查询参数三种方式一律 429），审计的近邻只能靠已入库论文补足。
4. **候选论文相关性**：39% 候选与 claim 零词面重合，属未解限制（详见第 18 节）。
5. **v7 pilot 的 q2/q3 未裁决**：`v7_only_opportunities.jsonl` 的人工 prior-art 盲评还没做，GO/NO-GO 门槛悬空。这是当前唯一阻塞"v7 是否值得保留"的待办。
6. **机制归因缺失**：production_e2e_v1 只证明 `Full V2 bundle > simple baselines`，没有 V2-no-audit 消融，因此"审计到底贡献了多少"仍未知。按预注册规则，这属于下一轮实验而非当前结论。
7. **v7 生产价值未测**：full-jury 漏斗与固定预算漏斗不同口径，v7 在真实预算约束下的产出价值需要单独实验。

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
