# Refactor Progress

## Phase 0 — ✅ 已完成

- [x] 创建 docs/refactor/ 文档 (00-03)
- [x] 审计全部核心文件
- [x] 修复 db.refresh = None (runner.py)
- [x] 修复澄清答案丢失 — 不再截断 user_input
- [x] 修复 Idea retry 条件 — 移除 active_count > 0 的错误 break
- [x] 移除 auto-promote — 替换为 _finish_with_insufficient_evidence
- [x] 增加 round retry 上限 (max_attempts=3, total_failed_budget=5, identical_error_detection)
- [x] 统一 policy 终止逻辑 — early_termination_check 移至 policy.py
- [x] 修复 generate_experiment conditional_go — 不再视为 good_idea
- [x] 新增 PhaseRun 模型 (models.py)
- [x] 新增 PhaseRun schema (schemas.py)
- [x] 新增任务状态枚举 (insufficient_evidence, more_research_required, auditing_gaps 等)
- [x] 新增 Alembic migration 0002_phase_runs.py
- [x] 前端 STATUS_LABELS 和 STATUS_COLORS 更新
- [x] 修改 test_fixes.py — 适配 auto_promote 废弃
- [x] 新增 test_phase0_fixes.py — 9 个测试

### Phase 1.5 修复（已完成）
- [x] Contract 对 ResearchState 的修改持久化 (state.contract_id)
- [x] Research Questions 驱动 query 生成
- [x] Contract/Questions 版本和失效机制 (version, superseded_at, input_hash)
- [x] PhaseRun 接入 runner
- [x] 新阶段状态进入恢复逻辑 (_INTERRUPTED_STATUSES)
- [x] 搜索失败后 runner 不继续报告和 Idea 阶段 (SearchLoopResult)

## Phase 1 — ✅ 已完成

- [x] ResearchContract 模型 + schema
- [x] ResearchQuestion 模型 + schema
- [x] Alembic migration 0003_contract_questions + 0004_contract_versioning
- [x] build_contract step
- [x] decompose_research_space step
- [x] runner.py 集成 — clarify → build_contract → decompose → search_loop
- [x] API route: GET /tasks/{id}/contract, GET /tasks/{id}/questions
- [x] 新增 test_phase1.py + test_control_plane.py

## Phase 2 — ✅ 已完成

- [x] EvidenceUnit 模型 + schema
- [x] SearchQueryRecord + SearchQueryPaper 模型
- [x] PaperRole 模型 + schema
- [x] CoverageRecord + QuestionEvidenceLink 模型
- [x] Alembic migrations 0005-0008
- [x] extract_evidence_units step (每轮执行)
- [x] update_coverage_matrix step (每轮执行)
- [x] coverage-driven query 生成 (generate_queries 重构)
- [x] coverage-driven stop (policy.py)

## Phase 2.2A — ✅ 已完成 (Final Closure)

- [x] RoundSearchResult dataclass + to_phase_payload/from_phase_payload
- [x] PhaseRun.output_json — 完整不截断 JSON 输出 (migration 0009)
- [x] phase_repo.get_completed_phase_output() — 精确恢复
- [x] _recover_round_search_result 从 output_json 精确恢复
- [x] retry 不污染 no_new_high_priority_count
- [x] readiness_gate.py — Phase2ReadinessResult + evaluate_phase2_readiness()
- [x] Clarify PhaseRun 接入
- [x] Bootstrap manifest 校验 (关键列检查)
- [x] 9 个新严格集成测试 (test_phase2a_final_closure.py)
- [x] CI backend ✅ + frontend ✅ (run #9, commit a9cf6d9)

**最终结果：PHASE_2_2A_PASS**

## Phase 3A — Gap 控制面骨架 + 闭包（✅ 已完成）

**目标**：建立 Gap 数据模型、migration、ResearchState 扩展、PhaseRun 契约和 Query 绑定。
不实现 Gap Mining 和 Gap Audit 的业务逻辑——仅搭建控制面骨架并关闭契约。

### 3A.1 数据模型（✅）
- [x] GapCandidate + GapEvidenceLink + GapAudit + NeighborComparison 模型
- [x] GapCandidate 结构化可证伪字段: target_setting, observed_problem, existing_coverage,
      missing_capability, claimed_delta, testable_hypothesis, falsification_condition,
      provenance_status
- [x] GapAudit 决策字段: evidence_for/against_gap_json, remaining_delta, novelty/audit_confidence,
      recommended_action (continue/narrow/more_search/reject), rejection_reason
- [x] NeighborComparison 结构化字段: shared_problem/mechanism/evaluation,
      covered/uncovered_claims_json, overlap_ratio
- [x] Status enum 冻结: candidate/auditing/audited/surviving/rejected/superseded

### 3A.2 Pydantic schemas（✅）
- [x] GapCandidateSchema 要求: falsification_condition, testable_hypothesis, claimed_delta,
      missing_capability, existing_coverage, observed_problem, target_setting,
      ≥1 question_id, ≥1 supporting_evidence_id
- [x] GapCandidateOut, GapAuditOut, NeighborComparisonOut, GapEvidenceLinkOut

### 3A.3 Alembic migrations（✅）
- [x] 0010_gap_tables (已发布): 创建 4 个 Gap 表 + search_query_records.target_gap_id 列
- [x] 0011_gap_control_plane_fix: FK 修复 + 结构化字段 + 两个 partial unique indexes

### 3A.4 ResearchState 扩展（✅）
- [x] active_gap_ids, surviving_gap_ids

### 3A.5 PhaseRun 契约（✅）
- [x] _INTERRUPTED_STATUSES 已包含 mining_gaps, auditing_gaps 等
- [x] 前端 STATUS_LABELS/COLORS 包含 mining_gaps

### 3A.6 Query 绑定（✅）
- [x] SearchQueryRecord.target_gap_id (FK to gap_candidates)
- [x] SearchQueryExecution.__post_init__: question_id 和 gap_id 不能同时为空
- [x] save_search_query 幂等包含 target_gap_id
- [x] 两个 partial unique indexes (discovery + gap)
- [x] get_queries_for_gap()

### 3A.7 Evidence 单一真相源（✅）
- [x] gap_evidence_links 是 Gap↔Evidence 的唯一业务真相
- [x] supporting/contradicting_evidence_ids_json 标记为 deprecated snapshot
- [x] API 从 gap_evidence_links 返回 Evidence

### 3A.8 gap_repo（✅）
- [x] create/get/list/supersede gap candidates
- [x] create/list/replace gap evidence links (跨 task 校验)
- [x] create/list gap audits (跨 task 校验)
- [x] create/list neighbor comparisons (跨 task 校验)
- [x] CrossTaskValidationError 阻断跨 task 关联

### 3A.9 只读 Gap API（✅）
- [x] GET /tasks/{task_id}/gaps
- [x] GET /gaps/{gap_id}
- [x] GET /gaps/{gap_id}/evidence
- [x] GET /gaps/{gap_id}/audits
- [x] GET /gaps/{gap_id}/neighbors
- [x] 默认排除 superseded gaps

### 3A.10 测试（✅）
- [x] 29 个 Phase 3A 测试 (12 基础 + 17 闭包)
- [x] FK 存在、migration roundtrip、ORM-Alembic 一致性
- [x] Query 不跨 Gap 串线、幂等
- [x] 跨 task 拒绝 (Contract/Evidence/Audit/NeighborComparison)
- [x] GapEvidenceLink 是 API Evidence 来源
- [x] Schema 缺 falsification_condition 失败
- [x] 5 个只读 API 测试
- [x] superseded gap 默认不出现
- [x] 153 tests total pass (136 old + 17 new)
- [x] PHASE3A_FK_OK + PHASE3A_ROUNDTRIP_MIGRATION_OK

## Phase 3B — Gap Mining（待实施）

- [ ] mine_gaps step — 从 Coverage Matrix 挖掘 Gap
- [ ] Gap API routes
- [ ] Gap Mining 测试

## Phase 3C — Gap Audit（待实施）

- [ ] generate_adversarial_queries
- [ ] audit_gaps step — 近邻审计
- [ ] Gap Audit 测试

## Phase 4 — Feasibility Gate + Idea Synthesis（禁止进入，直到 Phase 3 完整通过）

- [ ] GapGateResult 模型
- [ ] IdeaJudgment 模型
- [ ] run_feasibility_gates step
- [ ] synthesize_ideas step (重写)
- [ ] judge_ideas step (独立)
