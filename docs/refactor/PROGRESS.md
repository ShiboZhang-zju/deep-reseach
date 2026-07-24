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

## Phase 3A — Gap 控制面骨架（进行中）

**目标**：建立 Gap 相关数据模型、migration、ResearchState 扩展、PhaseRun 契约和 Query 绑定。
不实现 Gap Mining 和 Gap Audit 的业务逻辑——仅搭建控制面骨架。

- [ ] GapCandidate + GapEvidenceLink + GapAudit + NeighborComparison 数据模型
- [ ] Pydantic schemas (GapCandidateOut, GapCandidateSchema for LLM)
- [ ] Alembic migration 0010_gap_tables
- [ ] ResearchState 扩展 (active_gap_ids, surviving_gap_ids)
- [ ] PhaseRun 契约 (mining_gaps, auditing_gaps phase names)
- [ ] generate_queries 绑定 target_gap_id (query → gap 关联)
- [ ] 测试 + 验证

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
