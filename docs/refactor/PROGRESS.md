# Refactor Progress

## Phase 0 — implemented, hardening required

### 已完成
- [x] 创建 docs/refactor/ 文档 (00-03)
- [x] 审计全部核心文件
- [x] 修复 db.refresh = None (runner.py:258)
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
- [x] 运行全部测试 — 85 passed

### Phase 1.5 需修复
- [ ] MAX_ATTEMPTS_PER_ROUND 没有真正使用（只有全局预算）
- [ ] PhaseRun 尚未接入 runner
- [ ] 新阶段状态没有进入恢复逻辑 (_INTERRUPTED_STATUSES)
- [ ] 搜索失败后 runner 仍继续报告和 Idea 阶段

## Phase 1 — scaffold implemented, integration pending

### 已完成
- [x] ResearchContract 模型 (models.py)
- [x] ResearchQuestion 模型 (models.py)
- [x] ResearchContractSchema, ResearchDecompositionSchema (schemas.py)
- [x] ContractOut, ResearchQuestionOut API schemas (schemas.py)
- [x] Alembic migration 0003_contract_questions.py
- [x] build_contract.py step
- [x] decompose_research_space.py step
- [x] BUILD_CONTRACT + DECOMPOSE prompts (prompts.py)
- [x] runner.py 集成 — clarify → build_contract → decompose → search_loop
- [x] steps/__init__.py 导出新步骤
- [x] API route: GET /tasks/{id}/contract, GET /tasks/{id}/questions
- [x] main.py 注册 contracts router
- [x] 新增 test_phase1.py — 7 个测试

### Phase 1.5 需修复
- [ ] Contract 对 ResearchState 的修改没有持久化（state.contract_id 未保存）
- [ ] Research Questions 没有驱动 query（generate_queries 仍用旧逻辑）
- [ ] Contract/Questions 无版本和失效机制
- [ ] state.user_input = state.user_input 无意义代码
- [ ] Pydantic schema 缺少 Literal/Enum 约束和 validator
- [ ] API 返回 JSON 字符串而非结构化字段
- [ ] 缺少真实数据库集成测试（仅 mock 测试）

## Phase 1.5: 修复并验证 Phase 0/1 — 进行中

## Phase 2: Evidence Unit + Coverage Matrix — 待实施
