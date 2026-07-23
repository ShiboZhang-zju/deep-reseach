# Refactor Progress

## Phase 0: 审计与修复明显 Bug — ✅ 完成

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
- [x] 执行 phase_runs 表创建
- [x] 前端 STATUS_LABELS 和 STATUS_COLORS 更新
- [x] 修改 test_fixes.py — 适配 auto_promote 废弃
- [x] 新增 test_phase0_fixes.py — 9 个测试
- [x] 运行全部测试 — 85 passed

## Phase 1: Research Contract + Question Decomposition — ✅ 完成

### 已完成
- [x] ResearchContract 模型 (models.py)
- [x] ResearchQuestion 模型 (models.py)
- [x] ResearchContractSchema, ResearchDecompositionSchema (schemas.py)
- [x] ContractOut, ResearchQuestionOut API schemas (schemas.py)
- [x] Alembic migration 0003_contract_questions.py
- [x] 执行 research_contracts + research_questions 表创建
- [x] build_contract.py step — 从用户输入编译结构化研究契约
- [x] decompose_research_space.py step — 分解为5-12个可检索研究问题
- [x] BUILD_CONTRACT_SYSTEM/USER, DECOMPOSE_SYSTEM/USER prompts (prompts.py)
- [x] runner.py 集成 — clarify → build_contract → decompose → search_loop
- [x] steps/__init__.py 导出新步骤
- [x] API route: GET /tasks/{id}/contract, GET /tasks/{id}/questions
- [x] main.py 注册 contracts router
- [x] 新增 test_phase1.py — 7 个测试
- [x] 运行全部测试 — 92 passed

### 修改文件清单
1. `backend/app/db/models.py` — 新增 ResearchContract, ResearchQuestion 模型
2. `backend/app/schemas/schemas.py` — 新增 Phase 1 schemas
3. `backend/app/agent/prompts.py` — 新增 BUILD_CONTRACT + DECOMPOSE prompts
4. `backend/app/agent/steps/build_contract.py` — 新步骤
5. `backend/app/agent/steps/decompose_research_space.py` — 新步骤
6. `backend/app/agent/steps/__init__.py` — 导出新步骤
7. `backend/app/agent/runner.py` — 集成 build_contract + decompose 到主流程
8. `backend/app/api/routes/contracts.py` — 新 API routes
9. `backend/app/main.py` — 注册 contracts router
10. `backend/alembic/versions/0003_contract_questions.py` — 新迁移
11. `backend/tests/test_phase1.py` — 新测试

### 尚存问题
- 前端 Contract/Questions 展示待 Phase 5 实现
- README 待 Phase 6 更新
- 旧 clarify_topic 仍保留（兼容旧流程）

### 下一阶段
Phase 2: Evidence Unit + Coverage Matrix
