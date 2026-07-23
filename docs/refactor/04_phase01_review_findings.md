# Phase 0/1 Review Findings

> 审查基准提交：059bbee5d9059bcfe2e3134373689c77a3a9634d
> 审查日期：2026-07-23

## 必须修复的问题

### 1. Contract 对 ResearchState 的修改没有持久化

**文件**: `backend/app/agent/steps/build_contract.py`
**问题**: 创建 Contract 后修改了 `state.normalized_topic` 和 `state.keywords`，但没有调用 `task_repo.save_state()` 持久化。`state.contract_id` 也从未设置。
**影响**: 下游步骤无法通过 `state.contract_id` 获取 Contract。
**修复**: 创建 Contract 后执行 `state.contract_id = contract.id` + `task_repo.save_state()` + `db.commit()`。

### 2. Research Questions 没有驱动 query

**文件**: `backend/app/agent/steps/generate_queries.py`
**问题**: `generate_queries()` 仍使用旧逻辑（topic + keywords + gaps），完全忽略 `ResearchQuestion` 表和 `state.active_question_ids`。
**影响**: 检索不针对具体研究问题，覆盖度无法追踪。
**修复**: 新增 `select_target_questions()` 选择低覆盖问题，`generate_queries()` 输出包含 `target_question_id`。

### 3. 搜索失败后 runner 仍继续报告和 Idea 阶段

**文件**: `backend/app/agent/runner.py`
**问题**: `_run_search_loop()` 没有 return value，`run_task()` 不检查搜索是否成功就直接进入 analyze_papers。
**影响**: 搜索完全失败时仍尝试生成报告和 idea，浪费资源。
**修复**: `_run_search_loop()` 返回 `SearchLoopResult`，`run_task()` 只在 `completed/stopped_normally` 时继续。

### 4. MAX_ATTEMPTS_PER_ROUND 没有使用

**文件**: `backend/app/agent/runner.py`
**问题**: 定义了 `MAX_ATTEMPTS_PER_ROUND = 3` 但没有在 round 级别使用——只有全局 `TOTAL_FAILED_ROUND_BUDGET`。
**影响**: 单个 round 可以无限重试直到全局预算耗尽。
**修复**: 为每个 round 维护 `round_attempts[round_num]`，达到上限后该 round 失败。

### 5. PhaseRun 尚未接入 runner

**文件**: `backend/app/agent/runner.py`, `backend/app/db/models.py`
**问题**: `PhaseRun` 模型已定义但 `runner.py` 从未创建或查询 PhaseRun 记录。
**影响**: 无法追踪阶段执行状态，无法阶段级恢复。
**修复**: 新增 `phase_repo.py` + `phase_service.py`，实现 `execute_phase()` 包装器，接入 clarify/build_contract/decompose/search 四个阶段。

### 6. 新阶段状态没有进入恢复逻辑

**文件**: `backend/app/agent/runner.py`
**问题**: `_INTERRUPTED_STATUSES` 只包含旧状态，缺少 `building_contract`, `decomposing` 等。
**影响**: 任务在 build_contract 阶段崩溃后无法恢复。
**修复**: 补全 `_INTERRUPTED_STATUSES`，恢复逻辑优先读取 `PhaseRun(status="running")`。

### 7. Contract/Questions 无版本和失效机制

**文件**: `backend/app/db/models.py`, `backend/app/agent/steps/build_contract.py`, `decompose_research_space.py`
**问题**: Contract 没有 version/input_hash/superseded_at。Questions 只按 task_id 查询，不区分 contract 版本。
**影响**: 用户反馈改变研究方向后，旧 Contract/Questions 不会被失效。
**修复**: 新增 migration 0004 添加版本字段，build_contract 计算 input_hash 并做版本管理，decompose 按 contract_id 筛选。

### 8. Alembic 迁移路径没有真实验证

**文件**: `backend/app/main.py`, `backend/alembic/`
**问题**: `main.py` 仍调用 `Base.metadata.create_all()`，Alembic 迁移从未实际执行。`0001_baseline` 是空操作。
**影响**: 无法保证 schema 一致性，新数据库无法通过 Alembic 建表。
**修复**: 创建 `0000_initial_schema` 真实迁移，新增 `bootstrap_db.py`，停止生产路径 `create_all()`。

### 9. 当前测试主要为 mock 测试，缺少真实数据库集成测试

**文件**: `backend/tests/`
**问题**: 所有测试使用 MagicMock 模拟 db session，没有使用真实 SQLite 文件验证持久化。
**影响**: Contract/Questions 的跨 session 持久化、版本化、supersede 逻辑未经验证。
**修复**: 新增使用临时 SQLite 文件的集成测试。
