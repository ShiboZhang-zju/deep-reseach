# 03 Implementation Plan

## 实施顺序

### Phase 0: 审计与修复明显 Bug（当前阶段）

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 0.1 | 创建 docs/refactor/ 文档 | docs/refactor/ | ✅ |
| 0.2 | 修复 db.refresh = None | runner.py | 待修 |
| 0.3 | 修复澄清答案丢失 | runner.py, clarify_topic.py | 待修 |
| 0.4 | 修复 Idea retry 条件 | runner.py | 待修 |
| 0.5 | 移除 auto-promote | runner.py | 待修 |
| 0.6 | 增加 round retry 上限 | runner.py | 待修 |
| 0.7 | 统一 policy 终止逻辑 | policy.py, runner.py | 待修 |
| 0.8 | 修复 generate_experiment conditional_go | generate_experiment.py | 待修 |
| 0.9 | 新增 phase_runs 表 + 模型 | models.py, alembic | 待修 |
| 0.10 | 新增任务状态枚举 | 全项目 | 待修 |

### Phase 1: Research Contract + Question Decomposition

| # | 任务 | 文件 |
|---|------|------|
| 1.1 | ResearchContract 模型 + schema | models.py, schemas.py |
| 1.2 | ResearchQuestion 模型 + schema | models.py, schemas.py |
| 1.3 | Alembic migration 0003 | alembic/versions/ |
| 1.4 | build_contract step | steps/build_contract.py |
| 1.5 | decompose_research_space step | steps/decompose_research_space.py |
| 1.6 | 修复 clarify → build_contract 链路 | runner.py, clarify_topic.py |
| 1.7 | Contract API routes | api/routes/ |
| 1.8 | 前端 Contract 展示 | frontend/ |

### Phase 2: Evidence + Coverage

| # | 任务 | 文件 |
|---|------|------|
| 2.1 | EvidenceUnit 模型 + schema | models.py, schemas.py |
| 2.2 | SearchQuery 模型 + schema | models.py, schemas.py |
| 2.3 | PaperRole 模型 + schema | models.py, schemas.py |
| 2.4 | CoverageRecord + QuestionEvidenceLink 模型 | models.py |
| 2.5 | Alembic migration 0004 | alembic/versions/ |
| 2.6 | extract_evidence_units step | steps/extract_evidence.py |
| 2.7 | classify_paper_roles step | steps/classify_roles.py |
| 2.8 | update_coverage_matrix step | steps/update_coverage.py |
| 2.9 | coverage-driven query 生成 | steps/generate_queries.py (重构) |
| 2.10 | coverage-driven stop | policy.py (重构) |
| 2.11 | Evidence API routes | api/routes/ |

### Phase 3: Gap Mining + Audit

| # | 任务 | 文件 |
|---|------|------|
| 3.1 | GapCandidate 模型 + schema | models.py, schemas.py |
| 3.2 | GapAudit + NeighborComparison 模型 | models.py |
| 3.3 | Alembic migration 0005 | alembic/versions/ |
| 3.4 | mine_gaps step | steps/mine_gaps.py |
| 3.5 | generate_adversarial_queries | steps/audit_gaps.py |
| 3.6 | audit_gaps step | steps/audit_gaps.py |
| 3.7 | Gap API routes | api/routes/ |

### Phase 4: Feasibility Gate + Idea

| # | 任务 | 文件 |
|---|------|------|
| 4.1 | GapGateResult 模型 + schema | models.py |
| 4.2 | IdeaJudgment 模型 + schema | models.py |
| 4.3 | Alembic migration 0006 | alembic/versions/ |
| 4.4 | run_feasibility_gates step | steps/run_feasibility_gates.py |
| 4.5 | synthesize_ideas step (重写) | steps/synthesize_ideas.py |
| 4.6 | judge_ideas step (独立) | steps/judge_ideas.py |
| 4.7 | 修改 research_ideas 表 | migration |
| 4.8 | 移除旧 generate_ideas.py | (保留为 legacy) |
| 4.9 | Idea API 更新 | api/routes/ideas.py |

### Phase 5: 报告 + Wiki + 前端

| # | 任务 | 文件 |
|---|------|------|
| 5.1 | reports 表加 report_type | migration 0007 |
| 5.2 | generate_landscape_brief step | steps/generate_landscape.py |
| 5.3 | generate_final_memo step | steps/generate_memo.py |
| 5.4 | Wiki 降级 + evidence linkage | wiki_service.py |
| 5.5 | 前端新状态 + Contract + Gap + Evidence 展示 | frontend/ |

### Phase 6: 测试 + 文档

| # | 任务 | 文件 |
|---|------|------|
| 6.1 | Research Contract 测试 | tests/ |
| 6.2 | Search 测试 | tests/ |
| 6.3 | Evidence 测试 | tests/ |
| 6.4 | Coverage 测试 | tests/ |
| 6.5 | Gap 测试 | tests/ |
| 6.6 | Idea 测试 | tests/ |
| 6.7 | Runner 测试 | tests/ |
| 6.8 | API 测试 | tests/ |
| 6.9 | 前端构建 | frontend/ |
| 6.10 | README 更新 | README.md |
| 6.11 | .env.example 更新 | .env.example |
| 6.12 | 最终报告 | docs/refactor/04_final_refactor_report.md |

## 进度记录

每完成一个 Phase，在 `docs/refactor/PROGRESS.md` 记录。
