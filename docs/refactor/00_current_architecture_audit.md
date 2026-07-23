# 00 Current Architecture Audit

> 审计日期：2026-07-23
> 审计范围：backend/app/ 全部模块 + frontend/src/ 关键类型

## 1. 现有流程

```
用户输入 → clarify_topic → generate_queries → search_and_save_papers → score_papers
  → summarize_round → [循环5轮] → analyze_papers → wiki_ingest → build_clusters
  → generate_report → generate_and_score_ideas(5层验证) → [retry 2轮] → auto_promote
  → waiting_for_user_review → generate_experiments
```

## 2. 关键文件清单

### Agent 核心
| 文件 | 行数 | 职责 |
|------|------|------|
| `runner.py` | 745 | 主编排，含 search loop、ideas loop、auto_promote |
| `state.py` | 36 | ResearchState dataclass（序列化到 state_json） |
| `policy.py` | 26 | 终止条件判断（仅 max_rounds + high_priority_target） |
| `prompts.py` | 724 | 所有 LLM prompt 模板 |
| `steps/clarify_topic.py` | 19 | LLM 判断方向是否明确 |
| `steps/generate_queries.py` | 29 | LLM 生成 3-5 query |
| `steps/search_papers.py` | 119 | 多源检索 + 去重 + 保存 |
| `steps/score_papers.py` | 176 | LLM 逐篇评分 |
| `steps/summarize_round.py` | 30 | 轮次摘要 + gaps |
| `steps/analyze_papers.py` | 382 | PDF 下载 + PyMuPDF 提取 + LLM 结构化分析 |
| `steps/build_clusters.py` | 77 | Wiki clusters 或 LLM fallback 聚类 |
| `steps/generate_report.py` | 377 | STORM 两步式报告 |
| `steps/generate_ideas.py` | 750 | Idea 生成 + 5层验证 + 评分 |
| `steps/generate_experiment.py` | 112 | 实验方案生成 |

### 数据库模型（models.py）
| 表 | 用途 |
|----|------|
| research_tasks | 任务 |
| papers | 论文 |
| task_papers | 任务-论文关联（含评分） |
| research_rounds | 检索轮次 |
| reports | 报告 |
| research_ideas | Idea |
| experiment_plans | 实验方案 |
| agent_traces | 执行轨迹 |
| user_feedbacks | 用户反馈 |
| paper_chunks | RAG chunks |
| paper_analyses | 论文深度分析 |
| paper_citations | 引用关系 |
| wiki_pages | Wiki 页面 |

### Alembic 迁移
- 仅 `0001_baseline.py`（初始建表）
- 之后所有新表（paper_analyses, paper_citations, wiki_pages, paper_chunks）均通过 `Base.metadata.create_all()` 自动创建，**未通过 Alembic 迁移**

## 3. 已确认的 Bug 和架构问题

### P0 级 Bug（必须修复）

**Bug-1: 澄清答案丢失**（`runner.py:347-355`）
```python
if "\nClarifications:" in state.user_input and not state.normalized_topic:
    state.normalized_topic = state.user_input.split("\nClarifications:")[0].strip()
```
- 用户澄清回答只被截断保留原始输入，澄清内容未结构化进入 Contract
- 违反指令：禁止 `state.user_input.split("\nClarifications:")[0]`

**Bug-2: db.refresh = None**（`runner.py:258`）
```python
db.refresh = None  # noop
```
- 覆盖了 SQLAlchemy Session 的 `refresh` 方法，虽然标记为 noop 但是危险操作
- 违反指令：不得覆盖 SQLAlchemy Session 方法

**Bug-3: Idea retry 逻辑错误**（`runner.py:617`）
```python
if go_count > 0 or revise_count > 0 or active_count > 0:
    break
```
- `active_count > 0` 意味着即使所有 active ideas 都是 reject，也会 break
- 应该只在有 go 或 revise 时才停止

**Bug-4: auto_promote 逻辑**（`runner.py:696-744`）
- 当 max idea rounds 达到后，将分数 ≥0.55 的 reject idea 自动提升为 `conditional_go`
- 违反指令：禁止 auto-promote

**Bug-5: generate_experiment.py 中的 conditional_go**（`generate_experiment.py:56`）
```python
if decision == "go" or decision == "conditional_go":
    good_ideas.append(idea)
```
- `conditional_go` 不应被视为 good idea

### P1 级问题（架构缺陷）

**Arch-1: 终止逻辑分散**（`policy.py` + `runner.py:460-482`）
- `policy.py` 只检查 max_rounds 和 high_priority_target
- 实际的 early_termination（no_new_high + duplicate_rate）在 `runner._check_early_termination` 中

**Arch-2: state_json 存储业务真相**
- `knowledge_gaps: list[str]`、`high_priority_paper_ids` 等关键状态存在 state_json 中
- 不是结构化数据库表，无法查询和关联

**Arch-3: 论文角色缺失**
- 只有单一 `priority`（high/medium/low）基于总分
- 无 survey/seminal/direct_neighbor/benchmark/negative_result 角色分类

**Arch-4: 无 Evidence Unit 层**
- 从论文摘要直接跳到 Wiki/Report/Idea
- `paper_analyses` 有 6 个字段但不是可追溯的 evidence units

**Arch-5: Idea 生成无 Gap 前置**
- 直接从论文+报告生成 idea
- 无 Gap mining → audit → feasibility gate 流程

**Arch-6: 无独立 Judge**
- `generate_ideas.py` 中 `_score_idea` 既是生成者也是评分者
- 无独立 auditor

**Arch-7: 无 Research Contract**
- 只有 `normalized_topic` 字符串 + `keywords` 列表
- 无资源约束、实验偏好、novelty_bar 等

## 4. 依赖关系分析

### steps/__init__.py 导出
```python
from app.agent.steps import (
    clarify_topic,
    generate_queries,
    search_and_save_papers,
    score_papers,
    summarize_round,
    build_paper_clusters,
    generate_report,
    generate_and_score_ideas,
    generate_experiments,
)
```
- 所有 step 函数通过 `__init__` 导出
- runner.py 从 `__init__` 导入

### 数据流
```
state.user_input → clarify → state.normalized_topic + state.keywords
  → queries → search → papers → score → state.high_priority_paper_ids
  → analyze_papers → paper_analyses table
  → wiki → wiki_pages table
  → clusters → ClusterList (in-memory)
  → report → reports table
  → ideas → research_ideas table
```

## 5. 前端状态
- TypeScript types 与后端 Pydantic schemas 对齐
- STATUS_LABELS 包含现有所有状态
- 缺少新状态（insufficient_evidence, more_research_required, auditing_gaps 等）

## 6. README 与实际差异
- README 描述的流程图与实际实现基本一致
- README 中 `MVP 2` 列出的 PDF 全文解析已实现（analyze_papers）
- README 未记录 auto_promote、conditional_go 等逻辑
- README 未记录 RAG 在 Windows 上禁用的限制
