# 01 Target Architecture

> Evidence-grounded Research Gap Discovery and Idea Validation System

## 1. 核心理念

**当前**：文献检索后自动生成 Idea 的流水线
**目标**：Evidence-grounded Research Gap Discovery and Idea Validation System

每个 Research Idea 必须能说明：
1. 基于哪些论文证据
2. 最接近哪些已有工作
3. 经过了哪些反向检索
4. 仍然存在什么明确技术差异
5. 如何通过最小实验验证
6. 在什么情况下应该放弃

## 2. 目标主流程

```
用户输入研究方向
  → build_research_contract          (结构化研究方向 + 约束)
  → decompose_research_space          (分解为 research questions)
  → generate_discovery_queries        (多种 intent 的检索 query)
  → search_and_save_papers            (多源检索)
  → classify_paper_roles              (survey/seminal/neighbor/benchmark...)
  → extract_evidence_units            (chunk-level 证据提取)
  → update_coverage_matrix            (研究问题覆盖率)
  → [低覆盖问题驱动定向检索]
  → mine_gap_candidates               (挖掘 Gap)
  → generate_adversarial_queries      (对抗性检索)
  → audit_gap_candidates              (近邻审计)
  → run_feasibility_gates             (4 个可行性门)
  → synthesize_ideas                  (仅从 surviving gaps 生成)
  → independent_idea_judge            (独立评判)
  → waiting_for_user_review
  → generate_pilot_experiment         (最小实验方案)
  → generate_final_research_memo      (最终研究备忘录)
```

## 3. 新增数据库表

### Phase 1
- `research_contracts` — 结构化研究方向
- `research_questions` — 研究问题
- `research_axes` — 研究轴

### Phase 2
- `evidence_units` — 证据单元
- `coverage_records` — 覆盖记录
- `question_evidence_links` — 问题-证据关联
- `search_queries` — 结构化查询记录
- `paper_roles` — 论文角色

### Phase 3
- `gap_candidates` — Gap 候选
- `gap_evidence_links` — Gap-证据关联
- `gap_audits` — Gap 审计
- `neighbor_comparisons` — 近邻对比

### Phase 4
- `gap_gate_results` — 可行性门结果
- `idea_judgments` — 独立评判记录

### Phase 0
- `phase_runs` — 阶段执行记录

## 4. ResearchState 重构

```python
class ResearchState:
    task_id: str
    current_phase: str          # 替代 current_round 作为主进度指标

    contract_id: str | None
    active_question_ids: list[str]
    active_gap_ids: list[str]
    surviving_gap_ids: list[str]
    selected_idea_ids: list[str]

    current_round: int           # 保留用于检索循环
    budget_state: dict           # 预算追踪

    stop_reason: str
```

## 5. Runner 阶段化

```python
PHASES = [
    "clarify",
    "build_contract",
    "decompose",
    "discovery_search",
    "extract_evidence",
    "update_coverage",
    "gap_mining",
    "gap_audit",
    "feasibility_gate",
    "idea_synthesis",
    "idea_judgment",
    "user_review",
]
```

每个阶段：
- 有明确输入输出
- 有数据库状态（phase_runs 表）
- 有 trace
- 可单独重试
- 可恢复
- 可跳过已完成结果

## 6. 决策状态

合法最终结果：
```
GO
REVISE
REJECT
INSUFFICIENT_EVIDENCE
MORE_RESEARCH_REQUIRED
```

任务新增状态：
```
auditing_gaps
checking_feasibility
synthesizing_ideas
judging_ideas
insufficient_evidence
more_research_required
```

## 7. 报告拆分

### Research Landscape Brief（Evidence + Coverage 后）
- 研究问题树
- 论文角色分布
- 方法分类
- Benchmark 分类
- Evidence Coverage Matrix
- 已知限制
- 证据冲突
- 未覆盖区域

### Final Research Memo（Gap Audit + Idea Judge 后）
- 研究范围与约束
- 领域地图
- 关键 Evidence
- 候选 Gap
- Gap 审计结果
- 最近邻对比
- 被拒绝方向及原因
- surviving ideas
- 实验 Gate
- 推荐最小 Pilot
- 主要不确定性

## 8. Wiki 降级

Wiki 从"知识源"降级为"衍生展示层"：
```
Raw Paper → Evidence Unit → Wiki / Cluster / Report
```

禁止：Wiki 内容直接证明 Gap 或 Idea 成立

## 9. 旧流程兼容

- 旧 API 保持兼容
- 旧表不删除
- 旧任务仍可读取
- 旧流程可作为 fallback（通过 feature flag 控制）
- 最终主流程切换到新架构
