# 02 Database Migration Plan

## 1. 原则

- 使用 Alembic 创建迁移
- 不删除旧表
- 不破坏旧任务读取
- 为旧数据提供合理默认值
- 新增表通过 Alembic 迁移，不通过 `create_all()`

## 2. 迁移策略

### Migration 0002: Phase 0 — phase_runs + 修复

新增表：
- `phase_runs` — 阶段执行记录

不修改现有表结构。

### Migration 0003: Phase 1 — Research Contract + Questions

新增表：
- `research_contracts`
- `research_questions`
- `research_axes`

修改现有表：
- `research_tasks` 新增 `contract_id` 字段（nullable，旧任务为 NULL）

### Migration 0004: Phase 2 — Evidence + Coverage + Search Queries + Paper Roles

新增表：
- `evidence_units`
- `coverage_records`
- `question_evidence_links`
- `search_queries`
- `paper_roles`

### Migration 0005: Phase 3 — Gap + Audit

新增表：
- `gap_candidates`
- `gap_evidence_links`
- `gap_audits`
- `neighbor_comparisons`

### Migration 0006: Phase 4 — Gate + Judgment

新增表：
- `gap_gate_results`
- `idea_judgments`

修改现有表：
- `research_ideas` 新增 `source_gap_id` 字段（nullable，旧 ideas 为 NULL）
- `experiment_plans` 新增 `kill_criteria` 字段（nullable）

### Migration 0007: Phase 5 — Report 拆分

修改现有表：
- `reports` 新增 `report_type` 字段（默认 'full'，可选 'landscape_brief' | 'final_memo'）

## 3. 各表详细定义

### phase_runs
```sql
CREATE TABLE phase_runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    phase_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending/running/completed/failed/skipped
    attempt_count INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    input_version TEXT,
    output_version TEXT,
    error_message TEXT,
    round_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_phase_runs_task ON phase_runs(task_id);
CREATE INDEX idx_phase_runs_task_phase ON phase_runs(task_id, phase_name);
```

### research_contracts
```sql
CREATE TABLE research_contracts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    topic TEXT NOT NULL,
    target_problem TEXT,
    target_setting TEXT,
    desired_output TEXT,
    novelty_bar TEXT DEFAULT 'conference',
    preferred_directions_json TEXT DEFAULT '[]',
    excluded_directions_json TEXT DEFAULT '[]',
    gpu_available BOOLEAN,
    max_gpu_hours FLOAT,
    max_api_budget FLOAT,
    max_runtime_minutes INTEGER,
    allow_large_benchmark BOOLEAN DEFAULT TRUE,
    allow_model_training BOOLEAN DEFAULT TRUE,
    experiment_preferences_json TEXT DEFAULT '{}',
    key_terms_json TEXT DEFAULT '[]',
    time_scope_start INTEGER,
    time_scope_end INTEGER,
    status TEXT DEFAULT 'active',
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_contracts_task ON research_contracts(task_id);
```

### research_questions
```sql
CREATE TABLE research_questions (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    contract_id TEXT REFERENCES research_contracts(id),
    question TEXT NOT NULL,
    question_type TEXT NOT NULL,  -- problem/method/evaluation/dataset/resource/failure/application
    importance FLOAT DEFAULT 0.5,
    searchability FLOAT DEFAULT 0.5,
    status TEXT DEFAULT 'open',  -- open/partially_covered/covered/unavailable
    axis_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_rq_task ON research_questions(task_id);
CREATE INDEX idx_rq_status ON research_questions(status);
```

### search_queries
```sql
CREATE TABLE search_queries (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    query_text TEXT NOT NULL,
    intent TEXT NOT NULL,  -- survey/seminal/recent_work/benchmark/direct_neighbor/limitation/negative_result/gap_falsification/component_combination
    target_question_id TEXT REFERENCES research_questions(id),
    target_gap_id TEXT,  -- forward ref to gap_candidates
    expected_evidence_type TEXT,
    round_number INTEGER,
    status TEXT DEFAULT 'pending',
    result_count INTEGER DEFAULT 0,
    new_paper_count INTEGER DEFAULT 0,
    evidence_unit_count INTEGER DEFAULT 0,
    execution_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_sq_task ON search_queries(task_id);
CREATE INDEX idx_sq_intent ON search_queries(intent);
```

### paper_roles
```sql
CREATE TABLE paper_roles (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    paper_id TEXT NOT NULL REFERENCES papers(id),
    role TEXT NOT NULL,  -- survey/seminal/direct_neighbor/benchmark/method/negative_result/limitation_evidence/application
    confidence FLOAT DEFAULT 0.5,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_pr_task ON paper_roles(task_id);
CREATE INDEX idx_pr_paper ON paper_roles(paper_id);
CREATE INDEX idx_pr_role ON paper_roles(role);
```

### evidence_units
```sql
CREATE TABLE evidence_units (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    paper_id TEXT NOT NULL REFERENCES papers(id),
    evidence_type TEXT NOT NULL,  -- problem/method/result/limitation/dataset/metric/negative_result/future_work/comparison
    normalized_claim TEXT NOT NULL,
    original_span TEXT,
    section TEXT,
    page_number INTEGER,
    conditions_json TEXT DEFAULT '{}',
    metric_name TEXT,
    result_value TEXT,
    extraction_method TEXT DEFAULT 'llm',  -- llm/abstract_only/pdf_fulltext
    extraction_confidence FLOAT DEFAULT 0.5,
    verification_status TEXT DEFAULT 'unverified',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_eu_task ON evidence_units(task_id);
CREATE INDEX idx_eu_paper ON evidence_units(paper_id);
CREATE INDEX idx_eu_type ON evidence_units(evidence_type);
```

### coverage_records
```sql
CREATE TABLE coverage_records (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    question_id TEXT NOT NULL REFERENCES research_questions(id),
    coverage_score FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 0.0,
    supporting_evidence_count INTEGER DEFAULT 0,
    contradicting_evidence_count INTEGER DEFAULT 0,
    direct_neighbor_count INTEGER DEFAULT 0,
    unresolved_aspects_json TEXT DEFAULT '[]',
    unavailable_reason TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_cr_task ON coverage_records(task_id);
CREATE INDEX idx_cr_question ON coverage_records(question_id);
```

### question_evidence_links
```sql
CREATE TABLE question_evidence_links (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES research_questions(id),
    evidence_id TEXT NOT NULL REFERENCES evidence_units(id),
    relationship TEXT DEFAULT 'supports',  -- supports/contradicts/partially_supports
    relevance_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_qel_question ON question_evidence_links(question_id);
CREATE INDEX idx_qel_evidence ON question_evidence_links(evidence_id);
```

### gap_candidates
```sql
CREATE TABLE gap_candidates (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    statement TEXT NOT NULL,
    gap_type TEXT NOT NULL,  -- problem_gap/method_gap/evaluation_gap/dataset_gap/resource_gap/robustness_gap/negative_result_gap
    target_setting TEXT,
    observed_problem TEXT,
    existing_coverage TEXT,
    missing_capability TEXT,
    claimed_delta TEXT,
    testable_hypothesis TEXT,
    falsification_condition TEXT,
    supporting_evidence_ids_json TEXT DEFAULT '[]',
    contradicting_evidence_ids_json TEXT DEFAULT '[]',
    nearest_neighbor_ids_json TEXT DEFAULT '[]',
    status TEXT DEFAULT 'candidate',  -- candidate/auditing/audited/rejected/surviving
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_gc_task ON gap_candidates(task_id);
CREATE INDEX idx_gc_status ON gap_candidates(status);
```

### gap_evidence_links
```sql
CREATE TABLE gap_evidence_links (
    id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL REFERENCES gap_candidates(id),
    evidence_id TEXT NOT NULL REFERENCES evidence_units(id),
    relationship TEXT DEFAULT 'supports',  -- supports/contradicts
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_gel_gap ON gap_evidence_links(gap_id);
CREATE INDEX idx_gel_evidence ON gap_evidence_links(evidence_id);
```

### gap_audits
```sql
CREATE TABLE gap_audits (
    id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL REFERENCES gap_candidates(id),
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    coverage_status TEXT NOT NULL,  -- open/partially_covered/fully_covered/uncertain
    evidence_for_gap_json TEXT DEFAULT '[]',
    evidence_against_gap_json TEXT DEFAULT '[]',
    remaining_delta TEXT,
    novelty_confidence FLOAT DEFAULT 0.5,
    audit_confidence FLOAT DEFAULT 0.5,
    recommended_action TEXT,  -- continue/narrow/reject/more_search
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ga_gap ON gap_audits(gap_id);
CREATE INDEX idx_ga_task ON gap_audits(task_id);
```

### neighbor_comparisons
```sql
CREATE TABLE neighbor_comparisons (
    id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL REFERENCES gap_candidates(id),
    paper_id TEXT NOT NULL REFERENCES papers(id),
    shared_problem TEXT,
    shared_mechanism TEXT,
    shared_evaluation TEXT,
    differences_json TEXT DEFAULT '[]',
    covered_claims_json TEXT DEFAULT '[]',
    uncovered_claims_json TEXT DEFAULT '[]',
    overlap_ratio FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_nc_gap ON neighbor_comparisons(gap_id);
CREATE INDEX idx_nc_paper ON neighbor_comparisons(paper_id);
```

### gap_gate_results
```sql
CREATE TABLE gap_gate_results (
    id TEXT PRIMARY KEY,
    gap_id TEXT NOT NULL REFERENCES gap_candidates(id),
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    gate_name TEXT NOT NULL,  -- problem_reality/intervention_signal/observability/experimental_feasibility
    passed BOOLEAN NOT NULL,
    conditional BOOLEAN DEFAULT FALSE,
    details_json TEXT DEFAULT '{}',
    failure_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ggr_gap ON gap_gate_results(gap_id);
CREATE INDEX idx_ggr_gate ON gap_gate_results(gate_name);
```

### idea_judgments
```sql
CREATE TABLE idea_judgments (
    id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL REFERENCES research_ideas(id),
    task_id TEXT NOT NULL REFERENCES research_tasks(id),
    decision TEXT NOT NULL,  -- go/revise/reject/insufficient_evidence
    novelty_verdict TEXT,
    feasibility_verdict TEXT,
    evidence_verdict TEXT,
    experiment_verdict TEXT,
    blocking_issues_json TEXT DEFAULT '[]',
    revision_requirements_json TEXT DEFAULT '[]',
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_ij_idea ON idea_judgments(idea_id);
CREATE INDEX idx_ij_task ON idea_judgments(task_id);
```

## 4. 旧数据兼容

- 所有新增字段默认 nullable，旧记录保持 NULL
- `research_ideas.source_gap_id` 旧 ideas 为 NULL → API 返回时检查
- `reports.report_type` 旧报告默认 'full'
- 旧 `state_json` 仍可读取，新代码兼容旧格式
- 旧 `knowledge_gaps: list[str]` 保留但不作为主要数据源

## 5. 迁移执行

每个 migration 文件：
1. `upgrade()` — 创建新表/字段
2. `downgrade()` — 回滚（DROP TABLE / DROP COLUMN）

SQLite 不支持 ALTER TABLE ADD COLUMN with NOT NULL，所以新增字段必须 nullable 或有默认值。
