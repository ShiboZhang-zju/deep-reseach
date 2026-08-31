# production_e2e_v1 — Core Thesis Eval

**一句话原则**：先构造一个公平的三系统对照，让 E2E 实验能回答"复杂 pipeline 到底有没有价值"，再决定下一轮开发什么。

ResearchBench 与 RINoBench 已承担**局部能力**评估（retrieval / generation / ranking / novelty judging）。
本 harness 是第三条独立 eval，验证本项目核心主张：

> **同样的文献供给下，Evidence → Gap Audit → Intervention → Experiment 全链路
> 相对简单方案（直接生成 / 检索后生成）更能产生可信科研 Idea，更能拒绝假 Idea。**

不修改 production `app/` 代码、不动冻结的 eval-harness-v1、不新建数据库表。

## 实验设计（v1 冻结，改动须递增版本号）

### 三系统

| 系统 | 输入 | 链路 |
|------|------|------|
| A: Direct LLM | topic 文本 | 单次 LLM 调用直接产 Idea |
| B: Retrieval + LLM | topic + **Full V2 检索到的同一批论文**（top-K 摘要） | 读文献 → 单次 LLM 调用产 Idea |
| C: Full V2 | topic | Search → Evidence → Coverage → Gap → NPA Audit → Intervention → Experiment |

**设计点 1（检索条件控制）**：Baseline B 不自己检索，直接消费 Full V2 的论文导出
（`papers_export.jsonl`，按 final_score 排序取 top-K=20 的 title/year/venue/abstract）。
这样比较的是"同样的文献供给下，Evidence + Gap Audit 到底有没有增益"。
若 B 自检索（50 篇）而 V2 检索 300 篇，赢了也无法归因是检索强还是审计强。
self-retrieval Baseline B 留给 v2 单独测 adaptive retrieval 的增益。

**设计点 2（双 headline 指标，防 abstain 作弊）**：只看 False-open-gap rate 会激励
"全部 abstain"（rate=0 但零科研价值）。因此同时冻结两个 headline：

```
False Open Gap Rate  = 被独立验证已有 prior art 覆盖的 claimed gap / 全部声称开放的 gap   ↓
Credible Idea Yield  = 至少产出 1 个通过独立验证的可信 Idea 的 topic / 全部 topic          ↑
```

区分"系统 A：false gap 10% 但 90% abstain"与"系统 B：false gap 20% 但 70% topic 有可信 idea"。
Abstention rate 单独报告。最终表：

| System | False-open-gap ↓ | Credible Idea Yield ↑ | Novelty ↑ | Feasibility ↑ | Abstention | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Direct LLM | | | | | | |
| Retrieval + LLM | | | | | | |
| Full V2 | | | | | | |

Novelty / Feasibility / False-open-gap 由 Super Audit + 人工核对提供（见下），
Abstention 与 Cost 由 harness 自动统计。

**设计点 3（独立 Super Audit，不与 production auditor 共享盲区）**：不能把 production
Gap Audit 再跑一遍叫 independent validation——会共享 query family、检索源、模型偏好与
检索盲区，可能一起漏掉同一篇 killer paper。流程：

```
系统输出冻结
  → Independent Super Audit（不同 prompt 模板 / 更宽 query / 多源检索 / citation snowball）
  → candidate killer papers
  → 人工核对 FULL / PARTIAL / NONE 覆盖
```

人工只审核机器找到的关键 nearest prior art + 少量"未找到 killer"的对照样本，
不要求重新检索整个领域（成本可控）。

**设计点 4（主题分层，防 selection bias）**：不按"V2 跑成功过"挑题。24 个主题预分层：

| 分层 | 数量 | 测什么 |
|------|------|--------|
| narrow_mature（窄 + 文献成熟） | 12 | Gap / Novelty |
| emerging_sparse（新兴 / 稀疏） | 6 | retrieval + uncertainty |
| broad_ambiguous（宽泛 / 边界模糊） | 6 | abstention / scope control |

宽主题不是废样本：它们测试系统能否正确认识"当前不应该产 Idea"。

### 公平性控制（两处小调整）

1. **同一基础模型**：三系统共用 `get_llm()`（production 同款 provider：Qwen3.5-397B 主 +
   35B fallback）。Generator model / temperature（0.3，= production idea 生成链路
   `chat_json` 默认值）/ topic / 最终 Idea 输出 schema 完全相同，唯一变量是
   information / workflow。禁止 A 用 35B 而 V2 用 397B 的配置。
2. **固定评价单位**：每系统每 topic 最多提交 1 个 final Idea（Full V2 取 active ideas 中
   final_score 最高者；Baseline 由 schema 约束单 Idea 输出），否则明确 Abstain。
   不允许 A 出 10 个、B 出 5 个、V2 出 1 个的可笑比较。

### Abstain 语义

- Baseline A/B：LLM 输出 `decision=abstain`（对该领域没有足够可靠依据时诚实弃权）。
- Full V2：终态无 active idea（`waiting_for_user_review`/`more_research_required`/
  `insufficient_evidence`/`abstained` 且 ideas 为空）→ abstain，abstain_reason 记终态与
  stop_reason，保留语义区别（弃权原因进分析，不丢信息）。

## 目录与运行

```
backend/eval/production_e2e/
├── topics.jsonl           # 24 主题冻结（topic_id/stratum/topic）
├── schema.py              # 统一 Idea 输出契约（FinalIdea + E2EDecision）
├── baseline_direct.py     # A: Direct LLM
├── baseline_retrieval.py  # B: Retrieval + LLM（消费 V2 论文导出）
├── run_full_v2.py         # C: HTTP 驱动 production（需后端 localhost:8000 在跑）
├── super_audit.py         # 独立超级审计（候选 killer 检索）
├── evaluate.py            # 六列 headline 表聚合
└── README.md              # 本文件（设计冻结）
```

运行顺序（**B 依赖 C 的论文导出**）：

```bash
cd backend

# 0. 冻结主题集检视
python -m eval.production_e2e.topics_check     # （可选）打印分层统计

# 1. Baseline A（无依赖，可先跑）
python -m eval.production_e2e.baseline_direct --run-id pe2e_v1_direct

# 2. Full V2（需先启动后端；每 topic 20-60 min，串行跑 24 topic）
python -m eval.production_e2e.run_full_v2 --run-id pe2e_v1_fullv2

# 3. Baseline B（读 step 2 导出的 papers_export.jsonl）
python -m eval.production_e2e.baseline_retrieval --run-id pe2e_v1_retellm \
    --v2-run-dir ../eval_results/pe2e_v1_fullv2

# 4. 独立超级审计（对三系统的全部 idea / V2 的 surviving gaps）
python -m eval.production_e2e.super_audit --run-id pe2e_v1_audit \
    --systems pe2e_v1_direct pe2e_v1_retellm pe2e_v1_fullv2

# 5. 人工核对 super_audit 输出的候选 killer（FULL/PARTIAL/NONE），然后聚合
python -m eval.production_e2e.evaluate --runs pe2e_v1_direct pe2e_v1_retellm pe2e_v1_fullv2 \
    --audit pe2e_v1_audit
```

所有 run 产物在 `backend/eval_results/<run_id>/`：`config.json` / `predictions.jsonl` /
`papers_export.jsonl`（仅 Full V2）/ `metrics.json` / `summary.md`，支持 `--limit` / `--resume`
（按 topic_id 跳过已完成样本）。与 eval-harness-v1 的产物契约一致（attempts.jsonl 全量成本口径 +
predictions.jsonl 每 topic 唯一最终成功记录）。

## 成本口径（如实记录，不强行可比）

- Baseline A：llm_calls / tokens / latency（CallStats）。
- Baseline B：同上 + 消费的论文数（检索成本记在 Full V2 头上，README 如实声明）。
- Full V2：wall_clock_s + papers_count（+ best-effort 的 traces token 汇总）。
  V2 是全链路成本、baseline 是单调用成本，两者本来就不在一个口径上——
  表格中 Cost 列分别呈现，不合成单一数字。

## 结果裁决规则（预注册）

- Full Pipeline 增益主要来自 gap correctness → 继续加强 NPA / Audit；
- novelty 仍差 → 查 retrieval；
- novelty 好但 feasibility 差 → 优化 Intervention / Experiment;
- 与 Retrieval+LLM 基线无显著差异 → **删机制，而不是继续往系统上加东西**。

## 明确不做（v1）

self-retrieval Baseline B、新数据库表、DAG 状态机、judge framework、RLCF/偏好训练、
执行验证、来源可靠性加权。见调研报告 P2 暂缓清单。
