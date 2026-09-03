# v14 → v15 渐进式对抗审计 Paired Evaluation Protocol（预注册）

冻结时间：2026-09-02（v14 正式实验 pe2e_v1_fullv2 跑完之前；v15 未激活）。
冻结目的：防止跑完之后根据结果挑指标。本协议的任何改动必须：版本号递增 + 在文末「变更记录」注明原因与日期。

## 0. 两个待比较的运行

| 运行 | run-id | 审计策略 | 其他一切 |
|---|---|---|---|
| v14（baseline） | `pe2e_v1_fullv2` | GAP_SEARCH v14，`gap_audit_progressive=false` | 同 topics.jsonl / 同模型 / temp=0.3 / 同 clarify policy（protocol_violation）/ MAX_LLM_CALLS_PER_TASK=4000 / timeout 14400s |
| v15（treatment） | `pe2e_v1_fullv2_v15` | GAP_SEARCH v15，`gap_audit_progressive=true` | 与 v14 完全一致，唯一差异是 progressive 开关 |

v15 一次题都不多跑、不少跑；两者都基于同一份 topics.jsonl（24 题）。并发数不进入协议（它是运维参数，两版用当时实际值，但在 config.json extra 中留痕）。

## 1. 指标注册表（定义 + 取数源，缺一不收）

### 效率组（预期方向：全部下降）

| 指标 | 定义 | 取数源 |
|---|---|---|
| `audit_wall_time` | 任务内所有 audit_gaps PhaseRun 时长之和 | phase_runs 表 |
| `full_audit_count` | 走到 LLM 判决调用的审计次数（search_admission_status=PASS 且非预检停止） | gap_audits 表 + `gap_audit_pre_fulltext_stop` trace 排除法 |
| `progressive_early_stop_count` | P0-1 预检停止次数（v15 独有） | step_type=`gap_audit_pre_fulltext_stop` 的 AgentTrace 计数 |
| `pdf_attempted` | 近邻全文抽取尝试篇数 | `audit_neighbor_evidence` action trace 的 papers_attempted 求和（v15 波次记录 papers_attempted_total，取其和） |
| `pdf_succeeded` | verified/upgraded 全文证据覆盖的近邻篇数 | EvidenceUnit（verification_status ∈ {verified, upgraded}）按 neighbor 去重计数 |
| `llm_cost_in_audit` | 审计阶段 LLM token 消耗（主指标）；调用次数为次级代理 | audit 窗口内 AgentTrace.llm_tokens_used 求和。注：精确调用计数器若在激活时以纯计量代码补上则采用之，不补则 tokens 为准 |
| `audit_rounds_per_gap` | 每 gap 的 GapAudit 行数（含预检停止行） | gap_audits 表按 gap 分组 |

### 效度组（预期方向：一致，不设任何"改善"预期）

| 指标 | 定义 | 取数源 |
|---|---|---|
| `final_gap_status` | 每个 gap 终态（surviving/audited/rejected/closed） | gap_candidates.status |
| `final_novelty_confidence` | 每 gap 最终一次 GapAudit.novelty_confidence | gap_audits 表（按 created_at 取末条） |
| `surviving_gap_count` | 每任务 surviving gap 数 | gap_candidates |
| `tier_A_idea_count` | 每任务 confidence_tier=="A" 的 idea 数 | ResearchIdea.confidence_tier（枚举 A/B/C，"A" 为精确匹配） |
| `verdict_agreement` | 配对 gap 的终态一致率 | 见 §3 配对方法 |

**判读纪律**：效率组与效度组分开报告，绝不合并为单一分数。效度组任何指标恶化（verdict/surviving/tier_A 不一致）都必须给出 per-case 根因分析，即使效率收益很大。

## 2. 总量指标（补充，不作决策依据）

`NO_FULLTEXT_NEIGHBOR_EVIDENCE` failure code 在 verdict ceiling 检查中的发生率：v15 相对 v14 的变化。用于检测 P0-2 wave 停止是否过激（见 §4）。

## 3. 跨 run gap 配对方法（verdict agreement 与 EarlyStop Precision 的基础）

配对键：`normalize(claimed_delta)`（去空白、小写、去标点）+ topic_id 必须相同。
跨 run 检索随机性可能使 v15 某个 gap 在 v14 中无同键孪生：**必须报告配对成功率**（matched / v15 gaps），配对失败的 case 单列，不进 agreement 分母也不进分子。

## 4. 两个预注册安全指标

### 4.1 EarlyStop Precision（P0-1 的安全性）

```text
EarlyStop Precision
  = # { v15 预检停止的 gap，其 v14 孪生 gap 终态为「未通过」}
    / # { v15 预检停止的 gap（有 v14 孪生） }
```

「未通过」= rejected / closed / 终态仍为 uncertain 且从未 survive。
目标 ≈ 1.0。任何 misfire（v14 孪生最终 survive / 进 intervention）都必须逐案分析：是检索随机性还是预检逻辑缺陷。**若 Precision < 0.9，P0-1 判定失败，回滚预检（保留 P0-2）。**

### 4.2 PDF Saving 与过激检查（P0-2 的安全性）

```text
PDF Saving = 1 − (v15 pdf_attempted / v14 pdf_attempted)
```

同时检查 `NO_FULLTEXT_NEIGHBOR_EVIDENCE` 发生率。若 PDF Saving > 0 且该 failure code 发生率显著上升（> v14 的 1.5 倍），说明 wave 停止过激，需将 `audit_neighbor_evidence_wave_size` 调大或提高停止阈值后重测——此项调整属于协议 v2 变更，须记录。

## 5. 项目不变量（自本协议起生效，P0-3 必须遵守）

1. **复用事实，不复用判决**：Evidence cache（paper / fulltext / NeighborClaimCoverage / 评分缓存）跨 policy version 复用；Decision cache（confirmed / reject / narrow）永不跨 policy version 直接复用。policy version ≠ retrieval evidence identity。
2. **非对称门控**：任何廉价/预检路径只能杀死（reject / narrow / more_search），不能产生 confirmed。
3. **per-RQ 判定不被 task 级聚合替代**（沿袭既有约定）。
4. **efficiency 改动不得触碰 novelty validation 标准本身**：本优化的科学表述是「把 eager execution 改为 evidence-triggered execution，并跨轮复用已验证的事实证据」，不是「放宽了查新」。

## 6. P0-3 冻结设计（v15 激活与评估完成后再动工）

- **P0-3a（先行）**：审计输入增量——只对 `new_neighbors × unresolved_claims` 构造判决输入，沿用现有 audit schema 与 verdict pipeline。`unresolved = claims − claims_with_any_FULL`（集合减法，PARTIAL/NONE/UNCERTAIN 全部保持 unresolved）；已 FULL 的 claim 被 prior art 杀死，不再重复送判。persist delta coverage → `_derive_verdict_from_claims(全部持久化 coverage)`。**不拆 prompt contract，不改 DB schema。**
- **P0-3b（后行，仅在 P0-3a 验证 verdict 一致后）**：拆 `_AUDIT_SYSTEM` 为 Evidence Judge（Paper × Claim → FULL/PARTIAL/NONE/UNCERTAIN + rationale，只做事实判断）+ 代码化判决推导；仅 provisional_confirmed 时调用 Falsifier LLM（closest_killer_work / killer_query_terms / residual_uncertainty）。
- **P0-5（adaptive query 4→8→12）无限期推迟**：仅当 v15 激活后 profile 显示 `search_and_save_papers` 仍占 audit 40%+ 才立项；否则视为过度工程拒绝。

## 7. 决策规则（预注册，防"看到数据再定方向"）

```text
v15 激活 + paired analysis 完成
  ├─ 效度组全一致 且 PDF Saving 显著
  │     → P0-3a 立项（增量 claim 审计）
  ├─ 效度组不一致（Precision<0.9 或 verdict 漂移）
  │     → 先修 P0-1/P0-2 缺陷，禁止继续叠优化
  └─ audit_wall_time 下降不足 30% 且 search_and_save_papers 占比 > 40%
        → 重新评估 P0-5 优先级
```

## 变更记录

- v1（2026-09-02）：首版冻结。作者：AI 协助起草，用户裁决（渐进式对抗审计 + 增量重审路线、P0-3a/b 拆分、复用事实不复用判决不变量、EarlyStop Precision 指标）。
