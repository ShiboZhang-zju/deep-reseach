# Eval 主线汇报（任务分叉回归后）

> **重要说明：平行执行与复现验证**。主线指令由两个会话平行执行：eval agent 先行完成了 V2 clean rerun 与 V3 首跑（`v3_rinobench_277`，含 8f54fea CLI 映射修复），本会话独立重跑了同一主线（resume 完成 `v2_rb_ranking_50_clean` 剩余 33 样本、启动 `v3_rinobench_full277`）。因此 V2/V3 各有两次运行，**方向性结论完全一致，构成意外的复现验证**：
>
> | 指标（V3 vs V1） | run 1（eval agent, v3_rinobench_277） | run 2（本会话, v3_rinobench_full277） |
> |---|---:|---:|
> | Macro-F1 | 19.9% (+3.6pp) | 18.3% (+2.0pp) |
> | F1 class 2 | 34.6% (+25.6pp) | 27.1% (+18.0pp) |
> | Low-Novelty FP | 62.7% (-33.3pp) | 70.7% (-25.3pp) |
> | High-Novelty FN | 48.7% (+27.0pp) | 42.6% (+20.9pp) |
> | F1 class 1/5 | 仍为 0 | 仍为 0 |
> | Abstention / Cost | 0% / +8% tokens | 0% / +8% tokens |
>
> 两次独立运行的 FP/FN 幅度有 ~8pp run 间方差（MoE 服务端非确定性；75 个 gold≤2 样本约 6 个翻转），结论方向稳定但幅度表述需带区间。以下正文以本会话 run 为准，结论与 run 1 互证。

## 决策
- pe2e-001/002 pilot 判定为旁路实验：跑完只留档，不作为本轮验收，不自动启动 24 topics / super_audit / human_verdicts。当前状态：pe2e-002 已完成（waiting_for_user_review 留档），pe2e-001 在 auditing_gaps 阶段继续跑完。

## A+B：ResearchBench Ranking-50 V2 clean rerun → 冻结为 V2 Ranking Clean Baseline
- run dir: `eval_results/v2_rb_ranking_50_clean`（resume 接续完成，50/50 ok）
- transport failures: 2/1500 = **0.13% ≤ 5%** → 达标，称 V2 Ranking Clean Baseline
- parse failures: 0
- 官方 scorer（样本级）：overall_accuracy **0.81**；order_accuracy res 0.74 / fan_1_res 0.88；gold_wins both 0.74 / exactly_one 0.14 / neither 0.12；pair self_contradictory 0.187
- 比较级（与 V1 frozen 口径一致）：gold win **0.7100**（real-response 0.7109, n=1498）
- gold-first 0.6693（real 0.6702, n=750）/ negative-first 0.7507（real 0.7517, n=750）
- delta_position = **-0.0814**，bootstrap95%CI **[-0.1268, -0.0360]**（不跨 0，negative-first 呈现时更偏向 gold）
- P(select candidate_1): res order 0.6693（=P(gold)）/ fan order 0.2480（=P(negative)）
- 成本：1500 calls / 1,788,093 tokens / 5654s wall / 1192 tok/cmp
- **vs V1 frozen（directional 0.14, gf 0.20 / nf 0.08）**：criterion-first 在 clean 网络下仍 5 倍显著优于 V1 → 问题 1 答案：是

## C+D+E：RINoBench criterion-first-v3
- 实现核对：与预注册规格完全一致（hypothesis 逐字冻结、NoveltyJudgmentV3 八字段、ORDINAL_SEMANTICS_V3 五档语义、PRESENTATION_SIGNALS_BAN 禁则、一次 call/sample、temp=0、rank_policy CLI 默认 v1 不污染）
- run dir: `eval_results/v3_rinobench_full277`，277/277 ok，0 parse error，唯一变量 novelty_policy v1→v3
- **F. V1 vs V3**：

| Metric | V1 | V3 |
|---|---:|---:|
| Macro-F1 | 16.3% | **18.3%** (+2.0pp) |
| MAE | 0.942 | 0.989 |
| F1 class 1 / 5 | 0.0% / 0.0% | 0.0% / 0.0% |
| F1 class 2 | 9.1% | **27.1%** (+18pp) |
| F1 class 3 | 30.8% | 21.9% (-8.9pp) |
| F1 class 4 | 41.7% | 42.6% |
| Low-Novelty FP P(pred>=3\|gold<=2) | 96.0% | **70.7%** (-25.3pp) |
| High-Novelty FN P(pred<=3\|gold>=4) | 21.7% | **42.6%** (+20.9pp) |
| Abstention / Parse failure | 0% / 0% | 0% / 0% |
| Tokens/sample | 4160 | 4499 (+8%) |
| Latency ms/sample | 5215 | 6137 (+18%) |

- confusion：V1 histogram {2:6, 3:69, 4:202} → V3 {1:2, 2:58, 3:68, 4:149}；gold1/2→pred3/4 从 72 个减到 53 个；但 gold4/5→pred1/2 从 1 个增到 18 个

### 四个问题回答
1. gold 1/2 → pred 3/4 显著减少？**是**（72→53；Low-Novelty FP -25.3pp）——预注册假设的方向性成立
2. 是否引入新的 gold 4/5 → 1/2 误杀？**是，且严重**（1→18；High-Novelty FN 翻倍）——coverage-first 视角把"方向已有但机制实质不同"压成 2 分
3. 是否用大量 abstention 换 F1？**否**（abstention 0%）
4. cost 数量级增长？**否**（tokens +8%、latency +18%）

### 核心洞察
criterion-first principle 的效果强烈依赖任务形态：对 pairwise ranking（强制二选一）产生质变（0.14→0.71），对绝对 1-5 novelty 打分只是**把误差从"系统性高估（向 3/4 收缩）"换成"系统性低估（向 2/3 收缩）"**——两端类别（1/5）仍然零命中，calibration 仍是失败，只是失败方向反转。净收益 Macro-F1 +2pp 是两个方向相反的系统性偏差的残余。

## G. Guardrail
- Internal Research Validity Regression：**7/7 PASS**（tests/test_research_validity_regression.py）
- production `app/` 零 diff（git diff 确认）
- tests/eval 31 passed

## 停止点
按指令不启动：24-topic E2E、super_audit、human_verdicts、production triage 修改、ranking principle 迁移 production、V4。本轮两个问题均已回答，停止并汇报。
