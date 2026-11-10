# 固定预算 pilot —— 次指标结果（2026-11-02）

> **重要**：本文件只记录**次指标**（终态、executable 数、gap 类型覆盖、候选积压）。
> 预注册的**主指标**（盲评机会真率，设计 §6/§7）**尚未执行**——必须先完成 §8 匿名化导出
> + LLM 联网主评 + 人工抽查，才能按冻结规则判 GO / NO-GO / 平局。
> **在看到本文件时不应对 v7.2 与 v6 的优劣下结论。**
>
> 生成方式：`python scripts/_exp_summary.py`（只读 DB + run_matrix.json）。

## 1. 运行完整性

| 检查 | 结果 |
|---|---|
| 12 次运行全部完成 | ✅ 6 主题 × 2 臂 |
| 臂 stamp 正确（A=v7.2 / B=v6-exp） | ✅ 12/12 |
| gap 类型未越界（B 只 2 类，A ≤7 类） | ✅ 12/12 |
| 无工程性故障（设计 §7.5） | ✅ 12/12 终态合法 |
| 预算参数两臂一致 | ✅ 仅 `GAP_MINING_LEGACY_V6_MODE` 不同 |

工具：`scripts/_exp_verify.py`。

## 2. 逐次运行

| topic | stratum | 臂 | 终态 | 论文 | gaps | 已审 | 仍candidate | 类型 | exec | 方向 | cond |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pe2e-001 | narrow_mature | A | waiting_for_user_review | 270 | 9 | 2 | 7 | 6 | 1 | 3 | 0 |
| pe2e-001 | narrow_mature | B | more_research_required | 254 | 5 | 4 | 1 | 2 | 0 | 0 | 0 |
| pe2e-007 | narrow_mature | A | waiting_for_user_review | 204 | 12 | 2 | 10 | 7 | 1 | 9 | 0 |
| pe2e-007 | narrow_mature | B | abstained | 303 | 9 | 6 | 3 | 2 | 0 | 3 | 2 |
| pe2e-013 | emerging_sparse | A | waiting_for_user_review | 315 | 8 | 2 | 6 | 5 | 2 | 6 | 0 |
| pe2e-013 | emerging_sparse | B | more_research_required | 379 | 5 | 3 | 2 | 2 | 0 | 1 | 0 |
| pe2e-015 | emerging_sparse | A | waiting_for_user_review | 325 | 11 | 2 | 9 | 7 | 2 | 4 | 0 |
| pe2e-015 | emerging_sparse | B | more_research_required | 415 | 6 | 5 | 1 | 2 | 0 | 0 | 0 |
| pe2e-019 | broad_ambiguous | A | waiting_for_user_review | 350 | 11 | 2 | 9 | 6 | 2 | 3 | 0 |
| pe2e-019 | broad_ambiguous | B | **waiting_for_user_review** | 152 | 3 | 2 | 1 | 2 | **2** | 3 | 0 |
| pe2e-024 | broad_ambiguous | A | waiting_for_user_review | 303 | 10 | 2 | 8 | 6 | 1 | 5 | 1 |
| pe2e-024 | broad_ambiguous | B | more_research_required | 356 | 6 | 5 | 1 | 2 | 0 | 3 | 0 |

## 3. 臂汇总

| 指标 | 臂 A（v7.2） | 臂 B（v6） |
|---|---:|---:|
| ideas_ready | **6/6** | 1/6 |
| executable_candidate | **9** | 2 |
| research_direction_only | 30 | 10 |
| conditional_review | 1 | 2 |
| gap 候选总数 | 61 | 34 |
| 已审计 gap | **12** | **25** |
| 仍 `candidate` | 49（80.3%） | 9（26.5%） |
| 论文入库 | 1767 | 1859 |
| 平均 gap 类型/topic | **6.17** | 2.00（结构上限） |

## 4. 必须同时报告的两个反面事实

### 4.1 臂 A 的 executable 产出率不能直接与臂 B 比数量

臂 A 每主题只审计 2 个 gap（Top-2 预算），臂 B 审计 4–6 个（候选更少、竞争更弱，几乎全审）：

- 臂 A：9 个 executable / 12 个已审 gap
- 臂 B：2 个 executable / 25 个已审 gap

按**单位审计产出**看臂 A 远高；但这恰恰也可能是"审计太少、挑得精"的假象，
也可能是"v7.2 机会质量更高"。**这正是主指标（盲评真率）要区分的东西**，
次指标无法裁决。任何仅凭 `9 vs 2` 的结论都不成立。

### 4.2 饥饿机制在臂 A 未达设计目标（新发现，需修）

设计目标（README 第四轮修复验收）：`candidate` 占比 **<50%**。实测：

- 臂 A：**80.3%**（49/61）——未达标
- 臂 B：26.5%（9/34）——达标

原因分析（可验证）：臂 A 每轮 mined 8–12 个候选，但 budgeted 审计每轮只放行 2 个，
且每主题只跑 1–2 轮竞争。`gap_starvation_deferral_limit=3` 需要 3 轮竞争才退休，
而多数臂 A 任务在候选积累到 2 轮时就已终止（成功产出 Idea 后不再竞争）。
**即：饥饿机制的设计前提（多轮竞争）在 v7.2 的实际运行路径上不成立。**

这不是"机制写错了"，而是"触发条件没被满足"——但它意味着
**80% 的候选仍然以 `candidate` 状态沉默积压**，本轮修复对该指标无效。

> 这条也是次指标，但它与主指标正交（是产能/效率问题，不是质量问题），
> 可以在判决前单独修。

## 5. 下一步（按预注册顺序）

1. ✅ 运行完成、完整性核查
2. ⬜ **§8.1 匿名化导出**：从这 12 次运行导出 gap/idea 单元，剔除臂标识、乱序、mapping 封存
3. ⬜ **§8.2 LLM 联网主评**：沿用 v7 replay pilot pass2 口径（NONE/PARTIAL/FULL）
4. ⬜ **§8.2 人工抽查 ≥20%**（需用户参与；一致率 <70% 则主评作废）
5. ⬜ 解封 mapping，按 §7 冻结规则判定
