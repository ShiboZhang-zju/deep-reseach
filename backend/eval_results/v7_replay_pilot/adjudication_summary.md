# v7 Replay Pilot — 裁决存档（2026-10-29，pass2 定稿）

## 最终裁决（pass2，联网验证复核后）

| 指标 | pass1（LLM 预填） | **pass2（LLM 联网复核，最终）** |
|---|---|---|
| true_gap (NONE) | 4 | **2**（v7r-023 `a1e42604`、v7r-015 `3cf7b490`，均 boundary_gap） |
| partial | 18 | **21** |
| false_open (FULL) | 8 | **7**（23.3%） |
| q2 边际机会精度 | 13.3% | **6.7%（2/30）** |
| 门槛 [1] ≥1 human-true | MET | **MET**（2） |
| 门槛 [2] 不以假 gap 为主 | MET | **MET**（7 vs 23） |
| 门槛 [3] true gap 产出可信 Idea | MET（via ea4b38e0） | **NOT MET**（0/2） |
| **GO/NO-GO** | 名义 GO | **NO-GO（按预注册冻结规则）** |

**pass1→pass2 的两处翻转**（均因联网检索到具体 prior art）：
- `ea4b38e0`（v7r-005）NONE→PARTIAL：arXiv 2607.28308 用冻结路由/有序前缀干预直接检验了后续专家贡献
- `64883f30`（v7r-019）NONE→PARTIAL：arXiv 2610.05076 对 TTT 长程适应做了因果分解

翻转方向全部是"高估新颖性→被覆盖"，印证 pass1 的 NONE 判定系统性偏乐观。

## 门槛 [3] NOT MET 的取证分解

两个幸存 true gap 均无关联 Idea，但原因不同：

1. **`3cf7b490`（v7r-015）**：审计 `inconclusive` / `recommended_action=more_search`——
   **从未通过审计进入下游**，无 phenomenon plan、无 intervention、无 Idea 属预期。
   这是 v7 自家审计没能确认一个（事后被证为）真机会的问题。
2. **`a1e42604`（v7r-023）**：审计 `confirmed` + `surviving`（mechanical_override），
   且有 **22 个 passed intervention（15 个 A 档）**——但 **0 个 Idea**。同任务
   （da3ff420）同窗口（09-21 10:15–10:23）的 Idea 全部落在 v6 对应 gap `25ad78d6` 上。
   由于 triage 落选簇也会被存为 `research_direction_only` Idea（挂同一 gap_id），
   零 Idea 说明 experiment/idea 阶段**从未处理过该 gap**——是 replay 多轮扫描下
   gap 选择/幂等快照的产物，**不是"该 gap 无可转化素材"**（A 档素材在库）。

## 协议状态（引用本结论时必须一并披露）

- 两轮判定均为 **同一 LLM 评审**（pass1 纯参数记忆；pass2 联网验证，notes 逐行标注
  `[LLM pass2 web-verified ...; NOT independent human]`）——**始终不是协议设计的独立人工盲评**。
- pass2 相对 pass1 更强（翻转均有具体论文支撑），但独立性缺陷未消除。
- pass1 的 4 条 NONE 中 2 条自标 `target_only` 低置信——幸存的 2 条恰是其中之一
  （v7r-023 仍带 conf=低 与"高效 TTT 工作存在"的保留意见）。

## 预注册裁决规则的适用

门槛为合取条件，[3] NOT MET → **NO-GO**：v7 opportunity ontology **不按现状整体采纳**。
6 主题 pilot 只决定 GO/NO-GO，不做 per-type 删除。数据给出的方向性信号（供下一轮实验参考，
非 pilot 结论）：

- 真机会全部落在 `boundary_gap`（2/4 true）；`scaling_regime` 假阳性最高（4/7 FULL、0 true）；
  `mechanism_gap` 的 2 条 pass1-true 均被 pass2 联网 prior art 翻转。
- v7 的下游存在"confirmed + surviving + A 档 intervention 却零 Idea"的转化断点
  （`a1e42604` 取证如上）——若后续重跑 v7 实验，先修 gap→experiment 的选择/幂等路径。

## 若要重启 v7 评估的最低路径

1. 独立人工盲评（材料仍在 `human_review_blind.md`；重点 2 NONE + 7 FULL 共 9 行）
2. 修复 replay 下游对 surviving gap 的 idea 转化覆盖
3. 按 README §0.2 既定计划做**固定预算**的 v6 vs v7 生产价值对照（已知瓶颈 #7）——
   full-jury replay 的 NO-GO 不等同于固定预算下 v7 无价值

## 复现

```bash
cd backend
python -m eval.production_e2e.v7_blind_review score --pilot-dir eval_results/v7_replay_pilot
```

- 判定数据：`v7_only_opportunities.adjudicated.jsonl`（pass2 判定已写回）
- 聚合工具：`eval/production_e2e/v7_blind_review.py`（q2/q4/三条件，idea 关联实时查库）
