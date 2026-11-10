# v7 Replay Pilot — 裁决存档（2026-10-29）

## 裁决结果

| 指标 | 值 |
|---|---|
| 裁决覆盖 | **30/30 judged**（完整） |
| q2 边际机会精度 | true_gap (NONE) **4/30 = 13.3%**；partial 18；false_open (FULL) 8（26.7%） |
| q3 边际可信 Idea 产出 | 4 个 human-true gap 中 **1 个**（`ea4b38e0`，mechanism_gap）产出 4 个 `executable_candidate` Idea（0.738–0.778；全部被后续重跑代 superseded——证明该 gap 能产出过闸 Idea，但非当前 active 代） |
| GO/NO-GO 门槛 | **[1] MET**（4 个 NONE）· **[2] MET**（false_open 8 vs non-false 22）· **[3] MET**（1 个 human-true gap 有过闸 Idea）→ **名义 GO** |

## 分类型（仅裁决行）

| gap_type | 裁决 | true | partial | false_open |
|---|---:|---:|---:|---:|
| boundary_gap | 4 | 2 | 2 | 0 |
| contradiction | 4 | 0 | 3 | 1 |
| cross_domain_transfer | 4 | 0 | 3 | 1 |
| mechanism_gap | 5 | 2 | 3 | 0 |
| missing_evaluation | 3 | 0 | 3 | 0 |
| scaling_regime | 7 | 0 | 2 | **5** |
| untested_assumption | 3 | 0 | 2 | 1 |

- 真机会集中在 `boundary_gap` / `mechanism_gap`（各 2 个 true，且零 false_open）。
- `scaling_regime` 是最大的假阳性来源：7 个裁决 5 个 FULL、0 个 true——"某尺度区间没人研究"类主张最容易被已有工作覆盖。
- 其余类型多为 partial（主张部分成立，可收窄）。

## 协议偏差（必须在引用本结论时一并披露）

1. **裁决方式**：本轮 `human_verdicts.jsonl` 为 **LLM 预填 + 用户确认合并**（notes 中已逐行标注
   `[LLM-prefill, user-approved merge 2026-10-29; NOT independent human blind review]`），
   不是协议设计的独立人工盲评。盲评材料（`human_review_blind.md`）本身满足匿名化与
   信息隐藏要求，但裁决主体不满足独立性。
2. **2/4 的 true 判定被评审自己标注低置信**（`target_only` / "NONE 不代表可信缺口"）：
   `a1e42604`（v7r-023）、`3cf7b490`（v7r-015）是纯性能目标型主张（"提升 X%"），
   评审明确注明这类"无先例但仅为性能目标"不构成可信缺口。高置信的 true 判定是
   `64883f30`（v7r-019，conf=低-中）与 `ea4b38e0`（v7r-005，conf=低）。
3. **判定前信息污染**：v7r-008、v7r-028 在判定前接触过 `mech_override_review.llm_verdicts.jsonl`
   的同主题条目（标 `possible_contamination`）——两条恰为 FULL 判定，污染方向是
   高估覆盖（对 GO 不利方向的偏差，不威胁名义 GO，但复核时应优先重看）。
4. 2025–2026 文献覆盖不完整，NONE/PARTIAL 判定偏向高估新颖性。
5. 条件 [3] 的 4 个过闸 Idea 全部处于 `superseded` 状态（被后续重跑代取代）——
   证明"该 gap 产出过过闸 Idea"，但当前 active 代没有挂在它名下的 Idea。

## 冻结 GO 前的最低补救（建议）

对 **12 行**做独立人工复核即可消除偏差 1：4 个 NONE + 8 个 FULL（PARTIAL 不影响门槛
方向）。复核后重跑：

```bash
cd backend
python -m eval.production_e2e.v7_blind_review score --pilot-dir eval_results/v7_replay_pilot
```

若 12 行复核后仍有 ≥1 个高置信 NONE 且对应 gap 有过闸 Idea、且 false_open 不占多数，
则 GO 可冻结；否则按预注册裁决规则处理（不加机制，先修检索/准入）。

## 复现

- 判定数据：`v7_only_opportunities.adjudicated.jsonl`（判定已写回每行）
- 聚合工具：`eval/production_e2e/v7_blind_review.py score`（q2/q4/三条件全量重算，
  idea 关联实时查库）
- 盲评材料与映射：`human_review_blind.md` / `review_mapping.json`
