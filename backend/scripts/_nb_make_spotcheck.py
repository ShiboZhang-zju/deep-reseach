"""Render the stratified spot-check sheet for the neighbour-budget adjudication.

Why a spot check exists at all
------------------------------
The adjudication's primary metric is a reviewer's binary-ish judgement
(NONE / PARTIAL / FULL). That instrument is only as good as its agreement with
a careful human. The fixed-budget pilot left its own 18/91 spot check undone;
this experiment does not repeat that. A >=20% stratified sample of the SAME
anonymised units the reviewer will see is rendered to a sheet, judged by hand
against the source evidence, and the agreement rate is reported alongside the
main result.

What this sheet deliberately shows and hides
--------------------------------------------
Shows: the anonymous unit exactly as the reviewer receives it.
Hides: arm, topic id, task id, the pipeline's own verdict.

The arm of each sampled unit is written to a SEPARATE sealed file
(`spotcheck_key.json`) so the sheet itself stays blind. Reading the key before
filling in a judgement breaks the point of the check.

Usage:
    python scripts/_nb_make_spotcheck.py
"""

from __future__ import annotations

import json
import os
import textwrap

OUT_DIR = os.path.join("eval_results", "neighbor_budget")
INPUT = os.path.join(OUT_DIR, "adjudication_input.jsonl")
IDS = os.path.join(OUT_DIR, "spotcheck_ids.json")
MAPPING = os.path.join(OUT_DIR, "review_mapping.json")
SHEET = os.path.join(OUT_DIR, "human_spotcheck.md")
KEY = os.path.join(OUT_DIR, "spotcheck_key.json")


def _wrap(text, width=100, indent="    "):
    if not text:
        return f"{indent}(empty)"
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def main() -> int:
    with open(INPUT, encoding="utf-8") as fh:
        units = {json.loads(line)["anon_id"]: json.loads(line) for line in fh}
    with open(IDS, encoding="utf-8") as fh:
        ids = json.load(fh)
    with open(MAPPING, encoding="utf-8") as fh:
        mapping = json.load(fh)["units"]

    lines = [
        "# 近邻预算实验 — 人工抽查表（盲）",
        "",
        "**在做完判断前不要打开 `spotcheck_key.json`。**",
        "",
        "对每个单元独立判断，只依据下方呈现的单位内容与证据，不要推测它来自哪一组。",
        "",
        "判断口径（与独立评审一致）：",
        "",
        "- `NONE` — 存在可发表空白，未见先行工作覆盖该 delta",
        "- `PARTIAL` — 有先行工作部分覆盖，但该 delta 的特定操作化未被直接报告",
        "- `FULL` — 该 delta 已被先行工作实质覆盖",
        "- `UNCERTAIN` — 现有材料不足以判断",
        "",
        f"共 {len(ids)} 个单元（分层抽样 ≥20%）。",
        "",
        "---",
        "",
    ]

    for n, anon in enumerate(ids, start=1):
        u = units[anon]
        lines.append(f"## {n}. `{anon}` — {u['unit_type']}")
        lines.append("")
        if u["unit_type"] == "gap":
            lines.append(f"- 层：{u.get('topic_stratum')}")
            lines.append(f"- gap_type：{u.get('gap_type')}")
            lines.append("")
            lines.append("**描述**")
            lines.append(_wrap(u.get("description")))
            lines.append("")
            for label, key in (("目标设置", "target_setting"),
                               ("观察到的问题", "observed_problem"),
                               ("已有覆盖", "existing_coverage"),
                               ("缺失能力", "missing_capability"),
                               ("声称的 delta", "claimed_delta"),
                               ("可检验假设", "testable_hypothesis"),
                               ("证伪条件", "falsification_condition")):
                val = u.get(key)
                if val:
                    lines.append(f"**{label}**")
                    lines.append(_wrap(val))
                    lines.append("")
            ev = u.get("evidence") or []
            if ev:
                lines.append(f"**证据（{len(ev)} 条，论文身份已隐藏）**")
                for e in ev:
                    lines.append(f"- [{e.get('evidence_type')}] "
                                 f"{(e.get('claim') or '')[:300]}")
                lines.append("")
        else:
            lines.append(f"- 层：{u.get('topic_stratum')}")
            lines.append("")
            lines.append(f"**标题**：{u.get('title')}")
            lines.append("")
            for label, key in (("描述", "description"),
                               ("动机", "motivation"),
                               ("方法草图", "method_sketch"),
                               ("预期贡献", "expected_contribution")):
                val = u.get(key)
                if val:
                    lines.append(f"**{label}**")
                    lines.append(_wrap(val))
                    lines.append("")

        lines.append("**你的判断**：`[ ] NONE  [ ] PARTIAL  [ ] FULL  [ ] UNCERTAIN`")
        lines.append("")
        lines.append("**理由（必填，一句话）**：")
        lines.append("")
        lines.append("    ")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(SHEET, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(KEY, "w", encoding="utf-8") as fh:
        json.dump({
            "warning": "SEALED. Do not open until the sheet is fully judged.",
            "sampled": len(ids),
            "units": {a: {"arm": mapping[a]["arm"],
                          "topic_id": mapping[a]["topic_id"],
                          "unit_type": mapping[a]["unit_type"]} for a in ids},
        }, fh, ensure_ascii=False, indent=2)

    print(f"sheet  -> {SHEET}  ({len(ids)} units)")
    print(f"key    -> {KEY}  (SEALED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
