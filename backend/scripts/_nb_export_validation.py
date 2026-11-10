"""Export the out-of-sample validation units (pe2e-003, pe2e-008) for blind review.

Why a separate export
---------------------
These two topics were never used to derive the conjunction rule or the 46%
PARTIAL figure, so they are the only evidence that can speak to whether the
mining-prompt change helped. They are exported with the SAME blindness
discipline as the main run (no arm, no task id, no audit prose, no pipeline
verdict, paper identity scrubbed) into their OWN files so the main
adjudication artefacts are not disturbed.

Only gaps that reached an audit (`surviving` / `confirmed_runner_up` /
`inconclusive` / `rejected`) are exported: `candidate` gaps have no verdict.

Output:
  validation_units.jsonl   reviewer-facing, blind
  validation_key.json      SEALED mapping anon_id -> (topic, gap_id)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_nbe", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_nb_export_adjudication.py"))
_nbe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nbe)

OUT_DIR = os.path.join("eval_results", "neighbor_budget")
SEED = 20261109
PREFIX = "val-"

TOPICS = {
    "pe2e-003": "c5fcae2d-5087-4f50-91e6-b5b20e8b7113",
    "pe2e-008": "bfc0c0d5-2afd-4a90-a32b-3659f702d5e6",
}
AUDITED = {"surviving", "confirmed_runner_up", "inconclusive", "rejected"}


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def main() -> int:
    con = sqlite3.connect("file:deep_research.db?mode=ro", uri=True)
    units = []
    try:
        ph = ",".join("?" * len(AUDITED))
        for topic, tid in TOPICS.items():
            for row in con.execute(f"""
                SELECT id, gap_type, description, target_setting,
                       observed_problem, existing_coverage, missing_capability,
                       claimed_delta, testable_hypothesis,
                       falsification_condition, provenance_status, residual_gap
                FROM gap_candidates
                WHERE task_id=? AND status IN ({ph})
                ORDER BY created_at
            """, (tid, *sorted(AUDITED))).fetchall():
                u = _nbe._gap_unit(con, row, topic, "out_of_sample")
                u["_topic"] = topic
                units.append(u)
    finally:
        con.close()

    counts = ", ".join(
        "{}={}".format(t, sum(1 for u in units if u["_topic"] == t)) for t in TOPICS)
    print(f"collected {len(units)} audited gap units ({counts})")

    units.sort(key=lambda u: _h("gap", u["_gap_id"]))
    random.Random(SEED).shuffle(units)
    for i, u in enumerate(units, start=1):
        u["anon_id"] = f"{PREFIX}{i:03d}"

    clean = [{k: v for k, v in u.items() if not k.startswith("_")} for u in units]
    hits = _nbe._leak_check(clean)
    print(f"leak check: {'CLEAN' if not hits else 'LEAKED -> ' + str(hits)}")
    if hits:
        raise SystemExit("refusing to write a leaking validation file")

    inp = os.path.join(OUT_DIR, "validation_units.jsonl")
    with open(inp, "w", encoding="utf-8") as fh:
        for rec in clean:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    key = os.path.join(OUT_DIR, "validation_key.json")
    with open(key, "w", encoding="utf-8") as fh:
        json.dump({
            "warning": "SEALED until verdicts are written.",
            "note": ("Out-of-sample topics: never used to derive the conjunction "
                     "rule or the 46% PARTIAL figure."),
            "units": {u["anon_id"]: {"topic_id": u["_topic"], "gap_id": u["_gap_id"]}
                      for u in units},
        }, fh, ensure_ascii=False, indent=2)

    print(f"written -> {inp}")
    print(f"written -> {key} (SEALED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
