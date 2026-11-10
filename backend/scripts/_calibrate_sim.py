"""Calibrate the semantic-recall similarity threshold on real pilot data.

Why measure instead of picking a number
---------------------------------------
The retrieval failure was NOT "no results" — it was "12 results, all off-topic".
A similarity filter only helps if there is a gap between
  (a) the claim vs its genuine nearest neighbours, and
  (b) the claim vs the keyword-match noise that currently fills the pool.
If those distributions overlap, filtering cannot separate them and the whole
approach must be rejected before it is wired into the instrument.

Ground truth available without any new annotation:
  * POSITIVE set — the pipeline's own audit neighbours for each gap (papers the
    audit actually reasoned about; for a confirmed gap these are its nearest
    prior art by construction).
  * NEGATIVE set — the off-topic pool the adjudicator actually retrieved for
    fbp-089 (12 statistical-mediation / medical-GNN papers, none about MoE).

Both are embedded against the gap's claim text. The script reports the two
distributions and the maximin separating threshold.
"""

import json
import sqlite3
import statistics
import sys

sys.path.insert(0, ".")

from app.services import embedding_service  # noqa: E402

DB = "deep_research.db"
MAP = r"eval_results\fixed_budget_pilot\review_mapping.json"
INP = r"eval_results\fixed_budget_pilot\adjudication_input.jsonl"
V2 = r"eval_results\fixed_budget_pilot\verdicts.pass2.jsonl"


def _claim_text(unit: dict) -> str:
    parts = [unit.get("description") or "", unit.get("observed_problem") or "",
             unit.get("missing_capability") or "",
             unit.get("claimed_delta") or ""]
    return " ".join(p.strip() for p in parts if p.strip())[:2000]


def main() -> None:
    units = {json.loads(l)["anon_id"]: json.loads(l)
             for l in open(INP, encoding="utf-8") if l.strip()}
    mapping = json.load(open(MAP, encoding="utf-8"))["units"]
    con = sqlite3.connect(DB)

    pos_sims, neg_sims = [], []

    # POSITIVE: claim vs the audit's own neighbours (real prior art for it).
    n_gap = 0
    for aid, unit in units.items():
        if unit["unit_type"] != "gap":
            continue
        meta = mapping[aid]
        row = con.execute(
            "SELECT neighbor_paper_ids_json FROM gap_audits WHERE gap_id=? "
            "ORDER BY created_at DESC LIMIT 1", (meta["real_id"],)).fetchone()
        if not row or not row[0]:
            continue
        ids = json.loads(row[0])[:5]
        if not ids:
            continue
        q = ",".join("?" * len(ids))
        papers = con.execute(
            f"SELECT title, abstract FROM papers WHERE id IN ({q})", ids).fetchall()
        papers = [(t or "", (a or "")[:400]) for t, a in papers if t]
        if not papers:
            continue
        n_gap += 1
        claim = _claim_text(unit)
        vecs = embedding_service.embed_texts(
            [claim] + [f"{t} {a}" for t, a in papers])
        for i in range(1, len(vecs)):
            pos_sims.append(embedding_service.cosine_similarity(vecs[0], vecs[i]))

    # NEGATIVE: the known off-topic pool actually retrieved for fbp-089.
    p2 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(V2, encoding="utf-8") if l.strip()}
    noise = p2["fbp-089"]["retrieved"]
    claim = _claim_text(units["fbp-089"])
    vecs = embedding_service.embed_texts(
        [claim] + [f"{h['title']} {(h.get('abstract') or '')[:400]}"
                   for h in noise])
    neg_sims = [embedding_service.cosine_similarity(vecs[0], vecs[i])
                for i in range(1, len(vecs))]

    def stat(name, xs):
        if not xs:
            print(f"  {name}: (none)")
            return
        qs = statistics.quantiles(xs, n=10) if len(xs) >= 10 else []
        print(f"  {name}: n={len(xs)} min={min(xs):.3f} "
              f"p10={qs[0]:.3f} median={statistics.median(xs):.3f} "
              f"p90={qs[-1]:.3f} max={max(xs):.3f}")

    print(f"positive gaps used: {n_gap}")
    print("claim vs its own audit neighbours (SHOULD pass the filter):")
    stat("pos", pos_sims)
    print("\nclaim vs the off-topic fbp-089 pool (SHOULD be filtered out):")
    stat("neg", neg_sims)

    if pos_sims and neg_sims:
        best, best_bal = None, -1
        for t in [i / 100 for i in range(20, 71)]:
            tpr = sum(1 for s in pos_sims if s >= t) / len(pos_sims)
            fpr = sum(1 for s in neg_sims if s >= t) / len(neg_sims)
            bal = tpr - fpr
            if bal > best_bal:
                best, best_bal = t, bal
        tpr = sum(1 for s in pos_sims if s >= best) / len(pos_sims)
        fpr = sum(1 for s in neg_sims if s >= best) / len(neg_sims)
        print(f"\nbest balanced threshold: {best:.2f}  "
              f"recall(neighbours kept)={tpr:.0%}  "
              f"noise admitted={fpr:.0%}  margin={best_bal:+.2f}")
        print("\nNOTE: recall on neighbours is the number that matters most — "
              "if it is low, a threshold filter would delete real prior art "
              "and must not be used as a hard gate.")

    con.close()


if __name__ == "__main__":
    main()
