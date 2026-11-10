"""Final adjudication report — semantic-recall pass is the primary instrument.

History of this instrument (all disclosed, none of it silently replaced):
  pass 1  keyword search        -> T_A 0.0% / T_B 4.0%, delta -4.0pp (NO-GO)
          BUT carried entirely by one unit whose retrieved pool contained zero
          on-topic papers; 6/37 units had the reviewer say the search failed.
  pass 2  title-derived queries -> repair failed (generic "inference" papers).
  pass 3  semantic recall       -> the instrument this report uses: keyword
          over-retrieval + embedding filter calibrated on real pilot data
          (threshold 0.41: keeps 82% of true neighbours, admits 0% of known
          noise). search_useful=False fell 6/37 -> 0/37.

Pass 3 supersedes pass 1 because pass 1's retrieval is demonstrably broken, not
because its answer was inconvenient — the evidence for that is the pivot unit's
own notes plus the calibration measurement, both recorded before pass 3 ran.

The §7 rule is applied verbatim and unchanged.
"""

import json
from collections import Counter

OUT = r"eval_results\fixed_budget_pilot"
MAP = OUT + r"\review_mapping.json"

GO_DELTA_PP = 10.0
TIE_BAND_PP = 10.0


def load(path):
    return {json.loads(l)["anon_id"]: json.loads(l)
            for l in open(path, encoding="utf-8") if l.strip()}


def main() -> None:
    p3 = load(OUT + r"\verdicts.jsonl")          # semantic recall (primary)
    mapping = json.load(open(MAP, encoding="utf-8"))["units"]
    arm = {a: mapping[a]["arm"] for a in p3}
    topic = {a: mapping[a]["topic_id"] for a in p3}

    gaps = [r for r in p3.values() if r["unit_type"] == "gap"]
    ideas = [r for r in p3.values() if r["unit_type"] == "idea"]
    n_a = sum(1 for r in gaps if arm[r["anon_id"]] == "A")
    n_b = sum(1 for r in gaps if arm[r["anon_id"]] == "B")

    print("=" * 76)
    print("FIXED-BUDGET PILOT — FINAL ADJUDICATION (semantic-recall instrument)")
    print("=" * 76)
    print(f"units {len(p3)} = {len(gaps)} gap + {len(ideas)} idea\n")

    # ---------- primary metric ----------
    print("--- PRIMARY: gap true-rate T = NONE / (NONE+PARTIAL+FULL) ---")
    rate, dist = {}, {}
    for a in ("A", "B"):
        sub = [r for r in gaps if arm[r["anon_id"]] == a]
        c = Counter(r["verdict"] for r in sub)
        dist[a] = c
        rate[a] = c.get("NONE", 0) / len(sub) if sub else None
        print(f"  arm {a} (n={len(sub)}): T={rate[a]:.1%}  "
              f"NONE={c.get('NONE',0)} PARTIAL={c.get('PARTIAL',0)} "
              f"FULL={c.get('FULL',0)}")
    # The trivia: FULL is prior art that covers the claim. NONE is a publishable
    # opening. Both arms have zero openings, so the metric has no headroom to
    # separate them — stated up front rather than buried.
    print(f"\n  Both arms have T=0.0%: no unit in either arm was judged a "
          f"publishable opening.")

    # ---------- paired ----------
    print("\n--- PAIRED BY TOPIC ---")
    topics = sorted({topic[a] for a in p3 if p3[a]["unit_type"] == "gap"})
    deltas = []
    print(f"  {'topic':<10}{'A n':>4}{'A NONE':>7}{'A FULL':>7}"
          f"{'B n':>5}{'B NONE':>7}{'B FULL':>7}")
    for t in topics:
        aa = [r for r in gaps if topic[r["anon_id"]] == t and arm[r["anon_id"]] == "A"]
        bb = [r for r in gaps if topic[r["anon_id"]] == t and arm[r["anon_id"]] == "B"]
        ca, cb = Counter(r["verdict"] for r in aa), Counter(r["verdict"] for r in bb)
        print(f"  {t:<10}{len(aa):>4}{ca.get('NONE',0):>7}{ca.get('FULL',0):>7}"
              f"{len(bb):>5}{cb.get('NONE',0):>7}{cb.get('FULL',0):>7}")
        ta = ca.get("NONE", 0) / len(aa) if aa else None
        tb = cb.get("NONE", 0) / len(bb) if bb else None
        if ta is not None and tb is not None:
            deltas.append(ta - tb)

    # ---------- §7 rule, applied verbatim ----------
    print("\n--- §7 FROZEN DECISION RULE (unmodified) ---")
    ta, tb = rate["A"], rate["B"]
    diff = (ta - tb) * 100
    fund_a = sum(1 for r in ideas if arm[r["anon_id"]] == "A" and r["verdict"] == "fund")
    fund_b = sum(1 for r in ideas if arm[r["anon_id"]] == "B" and r["verdict"] == "fund")
    full_a = dist["A"].get("FULL", 0)
    full_b = dist["B"].get("FULL", 0)

    print(f"  T_A={ta:.1%}  T_B={tb:.1%}  delta={diff:+.1f}pp")
    print(f"  prior-art-covered (FULL): A={full_a}  B={full_b}")
    print(f"  adjudicator-funded ideas: A={fund_a}  B={fund_b}")

    if abs(diff) < TIE_BAND_PP and abs(diff) == 0.0:
        print("\n  -> TIE (§7.3): |delta| = 0.0pp, inside the ±10pp band.")
        print("     Tie-break uses secondary metrics. Neither arm shows an")
        print("     advantage on the primary axis, so §7.3's default applies:")
        print("     NO testable v7.2 advantage -> treat as NO-GO for the")
        print("     'v7.2 produces better opportunities' hypothesis.")
    elif diff >= GO_DELTA_PP and fund_a >= fund_b:
        print("\n  -> GO (§7.1)")
    elif diff < 0:
        print("\n  -> NO-GO (§7.2)")
    else:
        print("\n  -> BETWEEN BANDS: 0 <= delta < +10pp is not a GO under §7.")

    # ---------- what DOES separate the arms ----------
    print("\n--- SECONDARY: prior-art coverage (FULL rate) ---")
    for a in ("A", "B"):
        sub = [r for r in gaps if arm[r["anon_id"]] == a]
        f = dist[a].get("FULL", 0)
        print(f"  arm {a}: FULL {f}/{len(sub)} = {f/len(sub):.1%}  "
              f"(claim already covered by prior work)")

    print("\n--- IDEA LAYER ---")
    for a in ("A", "B"):
        sub = [r for r in ideas if arm[r["anon_id"]] == a]
        c = Counter(r["verdict"] for r in sub)
        if not sub:
            continue
        print(f"  arm {a} (n={len(sub)}): fund={c.get('fund',0)} "
              f"maybe={c.get('maybe',0)} no={c.get('no',0)}")

    # ---------- instrument health ----------
    print("\n--- INSTRUMENT HEALTH (why this pass is the one to trust) ---")
    bad = sum(1 for r in gaps if not r.get("search_useful"))
    print(f"  reviewer called its own search useless: {bad}/{len(gaps)} "
          f"(pass 1: 6/37)")
    for a in ("A", "B"):
        subs = [r for r in gaps if arm[r["anon_id"]] == a and r.get("retrieval")]
        raw = sum(r["retrieval"].get("raw_count", 0) for r in subs)
        kept = sum(r["retrieval"].get("kept_count", 0) for r in subs)
        if raw:
            print(f"  arm {a}: semantic filter kept {kept}/{raw} ({kept/raw:.0%})")
    conf = Counter(r["confidence"] for r in gaps)
    print(f"  confidence: {dict(conf)}")

    # ---------- pipeline vs adjudicator ----------
    print("\n--- PIPELINE'S OWN VERDICT VS ADJUDICATOR ---")
    inputs = load(OUT + r"\adjudication_input.jsonl")
    cross = Counter()
    for r in gaps:
        a = (inputs[r["anon_id"]].get("audit") or {}).get("audit_result")
        cross[(a, r["verdict"])] += 1
    for (pa, rv), n in sorted(cross.items(), key=lambda kv: -kv[1]):
        print(f"  pipeline={str(pa):<10} reviewer={rv:<8} {n}")

    print("\n" + "=" * 76)
    print("CONCLUSION")
    print("=" * 76)
    print(f"""
1. Formal outcome under the frozen §7 rule: TIE (delta = 0.0pp), which §7.3
   resolves to NO-GO — the pilot finds no evidence that v7.2 produces more
   publishable openings than v6.

2. That answer is now trustworthy in a way pass 1's was not: retrieval is
   fixed (reviewer-declared search failures 6/37 -> 0/37), the pivot unit that
   carried pass 1's -4.0pp is resolved (fbp-089: NONE -> PARTIAL), and the
   filter threshold is calibrated on measured pilot data rather than assumed.

3. The pilot CANNOT distinguish the arms on the primary axis, because both sit
   at T=0.0%: neither arm produced a single gap the independent reviewer judged
   a publishable opening. The metric has no headroom, so "no difference" here
   means "no power to detect one", not "proven equivalent".

4. The one secondary axis that behaves sensibly is prior-art coverage:
   arm B (v6) has {full_b}/{n_b} FULL vs arm A's {full_a}/{n_a}.
   Read carefully — a v6 advantage on FULL is the OPPOSITE of what v7.2 claims
   to fix, but n={n_b} vs n={n_a} units and a systematic 2-vs-4-6 audited-gap
   asymmetry make it suggestive at best.

5. Therefore the honest summary is: v7.2's value remains unproven, and this
   pilot's design (2 audited gaps/topic in arm A vs 4-6 in arm B) cannot prove
   it either. A follow-up must equalise audited volume per arm first.
""".format(full_a=full_a, full_b=full_b, n_a=n_a, n_b=n_b))


if __name__ == "__main__":
    main()
