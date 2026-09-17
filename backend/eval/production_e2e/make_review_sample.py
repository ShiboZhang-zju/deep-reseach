"""Build a calibration sample for the blind review.

Why a sample first
------------------
The full sheet is 91 submissions x ~10 candidates ≈ 900 adjudications. Before
spending that, calibrate on a small stratified set: it surfaces (a) whether the
snippets are readable enough to judge from, (b) whether two passes of the same
reviewer agree with themselves, and (c) whether the disclosure is sufficient to
handle noisy candidate lists.

Selection is stratified by system so all three arms are represented, and within a
system it prefers submissions that exercise the hard cases — ones with quoted
passages (easier to verify) alongside ones without (pure title judgement), and at
least one whose candidate list is mostly keyword noise.

Output is a standalone Markdown file, so the reviewer works from ONE artifact and
the full sheet stays untouched as the record of what was actually shown.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Source names can end up in the `venue` field when a paper has no real venue, so
# scrubbing source names is necessary even though the field is named "venue".
_SOURCE_NAMES = {
    "local_corpus", "openalex", "semantic_scholar", "arxiv",
    "citation_snowball", "unknown",
}


def _safe_venue(venue) -> str:
    """Venue for display, or "" when it is really a retrieval-source name."""
    text = (venue or "").strip()
    if not text or text.lower() in _SOURCE_NAMES:
        return ""
    return text


def _load(audit_dir: Path) -> list[dict]:
    rows = []
    for line in (audit_dir / "candidate_killer_papers.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _noise_ratio(rec: dict) -> float:
    """Fraction of candidates with no quoted passage (title-only judgement).

    Used only to ensure the sample includes the difficult case; it is NOT a
    relevance measure (that is what the reviewer is there to judge).
    """
    cands = rec.get("candidate_papers") or []
    if not cands:
        return 1.0
    without = sum(1 for c in cands if not c.get("snippet"))
    return without / len(cands)


def build_sample(records: list[dict], per_system: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_sys: dict[str, list[dict]] = {}
    for r in records:
        if r.get("target_type") != "idea":
            continue                      # ideas are what the headline scores
        by_sys.setdefault(str(r.get("system")), []).append(r)

    chosen: list[dict] = []
    for system in ("direct_llm", "retrieval_llm", "full_v2"):
        pool = by_sys.get(system) or []
        if not pool:
            continue
        # One submission that HAS quoted evidence, and the rest biased toward the
        # harder title-only case, so the calibration covers both extremes.
        with_quotes = [r for r in pool
                       if any(c.get("snippet") for c in (r.get("candidate_papers") or []))]
        picked: list[dict] = []
        if with_quotes:
            picked.append(rng.choice(with_quotes))
        rest = [r for r in pool if r not in picked]
        rest.sort(key=_noise_ratio, reverse=True)     # hardest first
        picked.extend(rest[:max(0, per_system - len(picked))])
        chosen.extend(picked)

    rng.shuffle(chosen)                   # mixing arms avoids order priming
    return chosen


def render(sample: list[dict], mapping: dict, out_path: Path) -> None:
    by_sid = {s["submission_id"]: s for s in mapping["submissions"]}
    lines = [
        "# Blind review — CALIBRATION SAMPLE",
        "",
        f"{len(sample)} submissions, drawn from all three systems and shuffled.",
        "",
        "Purpose: check the review PROCEDURE before committing to all 91. After "
        "this pass, answer:",
        "",
        "1. Could you decide FULL / PARTIAL / NONE for each candidate, or was the "
        "evidence insufficient?",
        "2. Did any snippet mislead you (bibliography, table, unrelated text)?",
        "3. Did the candidate-list quality line change how much you trusted a "
        "`true_gap` verdict?",
        "",
        "Record answers in `calibration_notes.md` beside this file.",
        "",
        "## How to fill `human_verdicts.jsonl`",
        "",
        "One row per submission, keyed by `submission_id` (do NOT add "
        "system/topic — they are restored automatically at scoring time):",
        "",
        "```json",
        '{"submission_id": "…", "verdict": "false_open", '
        '"idea_credible": true, "novelty": 3, "feasibility": 4, "notes": "…"}',
        "```",
        "",
        "- `verdict`: `\"false_open\"` if any candidate ALREADY covers the claimed "
        "contribution; `\"true_gap\"` if the contribution stands.",
        "- `idea_credible`: true only if the idea is both non-redundant AND "
        "testable as stated.",
        "- `novelty` / `feasibility`: 1-5.",
        "- `notes`: REQUIRED when the candidate list was mostly noise, since an "
        "all-noise list means the audit failed to find prior art — a different "
        "fact from `no prior art exists`.",
        "",
        "---",
        "",
        "## Submissions",
        "",
    ]
    for rec in sample:
        meta = by_sid.get(rec["submission_id"], {})
        cands = rec.get("candidate_papers") or []
        local_n = sum(1 for c in cands if c.get("source") == "local_corpus")
        snip_n = sum(1 for c in cands if c.get("snippet"))
        lines += [
            f"### {rec['submission_id']}",
            "",
            f"Topic: {rec.get('_topic') or meta.get('topic_id') or ''}",
            "",
            f"**Claim under audit:** {rec.get('claim') or ''}",
            "",
            f"*Candidate list: {len(cands)} items, {local_n} relevance-gated, "
            f"{snip_n} with a quoted passage. The rest are unfiltered keyword "
            f"matches — judge their titles critically.*",
            "",
        ]
        if rec.get("query_failure"):
            lines += [f"> RETRIEVAL FAILED: {rec['query_failure']}. An empty list "
                      f"here is NOT absence of prior art.", ""]
        for i, c in enumerate(cands, 1):
            year = c.get("year") or "n.d."
            venue = _safe_venue(c.get("venue"))
            suffix = f", {venue}" if venue else ""
            # Never print the retrieval source. Two distinct paths leaked it:
            # falling back to `c["source"]`, and `venue` itself carrying the
            # source name because the source is written where a venue belongs
            # when no real venue exists. Either way the sheet reveals the
            # retrieval path, and since local-corpus hits are the ones with
            # quoted passages it also reveals which arm is being read.
            lines.append(f"{i}. **{c.get('title') or ''}** ({year}{suffix})")
            if c.get("snippet"):
                where = c.get("section") or "full text"
                lines.append(f"   - matched passage ({where}): "
                             f"\"{' '.join(str(c['snippet']).split())}\"")
                lines.append("   - NOTE: a passage can match on shared vocabulary "
                             "while being about something else entirely (a survey "
                             "mentioning a topic is not prior art for a specific "
                             "mechanism). Judge the CANDIDATE, not the passage.")
            lines.append("   - verdict: FULL / PARTIAL / NONE")
        lines += ["", "---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sample] wrote {out_path} ({len(sample)} submissions)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a blind-review calibration sample")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--per-system", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    audit_dir = Path(args.audit_dir)
    records = _load(audit_dir)
    mapping = json.loads((audit_dir / "submission_mapping.json").read_text(
        encoding="utf-8"))
    sample = build_sample(records, args.per_system, args.seed)
    out = Path(args.out) if args.out else audit_dir / "human_review_calibration.md"
    render(sample, mapping, out)


if __name__ == "__main__":
    main()
