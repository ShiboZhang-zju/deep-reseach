"""Build the pre-registered calibration probes for rubric v2.

Two families, fixed BEFORE any v2 run (see RUBRIC_V2_PREREG.md §3):

  FORWARD probes (expect NONE)
      The delta is untested, but the topic is busy with adjacent work. These are
      the exact case v1 could not express: adjacency present, delta not
      established. If v2 still answers PARTIAL here, the fix failed (gate G1).

  REVERSE probes (expect FULL)
      Classic results whose delta IS established by a canonical paper. If v2
      answers NONE here, it has degenerated into a NONE-everywhere instrument.
      These are the anti-degeneracy floor (gate G2).

Every probe carries `expected_verdict` in a SEALED side file, not in the
reviewer-facing record, so the adjudicator cannot see the answer.

Usage:  python scripts/_nb_make_v2_probes.py
"""

from __future__ import annotations

import json
import os

OUT_DIR = os.path.join("eval_results", "neighbor_budget")
UNITS = os.path.join(OUT_DIR, "rubric_v2_probes.jsonl")
KEY = os.path.join(OUT_DIR, "rubric_v2_probes_key.json")

PROBES = [
    # ---------------- FORWARD: expected NONE ----------------
    {
        "anon_id": "p2v-N1",
        "unit_type": "gap",
        "topic_stratum": "probe_forward",
        "gap_type": "mechanism_gap",
        "expected": "NONE",
        "rationale": (
            "The assertion is that implementation language does not mediate "
            "learned weights once the FP operation sequence and seed are fixed. "
            "Adjacent work is abundant (floating-point non-associativity, "
            "cross-hardware divergence), but none tests the language variable "
            "with the operation sequence held byte-identical, so adjacency "
            "exists while the delta does not."),
        "description": (
            "Reproducibility studies attribute training-run divergence to seeds, "
            "hardware, and floating-point non-associativity. They never vary the "
            "implementation language while holding the operation sequence fixed."),
        "target_setting": (
            "Two implementations emitting a byte-identical sequence of "
            "floating-point operations, differing only in source language."),
        "observed_problem": (
            "Language is confounded with library defaults and operation order in "
            "every cross-implementation comparison reported so far."),
        "existing_coverage": (
            "Floating-point non-associativity and accumulation-order effects are "
            "well documented; cross-vendor mismatch is traced to accumulation "
            "order. The language variable itself is never isolated."),
        "missing_capability": (
            "A controlled test isolating language-level from arithmetic-level "
            "effects on learned parameters."),
        "claimed_delta": (
            "Implementation language is not a mediator of learned parameters once "
            "the floating-point operation sequence is held byte-identical; any "
            "reported language effect is an artefact of differing operation order "
            "rather than of the language itself."),
        "testable_hypothesis": (
            "Bitwise-identical parameter trajectories for the same operation "
            "sequence compiled under two languages."),
        "falsification_condition": (
            "Differing parameter trajectories despite byte-identical operation "
            "sequences and identical seeds."),
        "provenance_status": "verified",
        "evidence": [],
        "evidence_identity_redacted": True,
    },
    {
        "anon_id": "p2v-N2",
        "unit_type": "gap",
        "topic_stratum": "probe_forward",
        "gap_type": "mechanism_gap",
        "expected": "NONE",
        "rationale": (
            "Most papers measure whether load balancing reduces specialization. "
            "None isolates the balance gradient as the causal variable while "
            "holding data, schedule, and all other hyperparameters fixed, so the "
            "causal assertion is untested despite a very busy topic."),
        "description": (
            "Work on mixture-of-experts reports that auxiliary load-balancing "
            "losses weaken expert specialization, measured by selection "
            "frequencies across domains."),
        "target_setting": (
            "Matched training runs differing only in the balancing-loss weight, "
            "with data order, schedule and all other hyperparameters fixed."),
        "observed_problem": (
            "Specialization is compared across architectures and datasets, so the "
            "balancing gradient is confounded with capacity, data mixture and "
            "routing implementation."),
        "existing_coverage": (
            "Correlational evidence that balancing reduces specialization is "
            "extensive; bias-only and loss-free balancing variants are analysed."),
        "missing_capability": (
            "A single-variable causal test of the balancing gradient on "
            "specialization verdicts."),
        "claimed_delta": (
            "The balance gradient, not the routing statistics it produces, is what "
            "makes specialization verdicts unreliable: with data, schedule and "
            "capacity fixed, changing only the balancing weight flips the verdict "
            "while selection entropy is held approximately constant."),
        "testable_hypothesis": (
            "Specialization verdicts change sign under a balancing-weight sweep "
            "with matched selection entropy."),
        "falsification_condition": (
            "Verdicts staying invariant while balancing weight varies over an "
            "order of magnitude at matched entropy."),
        "provenance_status": "verified",
        "evidence": [],
        "evidence_identity_redacted": True,
    },
    # ---------------- REVERSE: expected FULL ----------------
    {
        "anon_id": "p2v-R1",
        "unit_type": "gap",
        "topic_stratum": "probe_reverse",
        "gap_type": "mechanism_gap",
        "expected": "FULL",
        "rationale": "Vaswani et al. 2017 establishes exactly this.",
        "description": (
            "Sequence transduction was dominated by recurrent and convolutional "
            "architectures with attention as an auxiliary component."),
        "target_setting": "Machine translation benchmarks (WMT 2014 En-De, En-Fr).",
        "observed_problem": (
            "Recurrent models preclude parallelisation over sequence positions and "
            "struggle with long-range dependencies."),
        "existing_coverage": "Recurrent and convolutional seq2seq with attention.",
        "missing_capability": "Fully attention-based transduction without recurrence.",
        "claimed_delta": (
            "A transformer architecture relying solely on attention mechanisms, "
            "with no recurrence and no convolution, establishes that recurrence is "
            "unnecessary: it achieves state-of-the-art translation quality while "
            "being more parallelisable than recurrent or convolutional models."),
        "testable_hypothesis": (
            "An attention-only model reaches at least the BLEU of the best "
            "recurrent/convolutional model on WMT 2014 En-De at lower training cost."),
        "falsification_condition": (
            "Attention-only architecture failing to match recurrent baselines."),
        "provenance_status": "verified",
        "evidence": [],
        "evidence_identity_redacted": True,
    },
    {
        "anon_id": "p2v-R2",
        "unit_type": "gap",
        "topic_stratum": "probe_reverse",
        "gap_type": "mechanism_gap",
        "expected": "FULL",
        "rationale": "Hu et al. 2021 (LoRA) establishes exactly this.",
        "description": (
            "Adapting large pretrained language models meant fine-tuning all "
            "parameters or adding adapter layers, both costly."),
        "target_setting": "Large pretrained language models (GPT-3 175B scale).",
        "observed_problem": (
            "Full fine-tuning requires storing and optimising a full parameter "
            "copy per task; adapters add inference latency."),
        "existing_coverage": "Adapters, prefix-tuning, full fine-tuning.",
        "missing_capability": (
            "Low-rank update injection that reduces trainable parameters by orders "
            "of magnitude without inference latency."),
        "claimed_delta": (
            "Freezing the pretrained weights and injecting trainable low-rank "
            "decomposition matrices into each layer reduces trainable parameters "
            "by roughly four orders of magnitude while matching full fine-tuning "
            "quality, because the adaptation needed for a downstream task has low "
            "intrinsic rank."),
        "testable_hypothesis": (
            "Low-rank adaptation matches full fine-tuning on GLUE with ~10,000x "
            "fewer trainable parameters and no added inference latency."),
        "falsification_condition": (
            "Low-rank adaptation failing to match full fine-tuning at that "
            "parameter budget."),
        "provenance_status": "verified",
        "evidence": [],
        "evidence_identity_redacted": True,
    },
    {
        "anon_id": "p2v-R3",
        "unit_type": "gap",
        "topic_stratum": "probe_reverse",
        "gap_type": "mechanism_gap",
        "expected": "FULL",
        "rationale": "Wei et al. 2022 (Chain-of-Thought) establishes exactly this.",
        "description": (
            "Large language models were evaluated mainly with direct answer "
            "prompting on reasoning benchmarks."),
        "target_setting": "Multi-step arithmetic, commonsense and symbolic reasoning.",
        "observed_problem": (
            "Direct prompting yields low accuracy on multi-step problems where the "
            "reasoning path must be composed."),
        "existing_coverage": "Direct-answer prompting, fine-tuning with rationales.",
        "missing_capability": (
            "Eliciting multi-step reasoning purely through prompting at scale."),
        "claimed_delta": (
            "Prompting a model to produce intermediate reasoning steps before the "
            "final answer substantially raises accuracy on multi-step reasoning, "
            "and this benefit emerges only above a critical model scale, indicating "
            "that the ability is latent in sufficiently large models rather than "
            "introduced by the prompt."),
        "testable_hypothesis": (
            "Chain-of-thought prompting improves GSM8K accuracy most sharply for "
            "models above roughly 100B parameters."),
        "falsification_condition": (
            "No scale-dependent benefit from chain-of-thought prompting."),
        "provenance_status": "verified",
        "evidence": [],
        "evidence_identity_redacted": True,
    },
]


def main() -> int:
    with open(UNITS, "w", encoding="utf-8") as fh:
        for p in PROBES:
            rec = {k: v for k, v in p.items()
                   if k not in ("expected", "rationale")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(KEY, "w", encoding="utf-8") as fh:
        json.dump({
            "warning": ("SEALED. Probe expectations were fixed in "
                        "RUBRIC_V2_PREREG.md before any v2 run."),
            "gates": {
                "G1_forward_min_none": 1,
                "G2_reverse_min_full": 3,
                "G3_verdict_vs_delta_established": ">=4/5",
            },
            "probes": {p["anon_id"]: {"expected": p["expected"],
                                      "rationale": p["rationale"]}
                       for p in PROBES},
        }, fh, ensure_ascii=False, indent=2)

    print(f"wrote {len(PROBES)} probes -> {UNITS}")
    for p in PROBES:
        print(f"  {p['anon_id']}  expect {p['expected']:6}  {p['topic_stratum']}")
    print(f"sealed expectations -> {KEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
