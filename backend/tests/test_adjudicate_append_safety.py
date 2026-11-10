"""Adjudication verdict files must never be truncated.

Incident (2026-11-02): `adjudicate_pilot.py` chose `mode = "w"` whenever
--resume was absent. Re-running it with `--only fbp-089` therefore wrote ONE
record over a complete 91-unit adjudication, and no backup existed — a day of
LLM adjudication was destroyed by a routine single-unit re-check.

The fix has two parts, both asserted here:
  1. the output file is always opened in append mode;
  2. the first append over a non-empty file leaves a .bak copy.

These tests read the script's source rather than executing a live model call,
because the property under test is "how the file is opened", and paying for
LLM tokens to observe that would be silly.
"""

import os
import re

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                      "adjudicate_pilot.py")


def _source() -> str:
    return open(SCRIPT, encoding="utf-8").read()


def _code_lines() -> list[str]:
    """Source with comments stripped — the incident is described in a comment,
    and a naive substring check would match the explanation instead of code."""
    out = []
    for line in _source().splitlines():
        stripped = line.split("#", 1)[0]
        if stripped.strip():
            out.append(stripped)
    return out


def test_never_opens_output_in_write_mode():
    code = "\n".join(_code_lines())
    # No mode assignment may evaluate to "w" for the verdict output.
    assert 'mode = "w"' not in code, (
        "the verdict writer must never truncate an existing adjudication")
    assert re.search(r'mode\s*=\s*"a"', code), "append mode must be explicit"


def test_backs_up_before_appending_over_existing_verdicts():
    src = _source()
    assert ".bak" in src, "an existing verdict file must be backed up first"
    assert "shutil.copy2" in src


def test_single_unit_rerun_flag_is_documented_as_additive():
    """The --only path must not be presented as a fresh pass."""
    src = _source()
    # A reader must be able to learn that --only appends, from the script.
    assert "accumulating" in src.lower() or "append" in src.lower()


def test_verdicts_are_keyed_by_anon_id_so_appends_are_safe():
    """Append safety depends on the key being stable and unique per unit."""
    src = _source()
    assert '"anon_id": unit["anon_id"]' in src
