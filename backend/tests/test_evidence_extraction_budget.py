"""Evidence extraction must supply what gap mining requires.

On a real run (task 5c2de9c7) limitation/negative_result units were 8% of the
108 extracted units, and 9 of 12 research questions were then rejected for
NO_LIMITATION_SIGNAL — no gap could be mined from them. Two independent defects
caused it: a paper's chunk budget was spent by section priority, so a long method
section crowded out the conclusion entirely, and the section name passed to the
extraction prompt was hard-coded to "", so the hints written for conclusion and
abstract never applied to anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PRIORITY = {"conclusion": 0, "method": 1, "experiment": 2, "abstract": 3, "introduction": 4}


def _chunks(section, count):
    return [(_PRIORITY[section], section, f"{section} text {index}", f"{section}-{index}")
            for index in range(count)]


def test_a_long_section_cannot_consume_the_whole_budget():
    from app.agent.steps.extract_evidence import interleave_section_chunks

    by_section = {
        "method": _chunks("method", 5),
        "experiment": _chunks("experiment", 4),
        "conclusion": _chunks("conclusion", 2),
    }

    selected = interleave_section_chunks(by_section, 6, _PRIORITY)

    assert len(selected) == 6
    sections = [item[1] for item in selected]
    # The conclusion carries the limitation statements gap admission requires;
    # under the old global sort, method+experiment filled all six slots.
    assert "conclusion" in sections
    assert sections.count("conclusion") == 2
    assert "method" in sections and "experiment" in sections


def test_conclusion_is_taken_first():
    from app.agent.steps.extract_evidence import interleave_section_chunks

    by_section = {"method": _chunks("method", 3), "conclusion": _chunks("conclusion", 3)}

    selected = interleave_section_chunks(by_section, 2, _PRIORITY)

    assert [item[1] for item in selected] == ["conclusion", "method"]


def test_budget_is_fully_used_when_one_section_is_short():
    from app.agent.steps.extract_evidence import interleave_section_chunks

    by_section = {"conclusion": _chunks("conclusion", 1), "method": _chunks("method", 9)}

    selected = interleave_section_chunks(by_section, 6, _PRIORITY)

    assert len(selected) == 6
    assert [item[1] for item in selected].count("conclusion") == 1
    assert [item[1] for item in selected].count("method") == 5


def test_selection_is_empty_when_there_is_nothing_to_extract():
    from app.agent.steps.extract_evidence import interleave_section_chunks

    assert interleave_section_chunks({}, 6, _PRIORITY) == []


class _CapturingLLM:
    def __init__(self):
        self.messages = None

    async def chat_json(self, messages, schema):
        self.messages = messages
        return schema(evidence_units=[])


@pytest.mark.asyncio
async def test_section_specific_hints_reach_the_model():
    """The hints only work if the real section name is passed through."""
    from types import SimpleNamespace

    from app.agent.steps.extract_evidence import _llm_extract_evidence

    paper = SimpleNamespace(title="A paper", abstract="abstract")

    llm = _CapturingLLM()
    await _llm_extract_evidence(llm, paper, "Discussion. We do not evaluate X.", "conclusion")
    conclusion_prompt = llm.messages[-1]["content"]
    assert "Limitations" in conclusion_prompt
    assert "negative_result" in conclusion_prompt

    llm = _CapturingLLM()
    await _llm_extract_evidence(llm, paper, "We train on 8 GPUs.", "method")
    method_prompt = llm.messages[-1]["content"]
    assert "boundaries" in method_prompt
    # The hint must not invite inferring restrictions the text does not state.
    assert "Never infer a restriction" in method_prompt

    llm = _CapturingLLM()
    await _llm_extract_evidence(llm, paper, "Some background.", "introduction")
    assert "Limitations" not in llm.messages[-1]["content"]
