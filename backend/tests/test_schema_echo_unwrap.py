"""Schema-echo recovery tests.

The local Qwen backend intermittently answers a `chat_json` call with the JSON
SCHEMA envelope instead of the data:

    {"properties": {"queries": [...]}, "required": ["queries"],
     "title": "AuditQueries", "type": "object"}

The real values are nested one level down inside `properties`. Validation then
fails on the missing top-level field even though the answer is present, burning
the provider's single repair retry and (on 2026-09-16) losing audit samples.

`_unwrap_schema_echo` repairs this deterministically. The tests below pin both
that the recovery works and that it never misfires on a legitimate payload.
"""

import pytest
from pydantic import BaseModel, Field

from app.llm.venus_provider import VenusProvider


class AuditQueriesLike(BaseModel):
    queries: list[str] = Field(min_length=2, max_length=8)


class Nested(BaseModel):
    value: str = ""
    count: int = 0


class TestUnwrapSchemaEcho:
    def test_recovers_values_nested_under_properties(self):
        echoed = {
            "properties": {"queries": ["a query", "another query"]},
            "required": ["queries"],
            "title": "AuditQueries",
            "type": "object",
        }
        assert VenusProvider._unwrap_schema_echo(echoed) == {
            "queries": ["a query", "another query"]}

    def test_recovers_multi_field_payload(self):
        echoed = {
            "properties": {"value": "hello", "count": 3},
            "required": ["value", "count"],
            "title": "Nested",
            "type": "object",
        }
        assert VenusProvider._unwrap_schema_echo(echoed) == {"value": "hello", "count": 3}

    def test_leaves_legitimate_payload_untouched(self):
        """A correct answer must never be rewritten.

        Guards the obvious failure mode of a recovery heuristic: a model whose
        schema legitimately HAS a `properties` field would be silently mangled.
        """
        good = {"queries": ["a", "b"]}
        assert VenusProvider._unwrap_schema_echo(good) == good

    def test_payload_with_real_content_beside_properties_is_not_unwrapped(self):
        """If the envelope carries its own answer fields, it is not a pure echo."""
        data = {"queries": ["a", "b"], "properties": {"unrelated": 1}}
        assert VenusProvider._unwrap_schema_echo(data) == data

    def test_empty_properties_is_not_unwrapped(self):
        # Nothing to recover from; hand it back so the caller raises properly.
        data = {"properties": {}, "type": "object"}
        assert VenusProvider._unwrap_schema_echo(data) == data

    @pytest.mark.parametrize("value", [None, [], "text", 42, {"type": "object"}])
    def test_non_dict_or_unrelated_input_passes_through(self, value):
        assert VenusProvider._unwrap_schema_echo(value) == value

    def test_recovered_payload_validates(self):
        """End to end: the unwrapped dict must satisfy the schema that failed."""
        echoed = {
            "properties": {"queries": ["q one", "q two", "q three"]},
            "required": ["queries"],
            "title": "AuditQueriesLike",
            "type": "object",
        }
        parsed = AuditQueriesLike.model_validate(
            VenusProvider._unwrap_schema_echo(echoed))
        assert parsed.queries == ["q one", "q two", "q three"]
