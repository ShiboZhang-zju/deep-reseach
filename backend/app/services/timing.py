"""Sub-stage timing collector for phase internals.

phase_service records ONE duration per phase (probe df744606: `mine_gaps 221s`,
`audit_gaps 1057s`, `generate_minimal_experiments 666s`). That is enough to know
which phase is slow and useless for knowing *why*: a phase is a fan-out over
lenses then a loop over candidates then several LLM calls, and the recorded
number collapses all of them.

This collector lets a phase attribute its own wall clock to named sub-stages
and dump the result into the step's existing trace payload (`stage_timings`),
mirroring what the extraction step already does per paper.
"""

import time
from collections import defaultdict
from contextlib import contextmanager


class StageTimings:
    """Accumulate named sub-stage durations; cheap and exception-safe.

    Repeated entries with the same name accumulate (a lens loop calls each
    lens once, but a candidate loop calls `audit` once per candidate), and a
    `*_count` is recorded alongside so per-item averages are derivable.
    """

    __slots__ = ("_totals", "_counts")

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._totals[name] += time.perf_counter() - started
            self._counts[name] += 1

    def add(self, name: str, seconds: float) -> None:
        """Record a duration measured elsewhere (e.g. inside a helper)."""
        self._totals[name] += max(seconds, 0.0)
        self._counts[name] += 1

    def as_dict(self) -> dict:
        """Serializable view: {stage: {seconds, ms, count, avg_ms}}.

        Seconds keep 2 decimals plus an exact `ms` field: sub-second stages
        (a fast embedding call, an in-memory dedup) round to 0.0 at 1 decimal,
        which reads as "this stage was free" and hides it from the report.
        """
        return {
            name: {
                "seconds": round(total, 2),
                "ms": int(total * 1000),
                "count": self._counts[name],
                "avg_ms": int(total / self._counts[name] * 1000)
                if self._counts[name] else 0,
            }
            for name, total in sorted(
                self._totals.items(), key=lambda kv: kv[1], reverse=True)
        }
