"""Aggregate accountable-area sign-offs into one departure conclusion.

Fuel, baggage, maintenance and other accountable areas each report their own
progress.  A :class:`DepartureReleaseBoard` folds that append-only report
stream into a single derived verdict for a flight:

* ``released``  - every required area's latest report is ``complete``;
* ``blocked``   - at least one area's latest report is ``blocked``, and the
  blocking areas with their reasons are listed explicitly;
* ``pending``   - nobody is blocked but some areas are still unreported or in
  progress.

Corrections never overwrite history.  A late correction appends a new report
at a new generation and the verdict - including the set of blocking areas and
their reasons - is recomputed from scratch for that generation.  Every verdict
is content-addressed with a digest and cached immutably, so two duty officers
reading the same generation cannot observe two different answers, and a reader
pinned to an old generation keeps seeing the same verdict after later
corrections arrive.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
from json import dumps
from threading import RLock
from typing import FrozenSet, Iterable, Mapping

from .models import Decision
from .timeutil import utc


class SignoffState(str, Enum):
    """State of one accountable area.

    ``PENDING`` is derived (no report yet) and cannot be filed directly.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class ReleaseStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    RELEASED = "released"


_FILED_STATES = frozenset({SignoffState.IN_PROGRESS, SignoffState.COMPLETE, SignoffState.BLOCKED})


@dataclass(frozen=True)
class ReleaseArea:
    """An area whose sign-off is required before departure."""

    area_id: str
    label: str = ""
    required_evidence: FrozenSet[str] = frozenset()

    def __post_init__(self):
        if not self.area_id:
            raise ValueError("area id required")
        object.__setattr__(self, "required_evidence", frozenset(self.required_evidence))


@dataclass(frozen=True)
class SignoffReport:
    """One immutable area report, linked into a tamper-evident chain."""

    sequence: int
    operation_id: str
    area: str
    state: SignoffState
    actor: str
    reported_at: datetime
    reason: str | None
    evidence: Mapping[str, str]
    supersedes: int | None
    previous_hash: str
    digest: str


@dataclass(frozen=True)
class AreaStatus:
    """Derived latest state of one area at a conclusion generation."""

    area: str
    state: SignoffState
    actor: str | None = None
    reported_at: datetime | None = None
    reason: str | None = None
    evidence: Mapping[str, str] = field(default_factory=dict)
    sequence: int | None = None
    updates: int = 0
    blocked_since: datetime | None = None


@dataclass(frozen=True)
class DepartureConclusion:
    """Immutable, content-addressed departure verdict for one generation."""

    flight_id: str
    generation: int
    status: ReleaseStatus
    areas: Mapping[str, AreaStatus]
    blockers: tuple[AreaStatus, ...]
    outstanding: tuple[AreaStatus, ...]
    basis: Mapping[str, int | None]
    digest: str

    @property
    def released(self) -> bool:
        return self.status is ReleaseStatus.RELEASED


def _canonical(payload: object) -> bytes:
    return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


class DepartureReleaseBoard:
    """Append-only sign-off log with generation-stamped derived verdicts.

    All reads and writes are serialized by one lock: a reader always observes
    a consistent prefix of the log.  Verdicts are cached by generation, so the
    same generation always resolves to the very same immutable object.
    """

    def __init__(self, flight_id: str, areas: Iterable[str | ReleaseArea]):
        if not flight_id:
            raise ValueError("flight id required")
        normalized = tuple(ReleaseArea(a) if isinstance(a, str) else a for a in areas)
        if not normalized:
            raise ValueError("at least one sign-off area required")
        ids = [a.area_id for a in normalized]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sign-off area")
        self.flight_id = flight_id
        self.areas: Mapping[str, ReleaseArea] = {a.area_id: a for a in normalized}
        self._reports: list[SignoffReport] = []
        self._by_operation: dict[str, SignoffReport] = {}
        self._verdicts: dict[int, DepartureConclusion] = {}
        self._last_hash = "0" * 64
        self._lock = RLock()

    @property
    def generation(self) -> int:
        with self._lock:
            return len(self._reports)

    def head_sequence(self) -> int:
        return self.generation

    # ------------------------------------------------------------------ writes

    def report(
        self,
        operation_id: str,
        area: str,
        state: SignoffState,
        actor: str,
        at: datetime,
        reason: str | None = None,
        evidence: Mapping[str, str] | None = None,
    ) -> SignoffReport:
        """File (or idempotently replay) one area progress report.

        A blocked report must carry a reason and a complete report must carry
        every piece of evidence configured for the area - an unsubstantiated
        "done" is exactly what the board exists to reject.
        """
        if not operation_id:
            raise ValueError("operation id required")
        if not actor:
            raise ValueError("actor required")
        spec = self.areas.get(area)
        if spec is None:
            raise ValueError(f"unknown sign-off area {area}")
        if state not in _FILED_STATES:
            raise ValueError(f"state {state!r} cannot be filed directly")
        reason = reason.strip() if isinstance(reason, str) else reason
        if state is SignoffState.BLOCKED and not reason:
            raise ValueError("blocked report requires a reason")
        evidence = dict(evidence or {})
        if state is SignoffState.COMPLETE:
            missing = sorted(set(spec.required_evidence) - {k for k, v in evidence.items() if str(v).strip()})
            if missing:
                raise ValueError(f"complete report for {area} missing evidence: {','.join(missing)}")
        at = utc(at)

        with self._lock:
            duplicate = self._by_operation.get(operation_id)
            if duplicate is not None:
                same = (
                    duplicate.area == area
                    and duplicate.state is state
                    and duplicate.actor == actor
                    and duplicate.reported_at == at
                    and duplicate.reason == reason
                    and dict(duplicate.evidence) == evidence
                )
                if not same:
                    raise ValueError("operation id reused with a different report")
                return duplicate

            sequence = len(self._reports) + 1
            supersedes = self._latest_sequence(area)
            digest = self._hash(sequence, operation_id, area, state, actor, at, reason, evidence, self._last_hash)
            record = SignoffReport(
                sequence,
                operation_id,
                area,
                state,
                actor,
                at,
                reason,
                evidence,
                supersedes,
                self._last_hash,
                digest,
            )
            self._reports.append(record)
            self._by_operation[operation_id] = record
            self._last_hash = digest
            return record

    def correct(
        self,
        operation_id: str,
        area: str,
        state: SignoffState,
        actor: str,
        at: datetime,
        reason: str | None = None,
        evidence: Mapping[str, str] | None = None,
    ) -> SignoffReport:
        """File a late correction.

        Thin alias for :meth:`report` that makes the correction intent explicit
        at call sites; append-only semantics are identical.
        """
        return self.report(operation_id, area, state, actor, at, reason, evidence)

    # ------------------------------------------------------------------- reads

    def history(self, area: str | None = None) -> tuple[SignoffReport, ...]:
        with self._lock:
            if area is None:
                return tuple(self._reports)
            if area not in self.areas:
                raise ValueError(f"unknown sign-off area {area}")
            return tuple(r for r in self._reports if r.area == area)

    def changes(self, after_sequence: int = 0) -> tuple[SignoffReport, ...]:
        with self._lock:
            return tuple(r for r in self._reports if r.sequence > after_sequence)

    def latest_report(self, area: str) -> SignoffReport | None:
        with self._lock:
            found = None
            for r in self._reports:
                if r.area == area:
                    found = r
            return found

    def conclusion(self, generation: int | None = None) -> DepartureConclusion:
        """Return the immutable verdict at ``generation`` (default: latest).

        A pinned generation is stable forever: later corrections create newer
        generations but never mutate a cached verdict.
        """
        with self._lock:
            current = len(self._reports)
            if generation is None:
                generation = current
            if generation < 0 or generation > current:
                raise ValueError(f"unknown generation {generation}")
            cached = self._verdicts.get(generation)
            if cached is not None:
                return cached
            verdict = self._compute(generation, self._reports[:generation])
            self._verdicts[generation] = verdict
            return verdict

    def can_release(self, generation: int | None = None) -> Decision:
        """Decision form of the verdict, suitable as a workflow requirement."""
        verdict = self.conclusion(generation)
        if verdict.status is ReleaseStatus.RELEASED:
            return Decision(
                True,
                "released",
                "all sign-off areas complete",
                {"generation": verdict.generation, "digest": verdict.digest},
            )
        if verdict.status is ReleaseStatus.BLOCKED:
            return Decision(
                False,
                "blocked",
                "; ".join(f"{b.area}: {b.reason}" for b in verdict.blockers),
                {
                    "generation": verdict.generation,
                    "digest": verdict.digest,
                    "blockers": [b.area for b in verdict.blockers],
                    "outstanding": [a.area for a in verdict.outstanding],
                },
            )
        return Decision(
            False,
            "pending",
            "sign-off areas not yet complete",
            {
                "generation": verdict.generation,
                "digest": verdict.digest,
                "outstanding": [a.area for a in verdict.outstanding],
            },
        )

    # --------------------------------------------------------------- integrity

    def verify_chain(self) -> bool:
        with self._lock:
            previous = "0" * 64
            for r in self._reports:
                if r.previous_hash != previous:
                    return False
                expected = self._hash(
                    r.sequence, r.operation_id, r.area, r.state, r.actor,
                    r.reported_at, r.reason, dict(r.evidence), previous,
                )
                if r.digest != expected:
                    return False
                previous = r.digest
            return True

    def verify_conclusion(self, conclusion: DepartureConclusion) -> bool:
        """Recompute a verdict digest independently from its visible content."""
        return conclusion.digest == self._verdict_digest(conclusion)

    # ----------------------------------------------------------------- helpers

    def _latest_sequence(self, area: str) -> int | None:
        found = None
        for r in self._reports:
            if r.area == area:
                found = r.sequence
        return found

    @staticmethod
    def _hash(
        sequence: int,
        operation_id: str,
        area: str,
        state: SignoffState,
        actor: str,
        at: datetime,
        reason: str | None,
        evidence: Mapping[str, str],
        previous_hash: str,
    ) -> str:
        raw = _canonical(
            {
                "sequence": sequence,
                "operation_id": operation_id,
                "area": area,
                "state": state.value,
                "actor": actor,
                "reported_at": utc(at).isoformat(),
                "reason": reason,
                "evidence": evidence,
                "previous_hash": previous_hash,
            }
        )
        return sha256(raw).hexdigest()

    def _compute(self, generation: int, prefix: list[SignoffReport]) -> DepartureConclusion:
        latest: dict[str, SignoffReport] = {}
        counts: dict[str, int] = defaultdict(int)
        for r in prefix:
            latest[r.area] = r
            counts[r.area] += 1

        statuses: dict[str, AreaStatus] = {}
        for area in sorted(self.areas):
            r = latest.get(area)
            blocked_since = None
            if r is not None and r.state is SignoffState.BLOCKED:
                # Start of the current unbroken blocked run for the area.
                run_start = None
                for item in prefix:
                    if item.area != area:
                        continue
                    if item.state is SignoffState.BLOCKED:
                        if run_start is None:
                            run_start = item.reported_at
                    else:
                        run_start = None
                blocked_since = run_start
            if r is None:
                statuses[area] = AreaStatus(area, SignoffState.PENDING)
            else:
                statuses[area] = AreaStatus(
                    area,
                    r.state,
                    r.actor,
                    r.reported_at,
                    r.reason,
                    dict(r.evidence),
                    r.sequence,
                    counts[area],
                    blocked_since,
                )

        blockers = tuple(s for s in statuses.values() if s.state is SignoffState.BLOCKED)
        outstanding = tuple(
            s for s in statuses.values() if s.state in (SignoffState.PENDING, SignoffState.IN_PROGRESS)
        )
        if blockers:
            status = ReleaseStatus.BLOCKED
        elif any(s.state is not SignoffState.COMPLETE for s in statuses.values()):
            status = ReleaseStatus.PENDING
        else:
            status = ReleaseStatus.RELEASED

        verdict = DepartureConclusion(
            self.flight_id,
            generation,
            status,
            {a: statuses[a] for a in sorted(statuses)},
            blockers,
            outstanding,
            {a: statuses[a].sequence for a in sorted(statuses)},
            "",
        )
        return replace(verdict, digest=self._verdict_digest(verdict))

    @staticmethod
    def _verdict_digest(verdict: DepartureConclusion) -> str:
        content = {
            "flight_id": verdict.flight_id,
            "generation": verdict.generation,
            "status": verdict.status.value,
            "areas": {
                area: {
                    "state": status.state.value,
                    "reason": status.reason,
                    "actor": status.actor,
                    "reported_at": status.reported_at.isoformat() if status.reported_at else None,
                    "sequence": status.sequence,
                    "evidence": dict(status.evidence),
                }
                for area, status in sorted(verdict.areas.items())
            },
        }
        return sha256(_canonical(content)).hexdigest()
