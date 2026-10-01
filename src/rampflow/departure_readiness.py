from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from json import dumps
from threading import RLock
from typing import Iterable, Mapping

from .models import Decision
from .timeutil import utc


class ReadinessStatus(str, Enum):
    SILENT = "silent"  # board-derived: no claim received from this department yet
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETE = "complete"


class DepartureVerdict(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    PENDING = "pending"


@dataclass(frozen=True)
class ReadinessClaim:
    """One department's report about a flight.

    Claims are immutable facts ordered by a per-(flight, department) sequence.
    A correction is simply a newer claim: it is never rejected for arriving
    late, only for not being newer than what the source already said.
    """

    flight_id: str
    department: str
    sequence: int
    status: ReadinessStatus
    reason: str
    reported_at: datetime
    received_at: datetime

    def __post_init__(self):
        if not self.flight_id or not self.department:
            raise ValueError("flight and department required")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        status = ReadinessStatus(self.status)
        if status is ReadinessStatus.SILENT:
            raise ValueError("departments cannot report silent")
        if status is ReadinessStatus.BLOCKED and not self.reason:
            raise ValueError("blocked claims must name a reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reported_at", utc(self.reported_at))
        object.__setattr__(self, "received_at", utc(self.received_at))


@dataclass(frozen=True)
class RecordedClaim:
    claim: ReadinessClaim
    generation: int


@dataclass(frozen=True)
class DepartmentState:
    department: str
    status: ReadinessStatus
    reason: str
    sequence: int
    reported_at: datetime | None
    received_at: datetime | None


@dataclass(frozen=True)
class ReadinessBlocker:
    department: str
    kind: str  # "reported-blocked" | "awaiting-confirmation"
    reason: str


def _canonical(payload: Mapping[str, object]) -> bytes:
    return dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _assessment_payload(
    flight_id: str,
    generation: int,
    verdict: DepartureVerdict,
    blockers: tuple[ReadinessBlocker, ...],
    departments: Mapping[str, DepartmentState],
) -> Mapping[str, object]:
    return {
        "flight_id": flight_id,
        "generation": generation,
        "verdict": DepartureVerdict(verdict).value,
        "blockers": [
            {"department": b.department, "kind": b.kind, "reason": b.reason}
            for b in blockers
        ],
        "departments": {
            name: {
                "status": ReadinessStatus(state.status).value,
                "reason": state.reason,
                "sequence": state.sequence,
                "reported_at": state.reported_at.isoformat() if state.reported_at else None,
                "received_at": state.received_at.isoformat() if state.received_at else None,
            }
            for name, state in departments.items()
        },
    }


def _assessment_digest(
    flight_id: str,
    generation: int,
    verdict: DepartureVerdict,
    blockers: tuple[ReadinessBlocker, ...],
    departments: Mapping[str, DepartmentState],
) -> str:
    return sha256(_canonical(_assessment_payload(flight_id, generation, verdict, blockers, departments))).hexdigest()


@dataclass(frozen=True)
class DepartureAssessment:
    """The departure conclusion for one flight at one board generation.

    Immutable and digest-stamped: two readers holding an assessment of the same
    generation can prove they see the same answer by comparing digests, and any
    holder can re-verify integrity with verify().
    """

    flight_id: str
    generation: int
    verdict: DepartureVerdict
    blockers: tuple[ReadinessBlocker, ...]
    departments: Mapping[str, DepartmentState]
    digest: str

    def verify(self) -> bool:
        return self.digest == _assessment_digest(
            self.flight_id, self.generation, self.verdict, self.blockers, self.departments
        )


class DepartureReadinessBoard:
    """Aggregates per-department readiness claims into one departure verdict.

    Every accepted claim advances a single board generation.  An assessment is a
    pure function of (flight, generation): it is derived from the append-only
    claim log, cached under its generation, and never changes afterwards, so two
    readers pinning the same generation always see the identical answer.  Late
    corrections are newer claims — they advance the generation and the next
    assessment recomputes the blocking reasons from scratch, while earlier
    generations stay reproducible for audit.  Readers that must agree with each
    other should pin the same generation; assess() stamps every verdict with
    the generation it was computed from.
    """

    def __init__(self, required_departments: Iterable[str]):
        required = tuple(sorted(set(required_departments)))
        if not required:
            raise ValueError("at least one required department")
        self._required = required
        self._log: dict[str, list[RecordedClaim]] = {}
        self._generation = 0
        self._assessments: dict[tuple[str, int], DepartureAssessment] = {}
        self._lock = RLock()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def required_departments(self) -> tuple[str, ...]:
        return self._required

    def flights(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._log))

    def submit(self, claim: ReadinessClaim, expected_generation: int | None = None) -> Decision:
        with self._lock:
            if expected_generation is not None and expected_generation != self._generation:
                return Decision(
                    False,
                    "stale-generation",
                    "board changed since the claim was prepared",
                    {"generation": self._generation},
                )
            log = self._log.get(claim.flight_id, ())
            last = None
            for rec in reversed(log):
                if rec.claim.department == claim.department:
                    last = rec
                    break
            if last is not None:
                if claim.sequence < last.claim.sequence:
                    return Decision(
                        False,
                        "stale-sequence",
                        "department already reported a newer sequence",
                        {"sequence": last.claim.sequence},
                    )
                if claim.sequence == last.claim.sequence:
                    if claim == last.claim:
                        return Decision(True, "duplicate", "claim already recorded", {"generation": self._generation})
                    return Decision(
                        False,
                        "sequence-conflict",
                        "sequence already used for a different claim",
                        {"sequence": last.claim.sequence},
                    )
            self._generation += 1
            self._log.setdefault(claim.flight_id, []).append(RecordedClaim(claim, self._generation))
            return Decision(True, "accepted", "claim recorded", {"generation": self._generation})

    def claims(self, flight_id: str) -> tuple[RecordedClaim, ...]:
        with self._lock:
            return tuple(self._log.get(flight_id, ()))

    def assess(self, flight_id: str, generation: int | None = None) -> DepartureAssessment:
        with self._lock:
            gen = self._generation if generation is None else generation
            if gen < 0 or gen > self._generation:
                raise ValueError(f"unknown generation {gen}")
            key = (flight_id, gen)
            cached = self._assessments.get(key)
            if cached is not None:
                return cached
            latest: dict[str, RecordedClaim] = {}
            for rec in self._log.get(flight_id, ()):
                if rec.generation <= gen:
                    latest[rec.claim.department] = rec
            departments: dict[str, DepartmentState] = {}
            for name in sorted(set(self._required) | set(latest)):
                rec = latest.get(name)
                if rec is None:
                    departments[name] = DepartmentState(name, ReadinessStatus.SILENT, "", 0, None, None)
                else:
                    c = rec.claim
                    departments[name] = DepartmentState(name, c.status, c.reason, c.sequence, c.reported_at, c.received_at)
            blockers = []
            for name in self._required:
                state = departments[name]
                if state.status is ReadinessStatus.COMPLETE:
                    continue
                if state.status is ReadinessStatus.BLOCKED:
                    blockers.append(ReadinessBlocker(name, "reported-blocked", state.reason))
                else:
                    blockers.append(ReadinessBlocker(name, "awaiting-confirmation", ""))
            if any(b.kind == "reported-blocked" for b in blockers):
                verdict = DepartureVerdict.BLOCKED
            elif blockers:
                verdict = DepartureVerdict.PENDING
            else:
                verdict = DepartureVerdict.READY
            blockers_t = tuple(blockers)
            assessment = DepartureAssessment(
                flight_id,
                gen,
                verdict,
                blockers_t,
                departments,
                _assessment_digest(flight_id, gen, verdict, blockers_t, departments),
            )
            self._assessments[key] = assessment
            return assessment
