"""Shared contracts — every module codes against these. Change only via PR reviewed by the lead.

Educational simulation only. No real medical diagnosis or treatment recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# ---------------------------------------------------------------- Contract 1: facts
# A fact is a tuple: ("Predicate", arg1, arg2, ...)
# Constants: "P1", "D1", "ICU1", "Cardiology", 125, 0.91
# Variables (rules / queries only): strings starting with "?"  e.g. "?x"
Fact = tuple
Theta = dict[str, Any]  # substitution {"?x": "P1"}


def is_var(term: Any) -> bool:
    return isinstance(term, str) and term.startswith("?")


RiskLabel = Literal["Low", "Medium", "High", "Critical"]
RISK_ORDER: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
BedType = Literal["ICU", "Emergency", "ExamRoom"]


# ---------------------------------------------------------------- Contract 2: logic -> search
@dataclass(frozen=True)
class PatientRequirement:
    pid: str
    priority: int            # 1..4, 4 = most urgent (Critical)
    bed_type: str            # "ICU" | "Emergency" | "ExamRoom"
    specialty: str           # "Cardiology" | "General" | ...
    waiting_since: int = 0   # tick at which the patient started waiting


# ---------------------------------------------------------------- Contract 3: search -> planning
@dataclass
class Allocation:
    # {"P1": {"bed": "ICU1", "doctor": "D1"}, "P3": {"bed": None, "doctor": None}}  None = deferred
    assignments: dict[str, dict[str, str | None]]
    cost: float
    stats: dict[str, Any] = field(default_factory=dict)  # algo, expanded, generated, max_frontier, time_ms


# ---------------------------------------------------------------- Contract 4: planning -> agent
@dataclass(frozen=True)
class Action:
    name: str                                   # "ReserveICUBed"
    args: tuple                                 # ("P1", "ICU1")
    pre: frozenset = frozenset()                # facts that must hold
    pre_neg: frozenset = frozenset()            # facts that must NOT hold
    add: frozenset = frozenset()
    delete: frozenset = frozenset()

    def __str__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.args))})"


Plan = list[Action]


def is_applicable(action: Action, state: set[Fact] | frozenset[Fact]) -> bool:
    return action.pre <= state and not (action.pre_neg & state)


def apply(action: Action, state: set[Fact] | frozenset[Fact]) -> frozenset[Fact]:
    return frozenset((set(state) - action.delete) | action.add)


def validate(plan: Plan, state: set[Fact] | frozenset[Fact]) -> tuple[bool, int | None]:
    """Simulate the plan. Returns (ok, index of first non-applicable action)."""
    s = frozenset(state)
    for i, a in enumerate(plan):
        if not is_applicable(a, s):
            return False, i
        s = apply(a, s)
    return True, None


# ---------------------------------------------------------------- Contract 5: events
EventType = Literal["patient_arrival", "resource_unavailable", "resource_available", "vitals_update"]


@dataclass
class Event:
    t: int
    type: EventType
    payload: dict[str, Any]  # patient dict | {"id": "ICU1"} | {"pid": "P1", "vitals": {...}}

    @staticmethod
    def from_dict(d: dict) -> "Event":
        d = dict(d)
        return Event(t=d.pop("t"), type=d.pop("type"), payload=d)


# ---------------------------------------------------------------- module interfaces (signatures only)
class MLPredictor(Protocol):
    def __call__(self, patient: dict) -> tuple[str, float]: ...          # Contract 6 patient -> ("High", 0.87)


class DLPredictor(Protocol):
    def __call__(self, history: list[dict]) -> tuple[str, float]: ...    # vitals history -> ("Critical", 0.91)


class Searcher(Protocol):
    def __call__(self, reqs: list[PatientRequirement], resources: dict, algo: str = "astar") -> Allocation: ...


class Planner(Protocol):
    def __call__(self, state: frozenset[Fact], goals: frozenset[Fact]) -> Plan | None: ...


DISCLAIMER = "Educational simulation only. No real medical diagnosis or treatment recommendation."
