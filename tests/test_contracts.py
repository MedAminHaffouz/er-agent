import json
from pathlib import Path

from er_agent.contracts import Action, Event, apply, is_applicable, validate

SCEN = Path(__file__).parent.parent / "scenarios"


def _assign(p, d):
    return Action("AssignDoctor", (p, d),
                  pre=frozenset({("PatientWaiting", p), ("Available", d)}),
                  add=frozenset({("DoctorAssigned", p, d)}),
                  delete=frozenset({("Available", d)}))


def test_action_applicability_and_effects():
    s = frozenset({("PatientWaiting", "P1"), ("Available", "D1")})
    a = _assign("P1", "D1")
    assert is_applicable(a, s)
    s2 = apply(a, s)
    assert ("DoctorAssigned", "P1", "D1") in s2 and ("Available", "D1") not in s2


def test_validate_detects_busy_doctor():
    s = frozenset({("PatientWaiting", "P1"), ("PatientWaiting", "P2"), ("Available", "D1")})
    ok, idx = validate([_assign("P1", "D1"), _assign("P2", "D1")], s)
    assert not ok and idx == 1


def test_scenarios_parse():
    for f in SCEN.glob("*.json"):
        data = json.loads(f.read_text())
        for e in data.get("events", []):
            Event.from_dict(e)
