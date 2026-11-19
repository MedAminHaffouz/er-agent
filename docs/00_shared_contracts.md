# 00 — Shared Contracts (everyone reads this first)

**Project:** GL4 AI — Integrated Intelligent Agent for Emergency Management
**Deadline:** Tue **Dec 1, 2026** (today: Nov 17 → 14 days)
**Stack:** Python 3.11, `numpy`, `pandas`, `scikit-learn`, `torch`, `streamlit`, `pytest`. No LLM (forbidden by the brief).

> Rule #1: the grade is about **cooperation between components**, not each algorithm alone.
> Every module must consume the previous one's output through the contracts below. No hard-coded shortcuts.

---

## Team split

| # | Who | Owns | Course part |
|---|-----|------|-------------|
| 1 | **Amin** (lead) | Contracts, DL (LSTM/GRU), Block 6 agent loop, dynamic replanning, UI/demo | Learning + integration |
| 2 | Member 2 | Block 1 world model + KR + JSON/events, **Block 5 Planning** (STRIPS) | KR + Planning |
| 3 | Member 3 | Block 2 logic engine: KB, unification, forward chaining, resolution + traces | Deduction |
| 4 | Member 4 | Block 3 synthetic datasets (tabular + temporal) + ML models + evaluation | Learning |
| 5 | Member 5 | Block 4 Search (UCS, A*), heuristic, benchmarks, **report assembly** | Search |

Why Block 1 + Block 5 together: both are about *states, actions, preconditions, effects* — one person owns the action schema end-to-end.

---

## Repo layout

```
er-agent/
├── er_agent/
│   ├── contracts.py        # Amin — dataclasses + type aliases below (frozen Nov 19)
│   ├── world/              # M2 — World, entities, events, to_facts()
│   ├── logic/              # M3 — KB, unify, forward_chain, resolution
│   ├── learning/ml/        # M4 — tabular models
│   ├── learning/dl/        # Amin — sequence models
│   ├── search/             # M5 — allocation problem, UCS, A*
│   ├── planning/           # M2 — STRIPS actions, planner, validator
│   └── agent/              # Amin — perceive→reason→search→plan→act loop
├── data/                   # M4 generator output (gitignored, regenerable with seed)
├── models/                 # trained weights (.joblib / .pt)
├── scenarios/              # JSON scenarios (static + dynamic)
├── ui/app.py               # Amin — Streamlit demo
├── tests/                  # each member: tests/test_<module>.py
└── report/                 # LaTeX, one section file per member
```

Branches: `feat/<module>` → PR into `main`, 1 reviewer minimum. `main` must always run `pytest`.

---

## Contract 1 — Facts (used by everyone)

```python
Fact = tuple            # ("Predicate", arg1, arg2, ...)
# constants: "P1", "D1", "ICU1", "Cardiology", 125, 0.91
# variables (rules/queries only): strings starting with "?"  →  "?x"
```

Examples:
```python
("Patient", "P1")
("HeartRate", "P1", 125)
("HasSymptom", "P1", "ChestPain")
("LowBloodPressure", "P1")          # discretized facts emitted by World (see below)
("Doctor", "D1"); ("Specialty", "D1", "Cardiology"); ("Available", "D1")
("Bed", "ICU1"); ("Type", "ICU1", "ICU"); ("Available", "ICU1")
("PatientWaiting", "P1")
("ML_Risk", "P1", "High", 0.87)     # inserted by agent from M4
("DL_Risk", "P1", "Critical", 0.91) # inserted by agent from Amin
```

**Discretization (M2 emits, M3 relies on it)** — keeps resolution symbolic, no arithmetic in the prover:

| Raw | Derived fact |
|-----|--------------|
| SpO2 < 90 | `LowOxygen(P)` |
| HR > 110 | `Tachycardia(P)` |
| BP = Low / SBP < 90 | `LowBloodPressure(P)` |
| Temp ≥ 38.5 | `Fever(P)` |
| Age ≥ 65 | `Elderly(P)` |

Thresholds live in one place: `er_agent/world/thresholds.py`.

## Contract 2 — Logic output → Search input

M3's reasoning must derive these predicates (names are fixed):

```python
("RiskLevel", P, "Low"|"Medium"|"High"|"Critical")   # final fused level
("Priority", P, 1..4)                                 # 4 = most urgent
("RequiresBed", P, "ICU"|"Emergency"|"ExamRoom")
("RequiresSpecialist", P, "Cardiology"|"General"|...)
("RequiresUrgentEvaluation", P)
```

Adapter (Amin): `requirements_from_kb(kb) -> list[PatientRequirement]`

```python
@dataclass
class PatientRequirement:
    pid: str; priority: int; bed_type: str; specialty: str; waiting_since: int
```

## Contract 3 — Search output → Planning input

```python
@dataclass
class Allocation:
    assignments: dict[str, dict]   # {"P1": {"bed": "ICU1", "doctor": "D1"}, "P3": {"bed": None, "doctor": None}}  None = deferred
    cost: float
    stats: dict                    # {"algo": "A*", "expanded": 132, "time_ms": 4.1, "max_frontier": 40}
```

## Contract 4 — Planning output → Agent

```python
@dataclass
class Action:
    name: str                 # "ReserveICUBed"
    args: tuple               # ("P1", "ICU1")
    pre: frozenset[Fact]      # positive preconditions
    pre_neg: frozenset[Fact]  # must NOT hold
    add: frozenset[Fact]
    delete: frozenset[Fact]

Plan = list[Action]
def is_applicable(action, state: set[Fact]) -> bool
def validate(plan, state) -> tuple[bool, int | None]  # (ok, index of first failing action)
```

## Contract 5 — Events (dynamic environment)

```json
{"t": 3, "type": "patient_arrival", "patient": {...}}
{"t": 4, "type": "resource_unavailable", "id": "ICU1"}
{"t": 5, "type": "vitals_update", "pid": "P1", "vitals": {"hr": 140, "sbp": 80, "spo2": 85, "temp": 39.2}}
{"t": 6, "type": "resource_available", "id": "D2"}
```

## Contract 6 — Patient input JSON (UI / scenarios)

```json
{
  "id": "P1", "age": 72,
  "vitals": {"hr": 125, "sbp": 85, "spo2": 88, "temp": 39.0, "rr": 26},
  "history": [{"t":1,"hr":105,"sbp":110,"spo2":95}, ... ],
  "symptoms": ["ChestPain", "ShortnessOfBreath"]
}
```

---

## Shared fixture (everyone tests against it)

`scenarios/base_example.json` = the brief's end-to-end example:
P1 (72, chest pain, SoB, low BP, SpO2 88, deteriorating history), P2 (45, moderate fever, stable), P3 (30, abdominal pain, stable).
Resources: D1 (Cardiology), D2 (General), EB1, EB2, ICU1.
`scenarios/dynamic_icu_lost.json` = base + event "P4 critical arrives" + "ICU1 unavailable".

Expected final answer for base: P1 → ICU1 + D1, P2 → EB1 + D2, P3 → EB2 (or deferred, depending on cost fn).

---

## Milestones

| Date | Milestone | Gate |
|------|-----------|------|
| **Thu Nov 19** | repo + `contracts.py` + scenarios pushed | Amin ✅ |
| **Sat Nov 22** | Every module runs standalone on the fixture, with tests | each |
| **Tue Nov 25** | Happy path end-to-end: JSON → plan | Amin + all |
| **Fri Nov 28** | Dynamic scenarios + search/ML/DL experiments done | Amin, M4, M5 |
| **Sun Nov 30** | Report compiled, demo rehearsed twice | M5 + all |
| **Tue Dec 1** | Submit | — |

Daily: 5-min async standup in the group chat — *done / next / blocked*.

## Disclaimer (must appear in UI + report)
Educational simulation only. No real medical diagnosis or treatment recommendation.
