# 02 — Member 2 · World Model & KR (Block 1) + Planning (Block 5)

**Code:** `er_agent/world/`, `er_agent/planning/`, `scenarios/` (with Amin)
**Report sections:** Block 1 (environment, PEAS, state, objectives), Block 5 (planning, search vs planning)
**Course part:** knowledge representation, planning (STRIPS — if not covered yet in class, read the planning chapter of the course book early)

You own the **state** of the world and the **actions** that change it. Same action schema serves both the planner and the agent's executor.

---

## Part A — Block 1: Environment & Knowledge Representation

### A1. Agent definition (report, short)
- [ ] **PEAS** table (Performance, Environment, Actuators, Sensors)
- [ ] Environment properties: partially observable? dynamic, sequential, discrete, single-agent — justify each
- [ ] Agent type: model-based + goal/utility-based → justify
- [ ] Agent objectives list (used by Search + Planning): prioritize critical, satisfy requirements, minimize waiting, no double assignment, efficient resource use

### A2. Entities & state
- [ ] Dataclasses: `Patient`, `Doctor`, `Nurse`, `Bed` (type ICU/Emergency/ExamRoom), `Equipment`, `Assignment`
- [ ] Patient status: `Waiting | UnderEvaluation | InICU | InEmergencyBed | Monitoring | Discharged`
- [ ] `World` class:
  - `from_json(path)`, `to_json()`
  - `to_facts() -> set[Fact]` (Contract 1, incl. **discretized** facts via `thresholds.py`)
  - `apply_event(event)` (Contract 5) — arrival, resource (un)available, vitals_update
  - `apply_action(action)` — add/delete effects on the fact set + entity objects stay in sync
  - `snapshot()` / `diff(other)` → used by UI to show what changed
- [ ] Input validation (bad JSON → clear error), unique IDs

### A3. Scenarios (with Amin)
- [ ] `base_example.json` — the brief's P1/P2/P3 example
- [ ] 2 more static: overload (6 patients, 1 ICU, 2 doctors), specialist mismatch (needs Cardiology, only General available)

## Part B — Block 5: Planning

### B1. Action schemas (STRIPS-style, Contract 4)

| Action | Pre | Add | Delete |
|--------|-----|-----|--------|
| `AssignDoctor(P,D)` | PatientWaiting(P), Available(D) | DoctorAssigned(P,D) | Available(D) |
| `PerformUrgentEvaluation(P,D)` | DoctorAssigned(P,D), RequiresUrgentEvaluation(P) | Evaluated(P) | PatientWaiting(P) |
| `ReserveICUBed(P,B)` | CriticalRisk(P), Available(B), Type(B,ICU) | Reserved(B,P) | Available(B) |
| `ReserveBed(P,B)` | Available(B), Type(B,Emergency) | Reserved(B,P) | Available(B) |
| `TransferPatient(P,B)` | Reserved(B,P), Evaluated(P) | PatientIn(P,B) | — |
| `StartMonitoring(P)` | PatientIn(P,B) | Monitoring(P) | — |
| `ReEvaluate(P,D)` | Monitoring(P), DoctorAssigned(P,D) | ReEvaluated(P) | — |
| `ReleaseDoctor(P,D)` | DoctorAssigned(P,D), ReEvaluated(P) | Available(D) | DoctorAssigned(P,D) |

Use **schemas with variables** + grounding (instantiate over current objects) — don't hand-write ground actions.

### B2. Goal from allocation
- [ ] `goals_from_allocation(alloc, kb) -> set[Fact]`
  e.g. P1→ICU1,D1 ⇒ `{PatientIn(P1,ICU1), Monitoring(P1), ReEvaluated(P1)}`

### B3. Planner
- [ ] **Forward state-space planner** (BFS + A* with h = #unsatisfied goal facts) — required
- [ ] Optional: backward (regression) planner or GraphPlan-lite for comparison
- [ ] `plan(state, goals) -> Plan | None`
- [ ] `validate(plan, state)` simulates the plan and returns the first failing index (agent uses it to detect invalid plans after events)
- [ ] Plan printout with pre/effects per step (for the trace panel)

### B4. Demonstrate correctness
- [ ] Test: cannot `TransferPatient(P1,ICU1)` before `ReserveICUBed`
- [ ] Test: cannot assign a busy doctor
- [ ] Test: after `ICU1` becomes unavailable, `validate()` fails at the reserve/transfer step
- [ ] Test: unreachable goal → returns `None` (agent must then re-search)

### B5. Report — Search vs Planning (required by the brief)
One clear paragraph + example: Search (M5) picks the **target configuration** (*who gets what*), Planning picks the **ordered action sequence** that reaches it under preconditions/effects (*how and in what order*).

## Done when
- `World.from_json("scenarios/base_example.json").to_facts()` matches the expected fact list (test)
- planner returns a valid plan for the base allocation in < 1 s
- all tests in `tests/test_world.py`, `tests/test_planning.py` pass

## Deadlines
| Date | Deliverable |
|------|-------------|
| Nov 19 | `World` + `to_facts()` + discretization merged (M3 & M4 need it) |
| Nov 21 | events + `apply_action` |
| Nov 23 | action schemas + forward planner + validate |
| Nov 26 | planner A* + tests + extra scenarios |
| Nov 29 | report sections |
