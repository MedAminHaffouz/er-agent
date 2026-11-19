# 01 — Amin · Lead · Deep Learning · Block 6 Integration

**Blocks:** 3 (DL half), 6 (all) + contracts + demo UI
**Code:** `er_agent/contracts.py`, `er_agent/learning/dl/`, `er_agent/agent/`, `ui/app.py`
**Report sections:** Architecture, DL component, Integration & dynamic environment, Demo walkthrough

---

## 1. Contracts (Day 1 — blocks everyone)

- [ ] **Nov 19** ✅ — write `contracts.py` (Fact, PatientRequirement, Allocation, Action, Plan, Event dataclasses) exactly as in `00_shared_contracts.md`
- [ ] **Nov 19** ✅ — write `scenarios/base_example.json` + `dynamic_icu_lost.json` (+ `dynamic_deterioration.json`)
- [ ] **Nov 19** ✅ — repo skeleton, `pyproject.toml`, `pytest` CI (GitHub Actions), branch rules
- [ ] Unblock people: review every PR touching an interface within 12h

## 2. Deep Learning — temporal risk (Block 3, DL half)

Input: vitals history `[T, F]`, F = (hr, sbp, spo2, temp, rr), T = 5–12 steps (padded/masked).
Output: `predict_dl(history) -> (label, prob)` → agent inserts `("DL_Risk", pid, label, prob)`.

- [ ] Get temporal dataset from M4 (`data/temporal.npz`, labels = risk at last step / trajectory class)
- [ ] Baseline: last-step features → logistic regression (to prove the sequence actually matters)
- [ ] Models: **GRU** and **LSTM** (1–2 layers, hidden 32–64, masking for variable length); compare
- [ ] Optional: 1D-CNN or tiny Transformer encoder for a 3rd column in the table
- [ ] Metrics: macro-F1, per-class recall (Critical recall is what matters), confusion matrix, calibration (reliability plot / ECE)
- [ ] Key experiment: **same final vitals, different trajectories** (stable vs deteriorating) → ML says same thing, DL differs. This is the figure that justifies DL vs ML in the report.
- [ ] Save `models/dl_gru.pt` + `scaler.json`; inference must run on CPU < 50 ms

## 3. Agent loop (Block 6)

```
perceive(event/JSON) → World.update()           [M2]
  → ML on current vitals, DL on history         [M4, Amin]
  → KB.reset_from(world.to_facts() + ML/DL facts)
  → KB.forward_chain() (+ resolution queries)   [M3]
  → requirements_from_kb(kb)                    [Amin adapter]
  → search(requirements, resources)             [M5]
  → plan(state, allocation)                     [M2]
  → execute step by step; before each step: is_applicable(action, state)?
       no  → REPLAN (full cycle from perceive)
  → on event: re-validate remaining plan → replan if invalid
```

- [ ] `Agent.step(event=None)` and `Agent.run(scenario)`
- [ ] `requirements_from_kb()` adapter (Contract 2)
- [ ] **Decision log**: every cycle records `{t, new_facts, ml, dl, derived, allocation, plan, reason_for_replan}` → JSON. This log *is* the demo and the report evidence.
- [ ] Replan triggers: (a) failed precondition, (b) new patient, (c) resource lost, (d) risk level change after vitals update
- [ ] Execution simulator: each action takes 1 tick; events injected at given `t`
- [ ] Monitoring loop: `StartMonitoring` → new vitals → DL re-run → if Critical, re-reason

## 4. Dynamic scenarios (must demo ≥ 1, aim for 3)

| Scenario | Trigger | Expected adaptation |
|----------|---------|---------------------|
| `dynamic_icu_lost` | ICU1 unavailable before transfer | P1 re-allocated (EB + monitoring / wait), plan rebuilt |
| `dynamic_second_critical` | P4 critical arrives, 1 ICU | search compares P1 vs P4 by priority/cost → one gets ICU, other escalated |
| `dynamic_deterioration` | P2 vitals degrade | DL flips P2 → Critical → new requirement → realloc |

## 5. UI (Streamlit, Block 1 interface + demo)

- [ ] Form to add patient (→ JSON → World), buttons for events (lose ICU, add patient, update vitals)
- [ ] Panels left→right: **Input → ML/DL predictions → KB new facts → derived facts + trace → allocation (A* vs UCS stats) → plan (step cursor)**
- [ ] Highlight in red what changed after an event (diff of facts / allocation / plan)
- [ ] Disclaimer banner

## 6. Done when
- `python -m er_agent.agent scenarios/base_example.json` prints a full log and a valid plan
- all 3 dynamic scenarios replan correctly, tested in `tests/test_agent.py`
- demo runs end-to-end in < 5 min, rehearsed twice (Nov 30)

## Dependencies
- Needs: M2 `World` + planner, M3 `KB`, M4 datasets + `predict_ml`, M5 `search`
- Blocks: everyone (contracts on Nov 19 — done)

## Risk notes
- If a module is late on Nov 22 → plug a stub with the same signature, keep integrating. Never wait.
- Keep the DL honest: synthetic data makes 99% accuracy trivial. Add noise / overlapping trajectories so results are discussable.
