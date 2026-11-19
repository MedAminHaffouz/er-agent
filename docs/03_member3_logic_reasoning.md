# 03 — Member 3 · Logical Reasoning, Deduction, Unification, Resolution (Block 2)

**Code:** `er_agent/logic/`
**Report section:** Block 2 (KB design, rules, unification, forward chaining, resolution, traces, ML/DL integration)
**Course part:** deduction (FOL, unification, resolution — the GL3 Logic Programming background helps a lot)

Implement the engine **yourself in Python** (the brief asks to *demonstrate* unification and resolution — a library hides exactly what's graded). Optional bonus: mirror the rule base in Prolog to cross-check answers.

---

## 1. Data structures (`logic/terms.py`)
- [ ] Terms: constants (`"P1"`, `125`, `0.91`), variables (`"?x"`), atoms = tuples (Contract 1)
- [ ] `Rule(premises: list[Fact], conclusion: Fact, name: str)` — Horn clauses
- [ ] `KB`: `add_fact`, `add_rule`, `facts`, `rules`, `reset_from(facts)`, `derived_facts()` (with provenance)

## 2. Unification (`logic/unify.py`)
- [ ] `unify(x, y, theta) -> dict | None` with **occurs check**
- [ ] `substitute(term, theta)`, `standardize_apart(rule)` (rename vars per use)
- [ ] Trace mode: log each binding step → `{"?x": "P1"}` shown in the UI
- [ ] Tests: `RequiresSpecialist(?x,Cardiology)` vs `RequiresSpecialist(P1,Cardiology)` → `{?x: P1}`; failure cases (constant clash, arity, occurs check)

## 3. Deduction — forward chaining (`logic/forward.py`)
- [ ] `forward_chain(kb) -> Trace` until fixpoint (no new facts)
- [ ] Each derived fact stores **provenance**: rule name + the facts that matched + θ
- [ ] `explain(fact)` → proof tree (recursive over provenance) — this is the "inference trace" the brief asks for

## 4. Resolution (`logic/resolution.py`)
- [ ] Convert rules/facts to **clausal form** (Horn → disjunction of literals)
- [ ] `ask(query) -> (bool, trace)` by **refutation**: add ¬query, resolve until empty clause □ or no new clauses
- [ ] Support queries with variables: `RequiresICU(?p)?` → return bindings (answer extraction)
- [ ] Strategy: set-of-support (start from ¬query) + depth/step limit to avoid blowup
- [ ] Trace printout, e.g.:
  ```
  1. ¬RequiresICU(P1)                         [negated goal]
  2. ¬CriticalRisk(x) ∨ RequiresICU(x)        [R7]
  3. ¬CriticalRisk(P1)                        [1,2  θ={x/P1}]
  4. ¬DL_Risk(x,Critical,p) ∨ CriticalRisk(x) [R5]
  5. ¬DL_Risk(P1,Critical,p)                  [3,4  θ={x/P1}]
  6. DL_Risk(P1,Critical,0.91)                [fact]
  7. □                                         [5,6  θ={p/0.91}]
  ```

## 5. Rule base (`logic/rules.py`) — the domain knowledge

Must derive the Contract 2 predicates (`RiskLevel`, `Priority`, `RequiresBed`, `RequiresSpecialist`, `RequiresUrgentEvaluation`). Starter set:

```
R1  HasSymptom(x,ChestPain) ∧ LowBloodPressure(x)        → HighRisk(x)
R2  LowOxygen(x) ∧ Tachycardia(x)                        → HighRisk(x)
R3  HighRisk(x)                                          → RequiresUrgentEvaluation(x)
R4  ML_Risk(x,High,p) ∧ HasSymptom(x,ChestPain)          → HighRisk(x)
R5  DL_Risk(x,Critical,p)                                → CriticalRisk(x)
R6  HighRisk(x) ∧ Elderly(x) ∧ LowOxygen(x)              → CriticalRisk(x)
R7  CriticalRisk(x)                                      → RequiresICU(x)
R8  RequiresICU(x)                                       → RequiresBed(x,ICU)
R9  HasSymptom(x,ChestPain)                              → RequiresSpecialist(x,Cardiology)
R10 CriticalRisk(x) → RiskLevel(x,Critical) ; HighRisk(x) ∧ ¬CriticalRisk(x) → RiskLevel(x,High) ...
```

- [ ] **Confidence threshold**: only accept `ML_Risk/DL_Risk` if `p ≥ 0.7` → needs a builtin `GreaterEq(p,0.7)` evaluated during matching (only builtin allowed; everything else symbolic)
- [ ] **Conflict policy** ML vs DL vs rules (e.g. ML=High, DL=Critical) → take max severity; document it
- [ ] Defaults (`RiskLevel=Low`, `RequiresBed=ExamRoom`, `RequiresSpecialist=General`) — negation-as-failure only in this final layer, document it as a closed-world assumption
- [ ] Priority mapping: Critical→4, High→3, Medium→2, Low→1

## 6. Integration hooks
- [ ] `KB.reset_from(facts)` must be cheap — agent rebuilds KB each cycle
- [ ] `kb.query_all(pattern)` → list of θ (used by Amin's adapter to extract requirements)
- [ ] Rule `CriticalRisk → RequiresICU` must work whether `CriticalRisk` came from rules *or* from DL → show both in report

## Done when
- on `base_example.json` + `ML_Risk(P1,High,0.87)` + `DL_Risk(P1,Critical,0.91)`: forward chaining derives `RequiresICU(P1)`, `RequiresSpecialist(P1,Cardiology)`, `Priority(P1,4)`; resolution proves `RequiresICU(P1)` with full trace
- ≥ 4 worked examples (unification, deduction, resolution success, resolution failure) in the report
- `tests/test_logic.py` green

## Deadlines
| Date | Deliverable |
|------|-------------|
| Nov 20 | unify + forward chaining + provenance |
| Nov 22 | rule base on fixture, Contract 2 predicates derived |
| Nov 24 | resolution refutation + trace + answer extraction |
| Nov 27 | edge cases, conflict policy, tests |
| Nov 29 | report section |
