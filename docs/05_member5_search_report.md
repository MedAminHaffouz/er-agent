# 05 — Member 5 · Search & Resource Allocation (Block 4) + Report Assembly

**Code:** `er_agent/search/`, `experiments/search_bench.py`, `report/` (main.tex)
**Report sections:** Block 4 + report editor (assembly, consistency, figures, bibliography)
**Course part:** search (uninformed + informed, heuristics)

---

## 1. Problem formulation (write this first, it's half the grade of Block 4)

Input: `list[PatientRequirement]` (Contract 2) + available resources from World.
Output: `Allocation` (Contract 3).

- [ ] **State**: tuple of per-patient assignment `(pid → (bed|None, doctor|None) | Unassigned)` + remaining resources. Make it **hashable** (frozenset / tuple) for the explored set.
- [ ] **Initial state**: all patients unassigned, all available resources free
- [ ] **Actions** (process patients in priority order to keep branching sane):
  - `Assign(P, bed, doctor)` — compatible bed type & specialty (or fallback: General doctor for any)
  - `Defer(P)` — patient stays waiting (with penalty)
- [ ] **Transition model**: resource leaves the free pool
- [ ] **Goal test**: every patient decided; every Critical patient has an ICU bed **or** is explicitly deferred only if no ICU exists; no resource assigned twice
- [ ] **Cost function** (justify each term in the report):
  ```
  step_cost = w_defer[priority]          if Defer   (e.g. 100, 30, 10, 3 for prio 4..1)
            + w_mismatch                 if non-ideal specialty / bed downgrade
            + w_waste                    if ICU used for non-critical
            + w_wait * waiting_since     (older waiting → costlier to defer)
  ```

## 2. Algorithms
- [ ] **Uniform Cost Search** (priority queue on g, explored set, path reconstruction)
- [ ] **A\*** with your heuristic h
- [ ] Optional 3rd: Greedy best-first or BFS → shows why optimality matters
- [ ] Stats in `Allocation.stats`: expanded nodes, generated nodes, max frontier, time, solution cost

## 3. Heuristic (must be designed + justified)
- [ ] h(n) = Σ over undecided patients of **min possible step cost** given remaining resources (e.g. 0 if a compatible free bundle exists, else `w_defer[prio]`)
- [ ] **prove admissibility** (never overestimates: each undecided patient costs at least its min) + discuss consistency
- [ ] optional second heuristic (e.g. relaxed: ignore doctor constraints) → compare h1 vs h2

## 4. Experiments (`experiments/search_bench.py`)
- [ ] generator of random instances: n patients ∈ {3, 5, 8, 10, 12}, varying scarcity
- [ ] for each: UCS vs A*(h1) vs A*(h2) → nodes expanded, time, cost (same cost = both optimal ✔)
- [ ] plots: nodes expanded vs n (log scale), time vs n
- [ ] scenario check on `base_example.json` and `dynamic_second_critical` (2 critical, 1 ICU) → explain the chosen allocation
- [ ] quality discussion: critical patients served %, average weighted waiting

## 5. Integration
```python
def search(reqs: list[PatientRequirement], resources: dict, algo="astar") -> Allocation
```
- [ ] must accept a re-search call mid-scenario (agent replans with fewer resources / more patients)
- [ ] no logic duplication: requirements come **only** from the KB (Block 2) — never re-derive risk here

## 6. Report assembly (you're the editor)
- [ ] **Nov 20**: `report/main.tex` skeleton, one `\input{sections/xx}` per member, shared `figures/`, bib
- [ ] Structure: Intro & objectives → Architecture (Amin) → B1 → B2 → B3 → B4 → B5 → B6 → Scenarios & results → Limitations → Conclusion → Disclaimer
- [ ] Consistency pass: same predicate names, same notation, same scenario across sections
- [ ] **Nov 30**: final compile, page check, PDF to the team for last read

## Done when
- on `base_example.json`: A* and UCS return the **same optimal cost**; A* expands fewer nodes
- benchmark table + 2 plots in `report/figures/`
- `tests/test_search.py` green (goal test, no double assignment, admissibility check on small cases by brute force)

## Deadlines
| Date | Deliverable |
|------|-------------|
| Nov 20 | formulation written (md/tex) + report skeleton |
| Nov 22 | UCS + A* on fixture, `search()` merged |
| Nov 25 | heuristic proof + h2 + tests |
| Nov 28 | benchmarks + plots |
| Nov 30 | report compiled |
