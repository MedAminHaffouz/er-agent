# er-agent — Integrated AI Agent for Emergency Management

GL4 Artificial Intelligence project, INSAT. **Deadline: Dec 1, 2026.**

> Educational simulation only. No real medical diagnosis or treatment recommendation.

One agent that cooperates across: knowledge representation → ML/DL risk evidence → logical reasoning
(unification, deduction, resolution) → search (resource allocation) → planning (action sequence) →
replanning when the environment changes.

## Setup

```bash
git clone <repo-url> && cd er-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Who owns what

| Folder | Owner | Block |
|--------|-------|-------|
| `er_agent/contracts.py`, `er_agent/agent/`, `er_agent/learning/dl/`, `ui/` | Amin (lead) | 3 (DL), 6 |
| `er_agent/world/`, `er_agent/planning/`, `scenarios/` | Member 2 | 1, 5 |
| `er_agent/logic/` | Member 3 | 2 |
| `er_agent/learning/ml/`, `er_agent/learning/data_gen.py` | Member 4 | 3 (ML + data) |
| `er_agent/search/`, `experiments/`, `report/` | Member 5 | 4 + report |

Full task lists: see `docs/` (one md per member) and `docs/00_shared_contracts.md`.

## Rules

1. **`contracts.py` is the API.** Code against it. Changing it = PR + lead review.
2. **Never push to `main`.** Branch `feat/<module>-<thing>` → PR → 1 review → merge.
3. **`main` must pass `pytest`.** CI runs on every PR.
4. Every module ships `tests/test_<module>.py` that runs on `scenarios/base_example.json`.
5. No LLM anywhere (project rule).
6. Late module? Commit a **stub with the right signature** so integration never blocks.
7. Daily async standup in the group chat: *done / next / blocked*.

## Milestones

| Date | Gate |
|------|------|
| Nov 22 | each module runs standalone on `base_example.json` |
| Nov 25 | end-to-end happy path: JSON → plan |
| Nov 28 | dynamic scenarios + experiments |
| Nov 30 | report compiled, demo rehearsed |
| Dec 1 | submit |
