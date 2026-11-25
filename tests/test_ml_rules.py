"""Member 4 bonus — decision tree -> Horn rules for the logic engine."""
import json
from pathlib import Path

import pandas as pd

from er_agent.contracts import is_var
from er_agent.learning import data_gen
from er_agent.learning.ml import rules_export as rx
from er_agent.learning.ml.features import patient_to_row

SCEN = Path(__file__).parent.parent / "scenarios"
WORLD_PREDICATES = {"LowOxygen", "Tachycardia", "LowBloodPressure", "Fever", "Elderly", "Tachypnea", "HasSymptom"}


def _facts_of(patient: dict) -> set[tuple]:
    """Contract-1 facts the World would emit for this patient (discretized, via the shared thresholds)."""
    row = pd.DataFrame([patient_to_row(patient)[0]])
    present = rx.discretize(row).iloc[0]
    return {tuple(patient["id"] if t == rx.X else t for t in rx.FACTS[name][0]) for name in present.index
            if present[name]}


def _fires(premises: list[tuple], facts: set[tuple], pid: str) -> bool:
    """Minimal single-variable matcher: bind ?x to pid and check every premise is a known fact."""
    return all(tuple(pid if is_var(t) else t for t in p) in facts for p in premises)


def test_shipped_rules_are_valid_horn_clauses():
    rules = rx.load_learned_rules()
    assert rules, "learned_rules.json is empty"
    names = [n for n, _, _ in rules]
    assert len(set(names)) == len(names) and all(n.startswith("Learned_R") for n in names)
    for _, prem, concl in rules:
        assert concl in {("HighRisk", "?x"), ("CriticalRisk", "?x")}
        for p in prem:
            assert isinstance(p, tuple) and p[0] in WORLD_PREDICATES and p[1] == "?x"
        assert {v for p in prem for v in p if is_var(v)} == {"?x"}        # range-restricted, one variable


def test_shipped_rules_meet_the_kb_confidence_threshold():
    meta = json.loads(rx.RULES_PATH.read_text())["rules"]
    assert all(r["val_precision"] >= rx.MIN_PRECISION and r["support_val"] >= rx.MIN_SUPPORT for r in meta)


def test_rules_on_the_brief_patients():
    pts = {p["id"]: p for p in json.loads((SCEN / "base_example.json").read_text())["patients"]}
    rules = rx.load_learned_rules()
    derived = {pid: {c for _, prem, c in rules if _fires(prem, _facts_of(p), pid)} for pid, p in pts.items()}
    assert ("HighRisk", "?x") in derived["P1"]           # chest pain + tachycardia + low BP + tachypnea
    assert not derived["P3"]                             # stable abdominal pain


def test_shipped_rules_are_in_sync_with_the_generator(tmp_path):
    data_gen.make_tabular(8000, seed=42).to_csv(tmp_path / "tabular.csv", index=False)
    rx.run(tmp_path / "tabular.csv", tmp_path / "rules.json", tmp_path / "exp", seed=42, with_real=False)
    assert json.loads((tmp_path / "rules.json").read_text()) == json.loads(rx.RULES_PATH.read_text()), \
        "learned_rules.json is stale: run python -m er_agent.learning.ml.rules_export"


def test_negated_branches_are_dropped_and_lower_bound_filters():
    assert rx.wilson_lower(0.72, 25) < 0.6 < rx.wilson_lower(0.86, 72)
    df = data_gen.make_tabular(3000, seed=1)
    f, y = rx.discretize(df), df.risk
    rules = rx.extract_rules(f.iloc[:2000], y.iloc[:2000], f.iloc[2000:], y.iloc[2000:])
    for r in rules:
        assert r["val"]["precision"] >= rx.MIN_PRECISION and r["val"]["precision_lower95"] >= rx.MIN_LOWER_BOUND
        assert set(r["premises"]) <= set(rx.FACTS)


def test_forward_chain_reaches_fixpoint_through_derived_premises():
    f = pd.DataFrame({k: [False, False] for k in rx.FACTS})
    f.loc[0, ["ChestPain", "LowBloodPressure", "Elderly", "LowOxygen"]] = True
    out = rx.forward_chain(rx.HAND_RULES, f)              # R1 -> HighRisk, then R6 -> CriticalRisk
    assert out.loc[0, "HighRisk"] and out.loc[0, "CriticalRisk"] and not out.loc[1].HighRisk

