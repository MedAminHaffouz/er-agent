"""Member 4 — datasets + ML risk model. Trains a small model in a tmp dir (models/ and data/ are gitignored)."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from er_agent.learning import data_gen
from er_agent.learning.ml import predict_ml, predict_proba_ml
from er_agent.learning.ml.features import patient_to_row
from er_agent.learning.ml.train import run

SCEN = Path(__file__).parent.parent / "scenarios"


@pytest.fixture(scope="module")
def base():
    return {p["id"]: p for p in json.loads((SCEN / "base_example.json").read_text())["patients"]}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("ml")
    data_gen.make_tabular(n=3000, seed=0).to_csv(d / "tabular.csv", index=False)
    run(d / "tabular.csv", d / "ml_rf.joblib", d / "exp", seed=0, quick=True)
    return d / "ml_rf.joblib"


# ------------------------------------------------------------------ datasets
def test_tabular_schema_imbalance_and_noise():
    df = data_gen.make_tabular(n=4000, seed=42)
    assert list(df.columns) == data_gen.TABULAR_COLUMNS
    dist = df.risk.value_counts(normalize=True)
    assert dist["Low"] > dist["Medium"] > dist["High"] > dist["Critical"] > 0.02     # imbalanced, all present
    assert 0.04 < df.attrs["label_noise_mask"].mean() < 0.10


def test_generators_are_seeded():
    pd.testing.assert_frame_equal(data_gen.make_tabular(500, seed=7), data_gen.make_tabular(500, seed=7))
    a, b = data_gen.make_temporal(200, seed=7, n_pairs=20), data_gen.make_temporal(200, seed=7, n_pairs=20)
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["y"], b["y"])


def test_temporal_shapes_mask_and_types():
    d = data_gen.make_temporal(n=300, seed=1, n_pairs=30)
    N, T, F = d["X"].shape
    assert (N, F) == (300, 5) and T == 12 and list(d["features"]) == data_gen.SEQ_FEATURES
    assert np.array_equal(d["mask"].sum(1), d["lengths"]) and d["lengths"].min() >= 5
    assert np.all(d["X"][~d["mask"]] == 0)                                           # post-padding with zeros
    assert set(np.unique(d["traj_type"])) == set(range(len(data_gen.TRAJ_TYPES)))
    assert set(np.unique(d["y"])) <= {0, 1, 2, 3}


def test_temporal_pairs_same_last_step_different_trend():
    d = data_gen.make_temporal(n=300, seed=1, n_pairs=50)
    differ = 0
    for k in range(50):
        i, j = np.flatnonzero(d["pair_id"] == k)
        assert np.array_equal(d["X"][i, d["lengths"][i] - 1], d["X"][j, d["lengths"][j] - 1])
        differ += d["y"][i] != d["y"][j]
    assert differ / 50 > 0.6          # the trend changes the label for most pairs -> DL must beat last-step ML


def test_brief_patients_get_expected_generator_labels(base):
    rows = [patient_to_row(base[p])[0] for p in ("P1", "P2", "P3")]
    cls = data_gen.score_to_class(data_gen.tabular_score(pd.DataFrame(rows)), data_gen.TAB_CUTS)
    assert [data_gen.CLASSES[c] for c in cls] == ["Critical", "Medium", "Low"]


# ------------------------------------------------------------------ features
def test_patient_to_row_maps_symptoms_and_reports_missing(base):
    row, missing = patient_to_row(base["P1"])
    assert row["chest_pain"] == 1 and row["sob"] == 1 and row["abdominal_pain"] == 0
    assert row["consciousness"] == "alert" and row["hr"] == 132
    assert missing == ["dbp"]                                    # Contract 6 has no dbp
    _, missing = patient_to_row({"id": "X", "vitals": {"hr": 100}})
    assert "age" in missing and "chest_pain" in missing


# ------------------------------------------------------------------ model + API
def test_predict_ml_on_brief(model_path, base):
    label, p = predict_ml(base["P1"], model_path)
    assert label in ("High", "Critical") and 0.5 <= p <= 1.0
    label, p = predict_ml(base["P3"], model_path)
    assert label in ("Low", "Medium")
    dist = predict_proba_ml(base["P2"], model_path)
    assert list(dist) == data_gen.CLASSES and abs(sum(dist.values()) - 1) < 1e-6


def test_predict_ml_handles_missing_fields(model_path, caplog):
    with caplog.at_level(logging.WARNING):
        label, p = predict_ml({"id": "PX", "vitals": {"hr": 150, "spo2": 82}}, model_path)
    assert label in data_gen.CLASSES and 0 <= p <= 1
    assert "imputed" in caplog.text


def test_training_artifacts(model_path):
    res = json.loads((model_path.parent / "exp" / "results.json").read_text())
    assert not set(data_gen.NOISE_FEATURES) & set(res["selected_features"])          # noise rejected
    assert len(res["comparison"]) == 8                                              # 4 models x {none, balanced}
    assert res["deployed"]["test"]["macro_f1"] > 0.5


# ------------------------------------------------------------------ real data (bonus) — no network needed
def test_clean_ktas_maps_schema_and_labels():
    from er_agent.learning.real_data import clean_ktas
    raw = pd.DataFrame({
        "Age": [70, 30], "HR": [120, "측불"], "SBP": [85, 120], "DBP": [50, 80], "Saturation": [88, np.nan],
        "BT": [39.0, 36.5], "RR": [26, 16], "Chief_complain": ["Chest pain, dyspnea", "abd pain"],
        "Mental": [2, 1], "NRS_pain": [np.nan, 5], "Injury": [1, 2], "Arrival mode": [2, 3],
        "KTAS_expert": [1, 4], "KTAS_RN": [2, 4], "Disposition": [3, 1],
    })
    df = clean_ktas(raw)
    assert df.risk.tolist() == ["Critical", "Low"]
    assert np.isnan(df.hr[1]) and df.hr[0] == 120                       # '측불' (unmeasurable) -> NaN
    assert df.chest_pain.tolist() == [1, 0] and df.sob.tolist() == [1, 0] and df.abdominal_pain.tolist() == [0, 1]
    assert df.consciousness.tolist() == ["confused", "alert"]
    assert set(data_gen.TABULAR_COLUMNS) - set(data_gen.NOISE_FEATURES) - {"dbp"} <= set(df.columns)


# ------------------------------------------------------------------ integration: what the agent will do
def _replay(name: str) -> list[tuple[int, dict]]:
    """Every (t, patient snapshot) the agent would score in a scenario, following Contract 5 events."""
    scen = json.loads((SCEN / name).read_text())
    base_s = json.loads((SCEN / scen["extends"]).read_text()) if "extends" in scen else scen
    patients = {p["id"]: json.loads(json.dumps(p)) for p in base_s["patients"]}
    snaps = [(0, p) for p in patients.values()]
    for e in scen.get("events", []):
        if e["type"] == "patient_arrival":
            patients[e["patient"]["id"]] = e["patient"]
            snaps.append((e["t"], e["patient"]))
        elif e["type"] == "vitals_update":
            p = json.loads(json.dumps(patients[e["pid"]]))
            p["vitals"] = {**p["vitals"], **e["vitals"]}
            p.setdefault("history", []).append({"t": e["t"], **e["vitals"]})
            patients[e["pid"]] = p
            snaps.append((e["t"], p))
    return snaps


@pytest.mark.parametrize("scenario", sorted(p.name for p in SCEN.glob("*.json")))
def test_ml_risk_facts_on_every_scenario(model_path, scenario):
    from er_agent.contracts import RISK_ORDER
    from er_agent.learning.ml import ml_risk_fact
    for _, patient in _replay(scenario):
        fact = ml_risk_fact(patient, model_path)
        assert len(fact) == 4 and fact[0] == "ML_Risk" and fact[1] == patient["id"]
        assert fact[2] in RISK_ORDER and isinstance(fact[3], float) and 0.0 <= fact[3] <= 1.0


def test_dynamic_scenarios_change_ml_risk(model_path):
    from er_agent.contracts import RISK_ORDER
    snaps = _replay("dynamic_deterioration.json")
    p2 = [predict_ml(p, model_path)[0] for _, p in snaps if p["id"] == "P2"]
    assert RISK_ORDER[p2[-1]] >= RISK_ORDER["High"] > RISK_ORDER[p2[0]]   # deterioration is picked up
    p4 = next(p for _, p in _replay("dynamic_icu_lost.json") if p["id"] == "P4")
    assert predict_ml(p4, model_path)[0] in ("High", "Critical")
