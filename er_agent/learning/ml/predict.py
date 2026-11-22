"""Integration API for the agent (owner: Member 4) — implements ``contracts.MLPredictor``.

    from er_agent.learning.ml import predict_ml
    predict_ml(patient)   # Contract 6 patient dict -> ("High", 0.87)

The agent inserts the result as ``("ML_Risk", pid, label, prob)``.
The model bundle is ``models/ml_rf.joblib`` (build it with ``python -m er_agent.learning.ml.train``),
loaded once and cached. Override the path with the ``ER_ML_MODEL`` env var or ``load_model(path)``.

Educational simulation only. No real medical diagnosis or treatment recommendation.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from er_agent.learning.ml.features import patient_to_row
from er_agent.learning.ml.train import MODEL_PATH

log = logging.getLogger(__name__)


def _default_path() -> Path:
    return Path(os.environ.get("ER_ML_MODEL", MODEL_PATH))


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ML model not found at {p}. Build it with: python -m er_agent.learning.ml.train")
    return joblib.load(p)


def load_model(path: str | Path | None = None) -> dict:
    return _load(str(Path(path) if path else _default_path()))


def predict_proba_ml(patient: dict, model_path: str | Path | None = None) -> dict[str, float]:
    """Full class distribution {"Low": .., "Medium": .., "High": .., "Critical": ..}."""
    bundle = load_model(model_path)
    row, missing = patient_to_row(patient)
    used_missing = [f for f in missing if f in bundle["features"]]
    if used_missing:
        filled = {f: bundle["train_medians"].get(f, "mode") for f in used_missing}
        log.warning("predict_ml(%s): missing %s -> imputed with training median/mode %s",
                    patient.get("id", "?"), used_missing, filled)
    X = pd.DataFrame([row])[bundle["features"]]
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(X)[0]
    by_class = dict(zip(map(str, pipe.classes_), map(float, proba)))
    return {c: by_class.get(c, 0.0) for c in bundle["classes"]}


def predict_ml(patient: dict, model_path: str | Path | None = None) -> tuple[str, float]:
    """Contract 6 patient -> (risk label, probability of that label)."""
    dist = predict_proba_ml(patient, model_path)
    labels = list(dist)
    k = int(np.argmax([dist[c] for c in labels]))
    return labels[k], round(dist[labels[k]], 4)


def ml_risk_fact(patient: dict, model_path: str | Path | None = None) -> tuple:
    """Contract 1 fact the agent inserts into the KB: ("ML_Risk", pid, label, prob)."""
    label, p = predict_ml(patient, model_path)
    return ("ML_Risk", patient["id"], label, p)
