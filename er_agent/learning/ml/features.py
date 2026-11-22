"""Contract 6 patient dict -> one tabular feature row (owner: Member 4).

Shared by training and inference so the agent never re-implements preprocessing.
Missing fields become NaN; the saved pipeline imputes them with the training median / mode.
"""
from __future__ import annotations

import math

from er_agent.learning.data_gen import (
    CATEGORICAL_FEATURES,
    NOISE_FEATURES,
    NUMERIC_FEATURES,
    SYMPTOM_FEATURES,
)

ALL_FEATURES = NUMERIC_FEATURES + SYMPTOM_FEATURES + CATEGORICAL_FEATURES + NOISE_FEATURES

# Contract 6 symptom names -> binary column
SYMPTOM_MAP = {
    "ChestPain": "chest_pain",
    "ShortnessOfBreath": "sob",
    "AbdominalPain": "abdominal_pain",
    "Fever": "fever_symptom",
}
CONFUSION_SYMPTOMS = {"Confusion", "AlteredConsciousness", "Unconscious"}

NAN = float("nan")


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def patient_to_row(patient: dict) -> tuple[dict, list[str]]:
    """Returns (row over ALL_FEATURES, list of fields that were missing and will be imputed)."""
    vitals = patient.get("vitals") or {}
    row: dict = {}
    missing: list[str] = []

    row["age"] = _num(patient.get("age"))
    for k in ("hr", "sbp", "dbp", "spo2", "temp", "rr"):
        row[k] = _num(vitals.get(k, patient.get(k)))           # nested (Contract 6) or flat

    symptoms = patient.get("symptoms")
    if symptoms is None:
        for col in SYMPTOM_FEATURES:
            row[col] = NAN
    else:
        present = set(symptoms)
        for name, col in SYMPTOM_MAP.items():
            row[col] = float(name in present)

    cons = patient.get("consciousness")
    if cons is None and symptoms is not None:
        cons = "confused" if CONFUSION_SYMPTOMS & set(symptoms) else "alert"
    row["consciousness"] = str(cons).lower() if cons is not None else NAN

    for col in NOISE_FEATURES:                                   # never in the deployed feature set
        row[col] = NAN

    for col in NUMERIC_FEATURES + SYMPTOM_FEATURES + CATEGORICAL_FEATURES:
        v = row[col]
        if isinstance(v, float) and math.isnan(v):
            missing.append(col)
    return row, missing
