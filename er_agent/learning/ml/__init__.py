"""Tabular ML risk prediction (owner: Member 4). Public API: ``predict_ml(patient) -> (label, prob)``."""
from er_agent.learning.ml.predict import load_model, ml_risk_fact, predict_ml, predict_proba_ml

__all__ = ["load_model", "ml_risk_fact", "predict_ml", "predict_proba_ml"]
