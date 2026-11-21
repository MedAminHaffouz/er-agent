"""Bonus: evaluate on REAL triage data (KTAS, Moon et al. 2019, CC BY 4.0) — owner: Member 4.

    python -m er_agent.learning.ml.real_eval    # needs models/ml_rf.joblib (train first) + internet once
        -> experiments/ml/real_results.{json,md}

Experiments (all scored against the expert KTAS label, mapped to our 4 classes):
  H  human reference    — triage nurse's KTAS vs expert KTAS
  A  real -> real       — DT / LR / RF, 3x repeated stratified 5-fold CV on the 1267 real visits, with the
                          Contract-6 features only ("shared") and with real-only extras ("full": pain score,
                          injury, ambulance arrival, AVPU level, dbp, + missing-value indicators)
  B  synthetic -> real  — the deployed model (trained only on synthetic data) applied as-is = sim-to-real gap
  C  synthetic + real   — same CV folds as A, synthetic rows added to each training fold
                          (C-weighted: real rows up-weighted so both sources carry the same total weight)

Analyses:
  * 95% bootstrap CIs (1000 resamples of patients) on macro-F1 and Critical recall for every row
  * why B fails: our generator's warning score vs. the expert KTAS level (Spearman), synthetic vs real
  * external validity: AUROC of P(High or Critical) for the hospital outcome "admitted / ICU / died"
    — a label none of the models were trained on
  * permutation importance of the best real-data model (which signals real triage uses)

The deployed model used by the agent is NOT changed by this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from er_agent.learning import data_gen, real_data
from er_agent.learning.data_gen import CLASSES
from er_agent.learning.ml.predict import load_model
from er_agent.learning.ml.train import (
    EXP_DIR,
    SEED,
    make_preprocessor,
    metrics_from_proba,
    proba_in_class_order,
)

MODELS = {
    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced",
                                           random_state=SEED),
    "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                           random_state=SEED, n_jobs=-1),
}
ADMITTED = {2, 3, 6}            # disposition: ward admission, ICU admission, death
N_BOOT = 1000


# ------------------------------------------------------------------------------------------ helpers
def cv_proba(model, X: pd.DataFrame, y: pd.Series, extra: tuple[pd.DataFrame, pd.Series] | None = None,
             real_weight: float = 1.0, n_repeats: int = 3) -> tuple[list[dict], np.ndarray]:
    """Repeated stratified 5-fold. Returns (metrics per repeat, out-of-fold proba of the first repeat)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    runs, proba, first = [], np.zeros((len(X), len(CLASSES))), None
    for k, (tr, te) in enumerate(cv.split(X, y)):
        X_tr, y_tr, fit_kw = X.iloc[tr], y.iloc[tr], {}
        if extra is not None:
            w = np.r_[np.full(len(tr), real_weight), np.ones(len(extra[1]))]
            X_tr, y_tr = pd.concat([X_tr, extra[0]]), pd.concat([y_tr, extra[1]])
            fit_kw = {"clf__sample_weight": w}
        proba[te] = proba_in_class_order(clone(model).fit(X_tr, y_tr, **fit_kw), X.iloc[te])
        if (k + 1) % 5 == 0:
            runs.append(metrics_from_proba(proba, y))
            if first is None:
                first = proba.copy()
            proba = np.zeros_like(proba)
    return runs, first


def bootstrap_ci(y: pd.Series, pred: np.ndarray, seed: int = SEED) -> dict:
    """95% percentile CIs over patient resamples for macro-F1 and Critical recall."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    f1s, recs = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        f1s.append(f1_score(y[i], pred[i], labels=CLASSES, average="macro", zero_division=0))
        if (y[i] == "Critical").any():
            recs.append(recall_score(y[i], pred[i], labels=["Critical"], average="macro", zero_division=0))
    return {"macro_f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "recall_critical_ci": [float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))]}


def admission_auroc(admitted: np.ndarray, proba: np.ndarray) -> float:
    """AUROC of P(High) + P(Critical) for the real outcome admitted/ICU/death."""
    return float(roc_auc_score(admitted, proba[:, CLASSES.index("High")] + proba[:, CLASSES.index("Critical")]))


def _pred(proba: np.ndarray) -> np.ndarray:
    return np.asarray(CLASSES)[proba.argmax(1)]


def summarize(runs: list[dict], oof: np.ndarray, y: pd.Series, admitted: np.ndarray) -> dict:
    keys = ["accuracy", "macro_f1", "macro_recall", "ece"]
    s = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    s.update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys})
    s["recall"] = {c: float(np.mean([r["recall"][c] for r in runs])) for c in CLASSES}
    s["confusion"] = runs[0]["confusion"]
    s.update(bootstrap_ci(y, _pred(oof)))
    s["admission_auroc"] = admission_auroc(admitted, oof)
    return s


def single(proba: np.ndarray, y: pd.Series, admitted: np.ndarray, pred: np.ndarray | None = None) -> dict:
    m = metrics_from_proba(proba, y, pred=pred)
    m.update(bootstrap_ci(y, _pred(proba) if pred is None else pred))
    m["admission_auroc"] = admission_auroc(admitted, proba)
    return m


# ------------------------------------------------------------------------------------------ main
def run(exp_dir: Path = EXP_DIR, synthetic_path: Path = data_gen.DATA_DIR / "tabular.csv") -> dict:
    real = real_data.load_ktas()
    y = real["risk"]
    admitted = real["disposition"].isin(ADMITTED).to_numpy().astype(int)
    bundle = load_model()
    shared = bundle["features"]                                   # exactly what the deployed model sees
    full = shared + [f for f in ["dbp", *real_data.EXTRA_FEATURES] if f not in shared]
    if not synthetic_path.exists():
        data_gen.generate(out_dir=synthetic_path.parent)
    syn = pd.read_csv(synthetic_path)

    res: dict = {
        "n_real": len(real),
        "class_distribution_real": y.value_counts(normalize=True).reindex(CLASSES).round(4).to_dict(),
        "class_distribution_synthetic": syn.risk.value_counts(normalize=True).reindex(CLASSES).round(4).to_dict(),
        "missing_rate_real": real[["hr", "sbp", "dbp", "spo2", "temp", "rr"]].isna().mean().round(3).to_dict(),
        "admitted_rate": float(admitted.mean()),
        "features_shared": shared, "features_full": full, "rows": {},
    }
    rows = res["rows"]

    # H: human reference. KTAS is ordinal -> score for AUROC = (6 - level), spread over a fake proba
    nurse = real["ktas_nurse"].map(real_data.KTAS_TO_RISK).to_numpy()
    onehot = np.eye(len(CLASSES))[[CLASSES.index(c) for c in nurse]]
    rows["H"] = single(onehot, y, admitted, pred=nurse)
    rows["H"]["admission_auroc"] = float(roc_auc_score(admitted, 6 - real["ktas_nurse"]))
    res["expert_ktas_admission_auroc"] = float(roc_auc_score(admitted, 6 - real["ktas_expert"]))

    # B: synthetic -> real
    rows["B"] = single(proba_in_class_order(bundle["pipeline"], real[shared]), y, admitted)
    print(f"B synthetic->real        macroF1={rows['B']['macro_f1']:.3f}")

    # A: real -> real
    for fs_name, fs in (("shared", shared), ("full", full)):
        for name, est in MODELS.items():
            pipe = Pipeline([("pre", make_preprocessor(fs, missing_indicator=fs_name == "full")), ("clf", est)])
            key = f"A|{name}|{fs_name}"
            rows[key] = summarize(*cv_proba(pipe, real[fs], y), y, admitted)
            print(f"{key:30s} macroF1={rows[key]['macro_f1']:.3f}")

    # C: synthetic + real -> real (plain and re-weighted)
    rf = Pipeline([("pre", make_preprocessor(shared)), ("clf", MODELS["RandomForest"])])
    extra = (syn[shared], syn["risk"])
    rows["C"] = summarize(*cv_proba(rf, real[shared], y, extra=extra), y, admitted)
    w = len(syn) / (0.8 * len(real))                              # equal total weight per source
    rows["C_weighted"] = summarize(*cv_proba(rf, real[shared], y, extra=extra, real_weight=w), y, admitted)
    print(f"C / C_weighted           macroF1={rows['C']['macro_f1']:.3f} / {rows['C_weighted']['macro_f1']:.3f}")

    # Why B fails: does our generator's label rule rank real patients like the experts do?
    syn_score = data_gen.tabular_score(syn)
    real_score = data_gen.tabular_score(real.fillna(real.median(numeric_only=True)))
    rank = {c: i for i, c in enumerate(CLASSES)}
    res["generator_score_vs_label_spearman"] = {
        "synthetic": float(pd.Series(syn_score).corr(syn.risk.map(rank), method="spearman")),
        "real": float(pd.Series(real_score).corr(y.map(rank), method="spearman")),
    }
    res["generator_score_by_class"] = {
        "synthetic": {c: float(np.median(syn_score[syn.risk == c])) for c in CLASSES},
        "real": {c: float(np.median(real_score[(y == c).to_numpy()])) for c in CLASSES},
    }

    # Which signals does real triage use? permutation importance of the best real model (RF full)
    X_tr, X_te, y_tr, y_te = train_test_split(real[full], y, test_size=0.3, stratify=y, random_state=SEED)
    best = Pipeline([("pre", make_preprocessor(full, missing_indicator=True)),
                     ("clf", MODELS["RandomForest"])]).fit(X_tr, y_tr)
    perm = permutation_importance(best, X_te, y_te, scoring="f1_macro", n_repeats=20, random_state=SEED, n_jobs=-1)
    res["real_permutation_importance"] = sorted(
        [{"feature": f, "mean": float(m), "std": float(s)}
         for f, m, s in zip(full, perm.importances_mean, perm.importances_std, strict=True)],
        key=lambda d: -d["mean"])

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "real_results.json").write_text(json.dumps(res, indent=2, default=float))
    (exp_dir / "real_results.md").write_text(to_markdown(res))
    print(f"saved {exp_dir}/real_results.{{json,md}}")
    return res


# ------------------------------------------------------------------------------------------ report
ROW_LABELS = {
    "H": "H — triage nurse vs expert (human reference)",
    "B": "B — deployed synthetic model, no retraining",
    "A|DecisionTree|shared": "A — Decision Tree, real CV, shared features",
    "A|LogisticRegression|shared": "A — Logistic Regression, real CV, shared",
    "A|RandomForest|shared": "A — Random Forest, real CV, shared",
    "A|DecisionTree|full": "A — Decision Tree, real CV, full features",
    "A|LogisticRegression|full": "A — Logistic Regression, real CV, full",
    "A|RandomForest|full": "A — Random Forest, real CV, full",
    "C": "C — RF synthetic + real, shared",
    "C_weighted": "C — RF synthetic + real (re-weighted), shared",
}


def to_markdown(r: dict) -> str:
    out = ["# Real-data evaluation — KTAS (Moon et al. 2019, PLOS ONE, CC BY 4.0)", "",
           ("Auto-generated by `python -m er_agent.learning.ml.real_eval`. Label = expert KTAS mapped "
           "1→Critical, 2→High, 3→Medium, 4/5→Low. 1267 adult ED visits, two Korean hospitals."), "",
           f"- class distribution real: {r['class_distribution_real']}",
           f"- class distribution synthetic: {r['class_distribution_synthetic']}",
           f"- missing rate per vital (real): {r['missing_rate_real']}",
           f"- admitted / ICU / died: {r['admitted_rate']:.1%}",
           f"- shared features (= deployed model): `{r['features_shared']}`",
           f"- full features: `{r['features_full']}` + missing-value indicators", "",
           "## Results", "",
           ("| experiment | acc | macro-F1 [95% CI] | recall Critical [95% CI] | recall L / M / H | ECE "
           "| AUROC admission |"),
           "|---|---|---|---|---|---|---|"]
    for key, label in ROW_LABELS.items():
        m = r["rows"][key]
        f1_ci, rc_ci = m["macro_f1_ci"], m["recall_critical_ci"]
        lmh = " / ".join(f"{m['recall'][c]:.2f}" for c in CLASSES[:3])
        ece = "—" if key == "H" else f"{m['ece']:.3f}"
        out.append(f"| {label} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} [{f1_ci[0]:.2f}, {f1_ci[1]:.2f}] | "
                   f"{m['recall']['Critical']:.2f} [{rc_ci[0]:.2f}, {rc_ci[1]:.2f}] | {lmh} | {ece} | "
                   f"{m['admission_auroc']:.3f} |")
    out += ["", f"Reference: the expert KTAS itself has AUROC {r['expert_ktas_admission_auroc']:.3f} for admission.",
            "", ("A/C: mean over 3 repeats of stratified 5-fold CV; CIs and AUROC from the first repeat's pooled "
            "out-of-fold predictions. Only 26 Critical patients → Critical recall CIs are wide."), "",
            "## Why the synthetic model does not transfer", "",
            ("Spearman correlation between our generator's warning score and the label: "
            f"synthetic {r['generator_score_vs_label_spearman']['synthetic']:.2f}, "
            f"real {r['generator_score_vs_label_spearman']['real']:.2f}."), "",
            "| class | median generator score (synthetic) | median generator score (real) |", "|---|---|---|"]
    for c in CLASSES:
        out.append(f"| {c} | {r['generator_score_by_class']['synthetic'][c]:.1f} | "
                   f"{r['generator_score_by_class']['real'][c]:.1f} |")
    out += ["", "## Permutation importance — Random Forest, full features, 30% held-out real data", "",
            "| feature | Δ macro-F1 when shuffled |", "|---|---|"]
    out += [f"| {d['feature']} | {d['mean']:.3f} ± {d['std']:.3f} |" for d in r["real_permutation_importance"]]
    out += ["", "Confusion matrix B (rows = true, cols = predicted, " + ", ".join(CLASSES) + "):", "", "```"]
    out += ["  ".join(f"{x:5d}" for x in row) for row in r["rows"]["B"]["confusion"]]
    out += ["```", ""]
    return "\n".join(out)


if __name__ == "__main__":
    run()
