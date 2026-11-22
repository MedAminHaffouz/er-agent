"""Train, compare and deploy the tabular risk models (owner: Member 4) — Block 3, ML half.

    python -m er_agent.learning.ml.train            # regenerates data if missing, writes:
        models/ml_rf.joblib                 deployed bundle (pipeline + metadata) used by predict_ml
        experiments/ml/results.json         every number below, machine-readable
        experiments/ml/results.md           comparison tables for the report

Steps: stratified 70/15/15 split (seed 42) -> feature selection (MI + permutation importance vs. noise columns)
-> DT / RF / LR / HistGB, each with and without class_weight="balanced" -> 5-fold GridSearchCV on the RF
(macro-F1) -> calibration check (ECE / Brier on val, CalibratedClassifierCV if it helps) -> test once.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from er_agent.learning import data_gen
from er_agent.learning.data_gen import (
    CATEGORICAL_FEATURES,
    CLASSES,
    NOISE_FEATURES,
    SYMPTOM_FEATURES,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "ml_rf.joblib"
EXP_DIR = ROOT / "experiments" / "ml"
SEED = 42


# ------------------------------------------------------------------------------------------ data
def load_split(data_path: Path, seed: int = SEED):
    if not data_path.exists():
        data_gen.generate(seed=seed, out_dir=data_path.parent)
    df = pd.read_csv(data_path)
    X, y = df.drop(columns="risk"), df["risk"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# ------------------------------------------------------------------------------------------ feature selection
def select_features(X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                    seed: int = SEED) -> tuple[list[str], pd.DataFrame]:
    """Keep a feature iff its mutual information (train) AND its permutation importance (val, macro-F1)
    both beat every pure-noise column.

    RF impurity importance is reported too but NOT used for the decision: it is biased towards continuous
    features (the noise columns get ~0.05, above rare binary symptoms such as `consciousness`).
    """
    def enc(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["consciousness"] = (d["consciousness"] == "confused").astype(int)
        return d
    Xn, Xv = enc(X), enc(X_val)
    discrete = [c in SYMPTOM_FEATURES + CATEGORICAL_FEATURES for c in Xn.columns]
    mi = mutual_info_classif(Xn, y, discrete_features=discrete, random_state=seed)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(Xn, y)
    perm = permutation_importance(rf, Xv, y_val, scoring="f1_macro", n_repeats=10, random_state=seed, n_jobs=-1)
    y_ord = y.map({c: i for i, c in enumerate(CLASSES)})
    corr = Xn.apply(lambda col: col.corr(y_ord, method="spearman"))
    table = pd.DataFrame({"mutual_info": mi, "perm_importance": perm.importances_mean,
                          "perm_std": perm.importances_std, "rf_impurity": rf.feature_importances_,
                          "spearman_vs_risk": corr.to_numpy()}, index=Xn.columns)
    noise = table.loc[NOISE_FEATURES]
    keep = table[(table.mutual_info > noise.mutual_info.max()) &
                 (table.perm_importance > noise.perm_importance.max())].index.tolist()
    table["kept"] = table.index.isin(keep)
    return keep, table.sort_values("perm_importance", ascending=False)


def make_preprocessor(features: list[str], missing_indicator: bool = False) -> ColumnTransformer:
    """``missing_indicator=True`` adds a 0/1 "was missing" column per numeric feature with NaNs in training
    (used on the real KTAS data, where SpO2 is missing for 55% of visits and missingness is informative)."""
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=missing_indicator)),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def candidate_models(seed: int = SEED) -> dict:
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                               random_state=seed),
    }


# ------------------------------------------------------------------------------------------ metrics
def reliability_bins(proba: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> dict:
    """Top-label reliability diagram data: per confidence bin, mean confidence, accuracy and count."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = {"conf": [], "acc": [], "count": []}
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out["conf"].append(float(conf[m].mean()))
            out["acc"].append(float((pred[m] == y_idx[m]).mean()))
            out["count"].append(int(m.sum()))
    return out


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(proba: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y_idx]
    return float(((proba - onehot) ** 2).sum(1).mean())


def proba_in_class_order(model, X) -> np.ndarray:
    """predict_proba with columns reordered to CLASSES (missing classes get probability 0)."""
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(c)] = raw[:, j]
    return out


def evaluate(model, X, y) -> dict:
    return metrics_from_proba(proba_in_class_order(model, X), y)


def metrics_from_proba(proba: np.ndarray, y: pd.Series, pred: np.ndarray | None = None) -> dict:
    """proba: [n, 4] in CLASSES order. ``pred`` overrides argmax (e.g. hard labels from a human rater)."""
    if pred is None:
        pred = np.asarray(CLASSES)[proba.argmax(1)]
    y_idx = pd.Series(y).map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "recall": dict(zip(CLASSES, map(float, r))),
        "f1": dict(zip(CLASSES, map(float, f))),
        "ece": expected_calibration_error(proba, y_idx),
        "brier": multiclass_brier(proba, y_idx),
        "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "reliability": reliability_bins(proba, y_idx),
    }


# ------------------------------------------------------------------------------------------ main
def run(data_path: Path, model_path: Path = MODEL_PATH, exp_dir: Path = EXP_DIR, seed: int = SEED,
        quick: bool = False) -> dict:
    t0 = time.time()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_split(data_path, seed)
    features, fs_table = select_features(X_tr, y_tr, X_va, y_va, seed)
    print(f"selected {len(features)} features: {features}")

    # 1) model x imbalance-handling comparison (fit on train, compare on val)
    comparison = {}
    for name, est in candidate_models(seed).items():
        for cw in (None, "balanced"):
            est_cw = clone(est).set_params(class_weight=cw)
            pipe = Pipeline([("pre", make_preprocessor(features)), ("clf", est_cw)]).fit(X_tr[features], y_tr)
            key = f"{name}|{'balanced' if cw else 'none'}"
            comparison[key] = {"train": evaluate(pipe, X_tr[features], y_tr),
                               "val": evaluate(pipe, X_va[features], y_va)}
            v = comparison[key]["val"]
            print(f"  {key:32s} val macroF1={v['macro_f1']:.3f} recall[Crit]={v['recall']['Critical']:.3f}"
                  f" ECE={v['ece']:.3f}")

    # 2) tune the main model (RF) with 5-fold CV on train, macro-F1
    grid = {"clf__n_estimators": [300], "clf__max_depth": [None, 12], "clf__min_samples_leaf": [3, 8],
            "clf__class_weight": [None, "balanced"]} if quick else {
            "clf__n_estimators": [300], "clf__max_depth": [None, 8, 14],
            "clf__min_samples_leaf": [1, 3, 8], "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced", "balanced_subsample"]}
    rf_pipe = Pipeline([("pre", make_preprocessor(features)),
                        ("clf", RandomForestClassifier(random_state=seed, n_jobs=1))])
    gs = GridSearchCV(rf_pipe, grid, scoring="f1_macro", n_jobs=-1, refit=True,
                      cv=StratifiedKFold(5, shuffle=True, random_state=seed)).fit(X_tr[features], y_tr)
    best = gs.best_estimator_
    print(f"RF grid best {gs.best_params_}  cv macroF1={gs.best_score_:.3f}")

    # 3) calibration: KB thresholds use p >= 0.7, so the probability must mean something
    calibrated = {}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(clone(best), method=method,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        calibrated[method] = cal.fit(X_tr[features], y_tr)
    cal_report = {"uncalibrated": evaluate(best, X_va[features], y_va)}
    cal_report.update({m: evaluate(c, X_va[features], y_va) for m, c in calibrated.items()})
    for k, v in cal_report.items():
        print(f"  calib {k:12s} val ECE={v['ece']:.3f} brier={v['brier']:.3f} macroF1={v['macro_f1']:.3f}")
    # pick lowest val ECE as long as macro-F1 does not drop by more than 1 point
    base_f1 = cal_report["uncalibrated"]["macro_f1"]
    eligible = [k for k, v in cal_report.items() if v["macro_f1"] >= base_f1 - 0.01]
    chosen = min(eligible, key=lambda k: cal_report[k]["ece"])
    deployed = best if chosen == "uncalibrated" else calibrated[chosen]

    # 4) final, single look at the test set
    test = evaluate(deployed, X_te[features], y_te)
    train = evaluate(deployed, X_tr[features], y_tr)
    print(f"deployed RF ({chosen}) test acc={test['accuracy']:.3f} macroF1={test['macro_f1']:.3f} "
          f"recall[Crit]={test['recall']['Critical']:.3f} ECE={test['ece']:.3f}  "
          f"(train macroF1={train['macro_f1']:.3f})")

    medians = X_tr[features].median(numeric_only=True).to_dict()
    bundle = {
        "pipeline": deployed, "features": features, "classes": CLASSES, "train_medians": medians,
        "meta": {"model": "RandomForest", "calibration": chosen, "best_params": gs.best_params_, "seed": seed,
                 "data": str(data_path.name), "test": test},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    results = {
        "n": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
        "class_distribution_train": y_tr.value_counts(normalize=True).reindex(CLASSES).round(4).to_dict(),
        "feature_selection": fs_table.round(4).reset_index(names="feature").to_dict(orient="records"),
        "selected_features": features,
        "comparison": comparison,
        "grid_search": {"best_params": gs.best_params_, "cv_macro_f1": gs.best_score_,
                        "n_candidates": len(gs.cv_results_["params"])},
        "calibration_val": cal_report, "calibration_chosen": chosen,
        "deployed": {"train": train, "test": test},
        "elapsed_s": round(time.time() - t0, 1),
    }
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (exp_dir / "results.md").write_text(to_markdown(results))
    print(f"saved {model_path} and {exp_dir}/results.{{json,md}} in {results['elapsed_s']}s")
    return results


def to_markdown(r: dict) -> str:
    out = ["# ML risk model — results (auto-generated by `python -m er_agent.learning.ml.train`)", ""]
    out += ["## Feature selection (train split)", "",
            "| feature | MI (train) | permutation imp. (val) | RF impurity | Spearman vs risk | kept |",
            "|---|---|---|---|---|---|"]
    for row in r["feature_selection"]:
        out.append(f"| {row['feature']} | {row['mutual_info']:.3f} | {row['perm_importance']:.3f} "
                   f"± {row['perm_std']:.3f} | {row['rf_impurity']:.3f} | {row['spearman_vs_risk']:.2f} | "
                   f"{'yes' if row['kept'] else 'no'} |")
    out += ["", "## Model comparison (validation split)", "",
            "| model | class_weight | acc | macro-P | macro-R | macro-F1 | recall Critical | ECE | train F1 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for key, d in r["comparison"].items():
        name, cw = key.split("|")
        v = d["val"]
        out.append(f"| {name} | {cw} | {v['accuracy']:.3f} | {v['macro_precision']:.3f} | {v['macro_recall']:.3f}"
                   f" | {v['macro_f1']:.3f} | {v['recall']['Critical']:.3f} | {v['ece']:.3f}"
                   f" | {d['train']['macro_f1']:.3f} |")
    g = r["grid_search"]
    out += ["", (f"## RF grid search: {g['n_candidates']} candidates x 5 folds, best CV macro-F1 = "
            f"{g['cv_macro_f1']:.3f}"), "", f"`{g['best_params']}`", "",
            "## Calibration (validation split)", "", "| variant | ECE | Brier | macro-F1 |", "|---|---|---|---|"]
    for k, v in r["calibration_val"].items():
        out.append(f"| {k} | {v['ece']:.3f} | {v['brier']:.3f} | {v['macro_f1']:.3f} |")
    t, tr = r["deployed"]["test"], r["deployed"]["train"]
    out += ["", f"Chosen: **{r['calibration_chosen']}**", "", "## Deployed model — test split", "",
            (f"accuracy {t['accuracy']:.3f}, macro-F1 {t['macro_f1']:.3f} (train {tr['macro_f1']:.3f}), "
            f"ECE {t['ece']:.3f}"), "", "| class | recall | F1 |", "|---|---|---|"]
    for c in CLASSES:
        out.append(f"| {c} | {t['recall'][c]:.3f} | {t['f1'][c]:.3f} |")
    out += ["", "Confusion matrix (rows = true, cols = predicted, order " + ", ".join(CLASSES) + "):", "", "```"]
    out += ["  ".join(f"{x:5d}" for x in row) for row in t["confusion"]]
    out += ["```", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and deploy the tabular ML risk model.")
    ap.add_argument("--data", type=Path, default=data_gen.DATA_DIR / "tabular.csv")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quick", action="store_true", help="small grid (CI / smoke test)")
    a = ap.parse_args(argv)
    run(a.data, a.out, EXP_DIR, a.seed, a.quick)


if __name__ == "__main__":
    main()
