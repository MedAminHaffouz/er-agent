# 04 — Member 4 · Datasets + Machine Learning Risk Prediction (Block 3, ML half)

**Code:** `er_agent/learning/data_gen.py`, `er_agent/learning/ml/`, `notebooks/ml_eval.ipynb`
**Report section:** Block 3 — dataset, preprocessing, models, evaluation, limitations, ML vs DL roles (shared with Amin)
**Course part:** learning

You produce **both datasets** (Amin's DL depends on your temporal one) and the tabular model.

---

## 1. Datasets (top priority — Amin is blocked on it)

Synthetic, seeded, regenerable. Real data option (e.g. public triage datasets on Kaggle/PhysioNet) is fine as a bonus but access/cleaning eats time — synthetic first.

### 1a. Tabular — `data/tabular.csv` (~5k–10k rows)
Features: `age, hr, sbp, dbp, spo2, temp, rr, chest_pain, sob, abdominal_pain, fever_symptom, consciousness (alert/confused)` + 1–2 noise features.
Label `risk ∈ {Low, Medium, High, Critical}`:
- [ ] base score inspired by an early-warning score (NEWS2-like point bands on rr, spo2, sbp, hr, temp, consciousness) + age + symptom bonuses
- [ ] add **label noise (~5–10%)** and overlapping distributions → otherwise accuracy = 100% and nothing to discuss
- [ ] realistic **class imbalance** (e.g. 50/30/15/5) → forces recall/F1 discussion
- [ ] document the generator in the report (it's your "dataset construction" part)

### 1b. Temporal — `data/temporal.npz` (for Amin)
- [ ] sequences `[N, T, F]`, T = 5–12, F = (hr, sbp, spo2, temp, rr), mask for variable length
- [ ] trajectory types: stable, slow deterioration, fast deterioration, recovery, noisy-stable
- [ ] label = risk class derived from trajectory (not only last step)
- [ ] **include pairs with identical last-step vitals but different trends** (key DL vs ML experiment)
- [ ] deliver by **Nov 21**

## 2. Preprocessing
- [ ] train/val/test split **stratified**, fixed seed
- [ ] `sklearn.Pipeline` + `ColumnTransformer`: scaling numeric, one-hot categoricals
- [ ] feature selection: correlation, mutual information or RF importance → keep/justify
- [ ] save pipeline with the model (`joblib`) — the agent must not re-implement preprocessing

## 3. Models
Train ≥ 3, justify each:
- [ ] Decision Tree (interpretable, can be compared with M3's rules — nice link!)
- [ ] Random Forest (robust, nonlinear, feature importance)
- [ ] Logistic Regression (baseline, calibrated probabilities)
- [ ] optional: SVM, k-NN, gradient boosting
- [ ] handle imbalance: `class_weight="balanced"` vs none — compare
- [ ] hyperparameter search (GridSearchCV, 5-fold) on the main model

## 4. Evaluation
- [ ] accuracy, macro precision/recall/F1, **per-class recall (Critical!)**, confusion matrix
- [ ] probability calibration check (the KB uses `p ≥ 0.7` thresholds → probabilities must mean something) → `CalibratedClassifierCV` if needed
- [ ] table comparing all models + 1 paragraph choosing the deployed one
- [ ] limitations: synthetic data, label noise, imbalance, overfitting check (train vs test gap), uncertainty

## 5. Integration API
```python
def predict_ml(patient: dict) -> tuple[str, float]:   # Contract 6 patient → ("High", 0.87)
```
- [ ] loads `models/ml_rf.joblib` once (module-level cache)
- [ ] handles missing fields (impute with training median, log a warning)
- [ ] tested on P1 of the brief → should give High/Critical with reasonable p

## 6. Bonus (if ahead)
- [ ] Export the Decision Tree as rules → feed to M3's KB as `Learned_R*` rules (symbolic ↔ learned bridge — strong report point)
- [ ] SHAP / permutation importance figure

## Done when
- `data_gen.py --seed 42` regenerates both datasets
- `predict_ml(P1)` works from the agent; `tests/test_ml.py` green
- evaluation notebook exports figures to `report/figures/`

## Deadlines
| Date | Deliverable |
|------|-------------|
| **Nov 21** | both datasets generated + pushed (or generator) |
| Nov 22 | models trained, `predict_ml` merged |
| Nov 26 | full evaluation + calibration + tuning |
| Nov 28 | (bonus) DT → rules |
| Nov 29 | report section + joint "ML vs DL roles" paragraph with Amin |
