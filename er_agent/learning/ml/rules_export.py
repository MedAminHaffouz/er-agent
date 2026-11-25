"""Bonus: Decision Tree -> symbolic Horn rules for the logic engine (owner: Member 4).

    python -m er_agent.learning.ml.rules_export
        -> er_agent/learning/ml/learned_rules.json   (shipped in the repo, loaded by the logic engine)
        -> experiments/ml/rules_results.{json,md}

Bridge between the learned and the symbolic parts of the agent:

1. Patients are discretized into the SAME facts the World emits (thresholds from ``world/thresholds.py``):
   LowOxygen, Tachycardia, LowBloodPressure, Fever, Elderly, Tachypnea, HasSymptom(x, ...).
2. One shallow decision tree per target is trained on those binary facts:
   HighRisk(x)     <=> label in {High, Critical}
   CriticalRisk(x) <=> label == Critical
3. Every leaf that predicts the target becomes a candidate rule: the path conditions are its premises.
   Horn clauses cannot contain negation, so "feature = 0" conditions are DROPPED (the rule is generalised),
   then the generalised rule is re-validated on held-out data.
4. A rule is kept only if its validation precision >= 0.7 (the same confidence threshold the KB applies to
   ML_Risk / DL_Risk facts), the 95% Wilson lower bound of that precision is >= 0.6, and support >= 20.
   The lower bound matters: choosing among many candidates on small support picks lucky rules (without it,
   two CriticalRisk rules passed at 0.72-0.73 on validation and fell to 0.47-0.63 on test).
   Rules subsumed by a more general kept rule are removed.

The result is a list of ``Learned_R*`` Horn rules with a single variable ``?x``, e.g.
    Learned_R2: Tachypnea(?x) -> HighRisk(?x)
which the logic engine loads with ``load_learned_rules()`` and wraps in its own ``Rule`` class.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from er_agent.learning import data_gen
from er_agent.learning.ml.train import EXP_DIR, SEED
from er_agent.world import thresholds as th

RULES_PATH = Path(__file__).with_name("learned_rules.json")
MIN_PRECISION = 0.7          # same threshold the KB uses for ML_Risk / DL_Risk
MIN_SUPPORT = 20
MIN_LOWER_BOUND = 0.6     # 95% Wilson lower bound on validation precision
MAX_DEPTH = 5
X = "?x"

# discretized fact -> (Contract-1 premise, function of the tabular row columns)
FACTS: dict[str, tuple[tuple, callable]] = {
    "LowOxygen":        (("LowOxygen", X),        lambda d: d.spo2 < th.SPO2_LOW),
    "Tachycardia":      (("Tachycardia", X),      lambda d: d.hr > th.HR_TACHY),
    "LowBloodPressure": (("LowBloodPressure", X), lambda d: d.sbp < th.SBP_LOW),
    "Fever":            (("Fever", X),            lambda d: d.temp >= th.TEMP_FEVER),
    "Elderly":          (("Elderly", X),          lambda d: d.age >= th.AGE_ELDERLY),
    "Tachypnea":        (("Tachypnea", X),        lambda d: d.rr >= th.RR_HIGH),
    "ChestPain":        (("HasSymptom", X, "ChestPain"),         lambda d: d.chest_pain == 1),
    "ShortnessOfBreath": (("HasSymptom", X, "ShortnessOfBreath"), lambda d: d.sob == 1),
    "AbdominalPain":    (("HasSymptom", X, "AbdominalPain"),     lambda d: d.abdominal_pain == 1),
    "Confusion":        (("HasSymptom", X, "Confusion"),         lambda d: d.consciousness == "confused"),
}
TARGETS = {"CriticalRisk": ("Critical",), "HighRisk": ("High", "Critical")}

# Member 3's starter rules (docs/03), restricted to the ones that use only these facts — for comparison
HAND_RULES = [
    ("R1", ["ChestPain", "LowBloodPressure"], "HighRisk"),
    ("R2", ["LowOxygen", "Tachycardia"], "HighRisk"),
    ("R6", ["HighRisk", "Elderly", "LowOxygen"], "CriticalRisk"),
]


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular rows -> boolean fact matrix (NaN vitals -> fact absent, like the World would do)."""
    return pd.DataFrame({name: fn(df).fillna(False).astype(bool).to_numpy() for name, (_, fn) in FACTS.items()},
                        index=df.index)


def targets(y: pd.Series) -> dict[str, np.ndarray]:
    return {t: y.isin(labels).to_numpy() for t, labels in TARGETS.items()}


# ------------------------------------------------------------------------------------------ extraction
def _positive_paths(tree: DecisionTreeClassifier, names: list[str]) -> list[frozenset[str]]:
    """Premise sets (positive conditions only) of every leaf whose majority (weighted) class is 1."""
    t = tree.tree_
    out: list[frozenset[str]] = []

    def walk(node: int, pos: frozenset[str]) -> None:
        if t.children_left[node] == -1:                           # leaf
            if t.value[node][0].argmax() == 1 and pos:
                out.append(pos)
            return
        f = names[t.feature[node]]
        walk(t.children_left[node], pos)                          # f = 0  -> negated literal, dropped
        walk(t.children_right[node], pos | {f})                   # f = 1  -> positive premise
    walk(0, frozenset())
    return out


def fire(premises: frozenset[str] | list[str], facts: pd.DataFrame) -> np.ndarray:
    m = np.ones(len(facts), dtype=bool)
    for p in premises:
        m &= facts[p].to_numpy()
    return m


def rule_stats(premises, facts: pd.DataFrame, target: np.ndarray) -> dict:
    m = fire(premises, facts)
    n = int(m.sum())
    return {"support": n, "precision": float(target[m].mean()) if n else 0.0,
            "recall": float(m[target].mean()) if target.any() else 0.0}


def wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    """Lower end of the 95% Wilson score interval of a proportion p observed on n cases."""
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - spread) / denom)


def extract_rules(facts_tr, y_tr, facts_va, y_va, seed: int = SEED, max_depth: int = MAX_DEPTH) -> list[dict]:
    rules: list[dict] = []
    t_tr, t_va = targets(y_tr), targets(y_va)
    for target in TARGETS:
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=30, class_weight="balanced",
                                      random_state=seed).fit(facts_tr, t_tr[target])
        cands = sorted(set(_positive_paths(tree, list(facts_tr.columns))), key=lambda s: (len(s), sorted(s)))
        kept: list[frozenset[str]] = []
        for prem in cands:
            if any(k <= prem for k in kept):                      # subsumed by a more general kept rule
                continue
            st = rule_stats(prem, facts_va, t_va[target])
            st["precision_lower95"] = wilson_lower(st["precision"], st["support"])
            if (st["precision"] >= MIN_PRECISION and st["precision_lower95"] >= MIN_LOWER_BOUND
                    and st["support"] >= MIN_SUPPORT):
                kept.append(prem)
                rules.append({"premises": sorted(prem), "conclusion": target, "val": st})
    rules.sort(key=lambda r: (r["conclusion"] != "CriticalRisk", -r["val"]["precision"]))
    for i, r in enumerate(rules, 1):
        r["name"] = f"Learned_R{i}"
    return rules


# ------------------------------------------------------------------------------------------ evaluation
def forward_chain(rules: list[tuple[str, list[str], str]], facts: pd.DataFrame) -> pd.DataFrame:
    """Tiny propositional forward chainer (single variable ?x) over the boolean fact matrix — used only to
    score rule sets here; the real engine is Member 3's."""
    f = facts.copy()
    for t in TARGETS:
        f[t] = False
    changed = True
    while changed:
        changed = False
        for _, prem, concl in rules:
            new = fire(prem, f) & ~f[concl].to_numpy()
            if new.any():
                f[concl] = f[concl].to_numpy() | new
                changed = True
    return f


def score_ruleset(rules, facts: pd.DataFrame, y: pd.Series) -> dict:
    derived, tg = forward_chain(rules, facts), targets(y)
    out = {}
    for t in TARGETS:
        pred, true = derived[t].to_numpy(), tg[t]
        tp = int((pred & true).sum())
        out[t] = {"precision": tp / pred.sum() if pred.sum() else 0.0, "recall": tp / true.sum(),
                  "fires": int(pred.sum())}
    return out


def as_triples(rules: list[dict]) -> list[tuple[str, list[str], str]]:
    return [(r["name"], r["premises"], r["conclusion"]) for r in rules]


# ------------------------------------------------------------------------------------------ public API
def to_horn(rule: dict) -> tuple[str, list[tuple], tuple]:
    """JSON rule -> (name, premises as Contract-1 facts with ?x, conclusion fact)."""
    return rule["name"], [FACTS[p][0] for p in rule["premises"]], (rule["conclusion"], X)


def load_learned_rules(path: Path = RULES_PATH) -> list[tuple[str, list[tuple], tuple]]:
    """For the logic engine:  for name, prem, concl in load_learned_rules(): kb.add_rule(Rule(prem, concl, name))"""
    return [(r["name"], [tuple(p) for p in r["premises"]], tuple(r["conclusion"]))
            for r in json.loads(Path(path).read_text())["rules"]]


def pretty(rule: dict) -> str:
    def lit(p):
        return f"{p[0]}({', '.join(map(str, p[1:]))})"
    name, prem, concl = to_horn(rule)
    return f"{name}: {' ∧ '.join(lit(p) for p in prem)} → {lit(concl)}"


# ------------------------------------------------------------------------------------------ main
def run(data_path: Path = data_gen.DATA_DIR / "tabular.csv", rules_path: Path = RULES_PATH,
        exp_dir: Path = EXP_DIR, seed: int = SEED, with_real: bool = True) -> dict:
    if not data_path.exists():
        data_gen.generate(seed=seed, out_dir=data_path.parent)
    df = pd.read_csv(data_path)
    y = df["risk"]
    # same split as train.py (70/15/15, stratified, same seed) -> test rows never seen
    tr, tmp = train_test_split(df.index, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    facts = discretize(df)

    rules = extract_rules(facts.loc[tr], y[tr], facts.loc[va], y[va], seed)
    for r in rules:
        r["test"] = rule_stats(r["premises"], facts.loc[te], targets(y[te])[r["conclusion"]])
    hand = [(n, p, c) for n, p, c in HAND_RULES]
    res = {
        "min_precision": MIN_PRECISION, "min_support": MIN_SUPPORT, "min_lower_bound": MIN_LOWER_BOUND,
        "max_depth": MAX_DEPTH,
        "thresholds": {k: getattr(th, k) for k in dir(th) if k.isupper()},
        "rules": rules,
        "test": {"learned": score_ruleset(as_triples(rules), facts.loc[te], y[te]),
                 "hand_R1_R2_R6": score_ruleset(hand, facts.loc[te], y[te]),
                 "learned_plus_hand": score_ruleset(as_triples(rules) + hand, facts.loc[te], y[te])},
        "hand_rule_stats_test": {},
    }
    tg_te = targets(y[te])
    hr_te = forward_chain(hand, facts.loc[te])                    # R6 needs HighRisk from R1/R2
    for n, p, c in hand:
        m = fire(p, hr_te)
        res["hand_rule_stats_test"][n] = {"support": int(m.sum()),
                                          "precision": float(tg_te[c][m].mean()) if m.any() else 0.0}

    if with_real:
        try:
            from er_agent.learning import real_data
            real = real_data.load_ktas(download=False)
            rf = discretize(real)
            res["real"] = {"learned": score_ruleset(as_triples(rules), rf, real.risk),
                           "hand_R1_R2_R6": score_ruleset(hand, rf, real.risk),
                           "per_rule": {r["name"]: rule_stats(r["premises"], rf, targets(real.risk)[r["conclusion"]])
                                        for r in rules}}
        except FileNotFoundError:
            res["real"] = None

    rules_path.write_text(json.dumps({
        "_doc": "Learned Horn rules (DT on discretized facts). Regenerate: python -m er_agent.learning.ml.rules_export",
        "variable": X,
        "rules": [{"name": r["name"], "premises": [list(FACTS[p][0]) for p in r["premises"]],
                   "conclusion": [r["conclusion"], X], "val_precision": round(r["val"]["precision"], 3),
                   "test_precision": round(r["test"]["precision"], 3), "support_val": r["val"]["support"]}
                  for r in rules]}, indent=2, ensure_ascii=False) + "\n")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "rules_results.json").write_text(json.dumps(res, indent=2, default=float))
    (exp_dir / "rules_results.md").write_text(to_markdown(res))
    for r in rules:
        print(f"{pretty(r):95s} val P={r['val']['precision']:.2f} n={r['val']['support']:4d} | "
              f"test P={r['test']['precision']:.2f}")
    print(f"saved {rules_path} and {exp_dir}/rules_results.{{json,md}}")
    return res


def to_markdown(r: dict) -> str:
    out = ["# Decision tree → Horn rules (auto-generated by `python -m er_agent.learning.ml.rules_export`)", "",
           (f"Kept iff validation precision ≥ {r['min_precision']}, its 95% Wilson lower bound ≥ "
            f"{r['min_lower_bound']} and support ≥ {r['min_support']} (tree depth {r['max_depth']})."), "",
           "| rule | val precision [lower 95%] | val support | val recall | test precision |",
           "|---|---|---|---|---|"]
    for x in r["rules"]:
        out.append(f"| `{pretty(x)}` | {x['val']['precision']:.2f} [{x['val']['precision_lower95']:.2f}] | "
                   f"{x['val']['support']} | {x['val']['recall']:.2f} | {x['test']['precision']:.2f} |")
    if not any(x["conclusion"] == "CriticalRisk" for x in r["rules"]):
        out += ["", ("No CriticalRisk rule passes the filter: on binary facts alone, Critical cannot be isolated "
                "reliably by a short rule. CriticalRisk stays derived from DL_Risk / ML_Risk (R5) and R6.")]
    out += ["", "## Rule sets as classifiers (test split, forward chaining to fixpoint)", "",
            "| rule set | HighRisk P | HighRisk R | CriticalRisk P | CriticalRisk R |", "|---|---|---|---|---|"]
    rows = [("learned", "learned rules"), ("hand_R1_R2_R6", "hand-written R1, R2, R6"),
            ("learned_plus_hand", "learned + hand")]
    for key, lab in rows:
        m = r["test"][key]
        out.append(f"| {lab} | {m['HighRisk']['precision']:.2f} | {m['HighRisk']['recall']:.2f} | "
                   f"{m['CriticalRisk']['precision']:.2f} | {m['CriticalRisk']['recall']:.2f} |")
    out += ["", "Hand-written rules alone (test): " + ", ".join(
        f"{n}: P={s['precision']:.2f} (n={s['support']})" for n, s in r["hand_rule_stats_test"].items())]
    if r.get("real"):
        out += ["", "## Same rules on the real KTAS data", "",
                "| rule set | HighRisk P | HighRisk R | CriticalRisk P | CriticalRisk R |", "|---|---|---|---|---|"]
        for key, lab in rows[:2]:
            m = r["real"][key]
            out.append(f"| {lab} | {m['HighRisk']['precision']:.2f} | {m['HighRisk']['recall']:.2f} | "
                       f"{m['CriticalRisk']['precision']:.2f} | {m['CriticalRisk']['recall']:.2f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    run()
