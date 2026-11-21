"""Report figures for Block 3 (ML half) — owner: Member 4.

    python -m er_agent.learning.ml.figures     # reads experiments/ml/*.json -> report/figures/ml_*.png

Static PNGs for the LaTeX report (light theme, print). Colours: categorical slot 1 blue / slot 2 orange,
one-hue blue ramp for magnitudes, gray only for de-emphasised / reference marks; text always in ink colours.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from er_agent.learning import data_gen, real_data
from er_agent.learning.data_gen import CLASSES
from er_agent.learning.ml.train import EXP_DIR, ROOT

FIG_DIR = ROOT / "report" / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"
MIN_BIN = 20
GRAY = "#c3c2b7"                       # de-emphasis / reference
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95",
                                                    "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "legend.frameon": False, "legend.labelcolor": INK2, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _hbar_grid(ax):
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ------------------------------------------------------------------------------------------ synthetic
def fig_confusion(cm: list[list[int]], title: str, out: Path) -> Path:
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised)")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, out)


def fig_reliability(res: dict, out: Path) -> Path:
    cal = res["calibration_val"]
    chosen = res["calibration_chosen"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY)
    ax.text(0.9, 0.84, "perfect", color=MUTED, fontsize=8, ha="right", rotation=45, rotation_mode="anchor")
    series = [("uncalibrated", ORANGE)] + ([(chosen, BLUE)] if chosen != "uncalibrated" else [])
    for name, color in series:
        rb = cal[name]["reliability"]
        keep = [i for i, n in enumerate(rb["count"]) if n >= MIN_BIN]       # sparse bins are pure noise
        lab = f"{name} (ECE {cal[name]['ece']:.3f})"
        ax.plot([rb["conf"][i] for i in keep], [rb["acc"][i] for i in keep], color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=lab)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.text(0.99, 0.31, f"bins with < {MIN_BIN} patients hidden", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xlabel("mean predicted probability (top class)")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration of the Random Forest (validation)")
    ax.legend(loc="upper left")
    return _save(fig, out)


def fig_feature_importance(res: dict, out: Path) -> Path:
    fs = pd.DataFrame(res["feature_selection"]).sort_values("perm_importance")
    colors = [BLUE if k else GRAY for k in fs.kept]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    y = np.arange(len(fs))
    axes[0].barh(y, fs.perm_importance, xerr=fs.perm_std, color=colors, height=0.6,
                 error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    axes[0].set_title("Permutation importance (val, Δ macro-F1)")
    axes[1].barh(y, fs.rf_impurity, color=colors, height=0.6)
    axes[1].set_title("RF impurity importance (biased)")
    axes[0].set_yticks(y, fs.feature)
    for ax in axes:
        _hbar_grid(ax)
    for f in data_gen.NOISE_FEATURES:                                  # point at the noise columns
        i = list(fs.feature).index(f)
        axes[1].annotate("pure noise", (fs.rf_impurity.iloc[i], i), xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=7.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    fig.legend(handles, ["kept", "dropped"], loc="lower right", ncol=2, bbox_to_anchor=(0.98, -0.02))
    return _save(fig, out)


def fig_model_comparison(res: dict, out: Path) -> Path:
    comp = res["comparison"]
    models = list(dict.fromkeys(k.split("|")[0] for k in comp))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    y = np.arange(len(models))
    h = 0.36
    for ax, metric, title in ((axes[0], "macro_f1", "Macro-F1 (validation)"),
                              (axes[1], "crit", "Recall on Critical (validation)")):
        for off, cw, color in ((-h / 2, "none", BLUE), (h / 2, "balanced", ORANGE)):
            vals = [comp[f"{m}|{cw}"]["val"]["recall"]["Critical"] if metric == "crit"
                    else comp[f"{m}|{cw}"]["val"][metric] for m in models]
            ax.barh(y + off, vals, height=h - 0.04, color=color, label=f"class_weight={cw}")
            for yi, v in zip(y + off, vals, strict=True):
                ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        _hbar_grid(ax)
    axes[0].set_yticks(y, models)
    axes[0].invert_yaxis()
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.07))
    return _save(fig, out)


# ------------------------------------------------------------------------------------------ real data
REAL_ROWS = [("H", "Triage nurse (human ref.)"), ("B", "Synthetic-trained model (deployed)"),
             ("A|RandomForest|shared", "RF trained on real, shared feats"),
             ("A|RandomForest|full", "RF trained on real, full feats"),
             ("C", "RF synthetic + real"), ("C_weighted", "RF synthetic + real, re-weighted")]


def fig_real_transfer(r: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    y = np.arange(len(REAL_ROWS))
    for ax, key, ci_key, title in ((axes[0], "macro_f1", "macro_f1_ci", "Macro-F1 vs expert KTAS"),
                                   (axes[1], "auroc", None, "AUROC for real admission/ICU/death")):
        vals, err = [], [[], []]
        for k, _ in REAL_ROWS:
            m = r["rows"][k]
            v = m["admission_auroc"] if key == "auroc" else m[key]
            vals.append(v)
            lo, hi = m[ci_key] if ci_key else (v, v)
            err[0].append(max(v - lo, 0))
            err[1].append(max(hi - v, 0))
        colors = [GRAY if k == "H" else BLUE for k, _ in REAL_ROWS]
        ax.barh(y, vals, color=colors, height=0.6, xerr=err if ci_key else None,
                error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
        for (k, _), yi, v, e in zip(REAL_ROWS, y, vals, err[1], strict=True):
            if ci_key:
                ax.text(v + e + 0.01, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
            else:                                                     # inside the bar: clear of the ref line
                ax.text(v - 0.008, yi, f"{v:.2f}", va="center", ha="right", fontsize=7.5,
                        color=INK if k == "H" else "white")
        ax.set_xlim(0.5 if key == "auroc" else 0, 1)
        ax.set_title(title)
        _hbar_grid(ax)
    axes[0].set_yticks(y, [lab for _, lab in REAL_ROWS])
    axes[0].invert_yaxis()
    axes[1].axvline(r["expert_ktas_admission_auroc"], color=INK2, lw=1, ls=":")
    axes[1].text(r["expert_ktas_admission_auroc"] + 0.005, -0.55, "expert KTAS", fontsize=7.5,
                 color=INK2, va="bottom")
    axes[1].set_ylim(len(REAL_ROWS) - 0.5, -0.9)
    fig.text(0.01, -0.03, "1267 real ED visits (Moon et al. 2019). Bars: 95% bootstrap CI. Gray = human reference.",
             fontsize=7.5, color=MUTED)
    return _save(fig, out)


def fig_real_importance(r: dict, out: Path) -> Path:
    imp = pd.DataFrame(r["real_permutation_importance"]).sort_values("mean")
    shared = set(r["features_shared"])
    colors = [BLUE if f in shared else ORANGE for f in imp.feature]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    y = np.arange(len(imp))
    ax.barh(y, imp["mean"], xerr=imp["std"], color=colors, height=0.6,
            error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    ax.set_yticks(y, imp.feature)
    _hbar_grid(ax)
    ax.set_xlabel("Δ macro-F1 when the feature is shuffled")
    ax.set_title("What real triage depends on (RF, real data)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["in our patient JSON / synthetic data", "real-only feature"], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.07))
    return _save(fig, out)


def fig_generator_score(out: Path, synthetic_path: Path = data_gen.DATA_DIR / "tabular.csv") -> Path:
    syn = pd.read_csv(synthetic_path)
    real = real_data.load_ktas(download=False)
    s_syn = data_gen.tabular_score(syn)
    s_real = data_gen.tabular_score(real.fillna(real.median(numeric_only=True)))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    pos = np.arange(len(CLASSES))
    for off, scores, labels, color, name in ((-0.2, s_syn, syn.risk, BLUE, "synthetic"),
                                             (0.2, s_real, real.risk, ORANGE, "real (KTAS expert)")):
        data = [scores[(labels == c).to_numpy()] for c in CLASSES]
        bp = ax.boxplot(data, positions=pos + off, widths=0.32, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "lw": 1.6}, whiskerprops={"color": color},
                        capprops={"color": color}, boxprops={"facecolor": color, "edgecolor": color})
        bp["boxes"][0].set_label(name)
    ax.set_xticks(pos, CLASSES)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("generator warning score")
    ax.set_xlabel("label")
    ax.set_title("Our label rule vs. real expert triage")
    ax.legend(loc="upper left")
    return _save(fig, out)


# ------------------------------------------------------------------------------------------ all
def make_all(exp_dir: Path = EXP_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    made = []
    res = json.loads((exp_dir / "results.json").read_text())
    made.append(fig_confusion(res["deployed"]["test"]["confusion"],
                              "Deployed RF — synthetic test set", fig_dir / "ml_confusion.png"))
    made.append(fig_reliability(res, fig_dir / "ml_reliability.png"))
    made.append(fig_feature_importance(res, fig_dir / "ml_feature_importance.png"))
    made.append(fig_model_comparison(res, fig_dir / "ml_model_comparison.png"))
    real_json = exp_dir / "real_results.json"
    if real_json.exists():
        r = json.loads(real_json.read_text())
        made.append(fig_real_transfer(r, fig_dir / "ml_real_transfer.png"))
        made.append(fig_real_importance(r, fig_dir / "ml_real_importance.png"))
        made.append(fig_confusion(r["rows"]["B"]["confusion"], "Synthetic-trained RF on real patients",
                                  fig_dir / "ml_real_confusion_synthetic.png"))
        made.append(fig_confusion(r["rows"]["A|RandomForest|full"]["confusion"],
                                  "Real-trained RF (full feats, CV)", fig_dir / "ml_real_confusion_real.png"))
        if real_data.KTAS_PATH.exists():
            made.append(fig_generator_score(fig_dir / "ml_real_generator_score.png"))
    return made


if __name__ == "__main__":
    for p in make_all():
        print(p)
