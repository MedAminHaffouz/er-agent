"""Synthetic dataset generator (owner: Member 4) — Block 3.

Produces two seeded, regenerable datasets:

* ``data/tabular.csv``  — one row per patient snapshot, label ``risk`` in {Low, Medium, High, Critical}
  (used by the ML models in ``er_agent.learning.ml``).
* ``data/temporal.npz`` — vitals sequences ``[N, T, F]`` with a mask (used by the DL models, Amin).

Usage::

    python -m er_agent.learning.data_gen --seed 42

Educational simulation only. No real medical diagnosis or treatment recommendation.

Design notes (summarised in the report, Block 3 "dataset construction"):

Tabular
    A latent severity ``s >= 0`` drives every vital sign (plus independent noise), so features are correlated
    but the class distributions overlap. The label comes from a NEWS2-inspired early-warning score
    (point bands on rr, spo2, sbp, hr, temp, consciousness) + age + symptom bonuses + a small Gaussian
    "clinician judgement" term, cut into 4 classes. Then ~7% label noise is applied (mostly to an adjacent
    class). The cut-points give a realistic imbalance close to 50/30/15/5. Two pure-noise columns are
    added so feature selection has something to reject.

Temporal
    Each sequence follows a severity path ``sev(t)`` of one of five trajectory types (stable, noisy_stable,
    slow_det, fast_det, recovery); vitals = patient baseline + sev(t) * deterioration direction + noise.
    The label depends on the *trajectory*, not only the last step: the 5-vital warning score of the last
    step is projected two steps ahead with the recent slope. A block of explicit *pairs* shares identical
    last-step vitals but has opposite trends (deterioration vs recovery) -> the key ML-vs-DL experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from er_agent.contracts import RISK_ORDER

CLASSES: list[str] = sorted(RISK_ORDER, key=RISK_ORDER.get)      # ["Low", "Medium", "High", "Critical"]
SEQ_FEATURES: list[str] = ["hr", "sbp", "spo2", "temp", "rr"]      # order of the F axis in temporal.npz
TRAJ_TYPES: list[str] = ["stable", "noisy_stable", "slow_det", "fast_det", "recovery"]

NUMERIC_FEATURES = ["age", "hr", "sbp", "dbp", "spo2", "temp", "rr"]
SYMPTOM_FEATURES = ["chest_pain", "sob", "abdominal_pain", "fever_symptom"]
CATEGORICAL_FEATURES = ["consciousness"]                           # "alert" | "confused"
NOISE_FEATURES = ["noise_uniform", "noise_gauss"]
TABULAR_COLUMNS = NUMERIC_FEATURES + SYMPTOM_FEATURES + CATEGORICAL_FEATURES + NOISE_FEATURES + ["risk"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ------------------------------------------------------------------------------------------ scoring
def _bands(x: np.ndarray, edges: list[float], points: list[int]) -> np.ndarray:
    """Piecewise-constant points: ``points[i]`` for ``edges[i-1] < x <= edges[i]`` (len(points) = len(edges)+1)."""
    return np.asarray(points)[np.searchsorted(edges, x, side="left")]


def news_vitals(hr, sbp, spo2, temp, rr) -> np.ndarray:
    """NEWS2-inspired points on the 5 sequence vitals (0..15). Vectorised; works on any broadcastable shape."""
    hr, sbp, spo2, temp, rr = (np.asarray(v, dtype=float) for v in (hr, sbp, spo2, temp, rr))
    p = _bands(rr, [8, 11, 20, 24], [3, 1, 0, 2, 3])
    p = p + _bands(spo2, [91, 93, 95], [3, 2, 1, 0])
    p = p + _bands(sbp, [90, 100, 110, 219], [3, 2, 1, 0, 3])
    p = p + _bands(hr, [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    p = p + _bands(temp, [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    return p.astype(float)


def tabular_score(df: pd.DataFrame) -> np.ndarray:
    """Deterministic part of the tabular label score (before judgement noise)."""
    s = news_vitals(df.hr, df.sbp, df.spo2, df.temp, df.rr)
    s += 3.0 * (df.consciousness == "confused")
    s += np.where(df.age >= 80, 2.0, np.where(df.age >= 65, 1.0, 0.0))
    cp, sob = df.chest_pain.astype(bool), df.sob.astype(bool)
    s += 2.0 * cp + 1.0 * sob + 1.0 * (cp & sob)
    s += 0.5 * df.abdominal_pain.astype(bool) + 1.0 * df.fever_symptom.astype(bool)
    return s.to_numpy(dtype=float)


TAB_CUTS = [3.0, 6.5, 11.0]      # score -> Low | Medium | High | Critical   (tuned for ~50/30/15/5)
SEQ_CUTS = [3.0, 6.5, 10.5]      # projected 5-vital score -> class


def score_to_class(score: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.searchsorted(cuts, score, side="right")


def _label_noise(y: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Flip ``rate`` of labels: 80% to an adjacent class, 20% to any other class. Returns (y_noisy, flipped)."""
    y = y.copy()
    k = len(CLASSES)
    flip = rng.random(len(y)) < rate
    idx = np.flatnonzero(flip)
    adjacent = rng.random(len(idx)) < 0.8
    step = rng.choice([-1, 1], size=len(idx))
    adj = np.clip(y[idx] + step, 0, k - 1)
    adj = np.where(adj == y[idx], y[idx] - step, adj)                  # bounced on an edge -> go the other way
    other = (y[idx] + rng.integers(1, k, size=len(idx))) % k
    y[idx] = np.where(adjacent, adj, other)
    return y, flip


# ------------------------------------------------------------------------------------------ tabular
def make_tabular(n: int = 8000, seed: int = 42, noise_rate: float = 0.07) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # latent severity: mostly mild, long right tail (gamma) -> natural imbalance
    s = rng.gamma(shape=1.3, scale=0.42, size=n)
    age = np.clip(rng.normal(48 + 8 * s, 19), 18, 98).round()

    hr = 78 + 26 * s + rng.normal(0, 13, n)
    brady = rng.random(n) < 0.03 * np.minimum(s, 2)                    # a few bradycardic decompensations
    hr = np.where(brady, rng.normal(45, 6, n), hr)
    sbp = 128 - 20 * s + rng.normal(0, 15, n)
    dbp = 0.6 * sbp + rng.normal(4, 7, n)
    spo2 = np.minimum(100, 97.5 - 3.6 * s + rng.normal(0, 1.8, n))
    feverish = rng.random(n) < np.clip(0.15 + 0.2 * s, 0, 0.8)
    temp = 36.9 + rng.normal(0, 0.35, n) + feverish * rng.gamma(2.0, 0.6, n)
    rr = 15 + 5 * s + rng.normal(0, 2.5, n)

    sig = lambda z: 1 / (1 + np.exp(-z))
    chest_pain = rng.random(n) < sig(-2.2 + 0.9 * s)
    sob = rng.random(n) < sig(-2.4 + 1.1 * s + 0.08 * (rr - 16))
    abdominal_pain = rng.random(n) < 0.18
    fever_symptom = (temp >= 38.0) & (rng.random(n) < 0.85) | (rng.random(n) < 0.04)
    confused = rng.random(n) < sig(-5.0 + 1.4 * s + 0.03 * (age - 50))

    df = pd.DataFrame({
        "age": age.astype(int),
        "hr": np.clip(hr, 30, 200).round().astype(int),
        "sbp": np.clip(sbp, 55, 230).round().astype(int),
        "dbp": np.clip(dbp, 30, 140).round().astype(int),
        "spo2": np.clip(spo2, 70, 100).round().astype(int),
        "temp": np.clip(temp, 34.0, 42.0).round(1),
        "rr": np.clip(rr, 6, 45).round().astype(int),
        "chest_pain": chest_pain.astype(int),
        "sob": sob.astype(int),
        "abdominal_pain": abdominal_pain.astype(int),
        "fever_symptom": fever_symptom.astype(int),
        "consciousness": np.where(confused, "confused", "alert"),
        "noise_uniform": rng.random(n).round(4),
        "noise_gauss": rng.normal(0, 1, n).round(4),
    })
    score = tabular_score(df) + rng.normal(0, 0.9, n)                  # clinician-judgement noise
    y, flipped = _label_noise(score_to_class(score, TAB_CUTS), noise_rate, rng)
    df["risk"] = np.asarray(CLASSES)[y]
    df.attrs["label_noise_mask"] = flipped
    return df


# ------------------------------------------------------------------------------------------ temporal
# per unit of severity: hr +, sbp -, spo2 -, temp +, rr +
_DIRECTION = np.array([45.0, -40.0, -10.0, 1.8, 12.0])
_BASE_MEAN = np.array([76.0, 126.0, 97.5, 36.9, 15.0])
_BASE_SD = np.array([8.0, 10.0, 1.0, 0.25, 1.5])
_NOISE_SD = np.array([3.0, 4.0, 0.6, 0.1, 1.0])
_CLIP_LO = np.array([30, 50, 60, 34.0, 6])
_CLIP_HI = np.array([210, 230, 100, 42.0, 50])


def _severity_path(kind: str, T: int, rng: np.random.Generator) -> np.ndarray:
    u = np.linspace(0, 1, T)
    if kind in ("stable", "noisy_stable"):
        return np.full(T, rng.uniform(0.0, 0.6)) + rng.normal(0, 0.03, T).cumsum() * 0.3
    if kind == "slow_det":
        a = rng.uniform(0.0, 0.45)
        return a + rng.uniform(0.25, 0.55) * u
    if kind == "fast_det":
        a = rng.uniform(0.0, 0.35)
        return a + rng.uniform(0.6, 1.1) * u ** 2                       # accelerating
    if kind == "recovery":
        b = rng.uniform(0.6, 1.1)
        return b - rng.uniform(0.35, 0.8) * np.sqrt(u)                  # fast then plateau
    raise ValueError(kind)


def _vitals_from_severity(sev: np.ndarray, base: np.ndarray, noise_mult: float,
                          rng: np.random.Generator) -> np.ndarray:
    x = base + np.maximum(sev, 0)[:, None] * _DIRECTION + rng.normal(0, 1, (len(sev), 5)) * _NOISE_SD * noise_mult
    x = np.clip(x, _CLIP_LO, _CLIP_HI)
    return np.round(x, 1)


def trajectory_label_score(seq: np.ndarray) -> float:
    """Projected warning score for one (unpadded) sequence [t, 5]: last-step score + 2 x recent slope."""
    scores = news_vitals(*seq.T)
    w = scores[-4:]
    slope = np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) >= 2 else 0.0
    return float(max(scores[-1] + 2.0 * slope, 0.0))


def make_temporal(n: int = 4000, seed: int = 42, t_min: int = 5, t_max: int = 12,
                  n_pairs: int = 400, noise_rate: float = 0.05) -> dict[str, np.ndarray]:
    """Returns arrays ready for ``np.savez``. ``n`` includes the ``2 * n_pairs`` paired sequences."""
    rng = np.random.default_rng(seed + 1)
    F = len(SEQ_FEATURES)
    X = np.zeros((n, t_max, F), dtype=np.float32)
    mask = np.zeros((n, t_max), dtype=bool)
    lengths = np.zeros(n, dtype=np.int64)
    traj = np.zeros(n, dtype=np.int64)
    pair_id = np.full(n, -1, dtype=np.int64)
    scores = np.zeros(n, dtype=np.float32)

    def put(i: int, seq: np.ndarray, kind: str) -> None:
        T = len(seq)
        X[i, :T], mask[i, :T], lengths[i], traj[i] = seq, True, T, TRAJ_TYPES.index(kind)
        scores[i] = trajectory_label_score(seq)

    # 1) explicit pairs: same last-step vitals, opposite trends
    i = 0
    for k in range(n_pairs):
        base = _BASE_MEAN + rng.normal(0, 1, F) * _BASE_SD
        end = rng.uniform(0.35, 0.8)
        T1, T2 = rng.integers(t_min, t_max + 1, size=2)
        up = end - rng.uniform(0.4, 0.7) * (1 - np.linspace(0, 1, T1))       # deteriorating towards `end`
        down = end + rng.uniform(0.4, 0.7) * (1 - np.linspace(0, 1, T2))     # recovering towards `end`
        a = _vitals_from_severity(up, base, 1.0, rng)
        b = _vitals_from_severity(down, base, 1.0, rng)
        b[-1] = a[-1]                                                        # identical last step
        put(i, a, "slow_det" if T1 > 7 else "fast_det"); pair_id[i] = k; i += 1
        put(i, b, "recovery"); pair_id[i] = k; i += 1

    # 2) free trajectories
    probs = np.array([0.35, 0.15, 0.2, 0.12, 0.18])                     # stable-heavy, like a real ER
    for j in range(i, n):
        kind = TRAJ_TYPES[rng.choice(len(TRAJ_TYPES), p=probs)]
        T = int(rng.integers(t_min, t_max + 1))
        base = _BASE_MEAN + rng.normal(0, 1, F) * _BASE_SD
        noise_mult = 2.2 if kind == "noisy_stable" else 1.0
        put(j, _vitals_from_severity(_severity_path(kind, T, rng), base, noise_mult, rng), kind)

    y_clean = score_to_class(scores, SEQ_CUTS)
    y, flipped = _label_noise(y_clean, noise_rate, rng)
    # pairs keep clean labels so the ML-vs-DL experiment measures the trend effect, not the noise
    is_pair = pair_id >= 0
    y = np.where(is_pair, y_clean, y)
    flipped &= ~is_pair

    perm = rng.permutation(n)                                           # shuffle (pairs stay identifiable)
    return {
        "X": X[perm], "mask": mask[perm], "lengths": lengths[perm], "y": y[perm].astype(np.int64),
        "traj_type": traj[perm], "pair_id": pair_id[perm], "score": scores[perm], "label_noise": flipped[perm],
        "classes": np.array(CLASSES), "features": np.array(SEQ_FEATURES), "traj_names": np.array(TRAJ_TYPES),
    }


# ------------------------------------------------------------------------------------------ CLI
def generate(seed: int = 42, out_dir: Path = DATA_DIR, n_tab: int = 8000, n_seq: int = 4000) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tab = make_tabular(n_tab, seed)
    tab.to_csv(out_dir / "tabular.csv", index=False)
    seq = make_temporal(n_seq, seed)
    np.savez_compressed(out_dir / "temporal.npz", **seq)
    return {"tabular": tab, "temporal": seq}


def _summary(tab: pd.DataFrame, seq: dict) -> str:
    lines = [f"tabular.csv  rows={len(tab)}"]
    dist = tab.risk.value_counts(normalize=True).reindex(CLASSES)
    lines.append("  class %:   " + "  ".join(f"{c}={p:.1%}" for c, p in dist.items()))
    lines.append(f"  label noise: {tab.attrs['label_noise_mask'].mean():.1%}")
    y = seq["y"]
    lines.append(f"temporal.npz X={seq['X'].shape} lengths {seq['lengths'].min()}..{seq['lengths'].max()}")
    lines.append("  class %:   " + "  ".join(f"{c}={np.mean(y == k):.1%}" for k, c in enumerate(CLASSES)))
    pid = seq["pair_id"]
    pairs = [np.flatnonzero(pid == k) for k in range(pid.max() + 1)]
    differ = np.mean([y[a] != y[b] for a, b in pairs])
    lines.append(f"  pairs={len(pairs)} (identical last step) with different label: {differ:.1%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--n-tab", type=int, default=8000)
    ap.add_argument("--n-seq", type=int, default=4000)
    a = ap.parse_args(argv)
    res = generate(a.seed, a.out, a.n_tab, a.n_seq)
    print(_summary(res["tabular"], res["temporal"]))
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
