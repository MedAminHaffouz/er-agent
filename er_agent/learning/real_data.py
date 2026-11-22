"""Real-data bonus (owner: Member 4): KTAS emergency-department triage dataset.

Source: Moon S-H, Shim JL, Park K-S, Park C-S (2019) "Triage accuracy and causes of mistriage using the Korean
Triage and Acuity Scale". PLOS ONE 14(9): e0216972. https://doi.org/10.1371/journal.pone.0216972
Supporting information S1 (data, 1267 adult ED visits, two Korean hospitals) — licence CC BY 4.0.
Also mirrored on Kaggle as "Emergency Service - Triage Application" (ilkeryildiz).

    python -m er_agent.learning.real_data       # downloads to data/real/ktas.xlsx and prints a summary

``load_ktas()`` maps the raw file onto our tabular schema (same column names as ``data/tabular.csv``) so the
same pipelines / ``patient_to_row`` conventions apply, plus a few real-only extra columns.

Label mapping KTAS_expert (1 = resuscitation ... 5 = non-urgent) -> our 4 classes:
    1 -> Critical, 2 -> High, 3 -> Medium, 4/5 -> Low

Educational simulation only. No real medical diagnosis or treatment recommendation.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from er_agent.learning.data_gen import DATA_DIR

KTAS_URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
            "&id=10.1371/journal.pone.0216972.s001")
KTAS_PATH = DATA_DIR / "real" / "ktas.xlsx"

KTAS_TO_RISK = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Low"}
EXTRA_FEATURES = ["nrs_pain", "injury", "arrival_ambulance", "mental_level"]   # real-only, not in Contract 6

_SYMPTOM_PATTERNS = {
    "chest_pain": r"chest",
    "sob": r"dyspn|sob\b|breath",
    "abdominal_pain": r"abd|epigastr",
    "fever_symptom": r"fever|febrile",
}


def download_ktas(dest: Path = KTAS_PATH, force: bool = False) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(KTAS_URL, headers={"User-Agent": "er-agent-course-project"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.startswith(b"PK"):                               # xlsx = zip
        raise RuntimeError(f"unexpected download from {KTAS_URL} (not an xlsx)")
    dest.write_bytes(data)
    return dest


def clean_ktas(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw S1 sheet -> our schema. Unmeasurable vitals ('측불' = 'could not measure') become NaN."""
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    cc = raw["Chief_complain"].astype(str).str.lower()
    mental = raw["Mental"].astype(int)                           # 1 alert, 2 verbal, 3 pain, 4 unresponsive
    df = pd.DataFrame({
        "age": num("Age"),
        "hr": num("HR"), "sbp": num("SBP"), "dbp": num("DBP"),
        "spo2": num("Saturation"), "temp": num("BT"), "rr": num("RR"),
        **{col: cc.str.contains(pat, regex=True).astype(int) for col, pat in _SYMPTOM_PATTERNS.items()},
        "consciousness": np.where(mental > 1, "confused", "alert"),
        "nrs_pain": num("NRS_pain").fillna(0),                   # NaN = patient reported no pain
        "injury": (raw["Injury"] == 2).astype(int),
        "arrival_ambulance": raw["Arrival mode"].isin([2, 4]).astype(int),
        "mental_level": mental,
        "ktas_expert": raw["KTAS_expert"].astype(int),
        "ktas_nurse": raw["KTAS_RN"].astype(int),
        "disposition": raw["Disposition"].astype(int),
        "chief_complaint": raw["Chief_complain"].astype(str),
    })
    df["risk"] = df["ktas_expert"].map(KTAS_TO_RISK)
    return df


def load_ktas(path: Path = KTAS_PATH, download: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        download_ktas(path)
    return clean_ktas(pd.read_excel(path))                       # needs openpyxl


def main() -> None:
    df = load_ktas()
    print(f"KTAS real data: {len(df)} rows from {KTAS_PATH}")
    print("risk %:", df.risk.value_counts(normalize=True).round(3).to_dict())
    print("missing %:", df[["hr", "sbp", "dbp", "spo2", "temp", "rr"]].isna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
