import os
import glob
import json
import math
import pandas as pd
from typing import Dict, Any, List, Optional


# ============================
# CONFIG
# ============================
DATASET_DIR = "../data_set_csv"
ONLY_PREFIX = "FAKE"       # "" for all later
TOP_N = 15                 # how many worst cases to list per coating
OUT_BASENAME = "worse_cases_report"

THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(THIS_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

OUT_CSV = os.path.join(MODEL_DIR, f"{OUT_BASENAME}.csv")
OUT_JSON = os.path.join(MODEL_DIR, f"{OUT_BASENAME}.json")


# ============================
# PARAMETER KEYS (in dataset CSV)
# ============================
KEYS = {
    # improvement rows (written by coating_writeback.py)
    "MAIN_IMP": "Main Coating Prediction Improvement (um)",
    "MAIN_VER": "Main Coating Prediction Verdict",
    "SEC_IMP": "Secondary Coating Prediction Improvement (um)",
    "SEC_VER": "Secondary Coating Prediction Verdict",

    # for context (from original dataset CSV)
    "SPEED": "Drawing Speed (m/min)",
    "TENSION": "Tension (g)",
    "FURNACE": "Furnace Temperature (°C)",
    "MAIN_CT": "Main Coating Temperature (°C)",
    "SEC_CT": "Secondary Coating Temperature (°C)",
    "BARE": "Bare Fiber Diameter (um)",

    "MAIN_TYPE": "Main Coating Type",
    "SEC_TYPE": "Secondary Coating Type",

    "D_MAIN_TH": "Main Coating Diameter Theoretical (um)",
    "D_MAIN_REAL": "Main Coating Diameter Measured (um)",
    "D_MAIN_CORR": "Main Coating Diameter Corrected Pred (um)",

    "D_SEC_TH": "Secondary Coating Diameter Theoretical (um)",
    "D_SEC_REAL": "Secondary Coating Diameter Measured (um)",
    "D_SEC_CORR": "Secondary Coating Diameter Corrected Pred (um)",
}


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in ("nan", "none"):
            return None
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def list_csvs() -> List[str]:
    pattern = f"{ONLY_PREFIX}*.csv" if ONLY_PREFIX else "*.csv"
    return sorted(glob.glob(os.path.join(DATASET_DIR, pattern)))


def load_params(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Parameter Name" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing Parameter Name/Value")
    df["Parameter Name"] = df["Parameter Name"].astype(str).str.strip()
    return df


def get_param(dfp: pd.DataFrame, key: str):
    hit = dfp.loc[dfp["Parameter Name"] == key, "Value"]
    if hit.empty:
        return None
    return hit.iloc[0]


def build_record(path: str) -> Dict[str, Any]:
    dfp = load_params(path)
    name = os.path.basename(path)

    rec = {"dataset_csv": name}

    # improvements + verdict
    rec["main_improvement_um"] = _to_float(get_param(dfp, KEYS["MAIN_IMP"]))
    rec["main_verdict"] = (str(get_param(dfp, KEYS["MAIN_VER"])).strip()
                           if get_param(dfp, KEYS["MAIN_VER"]) is not None else None)

    rec["sec_improvement_um"] = _to_float(get_param(dfp, KEYS["SEC_IMP"]))
    rec["sec_verdict"] = (str(get_param(dfp, KEYS["SEC_VER"])).strip()
                          if get_param(dfp, KEYS["SEC_VER"]) is not None else None)

    # context features
    rec["speed_m_min"] = _to_float(get_param(dfp, KEYS["SPEED"]))
    rec["tension_g"] = _to_float(get_param(dfp, KEYS["TENSION"]))
    rec["furnace_temp_c"] = _to_float(get_param(dfp, KEYS["FURNACE"]))
    rec["main_coat_temp_c"] = _to_float(get_param(dfp, KEYS["MAIN_CT"]))
    rec["sec_coat_temp_c"] = _to_float(get_param(dfp, KEYS["SEC_CT"]))
    rec["bare_fiber_um"] = _to_float(get_param(dfp, KEYS["BARE"]))

    rec["main_coating_type"] = (str(get_param(dfp, KEYS["MAIN_TYPE"])).strip()
                                if get_param(dfp, KEYS["MAIN_TYPE"]) is not None else None)
    rec["sec_coating_type"] = (str(get_param(dfp, KEYS["SEC_TYPE"])).strip()
                               if get_param(dfp, KEYS["SEC_TYPE"]) is not None else None)

    # diameters
    rec["D_main_th_um"] = _to_float(get_param(dfp, KEYS["D_MAIN_TH"]))
    rec["D_main_real_um"] = _to_float(get_param(dfp, KEYS["D_MAIN_REAL"]))
    rec["D_main_corr_um"] = _to_float(get_param(dfp, KEYS["D_MAIN_CORR"]))

    rec["D_sec_th_um"] = _to_float(get_param(dfp, KEYS["D_SEC_TH"]))
    rec["D_sec_real_um"] = _to_float(get_param(dfp, KEYS["D_SEC_REAL"]))
    rec["D_sec_corr_um"] = _to_float(get_param(dfp, KEYS["D_SEC_CORR"]))

    return rec


def print_top(df: pd.DataFrame, col_imp: str, col_ver: str, label: str):
    dfx = df.dropna(subset=[col_imp]).copy()
    if dfx.empty:
        print(f"\n[{label}] No improvement data found.")
        return

    dfx = dfx.sort_values(col_imp, ascending=True)  # most negative first
    top = dfx.head(TOP_N)

    print(f"\n=== WORST {label} cases (most negative improvement) ===")
    show_cols = [
        "dataset_csv",
        col_imp,
        col_ver,
        "main_coating_type", "sec_coating_type",
        "speed_m_min", "tension_g",
        "main_coat_temp_c", "sec_coat_temp_c",
        "furnace_temp_c",
    ]
    show_cols = [c for c in show_cols if c in top.columns]
    print(top[show_cols].to_string(index=False))


def main():
    paths = list_csvs()
    if not paths:
        raise SystemExit(f"No CSVs found in {DATASET_DIR} with prefix '{ONLY_PREFIX}'")

    rows = []
    bad = 0
    for p in paths:
        try:
            rows.append(build_record(p))
        except Exception:
            bad += 1

    df = pd.DataFrame(rows)

    # Save full report
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_dir": DATASET_DIR,
                "only_prefix": ONLY_PREFIX,
                "total_files": len(paths),
                "parsed_files": len(rows),
                "failed_files": bad,
                "top_n": TOP_N,
                "csv": OUT_CSV,
                "rows": df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("\n✅ Worse-cases report generated")
    print("Total files:", len(paths), "| Parsed:", len(rows), "| Failed:", bad)
    print("CSV:", OUT_CSV)
    print("JSON:", OUT_JSON)

    # Print worst cases for each coating
    print_top(df, "main_improvement_um", "main_verdict", "MAIN")
    print_top(df, "sec_improvement_um", "sec_verdict", "SECONDARY")


if __name__ == "__main__":
    main()