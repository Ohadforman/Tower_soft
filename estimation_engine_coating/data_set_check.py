import os
import glob
import pandas as pd
import numpy as np

# ============================
# CONFIG
# ============================
DATASET_DIR = "../data_set_csv"
OUT_TABLE = "training_table.csv"

# ✅ Only read dataset CSVs that start with "FAKE"
ONLY_PREFIX = "FAKE"  # set to "" to read all csvs


def _to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return np.nan


def load_param_csv(path: str) -> pd.DataFrame:
    """Reads Tower dataset CSV of format: Parameter Name, Value, Units (Units optional)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # normalize common variants
    if "Parameter Name" not in df.columns:
        for c in df.columns:
            if "parameter" in c.lower() and "name" in c.lower():
                df = df.rename(columns={c: "Parameter Name"})
    if "Value" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "value":
                df = df.rename(columns={c: "Value"})

    if "Parameter Name" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: missing required columns")

    df["Parameter Name"] = df["Parameter Name"].astype(str).str.strip()
    return df[["Parameter Name", "Value"]]


def get_value(df_params: pd.DataFrame, key: str):
    """Exact key match."""
    hit = df_params.loc[df_params["Parameter Name"] == key, "Value"]
    if hit.empty:
        return None
    return hit.iloc[0]


def build_row(file_path: str) -> dict:
    dfp = load_param_csv(file_path)
    fname = os.path.basename(file_path)

    # =====================================================
    # REQUIRED TARGET KEYS (EDIT THESE 4 TO MATCH YOUR CSV)
    # =====================================================
    MAIN_D_TH_KEY = "Main Coating Diameter Theoretical (um)"  # <-- edit if needed
    MAIN_D_REAL_KEY = "Main Coating Diameter Measured (um)"  # <-- edit if needed
    SEC_D_TH_KEY = "Secondary Coating Diameter Theoretical (um)"  # <-- edit if needed
    SEC_D_REAL_KEY = "Secondary Coating Diameter Measured (um)"  # <-- edit if needed

    main_th = _to_float(get_value(dfp, MAIN_D_TH_KEY))
    main_real = _to_float(get_value(dfp, MAIN_D_REAL_KEY))
    sec_th = _to_float(get_value(dfp, SEC_D_TH_KEY))
    sec_real = _to_float(get_value(dfp, SEC_D_REAL_KEY))

    row = {
        "dataset_csv": fname,
        "D_main_th_um": main_th,
        "D_main_real_um": main_real,
        "residual_main_um": (main_real - main_th)
        if (pd.notna(main_real) and pd.notna(main_th))
        else np.nan,
        "D_sec_th_um": sec_th,
        "D_sec_real_um": sec_real,
        "residual_sec_um": (sec_real - sec_th)
        if (pd.notna(sec_real) and pd.notna(sec_th))
        else np.nan,
    }

    # ============================
    # FEATURES (keep 5–10)
    # ============================
    feature_map = {
        "speed_m_min": "Drawing Speed (m/min)",
        "tension_g": "Tension (g)",
        "furnace_temp_c": "Furnace Temperature (°C)",
        "main_coat_temp_c": "Main Coating Temperature (°C)",
        "sec_coat_temp_c": "Secondary Coating Temperature (°C)",
        "bare_fiber_um": "Bare Fiber Diameter (um)",
    }

    for out_col, key in feature_map.items():
        row[out_col] = _to_float(get_value(dfp, key))

    # optional categorical
    cat_map = {
        "main_coating_type": "Main Coating Type",
        "sec_coating_type": "Secondary Coating Type",
        # "main_die_id":     "Main Die ID",
        # "sec_die_id":      "Secondary Die ID",
    }
    for out_col, key in cat_map.items():
        v = get_value(dfp, key)
        row[out_col] = None if v is None else str(v).strip()

    return row


def main():
    pattern = f"{ONLY_PREFIX}*.csv" if ONLY_PREFIX else "*.csv"
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, pattern)))
    if not paths:
        raise SystemExit(f"No '{pattern}' found in {DATASET_DIR}")

    rows = []
    bad = []
    for p in paths:
        try:
            rows.append(build_row(p))
        except Exception as e:
            bad.append((os.path.basename(p), str(e)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)

    # usable counts separately (because sometimes you may have only main or only secondary)
    main_ok = df.dropna(subset=["D_main_th_um", "D_main_real_um", "residual_main_um"])
    sec_ok = df.dropna(subset=["D_sec_th_um", "D_sec_real_um", "residual_sec_um"])

    print("✅ Wrote:", OUT_TABLE)
    print("Files pattern:", pattern)
    print("Total files:", len(df))
    print("Usable MAIN rows:", len(main_ok))
    print("Usable SECONDARY rows:", len(sec_ok))

    if bad:
        print("\n⚠️ Files that failed to parse:")
        for name, err in bad[:20]:
            print("-", name, "->", err)
        if len(bad) > 20:
            print("... plus", len(bad) - 20, "more")

    # Missing-rate views
    if len(main_ok) > 0:
        print("\nMissing per column (MAIN usable rows):")
        miss = main_ok.isna().mean().sort_values(ascending=False)
        print(miss.head(25))

    if len(sec_ok) > 0:
        print("\nMissing per column (SECONDARY usable rows):")
        miss = sec_ok.isna().mean().sort_values(ascending=False)
        print(miss.head(25))


if __name__ == "__main__":
    main()