import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any, Tuple


# ============================
# PATHS
# ============================
THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(THIS_DIR, "ml_models")

# Default (fallback) model paths
DEFAULT_MAIN_MODEL = os.path.join(MODEL_DIR, "ridge_residual_main.joblib")
DEFAULT_SEC_MODEL = os.path.join(MODEL_DIR, "ridge_residual_secondary.joblib")

# Best-model pointer created by step2p6
BEST_POINTER_FILE = os.path.join(MODEL_DIR, "best_models.json")
META_COMPARE_FILE = os.path.join(MODEL_DIR, "metadata_compare_ridge_huber.json")

# Old metadata (from step2p5) fallback
META_FILE = os.path.join(MODEL_DIR, "metadata.json")


# ============================
# DATASET CSV KEYS
# ============================
KEYS = {
    # theory
    "D_MAIN_TH": "Main Coating Diameter Theoretical (um)",
    "D_SEC_TH": "Secondary Coating Diameter Theoretical (um)",

    # measured (needed for improvement metrics)
    "D_MAIN_REAL": "Main Coating Diameter Measured (um)",
    "D_SEC_REAL": "Secondary Coating Diameter Measured (um)",

    # numeric features
    "SPEED": "Drawing Speed (m/min)",
    "TENSION": "Tension (g)",
    "FURNACE": "Furnace Temperature (°C)",
    "MAIN_CT": "Main Coating Temperature (°C)",
    "SEC_CT": "Secondary Coating Temperature (°C)",
    "BARE_D": "Bare Fiber Diameter (um)",

    # categoricals
    "MAIN_TYPE": "Main Coating Type",
    "SEC_TYPE": "Secondary Coating Type",
}

DEFAULT_NOMINAL = {
    "furnace_temp_c": 1900.0,
    "main_coat_temp_c": 45.0,
    "sec_coat_temp_c": 45.0,
}


def _to_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return np.nan


def _safe_div(a, b, eps=1e-9):
    try:
        return a / (b if abs(b) > eps else eps)
    except Exception:
        return np.nan


def _load_param_csv(dataset_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Parameter Name" not in df.columns:
        for c in df.columns:
            if "parameter" in c.lower() and "name" in c.lower():
                df = df.rename(columns={c: "Parameter Name"})
    if "Value" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "value":
                df = df.rename(columns={c: "Value"})

    if "Parameter Name" not in df.columns or "Value" not in df.columns:
        raise ValueError("Dataset CSV missing required columns: 'Parameter Name' and 'Value'")

    df["Parameter Name"] = df["Parameter Name"].astype(str).str.strip()
    return df[["Parameter Name", "Value"]]


def _get_value(dfp: pd.DataFrame, key: str):
    hit = dfp.loc[dfp["Parameter Name"] == key, "Value"]
    if hit.empty:
        return None
    return hit.iloc[0]


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_effective_metadata() -> Dict[str, Any]:
    """
    Prefer metadata_compare_ridge_huber.json (step2p6),
    otherwise use metadata.json (step2p5).
    """
    meta_compare = _read_json_if_exists(META_COMPARE_FILE)
    if meta_compare:
        return meta_compare
    return _read_json_if_exists(META_FILE)


def resolve_best_model_paths() -> Tuple[str, str]:
    """
    If best_models.json exists, use it. Otherwise use ridge defaults.
    """
    if os.path.exists(BEST_POINTER_FILE):
        with open(BEST_POINTER_FILE, "r", encoding="utf-8") as f:
            ptr = json.load(f)
        use_files = ptr.get("use_files", {})
        main_path = use_files.get("main", DEFAULT_MAIN_MODEL)
        sec_path = use_files.get("secondary", DEFAULT_SEC_MODEL)
        return main_path, sec_path

    return DEFAULT_MAIN_MODEL, DEFAULT_SEC_MODEL


def load_models():
    main_path, sec_path = resolve_best_model_paths()

    if not os.path.exists(main_path):
        raise FileNotFoundError("Main model not found: {}".format(main_path))
    if not os.path.exists(sec_path):
        raise FileNotFoundError("Secondary model not found: {}".format(sec_path))

    main_model = joblib.load(main_path)
    sec_model = joblib.load(sec_path)
    return main_model, sec_model, main_path, sec_path


def extract_engineered_features(dataset_csv_path: str) -> Dict[str, Any]:
    """
    Extracts engineered features exactly like training (Step 2.5/2.6).
    Also extracts measured diameters (for improvement metrics).
    """
    meta = load_effective_metadata()
    nominal = meta.get("nominal_used_for_deltas", DEFAULT_NOMINAL)

    dfp = _load_param_csv(dataset_csv_path)

    # theory + measured
    d_main_th = _to_float(_get_value(dfp, KEYS["D_MAIN_TH"]))
    d_sec_th = _to_float(_get_value(dfp, KEYS["D_SEC_TH"]))
    d_main_real = _to_float(_get_value(dfp, KEYS["D_MAIN_REAL"]))
    d_sec_real = _to_float(_get_value(dfp, KEYS["D_SEC_REAL"]))

    # features
    speed = _to_float(_get_value(dfp, KEYS["SPEED"]))
    tension = _to_float(_get_value(dfp, KEYS["TENSION"]))
    furnace = _to_float(_get_value(dfp, KEYS["FURNACE"]))
    main_ct = _to_float(_get_value(dfp, KEYS["MAIN_CT"]))
    sec_ct = _to_float(_get_value(dfp, KEYS["SEC_CT"]))
    bare_d = _to_float(_get_value(dfp, KEYS["BARE_D"]))

    main_type = _get_value(dfp, KEYS["MAIN_TYPE"])
    sec_type = _get_value(dfp, KEYS["SEC_TYPE"])
    main_type = None if main_type is None else str(main_type).strip()
    sec_type = None if sec_type is None else str(sec_type).strip()

    feats = {
        # base numeric
        "speed_m_min": speed,
        "tension_g": tension,
        "furnace_temp_c": furnace,
        "main_coat_temp_c": main_ct,
        "sec_coat_temp_c": sec_ct,
        "bare_fiber_um": bare_d,

        # engineered numeric
        "speed_over_tension": _safe_div(speed, tension) if (np.isfinite(speed) and np.isfinite(tension)) else np.nan,
        "delta_furnace": furnace - float(nominal["furnace_temp_c"]) if np.isfinite(furnace) else np.nan,
        "delta_main_coat_temp": main_ct - float(nominal["main_coat_temp_c"]) if np.isfinite(main_ct) else np.nan,
        "delta_sec_coat_temp": sec_ct - float(nominal["sec_coat_temp_c"]) if np.isfinite(sec_ct) else np.nan,

        # coupling for secondary
        "D_main_th_um_as_feat": d_main_th,

        # categoricals
        "main_coating_type": main_type,
        "sec_coating_type": sec_type,
    }

    return {
        "D_main_th_um": d_main_th,
        "D_sec_th_um": d_sec_th,
        "D_main_real_um": d_main_real,
        "D_sec_real_um": d_sec_real,
        "features": feats,
        "metadata": meta,
    }


def _apply_clamp(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None and value < low:
        return float(low)
    if high is not None and value > high:
        return float(high)
    return float(value)


def predict_corrected_from_dataset_csv(dataset_csv_path: str) -> Dict[str, Any]:
    """
    D_corrected = D_theory + clamp(predicted_residual)
    Uses metadata to match feature lists; uses best_models.json if present.
    """
    pack = extract_engineered_features(dataset_csv_path)

    d_main_th = pack["D_main_th_um"]
    d_sec_th = pack["D_sec_th_um"]
    d_main_real = pack["D_main_real_um"]
    d_sec_real = pack["D_sec_real_um"]

    feats = pack["features"]
    meta = pack["metadata"] or {}

    if np.isnan(d_main_th) or np.isnan(d_sec_th):
        raise ValueError("Missing theoretical diameter(s) in dataset CSV (main/secondary).")

    main_model, sec_model, main_path, sec_path = load_models()

    # Feature lists from metadata
    features_block = meta.get("features", {})
    if features_block.get("main") and features_block.get("secondary"):
        main_num = features_block["main"]["num"]
        main_cat = features_block["main"]["cat"]
        sec_num = features_block["secondary"]["num"]
        sec_cat = features_block["secondary"]["cat"]
    else:
        # fallback
        main_num = [
            "speed_m_min", "tension_g", "furnace_temp_c",
            "main_coat_temp_c", "sec_coat_temp_c", "bare_fiber_um",
            "speed_over_tension", "delta_furnace", "delta_main_coat_temp", "delta_sec_coat_temp"
        ]
        main_cat = ["main_coating_type", "sec_coating_type"]
        sec_num = main_num + ["D_main_th_um_as_feat"]
        sec_cat = main_cat

    X_main = pd.DataFrame([{**feats}], columns=main_num + main_cat)
    X_sec = pd.DataFrame([{**feats}], columns=sec_num + sec_cat)

    r_main = float(main_model.predict(X_main)[0])
    r_sec = float(sec_model.predict(X_sec)[0])

    # Clamp from metadata
    clamp = meta.get("clamp_predicted_residual_um", {})
    main_low = clamp.get("main", {}).get("low", None)
    main_high = clamp.get("main", {}).get("high", None)
    sec_low = clamp.get("secondary", {}).get("low", None)
    sec_high = clamp.get("secondary", {}).get("high", None)

    main_low = float(main_low) if main_low is not None else None
    main_high = float(main_high) if main_high is not None else None
    sec_low = float(sec_low) if sec_low is not None else None
    sec_high = float(sec_high) if sec_high is not None else None

    r_main_clamped = _apply_clamp(r_main, main_low, main_high)
    r_sec_clamped = _apply_clamp(r_sec, sec_low, sec_high)

    out = {
        "model_files_used": {"main": main_path, "secondary": sec_path},

        "D_main_th_um": float(d_main_th),
        "D_sec_th_um": float(d_sec_th),

        "D_main_real_um": None if np.isnan(d_main_real) else float(d_main_real),
        "D_sec_real_um": None if np.isnan(d_sec_real) else float(d_sec_real),

        "residual_main_pred_um": r_main,
        "residual_sec_pred_um": r_sec,

        "residual_main_clamped_um": r_main_clamped,
        "residual_sec_clamped_um": r_sec_clamped,

        "D_main_corrected_um": float(d_main_th + r_main_clamped),
        "D_sec_corrected_um": float(d_sec_th + r_sec_clamped),
    }
    return out


if __name__ == "__main__":
    test_csv = os.path.join("..", "data_set_csv", "FAKE_DRAW_0001.csv")
    out = predict_corrected_from_dataset_csv(test_csv)
    print("Test CSV:", test_csv)
    print("Model files used:", out["model_files_used"])
    for k, v in out.items():
        if k == "model_files_used":
            continue
        print(f"{k}: {v}")