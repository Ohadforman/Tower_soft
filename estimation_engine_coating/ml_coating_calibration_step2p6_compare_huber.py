import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error

import joblib

# ============================
# CONFIG
# ============================
TRAIN_TABLE = "training_table.csv"

THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(THIS_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# output names (we save both)
RIDGE_MAIN_PATH = os.path.join(MODEL_DIR, "ridge_residual_main.joblib")
RIDGE_SEC_PATH  = os.path.join(MODEL_DIR, "ridge_residual_secondary.joblib")
HUBER_MAIN_PATH = os.path.join(MODEL_DIR, "huber_residual_main.joblib")
HUBER_SEC_PATH  = os.path.join(MODEL_DIR, "huber_residual_secondary.joblib")

META_COMPARE_PATH = os.path.join(MODEL_DIR, "metadata_compare_ridge_huber.json")
BEST_POINTER_PATH = os.path.join(MODEL_DIR, "best_models.json")

TARGET_MAIN = "residual_main_um"
TARGET_SEC  = "residual_sec_um"

ALPHAS_TO_TRY = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

CLAMP_LOW_PCT = 1.0
CLAMP_HIGH_PCT = 99.0

NOMINAL = {"furnace_temp_c": 1900.0, "main_coat_temp_c": 45.0, "sec_coat_temp_c": 45.0}

BASE_NUM = [
    "speed_m_min", "tension_g", "furnace_temp_c",
    "main_coat_temp_c", "sec_coat_temp_c", "bare_fiber_um",
]
BASE_CAT = ["main_coating_type", "sec_coating_type"]

MAIN_REAL_COL = "D_main_real_um"
SEC_REAL_COL  = "D_sec_real_um"
MAIN_TH_COL   = "D_main_th_um"
SEC_TH_COL    = "D_sec_th_um"


def safe_div(a, b, eps=1e-9):
    try:
        return a / (b if abs(b) > eps else eps)
    except Exception:
        return np.nan


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speed_over_tension"] = [safe_div(s, t) for s, t in zip(out["speed_m_min"].values, out["tension_g"].values)]
    out["delta_furnace"] = out["furnace_temp_c"] - NOMINAL["furnace_temp_c"]
    out["delta_main_coat_temp"] = out["main_coat_temp_c"] - NOMINAL["main_coat_temp_c"]
    out["delta_sec_coat_temp"] = out["sec_coat_temp_c"] - NOMINAL["sec_coat_temp_c"]
    out["D_main_th_um_as_feat"] = out[MAIN_TH_COL].astype(float) if MAIN_TH_COL in out.columns else np.nan
    return out


@dataclass
class FitResult:
    model: object
    model_type: str
    alpha: Optional[float]
    loo_residual_mae: float
    baseline_diam_mae: float
    corrected_diam_mae: float
    improvement_pct: float


def build_pipeline(model_type: str, alpha: Optional[float], num_features: List[str], cat_features: List[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
    )

    if model_type == "ridge":
        if alpha is None:
            raise ValueError("alpha required for ridge")
        reg = Ridge(alpha=float(alpha), random_state=0)
    elif model_type == "huber":
        reg = HuberRegressor()
    else:
        raise ValueError("model_type must be 'ridge' or 'huber'")

    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def eval_before_after(df: pd.DataFrame, th_col: str, real_col: str, pred_residual: np.ndarray) -> Tuple[float, float, float]:
    y_real = df[real_col].astype(float).values
    y_th = df[th_col].astype(float).values
    baseline = float(mean_absolute_error(y_real, y_th))
    corrected = float(mean_absolute_error(y_real, y_th + pred_residual))
    imp = 0.0 if baseline <= 1e-12 else 100.0 * (baseline - corrected) / baseline
    return baseline, corrected, imp


def fit_target(df: pd.DataFrame, target: str, th_col: str, real_col: str,
               num_features: List[str], cat_features: List[str],
               model_type: str, label: str) -> FitResult:

    X = df[num_features + cat_features]
    y = df[target].astype(float)
    loo = LeaveOneOut()

    best_pipe = None
    best_alpha = None
    best_mae = None
    best_pred = None

    if model_type == "ridge":
        for a in ALPHAS_TO_TRY:
            pipe = build_pipeline("ridge", float(a), num_features, cat_features)
            pred = cross_val_predict(pipe, X, y, cv=loo)
            mae = float(mean_absolute_error(y, pred))
            print(f"[{label}][RIDGE] alpha={a:<6} LOO residual MAE={mae:.4f} um")
            if (best_mae is None) or (mae < best_mae):
                best_mae, best_alpha, best_pipe, best_pred = mae, float(a), pipe, pred
    else:
        pipe = build_pipeline("huber", None, num_features, cat_features)
        pred = cross_val_predict(pipe, X, y, cv=loo)
        mae = float(mean_absolute_error(y, pred))
        print(f"[{label}][HUBER] LOO residual MAE={mae:.4f} um")
        best_mae, best_alpha, best_pipe, best_pred = mae, None, pipe, pred

    baseline, corrected, imp = eval_before_after(df, th_col, real_col, np.asarray(best_pred, dtype=float))
    best_pipe.fit(X, y)

    return FitResult(
        model=best_pipe,
        model_type=model_type,
        alpha=best_alpha,
        loo_residual_mae=float(best_mae),
        baseline_diam_mae=baseline,
        corrected_diam_mae=corrected,
        improvement_pct=imp,
    )


def clamp_from_training(df: pd.DataFrame, target: str) -> Dict[str, float]:
    vals = df[target].astype(float).values
    return {
        "low": float(np.percentile(vals, CLAMP_LOW_PCT)),
        "high": float(np.percentile(vals, CLAMP_HIGH_PCT)),
        "pct": [CLAMP_LOW_PCT, CLAMP_HIGH_PCT],
    }


def main():
    df0 = pd.read_csv(TRAIN_TABLE)
    df = add_engineered_features(df0)

    df_main = df.dropna(subset=BASE_NUM + BASE_CAT + [TARGET_MAIN, MAIN_TH_COL, MAIN_REAL_COL]).copy()
    df_sec  = df.dropna(subset=BASE_NUM + BASE_CAT + [TARGET_SEC,  SEC_TH_COL,  SEC_REAL_COL]).copy()

    if len(df_main) < 10 or len(df_sec) < 10:
        raise SystemExit("Not enough rows to train (need ~10+).")

    # engineered feature lists (match Step 2.5)
    MAIN_NUM = BASE_NUM + ["speed_over_tension", "delta_furnace", "delta_main_coat_temp", "delta_sec_coat_temp"]
    MAIN_CAT = BASE_CAT
    SEC_NUM  = MAIN_NUM + ["D_main_th_um_as_feat"]
    SEC_CAT  = BASE_CAT

    # clamps based on training residual distribution (same for all models)
    clamps = {
        "main": clamp_from_training(df_main, TARGET_MAIN),
        "secondary": clamp_from_training(df_sec, TARGET_SEC),
    }

    results = {"ridge": {}, "huber": {}}

    print("\n========================")
    print("RIDGE (engineered)")
    print("========================")
    r_main = fit_target(df_main, TARGET_MAIN, MAIN_TH_COL, MAIN_REAL_COL, MAIN_NUM, MAIN_CAT, "ridge", "MAIN")
    r_sec  = fit_target(df_sec,  TARGET_SEC,  SEC_TH_COL,  SEC_REAL_COL,  SEC_NUM,  SEC_CAT,  "ridge", "SECONDARY")

    joblib.dump(r_main.model, RIDGE_MAIN_PATH)
    joblib.dump(r_sec.model,  RIDGE_SEC_PATH)

    results["ridge"] = {
        "main": r_main.__dict__.copy(),
        "secondary": r_sec.__dict__.copy(),
    }
    # remove non-serializable model objects
    results["ridge"]["main"].pop("model", None)
    results["ridge"]["secondary"].pop("model", None)

    print("\n========================")
    print("HUBER (engineered)")
    print("========================")
    h_main = fit_target(df_main, TARGET_MAIN, MAIN_TH_COL, MAIN_REAL_COL, MAIN_NUM, MAIN_CAT, "huber", "MAIN")
    h_sec  = fit_target(df_sec,  TARGET_SEC,  SEC_TH_COL,  SEC_REAL_COL,  SEC_NUM,  SEC_CAT,  "huber", "SECONDARY")

    joblib.dump(h_main.model, HUBER_MAIN_PATH)
    joblib.dump(h_sec.model,  HUBER_SEC_PATH)

    results["huber"] = {
        "main": h_main.__dict__.copy(),
        "secondary": h_sec.__dict__.copy(),
    }
    results["huber"]["main"].pop("model", None)
    results["huber"]["secondary"].pop("model", None)

    # Decide "best" per target by corrected diameter MAE
    best = {}
    for target_name in ["main", "secondary"]:
        ridge_mae = results["ridge"][target_name]["corrected_diam_mae"]
        huber_mae = results["huber"][target_name]["corrected_diam_mae"]
        if huber_mae < ridge_mae:
            best[target_name] = "huber"
        else:
            best[target_name] = "ridge"

    compare_meta = {
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_table": TRAIN_TABLE,
        "rows_main": int(len(df_main)),
        "rows_secondary": int(len(df_sec)),
        "features": {
            "main": {"num": MAIN_NUM, "cat": MAIN_CAT},
            "secondary": {"num": SEC_NUM, "cat": SEC_CAT},
        },
        "clamp_predicted_residual_um": clamps,
        "nominal_used_for_deltas": NOMINAL,
        "results": results,
        "best_by_corrected_diameter_mae": best,
        "model_files": {
            "ridge": {"main": RIDGE_MAIN_PATH, "secondary": RIDGE_SEC_PATH},
            "huber": {"main": HUBER_MAIN_PATH, "secondary": HUBER_SEC_PATH},
        }
    }

    with open(META_COMPARE_PATH, "w", encoding="utf-8") as f:
        json.dump(compare_meta, f, indent=2)

    # Small pointer file for predictor (optional)
    best_pointer = {
        "best": best,
        "use_files": {
            "main": compare_meta["model_files"][best["main"]]["main"],
            "secondary": compare_meta["model_files"][best["secondary"]]["secondary"],
        },
        "metadata_compare": META_COMPARE_PATH,
    }
    with open(BEST_POINTER_PATH, "w", encoding="utf-8") as f:
        json.dump(best_pointer, f, indent=2)

    print("\n✅ Saved comparison metadata:")
    print(" -", META_COMPARE_PATH)
    print(" -", BEST_POINTER_PATH)

    print("\nSummary (Corrected diameter MAE):")
    print(f" MAIN: ridge={results['ridge']['main']['corrected_diam_mae']:.4f} um | huber={results['huber']['main']['corrected_diam_mae']:.4f} um  => best={best['main']}")
    print(f" SEC : ridge={results['ridge']['secondary']['corrected_diam_mae']:.4f} um | huber={results['huber']['secondary']['corrected_diam_mae']:.4f} um  => best={best['secondary']}")


if __name__ == "__main__":
    main()