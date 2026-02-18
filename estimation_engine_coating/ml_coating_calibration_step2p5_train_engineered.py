import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple

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

# Put models INSIDE estimation_engine_coating/ml_models
THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(THIS_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MAIN_MODEL_PATH = os.path.join(MODEL_DIR, "ridge_residual_main.joblib")
SEC_MODEL_PATH = os.path.join(MODEL_DIR, "ridge_residual_secondary.joblib")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

TARGET_MAIN = "residual_main_um"
TARGET_SEC = "residual_sec_um"

# Choose model type: "ridge" or "huber"
MODEL_TYPE = "ridge"

# Ridge search (small-data friendly)
ALPHAS_TO_TRY = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# Clamp from training residual distribution percentiles
CLAMP_LOW_PCT = 1.0
CLAMP_HIGH_PCT = 99.0

# Nominal setpoints (used for engineered deltas)
NOMINAL = {
    "furnace_temp_c": 1900.0,
    "main_coat_temp_c": 45.0,
    "sec_coat_temp_c": 45.0,
}

# Base columns present in training_table.csv (from Step-1)
BASE_NUM = [
    "speed_m_min",
    "tension_g",
    "furnace_temp_c",
    "main_coat_temp_c",
    "sec_coat_temp_c",
    "bare_fiber_um",
]
BASE_CAT = [
    "main_coating_type",
    "sec_coating_type",
]

# Theory columns from Step-1 output
COL_D_MAIN_TH = "D_main_th_um"
COL_D_SEC_TH = "D_sec_th_um"


def safe_div(a, b, eps=1e-9):
    try:
        return a / (b if abs(b) > eps else eps)
    except Exception:
        return np.nan


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a small set of engineered features to improve tiny-data learning.
    Keep this limited and physically meaningful.
    """
    out = df.copy()

    out["speed_over_tension"] = [
        safe_div(s, t) for s, t in zip(out["speed_m_min"].values, out["tension_g"].values)
    ]

    out["delta_furnace"] = out["furnace_temp_c"] - NOMINAL["furnace_temp_c"]
    out["delta_main_coat_temp"] = out["main_coat_temp_c"] - NOMINAL["main_coat_temp_c"]
    out["delta_sec_coat_temp"] = out["sec_coat_temp_c"] - NOMINAL["sec_coat_temp_c"]

    # Secondary coupling: theoretical main diameter influences secondary behavior
    if COL_D_MAIN_TH in out.columns:
        out["D_main_th_um_as_feat"] = out[COL_D_MAIN_TH].astype(float)
    else:
        out["D_main_th_um_as_feat"] = np.nan

    return out


@dataclass
class FitResult:
    model: object
    alpha: Optional[float]
    loo_mae: float
    baseline_mae: float
    improved_mae: float
    improvement_pct: float


def build_pipeline(
    model_type: str,
    alpha: Optional[float],
    num_features: List[str],
    cat_features: List[str],
) -> Pipeline:
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
        raise ValueError("MODEL_TYPE must be 'ridge' or 'huber'")

    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def evaluate_before_after(
    df: pd.DataFrame,
    th_col: str,
    real_col: str,
    pred_residual: np.ndarray
) -> Tuple[float, float, float]:
    """
    Returns baseline MAE (theory), improved MAE (theory+pred), improvement %.
    """
    y_real = df[real_col].astype(float).values
    y_th = df[th_col].astype(float).values

    baseline_mae = float(mean_absolute_error(y_real, y_th))
    y_corr = y_th + pred_residual
    improved_mae = float(mean_absolute_error(y_real, y_corr))

    if baseline_mae <= 1e-12:
        improvement_pct = 0.0
    else:
        improvement_pct = 100.0 * (baseline_mae - improved_mae) / baseline_mae

    return baseline_mae, improved_mae, improvement_pct


def fit_one_target(
    df: pd.DataFrame,
    target: str,
    num_features: List[str],
    cat_features: List[str],
    th_col_for_eval: str,
    real_col_for_eval: str,
    label: str
) -> FitResult:

    X = df[num_features + cat_features]
    y = df[target].astype(float)

    loo = LeaveOneOut()

    best_alpha = None  # type: Optional[float]
    best_pipe = None
    best_mae = None
    best_pred = None

    if MODEL_TYPE == "ridge":
        for a in ALPHAS_TO_TRY:
            pipe = build_pipeline("ridge", float(a), num_features, cat_features)
            y_pred = cross_val_predict(pipe, X, y, cv=loo)
            mae = float(mean_absolute_error(y, y_pred))
            print(f"[{label}] alpha={a:<6}  LOO residual MAE={mae:.4f} um")
            if (best_mae is None) or (mae < best_mae):
                best_mae = mae
                best_alpha = float(a)
                best_pipe = pipe
                best_pred = y_pred
    else:
        pipe = build_pipeline("huber", None, num_features, cat_features)
        y_pred = cross_val_predict(pipe, X, y, cv=loo)
        mae = float(mean_absolute_error(y, y_pred))
        print(f"[{label}] HUBER  LOO residual MAE={mae:.4f} um")
        best_mae = mae
        best_alpha = None
        best_pipe = pipe
        best_pred = y_pred

    baseline_mae, improved_mae, improvement_pct = evaluate_before_after(
        df=df,
        th_col=th_col_for_eval,
        real_col=real_col_for_eval,
        pred_residual=np.asarray(best_pred, dtype=float)
    )

    # Fit final model on full data
    best_pipe.fit(X, y)

    return FitResult(
        model=best_pipe,
        alpha=best_alpha,
        loo_mae=float(best_mae),
        baseline_mae=baseline_mae,
        improved_mae=improved_mae,
        improvement_pct=improvement_pct,
    )


def main():
    df0 = pd.read_csv(TRAIN_TABLE)

    # Add engineered features
    df = add_engineered_features(df0)

    MAIN_REAL_COL = "D_main_real_um"
    SEC_REAL_COL = "D_sec_real_um"
    MAIN_TH_COL = "D_main_th_um"
    SEC_TH_COL = "D_sec_th_um"

    df_main = df.dropna(subset=BASE_NUM + BASE_CAT + [TARGET_MAIN, MAIN_TH_COL, MAIN_REAL_COL]).copy()
    df_sec = df.dropna(subset=BASE_NUM + BASE_CAT + [TARGET_SEC, SEC_TH_COL, SEC_REAL_COL]).copy()

    if len(df_main) < 10 or len(df_sec) < 10:
        raise SystemExit("Not enough rows to train. Need at least ~10 per target.")

    MAIN_NUM = BASE_NUM + [
        "speed_over_tension",
        "delta_furnace",
        "delta_main_coat_temp",
        "delta_sec_coat_temp",
    ]
    MAIN_CAT = BASE_CAT

    SEC_NUM = MAIN_NUM + ["D_main_th_um_as_feat"]
    SEC_CAT = BASE_CAT

    print("\n============================")
    print("Training MAIN residual model")
    print("============================")
    res_main = fit_one_target(
        df=df_main,
        target=TARGET_MAIN,
        num_features=MAIN_NUM,
        cat_features=MAIN_CAT,
        th_col_for_eval=MAIN_TH_COL,
        real_col_for_eval=MAIN_REAL_COL,
        label="MAIN",
    )

    print("\n===============================")
    print("Training SECONDARY residual model")
    print("===============================")
    res_sec = fit_one_target(
        df=df_sec,
        target=TARGET_SEC,
        num_features=SEC_NUM,
        cat_features=SEC_CAT,
        th_col_for_eval=SEC_TH_COL,
        real_col_for_eval=SEC_REAL_COL,
        label="SECONDARY",
    )

    # Compute clamps from training residuals
    main_resid = df_main[TARGET_MAIN].astype(float).values
    sec_resid = df_sec[TARGET_SEC].astype(float).values

    clamp_main = [
        float(np.percentile(main_resid, CLAMP_LOW_PCT)),
        float(np.percentile(main_resid, CLAMP_HIGH_PCT)),
    ]
    clamp_sec = [
        float(np.percentile(sec_resid, CLAMP_LOW_PCT)),
        float(np.percentile(sec_resid, CLAMP_HIGH_PCT)),
    ]

    # Save models
    joblib.dump(res_main.model, MAIN_MODEL_PATH)
    joblib.dump(res_sec.model, SEC_MODEL_PATH)

    # Save metadata
    meta = {
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": MODEL_TYPE,
        "train_table": TRAIN_TABLE,
        "rows_main": int(len(df_main)),
        "rows_secondary": int(len(df_sec)),
        "targets": {"main": TARGET_MAIN, "secondary": TARGET_SEC},
        "features": {
            "main": {"num": MAIN_NUM, "cat": MAIN_CAT},
            "secondary": {"num": SEC_NUM, "cat": SEC_CAT},
        },
        "ridge_alpha": {
            "main": res_main.alpha,
            "secondary": res_sec.alpha,
        },
        "cv": {
            "method": "LeaveOneOut",
            "residual_mae_um": {
                "main": res_main.loo_mae,
                "secondary": res_sec.loo_mae,
            },
            "diameter_mae_um": {
                "main_baseline_theory": res_main.baseline_mae,
                "main_corrected": res_main.improved_mae,
                "secondary_baseline_theory": res_sec.baseline_mae,
                "secondary_corrected": res_sec.improved_mae,
            },
            "diameter_improvement_pct": {
                "main": res_main.improvement_pct,
                "secondary": res_sec.improvement_pct,
            }
        },
        "clamp_predicted_residual_um": {
            "main": {"low": clamp_main[0], "high": clamp_main[1], "pct": [CLAMP_LOW_PCT, CLAMP_HIGH_PCT]},
            "secondary": {"low": clamp_sec[0], "high": clamp_sec[1], "pct": [CLAMP_LOW_PCT, CLAMP_HIGH_PCT]},
        },
        "nominal_used_for_deltas": NOMINAL,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Saved engineered models + metadata:")
    print(" -", MAIN_MODEL_PATH, f"(alpha={res_main.alpha}, residual LOO MAE={res_main.loo_mae:.4f} um)")
    print(" -", SEC_MODEL_PATH, f"(alpha={res_sec.alpha}, residual LOO MAE={res_sec.loo_mae:.4f} um)")
    print(" -", META_PATH)

    print("\nBefore vs After (diameter MAE):")
    print(f" MAIN:      theory MAE={res_main.baseline_mae:.4f} um  -> corrected MAE={res_main.improved_mae:.4f} um  ({res_main.improvement_pct:+.2f}%)")
    print(f" SECONDARY: theory MAE={res_sec.baseline_mae:.4f} um  -> corrected MAE={res_sec.improved_mae:.4f} um  ({res_sec.improvement_pct:+.2f}%)")


if __name__ == "__main__":
    main()