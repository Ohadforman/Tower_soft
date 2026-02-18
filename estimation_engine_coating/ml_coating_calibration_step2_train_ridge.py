import os
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import joblib


# ============================
# CONFIG
# ============================
TRAIN_TABLE = "training_table.csv"
OUT_DIR = "ml_models"
os.makedirs(OUT_DIR, exist_ok=True)

# targets (residuals)
TARGET_MAIN = "residual_main_um"
TARGET_SEC  = "residual_sec_um"

# features (numeric + categorical)
NUM_FEATURES = [
    "speed_m_min",
    "tension_g",
    "furnace_temp_c",
    "main_coat_temp_c",
    "sec_coat_temp_c",
    "bare_fiber_um",
]
CAT_FEATURES = [
    "main_coating_type",
    "sec_coating_type",
]

# Ridge strength (we'll tune lightly; small data likes more regularization)
ALPHAS_TO_TRY = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]


@dataclass
class ModelResult:
    alpha: float
    mae: float
    model: object


def build_pipeline(alpha: float) -> Pipeline:
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
            ("num", num_pipe, NUM_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=alpha, random_state=0)

    return Pipeline(steps=[
        ("pre", pre),
        ("ridge", model),
    ])


def pick_best_alpha(X: pd.DataFrame, y: pd.Series, label: str) -> ModelResult:
    loo = LeaveOneOut()

    best = None
    for a in ALPHAS_TO_TRY:
        pipe = build_pipeline(alpha=a)
        # LOO predictions
        y_pred = cross_val_predict(pipe, X, y, cv=loo, n_jobs=None)
        mae = float(mean_absolute_error(y, y_pred))

        print(f"[{label}] alpha={a:<6}  LOO MAE={mae:.4f} um")

        if (best is None) or (mae < best.mae):
            best = ModelResult(alpha=a, mae=mae, model=pipe)

    # fit best on full data
    best.model.fit(X, y)
    return best


def main():
    df = pd.read_csv(TRAIN_TABLE)

    # keep only rows with both targets available (you have all 60, but keep robust)
    df_main = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET_MAIN]).copy()
    df_sec  = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET_SEC]).copy()

    if len(df_main) < 10 or len(df_sec) < 10:
        raise SystemExit("Not enough rows to train. Need at least ~10 per target.")

    X_main = df_main[NUM_FEATURES + CAT_FEATURES]
    y_main = df_main[TARGET_MAIN].astype(float)

    X_sec = df_sec[NUM_FEATURES + CAT_FEATURES]
    y_sec = df_sec[TARGET_SEC].astype(float)

    print("\n============================")
    print("Training MAIN residual model")
    print("============================")
    best_main = pick_best_alpha(X_main, y_main, label="MAIN")

    print("\n===============================")
    print("Training SECONDARY residual model")
    print("===============================")
    best_sec = pick_best_alpha(X_sec, y_sec, label="SECONDARY")

    # Save
    main_path = os.path.join(OUT_DIR, "ridge_residual_main.joblib")
    sec_path  = os.path.join(OUT_DIR, "ridge_residual_secondary.joblib")

    joblib.dump(best_main.model, main_path)
    joblib.dump(best_sec.model, sec_path)

    print("\n✅ Saved models:")
    print(" -", main_path, f"(alpha={best_main.alpha}, LOO MAE={best_main.mae:.4f} um)")
    print(" -", sec_path,  f"(alpha={best_sec.alpha},  LOO MAE={best_sec.mae:.4f} um)")

    # Quick sanity: show how to compute corrected predictions
    # D_corrected = D_th + predicted_residual
    df_demo = df.head(5).copy()
    X_demo = df_demo[NUM_FEATURES + CAT_FEATURES]
    pred_main = best_main.model.predict(X_demo)
    pred_sec  = best_sec.model.predict(X_demo)

    df_demo["D_main_pred_corrected_um"] = df_demo["D_main_th_um"] + pred_main
    df_demo["D_sec_pred_corrected_um"]  = df_demo["D_sec_th_um"] + pred_sec

    print("\nExample corrected predictions (first 5 rows):")
    cols = [
        "dataset_csv",
        "D_main_th_um", "D_main_real_um", "D_main_pred_corrected_um",
        "D_sec_th_um", "D_sec_real_um", "D_sec_pred_corrected_um",
    ]
    print(df_demo[cols].to_string(index=False))


if __name__ == "__main__":
    main()