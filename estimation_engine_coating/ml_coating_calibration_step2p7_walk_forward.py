import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# ============================
# CONFIG
# ============================
TRAIN_TABLE = "training_table.csv"

# Use the same feature definitions as Step 2.5 / 2.6
NOMINAL = {"furnace_temp_c": 1900.0, "main_coat_temp_c": 45.0, "sec_coat_temp_c": 45.0}

BASE_NUM = [
    "speed_m_min", "tension_g", "furnace_temp_c",
    "main_coat_temp_c", "sec_coat_temp_c", "bare_fiber_um",
]
BASE_CAT = ["main_coating_type", "sec_coating_type"]

MAIN_TH_COL = "D_main_th_um"
MAIN_REAL_COL = "D_main_real_um"
MAIN_TARGET = "residual_main_um"

SEC_TH_COL = "D_sec_th_um"
SEC_REAL_COL = "D_sec_real_um"
SEC_TARGET = "residual_sec_um"

# Walk-forward settings
START_TRAIN = 20      # first train size
TEST_BLOCK = 5        # test chunk size
STEP = 5              # move forward by this many each iteration

# Keep alphas fixed from your best Ridge results (engineered)
RIDGE_ALPHA_MAIN = 3.0
RIDGE_ALPHA_SEC  = 10.0

OUT_JSON = os.path.join(os.path.dirname(__file__), "ml_models", "walk_forward_report.json")


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


def build_ridge_pipeline(alpha: float, num_features: List[str], cat_features: List[str]) -> Pipeline:
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
    return Pipeline(steps=[("pre", pre), ("ridge", Ridge(alpha=float(alpha), random_state=0))])


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))


def walk_forward_one(
    df: pd.DataFrame,
    th_col: str,
    real_col: str,
    target_resid_col: str,
    num_features: List[str],
    cat_features: List[str],
    alpha: float,
    label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:

    rows = []
    n = len(df)

    for train_end in range(START_TRAIN, n - 1, STEP):
        test_start = train_end
        test_end = min(train_end + TEST_BLOCK, n)
        if test_end <= test_start:
            break

        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()

        pipe = build_ridge_pipeline(alpha, num_features, cat_features)
        X_tr = train[num_features + cat_features]
        y_tr = train[target_resid_col].astype(float)

        X_te = test[num_features + cat_features]
        y_te_real = test[real_col].astype(float).values
        y_te_th = test[th_col].astype(float).values

        pipe.fit(X_tr, y_tr)
        pred_resid = pipe.predict(X_te).astype(float)

        # baseline prediction: theory
        baseline = mae(y_te_real, y_te_th)
        corrected = mae(y_te_real, y_te_th + pred_resid)

        rows.append({
            "label": label,
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "test_range_idx": [int(test_start), int(test_end - 1)],
            "baseline_mae_um": baseline,
            "corrected_mae_um": corrected,
            "improvement_pct": 0.0 if baseline <= 1e-12 else 100.0 * (baseline - corrected) / baseline,
        })

    # overall (avg over blocks)
    if rows:
        overall = {
            "baseline_mae_um_mean": float(np.mean([r["baseline_mae_um"] for r in rows])),
            "corrected_mae_um_mean": float(np.mean([r["corrected_mae_um"] for r in rows])),
            "improvement_pct_mean": float(np.mean([r["improvement_pct"] for r in rows])),
            "blocks": int(len(rows)),
        }
    else:
        overall = {"baseline_mae_um_mean": np.nan, "corrected_mae_um_mean": np.nan, "improvement_pct_mean": np.nan, "blocks": 0}

    return rows, overall


def main():
    df0 = pd.read_csv(TRAIN_TABLE)

    # Sort by dataset name (good enough for FAKE; for real data we can sort by timestamp)
    df0 = df0.sort_values("dataset_csv").reset_index(drop=True)

    df = add_engineered_features(df0)

    # feature sets (match Step 2.5)
    MAIN_NUM = BASE_NUM + ["speed_over_tension", "delta_furnace", "delta_main_coat_temp", "delta_sec_coat_temp"]
    MAIN_CAT = BASE_CAT
    SEC_NUM = MAIN_NUM + ["D_main_th_um_as_feat"]
    SEC_CAT = BASE_CAT

    # drop NA
    df_main = df.dropna(subset=MAIN_NUM + MAIN_CAT + [MAIN_TH_COL, MAIN_REAL_COL, MAIN_TARGET]).copy()
    df_sec = df.dropna(subset=SEC_NUM + SEC_CAT + [SEC_TH_COL, SEC_REAL_COL, SEC_TARGET]).copy()

    print("\n=== Walk-forward validation ===")
    print("Rows MAIN:", len(df_main), "| Rows SEC:", len(df_sec))
    print("START_TRAIN:", START_TRAIN, "TEST_BLOCK:", TEST_BLOCK, "STEP:", STEP)

    main_rows, main_overall = walk_forward_one(
        df_main, MAIN_TH_COL, MAIN_REAL_COL, MAIN_TARGET, MAIN_NUM, MAIN_CAT, RIDGE_ALPHA_MAIN, "MAIN"
    )
    sec_rows, sec_overall = walk_forward_one(
        df_sec, SEC_TH_COL, SEC_REAL_COL, SEC_TARGET, SEC_NUM, SEC_CAT, RIDGE_ALPHA_SEC, "SECONDARY"
    )

    print("\nMAIN overall:")
    print(main_overall)
    print("\nSECONDARY overall:")
    print(sec_overall)

    report = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_table": TRAIN_TABLE,
        "sort": "dataset_csv",
        "settings": {
            "START_TRAIN": START_TRAIN,
            "TEST_BLOCK": TEST_BLOCK,
            "STEP": STEP,
            "alpha_main": RIDGE_ALPHA_MAIN,
            "alpha_secondary": RIDGE_ALPHA_SEC,
        },
        "main": {"blocks": main_rows, "overall": main_overall},
        "secondary": {"blocks": sec_rows, "overall": sec_overall},
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Saved walk-forward report:", OUT_JSON)


if __name__ == "__main__":
    main()