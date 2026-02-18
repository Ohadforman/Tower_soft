import os
import math
import pandas as pd
from typing import Optional, Dict, Any

from estimation_engine_coating.coating_predictor import predict_corrected_from_dataset_csv


OUT_KEYS = {
    "MAIN_CORR": "Main Coating Diameter Corrected Pred (um)",
    "SEC_CORR":  "Secondary Coating Diameter Corrected Pred (um)",
    "MAIN_RES":  "Main Coating Residual Pred (um)",
    "SEC_RES":   "Secondary Coating Residual Pred (um)",

    "MAIN_IMP_UM": "Main Coating Prediction Improvement (um)",
    "MAIN_IMP_PCT": "Main Coating Prediction Improvement (%)",
    "MAIN_VERDICT": "Main Coating Prediction Verdict",

    "SEC_IMP_UM": "Secondary Coating Prediction Improvement (um)",
    "SEC_IMP_PCT": "Secondary Coating Prediction Improvement (%)",
    "SEC_VERDICT": "Secondary Coating Prediction Verdict",
}

OUT_UNITS = {
    OUT_KEYS["MAIN_CORR"]: "um",
    OUT_KEYS["SEC_CORR"]: "um",
    OUT_KEYS["MAIN_RES"]: "um",
    OUT_KEYS["SEC_RES"]: "um",

    OUT_KEYS["MAIN_IMP_UM"]: "um",
    OUT_KEYS["MAIN_IMP_PCT"]: "%",
    OUT_KEYS["MAIN_VERDICT"]: "",

    OUT_KEYS["SEC_IMP_UM"]: "um",
    OUT_KEYS["SEC_IMP_PCT"]: "%",
    OUT_KEYS["SEC_VERDICT"]: "",
}


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Parameter Name" not in df.columns:
        raise ValueError("CSV must contain 'Parameter Name'")
    if "Value" not in df.columns:
        raise ValueError("CSV must contain 'Value'")
    if "Units" not in df.columns:
        df["Units"] = ""
    return df


def _upsert(df: pd.DataFrame, name: str, value: Any, units: str = "") -> pd.DataFrame:
    df = df.copy()
    mask = df["Parameter Name"].astype(str).str.strip() == name
    if mask.any():
        idx = df.index[mask][0]
        df.at[idx, "Value"] = value
        df.at[idx, "Units"] = units
    else:
        df = pd.concat([df, pd.DataFrame([{
            "Parameter Name": name,
            "Value": value,
            "Units": units
        }])], ignore_index=True)
    return df


def _improvement(real: float, theory: float, corrected: float):
    if not all(map(math.isfinite, [real, theory, corrected])):
        return None, None, "N/A"

    err_th = abs(real - theory)
    err_corr = abs(real - corrected)

    delta = err_th - err_corr
    pct = 0.0 if err_th <= 1e-12 else 100.0 * delta / err_th

    if abs(delta) < 1e-9:
        verdict = "Same"
    elif delta > 0:
        verdict = "Improved"
    else:
        verdict = "Worse"

    return delta, pct, verdict


def write_corrected_predictions_into_dataset_csv(
    dataset_csv_path: str,
    save_residuals: bool = True,
    make_backup: bool = True
) -> Dict[str, Any]:

    if not os.path.exists(dataset_csv_path):
        raise FileNotFoundError(dataset_csv_path)

    pred = predict_corrected_from_dataset_csv(dataset_csv_path)

    # optional backup
    backup_path: Optional[str] = None
    if make_backup:
        backup_path = dataset_csv_path + ".bak"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(dataset_csv_path, backup_path)

    df = pd.read_csv(dataset_csv_path)
    df = _ensure_cols(df)
    df["Parameter Name"] = df["Parameter Name"].astype(str).str.strip()

    # values
    d_main_th = float(pred["D_main_th_um"])
    d_sec_th = float(pred["D_sec_th_um"])
    d_main_corr = float(pred["D_main_corrected_um"])
    d_sec_corr = float(pred["D_sec_corrected_um"])

    d_main_real = pred.get("D_main_real_um", None)
    d_sec_real = pred.get("D_sec_real_um", None)

    # Write corrected
    df = _upsert(df, OUT_KEYS["MAIN_CORR"], round(d_main_corr, 6), OUT_UNITS[OUT_KEYS["MAIN_CORR"]])
    df = _upsert(df, OUT_KEYS["SEC_CORR"],  round(d_sec_corr, 6),  OUT_UNITS[OUT_KEYS["SEC_CORR"]])

    # residuals
    if save_residuals:
        df = _upsert(df, OUT_KEYS["MAIN_RES"], round(float(pred["residual_main_clamped_um"]), 6), OUT_UNITS[OUT_KEYS["MAIN_RES"]])
        df = _upsert(df, OUT_KEYS["SEC_RES"],  round(float(pred["residual_sec_clamped_um"]), 6),  OUT_UNITS[OUT_KEYS["SEC_RES"]])

    # improvement metrics (only if measured exists)
    written = {
        OUT_KEYS["MAIN_CORR"]: d_main_corr,
        OUT_KEYS["SEC_CORR"]: d_sec_corr,
    }

    if d_main_real is not None:
        imp_um, imp_pct, verdict = _improvement(float(d_main_real), d_main_th, d_main_corr)
        if imp_um is not None:
            df = _upsert(df, OUT_KEYS["MAIN_IMP_UM"], round(float(imp_um), 6), OUT_UNITS[OUT_KEYS["MAIN_IMP_UM"]])
            df = _upsert(df, OUT_KEYS["MAIN_IMP_PCT"], round(float(imp_pct), 2), OUT_UNITS[OUT_KEYS["MAIN_IMP_PCT"]])
            df = _upsert(df, OUT_KEYS["MAIN_VERDICT"], verdict, "")
            written[OUT_KEYS["MAIN_IMP_UM"]] = float(imp_um)
            written[OUT_KEYS["MAIN_IMP_PCT"]] = float(imp_pct)
            written[OUT_KEYS["MAIN_VERDICT"]] = verdict

    if d_sec_real is not None:
        imp_um, imp_pct, verdict = _improvement(float(d_sec_real), d_sec_th, d_sec_corr)
        if imp_um is not None:
            df = _upsert(df, OUT_KEYS["SEC_IMP_UM"], round(float(imp_um), 6), OUT_UNITS[OUT_KEYS["SEC_IMP_UM"]])
            df = _upsert(df, OUT_KEYS["SEC_IMP_PCT"], round(float(imp_pct), 2), OUT_UNITS[OUT_KEYS["SEC_IMP_PCT"]])
            df = _upsert(df, OUT_KEYS["SEC_VERDICT"], verdict, "")
            written[OUT_KEYS["SEC_IMP_UM"]] = float(imp_um)
            written[OUT_KEYS["SEC_IMP_PCT"]] = float(imp_pct)
            written[OUT_KEYS["SEC_VERDICT"]] = verdict

    df.to_csv(dataset_csv_path, index=False)

    return {
        "ok": True,
        "dataset_csv": dataset_csv_path,
        "backup": backup_path,
        "written": written,
    }


if __name__ == "__main__":
    test_csv = os.path.join("..", "data_set_csv", "FAKE_DRAW_0001.csv")
    out = write_corrected_predictions_into_dataset_csv(test_csv, save_residuals=True, make_backup=True)
    print("✅ Wrote:", out["dataset_csv"])
    print("Backup:", out["backup"])
    for k, v in out["written"].items():
        print(f"{k}: {v}")