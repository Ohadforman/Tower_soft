import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

import joblib

from estimation_engine_coating.coating_predictor import (
    extract_engineered_features,
    load_effective_metadata,
    resolve_best_model_paths,
)


def _get_pipeline_parts(pipe):
    pre = pipe.named_steps.get("pre")
    reg = pipe.named_steps.get("reg") or pipe.named_steps.get("ridge") or pipe.named_steps.get("huber")
    return pre, reg


def _get_feature_names(pre) -> List[str]:
    # sklearn >= 1.0 usually supports get_feature_names_out()
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        # fallback: generic names
        return [f"f{i}" for i in range(pre.transformers_[0][1].named_steps["scaler"].n_features_in_)]


def top_global_coeffs(pipe, top_n: int = 25) -> List[Tuple[str, float]]:
    pre, reg = _get_pipeline_parts(pipe)
    names = _get_feature_names(pre)
    coef = np.asarray(reg.coef_, dtype=float).ravel()

    pairs = list(zip(names, coef))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


def explain_one_sample(pipe, X_row: pd.DataFrame, top_n: int = 20) -> List[Tuple[str, float]]:
    pre, reg = _get_pipeline_parts(pipe)
    names = _get_feature_names(pre)

    X_trans = pre.transform(X_row)
    # X_trans can be sparse
    try:
        X_vec = np.asarray(X_trans.todense()).ravel()
    except Exception:
        X_vec = np.asarray(X_trans).ravel()

    coef = np.asarray(reg.coef_, dtype=float).ravel()
    contrib = X_vec * coef

    pairs = list(zip(names, contrib))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


def build_X_from_dataset_csv(dataset_csv_path: str, meta: Dict[str, Any], which: str) -> pd.DataFrame:
    pack = extract_engineered_features(dataset_csv_path)
    feats = pack["features"]

    features_block = meta.get("features", {})
    if features_block.get("main") and features_block.get("secondary"):
        if which == "main":
            num = features_block["main"]["num"]
            cat = features_block["main"]["cat"]
        else:
            num = features_block["secondary"]["num"]
            cat = features_block["secondary"]["cat"]
    else:
        # fallback
        base_num = [
            "speed_m_min", "tension_g", "furnace_temp_c",
            "main_coat_temp_c", "sec_coat_temp_c", "bare_fiber_um",
            "speed_over_tension", "delta_furnace", "delta_main_coat_temp", "delta_sec_coat_temp"
        ]
        base_cat = ["main_coating_type", "sec_coating_type"]
        if which == "main":
            num, cat = base_num, base_cat
        else:
            num, cat = base_num + ["D_main_th_um_as_feat"], base_cat

    X = pd.DataFrame([{**feats}], columns=num + cat)
    return X


def main():
    meta = load_effective_metadata()
    main_path, sec_path = resolve_best_model_paths()

    print("\n=== Ridge Feature Attribution ===")
    print("Main model:", main_path)
    print("Sec  model:", sec_path)

    main_model = joblib.load(main_path)
    sec_model = joblib.load(sec_path)

    print("\n--- TOP COEFFICIENTS (MAIN) ---")
    for name, w in top_global_coeffs(main_model, top_n=25):
        print(f"{w:+.6f}  {name}")

    print("\n--- TOP COEFFICIENTS (SECONDARY) ---")
    for name, w in top_global_coeffs(sec_model, top_n=25):
        print(f"{w:+.6f}  {name}")

    # Optional: explain a sample CSV if exists (FAKE first)
    test_csv = os.path.join("../data_set_csv", "FAKE_DRAW_0001.csv")
    if os.path.exists(test_csv):
        print("\n--- PER-SAMPLE CONTRIBUTIONS (FAKE_DRAW_0001.csv) ---")

        X_main = build_X_from_dataset_csv(test_csv, meta, which="main")
        pred_main = float(main_model.predict(X_main)[0])
        print(f"\nMAIN predicted residual (um): {pred_main:+.6f}")
        for name, c in explain_one_sample(main_model, X_main, top_n=15):
            print(f"{c:+.6f}  {name}")

        X_sec = build_X_from_dataset_csv(test_csv, meta, which="secondary")
        pred_sec = float(sec_model.predict(X_sec)[0])
        print(f"\nSECONDARY predicted residual (um): {pred_sec:+.6f}")
        for name, c in explain_one_sample(sec_model, X_sec, top_n=15):
            print(f"{c:+.6f}  {name}")
    else:
        print("\n(No FAKE_DRAW_0001.csv found for per-sample explanation.)")


if __name__ == "__main__":
    main()