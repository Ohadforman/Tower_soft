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
ONLY_PREFIX = "FAKE"  # "" for all
WINDOWS = [10, 20, 50]

THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(THIS_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)
OUT_JSON = os.path.join(MODEL_DIR, "trust_report.json")

# Parameter names written by coating_writeback.py
KEYS = {
    "MAIN_IMP": "Main Coating Prediction Improvement (um)",
    "MAIN_VER": "Main Coating Prediction Verdict",
    "SEC_IMP": "Secondary Coating Prediction Improvement (um)",
    "SEC_VER": "Secondary Coating Prediction Verdict",
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
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, pattern)))
    return paths


def load_param_table(path: str) -> pd.DataFrame:
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


def compute_window_stats(records: List[Dict[str, Any]], window: int) -> Dict[str, Any]:
    tail = records[-window:] if len(records) >= window else records[:]
    if not tail:
        return {"window": window, "n": 0}

    def stats_for(prefix: str) -> Dict[str, Any]:
        imp_key = KEYS[f"{prefix}_IMP"]
        ver_key = KEYS[f"{prefix}_VER"]

        imps = []
        verdicts = []

        for r in tail:
            imp = r.get(imp_key, None)
            ver = r.get(ver_key, None)
            if imp is not None:
                imps.append(float(imp))
            if ver is not None:
                verdicts.append(str(ver))

        n_imp = len(imps)
        n_ver = len(verdicts)

        improved = sum(1 for v in verdicts if v == "Improved")
        worse = sum(1 for v in verdicts if v == "Worse")
        same = sum(1 for v in verdicts if v == "Same")

        out = {
            "n_with_improvement_um": n_imp,
            "n_with_verdict": n_ver,
            "pct_improved": (100.0 * improved / n_ver) if n_ver else None,
            "pct_worse": (100.0 * worse / n_ver) if n_ver else None,
            "pct_same": (100.0 * same / n_ver) if n_ver else None,
            "mean_improvement_um": (sum(imps) / n_imp) if n_imp else None,
            "median_improvement_um": (float(pd.Series(imps).median()) if n_imp else None),
            "best_improvement_um": (max(imps) if n_imp else None),
            "worst_improvement_um": (min(imps) if n_imp else None),
        }
        return out

    return {
        "window": window,
        "n": len(tail),
        "main": stats_for("MAIN"),
        "secondary": stats_for("SEC"),
    }


def main():
    paths = list_csvs()
    if not paths:
        raise SystemExit(f"No CSVs found in {DATASET_DIR} with prefix '{ONLY_PREFIX}'")

    records: List[Dict[str, Any]] = []
    bad = 0

    for p in paths:
        try:
            dfp = load_param_table(p)
            rec = {
                "dataset_csv": os.path.basename(p),
                KEYS["MAIN_IMP"]: _to_float(get_param(dfp, KEYS["MAIN_IMP"])),
                KEYS["SEC_IMP"]: _to_float(get_param(dfp, KEYS["SEC_IMP"])),
                KEYS["MAIN_VER"]: get_param(dfp, KEYS["MAIN_VER"]),
                KEYS["SEC_VER"]: get_param(dfp, KEYS["SEC_VER"]),
            }
            # normalize verdict strings
            for k in (KEYS["MAIN_VER"], KEYS["SEC_VER"]):
                if rec[k] is not None:
                    rec[k] = str(rec[k]).strip()
            records.append(rec)
        except Exception:
            bad += 1

    print("\n=== ML Trust Tracker ===")
    print("Files:", len(paths), "| Parsed:", len(records), "| Failed:", bad)

    report = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": DATASET_DIR,
        "only_prefix": ONLY_PREFIX,
        "total_files": len(paths),
        "parsed_files": len(records),
        "failed_files": bad,
        "windows": [],
    }

    for w in WINDOWS:
        stats = compute_window_stats(records, w)
        report["windows"].append(stats)

        print(f"\n--- Last {stats['window']} draws (n={stats['n']}) ---")

        m = stats["main"]
        s = stats["secondary"]

        print("MAIN:")
        print(f"  Improved%: {m['pct_improved']:.1f} | Worse%: {m['pct_worse']:.1f} | Same%: {m['pct_same']:.1f}" if m["pct_improved"] is not None else "  (no verdicts)")
        if m["mean_improvement_um"] is not None:
            print(f"  Mean Δerr (um): {m['mean_improvement_um']:+.3f} | Median: {m['median_improvement_um']:+.3f} | Best: {m['best_improvement_um']:+.3f} | Worst: {m['worst_improvement_um']:+.3f}")

        print("SECONDARY:")
        print(f"  Improved%: {s['pct_improved']:.1f} | Worse%: {s['pct_worse']:.1f} | Same%: {s['pct_same']:.1f}" if s["pct_improved"] is not None else "  (no verdicts)")
        if s["mean_improvement_um"] is not None:
            print(f"  Mean Δerr (um): {s['mean_improvement_um']:+.3f} | Median: {s['median_improvement_um']:+.3f} | Best: {s['best_improvement_um']:+.3f} | Worst: {s['worst_improvement_um']:+.3f}")

    # Save JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Saved:", OUT_JSON)


if __name__ == "__main__":
    main()