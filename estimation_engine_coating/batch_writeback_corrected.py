import os
import glob
import pandas as pd
from typing import List, Dict, Any

from estimation_engine_coating.coating_writeback import write_corrected_predictions_into_dataset_csv


DATASET_DIR = "../data_set_csv"
ONLY_PREFIX = "FAKE"   # set to "" for all real later
MAKE_BACKUP = True
SAVE_RESIDUALS = True


def list_target_csvs() -> List[str]:
    pattern = f"{ONLY_PREFIX}*.csv" if ONLY_PREFIX else "*.csv"
    return sorted(glob.glob(os.path.join(DATASET_DIR, pattern)))


def main():
    paths = list_target_csvs()
    if not paths:
        raise SystemExit(f"No CSVs found in {DATASET_DIR} with prefix '{ONLY_PREFIX}'")

    ok = 0
    fail = 0
    results: List[Dict[str, Any]] = []

    for p in paths:
        try:
            out = write_corrected_predictions_into_dataset_csv(
                dataset_csv_path=p,
                save_residuals=SAVE_RESIDUALS,
                make_backup=MAKE_BACKUP
            )
            ok += 1
            results.append({
                "file": os.path.basename(p),
                "ok": True,
                "written": out.get("written", {})
            })
        except Exception as e:
            fail += 1
            results.append({
                "file": os.path.basename(p),
                "ok": False,
                "error": str(e)
            })

    print("\n✅ Batch writeback finished")
    print("Total:", len(paths), "| OK:", ok, "| Failed:", fail)

    if fail:
        print("\nFirst failures:")
        for r in results:
            if not r["ok"]:
                print("-", r["file"], "->", r["error"])
                break

    df = pd.DataFrame(results)

    # Make preview easier to read (flatten written dict for first few columns)
    if "written" in df.columns:
        # pull a few common fields if exist
        def getw(d, k):
            try:
                return d.get(k)
            except Exception:
                return None

        df["main_corr_um"] = df["written"].apply(lambda d: getw(d, "Main Coating Diameter Corrected Pred (um)"))
        df["sec_corr_um"] = df["written"].apply(lambda d: getw(d, "Secondary Coating Diameter Corrected Pred (um)"))
        df["main_verdict"] = df["written"].apply(lambda d: getw(d, "Main Coating Prediction Verdict"))
        df["sec_verdict"] = df["written"].apply(lambda d: getw(d, "Secondary Coating Prediction Verdict"))

    print("\nPreview:")
    cols = [c for c in ["file", "ok", "main_corr_um", "sec_corr_um", "main_verdict", "sec_verdict"] if c in df.columns]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()