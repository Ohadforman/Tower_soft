# app_io/dataset.py
from __future__ import annotations
import os
import pandas as pd
from app_io.paths import P, ensure_dir, safe_filename

def ensure_dataset_dir() -> str:
    return ensure_dir(P.dataset_dir)

def dataset_path(filename: str) -> str:
    ensure_dataset_dir()
    return os.path.join(P.dataset_dir, safe_filename(filename))

def list_dataset_csvs() -> list[str]:
    ensure_dataset_dir()
    out = []
    for f in os.listdir(P.dataset_dir):
        if f.lower().endswith(".csv"):
            out.append(f)
    return sorted(out)

def most_recent_csv() -> str:
    files = list_dataset_csvs()
    if not files:
        return ""
    paths = [os.path.join(P.dataset_dir, f) for f in files]
    newest = max(paths, key=lambda p: os.path.getmtime(p))
    return os.path.basename(newest)

def append_rows_to_dataset_csv(dataset_csv_filename: str, rows: list[dict]):
    if not dataset_csv_filename:
        return False, "No dataset CSV name."
    path = dataset_path(dataset_csv_filename)
    if not os.path.exists(path):
        return False, f"Dataset CSV not found: {path}"

    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception as e:
        return False, f"Failed to read dataset csv: {e}"

    if "Parameter Name" not in df.columns or "Value" not in df.columns:
        return False, "Dataset CSV has unexpected format (needs 'Parameter Name' and 'Value')."

    if "Units" not in df.columns:
        df["Units"] = ""

    add_df = pd.DataFrame(rows)
    if "Units" not in add_df.columns:
        add_df["Units"] = ""

    out = pd.concat([df, add_df], ignore_index=True)

    try:
        out.to_csv(path, index=False)
    except Exception as e:
        return False, f"Failed to write dataset csv: {e}"

    return True, f"Updated dataset CSV: {dataset_csv_filename}"