# helpers/dataset_io.py
from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd


# ==========================================================
# Directory + listing helpers
# ==========================================================
def ensure_dataset_dir(dataset_dir: str) -> None:
    if not dataset_dir:
        # don't mkdir("") ever
        return
    os.makedirs(dataset_dir, exist_ok=True)


def list_dataset_csvs(dataset_dir: str, full_paths: bool = False) -> List[str]:
    ensure_dataset_dir(dataset_dir)
    if not dataset_dir or not os.path.exists(dataset_dir):
        return []
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    files = sorted(files)
    if full_paths:
        return [os.path.join(dataset_dir, f) for f in files]
    return files


def most_recent_csv(dataset_dir: str) -> Optional[str]:
    ensure_dataset_dir(dataset_dir)
    if not dataset_dir or not os.path.exists(dataset_dir):
        return None
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    if not files:
        return None
    latest_path = max(files, key=os.path.getmtime)
    return os.path.basename(latest_path)


# ==========================================================
# Path resolver (the missing piece)
# ==========================================================
def resolve_dataset_csv_path(csv_path_or_name: str, dataset_dir: Optional[str] = None) -> str:
    """
    Accepts either:
      - full path: /.../data_set_csv/ABC.csv
      - filename only: ABC.csv  -> joined with dataset_dir

    dataset_dir:
      - if None, try env DATASET_DIR, else default "data_set_csv"
    """
    s = (csv_path_or_name or "").strip()
    if not s:
        return ""

    # If already looks like a path (absolute or contains a separator), keep it
    if os.path.isabs(s) or (os.path.sep in s) or ("/" in s) or ("\\" in s):
        return os.path.normpath(s)

    base_dir = (dataset_dir or os.getenv("DATASET_DIR") or "data_set_csv").strip()
    if not base_dir:
        base_dir = "data_set_csv"

    return os.path.normpath(os.path.join(base_dir, s))


# ==========================================================
# Read / append
# ==========================================================
def read_dataset_csv(csv_path: str) -> Optional[pd.DataFrame]:
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path, keep_default_na=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    for c in ["Parameter Name", "Value", "Units"]:
        if c not in df.columns:
            df[c] = ""
    return df[["Parameter Name", "Value", "Units"]]


def append_rows_to_dataset_csv(
    csv_path_or_name: str,
    rows: List[Dict[str, Any]],
    dataset_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Safe append.
    - csv_path_or_name can be filename or full path.
    - dataset_dir optional; if missing uses env DATASET_DIR or default "data_set_csv".
    """
    # sanitize
    rows = rows or []
    csv_path = resolve_dataset_csv_path(csv_path_or_name, dataset_dir=dataset_dir)

    if not csv_path:
        return False, "No CSV path provided."

    out_dir = os.path.dirname(csv_path)
    if not out_dir:
        # this is the exact bug you hit before
        return False, f"Invalid dataset CSV path (no directory): {csv_path}"

    os.makedirs(out_dir, exist_ok=True)

    # create file if missing
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Parameter Name", "Value", "Units"]).to_csv(csv_path, index=False)

    df = read_dataset_csv(csv_path)
    if df is None:
        df = pd.DataFrame(columns=["Parameter Name", "Value", "Units"])

    new_rows = pd.DataFrame(rows)
    for c in ["Parameter Name", "Value", "Units"]:
        if c not in new_rows.columns:
            new_rows[c] = ""

    out = pd.concat([df, new_rows[["Parameter Name", "Value", "Units"]]], ignore_index=True)
    out.to_csv(csv_path, index=False)

    return True, f"Appended {len(rows)} rows to {os.path.basename(csv_path)}."