# helpers/dataset_csv_io.py
from __future__ import annotations

import os
from typing import List, Dict, Tuple

import pandas as pd

from app_io.paths import P, dataset_csv_path


def append_rows_to_dataset_csv(selected_csv: str, rows: List[Dict]) -> Tuple[bool, str]:
    """
    Appends rows (Parameter Name, Value, Units) to a dataset CSV.
    selected_csv can be:
      - filename only: ABC.csv
      - full path: /.../data_set_csv/ABC.csv
    Returns: (ok, message)
    """
    if not selected_csv:
        return False, "No CSV selected."

    # Resolve path
    csv_path = selected_csv if os.path.exists(selected_csv) else dataset_csv_path(os.path.basename(selected_csv))
    os.makedirs(P.dataset_dir, exist_ok=True)

    # Ensure file exists with correct columns
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Parameter Name", "Value", "Units"]).to_csv(csv_path, index=False)

    try:
        df = pd.read_csv(csv_path, keep_default_na=False)
    except Exception as e:
        return False, f"Failed reading dataset CSV: {e}"

    # Ensure columns exist
    for col in ["Parameter Name", "Value", "Units"]:
        if col not in df.columns:
            df[col] = ""

    df = df[["Parameter Name", "Value", "Units"]]

    new_df = pd.DataFrame(rows)
    for col in ["Parameter Name", "Value", "Units"]:
        if col not in new_df.columns:
            new_df[col] = ""

    new_df = new_df[["Parameter Name", "Value", "Units"]]

    try:
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        return True, f"Saved {len(new_df)} rows into {os.path.basename(csv_path)}"
    except Exception as e:
        return False, f"Failed writing dataset CSV: {e}"