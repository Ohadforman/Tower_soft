# helpers/orders_io.py
from __future__ import annotations

import pandas as pd

from app_io.paths import P
from helpers.dataset_io import list_dataset_csvs
from helpers.constants import (
    STATUS_COL,
    STATUS_UPDATED_COL,
    FAILED_DESC_COL,
    TRY_COUNT_COL,
    LAST_TRY_TIME_COL,
    LAST_TRY_DATASET_COL,
)


# Columns that might contain the dataset csv link (we check in this order)
DATASET_LINK_COLS = [
    "Assigned Dataset CSV",
    "Failed CSV",
    "Active CSV",
    "Done CSV",
]


def find_associated_dataset_csv(order_row: pd.Series) -> str:
    """
    Returns the best-known dataset CSV filename for an order row.
    Checks several possible columns (Assigned/Failed/Active/Done).
    """
    for col in DATASET_LINK_COLS:
        if col in order_row.index:
            val = str(order_row.get(col, "")).strip()
            if val and val.lower() != "nan":
                return val
    return ""


def ensure_orders_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the minimal columns we rely on exist.
    (Non-destructive: adds columns if missing)
    """
    if STATUS_COL not in df.columns:
        df[STATUS_COL] = "Pending"
    if FAILED_DESC_COL not in df.columns:
        df[FAILED_DESC_COL] = ""
    if STATUS_UPDATED_COL not in df.columns:
        df[STATUS_UPDATED_COL] = ""

    if TRY_COUNT_COL not in df.columns:
        df[TRY_COUNT_COL] = 0
    if LAST_TRY_TIME_COL not in df.columns:
        df[LAST_TRY_TIME_COL] = ""
    if LAST_TRY_DATASET_COL not in df.columns:
        df[LAST_TRY_DATASET_COL] = ""

    return df


def read_orders_csv(path: str | None = None) -> pd.DataFrame:
    """
    Reads orders CSV (default: P.orders_csv). Returns empty DF if missing.
    """
    path = path or P.orders_csv
    if not path:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        # keep it simple; UI layer can catch/print if needed
        return pd.DataFrame()

    return ensure_orders_cols(df)


def write_orders_csv(df: pd.DataFrame, path: str | None = None) -> bool:
    """
    Writes orders CSV (default: P.orders_csv).
    """
    path = path or P.orders_csv
    try:
        df.to_csv(path, index=False)
        return True
    except Exception:
        return False


def count_draw_orders(path: str | None = None) -> int:
    """
    Authoritative draw count from draw_orders.csv (not logs / dataset folder file-count).
    Counts meaningful order rows and ignores fully blank template rows.
    """
    df = read_orders_csv(path=path)
    if df is None or df.empty:
        return 0

    # Keep rows that have at least one meaningful order field.
    cols = [c for c in ["Timestamp", "Fiber Project", "Preform Number", "Status"] if c in df.columns]
    if not cols:
        return int(len(df))

    keep_mask = pd.Series(False, index=df.index)
    for c in cols:
        keep_mask = keep_mask | df[c].astype(str).str.strip().ne("")
    return int(keep_mask.sum())


def count_dataset_draws(dataset_dir: str | None = None) -> int:
    """
    Count dataset draw CSV files directly from dataset directory.
    This reflects actual draw files in data_set_csv.
    """
    try:
        ddir = dataset_dir or P.dataset_dir
        return int(len(list_dataset_csvs(ddir, full_paths=False)))
    except Exception:
        return 0
