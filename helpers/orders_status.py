import datetime as dt
import pandas as pd

from helpers.constants import FAILED_DESC_COL, STATUS_COL, STATUS_UPDATED_COL
from orders.lifecycle import (
    auto_move_failed_to_pending_after_days as lifecycle_auto_move_failed_to_pending_after_days,
    ensure_orders_cols as lifecycle_ensure_orders_cols,
)

FAILED_REASON_COL = FAILED_DESC_COL


def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_dt_safe(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(str(x), errors="coerce")
    except Exception:
        return None


def ensure_orders_cols(
    df: pd.DataFrame,
    status_col: str = STATUS_COL,
    status_updated_col: str = STATUS_UPDATED_COL,
    failed_reason_col: str = FAILED_REASON_COL,
) -> pd.DataFrame:
    # Compatibility shim: keep old import path but use lifecycle implementation.
    return lifecycle_ensure_orders_cols(
        df,
        status_col=status_col,
        status_updated_col=status_updated_col,
        failed_reason_col=failed_reason_col,
    )


def auto_move_failed_to_pending_after_days(days: int, orders_file: str):
    return lifecycle_auto_move_failed_to_pending_after_days(days=days, orders_file=orders_file)
