import os
import datetime as dt
import pandas as pd


def _now_str():
    # keep it simple (local machine time)
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_dt_safe(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(str(x), errors="coerce")
    except Exception:
        return None


def ensure_orders_cols(
    df: pd.DataFrame,
    status_col: str = "Status",
    status_updated_col: str = "Status Updated At",
    failed_reason_col: str = "Failed Reason",
) -> pd.DataFrame:
    """Ensure required columns exist (non-breaking)."""
    if status_col not in df.columns:
        df[status_col] = "Pending"
    if status_updated_col not in df.columns:
        df[status_updated_col] = ""
    if failed_reason_col not in df.columns:
        df[failed_reason_col] = ""
    return df


def auto_move_failed_to_pending_after_days(
    days: int = 4,
    orders_file: str = "draw_orders.csv",
    status_col: str = "Status",
    status_updated_col: str = "Status Updated At",
):
    """
    If Status == Failed and Status Updated At is older than <days>, move back to Pending.
    Returns (changed: bool, moved_count: int, msg: str)
    """
    if not os.path.exists(orders_file):
        return False, 0, f"{orders_file} not found."

    try:
        df = pd.read_csv(orders_file)
    except Exception as e:
        return False, 0, f"Failed to read {orders_file}: {e}"

    df = ensure_orders_cols(df, status_col=status_col, status_updated_col=status_updated_col)

    if df.empty:
        return False, 0, "No orders."

    now = pd.Timestamp(dt.datetime.now())
    cutoff = now - pd.Timedelta(days=days)

    moved = 0
    for i in range(len(df)):
        if str(df.at[i, status_col]).strip().lower() != "failed":
            continue

        t = _parse_dt_safe(df.at[i, status_updated_col])

        # If we have no timestamp, set it now (so it will age properly next time)
        if t is None or pd.isna(t):
            df.at[i, status_updated_col] = _now_str()
            continue

        if t < cutoff:
            df.at[i, status_col] = "Pending"
            df.at[i, status_updated_col] = _now_str()
            moved += 1

    # Always write back (small file, safe)
    try:
        df.to_csv(orders_file, index=False)
    except Exception as e:
        return False, moved, f"Failed to write {orders_file}: {e}"

    if moved > 0:
        return True, moved, f"Moved {moved} Failed order(s) back to Pending (older than {days} days)."
    return False, 0, "No Failed orders needed moving."