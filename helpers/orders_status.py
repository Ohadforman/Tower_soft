import datetime as dt
import os
import pandas as pd

STATUS_COL = "Status"
STATUS_UPDATED_COL = "Status Updated At"
FAILED_REASON_COL = "Failed Description"

def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_dt_safe(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(str(x), errors="coerce")
    except Exception:
        return None

def ensure_orders_cols(df: pd.DataFrame) -> pd.DataFrame:
    if STATUS_COL not in df.columns:
        df[STATUS_COL] = "Pending"
    if STATUS_UPDATED_COL not in df.columns:
        df[STATUS_UPDATED_COL] = ""
    if FAILED_REASON_COL not in df.columns:
        df[FAILED_REASON_COL] = ""
    return df

def auto_move_failed_to_pending_after_days(days: int, orders_file: str):
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

    df = ensure_orders_cols(df)
    if df.empty:
        return False, 0, "No orders."

    now = pd.Timestamp(dt.datetime.now())
    cutoff = now - pd.Timedelta(days=days)

    moved = 0
    for i in range(len(df)):
        if str(df.at[i, STATUS_COL]).strip().lower() != "failed":
            continue

        t = _parse_dt_safe(df.at[i, STATUS_UPDATED_COL])
        if t is None or pd.isna(t):
            df.at[i, STATUS_UPDATED_COL] = _now_str()
            continue

        if t < cutoff:
            df.at[i, STATUS_COL] = "Pending"
            df.at[i, STATUS_UPDATED_COL] = _now_str()
            moved += 1

    try:
        df.to_csv(orders_file, index=False)
    except Exception as e:
        return False, moved, f"Failed to write {orders_file}: {e}"

    if moved > 0:
        return True, moved, f"Moved {moved} Failed order(s) back to Pending (older than {days} days)."
    return False, 0, "No Failed orders needed moving."