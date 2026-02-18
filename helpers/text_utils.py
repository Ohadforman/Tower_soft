# helpers/text_utils.py
from __future__ import annotations
import pandas as pd
import datetime as dt

def safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

def to_float(v, default: float = 0.0) -> float:
    try:
        s = safe_str(v)
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def safe_int(v, default: int = 0) -> int:
    try:
        s = safe_str(v)
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fmt_km(v):
    s = safe_str(v)
    if s == "":
        return "—"
    try:
        return f"{float(s):.2f}"
    except Exception:
        return s