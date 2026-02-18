# helpers/params_io.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def get_float_param(df_params: pd.DataFrame, name: str, default: float = np.nan) -> float:
    """
    Read a numeric parameter from a dataset params CSV (Parameter Name / Value).
    Returns float(default) if missing or not parseable.
    """
    try:
        m = df_params["Parameter Name"].astype(str).str.strip().eq(str(name).strip())
        if m.any():
            v = df_params.loc[m, "Value"].iloc[-1]
            return float(str(v).strip())
    except Exception:
        pass
    return float(default)


def get_value(df_params: pd.DataFrame, name: str, default: Any = None) -> Any:
    """
    Get the last value for a parameter name (exact match).
    """
    try:
        hit = df_params.loc[df_params["Parameter Name"].astype(str) == str(name), "Value"]
        if hit.empty:
            return default
        return hit.iloc[-1]
    except Exception:
        return default


def param_map(df_params: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert (Parameter Name, Value) rows to a dict (last value wins).
    """
    out: Dict[str, Any] = {}
    if df_params is None or df_params.empty:
        return out

    for _, r in df_params.iterrows():
        k = str(r.get("Parameter Name", "")).strip()
        if not k:
            continue
        out[k] = r.get("Value", "")
    return out