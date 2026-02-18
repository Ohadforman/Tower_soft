# helpers/format_utils.py
from __future__ import annotations

import numpy as np
from helpers.text_utils import to_float, safe_str


def fmt_float(v, digits: int = 2, default: str = "—") -> str:
    """
    Safe float formatter:
      - accepts numbers or strings
      - returns default if empty/nan/unparseable
    """
    x = to_float(v, default=np.nan)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    return f"{x:.{int(digits)}f}"


def fmt_int(v, default: str = "—") -> str:
    s = safe_str(v)
    if s == "":
        return default
    try:
        return str(int(float(s)))
    except Exception:
        return default