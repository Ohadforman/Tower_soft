# helpers/json_io.py
from __future__ import annotations
import json
import os
from typing import Any, Optional

def load_json(path: str, default: Optional[Any] = None) -> Any:
    """
    Safe JSON loader. Returns `default` if missing/bad.
    """
    if default is None:
        default = {}
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default