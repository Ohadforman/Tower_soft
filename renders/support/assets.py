# helpers/assets.py
from __future__ import annotations
import base64
import os

def get_base64_image(path: str) -> str:
    """
    Returns base64 string of an image file. Empty string if missing.
    """
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""