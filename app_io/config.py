# app_io/config.py
from __future__ import annotations
import json, os
from app_io.paths import P

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_coating_config() -> dict:
    return load_json(P.coating_config_json)

def coating_options_from_cfg(cfg: dict) -> list[str]:
    coats = (cfg or {}).get("coatings", {})
    if isinstance(coats, dict):
        return [str(k).strip() for k in coats.keys() if str(k).strip()]
    return []