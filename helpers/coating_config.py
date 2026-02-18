# helpers/coating_config.py
import json
import os
from typing import Dict, Any, List


def load_config_coating_json(path: str) -> Dict[str, Any]:
    """Loads coating+die config from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Coating config not found: {os.path.abspath(path)}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coating_options_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    coats = (cfg or {}).get("coatings", {})
    if isinstance(coats, dict):
        return [str(k).strip() for k in coats.keys() if str(k).strip()]
    return []

def load_coating_config(path: str = "config_coating.json") -> Dict[str, Any]:
    """
    Backward-compatible alias used by renders/coating.py and Process Setup.
    If you pass a path (recommended: P.coating_config_json), it will use it.
    """
    return load_config_coating_json(path)