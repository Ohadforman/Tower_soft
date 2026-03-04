from __future__ import annotations

import json
import os
from datetime import datetime

from app_io.paths import P


def log_event(event: str, **fields) -> None:
    try:
        os.makedirs(P.logs_dir, exist_ok=True)
        day = datetime.now().strftime("%Y%m%d")
        path = os.path.join(P.logs_dir, f"app_events_{day}.log")
        payload = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **fields,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        # Logging must never break app flow.
        pass
