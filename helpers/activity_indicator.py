import csv
import json
import os
from datetime import datetime
from typing import Optional


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def record_activity_start(
    *,
    indicator_json_path: str,
    events_csv_path: str,
    activity_type: str,
    title: str = "",
    actor: str = "",
    source: str = "",
    meta: Optional[dict] = None,
) -> None:
    """
    Mark current active activity + append a start event.
    This is used as software-side signal for future physical indicator integration.
    """
    ts = _now_ts()
    meta = meta or {}

    _ensure_parent(indicator_json_path)
    _ensure_parent(events_csv_path)

    state = {
        "active": True,
        "event": "start",
        "event_ts": ts,
        "activity_type": str(activity_type or "").strip(),
        "title": str(title or "").strip(),
        "actor": str(actor or "").strip(),
        "source": str(source or "").strip(),
        "meta": meta,
    }
    with open(indicator_json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    header = ["event_ts", "event", "activity_type", "title", "actor", "source", "meta_json"]
    write_header = not os.path.exists(events_csv_path) or os.path.getsize(events_csv_path) == 0
    with open(events_csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(
            [
                ts,
                "start",
                state["activity_type"],
                state["title"],
                state["actor"],
                state["source"],
                json.dumps(meta, ensure_ascii=False),
            ]
        )
