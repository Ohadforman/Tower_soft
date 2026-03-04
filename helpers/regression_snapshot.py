from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd


def _count_status(df: pd.DataFrame, col: str = "Status") -> Dict[str, int]:
    if df is None or df.empty or col not in df.columns:
        return {}
    s = df[col].astype(str).str.strip().str.lower()
    out = s.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in out.items()}


def compute_snapshot(P) -> Dict:
    snap = {
        "orders_total": 0,
        "orders_status": {},
        "parts_total": 0,
        "parts_status": {},
        "schedule_total": 0,
        "latest_log_rows": 0,
    }

    if os.path.exists(P.orders_csv):
        dfo = pd.read_csv(P.orders_csv, keep_default_na=False)
        snap["orders_total"] = int(len(dfo))
        snap["orders_status"] = _count_status(dfo, "Status")

    if os.path.exists(P.parts_orders_csv):
        dfp = pd.read_csv(P.parts_orders_csv, keep_default_na=False)
        snap["parts_total"] = int(len(dfp))
        snap["parts_status"] = _count_status(dfp, "Status")

    if os.path.exists(P.schedule_csv):
        dfs = pd.read_csv(P.schedule_csv, keep_default_na=False)
        snap["schedule_total"] = int(len(dfs))

    if os.path.isdir(P.logs_dir):
        logs = [os.path.join(P.logs_dir, f) for f in os.listdir(P.logs_dir) if f.lower().endswith(".csv")]
        if logs:
            latest = max(logs, key=os.path.getmtime)
            dfl = pd.read_csv(latest, keep_default_na=False)
            snap["latest_log_rows"] = int(len(dfl))

    return snap


def load_baseline(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(path: str, snapshot: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
