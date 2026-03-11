import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from helpers.parts_inventory import decrement_part, increment_part, is_non_consumable_part


RESERVATION_COLS = [
    "reservation_id",
    "reservation_ts",
    "task_id",
    "component",
    "task",
    "part_name",
    "qty",
    "state",  # ACTIVE / CONSUMED / RELEASED
    "actor",
    "note",
    "updated_ts",
]


def _safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _task_key(task_id: str, component: str, task: str) -> str:
    return "|".join([_safe_str(task_id).lower(), _safe_str(component).lower(), _safe_str(task).lower()])


def _ensure_df(path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except Exception:
            df = pd.DataFrame(columns=RESERVATION_COLS)
    else:
        df = pd.DataFrame(columns=RESERVATION_COLS)
    for c in RESERVATION_COLS:
        if c not in df.columns:
            df[c] = ""
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0).astype(float)
    return df[RESERVATION_COLS].copy()


def _save_df(path: str, df: pd.DataFrame) -> None:
    out = df.copy()
    for c in RESERVATION_COLS:
        if c not in out.columns:
            out[c] = ""
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0).astype(float)
    out[RESERVATION_COLS].to_csv(path, index=False)


def list_task_reservations(path: str, *, task_id: str, component: str, task: str, active_only: bool = False) -> pd.DataFrame:
    df = _ensure_df(path)
    key = _task_key(task_id, component, task)
    cur_key = df.apply(
        lambda r: _task_key(r.get("task_id", ""), r.get("component", ""), r.get("task", "")),
        axis=1,
    )
    out = df[cur_key.eq(key)].copy()
    if active_only:
        out = out[out["state"].astype(str).str.upper().eq("ACTIVE")].copy()
    return out


def reserve_parts_for_task(
    *,
    reservations_csv_path: str,
    inventory_csv_path: str,
    task_id: str,
    component: str,
    task: str,
    parts: List[str],
    qty_per_part: float = 1.0,
    actor: str = "",
    note: str = "",
) -> Dict[str, object]:
    if not parts:
        return {"created": 0, "missing": [], "skipped_existing": 0}
    qty_per_part = max(0.0, float(qty_per_part))
    if qty_per_part <= 0:
        return {"created": 0, "missing": [], "skipped_existing": 0}

    df = _ensure_df(reservations_csv_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid_seed = int(datetime.now().timestamp() * 1000)
    created = 0
    missing = []
    skipped_existing = 0

    for i, p in enumerate(parts):
        part_name = _safe_str(p)
        if not part_name:
            continue
        # Skip when already active for this same task + part.
        active = list_task_reservations(
            reservations_csv_path,
            task_id=task_id,
            component=component,
            task=task,
            active_only=True,
        )
        if not active.empty:
            has_same = active["part_name"].astype(str).str.strip().str.lower().eq(part_name.lower()).any()
            if has_same:
                skipped_existing += 1
                continue

        ok = decrement_part(inventory_csv_path, part_name, qty=qty_per_part)
        if not ok:
            missing.append(part_name)
            continue

        row = {
            "reservation_id": str(rid_seed + i),
            "reservation_ts": now,
            "task_id": _safe_str(task_id),
            "component": _safe_str(component),
            "task": _safe_str(task),
            "part_name": part_name,
            "qty": float(qty_per_part),
            "state": "ACTIVE",
            "actor": _safe_str(actor),
            "note": _safe_str(note),
            "updated_ts": now,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        created += 1

    _save_df(reservations_csv_path, df)
    return {"created": int(created), "missing": missing, "skipped_existing": int(skipped_existing)}


def release_task_reservations(
    *,
    reservations_csv_path: str,
    inventory_csv_path: str,
    task_id: str,
    component: str,
    task: str,
    actor: str = "",
    note: str = "",
) -> Dict[str, object]:
    df = _ensure_df(reservations_csv_path)
    key = _task_key(task_id, component, task)
    mask = df.apply(
        lambda r: _task_key(r.get("task_id", ""), r.get("component", ""), r.get("task", "")) == key,
        axis=1,
    ) & df["state"].astype(str).str.upper().eq("ACTIVE")
    act = df[mask].copy()
    if act.empty:
        return {"released": 0}

    released = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, r in act.iterrows():
        part_name = _safe_str(r.get("part_name", ""))
        qty = float(pd.to_numeric(r.get("qty", 0.0), errors="coerce") or 0.0)
        if part_name and qty > 0:
            # Tools are non-consumable and were not decremented on reserve.
            if not is_non_consumable_part(inventory_csv_path, part_name):
                increment_part(inventory_csv_path, part_name, qty=qty, notes="Released maintenance reservation")
            released += 1
    df.loc[mask, "state"] = "RELEASED"
    df.loc[mask, "updated_ts"] = now
    if note:
        df.loc[mask, "note"] = _safe_str(note)
    if actor:
        df.loc[mask, "actor"] = _safe_str(actor)
    _save_df(reservations_csv_path, df)
    return {"released": int(released)}


def consume_task_reservations(
    *,
    reservations_csv_path: str,
    task_id: str,
    component: str,
    task: str,
    actor: str = "",
    note: str = "",
) -> Dict[str, object]:
    df = _ensure_df(reservations_csv_path)
    key = _task_key(task_id, component, task)
    mask = df.apply(
        lambda r: _task_key(r.get("task_id", ""), r.get("component", ""), r.get("task", "")) == key,
        axis=1,
    ) & df["state"].astype(str).str.upper().eq("ACTIVE")
    act = df[mask].copy()
    if act.empty:
        return {"consumed": 0}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[mask, "state"] = "CONSUMED"
    df.loc[mask, "updated_ts"] = now
    if note:
        df.loc[mask, "note"] = _safe_str(note)
    if actor:
        df.loc[mask, "actor"] = _safe_str(actor)
    _save_df(reservations_csv_path, df)
    return {"consumed": int(len(act))}


def has_active_reservation(
    path: str,
    *,
    task_id: str,
    component: str,
    task: str,
) -> bool:
    x = list_task_reservations(
        path,
        task_id=task_id,
        component=component,
        task=task,
        active_only=True,
    )
    return not x.empty
