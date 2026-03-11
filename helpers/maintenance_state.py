import os
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd


STATE_COLS = [
    "task_key",
    "task_id",
    "component",
    "task",
    "state",
    "updated_ts",
    "updated_by",
    "note",
]

ALLOWED_STATES = {
    "PLANNED",
    "PREP_NEEDED",
    "PREP_READY",
    "IN_PROGRESS",
    "DONE",
    "RESCHEDULED",
    "BLOCKED_PARTS",
}

TRANSITIONS = {
    "PLANNED": {"PREP_NEEDED", "PREP_READY", "IN_PROGRESS", "BLOCKED_PARTS", "RESCHEDULED"},
    "PREP_NEEDED": {"PREP_READY", "IN_PROGRESS", "BLOCKED_PARTS", "RESCHEDULED"},
    "PREP_READY": {"IN_PROGRESS", "BLOCKED_PARTS", "RESCHEDULED"},
    "IN_PROGRESS": {"DONE", "BLOCKED_PARTS", "RESCHEDULED"},
    "DONE": {"PLANNED", "PREP_NEEDED", "RESCHEDULED"},
    "RESCHEDULED": {"PLANNED", "PREP_NEEDED", "PREP_READY", "BLOCKED_PARTS"},
    "BLOCKED_PARTS": {"PREP_NEEDED", "PREP_READY", "IN_PROGRESS", "RESCHEDULED"},
}


def _safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _task_key_from_row(row) -> str:
    tid = _safe_str(row.get("Task_ID", "")).lower()
    comp = _safe_str(row.get("Component", "")).lower()
    task = _safe_str(row.get("Task", "")).lower()
    return f"{tid}|{comp}|{task}"


def _ensure_df(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except Exception:
            df = pd.DataFrame(columns=STATE_COLS)
    else:
        df = pd.DataFrame(columns=STATE_COLS)
    for c in STATE_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[STATE_COLS].copy()


def _write_df(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    for c in STATE_COLS:
        if c not in out.columns:
            out[c] = ""
    out[STATE_COLS].to_csv(path, index=False)


def normalize_state(v: str) -> str:
    s = _safe_str(v).upper()
    if s in ALLOWED_STATES:
        return s
    return ""


def default_state_for_status(status: str, parts_missing: bool = False, conditional_parts: bool = False) -> str:
    st = _safe_str(status).upper()
    if parts_missing and not conditional_parts:
        return "BLOCKED_PARTS"
    if st in {"OVERDUE", "DUE SOON", "ROUTINE"}:
        return "PREP_NEEDED"
    if st == "OK":
        return "PLANNED"
    return "PLANNED"


def merge_state_into_df(
    df: pd.DataFrame,
    state_csv_path: str,
    *,
    status_col: str = "Status",
    parts_missing_col: str = "",
    conditional_col: str = "",
) -> pd.DataFrame:
    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if isinstance(out, pd.DataFrame) and "Lifecycle_State" not in out.columns:
            out["Lifecycle_State"] = ""
        return out

    state_df = _ensure_df(state_csv_path)
    state_map: Dict[str, str] = {}
    for _, r in state_df.iterrows():
        k = _safe_str(r.get("task_key", ""))
        if not k:
            continue
        state_map[k] = normalize_state(r.get("state", ""))

    out = df.copy()
    vals = []
    for _, r in out.iterrows():
        key = _task_key_from_row(r)
        cur = normalize_state(state_map.get(key, ""))
        if not cur:
            parts_missing = bool(r.get(parts_missing_col, False)) if parts_missing_col else False
            conditional_parts = bool(r.get(conditional_col, False)) if conditional_col else False
            cur = default_state_for_status(r.get(status_col, ""), parts_missing=parts_missing, conditional_parts=conditional_parts)
        vals.append(cur)
    out["Lifecycle_State"] = vals
    return out


def set_task_state(
    state_csv_path: str,
    row,
    new_state: str,
    *,
    actor: str = "",
    note: str = "",
    force: bool = False,
) -> Tuple[bool, str]:
    ns = normalize_state(new_state)
    if not ns:
        return False, f"Invalid state: {new_state}"

    df = _ensure_df(state_csv_path)
    key = _task_key_from_row(row)
    if not key:
        return False, "Task key is empty"

    existing = df[df["task_key"].astype(str).str.strip().str.lower().eq(key.lower())]
    old_state = ""
    if not existing.empty:
        old_state = normalize_state(existing.iloc[0].get("state", ""))
    if not old_state:
        old_state = default_state_for_status(row.get("Status", ""))

    if (not force) and old_state in TRANSITIONS and ns not in TRANSITIONS.get(old_state, set()):
        return False, f"Transition not allowed: {old_state} -> {ns}"

    payload = {
        "task_key": key,
        "task_id": _safe_str(row.get("Task_ID", "")),
        "component": _safe_str(row.get("Component", "")),
        "task": _safe_str(row.get("Task", "")),
        "state": ns,
        "updated_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_by": _safe_str(actor),
        "note": _safe_str(note),
    }

    if not existing.empty:
        idx = existing.index[0]
        for k, v in payload.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    _write_df(state_csv_path, df)
    return True, ""


def set_tasks_state(
    state_csv_path: str,
    rows_df: pd.DataFrame,
    new_state: str,
    *,
    actor: str = "",
    note: str = "",
    force: bool = False,
) -> Tuple[int, int]:
    if rows_df is None or rows_df.empty:
        return 0, 0
    ok, fail = 0, 0
    for _, r in rows_df.iterrows():
        success, _ = set_task_state(
            state_csv_path,
            r,
            new_state,
            actor=actor,
            note=note,
            force=force,
        )
        if success:
            ok += 1
        else:
            fail += 1
    return ok, fail

