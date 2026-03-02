from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from app_io.path_health import build_path_health_report
from helpers.backup_io import create_backup_snapshot
from helpers.csv_schema import REQUIRED_CSV_COLUMNS, validate_csv_schema


def _fmt_ts(path: str) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


def _collect_configured_paths(P) -> dict:
    out = {}
    for key, val in sorted(getattr(P, "__dict__", {}).items()):
        if not isinstance(val, str):
            continue
        if key.endswith(("_dir", "_csv", "_json", "_path", "_image", "_log")):
            out[key] = val
    return out


@st.cache_data(show_spinner=False)
def _build_path_rows_and_summary(tracked_items: tuple, refresh_token: int):
    tracked = dict(tracked_items)
    report_obj = SimpleNamespace(**tracked)
    report = build_path_health_report(report_obj, critical_keys=list(tracked), legacy_aliases={})
    summary = report.get("summary", {})

    rows = []
    for name, path in tracked.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists and os.path.isfile(path) else 0
        rows.append(
            {
                "key": name,
                "exists": exists,
                "size_bytes": size,
                "modified": _fmt_ts(path) if exists else "missing",
                "path": path,
            }
        )
    return summary, pd.DataFrame(rows)


def _discover_operational_files(dirs: tuple, refresh_token: int) -> pd.DataFrame:
    allowed = {".csv", ".json", ".txt", ".log", ".duckdb", ".db", ".xlsx"}
    rows = []
    for group, d in dirs:
        if not os.path.isdir(d):
            continue
        for base, _, files in os.walk(d):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in allowed:
                    continue
                p = os.path.join(base, name)
                rows.append(
                    {
                        "group": group,
                        "file": name,
                        "ext": ext,
                        "size_bytes": os.path.getsize(p),
                        "modified": _fmt_ts(p),
                        "path": p,
                    }
                )
    rows.sort(key=lambda r: (r["group"], r["file"]))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _build_schema_rows(path_map_items: tuple, refresh_token: int) -> pd.DataFrame:
    path_map = dict(path_map_items)
    schema_rows = []
    for key, required_cols in REQUIRED_CSV_COLUMNS.items():
        p = path_map.get(key)
        if not p or not os.path.exists(p):
            schema_rows.append({"csv": key, "ok": False, "rows": 0, "missing_columns": "file missing"})
            continue
        result = validate_csv_schema(p, required_cols)
        schema_rows.append(
            {
                "csv": key,
                "ok": result.ok,
                "rows": result.row_count,
                "missing_columns": ", ".join(result.missing_columns) if result.missing_columns else "",
            }
        )
    return pd.DataFrame(schema_rows)


def render_data_diagnostics_tab(P) -> None:
    st.title("🩺 Data Diagnostics")
    st.caption("Read-only health view for paths and core data files.")

    root_fallback_db = os.path.abspath(os.path.join(P.root_dir, "data", "tower.duckdb"))
    active_db = os.path.abspath(P.duckdb_path)
    db_mode = "fallback" if active_db == root_fallback_db else "local"
    st.info(f"DuckDB mode: `{db_mode}`  |  path: `{active_db}`")
    st.session_state.setdefault("diag_refresh_token", 0)
    if st.button("Refresh diagnostics", key="refresh_diagnostics_btn", use_container_width=True):
        st.session_state["diag_refresh_token"] += 1
        st.cache_data.clear()
    refresh_token = int(st.session_state.get("diag_refresh_token", 0))

    st.subheader("Backup")
    b1, b2 = st.columns([1, 2])
    b1.code(P.backups_dir)
    snapshots = []
    if os.path.isdir(P.backups_dir):
        snapshots = sorted(
            [d for d in os.listdir(P.backups_dir) if os.path.isdir(os.path.join(P.backups_dir, d))],
            reverse=True,
        )
    b2.metric("Snapshots", len(snapshots))
    if st.button("Create Backup Snapshot", key="create_backup_snapshot_btn", use_container_width=True):
        with st.spinner("Creating backup snapshot..."):
            result = create_backup_snapshot(P)
        st.session_state["diag_refresh_token"] += 1
        st.cache_data.clear()
        if result.errors:
            st.warning(
                f"Backup completed with warnings: files={result.copied_files}, errors={len(result.errors)}\n"
                f"Snapshot: {result.snapshot_dir}"
            )
            with st.expander("Backup errors"):
                for e in result.errors:
                    st.text(e)
        else:
            st.success(f"Backup created: {result.snapshot_dir} (files={result.copied_files})")
    if snapshots:
        st.caption(f"Latest snapshot: {snapshots[0]}")

    st.subheader("Full App Tests")
    st.caption("Runs run_app_tests.py and shows the complete report output.")
    st.session_state.setdefault("diag_app_tests_output", "")
    st.session_state.setdefault("diag_app_tests_code", None)
    st.session_state.setdefault("diag_app_tests_ran_at", "")
    if st.button("Run Full App Tests", key="run_full_app_tests_btn", use_container_width=True):
        cmd = [sys.executable, os.path.join(P.root_dir, "run_app_tests.py")]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=P.root_dir)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        st.session_state["diag_app_tests_output"] = out.strip()
        st.session_state["diag_app_tests_code"] = int(proc.returncode)
        st.session_state["diag_app_tests_ran_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    last_code = st.session_state.get("diag_app_tests_code")
    if last_code is not None:
        ran_at = st.session_state.get("diag_app_tests_ran_at", "")
        if last_code == 0:
            st.success(f"Last run passed (exit code 0) at {ran_at}")
        else:
            st.error(f"Last run failed (exit code {last_code}) at {ran_at}")
        st.code(st.session_state.get("diag_app_tests_output", ""), language="text")

    tracked = _collect_configured_paths(P)
    tracked_items = tuple(tracked.items())
    summary, paths_df = _build_path_rows_and_summary(tracked_items, refresh_token)
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical issues", int(summary.get("critical_issues", 0)))
    c2.metric("Issues", int(summary.get("issues", 0)))
    c3.metric("Checked paths", len(tracked))

    st.subheader("Configured Paths (All from P)")
    st.dataframe(paths_df, use_container_width=True, hide_index=True)

    st.subheader("Discovered Data Files (CSV/JSON/LOG/DB)")
    dirs = (
        ("data", P.data_dir),
        ("logs", P.logs_dir),
        ("maintenance", P.maintenance_dir),
        ("parts", P.parts_dir),
        ("hooks", P.hooks_dir),
        ("reports", P.reports_dir),
        ("dataset", P.dataset_dir),
        ("config", P.config_dir),
        ("state", P.state_dir),
    )
    discovered = _discover_operational_files(dirs, refresh_token)
    if discovered.empty:
        st.info("No data files found in configured directories.")
    else:
        st.dataframe(discovered, use_container_width=True, hide_index=True)

    st.subheader("CSV Schemas")
    path_map = {
        "orders": P.orders_csv,
        "parts_orders": P.parts_orders_csv,
        "schedule": P.schedule_csv,
        "tower_temps": P.tower_temps_csv,
        "tower_containers": P.tower_containers_csv,
    }
    schema_df = _build_schema_rows(tuple(path_map.items()), refresh_token)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
