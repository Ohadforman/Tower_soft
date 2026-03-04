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
    item_by_key = {it.get("key"): it for it in report.get("items", [])}

    rows = []
    for name, path in tracked.items():
        item = item_by_key.get(name, {})
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists and os.path.isfile(path) else 0
        readable = bool(item.get("readable", exists and os.access(path, os.R_OK)))
        writable = bool(item.get("writable", os.access(path if exists else (os.path.dirname(path) or "."), os.W_OK)))
        healthy = bool(item.get("healthy", exists and readable and writable))
        status = "✅ READY" if healthy else "❌ BLOCKED"
        rows.append(
            {
                "key": name,
                "status": status,
                "exists": exists,
                "readable": readable,
                "writable": writable,
                "healthy": healthy,
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
    st.markdown(
        """
        <style>
          .diag-top-spacer{ height: 8px; }
          .diag-title{
            font-size: 1.62rem;
            font-weight: 900;
            margin: 0;
            padding-top: 4px;
            line-height: 1.2;
            color: rgba(236,248,255,0.98);
            text-shadow: 0 0 14px rgba(86,178,255,0.22);
          }
          .diag-sub{
            margin: 4px 0 8px 0;
            font-size: 0.92rem;
            color: rgba(188,224,248,0.88);
          }
          .diag-line{
            height: 1px;
            margin: 0 0 12px 0;
            background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
          }
          .diag-section{
            margin-top: 8px;
            margin-bottom: 8px;
            padding-left: 8px;
            border-left: 3px solid rgba(120,200,255,0.62);
            font-size: 1.04rem;
            font-weight: 820;
            color: rgba(230,246,255,0.98);
          }
          .diag-perm-wrap{
            display:grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 10px 0;
          }
          .diag-perm-card{
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid rgba(128,206,255,0.26);
            background: linear-gradient(180deg, rgba(14,32,56,0.30), rgba(8,16,28,0.22));
          }
          .diag-perm-k{
            font-size: 0.80rem;
            color: rgba(188,224,248,0.90);
            margin-bottom: 2px;
          }
          .diag-perm-v{
            font-size: 1.15rem;
            font-weight: 860;
            color: rgba(230,246,255,0.98);
          }
          .diag-perm-good{
            border-color: rgba(118,236,160,0.44);
            box-shadow: 0 0 0 1px rgba(118,236,160,0.18);
          }
          .diag-perm-bad{
            border-color: rgba(255,120,120,0.48);
            box-shadow: 0 0 0 1px rgba(255,120,120,0.20);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="diag-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="diag-title">🩺 Data Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="diag-sub">Read-only health view for paths, backups, audits, and core data schemas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="diag-line"></div>', unsafe_allow_html=True)

    root_fallback_db = os.path.abspath(os.path.join(P.root_dir, "data", "tower.duckdb"))
    active_db = os.path.abspath(P.duckdb_path)
    db_mode = "fallback" if active_db == root_fallback_db else "local"
    st.info(f"DuckDB mode: `{db_mode}`  |  path: `{active_db}`")
    st.session_state.setdefault("diag_refresh_token", 0)
    if st.button("Refresh diagnostics", key="refresh_diagnostics_btn", use_container_width=True):
        st.session_state["diag_refresh_token"] += 1
        st.cache_data.clear()
    refresh_token = int(st.session_state.get("diag_refresh_token", 0))

    st.markdown('<div class="diag-section">Backup</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="diag-section">Full App Tests</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="diag-section">Path Permissions Audit</div>', unsafe_allow_html=True)
    st.caption("Deep read/write/list/parse access audit for all configured paths.")
    st.session_state.setdefault("diag_perm_audit_output", "")
    st.session_state.setdefault("diag_perm_audit_code", None)
    st.session_state.setdefault("diag_perm_audit_ran_at", "")
    if st.button("Run Path Permissions Audit", key="run_perm_audit_btn", use_container_width=True):
        cmd = [sys.executable, os.path.join(P.root_dir, "run_path_permissions_audit.py")]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=P.root_dir)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        st.session_state["diag_perm_audit_output"] = out.strip()
        st.session_state["diag_perm_audit_code"] = int(proc.returncode)
        st.session_state["diag_perm_audit_ran_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audit_code = st.session_state.get("diag_perm_audit_code")
    if audit_code is not None:
        ran_at = st.session_state.get("diag_perm_audit_ran_at", "")
        if audit_code == 0:
            st.success(f"Permissions audit passed (exit code 0) at {ran_at}")
        else:
            st.error(f"Permissions audit failed (exit code {audit_code}) at {ran_at}")
        st.code(st.session_state.get("diag_perm_audit_output", ""), language="text")

    tracked = _collect_configured_paths(P)
    tracked_items = tuple(tracked.items())
    summary, paths_df = _build_path_rows_and_summary(tracked_items, refresh_token)
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical issues", int(summary.get("critical_issues", 0)))
    c2.metric("Issues", int(summary.get("issues", 0)))
    c3.metric("Checked paths", len(tracked))

    healthy_count = int(paths_df["healthy"].sum()) if not paths_df.empty and "healthy" in paths_df.columns else 0
    blocked_count = int((~paths_df["healthy"]).sum()) if not paths_df.empty and "healthy" in paths_df.columns else 0
    readable_count = int(paths_df["readable"].sum()) if not paths_df.empty and "readable" in paths_df.columns else 0
    writable_count = int(paths_df["writable"].sum()) if not paths_df.empty and "writable" in paths_df.columns else 0
    total = len(paths_df)

    st.markdown(
        f"""
        <div class="diag-perm-wrap">
          <div class="diag-perm-card {'diag-perm-good' if blocked_count == 0 else 'diag-perm-bad'}">
            <div class="diag-perm-k">Path Access Status</div>
            <div class="diag-perm-v">{'✅ ALL READY' if blocked_count == 0 else f'❌ {blocked_count} BLOCKED'}</div>
          </div>
          <div class="diag-perm-card">
            <div class="diag-perm-k">Readable Paths</div>
            <div class="diag-perm-v">{readable_count}/{total}</div>
          </div>
          <div class="diag-perm-card">
            <div class="diag-perm-k">Writable Paths</div>
            <div class="diag-perm-v">{writable_count}/{total}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="diag-section">Configured Paths (All from P)</div>', unsafe_allow_html=True)
    show_cols = ["status", "key", "exists", "readable", "writable", "healthy", "modified", "path"]
    show_cols = [c for c in show_cols if c in paths_df.columns]
    if not paths_df.empty:
        st.dataframe(paths_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No configured paths found.")

    st.markdown('<div class="diag-section">Discovered Data Files (CSV/JSON/LOG/DB)</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="diag-section">CSV Schemas</div>', unsafe_allow_html=True)
    path_map = {
        "orders": P.orders_csv,
        "parts_orders": P.parts_orders_csv,
        "schedule": P.schedule_csv,
        "tower_temps": P.tower_temps_csv,
        "tower_containers": P.tower_containers_csv,
    }
    schema_df = _build_schema_rows(tuple(path_map.items()), refresh_token)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
