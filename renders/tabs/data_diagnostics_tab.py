from __future__ import annotations

import os
import json
import subprocess
import sys
import re
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from app_io.path_health import build_path_health_report
from helpers.backup_io import create_backup_snapshot, create_diagnostics_bundle
from helpers.csv_schema import REQUIRED_CSV_COLUMNS, validate_csv_schema
from helpers.error_registry import get_error_help


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

        # DuckDB file is often created lazily on first connection.
        # If parent directory is writable, treat missing file as ready.
        if name == "duckdb_path" and (not exists):
            parent = os.path.dirname(path) or "."
            parent_writable = os.path.isdir(parent) and os.access(parent, os.W_OK)
            if parent_writable:
                exists = True
                readable = True
                writable = True
                healthy = True

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


@st.cache_data(show_spinner=False)
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


def _render_error_guidance_from_outputs(*outputs: str) -> None:
    fail_code_re = re.compile(r"^\[FAIL\]\s+\[((?:PF|AT|EV)-\d{2})\]", flags=re.MULTILINE)
    detected: list[str] = []
    for out in outputs:
        if not out:
            continue
        detected.extend(fail_code_re.findall(out))

    # Preserve order and uniqueness.
    seen = set()
    codes = [c for c in detected if not (c in seen or seen.add(c))]
    if not codes:
        return

    st.markdown('<div class="diag-section">Error Code Guidance</div>', unsafe_allow_html=True)
    st.caption("Detected failing codes with quick fix instructions.")
    rows = [get_error_help(c) for c in codes]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    unique_docs = sorted({row.get("doc", "") for row in rows if row.get("doc", "")})
    if unique_docs:
        st.caption("Useful docs")
        st.markdown(" | ".join([f"`{d}`" for d in unique_docs]))


def _extract_missing_imports_from_env_output(text: str) -> list[str]:
    if not text:
        return []
    # Example line:
    # [FAIL] [EV-04] Required package imports -> RuntimeError: missing imports: plotly, duckdb, reportlab, PyPDF2
    m = re.search(r"missing imports:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1)
    names = [x.strip() for x in raw.split(",") if x.strip()]
    # preserve order and uniqueness
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _safe_auto_repair(P) -> list[str]:
    actions: list[str] = []
    dir_keys = [
        "data_dir",
        "config_dir",
        "state_dir",
        "assets_dir",
        "images_dir",
        "dataset_dir",
        "logs_dir",
        "reports_dir",
        "backups_dir",
        "maintenance_dir",
        "hooks_dir",
        "parts_dir",
    ]
    for key in dir_keys:
        d = getattr(P, key, "")
        if isinstance(d, str) and d:
            os.makedirs(d, exist_ok=True)
            actions.append(f"ensured dir: {key}")

    defaults = [
        (P.selected_csv_json, {"selected_csv": ""}),
        (P.container_levels_prev_json, {}),
        (P.coating_stock_json, {}),
    ]
    for path, payload in defaults:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            actions.append(f"created file: {os.path.basename(path)}")
    return actions


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
          .diag-status-card{
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid rgba(128,206,255,0.22);
            background: linear-gradient(180deg, rgba(14,32,56,0.26), rgba(8,16,28,0.20));
            min-height: 82px;
          }
          .diag-status-k{
            font-size: 0.80rem;
            color: rgba(188,224,248,0.90);
            margin-bottom: 4px;
          }
          .diag-status-v{
            font-size: 1.55rem;
            font-weight: 880;
            line-height: 1.1;
          }
          .diag-ok{ color: rgba(118,236,160,0.96); text-shadow: 0 0 12px rgba(118,236,160,0.28); }
          .diag-bad{ color: rgba(255,120,120,0.96); text-shadow: 0 0 12px rgba(255,120,120,0.26); }
          .diag-na{ color: rgba(186,206,224,0.92); }
          div[data-testid="stButton"] > button{
            border-radius: 12px !important;
            border: 1px solid rgba(138,214,255,0.58) !important;
            background: linear-gradient(180deg, rgba(28,74,120,0.72), rgba(12,36,68,0.66)) !important;
            color: rgba(236,248,255,0.98) !important;
            box-shadow: 0 8px 18px rgba(8,30,58,0.32), 0 0 12px rgba(74,170,255,0.18) !important;
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease !important;
          }
          div[data-testid="stButton"] > button:hover{
            transform: translateY(-1px);
            border-color: rgba(188,238,255,0.86) !important;
            box-shadow: 0 12px 24px rgba(8,30,58,0.36), 0 0 16px rgba(96,194,255,0.30) !important;
          }
          div[data-testid="stButton"] > button[kind="primary"]{
            border-color: rgba(170,232,255,0.84) !important;
            background: linear-gradient(180deg, rgba(76,168,255,0.90), rgba(32,98,172,0.88)) !important;
            box-shadow: 0 14px 24px rgba(12, 68, 124, 0.40), 0 0 18px rgba(96,194,255,0.34) !important;
          }
          div[data-baseweb="tag"],
          span[data-baseweb="tag"]{
            background: linear-gradient(180deg, rgba(70,160,238,0.94), rgba(32,96,168,0.92)) !important;
            border: 1px solid rgba(170,232,255,0.82) !important;
            color: rgba(244,252,255,0.99) !important;
          }
          div[data-baseweb="tag"] *,
          span[data-baseweb="tag"] *{
            color: rgba(244,252,255,0.99) !important;
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
    refresh_token = int(st.session_state.get("diag_refresh_token", 0))

    st.markdown('<div class="diag-section">Control Panel</div>', unsafe_allow_html=True)
    st.caption("Run all validation checks from one button, then use maintenance actions as needed.")

    def _run_and_store(prefix: str, script_name: str) -> None:
        cmd = [sys.executable, os.path.join(P.root_dir, "scripts", "cli", script_name)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=P.root_dir)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        st.session_state[f"{prefix}_output"] = out.strip()
        st.session_state[f"{prefix}_code"] = int(proc.returncode)
        st.session_state[f"{prefix}_ran_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _sync_statuses_from_full_health_output(out_text: str, ran_at: str) -> None:
        if not out_text:
            return
        mapping = {
            "App tests": "diag_app_tests",
            "Path permissions audit": "diag_perm_audit",
            "Environment pretest": "diag_env_pretest",
            "Release check": "diag_release_check",
        }
        for line in out_text.splitlines():
            s = line.strip()
            for label, prefix in mapping.items():
                if s.endswith(label) and s.startswith("[PASS]"):
                    st.session_state[f"{prefix}_code"] = 0
                    st.session_state[f"{prefix}_ran_at"] = ran_at
                elif s.endswith(label) and s.startswith("[FAIL]"):
                    st.session_state[f"{prefix}_code"] = 1
                    st.session_state[f"{prefix}_ran_at"] = ran_at

    if st.button("▶ Run All (Full Health)", key="run_all_full_health_top_btn", use_container_width=True):
        _run_and_store("diag_full_health", "run_full_health_check.py")
        _sync_statuses_from_full_health_output(
            st.session_state.get("diag_full_health_output", ""),
            st.session_state.get("diag_full_health_ran_at", ""),
        )
    st.caption("Run All (Full Health): runs Preflight + App Tests + Path Audit + Env Pretest + Release Check.")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Refresh", key="refresh_diagnostics_btn", use_container_width=True):
        st.session_state["diag_refresh_token"] += 1
        st.cache_data.clear()
    if c2.button("Safe Auto Repair", key="run_safe_auto_repair_btn", use_container_width=True):
        actions = _safe_auto_repair(P)
        st.session_state["diag_auto_repair_actions"] = actions
        st.session_state["diag_refresh_token"] += 1
        st.cache_data.clear()
    if c3.button("Create Backup", key="create_backup_snapshot_btn", use_container_width=True):
        with st.spinner("Creating backup snapshot..."):
            result = create_backup_snapshot(P)
        st.session_state["diag_refresh_token"] += 1
        st.cache_data.clear()
        st.session_state["diag_backup_result"] = {
            "snapshot_dir": result.snapshot_dir,
            "copied_files": result.copied_files,
            "errors": result.errors,
        }
    if c4.button("Export Bundle", key="export_diag_bundle_btn", use_container_width=True):
        with st.spinner("Building diagnostics bundle..."):
            bundle = create_diagnostics_bundle(P)
        st.session_state["diag_bundle_path"] = bundle.bundle_path
        st.session_state["diag_bundle_errors"] = bundle.errors
        st.session_state["diag_bundle_stats"] = (bundle.included_files, bundle.total_bytes)
    d1, d2, d3, d4 = st.columns(4)
    d1.caption("Reload diagnostics data and clear Streamlit cache.")
    d2.caption("Create missing dirs/files safely (no destructive changes).")
    d3.caption("Create a full snapshot under backups for rollback/debug.")
    d4.caption("Export diagnostics bundle (reports + checks + key artifacts).")

    # Keep state keys initialized
    st.session_state.setdefault("diag_app_tests_output", "")
    st.session_state.setdefault("diag_app_tests_code", None)
    st.session_state.setdefault("diag_app_tests_ran_at", "")
    st.session_state.setdefault("diag_perm_audit_output", "")
    st.session_state.setdefault("diag_perm_audit_code", None)
    st.session_state.setdefault("diag_perm_audit_ran_at", "")
    st.session_state.setdefault("diag_env_pretest_output", "")
    st.session_state.setdefault("diag_env_pretest_code", None)
    st.session_state.setdefault("diag_env_pretest_ran_at", "")
    st.session_state.setdefault("diag_release_check_output", "")
    st.session_state.setdefault("diag_release_check_code", None)
    st.session_state.setdefault("diag_release_check_ran_at", "")
    st.session_state.setdefault("diag_full_health_output", "")
    st.session_state.setdefault("diag_full_health_code", None)
    st.session_state.setdefault("diag_full_health_ran_at", "")

    # Compact status row (green/red)
    def _status_html(label: str, code):
        if code is None:
            cls, txt = "diag-na", "N/A"
        elif int(code) == 0:
            cls, txt = "diag-ok", "OK"
        else:
            cls, txt = "diag-bad", "FAIL"
        return (
            f'<div class="diag-status-card">'
            f'<div class="diag-status-k">{label}</div>'
            f'<div class="diag-status-v {cls}">{txt}</div>'
            f"</div>"
        )

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.markdown(_status_html("App Tests", st.session_state.get("diag_app_tests_code")), unsafe_allow_html=True)
    s2.markdown(_status_html("Path Audit", st.session_state.get("diag_perm_audit_code")), unsafe_allow_html=True)
    s3.markdown(_status_html("Env", st.session_state.get("diag_env_pretest_code")), unsafe_allow_html=True)
    s4.markdown(_status_html("Release", st.session_state.get("diag_release_check_code")), unsafe_allow_html=True)
    s5.markdown(_status_html("Full Health", st.session_state.get("diag_full_health_code")), unsafe_allow_html=True)

    # Backup and bundle quick statuses
    snapshots = []
    if os.path.isdir(P.backups_dir):
        snapshots = sorted(
            [d for d in os.listdir(P.backups_dir) if os.path.isdir(os.path.join(P.backups_dir, d))],
            reverse=True,
        )
    st.caption(f"Backups dir: `{P.backups_dir}` | Snapshots: {len(snapshots)}")

    actions = st.session_state.get("diag_auto_repair_actions", [])
    if actions:
        st.success(f"Auto repair completed ({len(actions)} actions).")
        with st.expander("Auto repair actions"):
            for a in actions:
                st.text(a)

    backup_result = st.session_state.get("diag_backup_result")
    if backup_result:
        if backup_result.get("errors"):
            st.warning(
                f"Backup warnings: files={backup_result.get('copied_files', 0)}, "
                f"errors={len(backup_result.get('errors', []))}"
            )
            with st.expander("Backup errors"):
                for e in backup_result.get("errors", []):
                    st.text(e)
        else:
            st.success(
                f"Backup created: {backup_result.get('snapshot_dir', '')} "
                f"(files={backup_result.get('copied_files', 0)})"
            )

    bundle_path = st.session_state.get("diag_bundle_path", "")
    if bundle_path and os.path.isfile(bundle_path):
        files_count, total_bytes = st.session_state.get("diag_bundle_stats", (0, 0))
        st.success(f"Bundle ready: {os.path.basename(bundle_path)} (files={files_count}, bytes={total_bytes})")
        with open(bundle_path, "rb") as f:
            data = f.read()
        st.download_button(
            "Download Diagnostics Bundle",
            data=data,
            file_name=os.path.basename(bundle_path),
            mime="application/zip",
            use_container_width=True,
            key="download_diag_bundle_btn",
        )
        errs = st.session_state.get("diag_bundle_errors", [])
        if errs:
            with st.expander("Bundle warnings"):
                for e in errs:
                    st.text(e)

    with st.expander("Run Outputs", expanded=False):
        for name, code_key, at_key, out_key in [
            ("App Tests", "diag_app_tests_code", "diag_app_tests_ran_at", "diag_app_tests_output"),
            ("Path Permissions Audit", "diag_perm_audit_code", "diag_perm_audit_ran_at", "diag_perm_audit_output"),
            ("Environment Pretest", "diag_env_pretest_code", "diag_env_pretest_ran_at", "diag_env_pretest_output"),
            ("Release Check", "diag_release_check_code", "diag_release_check_ran_at", "diag_release_check_output"),
            ("Full Health Check", "diag_full_health_code", "diag_full_health_ran_at", "diag_full_health_output"),
        ]:
            code = st.session_state.get(code_key)
            if code is None:
                continue
            ran_at = st.session_state.get(at_key, "")
            if code == 0:
                st.success(f"{name}: PASS (exit 0) at {ran_at}")
            else:
                st.error(f"{name}: FAIL (exit {code}) at {ran_at}")
            st.code(st.session_state.get(out_key, ""), language="text")

    with st.expander("EV-04 Package Auto Fix", expanded=False):
        st.caption("If EV-04 failed, install missing packages for the active Python environment.")
        env_out = st.session_state.get("diag_env_pretest_output", "")
        missing_imports = _extract_missing_imports_from_env_output(env_out)
        if missing_imports:
            st.warning(f"Detected missing imports: {', '.join(missing_imports)}")
            install_cmd = [sys.executable, "-m", "pip", "install", *missing_imports]
            st.code(" ".join(install_cmd), language="bash")
            if st.button("Install Missing EV-04 Packages", key="install_ev04_packages_btn", use_container_width=True):
                proc = subprocess.run(install_cmd, capture_output=True, text=True, cwd=P.root_dir)
                out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
                st.session_state["diag_ev04_install_code"] = int(proc.returncode)
                st.session_state["diag_ev04_install_output"] = out.strip()
                st.session_state["diag_ev04_install_ran_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.info("No missing package list detected from latest env pretest output.")

        ev04_install_code = st.session_state.get("diag_ev04_install_code")
        if ev04_install_code is not None:
            ran_at = st.session_state.get("diag_ev04_install_ran_at", "")
            if ev04_install_code == 0:
                st.success(f"Package install completed (exit code 0) at {ran_at}. Re-run Environment Pretest.")
            else:
                st.error(f"Package install failed (exit code {ev04_install_code}) at {ran_at}")
            st.code(st.session_state.get("diag_ev04_install_output", ""), language="text")

    _render_error_guidance_from_outputs(
        st.session_state.get("diag_app_tests_output", ""),
        st.session_state.get("diag_perm_audit_output", ""),
        st.session_state.get("diag_release_check_output", ""),
        st.session_state.get("diag_env_pretest_output", ""),
        st.session_state.get("diag_full_health_output", ""),
    )

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

    with st.expander("Configured Paths (All from P)", expanded=False):
        show_cols = ["status", "key", "exists", "readable", "writable", "healthy", "modified", "path"]
        show_cols = [c for c in show_cols if c in paths_df.columns]
        if not paths_df.empty:
            st.dataframe(paths_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No configured paths found.")

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
    with st.expander("Discovered Data Files (CSV/JSON/LOG/DB)", expanded=False):
        discovered = _discover_operational_files(dirs, refresh_token)
        if discovered.empty:
            st.info("No data files found in configured directories.")
        else:
            st.dataframe(discovered, use_container_width=True, hide_index=True)

    path_map = {
        "orders": P.orders_csv,
        "parts_orders": P.parts_orders_csv,
        "schedule": P.schedule_csv,
        "tower_temps": P.tower_temps_csv,
        "tower_containers": P.tower_containers_csv,
    }
    with st.expander("CSV Schemas", expanded=False):
        schema_df = _build_schema_rows(tuple(path_map.items()), refresh_token)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
