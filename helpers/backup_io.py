from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from helpers.app_logger import log_event

INCLUDE_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".log",
    ".duckdb",
    ".db",
    ".xlsx",
    ".jpg",
    ".jpeg",
    ".png",
    ".pdf",
}


@dataclass
class BackupResult:
    snapshot_dir: str
    copied_files: int
    copied_bytes: int
    errors: List[str]
    manifest_path: str


@dataclass
class DiagnosticsBundleResult:
    bundle_path: str
    included_files: int
    total_bytes: int
    errors: List[str]


def _iter_files(root_dir: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root_dir):
        return out
    for base, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in INCLUDE_EXTENSIONS:
                out.append(os.path.join(base, name))
    return out


def _safe_rel(path: str, base: str) -> str:
    try:
        rel = os.path.relpath(path, base)
    except Exception:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")


def create_backup_snapshot(P) -> BackupResult:
    os.makedirs(P.backups_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(P.backups_dir, f"snapshot_{ts}")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Directories to scan for operational files.
    source_dirs: List[Tuple[str, str]] = [
        ("data", P.data_dir),
        ("config", P.config_dir),
        ("state", P.state_dir),
        ("logs", P.logs_dir),
        ("maintenance", P.maintenance_dir),
        ("parts", P.parts_dir),
        ("hooks", P.hooks_dir),
        ("reports", P.reports_dir),
        ("dataset", P.dataset_dir),
    ]

    copied_files = 0
    copied_bytes = 0
    errors: List[str] = []
    copied_index: Dict[str, str] = {}

    for tag, src_dir in source_dirs:
        for src in _iter_files(src_dir):
            rel = _safe_rel(src, src_dir)
            dst = os.path.join(snapshot_dir, tag, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
                copied_files += 1
                copied_bytes += os.path.getsize(dst)
                copied_index[src] = dst
            except Exception as e:
                errors.append(f"{src}: {type(e).__name__}: {e}")

    # If DB lives outside project root (per-user local), include it explicitly.
    if isinstance(P.duckdb_path, str) and os.path.isfile(P.duckdb_path):
        db_dst = os.path.join(snapshot_dir, "external", "duckdb", os.path.basename(P.duckdb_path))
        os.makedirs(os.path.dirname(db_dst), exist_ok=True)
        try:
            shutil.copy2(P.duckdb_path, db_dst)
            copied_files += 1
            copied_bytes += os.path.getsize(db_dst)
            copied_index[P.duckdb_path] = db_dst
        except Exception as e:
            errors.append(f"{P.duckdb_path}: {type(e).__name__}: {e}")

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_dir": snapshot_dir,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "errors": errors,
        "sources": source_dirs,
    }
    manifest_path = os.path.join(snapshot_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log_event(
        "backup_snapshot_created",
        snapshot_dir=snapshot_dir,
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        errors=len(errors),
    )

    return BackupResult(
        snapshot_dir=snapshot_dir,
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        errors=errors,
        manifest_path=manifest_path,
    )


def _latest_by_prefix(folder: str, prefix: str, suffixes: tuple[str, ...]) -> List[str]:
    if not os.path.isdir(folder):
        return []
    matches: List[str] = []
    for name in os.listdir(folder):
        if not name.startswith(prefix):
            continue
        low = name.lower()
        if not any(low.endswith(s) for s in suffixes):
            continue
        matches.append(os.path.join(folder, name))
    if not matches:
        return []
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    newest_ts = os.path.getmtime(matches[0])
    return [p for p in matches if os.path.getmtime(p) == newest_ts]


def create_diagnostics_bundle(P) -> DiagnosticsBundleResult:
    os.makedirs(P.backups_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_path = os.path.join(P.backups_dir, f"diagnostics_bundle_{ts}.zip")

    to_include: List[tuple[str, str]] = []
    errors: List[str] = []

    key_files = [
        P.coating_config_json,
        P.protocols_json,
        P.pid_config_json,
        P.dies_config_json,
        P.orders_csv,
        P.parts_orders_csv,
        P.schedule_csv,
        P.tower_temps_csv,
        P.tower_containers_csv,
        P.selected_csv_json,
    ]
    for p in key_files:
        if isinstance(p, str) and os.path.isfile(p):
            arc = os.path.join("snapshot", os.path.relpath(p, P.root_dir))
            to_include.append((p, arc))

    latest_checks = _latest_by_prefix(os.path.join(P.reports_dir, "checks"), "all_checks_", (".json",))
    for p in latest_checks:
        to_include.append((p, os.path.join("reports", "checks", os.path.basename(p))))

    latest_audits = _latest_by_prefix(
        os.path.join(P.reports_dir, "path_audit"), "path_permissions_", (".json", ".csv")
    )
    for p in latest_audits:
        to_include.append((p, os.path.join("reports", "path_audit", os.path.basename(p))))

    app_logs: List[str] = []
    if os.path.isdir(P.logs_dir):
        for name in os.listdir(P.logs_dir):
            if name.startswith("app_events_") and name.endswith(".log"):
                app_logs.append(os.path.join(P.logs_dir, name))
    app_logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for p in app_logs[:2]:
        to_include.append((p, os.path.join("logs", os.path.basename(p))))

    included_files = 0
    total_bytes = 0
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in to_include:
            try:
                zf.write(src, arc)
                included_files += 1
                total_bytes += os.path.getsize(src)
            except Exception as e:
                errors.append(f"{src}: {type(e).__name__}: {e}")

        manifest = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bundle_path": bundle_path,
            "included_files": included_files,
            "total_bytes": total_bytes,
            "items": [arc for _, arc in to_include],
            "errors": errors,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    return DiagnosticsBundleResult(
        bundle_path=bundle_path,
        included_files=included_files,
        total_bytes=total_bytes,
        errors=errors,
    )
