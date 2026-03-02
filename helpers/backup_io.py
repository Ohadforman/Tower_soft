from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple


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

    return BackupResult(
        snapshot_dir=snapshot_dir,
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        errors=errors,
        manifest_path=manifest_path,
    )
