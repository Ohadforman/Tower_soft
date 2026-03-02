#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from helpers.csv_schema import REQUIRED_CSV_COLUMNS, validate_csv_schema
from app_io.paths import P


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def _is_under(path: str, base: str) -> bool:
    return os.path.abspath(path).startswith(os.path.abspath(base) + os.sep)


def check_folder_layout() -> None:
    checks = {
        "orders_csv": (P.orders_csv, os.path.join(P.root_dir, "data")),
        "parts_orders_csv": (P.parts_orders_csv, os.path.join(P.root_dir, "data")),
        "schedule_csv": (P.schedule_csv, os.path.join(P.root_dir, "data")),
        "tower_temps_csv": (P.tower_temps_csv, os.path.join(P.root_dir, "data")),
        "tower_containers_csv": (P.tower_containers_csv, os.path.join(P.root_dir, "data")),
        "coating_config_json": (P.coating_config_json, os.path.join(P.root_dir, "config")),
        "protocols_json": (P.protocols_json, os.path.join(P.root_dir, "config")),
        "home_bg_image": (P.home_bg_image, os.path.join(P.root_dir, "assets")),
    }
    bad = [k for k, (v, base) in checks.items() if not _is_under(v, base)]
    if bad:
        raise RuntimeError("wrong folder mapping: " + ", ".join(bad))


def check_required_files() -> None:
    required = [
        P.orders_csv,
        P.parts_orders_csv,
        P.schedule_csv,
        P.tower_temps_csv,
        P.tower_containers_csv,
        P.coating_config_json,
        P.protocols_json,
        P.dies_config_json,
        P.pid_config_json,
        P.home_bg_image,
        P.logo_image,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("missing: " + "; ".join(missing))


def check_backup_dir() -> None:
    os.makedirs(P.backups_dir, exist_ok=True)
    if not os.path.isdir(P.backups_dir):
        raise RuntimeError(f"backup dir missing: {P.backups_dir}")
    if not os.access(P.backups_dir, os.W_OK):
        raise RuntimeError(f"backup dir not writable: {P.backups_dir}")


def check_json_parse() -> None:
    for p in [P.coating_config_json, P.protocols_json, P.dies_config_json, P.pid_config_json]:
        with open(p, "r", encoding="utf-8") as f:
            json.load(f)


def check_duckdb_local_policy() -> None:
    db_path = os.path.abspath(P.duckdb_path)
    root_data_dir = os.path.abspath(os.path.join(P.root_dir, "data"))
    base = os.path.basename(db_path)
    if not (base == "tower.duckdb" or (base.startswith("tower_") and base.endswith(".duckdb"))):
        raise RuntimeError(f"unexpected db filename: {db_path}")
    # Allow either user-local path OR fallback under project data/.
    in_project_data = db_path.startswith(root_data_dir + os.sep)
    if db_path.startswith(os.path.abspath(P.root_dir) + os.sep) and not in_project_data:
        raise RuntimeError(f"duckdb should be user-local or fallback data path: {db_path}")
    parent = os.path.dirname(db_path) or "."
    if not os.path.isdir(parent):
        raise RuntimeError(f"duckdb parent dir missing: {parent}")
    if not os.access(parent, os.W_OK):
        raise RuntimeError(f"duckdb parent not writable: {parent}")


def check_csv_schemas() -> None:
    path_map = {
        "orders": P.orders_csv,
        "parts_orders": P.parts_orders_csv,
        "schedule": P.schedule_csv,
        "tower_temps": P.tower_temps_csv,
        "tower_containers": P.tower_containers_csv,
    }
    failures = []
    for key, required in REQUIRED_CSV_COLUMNS.items():
        result = validate_csv_schema(path_map[key], required)
        if not result.ok:
            failures.append(f"{key}: missing {result.missing_columns}")
    if failures:
        raise RuntimeError("; ".join(failures))


def run() -> int:
    started = time.time()
    checks = [
        ("Folder layout", check_folder_layout),
        ("Required files", check_required_files),
        ("Backup dir", check_backup_dir),
        ("JSON parse", check_json_parse),
        ("DuckDB local policy", check_duckdb_local_policy),
        ("CSV schemas", check_csv_schemas),
    ]

    results: list[CheckResult] = []
    for name, fn in checks:
        try:
            fn()
            results.append(CheckResult(name=name, ok=True))
        except Exception as e:
            results.append(CheckResult(name=name, ok=False, details=f"{type(e).__name__}: {e}"))

    failed = 0
    print("=== Preflight Check ===")
    for r in results:
        if r.ok:
            print(f"[PASS] {r.name}")
        else:
            failed += 1
            print(f"[FAIL] {r.name} -> {r.details}")

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.2f}s")

    if failed:
        print(f"Result: FAILED ({failed} checks)")
        return 1

    print("Result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
