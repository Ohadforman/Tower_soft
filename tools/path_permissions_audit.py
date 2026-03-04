#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_path_key(key: str) -> bool:
    return key.endswith(("_dir", "_csv", "_json", "_path", "_image", "_log"))


def _kind_from_key(key: str, value: str) -> str:
    if key.endswith("_dir"):
        return "dir"
    if os.path.isdir(value):
        return "dir"
    return "file"


def _probe_dir_write(path: str) -> Tuple[bool, str]:
    probe = os.path.join(path, ".perm_probe_tmp")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _probe_parent_write(path: str) -> Tuple[bool, str]:
    parent = os.path.dirname(path) or "."
    probe = os.path.join(parent, ".perm_probe_tmp")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _parse_file(path: str) -> Tuple[bool, str]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            return True, "json_ok"
        if ext == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            if not header:
                return False, "empty_csv_header"
            return True, f"csv_cols={len(header.split(','))}"
        with open(path, "rb") as f:
            f.read(64)
        return True, "read_ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def build_permission_report(P, critical_keys: List[str]) -> Dict:
    items = []
    critical = set(critical_keys)
    for key, value in sorted(getattr(P, "__dict__", {}).items()):
        if not isinstance(value, str) or not _is_path_key(key):
            continue

        started = time.perf_counter()
        kind = _kind_from_key(key, value)
        exists = os.path.exists(value)
        readable = exists and os.access(value, os.R_OK)
        writable = False
        listable = False
        parse_ok = None
        parse_info = ""
        write_probe_ok = False
        write_probe_info = ""

        if kind == "dir":
            if exists:
                writable = os.access(value, os.W_OK)
                try:
                    _ = os.listdir(value)
                    listable = True
                except Exception:
                    listable = False
                write_probe_ok, write_probe_info = _probe_dir_write(value) if writable else (False, "dir_not_writable")
            else:
                parent = os.path.dirname(value) or "."
                writable = os.path.isdir(parent) and os.access(parent, os.W_OK)
                write_probe_ok, write_probe_info = _probe_parent_write(value) if writable else (False, "parent_not_writable")
        else:
            if exists:
                writable = os.access(value, os.W_OK)
                parse_ok, parse_info = _parse_file(value)
                write_probe_ok, write_probe_info = _probe_parent_write(value)
            else:
                parent = os.path.dirname(value) or "."
                writable = os.path.isdir(parent) and os.access(parent, os.W_OK)
                write_probe_ok, write_probe_info = _probe_parent_write(value) if writable else (False, "parent_not_writable")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if kind == "file":
            # For generated/runtime files (e.g., duckdb), "missing but writable parent" is still healthy.
            healthy = (exists and readable and write_probe_ok) or ((not exists) and write_probe_ok)
        else:
            healthy = ((exists and listable and write_probe_ok) or write_probe_ok)
        items.append(
            {
                "key": key,
                "path": value,
                "kind": kind,
                "critical": key in critical,
                "exists": exists,
                "readable": readable,
                "writable": writable,
                "listable": listable,
                "parse_ok": parse_ok,
                "parse_info": parse_info,
                "write_probe_ok": write_probe_ok,
                "write_probe_info": write_probe_info,
                "healthy": healthy,
                "elapsed_ms": elapsed_ms,
            }
        )

    issues = [x for x in items if not x["healthy"]]
    critical_issues = [x for x in issues if x["critical"]]
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_paths": len(items),
        "healthy_paths": len(items) - len(issues),
        "issues": len(issues),
        "critical_issues": len(critical_issues),
        "total_check_ms": int(sum(x["elapsed_ms"] for x in items)),
    }
    return {"summary": summary, "items": items, "issues": issues, "critical_issues": critical_issues}


def write_report_files(report: Dict, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"path_permissions_{ts}.json")
    csv_path = os.path.join(out_dir, f"path_permissions_{ts}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = report.get("items", [])
    fieldnames = [
        "key",
        "kind",
        "critical",
        "exists",
        "readable",
        "writable",
        "listable",
        "parse_ok",
        "parse_info",
        "write_probe_ok",
        "write_probe_info",
        "healthy",
        "elapsed_ms",
        "path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Path permissions and accessibility audit")
    parser.add_argument("--out-dir", default="", help="Output folder for reports (default: <reports_dir>/path_audit)")
    args = parser.parse_args()

    from app_io.paths import P
    from helpers.app_logger import log_event

    critical = [
        "orders_csv",
        "parts_orders_csv",
        "schedule_csv",
        "tower_temps_csv",
        "tower_containers_csv",
        "coating_config_json",
        "protocols_json",
        "dies_config_json",
        "pid_config_json",
        "duckdb_path",
        "backups_dir",
        "logs_dir",
        "dataset_dir",
    ]

    report = build_permission_report(P, critical_keys=critical)
    out_dir = args.out_dir.strip() or os.path.join(P.reports_dir, "path_audit")
    json_path, csv_path = write_report_files(report, out_dir)

    summary = report["summary"]
    print("=== Path Permissions Audit ===")
    print(json.dumps(summary, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report : {csv_path}")

    issues = report.get("critical_issues", [])
    if issues:
        print("\nCritical issues:")
        for x in issues:
            print(f"- {x['key']}: {x.get('write_probe_info') or x.get('parse_info') or 'not healthy'} ({x['path']})")
        log_event("path_permissions_audit", ok=False, critical_issues=len(issues), json_report=json_path, csv_report=csv_path)
        return 2
    log_event("path_permissions_audit", ok=True, critical_issues=0, json_report=json_path, csv_report=csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
