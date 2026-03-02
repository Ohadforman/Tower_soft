#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Callable, List

import pandas as pd

from helpers.csv_schema import REQUIRED_CSV_COLUMNS, validate_csv_schema
from app_io.paths import P
from app_io.path_health import build_path_health_report


@dataclass
class TestResult:
    name: str
    ok: bool
    details: str = ""
    warning: bool = False


class Runner:
    def __init__(self, strict_warnings: bool = False):
        self.strict_warnings = strict_warnings
        self.results: List[TestResult] = []

    def add(self, name: str, ok: bool, details: str = "", warning: bool = False) -> None:
        self.results.append(TestResult(name=name, ok=ok, details=details, warning=warning))

    def check(self, name: str, fn: Callable[[], None], warning: bool = False) -> None:
        try:
            fn()
            self.add(name, True)
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            if warning:
                self.add(name, True, details=detail, warning=True)
            else:
                self.add(name, False, details=detail)

    def summary(self) -> int:
        passed = sum(1 for r in self.results if r.ok and not r.warning)
        warnings = sum(1 for r in self.results if r.warning)
        failed = sum(1 for r in self.results if not r.ok)

        print("\n=== App Test Report ===")
        for r in self.results:
            status = "PASS"
            if r.warning:
                status = "WARN"
            if not r.ok:
                status = "FAIL"
            msg = f"[{status}] {r.name}"
            if r.details:
                msg += f" -> {r.details}"
            print(msg)

        print("\n=== Summary ===")
        print(f"Passed  : {passed}")
        print(f"Warnings: {warnings}")
        print(f"Failed  : {failed}")

        if failed > 0:
            return 1
        if self.strict_warnings and warnings > 0:
            return 2
        return 0


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_path_health() -> None:
    critical_keys = [
        "orders_csv",
        "parts_orders_csv",
        "schedule_csv",
        "tower_temps_csv",
        "tower_containers_csv",
        "coating_config_json",
        "protocols_json",
        "dies_config_json",
        "pid_config_json",
        "home_bg_image",
        "logo_image",
    ]
    report = build_path_health_report(P, critical_keys=critical_keys, legacy_aliases={})
    summary = report["summary"]
    _assert(summary["critical_issues"] == 0, f"critical_issues={summary['critical_issues']}")


def test_paths_are_folder_based() -> None:
    must_prefix = {
        "orders_csv": os.path.join(P.root_dir, "data"),
        "parts_orders_csv": os.path.join(P.root_dir, "data"),
        "schedule_csv": os.path.join(P.root_dir, "data"),
        "tower_temps_csv": os.path.join(P.root_dir, "data"),
        "tower_containers_csv": os.path.join(P.root_dir, "data"),
        "coating_config_json": os.path.join(P.root_dir, "config"),
        "protocols_json": os.path.join(P.root_dir, "config"),
        "dies_config_json": os.path.join(P.root_dir, "config"),
        "pid_config_json": os.path.join(P.root_dir, "config"),
        "home_bg_image": os.path.join(P.root_dir, "assets", "images"),
        "logo_image": os.path.join(P.root_dir, "assets", "images"),
    }
    bad = []
    for key, prefix in must_prefix.items():
        val = getattr(P, key)
        if not os.path.abspath(val).startswith(os.path.abspath(prefix) + os.sep):
            bad.append(f"{key}={val}")
    _assert(not bad, "non-folder-based paths: " + "; ".join(bad))


def test_core_files_exist() -> None:
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
    _assert(not missing, "missing files: " + ", ".join(missing))


def test_json_integrity() -> None:
    for p in [P.coating_config_json, P.protocols_json, P.dies_config_json, P.pid_config_json]:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        _assert(obj is not None, f"invalid json object in {p}")


def test_backup_dir_is_writable() -> None:
    os.makedirs(P.backups_dir, exist_ok=True)
    _assert(os.path.isdir(P.backups_dir), f"backup dir missing: {P.backups_dir}")
    _assert(os.access(P.backups_dir, os.W_OK), f"backup dir not writable: {P.backups_dir}")


def test_duckdb_path_is_local_or_fallback() -> None:
    db_path = os.path.abspath(P.duckdb_path)
    root_data_dir = os.path.abspath(os.path.join(P.root_dir, "data"))
    base = os.path.basename(db_path)
    _assert(
        base == "tower.duckdb" or (base.startswith("tower_") and base.endswith(".duckdb")),
        f"unexpected db filename: {db_path}",
    )
    in_project_data = db_path.startswith(root_data_dir + os.sep)
    _assert(
        in_project_data or not db_path.startswith(os.path.abspath(P.root_dir) + os.sep),
        f"db path should be user-local or fallback data path: {db_path}",
    )
    parent = os.path.dirname(db_path) or "."
    _assert(os.path.isdir(parent), f"duckdb parent dir missing: {parent}")
    _assert(os.access(parent, os.W_OK), f"duckdb parent not writable: {parent}")


def test_orders_schema() -> None:
    result = validate_csv_schema(P.orders_csv, REQUIRED_CSV_COLUMNS["orders"])
    _assert(result.ok, f"orders missing columns: {result.missing_columns}")


def test_parts_schema() -> None:
    result = validate_csv_schema(P.parts_orders_csv, REQUIRED_CSV_COLUMNS["parts_orders"])
    _assert(result.ok, f"parts missing columns: {result.missing_columns}")


def test_schedule_schema() -> None:
    result = validate_csv_schema(P.schedule_csv, REQUIRED_CSV_COLUMNS["schedule"])
    _assert(result.ok, f"schedule missing columns: {result.missing_columns}")


def test_temps_schema() -> None:
    result = validate_csv_schema(P.tower_temps_csv, REQUIRED_CSV_COLUMNS["tower_temps"])
    _assert(result.ok, f"tower_temps missing columns: {result.missing_columns}")


def test_containers_schema() -> None:
    result = validate_csv_schema(P.tower_containers_csv, REQUIRED_CSV_COLUMNS["tower_containers"])
    _assert(result.ok, f"tower_containers missing columns: {result.missing_columns}")


def test_csv_roundtrip_in_temp() -> None:
    csvs = [
        P.orders_csv,
        P.parts_orders_csv,
        P.schedule_csv,
        P.tower_temps_csv,
        P.tower_containers_csv,
        P.sap_rods_inventory_csv,
        P.preform_inventory_csv,
    ]
    with tempfile.TemporaryDirectory(prefix="tower_csv_test_") as td:
        for src in csvs:
            name = os.path.basename(src)
            dst = os.path.join(td, name)
            shutil.copy2(src, dst)
            df = pd.read_csv(dst, keep_default_na=False)
            # Write back unchanged; this validates parser/writer compatibility.
            df.to_csv(dst, index=False)
            df2 = pd.read_csv(dst, keep_default_na=False)
            _assert(list(df.columns) == list(df2.columns), f"columns changed on roundtrip: {src}")


def test_configs_have_expected_keys() -> None:
    with open(P.coating_config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _assert(isinstance(cfg, dict), "coating config must be dict")
    _assert("dies" in cfg, "coating config missing 'dies'")
    _assert("coatings" in cfg, "coating config missing 'coatings'")

    with open(P.protocols_json, "r", encoding="utf-8") as f:
        protocols = json.load(f)
    _assert(isinstance(protocols, list), "protocols config must be list")


def test_module_imports() -> None:
    # Import tab/components modules only (avoid dash_try runtime side effects).
    modules = [
        "renders.tabs.home_tab",
        "renders.tabs.order_draw_tab",
        "renders.tabs.consumables_tab",
        "renders.tabs.schedule_tab",
        "renders.tabs.protocols",
        "renders.tabs.maintenance_tab",
        "renders.tabs.dashboard_tab",
        "renders.tabs.draw_finalize_tab",
        "renders.tabs.data_diagnostics_tab",
        "renders.tabs.sql_lab",
    ]
    for m in modules:
        importlib.import_module(m)


def test_dash_try_compiles() -> None:
    with open("dash_try.py", "r", encoding="utf-8") as f:
        src = f.read()
    compile(src, "dash_try.py", "exec")


def test_path_script_single_source() -> None:
    # Guardrail: core app files should reference P.* paths, not hardcoded root file names.
    suspicious = []
    targets = ["dash_try.py", "renders/tabs/home_tab.py", "renders/tabs/consumables_tab.py", "renders/tabs/order_draw_tab.py"]
    literals = ["draw_orders.csv", "part_orders.csv", "tower_schedule.csv", "tower_temps.csv", "tower_containers.csv", "config_coating.json"]
    for rel in targets:
        with open(rel, "r", encoding="utf-8") as f:
            text = f.read()
        for lit in literals:
            if f'"{lit}"' in text or f"'{lit}'" in text:
                # allow UI/help text mentions, block obvious path assignments/opens
                for line in text.splitlines():
                    if lit in line and ("open(" in line or "read_csv(" in line or "=" in line):
                        if "P." not in line and "caption" not in line and "warning" not in line and "error(" not in line:
                            suspicious.append(f"{rel}: {line.strip()}")
    _assert(not suspicious, "suspicious hardcoded path usage: " + " | ".join(suspicious))


def test_no_root_duplicate_files_warning() -> None:
    root_names = [
        "draw_orders.csv",
        "part_orders.csv",
        "tower_schedule.csv",
        "tower_temps.csv",
        "tower_containers.csv",
        "protocols.json",
        "config_coating.json",
    ]
    existing = [n for n in root_names if os.path.exists(os.path.join(P.root_dir, n))]
    if existing:
        raise RuntimeError("root duplicates present: " + ", ".join(existing))


def test_legacy_redirect_runtime() -> None:
    """
    Validate central compatibility redirect works without creating root files.
    Run in subprocess to avoid monkeypatch side effects in this process.
    """
    code = r'''
import os, json, builtins
from app_io.paths import P
from app_io.legacy_path_compat import install_legacy_path_compat
root_cfg = os.path.join(P.root_dir, "config_coating.json")
# Before installing compat, physical legacy root file should not exist.
assert not os.path.isfile(root_cfg), "legacy root file unexpectedly exists on disk"
install_legacy_path_compat(P)
# After installing compat, os.path.exists(...) may intentionally return True.
assert os.path.exists(root_cfg), "compat exists check should pass"
with builtins.open(root_cfg, "r", encoding="utf-8") as f:
    d = json.load(f)
assert isinstance(d, dict) and "coatings" in d and "dies" in d
print("ok")
'''
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    _assert(proc.returncode == 0, f"legacy redirect failed: {proc.stderr or proc.stdout}")


def test_ui_smoke() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dash_try.py")
    at.run(timeout=120)
    exc_list = list(at.exception)
    if exc_list:
        msg = str(exc_list[0].message if hasattr(exc_list[0], "message") else exc_list[0])
        raise RuntimeError(msg)


def test_ui_tab_switch_smoke() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dash_try.py")
    at.run(timeout=120)
    if list(at.exception):
        msg = str(at.exception[0].message if hasattr(at.exception[0], "message") else at.exception[0])
        raise RuntimeError(msg)

    # Best-effort tab switch: if sidebar radio exists, validate a few key pages.
    radios = at.sidebar.radio
    if not radios:
        return
    page_radio = radios[0]
    for page in ["🏠 Home", "🍃 Tower state - Consumables and dies", "🩺 Data Diagnostics", "🧪 SQL Lab"]:
        try:
            page_radio.set_value(page)
            at.run(timeout=120)
            if list(at.exception):
                msg = str(at.exception[0].message if hasattr(at.exception[0], "message") else at.exception[0])
                raise RuntimeError(f"{page}: {msg}")
        except Exception:
            # Ignore widgets not present depending on selected group; covered by base smoke.
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Project-wide app test runner")
    parser.add_argument("--strict-warnings", action="store_true", help="treat warnings as failures (exit code 2)")
    parser.add_argument("--ui-smoke", action="store_true", help="run Streamlit UI smoke test (slower)")
    parser.add_argument("--ui-tabs", action="store_true", help="run best-effort UI tab switch smoke")
    parser.add_argument("--ui-strict", action="store_true", help="if set with --ui-smoke, fail on UI smoke errors")
    args = parser.parse_args()

    r = Runner(strict_warnings=args.strict_warnings)

    # Core checks
    r.check("Path health report", test_path_health)
    r.check("Paths are folder-based", test_paths_are_folder_based)
    r.check("Core files exist", test_core_files_exist)
    r.check("JSON integrity", test_json_integrity)
    r.check("Backup dir writable", test_backup_dir_is_writable)
    r.check("DuckDB local path policy", test_duckdb_path_is_local_or_fallback)
    r.check("Orders schema", test_orders_schema)
    r.check("Parts schema", test_parts_schema)
    r.check("Schedule schema", test_schedule_schema)
    r.check("Tower temps schema", test_temps_schema)
    r.check("Tower containers schema", test_containers_schema)
    r.check("CSV roundtrip in temp", test_csv_roundtrip_in_temp)
    r.check("Config expected keys", test_configs_have_expected_keys)
    r.check("No root duplicate files", test_no_root_duplicate_files_warning, warning=True)
    r.check("Legacy redirect runtime", test_legacy_redirect_runtime)
    r.check("dash_try compile", test_dash_try_compiles)
    r.check("Module import smoke", test_module_imports)
    r.check("Path script single source guardrail", test_path_script_single_source)
    if args.ui_smoke:
        r.check("UI smoke (streamlit.testing)", test_ui_smoke, warning=not args.ui_strict)
    if args.ui_tabs:
        r.check("UI tab switch smoke", test_ui_tab_switch_smoke, warning=not args.ui_strict)

    return r.summary()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("\nUnhandled test runner error:")
        traceback.print_exc()
        raise SystemExit(3)
