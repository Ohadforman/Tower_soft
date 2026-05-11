from __future__ import annotations

import csv
import json
import mimetypes
import calendar
import os
import math
import re
import sys
import sqlite3
import base64
import subprocess
import hashlib
import threading
import shutil
from html import escape as escape_html
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd
try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR_ENV = str(os.environ.get("TOWER_REBUILD_ROOT_DIR", "") or "").strip()
DATA_DIR_ENV = str(os.environ.get("TOWER_REBUILD_DATA_DIR", "") or "").strip()
ROOT_DIR = Path(ROOT_DIR_ENV).expanduser().resolve() if ROOT_DIR_ENV else BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve() if DATA_DIR_ENV else ROOT_DIR / "data"

DRAW_ORDERS = DATA_DIR / "draw_orders.csv"
TOWER_SCHEDULE = DATA_DIR / "tower_schedule.csv"
PART_ORDERS = DATA_DIR / "part_orders.csv"
PARTS_INVENTORY = DATA_DIR / "parts_inventory.csv"
PARTS_LOCATIONS = DATA_DIR / "parts_locations.csv"
PARTS_COMPANIES = DATA_DIR / "parts_companies.csv"
PROJECTS_FIBER = DATA_DIR / "projects_fiber.csv"
PROJECTS_TEMPLATES = DATA_DIR / "projects_fiber_templates.csv"
SAP_RODS_INVENTORY = DATA_DIR / "sap_rods_inventory.csv"
SELECTED_CSV_JSON = DATA_DIR / "selected_csv.json"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.maintenance_status import compute_maintenance_status_df, load_maintenance_folder_df

MAINTENANCE_DIR = ROOT_DIR / "maintenance"
COATING_CONFIG = ROOT_DIR / "config" / "config_coating.json"
PID_CONFIG = ROOT_DIR / "config" / "pid_config.json"
CONTAINER_CONFIG = ROOT_DIR / "config" / "container_config.json"
DIES_CONFIG = ROOT_DIR / "config" / "dies_6station.json"
HEATER_CONFIG = ROOT_DIR / "config" / "heater_config.json"
DATASET_DIR = ROOT_DIR / "data_set_csv"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
REPORT_CENTER_DIR = REPORTS_DIR / "report_center"
DEVELOPMENT_MEDIA_DIR = ROOT_DIR / "development_media"
BACKUPS_DIR = ROOT_DIR / "backups"
DUCKDB_PATH = DATA_DIR / "tower.duckdb"
STATE_DIR = ROOT_DIR / "state"
DASHBOARD_EXPORTS_DIR = STATE_DIR / "dashboard_exports"
DOWNLOADS_DIR = Path.home() / "Downloads"
COATING_STOCK = STATE_DIR / "coating_type_stock.json"
CONTAINER_SNAPSHOT = STATE_DIR / "container_levels_prev.json"
MAINTENANCE_STATE = MAINTENANCE_DIR / "maintenance_task_state.csv"
MAINTENANCE_ACTIONS = MAINTENANCE_DIR / "maintenance_actions_log.csv"
MAINTENANCE_WAITS = MAINTENANCE_DIR / "maintenance_wait_parts_log.csv"
MAINTENANCE_RUNTIME = MAINTENANCE_DIR / "_app_state.json"
MAINTENANCE_PACKAGE_PHOTOS_DIR = STATIC_DIR / "uploads" / "maintenance_packages"
TOWER_TEMPS = DATA_DIR / "tower_temps.csv"
TOWER_CONTAINERS = DATA_DIR / "tower_containers.csv"
PREFORM_INVENTORY = DATA_DIR / "preform_inventory.csv"
FAULTS_LOG = MAINTENANCE_DIR / "faults_log.csv"
FAULTS_ACTIONS_LOG = MAINTENANCE_DIR / "faults_actions_log.csv"
ARGON_MONTHLY_REPORT = REPORTS_DIR / "gas" / "argon_monthly_report.csv"
MANUALS_DIR = ROOT_DIR / "manuals"
EXTERNAL_MANUALS_DIR = ROOT_DIR.parent / "manuals"
MANUAL_INDEX_SCRIPT = BASE_DIR / "tools" / "extract_manual_index.py"
MANUAL_PAGE_RENDER_SCRIPT = BASE_DIR / "tools" / "render_manual_page.swift"
MANUAL_PAGE_CACHE_DIR = STATE_DIR / "manual_page_cache"
BUNDLED_RUNTIME_PYTHON = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "bin" / "python3"
HELPER_PYTHON_ENV = str(os.environ.get("TOWER_REBUILD_HELPER_PYTHON", "") or "").strip()
MANUAL_PAGE_MODE_ENV = str(os.environ.get("TOWER_REBUILD_MANUAL_PAGE_MODE", "") or "").strip().lower()
DEFAULT_BIND_HOST = str(os.environ.get("TOWER_REBUILD_HOST", "") or "").strip() or "127.0.0.1"
try:
    DEFAULT_BIND_PORT = int(str(os.environ.get("TOWER_REBUILD_PORT", "") or "").strip() or "8010")
except ValueError:
    DEFAULT_BIND_PORT = 8010

PROJECTS_COL = "Fiber Project"
GEOMETRY_COL = "Fiber Geometry Type"
GOOD_ZONES_COL = "Good Zones Count (required length zones)"
LENGTH_COL = "Required Length (m) (for T&M+costumer)"
MAIN_TEMP_COL = "Main Coating Temperature (°C)"
SECONDARY_TEMP_COL = "Secondary Coating Temperature (°C)"
FIBER_TOL_COL = "Fiber Diameter Tol (± µm)"
MAIN_TOL_COL = "Main Coating Diameter Tol (± µm)"
SECONDARY_TOL_COL = "Secondary Coating Diameter Tol (± µm)"
TIGER_CUT_COL = "Tiger Cut (%)"
OCT_F2F_COL = "Octagonal F2F (mm)"
SCHEDULE_PASSWORD = "DORON"
SCHEDULE_REQUIRED_COLS = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
ORDER_DRAW_GEOMETRY_OPTIONS = [
    "",
    "PANDA - PM",
    "TIGER - PM",
    "Octagonal",
    "ROUND",
    "STEP INDEX",
    "Ring Core",
    "Hollow Core",
    "Photonic Crystal",
    "Custom (write in Notes)",
]
TEMPLATE_FIELDS = [
    PROJECTS_COL,
    GEOMETRY_COL,
    TIGER_CUT_COL,
    OCT_F2F_COL,
    "Fiber Diameter (µm)",
    FIBER_TOL_COL,
    "Main Coating Diameter (µm)",
    MAIN_TOL_COL,
    "Secondary Coating Diameter (µm)",
    SECONDARY_TOL_COL,
    "Tension (g)",
    "Draw Speed (m/min)",
    "Main Coating",
    "Secondary Coating",
    MAIN_TEMP_COL,
    SECONDARY_TEMP_COL,
    "Notes Default",
]
PART_STATUS_ORDER = [
    "Opened",
    "Wait for Approval",
    "Approved",
    "Ordered",
    "Received",
    "Archived",
]
REPORT_CENTER_SECTIONS = [
    "Executive Summary",
    "Resources: Gas + SAP + Preforms",
    "Draw Outcomes (Done/Failed + Notes)",
    "Parts Orders Status",
    "Schedule: Past Week + Next Week",
    "Maintenance + Faults",
    "Maintenance Tests + Measurements",
    "Consumables Snapshot",
]

CONSUMABLE_TEMP_FIELDS = [
    "die_holder_primary_c",
    "die_holder_secondary_c",
    "A_container_c",
    "A_pipe_c",
    "B_container_c",
    "B_pipe_c",
    "C_container_c",
    "C_pipe_c",
    "D_container_c",
    "D_pipe_c",
]
CONSUMABLE_TEMP_SETPOINT_KEYS = {
    "die_holder_primary_c": "die_holder_primary_temp_c",
    "die_holder_secondary_c": "die_holder_secondary_temp_c",
    "A_container_c": "A_container_c",
    "A_pipe_c": "A_pipe_c",
    "B_container_c": "B_container_c",
    "B_pipe_c": "B_pipe_c",
    "C_container_c": "C_container_c",
    "C_pipe_c": "C_pipe_c",
    "D_container_c": "D_container_c",
    "D_pipe_c": "D_pipe_c",
}


def consumable_temp_setpoint_csv_field(field: str) -> str:
    return f"{field[:-2]}_sp_c" if field.endswith("_c") else f"{field}_sp"


@dataclass
class JsonResponse:
    body: dict
    status: int = 200


_PARTS_MANUAL_INDEX_CACHE: dict[str, object] = {"signature": None, "payload": None}
_MANUAL_PAGE_RENDER_LOCK = threading.Lock()
_MANUAL_PAGE_RENDER_EVENTS: dict[str, threading.Event] = {}
_MANUAL_PAGE_PREFETCH_LOCK = threading.Lock()
_MANUAL_PAGE_PREFETCH_QUEUED: set[str] = set()
_MANUAL_PAGE_PRIORITY_PREFETCH_QUEUED: set[str] = set()
_MANUAL_PAGE_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="manual-page-prefetch")
_MANUAL_PAGE_PRIORITY_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manual-page-prefetch-priority")
_HELPER_PYTHON_CACHE: dict[tuple[str, ...], Path | None] = {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_fieldnames(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def read_json_value(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json_value(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _python_candidate_paths() -> list[Path]:
    root_venv_bin = ROOT_DIR / ".venv" / "bin" / "python"
    root_venv_win = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    base_venv_bin = BASE_DIR / ".venv" / "bin" / "python"
    base_venv_win = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    raw_candidates = [
        HELPER_PYTHON_ENV,
        sys.executable,
        root_venv_bin,
        root_venv_win,
        base_venv_bin,
        base_venv_win,
        BUNDLED_RUNTIME_PYTHON,
        shutil.which("python3") or "",
        shutil.which("python") or "",
    ]
    candidates: list[Path] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _python_supports_modules(candidate: Path, required_modules: tuple[str, ...]) -> bool:
    if not required_modules:
        return True
    try:
        result = subprocess.run(
            [
                str(candidate),
                "-c",
                "import importlib.util, sys; modules = tuple(sys.argv[1:]); missing = [name for name in modules if importlib.util.find_spec(name) is None]; raise SystemExit(0 if not missing else 1)",
                *required_modules,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def resolve_helper_python(required_modules: tuple[str, ...] = ("pypdf",)) -> Path | None:
    cache_key = tuple(required_modules)
    if cache_key in _HELPER_PYTHON_CACHE:
        return _HELPER_PYTHON_CACHE[cache_key]
    for candidate in _python_candidate_paths():
        if not candidate.exists() or not candidate.is_file():
            continue
        if _python_supports_modules(candidate, cache_key):
            resolved = candidate.resolve()
            _HELPER_PYTHON_CACHE[cache_key] = resolved
            return resolved
    _HELPER_PYTHON_CACHE[cache_key] = None
    return None


def manual_page_render_mode() -> str:
    if MANUAL_PAGE_MODE_ENV in {"image", "page-image", "png"}:
        return "image"
    if MANUAL_PAGE_MODE_ENV in {"pdf", "pdf-inline", "inline"}:
        return "pdf-inline"
    renderer_binary = MANUAL_PAGE_CACHE_DIR / "render_manual_page"
    if sys.platform == "darwin" and MANUAL_PAGE_RENDER_SCRIPT.exists() and (shutil.which("xcrun") or renderer_binary.exists()):
        return "image"
    return "pdf-inline"


def ensure_runtime_directories() -> None:
    for path in (
        STATE_DIR,
        DASHBOARD_EXPORTS_DIR,
        REPORTS_DIR,
        REPORT_CENTER_DIR,
        BACKUPS_DIR,
        DEVELOPMENT_MEDIA_DIR,
        MANUAL_PAGE_CACHE_DIR,
        STATIC_DIR / "uploads",
    ):
        path.mkdir(parents=True, exist_ok=True)


TRACKED_PATH_OVERRIDES_FILE = STATE_DIR / "tracked_path_overrides.json"
TRACKED_PATH_SPECS = (
    {"key": "orders_csv", "label": "Orders CSV", "global_name": "DRAW_ORDERS", "kind": "file", "default": DRAW_ORDERS},
    {"key": "parts_orders_csv", "label": "Parts Orders CSV", "global_name": "PART_ORDERS", "kind": "file", "default": PART_ORDERS},
    {"key": "schedule_csv", "label": "Schedule CSV", "global_name": "TOWER_SCHEDULE", "kind": "file", "default": TOWER_SCHEDULE},
    {"key": "selected_csv_json", "label": "Selected CSV JSON", "global_name": "SELECTED_CSV_JSON", "kind": "file", "default": SELECTED_CSV_JSON},
    {"key": "dataset_dir", "label": "Dataset Workspace", "global_name": "DATASET_DIR", "kind": "dir", "default": DATASET_DIR},
    {"key": "logs_dir", "label": "Logs Workspace", "global_name": "LOGS_DIR", "kind": "dir", "default": LOGS_DIR},
    {"key": "reports_dir", "label": "Reports Workspace", "global_name": "REPORTS_DIR", "kind": "dir", "default": REPORTS_DIR},
    {"key": "backups_dir", "label": "Backups Workspace", "global_name": "BACKUPS_DIR", "kind": "dir", "default": BACKUPS_DIR},
    {"key": "duckdb_path", "label": "DuckDB File", "global_name": "DUCKDB_PATH", "kind": "file", "default": DUCKDB_PATH},
)
TRACKED_PATH_SPEC_MAP = {str(item["key"]): item for item in TRACKED_PATH_SPECS}


def normalize_tracked_path_value(value: str | Path) -> Path:
    text = str(value or "").strip()
    candidate = Path(text).expanduser() if text else ROOT_DIR
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


def tracked_path_defaults() -> dict[str, Path]:
    return {str(item["key"]): Path(item["default"]) for item in TRACKED_PATH_SPECS}


def load_tracked_path_overrides() -> dict[str, Path]:
    raw = read_json_dict(TRACKED_PATH_OVERRIDES_FILE)
    overrides: dict[str, Path] = {}
    for item in TRACKED_PATH_SPECS:
        key = str(item["key"])
        value = raw.get(key)
        if not value:
            continue
        overrides[key] = normalize_tracked_path_value(value)
    return overrides


def apply_tracked_path_overrides(overrides: dict[str, Path] | None = None) -> None:
    values = tracked_path_defaults()
    for key, path in (overrides or {}).items():
        if key in values:
            values[key] = normalize_tracked_path_value(path)
    module_globals = globals()
    for item in TRACKED_PATH_SPECS:
        module_globals[str(item["global_name"])] = values[str(item["key"])]
    module_globals["REPORT_CENTER_DIR"] = module_globals["REPORTS_DIR"] / "report_center"
    module_globals["ARGON_MONTHLY_REPORT"] = module_globals["REPORTS_DIR"] / "gas" / "argon_monthly_report.csv"


def save_tracked_path_overrides(overrides: dict[str, Path]) -> None:
    defaults = tracked_path_defaults()
    payload = {
        key: str(normalize_tracked_path_value(path))
        for key, path in overrides.items()
        if key in defaults and normalize_tracked_path_value(path) != defaults[key]
    }
    if payload:
        write_json_value(TRACKED_PATH_OVERRIDES_FILE, payload)
        return
    try:
        TRACKED_PATH_OVERRIDES_FILE.unlink()
    except FileNotFoundError:
        pass


def current_tracked_paths() -> list[tuple[dict[str, object], Path]]:
    module_globals = globals()
    return [
        (item, Path(module_globals[str(item["global_name"])]))
        for item in TRACKED_PATH_SPECS
    ]


apply_tracked_path_overrides(load_tracked_path_overrides())
ensure_runtime_directories()

FULL_BACKUP_INTERVAL = timedelta(days=7)
FULL_BACKUP_POLICY_LABEL = "Runs inside the app once every 7 days while the app is active"
FULL_BACKUP_LOCK = threading.Lock()

FULL_BACKUP_SOURCES = (
    {"label": "Core data", "path": DATA_DIR, "target": "data"},
    {"label": "Dataset CSVs", "path": DATASET_DIR, "target": "data_set_csv"},
    {"label": "Logs", "path": LOGS_DIR, "target": "logs"},
    {"label": "Reports", "path": REPORTS_DIR, "target": "reports"},
    {"label": "Maintenance", "path": MAINTENANCE_DIR, "target": "maintenance"},
    {"label": "State", "path": STATE_DIR, "target": "state"},
    {"label": "Config", "path": ROOT_DIR / "config", "target": "config"},
    {"label": "Development media", "path": DEVELOPMENT_MEDIA_DIR, "target": "development_media"},
    {"label": "App uploads", "path": STATIC_DIR / "uploads", "target": "tower_rebuild/static/uploads"},
)


def path_is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def build_full_backup_sources() -> list[dict[str, object]]:
    sources = [
        {"label": str(item["label"]), "path": Path(item["path"]), "target": str(item["target"])}
        for item in FULL_BACKUP_SOURCES
    ]
    covered_dirs = [Path(item["path"]).resolve() for item in sources if Path(item["path"]).exists() and Path(item["path"]).is_dir()]
    covered_files = {Path(item["path"]).resolve() for item in sources if Path(item["path"]).exists() and Path(item["path"]).is_file()}
    for spec, current_path in current_tracked_paths():
        key = str(spec["key"])
        resolved_path = Path(current_path).resolve()
        if key == "backups_dir" or resolved_path == BACKUPS_DIR.resolve():
            continue
        if resolved_path in covered_files or any(path_is_within(resolved_path, directory) for directory in covered_dirs):
            continue
        target_name = key if str(spec["kind"]) == "dir" else f"{key}{resolved_path.suffix}"
        sources.append({
            "label": f"{spec['label']} override",
            "path": resolved_path,
            "target": f"external_paths/{target_name}",
        })
        if resolved_path.is_dir():
            covered_dirs.append(resolved_path)
        else:
            covered_files.add(resolved_path)
    return sources


def list_backup_directories(prefix: str | None = None) -> list[Path]:
    if not BACKUPS_DIR.exists():
        return []
    folders = [path for path in BACKUPS_DIR.iterdir() if path.is_dir()]
    if prefix:
        folders = [path for path in folders if path.name.startswith(prefix)]
    return sorted(folders, key=lambda path: path.stat().st_mtime, reverse=True)


def latest_backup_directory(prefix: str | None = None) -> Path | None:
    folders = list_backup_directories(prefix)
    return folders[0] if folders else None


def latest_backup_snapshot(prefix: str | None = None) -> dict[str, str] | None:
    latest = latest_backup_directory(prefix)
    if not latest:
        return None
    return {
        "name": latest.name,
        "path": str(latest),
        "modified": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
    }


def next_backup_snapshot_dir(prefix: str = "full_backup") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = BACKUPS_DIR / f"{prefix}_{stamp}"
    version = 2
    while candidate.exists():
        candidate = BACKUPS_DIR / f"{prefix}_{stamp}__{version}"
        version += 1
    return candidate


def create_full_backup_snapshot(trigger: str = "manual") -> dict[str, object]:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_dir = next_backup_snapshot_dir("full_backup")
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for item in build_full_backup_sources():
        source = Path(item["path"])
        target = snapshot_dir / str(item["target"])
        if not source.exists():
            missing.append({"label": str(item["label"]), "path": str(source)})
            continue
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied.append({"label": str(item["label"]), "path": str(source), "target": str(target.relative_to(snapshot_dir))})
    manifest = {
        "kind": "full_backup",
        "trigger": trigger,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "policy_label": FULL_BACKUP_POLICY_LABEL,
        "copied": copied,
        "missing": missing,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "name": snapshot_dir.name,
        "path": str(snapshot_dir),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "modified": datetime.fromtimestamp(snapshot_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "trigger": trigger,
    }


def ensure_app_weekly_full_backup() -> dict[str, object] | None:
    with FULL_BACKUP_LOCK:
        latest = latest_backup_directory("full_backup_")
        if latest:
            latest_modified = datetime.fromtimestamp(latest.stat().st_mtime)
            if datetime.now() - latest_modified < FULL_BACKUP_INTERVAL:
                return None
        try:
            return create_full_backup_snapshot(trigger="app-weekly")
        except Exception:
            return None


def slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "export"


def dedupe_clean_strings(values) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return clean


def parseBuilderMediaPaths(value) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return dedupe_clean_strings(part.strip() for part in text.split(";"))


def parseBuilderPhotoEntries(value) -> list[dict[str, str]]:
    text = str(value or "").strip()
    if not text:
        return []
    raw_items = []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_items = parsed
        except Exception:
            raw_items = []
    if not raw_items:
        raw_items = [{"path": path} for path in parseBuilderMediaPaths(text)]
    seen: set[str] = set()
    clean: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        temp_id = str(item.get("temp_id", "") or item.get("tempId", "")).strip()
        name = str(item.get("name", "")).strip()
        step_key = str(item.get("step_key", "") or item.get("stepKey", "")).strip()
        step_label = str(item.get("step_label", "") or item.get("stepLabel", "")).strip()
        if not path and not temp_id:
            continue
        dedupe_key = f"path:{path}" if path else f"temp:{temp_id}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        clean.append(
            {
                "path": path,
                "temp_id": temp_id,
                "name": name or (os.path.basename(path) if path else ""),
                "step_key": step_key,
                "step_label": step_label,
            }
        )
    return clean


def get_maintenance_runtime() -> dict[str, float]:
    stored = read_json_dict(MAINTENANCE_RUNTIME)
    defaults = {
        "furnace_hours": 0.0,
        "uv1_hours": 0.0,
        "uv2_hours": 0.0,
        "draw_count": 0.0,
    }
    runtime: dict[str, float] = {}
    source_map = {
        "furnace_hours": "furnace_hours",
        "uv1_hours": "uv1_hours",
        "uv2_hours": "uv2_hours",
        "draw_count": "last_draw_count",
    }
    for key, default in defaults.items():
        try:
            runtime[key] = float(stored.get(source_map[key], default))
        except (TypeError, ValueError):
            runtime[key] = float(default)
    return runtime


def get_maintenance_runtime_context() -> dict:
    stored = read_json_dict(MAINTENANCE_RUNTIME)
    runtime = get_maintenance_runtime()
    current_date_raw = str(stored.get("current_date", "")).strip()
    try:
        current_date = datetime.strptime(current_date_raw, "%Y-%m-%d").date() if current_date_raw else datetime.now().date()
    except ValueError:
        current_date = datetime.now().date()
    try:
        warn_days = int(float(stored.get("warn_days", 14)))
    except (TypeError, ValueError):
        warn_days = 14
    try:
        warn_hours = float(stored.get("warn_hours", 50.0))
    except (TypeError, ValueError):
        warn_hours = 50.0
    return {
        **runtime,
        "current_date": current_date,
        "warn_days": warn_days,
        "warn_hours": warn_hours,
    }


def read_maintenance_tracker_rows() -> list[dict[str, str]]:
    if not MAINTENANCE_DIR.exists():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(MAINTENANCE_DIR.glob("*.xlsx")):
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        rename_map = {
            "Equipment": "Component",
            "Task ID": "Task_ID",
            "Task Name": "Task",
            "Tracking Mode": "Tracking_Mode",
            "Required Parts": "Required_Parts",
            "Group": "Task_Group",
            "Planning months": "Planning_Window_Months",
            "Estimated duration min": "Est_Duration_Min",
            "Manual Page": "Page",
            "Document Name": "Manual_Name",
            "Document File/Link": "Document",
            "Last Done Date": "Last_Done_Date",
            "Last Done Hours": "Last_Done_Hours",
            "Last_Done_Draw": "Last_Done_Draw",
        }
        df = df.rename(columns=rename_map)
        for column in ["Component", "Task_ID", "Task", "Task_Group", "Tracking_Mode", "Required_Parts"]:
            if column not in df.columns:
                df[column] = ""
        df["Source_File"] = path.name
        for row in df.to_dict(orient="records"):
            component = str(row.get("Component", "")).strip()
            task = str(row.get("Task", "")).strip()
            if not component or not task:
                continue
            rows.append({key: "" if (value is None or (isinstance(value, float) and math.isnan(value))) else str(value) for key, value in row.items()})
    return rows


def split_required_parts(value: str | None) -> list[str]:
    text = str(value or "").replace("\n", ";").replace("/", ";")
    output = []
    for item in text.split(";"):
        part = item.strip()
        if part and part.lower() != "nan":
            output.append(part)
    return dedupe_strings(output)


def dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def ensure_parts_company(company_name: str) -> None:
    company = str(company_name or "").strip()
    if not company:
        return
    rows = read_csv_rows(PARTS_COMPANIES)
    fieldnames = read_csv_fieldnames(PARTS_COMPANIES) or ["Company", "Last Updated"]
    if any(str(row.get("Company", "")).strip().lower() == company.lower() for row in rows):
        return
    rows.append(
        {
            "Company": company,
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_csv_rows(PARTS_COMPANIES, rows, fieldnames)


def to_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def normalize_recurrence(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "nan"}:
        return "none"
    if text == "weekly":
        return "weekly"
    if text == "monthly":
        return "monthly"
    if text in {"every 3 months", "3 months", "quarterly"}:
        return "quarterly"
    if text in {"every 6 months", "6 months", "semiannual", "semi-annually"}:
        return "semiannual"
    if text == "yearly":
        return "yearly"
    return "none"


def next_recurrence_dt(dt: datetime, recurrence: str) -> datetime:
    if recurrence == "weekly":
        return dt + timedelta(weeks=1)
    if recurrence == "monthly":
        month = dt.month + 1
        year = dt.year
        if month > 12:
            month = 1
            year += 1
        return dt.replace(year=year, month=month)
    if recurrence == "quarterly":
        month = dt.month + 3
        year = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        return dt.replace(year=year, month=month)
    if recurrence == "semiannual":
        month = dt.month + 6
        year = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        return dt.replace(year=year, month=month)
    if recurrence == "yearly":
        return dt.replace(year=dt.year + 1)
    return dt


def month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def add_months(dt: datetime, months: int) -> datetime:
    month = dt.month + months
    year = dt.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def build_demo_schedule_events(anchor: datetime) -> list[dict]:
    demo_day = anchor + timedelta(days=2)
    week_span_start = (anchor + timedelta(days=1)).replace(hour=18, minute=30)
    week_span_end = (anchor + timedelta(days=4)).replace(hour=9, minute=30)
    month_span_start = (anchor + timedelta(days=9)).replace(hour=8, minute=0)
    month_span_end = (anchor + timedelta(days=18)).replace(hour=17, minute=30)
    slots = [
        ("Maintenance", demo_day.replace(hour=7, minute=30), demo_day.replace(hour=9, minute=0), "DEMO | Morning maintenance window"),
        ("Drawing", demo_day.replace(hour=9, minute=15), demo_day.replace(hour=11, minute=45), "DEMO | Draw order cluster A"),
        ("Management Event", demo_day.replace(hour=10, minute=0), demo_day.replace(hour=10, minute=40), "DEMO | Shift coordination review"),
        ("Drawing", demo_day.replace(hour=12, minute=15), demo_day.replace(hour=16, minute=0), "DEMO | Draw order cluster B"),
        ("Stop", demo_day.replace(hour=16, minute=15), demo_day.replace(hour=18, minute=0), "DEMO | Cooling / stop block"),
        ("Maintenance", demo_day.replace(hour=18, minute=15), demo_day.replace(hour=19, minute=45), "DEMO | End-of-day coating check"),
        ("Management Event", week_span_start, week_span_end, "DEMO | Multi-day planning window"),
        ("Maintenance", month_span_start, month_span_end, "DEMO | Extended maintenance hold"),
    ]
    demo_events = []
    for event_type, start, end, description in slots:
        demo_events.append(
            {
                "event_type": event_type,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "start_label": start.strftime("%Y-%m-%d %H:%M"),
                "end_label": end.strftime("%Y-%m-%d %H:%M"),
                "date_label": start.strftime("%b %d"),
                "day_key": start.strftime("%Y-%m-%d"),
                "weekday_label": start.strftime("%a"),
                "description": description,
                "recurrence": "demo",
                "duration_hours": round((end - start).total_seconds() / 3600.0, 2),
                "is_demo": True,
            }
        )
    return demo_events


def summarize_draw_orders() -> dict:
    rows = read_csv_rows(DRAW_ORDERS)
    status_counts: dict[str, int] = {}
    recent = []
    monthly_counts: dict[str, int] = {}
    for row in rows:
        status = (row.get("Status") or "Unknown").strip() or "Unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        updated_at = row.get("Status Updated At") or row.get("Timestamp") or ""
        dt = parse_dt(updated_at)
        if dt:
            month_key = dt.strftime("%m/%d")
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        recent.append(
            {
                "preform": row.get("Preform Number", ""),
                "project": row.get("Fiber Project", ""),
                "status": status,
                "priority": row.get("Priority", "Normal"),
                "length": row.get("Length (m)") or row.get("Required Length (m) (for T&M+costumer)") or "0",
                "updated_at": updated_at,
            }
        )

    def sort_key(item: dict) -> datetime:
        return parse_dt(item.get("updated_at")) or datetime.min

    recent.sort(key=sort_key, reverse=True)
    total = len(rows)
    done = status_counts.get("Done", 0)
    failed = status_counts.get("Failed", 0)
    active = total - done - failed
    in_progress = status_counts.get("In Progress", 0)
    scheduled = status_counts.get("Scheduled", 0)
    pending = status_counts.get("Pending", 0)
    return {
        "total": total,
        "active": active,
        "in_progress": in_progress,
        "scheduled": scheduled,
        "pending": pending,
        "done": done,
        "failed": failed,
        "status_counts": status_counts,
        "recent": recent[:6],
        "activity_series": [{"label": key, "value": value} for key, value in list(monthly_counts.items())[-8:]],
    }


def summarize_schedule() -> dict:
    rows = read_csv_rows(TOWER_SCHEDULE)
    events = []
    master_rows = []
    daily_load: dict[str, dict[str, float]] = {}
    now = datetime.now()
    anchor = now.replace(hour=0, minute=0, second=0, microsecond=0)
    range_start = anchor - timedelta(days=7)
    range_end = add_months(anchor, 3)

    for row in rows:
        start = parse_dt(row.get("Start DateTime"))
        end = parse_dt(row.get("End DateTime"))
        master_rows.append(
            {
                "index": len(master_rows),
                "event_type": row.get("Event Type", "Unknown"),
                "start": row.get("Start DateTime", ""),
                "end": row.get("End DateTime", ""),
                "description": (row.get("Description") or "").strip(),
                "recurrence": normalize_recurrence(row.get("Recurrence", "")) or "none",
            }
        )
        duration_hours = round(((end - start).total_seconds() / 3600.0), 2) if start and end else 0
        events.append(
            {
                "event_type": row.get("Event Type", "Unknown"),
                "start": row.get("Start DateTime", ""),
                "end": row.get("End DateTime", ""),
                "description": (row.get("Description") or "").strip(),
                "recurrence": normalize_recurrence(row.get("Recurrence", "")),
                "sort": start or datetime.max,
                "duration_hours": duration_hours,
            }
        )

    expanded_events = []
    for item in events:
        start = item["sort"]
        end = parse_dt(item["end"])
        if not start or start == datetime.max or not end:
            continue

        recurrence = item["recurrence"]
        duration = end - start

        if recurrence == "none":
            occurrences = [(start, end)]
        else:
            occurrences = []
            occ_start = start
            occ_end = end
            safety = 0
            while occ_end < range_start and safety < 5000:
                occ_start = next_recurrence_dt(occ_start, recurrence)
                occ_end = occ_start + duration
                safety += 1
            safety = 0
            while occ_start <= range_end and safety < 5000:
                occurrences.append((occ_start, occ_end))
                occ_start = next_recurrence_dt(occ_start, recurrence)
                occ_end = occ_start + duration
                safety += 1

        for occ_start, occ_end in occurrences:
            if occ_end < range_start or occ_start > range_end:
                continue
            hours = round((occ_end - occ_start).total_seconds() / 3600.0, 2)
            day_key = occ_start.strftime("%m/%d")
            bucket = daily_load.setdefault(day_key, {"events": 0, "hours": 0.0})
            bucket["events"] += 1
            bucket["hours"] += hours
            expanded_events.append(
                {
                    "event_type": item["event_type"],
                    "start": occ_start.isoformat(),
                    "end": occ_end.isoformat(),
                    "start_label": occ_start.strftime("%Y-%m-%d %H:%M"),
                    "end_label": occ_end.strftime("%Y-%m-%d %H:%M"),
                    "date_label": occ_start.strftime("%b %d"),
                    "day_key": occ_start.strftime("%Y-%m-%d"),
                    "weekday_label": occ_start.strftime("%a"),
                    "description": item["description"],
                    "recurrence": recurrence,
                    "duration_hours": hours,
                    "is_demo": False,
                }
            )

    for demo_event in build_demo_schedule_events(anchor):
        demo_start = parse_dt(demo_event["start"])
        if demo_start:
            day_key = demo_start.strftime("%m/%d")
            bucket = daily_load.setdefault(day_key, {"events": 0, "hours": 0.0})
            bucket["events"] += 1
            bucket["hours"] += demo_event["duration_hours"]
        expanded_events.append(demo_event)

    expanded_events.sort(key=lambda item: item["start"])
    upcoming = expanded_events[:12]
    type_counts: dict[str, int] = {}
    for item in expanded_events:
        key = item["event_type"]
        type_counts[key] = type_counts.get(key, 0) + 1
    daily_series = [
        {"label": key, "events": value["events"], "hours": round(value["hours"], 2)}
        for key, value in list(daily_load.items())[-10:]
    ]

    month_buckets: dict[str, dict] = {}
    for item in expanded_events:
        dt = parse_dt(item["start"])
        if not dt:
            continue
        bucket_key = dt.strftime("%Y-%m")
        bucket = month_buckets.setdefault(
            bucket_key,
            {"label": dt.strftime("%b %Y"), "events": 0, "hours": 0.0},
        )
        bucket["events"] += 1
        bucket["hours"] += item["duration_hours"]

    return {
        "upcoming": upcoming,
        "type_counts": type_counts,
        "total": len(expanded_events),
        "daily_series": daily_series,
        "expanded_events": expanded_events,
        "master_rows": master_rows,
        "month_series": [
            {"key": key, "label": value["label"], "events": value["events"], "hours": round(value["hours"], 2)}
            for key, value in sorted(month_buckets.items())
        ],
        "timeline_anchor": anchor.strftime("%Y-%m-%d"),
    }


def save_schedule_master_action(payload: dict) -> JsonResponse:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return JsonResponse({"ok": False, "message": "Rows payload is required."}, 400)
    fieldnames = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
    cleaned_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cleaned_rows.append(
            {
                "Event Type": str(row.get("event_type", "")).strip(),
                "Start DateTime": str(row.get("start", "")).strip(),
                "End DateTime": str(row.get("end", "")).strip(),
                "Description": str(row.get("description", "")).strip(),
                "Recurrence": normalize_recurrence(row.get("recurrence", "")) or "none",
            }
        )
    write_csv_rows(TOWER_SCHEDULE, cleaned_rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Schedule master saved.", "bootstrap": build_bootstrap_payload().body})


def add_schedule_event_action(payload: dict) -> JsonResponse:
    event_type = str(payload.get("eventType", "")).strip()
    start = str(payload.get("start", "")).strip()
    end = str(payload.get("end", "")).strip()
    description = str(payload.get("description", "")).strip()
    recurrence = normalize_recurrence(payload.get("recurrence", "")) or "none"
    if not event_type or not start or not end:
        return JsonResponse({"ok": False, "message": "Event type, start, and end are required."}, 400)
    fieldnames = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
    rows = read_csv_rows(TOWER_SCHEDULE)
    rows.append(
        {
            "Event Type": event_type,
            "Start DateTime": start,
            "End DateTime": end,
            "Description": description,
            "Recurrence": recurrence,
        }
    )
    write_csv_rows(TOWER_SCHEDULE, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Event added to schedule.", "bootstrap": build_bootstrap_payload().body})


def delete_schedule_event_action(payload: dict) -> JsonResponse:
    try:
        index = int(payload.get("index"))
    except (TypeError, ValueError):
        return JsonResponse({"ok": False, "message": "Valid event index is required."}, 400)
    rows = read_csv_rows(TOWER_SCHEDULE)
    fieldnames = read_csv_fieldnames(TOWER_SCHEDULE) or ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
    if index < 0 or index >= len(rows):
        return JsonResponse({"ok": False, "message": "Event not found."}, 404)
    del rows[index]
    write_csv_rows(TOWER_SCHEDULE, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Event deleted.", "bootstrap": build_bootstrap_payload().body})


def summarize_part_orders() -> dict:
    rows = read_csv_rows(PART_ORDERS)
    status_counts: dict[str, int] = {}
    open_orders = []
    all_orders = []
    maintenance_open = 0
    approval_queue = []
    ready_to_order = []
    ordered_open = []
    received_pending_inventory = []
    maintenance_linked = []
    project_names = dedupe_strings([item.get(PROJECTS_COL, "") for item in read_csv_rows(PROJECTS_FIBER)])
    company_names = [row.get("Company", "") for row in read_csv_rows(PARTS_COMPANIES)]
    for index, row in enumerate(rows):
        status = (row.get("Status") or "Unknown").strip() or "Unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        project_name = row.get("Project Name", "")
        company = row.get("Company", "")
        maintenance_component = row.get("Maintenance Component", "")
        maintenance_task = row.get("Maintenance Task", "")
        details = (row.get("Details") or "").strip()
        inventory_synced = (row.get("Inventory Synced") or "").strip()
        received_state = (row.get("Received State") or "").strip()
        item = {
            "index": index,
            "status": status,
            "part_name": row.get("Part Name", ""),
            "serial_number": row.get("Serial Number", ""),
            "project": project_name,
            "details": details,
            "opened_by": row.get("Opened By", ""),
            "approval_requested_from": row.get("Approval Requested From", ""),
            "approved": row.get("Approved", ""),
            "approved_by": row.get("Approved By", ""),
            "approval_date": row.get("Approval Date", ""),
            "received_date": row.get("Received Date", ""),
            "received_state": received_state,
            "ordered_by": row.get("Ordered By", ""),
            "date_ordered": row.get("Date Ordered", ""),
            "company": company,
            "inventory_synced": inventory_synced,
            "maintenance_component": maintenance_component,
            "maintenance_task": maintenance_task,
            "maintenance_task_id": row.get("Maintenance Task ID", ""),
            "wait_id": row.get("Wait ID", ""),
            "origin": "Maintenance" if maintenance_component or maintenance_task or str(project_name).strip().lower() == "maintenance" else "General",
        }
        all_orders.append(item)
        if status not in {"Received", "Closed"}:
            open_orders.append(item)
        if str(project_name).strip().lower() == "maintenance" and status not in {"Received", "Closed"}:
            maintenance_open += 1
        if company:
            company_names.append(company)
        if status == "Wait for Approval":
            approval_queue.append(item)
        if status == "Approved":
            ready_to_order.append(item)
        if status == "Ordered":
            ordered_open.append(item)
        if status == "Received" and (inventory_synced != "Yes" or not received_state):
            received_pending_inventory.append(item)
        if maintenance_component or maintenance_task or str(project_name).strip().lower() == "maintenance":
            maintenance_linked.append(item)
    sorted_status_counts = {key: status_counts.get(key, 0) for key in PART_STATUS_ORDER if key in status_counts}
    for key, value in status_counts.items():
        if key not in sorted_status_counts:
            sorted_status_counts[key] = value
    return {
        "status_counts": sorted_status_counts,
        "all_orders": all_orders,
        "open_orders": open_orders[:8],
        "maintenance_open": maintenance_open,
        "total": len(rows),
        "status_series": [{"label": key, "value": value} for key, value in sorted_status_counts.items()],
        "queues": {
            "approval": approval_queue[:8],
            "approved": ready_to_order[:8],
            "ordered": ordered_open[:8],
            "received_pending": received_pending_inventory[:8],
            "maintenance_linked": maintenance_linked[:8],
        },
        "queue_counts": {
            "approval": len(approval_queue),
            "approved": len(ready_to_order),
            "ordered": len(ordered_open),
            "received_pending": len(received_pending_inventory),
            "maintenance_linked": len(maintenance_linked),
        },
        "project_names": dedupe_strings(project_names + [item["project"] for item in all_orders]),
        "company_names": dedupe_strings(company_names),
        "status_order": PART_STATUS_ORDER,
    }


def summarize_inventory() -> dict:
    rows = read_csv_rows(PARTS_INVENTORY)
    low_stock = []
    component_pressure: dict[str, int] = {}
    mounted_rows = []
    for row in rows:
        quantity = to_float(row.get("Quantity"))
        min_level = to_float(row.get("Min Level"))
        location = row.get("Location", "")
        if str(location).strip().lower() == "mounted":
            mounted_rows.append(
                {
                    "part_name": row.get("Part Name", ""),
                    "serial_number": row.get("Serial Number", ""),
                    "quantity": quantity,
                    "component": row.get("Component", ""),
                    "location": location,
                }
            )
        if quantity <= min_level:
            component = row.get("Component", "") or "Unknown"
            component_pressure[component] = component_pressure.get(component, 0) + 1
            low_stock.append(
                {
                    "part_name": row.get("Part Name", ""),
                    "component": row.get("Component", ""),
                    "quantity": quantity,
                    "min_level": min_level,
                    "location": row.get("Location", ""),
                }
            )
    low_stock.sort(key=lambda item: (item["quantity"], item["part_name"]))
    pressure_series = [
        {"label": key, "value": value}
        for key, value in sorted(component_pressure.items(), key=lambda item: item[1], reverse=True)[:6]
    ]
    return {
        "low_stock": low_stock[:8],
        "low_stock_total": len(low_stock),
        "tracked_parts": len(rows),
        "pressure_series": pressure_series,
        "mounted_rows": mounted_rows[:12],
        "inventory_rows": [
            {
                "part_name": row.get("Part Name", ""),
                "item_type": row.get("Item Type", ""),
                "component": row.get("Component", ""),
                "supplier": row.get("Supplier", ""),
                "serial_number": row.get("Serial Number", ""),
                "location": row.get("Location", ""),
                "location_serial": row.get("Location Serial", ""),
                "quantity": to_float(row.get("Quantity")),
                "min_level": to_float(row.get("Min Level")),
                "notes": row.get("Notes", ""),
            }
            for row in rows
        ],
        "location_names": dedupe_strings([row.get("Location Name", "") for row in read_csv_rows(PARTS_LOCATIONS)]),
    }


def get_parts_manual_roots() -> list[Path]:
    return [root for root in [MANUALS_DIR, EXTERNAL_MANUALS_DIR] if root.exists() and root.is_dir()]


def pick_parts_manuals_dir() -> Path | None:
    roots = get_parts_manual_roots()
    if not roots:
        return None
    ranked = sorted(
        roots,
        key=lambda root: (len(list(root.glob("*.pdf"))), int(root.stat().st_mtime)),
        reverse=True,
    )
    return ranked[0]


def empty_parts_manual_index(message: str = "") -> dict[str, object]:
    return {
        "ok": True,
        "manuals": [],
        "rows": [],
        "totals": {"manual_count": 0, "row_count": 0},
        "message": message,
    }


def build_parts_manual_signature(root: Path | None) -> tuple[object, ...]:
    if root is None:
        return ("missing",)
    entries: list[tuple[str, int, int]] = []
    for pdf_path in sorted(root.glob("*.pdf")):
        try:
            stats = pdf_path.stat()
        except OSError:
            continue
        entries.append((pdf_path.name, int(stats.st_mtime), int(stats.st_size)))
    return (str(root), *entries)


def get_parts_manual_index() -> dict[str, object]:
    manuals_root = pick_parts_manuals_dir()
    signature = build_parts_manual_signature(manuals_root)
    cached_signature = _PARTS_MANUAL_INDEX_CACHE.get("signature")
    cached_payload = _PARTS_MANUAL_INDEX_CACHE.get("payload")
    if cached_signature == signature and isinstance(cached_payload, dict):
        return cached_payload
    if manuals_root is None:
        payload = empty_parts_manual_index("No manuals directory found.")
    elif not MANUAL_INDEX_SCRIPT.exists():
        payload = empty_parts_manual_index("Manual index script is missing.")
    else:
        helper_python = resolve_helper_python(("pypdf",))
        if helper_python is None:
            payload = empty_parts_manual_index("No compatible PDF helper runtime with pypdf was found.")
        else:
            try:
                result = subprocess.run(
                    [str(helper_python), str(MANUAL_INDEX_SCRIPT), str(manuals_root)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                payload = json.loads(result.stdout or "{}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
                payload = empty_parts_manual_index("Manual index build failed.")
            else:
                payload["message"] = payload.get("message", "")
                payload["source_dir"] = str(manuals_root)
                payload["runtime_python"] = str(helper_python)
    if isinstance(payload, dict):
        payload["render_mode"] = manual_page_render_mode()
    _PARTS_MANUAL_INDEX_CACHE["signature"] = signature
    _PARTS_MANUAL_INDEX_CACHE["payload"] = payload
    return payload


def build_parts_manual_index_payload() -> JsonResponse:
    return JsonResponse(get_parts_manual_index())


def summarize_parts_manual_lookup() -> dict[str, object]:
    manuals_root = pick_parts_manuals_dir()
    cached_payload = _PARTS_MANUAL_INDEX_CACHE.get("payload")
    totals = cached_payload.get("totals") if isinstance(cached_payload, dict) else {}
    helper_python = resolve_helper_python(("pypdf",))
    manual_count = len(list(manuals_root.glob("*.pdf"))) if manuals_root else 0
    return {
        "manual_count": manual_count,
        "row_count": int((totals or {}).get("row_count", 0)),
        "message": str((cached_payload or {}).get("message", "")) if isinstance(cached_payload, dict) else "",
        "render_mode": manual_page_render_mode(),
        "helper_python": str(helper_python) if helper_python else "",
    }


def _manual_page_render_artifacts(pdf_path: Path, page_number: int) -> tuple[int, str, Path]:
    safe_page = max(1, int(page_number or 1))
    MANUAL_PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    render_signature = hashlib.sha1(
        "::".join(
            [
                str(pdf_path.resolve()),
                str(pdf_path.stat().st_mtime_ns),
                str(safe_page),
                str(MANUAL_PAGE_RENDER_SCRIPT.stat().st_mtime_ns if MANUAL_PAGE_RENDER_SCRIPT.exists() else 0),
            ]
        ).encode("utf-8")
    ).hexdigest()
    output_path = MANUAL_PAGE_CACHE_DIR / f"{render_signature}.png"
    return safe_page, render_signature, output_path


def _manual_page_renderer_env() -> dict[str, str]:
    MANUAL_PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clang_cache = MANUAL_PAGE_CACHE_DIR / "clang_module_cache"
    swift_cache = MANUAL_PAGE_CACHE_DIR / "swift_module_cache"
    clang_cache.mkdir(parents=True, exist_ok=True)
    swift_cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CLANG_MODULE_CACHE_PATH"] = str(clang_cache)
    env["SWIFT_MODULE_CACHE_PATH"] = str(swift_cache)
    xcode_developer_dir = Path("/Applications/Xcode.app/Contents/Developer")
    if xcode_developer_dir.exists():
        env["DEVELOPER_DIR"] = str(xcode_developer_dir)
    return env


def _ensure_manual_page_renderer_binary(env: dict[str, str]) -> Path:
    if not MANUAL_PAGE_RENDER_SCRIPT.exists():
        raise FileNotFoundError("Manual page render script is missing.")
    renderer_binary = MANUAL_PAGE_CACHE_DIR / "render_manual_page"
    renderer_source_mtime = MANUAL_PAGE_RENDER_SCRIPT.stat().st_mtime_ns
    binary_needs_rebuild = (
        not renderer_binary.exists()
        or renderer_binary.stat().st_mtime_ns < renderer_source_mtime
    )
    if binary_needs_rebuild:
        subprocess.run(
            ["xcrun", "swiftc", "-O", str(MANUAL_PAGE_RENDER_SCRIPT), "-o", str(renderer_binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    return renderer_binary


def _render_manual_page_image_once(pdf_path: Path, safe_page: int, output_path: Path) -> Path:
    env = _manual_page_renderer_env()
    renderer_binary = _ensure_manual_page_renderer_binary(env)

    result = subprocess.run(
        [str(renderer_binary), str(pdf_path), str(safe_page), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    rendered_path = Path((result.stdout or "").strip() or output_path)
    if rendered_path.exists() and rendered_path.is_file():
        return rendered_path
    raise FileNotFoundError("Manual page image render failed.")


def render_manual_page_image(pdf_path: Path, page_number: int) -> Path:
    if manual_page_render_mode() != "image":
        raise RuntimeError("Manual page image rendering is not enabled on this platform.")
    safe_page, render_signature, output_path = _manual_page_render_artifacts(pdf_path, page_number)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    owns_render = False
    with _MANUAL_PAGE_RENDER_LOCK:
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        render_event = _MANUAL_PAGE_RENDER_EVENTS.get(render_signature)
        if render_event is None:
            render_event = threading.Event()
            _MANUAL_PAGE_RENDER_EVENTS[render_signature] = render_event
            owns_render = True
    if owns_render:
        try:
            return _render_manual_page_image_once(pdf_path, safe_page, output_path)
        finally:
            with _MANUAL_PAGE_RENDER_LOCK:
                _MANUAL_PAGE_RENDER_EVENTS.pop(render_signature, None)
                render_event.set()
    render_event.wait(timeout=125)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    return _render_manual_page_image_once(pdf_path, safe_page, output_path)


def _prefetch_manual_page_image_task(pdf_path: Path, page_number: int, render_signature: str, priority: bool = False) -> None:
    try:
        render_manual_page_image(pdf_path, page_number)
    except Exception:
        return
    finally:
        with _MANUAL_PAGE_PREFETCH_LOCK:
            if priority:
                _MANUAL_PAGE_PRIORITY_PREFETCH_QUEUED.discard(render_signature)
            else:
                _MANUAL_PAGE_PREFETCH_QUEUED.discard(render_signature)


def schedule_manual_page_prefetch(pdf_path: Path, page_numbers: list[int], priority: bool = False) -> int:
    if manual_page_render_mode() != "image":
        return 0
    scheduled = 0
    seen_pages: set[int] = set()
    for page_number in page_numbers:
        safe_page, render_signature, output_path = _manual_page_render_artifacts(pdf_path, page_number)
        if safe_page in seen_pages:
            continue
        seen_pages.add(safe_page)
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        with _MANUAL_PAGE_PREFETCH_LOCK:
            queued_signatures = _MANUAL_PAGE_PRIORITY_PREFETCH_QUEUED if priority else _MANUAL_PAGE_PREFETCH_QUEUED
            if render_signature in queued_signatures:
                continue
            queued_signatures.add(render_signature)
        executor = _MANUAL_PAGE_PRIORITY_PREFETCH_EXECUTOR if priority else _MANUAL_PAGE_PREFETCH_EXECUTOR
        executor.submit(_prefetch_manual_page_image_task, pdf_path, safe_page, render_signature, priority)
        scheduled += 1
    return scheduled


def summarize_maintenance_rebuild() -> dict:
    runtime_context = get_maintenance_runtime_context()
    tasks = read_maintenance_tracker_rows()
    state_rows = read_csv_rows(MAINTENANCE_STATE)
    action_rows = read_csv_rows(MAINTENANCE_ACTIONS)
    wait_rows = [row for row in read_csv_rows(MAINTENANCE_WAITS) if not str(row.get("resolved_ts", "")).strip()]
    part_orders = read_csv_rows(PART_ORDERS)
    inventory_rows = read_csv_rows(PARTS_INVENTORY)
    schedule = summarize_schedule()
    work_package_rows = read_csv_rows(MAINTENANCE_DIR / "maintenance_work_packages.csv")
    fault_rows = read_csv_rows(FAULTS_LOG)
    fault_action_rows = read_csv_rows(FAULTS_ACTIONS_LOG)
    execute_status_rank = {
        "SCHEDULED": 0,
        "IN_PROGRESS": 1,
        "PREP_READY": 2,
        "WAIT FOR PART": 3,
        "BLOCKED_PARTS": 4,
    }
    due_lookup: dict[str, dict[str, object]] = {}
    try:
        helper_df = load_maintenance_folder_df(str(MAINTENANCE_DIR))
        helper_status_df = compute_maintenance_status_df(
            helper_df,
            current_draw_count=int(runtime_context["draw_count"]),
            furnace_hours=float(runtime_context["furnace_hours"]),
            uv1_hours=float(runtime_context["uv1_hours"]),
            uv2_hours=float(runtime_context["uv2_hours"]),
            warn_days=int(runtime_context["warn_days"]),
            warn_hours=float(runtime_context["warn_hours"]),
            current_date=runtime_context["current_date"],
        )
        for helper_row in helper_status_df.to_dict(orient="records"):
            helper_task_id = str(helper_row.get("Task_ID", "")).strip()
            helper_component = str(helper_row.get("Component", "")).strip().lower()
            helper_task = str(helper_row.get("Task", "")).strip().lower()
            helper_key = helper_task_id or f"{helper_component}::{helper_task}"
            if not helper_key:
                continue
            next_due_date = helper_row.get("Next_Due_Date")
            if hasattr(next_due_date, "isoformat"):
                next_due_date = next_due_date.isoformat()
            due_lookup[helper_key] = {
                "next_due_date": str(next_due_date or "").strip(),
                "next_due_hours": helper_row.get("Next_Due_Hours"),
                "next_due_draw": helper_row.get("Next_Due_Draw"),
                "hours_source": str(helper_row.get("Hours_Source", "")).strip(),
                "timing_status": str(helper_row.get("Status", "")).strip(),
            }
    except Exception:
        due_lookup = {}

    def maintenance_signature(task_id: str, component: str, task: str) -> str:
        raw_task_id = str(task_id or "").strip()
        normalized_task_id = raw_task_id
        if raw_task_id:
            embedded_task_id = re.search(r"([A-Za-z0-9]+-MNT-\d+)", raw_task_id, re.IGNORECASE)
            if embedded_task_id:
                normalized_task_id = embedded_task_id.group(1)
            elif raw_task_id.lower().startswith("demo-"):
                normalized_task_id = re.sub(r"^demo-[^-]+-", "", raw_task_id, flags=re.IGNORECASE)
                normalized_task_id = re.sub(r"-\d+$", "", normalized_task_id)
        base = str(normalized_task_id or "").strip().lower() or f"{str(component or '').strip().lower()}::{str(task or '').strip().lower()}"
        return base

    state_by_task = {str(row.get("task_id", "")).strip(): row for row in state_rows if str(row.get("task_id", "")).strip()}
    state_by_signature = {
        maintenance_signature(row.get("task_id", ""), row.get("component", ""), row.get("task", "")): row
        for row in state_rows
    }
    wait_by_task = {str(row.get("maintenance_task_id", "")).strip(): row for row in wait_rows if str(row.get("maintenance_task_id", "")).strip()}
    wait_by_signature = {
        maintenance_signature(row.get("maintenance_task_id", ""), row.get("maintenance_component", ""), row.get("maintenance_task", "")): row
        for row in wait_rows
    }
    work_package_by_task = {str(row.get("Task_ID", "")).strip(): row for row in work_package_rows if str(row.get("Task_ID", "")).strip()}
    work_package_by_signature = {
        maintenance_signature(row.get("Task_ID", ""), row.get("Component", ""), row.get("Task", "")): row
        for row in work_package_rows
    }
    linked_orders_by_task: dict[str, list[dict[str, str]]] = {}
    for row in part_orders:
        task_id = str(row.get("Maintenance Task ID", "")).strip()
        if not task_id:
            continue
        linked_orders_by_task.setdefault(task_id, []).append(row)

    stock_names = {str(row.get("Part Name", "")).strip().lower() for row in inventory_rows if to_float(row.get("Quantity")) > 0 and str(row.get("Location", "")).strip().lower() != "mounted"}
    completed_ids = [str(row.get("maintenance_task_id", "")).strip() for row in action_rows if str(row.get("maintenance_task_id", "")).strip()]
    recent_actions = [
        {
            "task_id": str(row.get("maintenance_task_id", "")).strip(),
            "component": row.get("maintenance_component", ""),
            "task": row.get("maintenance_task", ""),
            "done_date": row.get("maintenance_done_date", ""),
            "note": row.get("maintenance_note", ""),
        }
        for row in action_rows[-10:]
    ][::-1]

    task_rows = []
    prep_queue = []
    execute_queue = []
    blocked_tracker = []
    seen_task_signatures: set[str] = set()
    for row in tasks:
        task_id = str(row.get("Task_ID", "")).strip()
        task_signature = maintenance_signature(task_id, row.get("Component", ""), row.get("Task", ""))
        if task_signature in seen_task_signatures:
            continue
        seen_task_signatures.add(task_signature)
        helper_key = task_id or f"{str(row.get('Component', '')).strip().lower()}::{str(row.get('Task', '')).strip().lower()}"
        due_meta = due_lookup.get(helper_key, {})
        required_parts = split_required_parts(row.get("Required_Parts", ""))
        missing_parts = [part for part in required_parts if part.lower() not in stock_names]
        linked_orders = linked_orders_by_task.get(task_id, [])
        open_linked = [item for item in linked_orders if str(item.get("Status", "")).strip() in {"Opened", "Wait for Approval", "Approved", "Ordered", "Received"}]
        received_waiting_sync = [
            item for item in open_linked
            if str(item.get("Status", "")).strip() == "Received"
            and str(item.get("Inventory Synced", "")).strip().lower() != "yes"
        ]
        current_state = str((state_by_task.get(task_id) or state_by_signature.get(task_signature) or {}).get("state", "")).strip() or ""
        flow_state = "Ready for preparation"
        if received_waiting_sync:
            flow_state = "Received, waiting inventory action"
        elif open_linked:
            flow_state = "Wait for order"
        elif missing_parts:
            flow_state = "Missing parts, no linked order yet"
        if task_id in wait_by_task or task_signature in wait_by_signature:
            current_state = current_state or "WAIT FOR PART"
        if not current_state:
            current_state = "PREP_READY" if flow_state == "Ready for preparation" else "BLOCKED_PARTS"
        item = {
            "task_id": task_id,
            "component": row.get("Component", ""),
            "task": row.get("Task", ""),
            "task_group": row.get("Task_Group", ""),
            "tracking_mode": row.get("Tracking_Mode", ""),
            "hours_source": row.get("Hours_Source", "") or due_meta.get("hours_source", ""),
            "required_parts": required_parts,
            "missing_parts": missing_parts,
            "status": current_state,
            "timing_status": str(due_meta.get("timing_status", "")),
            "flow_state": flow_state,
            "source_file": row.get("Source_File", ""),
            "last_done_date": row.get("Last_Done_Date", ""),
            "last_done_hours": row.get("Last_Done_Hours", ""),
            "last_done_draw": row.get("Last_Done_Draw", ""),
            "next_due_date": str(due_meta.get("next_due_date", "")),
            "next_due_hours": due_meta.get("next_due_hours", ""),
            "next_due_draw": due_meta.get("next_due_draw", ""),
            "est_duration_min": row.get("Est_Duration_Min", ""),
            "procedure_summary": row.get("Procedure Summary", ""),
            "safety_notes": row.get("Safety/Notes", ""),
            "manual_name": row.get("Manual_Name", ""),
            "manual_link": row.get("Document", ""),
            "manual_page": row.get("Page", ""),
            "wait_note": str(
                (
                    wait_by_task.get(task_id)
                    or wait_by_signature.get(task_signature)
                    or {}
                ).get("reason", "")
                or (
                    wait_by_task.get(task_id)
                    or wait_by_signature.get(task_signature)
                    or {}
                ).get("note", "")
            ).strip(),
            "linked_open_count": len(open_linked),
            "linked_received_waiting_sync": len(received_waiting_sync),
            "linked_ready_count": len([item for item in linked_orders if str(item.get("Status", "")).strip() == "Received" and str(item.get("Inventory Synced", "")).strip().lower() == "yes"]),
            "work_package": {
                "preparation_checklist": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Preparation_Checklist", "")).strip(),
                "safety_protocol": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Safety_Protocol", "")).strip(),
                "safety_fall_risk": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Safety_Fall_Risk", "")).strip(),
                "safety_tnm_presence": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Safety_TnM_Presence", "")).strip(),
                "procedure_steps": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Procedure_Steps", "")).strip(),
                "procedure_photos": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Procedure_Photos", "")).strip(),
                "sanity_checklist": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Sanity_Checklist", "")).strip(),
                "sanity_results": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Sanity_Results", "")).strip(),
                "supplier_name": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Supplier_Name", "")).strip(),
                "supplier_details": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Supplier_Details", "")).strip(),
                "draw_stop_plan": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Draw_Stop_Plan", "")).strip(),
                "est_stop_min": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Est_Stop_Min", "")).strip(),
                "completion_criteria": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Completion_Criteria", "")).strip(),
                "last_updated": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Last_Updated", "")).strip(),
                "updated_by": str((work_package_by_task.get(task_id) or work_package_by_signature.get(task_signature) or {}).get("Updated_By", "")).strip(),
            },
            "linked_orders": [
                {
                    "part_name": order.get("Part Name", ""),
                    "status": order.get("Status", ""),
                    "details": order.get("Details", ""),
                }
                for order in linked_orders[:6]
            ],
        }
        task_rows.append(item)
        if current_state in {"PREP_READY", "BLOCKED_PARTS", "WAIT FOR PART"}:
            prep_queue.append(item)
        if current_state in {"PREP_READY", "IN_PROGRESS", "BLOCKED_PARTS", "WAIT FOR PART", "SCHEDULED"}:
            execute_queue.append(item)
        if flow_state != "Ready for preparation" or current_state in {"BLOCKED_PARTS", "WAIT FOR PART"}:
            blocked_tracker.append(item)

    task_rows.sort(key=lambda item: (item["status"], item["component"], item["task"]))
    prep_queue.sort(key=lambda item: (item["status"], item["component"], item["task"]))
    execute_queue.sort(key=lambda item: (execute_status_rank.get(str(item["status"]).upper(), 99), item["component"], item["task"]))
    blocked_tracker.sort(key=lambda item: (item["status"], item["component"], item["task"]))
    maintenance_events = [
        item for item in schedule.get("upcoming", [])
        if "maintenance" in str(item.get("event_type", "")).lower()
    ]
    prep_events = [
        item for item in schedule.get("upcoming", [])
        if any(token in str(item.get("event_type", "")).lower() for token in ["maintenance preparation", "maintenance parts check"])
    ]
    smart_todo = (blocked_tracker[:4] + [item for item in prep_queue if item["task_id"] not in {row["task_id"] for row in blocked_tracker[:4]}][:4])[:8]
    faults_recent = [
        {
            "fault_id": str(row.get("fault_id", "")).strip(),
            "ts": str(row.get("fault_ts", "")).strip(),
            "component": str(row.get("fault_component", "")).strip(),
            "title": str(row.get("fault_title", "")).strip(),
            "severity": str(row.get("fault_severity", "")).strip() or "medium",
            "related_draw": str(row.get("fault_related_draw", "")).strip(),
        }
        for row in fault_rows[-16:]
    ][::-1]
    fault_actions_by_id: dict[str, list[dict[str, str]]] = {}
    for row in fault_action_rows:
        fault_id = str(row.get("fault_id", "")).strip()
        if not fault_id:
            continue
        fault_actions_by_id.setdefault(fault_id, []).append(row)
    component_fault_counts: dict[str, int] = {}
    for row in fault_rows:
        component = str(row.get("fault_component", "")).strip() or "Unknown"
        component_fault_counts[component] = component_fault_counts.get(component, 0) + 1
    fault_hotspots = [
        {"component": component, "count": count}
        for component, count in sorted(component_fault_counts.items(), key=lambda item: (-item[1], item[0]))
    ][:8]
    correlation_watch = []
    task_components = {str(item.get("component", "")).strip().lower(): item for item in task_rows}
    for fault in faults_recent:
        component_key = str(fault.get("component", "")).strip().lower()
        related_task = task_components.get(component_key)
        correlation_watch.append(
            {
                "fault_title": fault["title"],
                "fault_component": fault["component"],
                "fault_severity": fault["severity"],
                "linked_task": related_task["task"] if related_task else "",
                "linked_status": related_task["status"] if related_task else "",
            }
        )
    runtime = get_maintenance_runtime()

    return {
        "metrics": [
            {"label": "Tasks", "value": len(task_rows)},
            {"label": "Prep Queue", "value": len(prep_queue)},
            {"label": "Blocked", "value": len([item for item in task_rows if item["status"] in {"BLOCKED_PARTS", "WAIT FOR PART"}])},
            {"label": "Recent Actions", "value": len(action_rows)},
        ],
        "tasks": task_rows,
        "prep_queue": prep_queue[:24],
        "execute_queue": execute_queue[:24],
        "blocked_tracker": blocked_tracker[:24],
        "recent_actions": recent_actions,
        "completed_ids": completed_ids[-12:],
        "maintenance_events": maintenance_events[:8],
        "prep_events": prep_events[:8],
        "smart_todo": smart_todo,
        "faults_recent": faults_recent,
        "fault_hotspots": fault_hotspots,
        "fault_actions_total": len(fault_action_rows),
        "correlation_watch": correlation_watch[:10],
        "timeline_runtime": runtime,
    }


def save_maintenance_runtime_action(payload: dict) -> JsonResponse:
    current = get_maintenance_runtime()
    stored = read_json_dict(MAINTENANCE_RUNTIME)
    updated = dict(stored)
    field_map = {
        "furnaceHours": "furnace_hours",
        "uv1Hours": "uv1_hours",
        "uv2Hours": "uv2_hours",
        "drawCount": "last_draw_count",
    }
    runtime_key_map = {
        "furnaceHours": "furnace_hours",
        "uv1Hours": "uv1_hours",
        "uv2Hours": "uv2_hours",
        "drawCount": "draw_count",
    }
    for payload_key, state_key in field_map.items():
        raw = payload.get(payload_key, current[runtime_key_map[payload_key]])
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return JsonResponse({"ok": False, "message": f"Invalid value for {payload_key}."}, 400)
        updated[state_key] = int(round(value)) if state_key == "last_draw_count" else value
    updated["current_date"] = datetime.now().strftime("%Y-%m-%d")
    write_json_value(MAINTENANCE_RUNTIME, updated)
    return JsonResponse({"ok": True, "message": "Maintenance runtime updated.", "bootstrap": build_bootstrap_payload().body})


def save_maintenance_work_package_action(payload: dict) -> JsonResponse:
    task_id = str(payload.get("taskId", "")).strip()
    component = str(payload.get("component", "")).strip()
    task = str(payload.get("task", "")).strip()
    if not task_id or not component or not task:
        return JsonResponse({"ok": False, "message": "Task id, component, and task are required."}, 400)
    path = MAINTENANCE_DIR / "maintenance_work_packages.csv"
    rows = read_csv_rows(path)
    fieldnames = read_csv_fieldnames(path) or [
        "Task_ID", "Component", "Task", "Task_Group", "Required_Parts", "Preparation_Checklist", "Safety_Protocol",
        "Safety_Fall_Risk", "Safety_TnM_Presence", "Procedure_Steps", "Procedure_Photos", "Sanity_Checklist", "Draw_Stop_Plan",
        "Est_Stop_Min", "Completion_Criteria", "Supplier_Name", "Supplier_Details", "Sanity_Results", "Last_Updated", "Updated_By",
    ]
    for extra_field in ["Supplier_Name", "Supplier_Details", "Sanity_Checklist", "Sanity_Results"]:
        if extra_field not in fieldnames:
            fieldnames.append(extra_field)
    index = next((i for i, row in enumerate(rows) if str(row.get("Task_ID", "")).strip() == task_id), None)
    base_row = rows[index] if index is not None else {key: "" for key in fieldnames}
    photo_entries = parseBuilderPhotoEntries(payload.get("procedurePhotos", ""))
    upload_items = payload.get("photoUploads", []) or []
    if upload_items:
        photo_dir = MAINTENANCE_PACKAGE_PHOTOS_DIR / slugify(component) / slugify(task_id)
        photo_dir.mkdir(parents=True, exist_ok=True)
        uploaded_paths: dict[str, str] = {}
        for item in upload_items:
            temp_id = str(item.get("temp_id", "")).strip()
            filename = os.path.basename(str(item.get("name", "")).strip())
            content = str(item.get("content", "")).strip()
            if not filename or not content:
                continue
            try:
                raw = base64.b64decode(content)
            except Exception:
                continue
            candidate = photo_dir / filename
            stem = candidate.stem
            suffix = candidate.suffix
            version = 2
            while candidate.exists():
                candidate = photo_dir / f"{stem}__{version}{suffix}"
                version += 1
            candidate.write_bytes(raw)
            saved_path = "/" + str(candidate.relative_to(STATIC_DIR)).replace(os.sep, "/")
            if temp_id:
                uploaded_paths[temp_id] = saved_path
            else:
                photo_entries.append({"path": saved_path, "name": filename, "step_key": "", "step_label": ""})
        resolved_entries: list[dict[str, str]] = []
        for entry in photo_entries:
            path = str(entry.get("path", "")).strip()
            temp_id = str(entry.get("temp_id", "")).strip()
            if not path and temp_id:
                path = uploaded_paths.get(temp_id, "")
            if not path:
                continue
            resolved_entries.append(
                {
                    "path": path,
                    "name": str(entry.get("name", "")).strip() or os.path.basename(path),
                    "step_key": str(entry.get("step_key", "")).strip(),
                    "step_label": str(entry.get("step_label", "")).strip(),
                }
            )
        photo_entries = resolved_entries
    photo_entries = [
        entry
        for entry in parseBuilderPhotoEntries(json.dumps(photo_entries, ensure_ascii=False))
        if str(entry.get("path", "")).strip()
    ]
    base_row["Task_ID"] = task_id
    base_row["Component"] = component
    base_row["Task"] = task
    base_row["Task_Group"] = str(payload.get("taskGroup", base_row.get("Task_Group", ""))).strip()
    base_row["Required_Parts"] = str(payload.get("requiredParts", base_row.get("Required_Parts", ""))).strip()
    base_row["Preparation_Checklist"] = str(payload.get("preparationChecklist", "")).strip()
    base_row["Safety_Protocol"] = str(payload.get("safetyProtocol", "")).strip()
    base_row["Safety_Fall_Risk"] = str(payload.get("safetyFallRisk", "")).strip()
    base_row["Safety_TnM_Presence"] = str(payload.get("safetyTnmPresence", "")).strip()
    base_row["Procedure_Steps"] = str(payload.get("procedureSteps", "")).strip()
    base_row["Procedure_Photos"] = json.dumps(photo_entries, ensure_ascii=False)
    base_row["Sanity_Checklist"] = str(payload.get("sanityChecklist", "")).strip()
    base_row["Sanity_Results"] = str(payload.get("sanityResults", "")).strip()
    base_row["Draw_Stop_Plan"] = str(payload.get("drawStopPlan", "")).strip()
    base_row["Est_Stop_Min"] = str(payload.get("estStopMin", "")).strip()
    base_row["Completion_Criteria"] = str(payload.get("completionCriteria", "")).strip()
    base_row["Supplier_Name"] = str(payload.get("supplierName", "")).strip()
    base_row["Supplier_Details"] = str(payload.get("supplierDetails", "")).strip()
    base_row["Last_Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_row["Updated_By"] = "rebuild"
    if index is None:
        rows.append(base_row)
    else:
        rows[index] = base_row
    write_csv_rows(path, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Maintenance work package saved.", "bootstrap": build_bootstrap_payload().body})


def update_inventory_stock_action(payload: dict) -> JsonResponse:
    rows = read_csv_rows(PARTS_INVENTORY)
    fieldnames = read_csv_fieldnames(PARTS_INVENTORY)
    if not fieldnames:
        fieldnames = ["Part Name", "Item Type", "Component", "Supplier", "Serial Number", "Location", "Location Serial", "Quantity", "Min Level", "Notes", "Last Updated"]
    elif "Supplier" not in fieldnames:
        insert_at = fieldnames.index("Component") + 1 if "Component" in fieldnames else len(fieldnames)
        fieldnames = fieldnames[:insert_at] + ["Supplier"] + fieldnames[insert_at:]
    part_name = str(payload.get("partName", "")).strip()
    if not part_name:
        return JsonResponse({"ok": False, "message": "Part name is required."}, 400)
    serial = str(payload.get("serialNumber", "")).strip()
    mode = str(payload.get("mode", "add")).strip()
    qty = max(0.0 if mode == "edit" else 0.01, to_float(str(payload.get("quantity", "1"))))
    location = str(payload.get("location", "")).strip()
    item_type = str(payload.get("itemType", "Part")).strip() or "Part"
    component = str(payload.get("component", "Tower Parts")).strip() or "Tower Parts"
    supplier = str(payload.get("supplier", "")).strip()
    min_level = str(payload.get("minLevel", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    location_serial_map = {
        str(row.get("Location Name", "")).strip(): str(row.get("Location Serial", "")).strip()
        for row in read_csv_rows(PARTS_LOCATIONS)
        if str(row.get("Location Name", "")).strip()
    }
    now_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_index = None
    for index, row in enumerate(rows):
        if str(row.get("Part Name", "")).strip().lower() != part_name.lower():
            continue
        if str(row.get("Serial Number", "")).strip().lower() != serial.lower():
            continue
        if mode == "use":
            match_index = index
            break
        existing_location = str(row.get("Location", "")).strip().lower()
        desired_location = location.lower()
        if existing_location == desired_location:
            match_index = index
            break
    if mode == "add":
        if match_index is None:
            rows.append(
                {
                    "Part Name": part_name,
                    "Item Type": item_type,
                    "Component": component,
                    "Supplier": supplier,
                    "Serial Number": serial,
                    "Location": location,
                    "Location Serial": location_serial_map.get(location, "MOUNTED" if location.lower() == "mounted" else ""),
                    "Quantity": f"{qty:g}",
                    "Min Level": min_level or "0",
                    "Notes": notes or "Quick inventory add",
                    "Last Updated": now_label,
                }
            )
        else:
            current_qty = to_float(rows[match_index].get("Quantity"))
            rows[match_index]["Quantity"] = f"{current_qty + qty:g}"
            rows[match_index]["Item Type"] = item_type
            rows[match_index]["Component"] = component
            rows[match_index]["Supplier"] = supplier if supplier else str(rows[match_index].get("Supplier", "")).strip()
            rows[match_index]["Location"] = location
            rows[match_index]["Location Serial"] = location_serial_map.get(location, "MOUNTED" if location.lower() == "mounted" else rows[match_index].get("Location Serial", ""))
            if min_level:
                rows[match_index]["Min Level"] = min_level
            if notes:
                rows[match_index]["Notes"] = notes
            rows[match_index]["Last Updated"] = now_label
        write_csv_rows(PARTS_INVENTORY, rows, fieldnames)
        return JsonResponse({"ok": True, "message": "Inventory stock increased.", "bootstrap": build_bootstrap_payload().body})
    if mode == "new":
        for row in rows:
            if str(row.get("Part Name", "")).strip().lower() != part_name.lower():
                continue
            if str(row.get("Serial Number", "")).strip().lower() != serial.lower():
                continue
            if str(row.get("Location", "")).strip().lower() != location.lower():
                continue
            return JsonResponse({"ok": False, "message": "Inventory row already exists. Use edit or add stock instead."}, 400)
        rows.append(
            {
                "Part Name": part_name,
                "Item Type": item_type,
                "Component": component,
                "Supplier": supplier,
                "Serial Number": serial,
                "Location": location,
                "Location Serial": location_serial_map.get(location, "MOUNTED" if location.lower() == "mounted" else ""),
                "Quantity": f"{qty:g}",
                "Min Level": min_level or "0",
                "Notes": notes or "New inventory row",
                "Last Updated": now_label,
            }
        )
        write_csv_rows(PARTS_INVENTORY, rows, fieldnames)
        return JsonResponse({"ok": True, "message": "Inventory row created.", "bootstrap": build_bootstrap_payload().body})
    if mode == "edit":
        try:
            edit_index = int(payload.get("inventoryEditIndex", ""))
        except (TypeError, ValueError):
            return JsonResponse({"ok": False, "message": "Choose which inventory row to edit."}, 400)
        if edit_index < 0 or edit_index >= len(rows):
            return JsonResponse({"ok": False, "message": "Inventory row was not found."}, 400)
        current_row = rows[edit_index]
        rows[edit_index]["Part Name"] = part_name or str(current_row.get("Part Name", "")).strip()
        rows[edit_index]["Item Type"] = item_type or str(current_row.get("Item Type", "")).strip() or "Part"
        rows[edit_index]["Component"] = component or str(current_row.get("Component", "")).strip() or "Tower Parts"
        rows[edit_index]["Supplier"] = supplier if supplier else str(current_row.get("Supplier", "")).strip()
        rows[edit_index]["Serial Number"] = serial
        rows[edit_index]["Location"] = location
        rows[edit_index]["Location Serial"] = location_serial_map.get(location, "MOUNTED" if location.lower() == "mounted" else str(current_row.get("Location Serial", "")).strip())
        rows[edit_index]["Quantity"] = f"{qty:g}"
        rows[edit_index]["Min Level"] = min_level if min_level != "" else str(current_row.get("Min Level", "")).strip() or "0"
        rows[edit_index]["Notes"] = notes if notes else str(current_row.get("Notes", "")).strip()
        rows[edit_index]["Last Updated"] = now_label
        write_csv_rows(PARTS_INVENTORY, rows, fieldnames)
        return JsonResponse({"ok": True, "message": "Inventory row updated.", "bootstrap": build_bootstrap_payload().body})
    if match_index is None:
        return JsonResponse({"ok": False, "message": "Part was not found in inventory."}, 400)
    current_qty = to_float(rows[match_index].get("Quantity"))
    remaining = max(0.0, current_qty - qty)
    rows[match_index]["Quantity"] = f"{remaining:g}"
    rows[match_index]["Last Updated"] = now_label
    rows[match_index]["Notes"] = notes or "Quick inventory use"
    write_csv_rows(PARTS_INVENTORY, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Inventory stock decreased.", "bootstrap": build_bootstrap_payload().body})


def sync_part_order_into_inventory(row: dict[str, str], payload: dict) -> None:
    rows = read_csv_rows(PARTS_INVENTORY)
    fieldnames = read_csv_fieldnames(PARTS_INVENTORY)
    if not fieldnames:
        fieldnames = ["Part Name", "Item Type", "Component", "Supplier", "Serial Number", "Location", "Location Serial", "Quantity", "Min Level", "Notes", "Last Updated"]
    elif "Supplier" not in fieldnames:
        insert_at = fieldnames.index("Component") + 1 if "Component" in fieldnames else len(fieldnames)
        fieldnames = fieldnames[:insert_at] + ["Supplier"] + fieldnames[insert_at:]
    location_serial_map = {
        str(item.get("Location Name", "")).strip(): str(item.get("Location Serial", "")).strip()
        for item in read_csv_rows(PARTS_LOCATIONS)
        if str(item.get("Location Name", "")).strip()
    }
    inventory_action = str(payload.get("inventoryAction", "")).strip()
    if inventory_action == "Locate in inventory":
        location = str(payload.get("inventoryLocation", "")).strip()
    elif inventory_action == "Mount on machine":
        location = str(payload.get("inventoryLocation", "")).strip() or "Mounted"
    else:
        location = ""
    if not location:
        return
    part_name = str(payload.get("partName", row.get("Part Name", ""))).strip()
    serial = str(payload.get("serialNumber", row.get("Serial Number", ""))).strip()
    quantity = max(0.01, to_float(str(payload.get("inventoryQuantity", "1"))))
    item_type = str(payload.get("inventoryItemType", "")).strip() or "Part"
    component = str(payload.get("maintenanceComponent", row.get("Maintenance Component", ""))).strip() or "Tower Parts"
    supplier = str(payload.get("inventorySupplier", row.get("Company", ""))).strip()
    min_level = str(payload.get("inventoryMinLevel", "")).strip() or "0"
    notes = str(payload.get("inventoryNotes", "")).strip() or f"Auto from received order ({inventory_action})"
    now_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_index = None
    for index, inv_row in enumerate(rows):
        if str(inv_row.get("Part Name", "")).strip().lower() != part_name.lower():
            continue
        if str(inv_row.get("Serial Number", "")).strip().lower() != serial.lower():
            continue
        if str(inv_row.get("Location", "")).strip().lower() != location.lower():
            continue
        match_index = index
        break
    if match_index is None:
        rows.append(
            {
                "Part Name": part_name,
                "Item Type": item_type,
                "Component": component,
                "Supplier": supplier,
                "Serial Number": serial,
                "Location": location,
                "Location Serial": location_serial_map.get(location, "MOUNTED" if location.lower() == "mounted" else ""),
                "Quantity": f"{quantity:g}",
                "Min Level": min_level,
                "Notes": notes,
                "Last Updated": now_label,
            }
        )
    else:
        current_qty = to_float(rows[match_index].get("Quantity"))
        rows[match_index]["Quantity"] = f"{current_qty + quantity:g}"
        rows[match_index]["Item Type"] = item_type
        rows[match_index]["Component"] = component
        rows[match_index]["Supplier"] = supplier if supplier else str(rows[match_index].get("Supplier", "")).strip()
        rows[match_index]["Location"] = location
        rows[match_index]["Location Serial"] = location_serial_map.get(location, "MOUNTED" if location.lower() == "mounted" else rows[match_index].get("Location Serial", ""))
        rows[match_index]["Notes"] = notes
        rows[match_index]["Last Updated"] = now_label
        if min_level:
            rows[match_index]["Min Level"] = min_level
    write_csv_rows(PARTS_INVENTORY, rows, fieldnames)


def delete_part_order_action(payload: dict) -> JsonResponse:
    rows = read_csv_rows(PART_ORDERS)
    fieldnames = read_csv_fieldnames(PART_ORDERS)
    try:
        index = int(payload.get("index"))
    except (TypeError, ValueError):
        return JsonResponse({"ok": False, "message": "Order selection is invalid."}, 400)
    if index < 0 or index >= len(rows):
        return JsonResponse({"ok": False, "message": "Order not found."}, 400)
    del rows[index]
    write_csv_rows(PART_ORDERS, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Part order deleted.", "bootstrap": build_bootstrap_payload().body})


def set_maintenance_state_action(payload: dict) -> JsonResponse:
    task_id = str(payload.get("taskId", "")).strip()
    component = str(payload.get("component", "")).strip()
    task = str(payload.get("task", "")).strip()
    state = str(payload.get("state", "")).strip()
    note = str(payload.get("note", "")).strip()
    if not task_id or not state:
      return JsonResponse({"ok": False, "message": "Task and state are required."}, 400)
    rows = read_csv_rows(MAINTENANCE_STATE)
    fieldnames = read_csv_fieldnames(MAINTENANCE_STATE) or ["task_key", "task_id", "component", "task", "state", "updated_ts", "updated_by", "note"]
    task_key = f"{task_id.lower()}::{component.lower()}::{task.lower()}"
    updated = False
    for row in rows:
        if str(row.get("task_id", "")).strip() == task_id:
            row["task_key"] = task_key
            row["component"] = component
            row["task"] = task
            row["state"] = state
            row["updated_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row["updated_by"] = "rebuild"
            row["note"] = note
            updated = True
            break
    if not updated:
        rows.append(
            {
                "task_key": task_key,
                "task_id": task_id,
                "component": component,
                "task": task,
                "state": state,
                "updated_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_by": "rebuild",
                "note": note,
            }
        )
    write_csv_rows(MAINTENANCE_STATE, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Maintenance state updated.", "bootstrap": build_bootstrap_payload().body})


def complete_maintenance_task_action(payload: dict) -> JsonResponse:
    task_id = str(payload.get("taskId", "")).strip()
    component = str(payload.get("component", "")).strip()
    task = str(payload.get("task", "")).strip()
    mode = str(payload.get("trackingMode", "")).strip()
    note = str(payload.get("note", "")).strip()
    if not task_id:
        return JsonResponse({"ok": False, "message": "Task is required."}, 400)
    rows = read_csv_rows(MAINTENANCE_ACTIONS)
    fieldnames = read_csv_fieldnames(MAINTENANCE_ACTIONS) or [
        "maintenance_id", "maintenance_ts", "maintenance_component", "maintenance_task", "maintenance_task_id",
        "maintenance_mode", "maintenance_hours_source", "maintenance_done_date", "maintenance_done_hours",
        "maintenance_done_draw", "maintenance_source_file", "maintenance_actor", "maintenance_note",
    ]
    rows.append(
        {
            "maintenance_id": str(int(datetime.now().timestamp() * 1000)),
            "maintenance_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "maintenance_component": component,
            "maintenance_task": task,
            "maintenance_task_id": task_id,
            "maintenance_mode": mode,
            "maintenance_hours_source": "",
            "maintenance_done_date": datetime.now().strftime("%Y-%m-%d"),
            "maintenance_done_hours": "",
            "maintenance_done_draw": "",
            "maintenance_source_file": "",
            "maintenance_actor": "rebuild",
            "maintenance_note": note,
        }
    )
    write_csv_rows(MAINTENANCE_ACTIONS, rows, fieldnames)
    set_maintenance_state_action({"taskId": task_id, "component": component, "task": task, "state": "DONE_NOW", "note": note})
    return JsonResponse({"ok": True, "message": "Maintenance action logged.", "bootstrap": build_bootstrap_payload().body})


def create_maintenance_part_orders_action(payload: dict) -> JsonResponse:
    task_entries = payload.get("tasks")
    if isinstance(task_entries, list) and task_entries:
        normalized_tasks = [
            {
                "task_id": str(item.get("taskId", "")).strip(),
                "component": str(item.get("component", "")).strip(),
                "task": str(item.get("task", "")).strip(),
                "parts": item.get("parts", []) or [],
            }
            for item in task_entries
            if str(item.get("taskId", "")).strip()
        ]
    else:
        task_id = str(payload.get("taskId", "")).strip()
        component = str(payload.get("component", "")).strip()
        task = str(payload.get("task", "")).strip()
        parts = payload.get("parts", []) or []
        normalized_tasks = [{
            "task_id": task_id,
            "component": component,
            "task": task,
            "parts": parts,
        }] if task_id else []

    if not normalized_tasks:
        return JsonResponse({"ok": False, "message": "Task and parts are required."}, 400)
    rows = read_csv_rows(PART_ORDERS)
    fieldnames = read_csv_fieldnames(PART_ORDERS)
    active_status = {"Opened", "Wait for Approval", "Approved", "Ordered", "Received"}
    created = 0
    touched_tasks: list[tuple[str, str, str]] = []
    for task_entry in normalized_tasks:
        task_id = task_entry["task_id"]
        component = task_entry["component"]
        task = task_entry["task"]
        parts = task_entry["parts"]
        if not task_id or not parts:
            continue
        for part_name in parts:
            part = str(part_name or "").strip()
            if not part:
                continue
            exists = any(
                str(row.get("Part Name", "")).strip().lower() == part.lower()
                and str(row.get("Maintenance Task ID", "")).strip() == task_id
                and str(row.get("Status", "")).strip() in active_status
                for row in rows
            )
            if exists:
                continue
            rows.append(
                {
                    "Status": "Opened",
                    "Part Name": part,
                    "Serial Number": "",
                    "Project Name": "Maintenance",
                    "Details": f"Maintenance hold: {component} — {task} (Task ID: {task_id}).",
                    "Opened By": "rebuild",
                    "Approval Requested From": "",
                    "Approved": "No",
                    "Approved By": "",
                    "Approval Date": "",
                    "Received Date": "",
                    "Received State": "",
                    "Ordered By": "",
                    "Date Ordered": "",
                    "Company": "",
                    "Inventory Synced": "",
                    "Maintenance Component": component,
                    "Maintenance Task": task,
                    "Maintenance Task ID": task_id,
                    "Wait ID": "",
                }
            )
            created += 1
        touched_tasks.append((task_id, component, task))
    write_csv_rows(PART_ORDERS, rows, fieldnames)
    for task_id, component, task in touched_tasks:
        set_maintenance_state_action({
            "taskId": task_id,
            "component": component,
            "task": task,
            "state": "WAIT FOR PART",
            "note": "Parts ordered from prep horizon.",
        })
    return JsonResponse({"ok": True, "message": f"Created {created} maintenance part order(s).", "bootstrap": build_bootstrap_payload().body})


def schedule_maintenance_tasks_action(payload: dict) -> JsonResponse:
    event_type = str(payload.get("eventType", "")).strip() or "Maintenance"
    recurrence = normalize_recurrence(payload.get("recurrence", "")) or "none"
    task_entries = payload.get("tasks")
    window_entries = payload.get("windows")
    normalized_windows = []
    if isinstance(window_entries, list) and window_entries:
        normalized_windows = [
            {
                "start": str(item.get("start", "")).strip(),
                "end": str(item.get("end", "")).strip(),
                "label": str(item.get("label", "")).strip(),
            }
            for item in window_entries
            if str(item.get("start", "")).strip() and str(item.get("end", "")).strip()
        ]
    else:
        start = str(payload.get("start", "")).strip()
        end = str(payload.get("end", "")).strip()
        if start and end:
            normalized_windows = [{
                "start": start,
                "end": end,
                "label": str(payload.get("label", "")).strip(),
            }]
    if not normalized_windows:
        return JsonResponse({"ok": False, "message": "Schedule window is required."}, 400)
    if isinstance(task_entries, list) and task_entries:
        normalized_tasks = [
            {
                "task_id": str(item.get("taskId", "")).strip(),
                "component": str(item.get("component", "")).strip(),
                "task": str(item.get("task", "")).strip(),
            }
            for item in task_entries
            if str(item.get("taskId", "")).strip()
        ]
    else:
        task_id = str(payload.get("taskId", "")).strip()
        normalized_tasks = [{
            "task_id": task_id,
            "component": str(payload.get("component", "")).strip(),
            "task": str(payload.get("task", "")).strip(),
        }] if task_id else []
    if not normalized_tasks:
        return JsonResponse({"ok": False, "message": "Task is required."}, 400)

    rows = read_csv_rows(TOWER_SCHEDULE)
    fieldnames = read_csv_fieldnames(TOWER_SCHEDULE) or SCHEDULE_REQUIRED_COLS[:]
    created = 0
    for index, task_entry in enumerate(normalized_tasks):
        task_id = task_entry["task_id"]
        component = task_entry["component"]
        task = task_entry["task"]
        selected_window = normalized_windows[index % len(normalized_windows)]
        start = selected_window["start"]
        end = selected_window["end"]
        scheduled_for_label = selected_window["label"] or (parse_dt(start).strftime("%b %d %H:%M") if parse_dt(start) else start)
        description = f"Maintenance scheduled: {component} — {task} (Task ID: {task_id})."
        exists = any(
            str(row.get("Start DateTime", "")).strip() == start
            and f"Task ID: {task_id}" in str(row.get("Description", ""))
            for row in rows
        )
        if not exists:
            rows.append(
                {
                    "Event Type": event_type,
                    "Start DateTime": start,
                    "End DateTime": end,
                    "Description": description,
                    "Recurrence": recurrence,
                }
            )
            created += 1
        set_maintenance_state_action({
            "taskId": task_id,
            "component": component,
            "task": task,
            "state": "SCHEDULED",
            "note": f"Scheduled from prep horizon for {scheduled_for_label}.",
        })
    write_csv_rows(TOWER_SCHEDULE, rows, fieldnames)
    return JsonResponse({"ok": True, "message": f"Scheduled {len(normalized_tasks)} maintenance task(s).", "bootstrap": build_bootstrap_payload().body})


def unmount_inventory_item_action(payload: dict) -> JsonResponse:
    rows = read_csv_rows(PARTS_INVENTORY)
    fieldnames = read_csv_fieldnames(PARTS_INVENTORY)
    part_name = str(payload.get("partName", "")).strip()
    serial = str(payload.get("serialNumber", "")).strip()
    qty = max(0.01, to_float(str(payload.get("quantity", "1"))))
    if not part_name:
        return JsonResponse({"ok": False, "message": "Mounted part is required."}, 400)
    match_index = None
    for index, row in enumerate(rows):
        if str(row.get("Part Name", "")).strip().lower() != part_name.lower():
            continue
        if str(row.get("Location", "")).strip().lower() != "mounted":
            continue
        if serial and str(row.get("Serial Number", "")).strip().lower() != serial.lower():
            continue
        match_index = index
        break
    if match_index is None:
        return JsonResponse({"ok": False, "message": "Mounted row was not found."}, 400)
    current_qty = to_float(rows[match_index].get("Quantity"))
    remaining = max(0.0, current_qty - qty)
    rows[match_index]["Quantity"] = f"{remaining:g}"
    rows[match_index]["Last Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows[match_index]["Notes"] = "Unmounted from machine"
    if remaining == 0:
        rows[match_index]["Location"] = ""
        rows[match_index]["Location Serial"] = ""
        if not str(rows[match_index].get("Component", "")).strip():
            rows[match_index]["Component"] = "Tower Parts"
    write_csv_rows(PARTS_INVENTORY, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Mounted inventory updated.", "bootstrap": build_bootstrap_payload().body})


def list_recent_files(directory: Path, suffixes: tuple[str, ...] = (".csv",), limit: int = 6) -> list[dict]:
    if not directory.exists() or not directory.is_dir():
        return []
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    output = []
    for path in files[:limit]:
        stat = path.stat()
        output.append(
            {
                "name": path.name,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "size_kb": round(stat.st_size / 1024, 1),
            }
        )
    return output


def list_csv_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv"], key=lambda p: p.stat().st_mtime, reverse=True)


def analyze_log_file(log_name: str | None = None, sample_limit: int = 1600) -> dict:
    files = list_csv_files(LOGS_DIR)
    if not files:
        return {
            "selected_file": "",
            "available_logs": [],
            "x_options": [],
            "numeric_columns": [],
            "rows": [],
            "sample_count": 0,
            "total_rows": 0,
        }
    by_name = {path.name: path for path in files}
    selected_path = by_name.get(str(log_name or "").strip(), files[0])
    rows = read_csv_rows(selected_path)
    if not rows:
        return {
            "selected_file": selected_path.name,
            "available_logs": [path.name for path in files],
            "x_options": [],
            "numeric_columns": [],
            "rows": [],
            "sample_count": 0,
            "total_rows": 0,
        }

    def normalize_x_value(raw_value: object, fallback_index: int) -> tuple[float, str]:
        text_value = str(raw_value or "").strip()
        if not text_value:
            return float(fallback_index), "index"
        parsed_dt = parse_dt(text_value)
        if parsed_dt:
            return parsed_dt.timestamp(), "datetime"
        for pattern in ("%d/%m/%Y %H:%M:%S%f", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text_value, pattern).timestamp(), "datetime"
            except ValueError:
                continue
        numeric_value = to_float(text_value)
        if numeric_value is not None:
            return float(numeric_value), "numeric"
        return float(fallback_index), "index"

    headers = list(rows[0].keys())
    numeric_columns = []
    for column in headers:
        non_empty = [row.get(column, "") for row in rows[:400] if str(row.get(column, "")).strip()]
        if not non_empty:
            continue
        numeric_count = sum(1 for value in non_empty if str(value).replace(".", "", 1).replace("-", "", 1).isdigit())
        if numeric_count / max(1, len(non_empty)) >= 0.75:
            numeric_columns.append(column)

    sampled_rows = rows[:sample_limit]
    suggested_x = "Date/Time" if "Date/Time" in headers else ("Timestamp" if "Timestamp" in headers else headers[0])
    compact_rows = []
    x_series: dict[str, list[float]] = {column: [] for column in headers}
    x_display: dict[str, list[str]] = {column: [] for column in headers}
    x_kinds: dict[str, str] = {}
    for index, row in enumerate(sampled_rows):
        item = {"__index": index, "__x": str(row.get(suggested_x, "") or index)}
        for column in numeric_columns:
            item[column] = to_float(row.get(column))
        for column in headers:
            normalized_value, detected_kind = normalize_x_value(row.get(column, ""), index)
            x_series[column].append(normalized_value)
            x_display[column].append(str(row.get(column, "") or ""))
            x_kinds[column] = detected_kind if x_kinds.get(column) in (None, "index") else x_kinds[column]
            if x_kinds.get(column) is None:
                x_kinds[column] = detected_kind
            elif x_kinds[column] == "index" and detected_kind != "index":
                x_kinds[column] = detected_kind
        compact_rows.append(item)

    suggested_y = [column for column in numeric_columns if column != suggested_x][:4]
    length_candidates = [column for column in headers if "length" in str(column).lower()]
    return {
        "selected_file": selected_path.name,
        "available_logs": [path.name for path in files],
        "x_options": headers,
        "numeric_columns": numeric_columns,
        "rows": compact_rows,
        "x_series": x_series,
        "x_display": x_display,
        "x_kinds": x_kinds,
        "sample_count": len(compact_rows),
        "total_rows": len(rows),
        "suggested_x": suggested_x,
        "suggested_y": suggested_y,
        "latest_modified": datetime.fromtimestamp(selected_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "length_column": length_candidates[0] if length_candidates else "",
    }


def load_coating_options() -> list[str]:
    config = read_json_dict(COATING_CONFIG)
    coatings = config.get("coatings", {})
    if not isinstance(coatings, dict):
        return []
    return dedupe_strings(list(coatings.keys()))


def sort_order_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def sort_key(row: dict[str, str]) -> datetime:
        return parse_dt(row.get("Timestamp")) or datetime.min

    return sorted(rows, key=sort_key, reverse=True)


def fmt_pm(value: str | None, tol: str | None, unit: str = "µm") -> str:
    num = to_float(value)
    tol_num = to_float(tol)
    if num <= 0:
        return ""
    if tol_num > 0:
        return f"{num:g} ± {tol_num:g} {unit}"
    return f"{num:g} {unit}"


def format_order_for_ui(row: dict[str, str], index: int) -> dict:
    geometry = (row.get(GEOMETRY_COL) or "Unspecified").strip() or "Unspecified"
    status = (row.get("Status") or "Unknown").strip() or "Unknown"
    return {
        "index": index,
        "preform": row.get("Preform Number", ""),
        "project": row.get(PROJECTS_COL, ""),
        "opener": row.get("Order Opener", ""),
        "status": status,
        "priority": row.get("Priority", "Normal"),
        "timestamp": row.get("Timestamp", ""),
        "next_draw": row.get("Next Planned Draw Date", "") or row.get("Desired Date", ""),
        "desired_date": row.get("Desired Date", ""),
        "geometry": geometry,
        "length": row.get(LENGTH_COL, "") or "0",
        "good_zones": row.get(GOOD_ZONES_COL, "") or "0",
        "notes": row.get("Notes", ""),
        "main_coating": row.get("Main Coating", ""),
        "secondary_coating": row.get("Secondary Coating", ""),
        "main_temp": row.get(MAIN_TEMP_COL, ""),
        "secondary_temp": row.get(SECONDARY_TEMP_COL, ""),
        "fiber_spec": fmt_pm(row.get("Fiber Diameter (µm)"), row.get(FIBER_TOL_COL)),
        "main_spec": fmt_pm(row.get("Main Coating Diameter (µm)"), row.get(MAIN_TOL_COL)),
        "secondary_spec": fmt_pm(row.get("Secondary Coating Diameter (µm)"), row.get(SECONDARY_TOL_COL)),
        "tiger_cut": row.get(TIGER_CUT_COL, ""),
        "oct_f2f": row.get(OCT_F2F_COL, ""),
    }


def build_schedule_description(order: dict[str, str], order_index: int, preform_number: str) -> str:
    geometry = str(order.get(GEOMETRY_COL, "")).strip()
    priority = str(order.get("Priority", "Normal")).strip()
    selected_project = str(order.get(PROJECTS_COL, "")).strip()
    length_required = str(order.get(LENGTH_COL, "")).strip()
    good_zones = str(order.get(GOOD_ZONES_COL, "")).strip()
    desc_lines = [
        f"ORDER #{order_index} | Priority: {priority}",
        f"Fiber: {selected_project} | Geometry: {geometry} | Preform: {preform_number}",
        f"Required Length: {length_required} m | Good Zones Count: {good_zones}",
    ]
    diameter_bits = []
    fiber_spec = fmt_pm(order.get("Fiber Diameter (µm)"), order.get(FIBER_TOL_COL))
    main_spec = fmt_pm(order.get("Main Coating Diameter (µm)"), order.get(MAIN_TOL_COL))
    secondary_spec = fmt_pm(order.get("Secondary Coating Diameter (µm)"), order.get(SECONDARY_TOL_COL))
    if fiber_spec:
        diameter_bits.append(f"Fiber {fiber_spec}")
    if main_spec:
        diameter_bits.append(f"Coat1 {main_spec}")
    if secondary_spec:
        diameter_bits.append(f"Coat2 {secondary_spec}")
    if diameter_bits:
        desc_lines.append("Diameters: " + " | ".join(diameter_bits))
    if geometry == "TIGER - PM" and to_float(order.get(TIGER_CUT_COL)) > 0:
        desc_lines.append(f"Tiger Cut: {to_float(order.get(TIGER_CUT_COL)):.1f}%")
    if geometry == "Octagonal" and to_float(order.get(OCT_F2F_COL)) > 0:
        desc_lines.append(f"Oct F2F: {to_float(order.get(OCT_F2F_COL)):.2f} mm")
    if to_float(order.get(MAIN_TEMP_COL)) > 0:
        desc_lines.append(f"Main Coat Temp: {to_float(order.get(MAIN_TEMP_COL)):.0f}°C")
    if to_float(order.get(SECONDARY_TEMP_COL)) > 0:
        desc_lines.append(f"Sec Coat Temp: {to_float(order.get(SECONDARY_TEMP_COL)):.0f}°C")
    notes = str(order.get("Notes", "")).strip()
    if notes:
        desc_lines.append(f"Notes: {notes}")
    return " | ".join([line for line in desc_lines if line.strip()])


def summarize_order_draw() -> dict:
    raw_orders = sort_order_rows(read_csv_rows(DRAW_ORDERS))
    orders = [format_order_for_ui(row, index) for index, row in enumerate(raw_orders)]
    projects = dedupe_strings([row.get(PROJECTS_COL, "") for row in read_csv_rows(PROJECTS_FIBER)])
    templates = read_csv_rows(PROJECTS_TEMPLATES)
    templates_by_project = {}
    for row in templates:
        project = str(row.get(PROJECTS_COL, "")).strip()
        if project:
            templates_by_project[project] = {field: row.get(field, "") for field in TEMPLATE_FIELDS}
    sap_items = read_csv_rows(SAP_RODS_INVENTORY)
    status_counts: dict[str, int] = {}
    geometry_counts: dict[str, int] = {}
    pending_orders = []
    scheduled_orders = []
    completed_orders = []
    for row in orders:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        geometry = row["geometry"]
        geometry_counts[geometry] = geometry_counts.get(geometry, 0) + 1
        if status == "Pending":
            pending_orders.append(row)
        elif status in {"Scheduled", "In Progress"}:
            scheduled_orders.append(row)
        else:
            completed_orders.append(row)
    sap_sets = next((row for row in sap_items if str(row.get("Item", "")).strip().lower() == "sap rods set"), sap_items[0] if sap_items else {})
    sap_count = to_float(sap_sets.get("Count"))
    return {
        "status_counts": status_counts,
        "geometry_series": [{"label": key, "value": value} for key, value in sorted(geometry_counts.items(), key=lambda item: item[1], reverse=True)[:6]],
        "all_orders": orders[:18],
        "pending_orders": pending_orders,
        "scheduled_orders": scheduled_orders,
        "completed_orders": completed_orders[:18],
        "queue_counts": {
            "pending": len(pending_orders),
            "scheduled": len(scheduled_orders),
            "history": len(completed_orders),
        },
        "active_count": len(pending_orders) + len(scheduled_orders),
        "project_count": len(projects),
        "template_count": len(templates),
        "sap_item_count": len(sap_items),
        "project_names": projects,
        "template_project_names": sorted(templates_by_project.keys()),
        "templates_by_project": templates_by_project,
        "coating_options": load_coating_options(),
        "form_config": {
            "geometry_options": ORDER_DRAW_GEOMETRY_OPTIONS,
        },
        "sap_summary": {
            "item": sap_sets.get("Item", "SAP Rods Set"),
            "count": sap_count,
            "units": sap_sets.get("Units", "sets"),
            "last_updated": sap_sets.get("Last Updated", ""),
            "notes": sap_sets.get("Notes", ""),
            "low": sap_count < 1,
        },
    }


def summarize_dashboard() -> dict:
    draws = summarize_draw_orders()
    schedule = summarize_schedule()
    parts = summarize_part_orders()
    inventory = summarize_inventory()
    log_csvs = list_csv_files(LOGS_DIR)
    dataset_csvs = list_csv_files(DATASET_DIR)
    dataset_files = list_recent_files(DATASET_DIR)
    log_files = list_recent_files(LOGS_DIR)
    report_files = list_recent_files(REPORTS_DIR, suffixes=(".csv", ".json", ".pdf", ".log"))
    return {
        "metrics": [
            {"label": "Dataset CSVs", "value": len(list(DATASET_DIR.glob("*.csv"))) if DATASET_DIR.exists() else 0},
            {"label": "Log CSVs", "value": len(list(LOGS_DIR.glob("*.csv"))) if LOGS_DIR.exists() else 0},
            {"label": "Active Draws", "value": draws["active"]},
            {"label": "Open Part Orders", "value": len(parts["open_orders"])},
        ],
        "draw_status_series": [{"label": key, "value": value} for key, value in draws["status_counts"].items()],
        "schedule_series": schedule["daily_series"],
        "inventory_series": inventory["pressure_series"],
        "dataset_files": dataset_files,
        "log_files": log_files,
        "report_files": report_files,
        "available_logs": [path.name for path in log_csvs],
        "latest_log": log_csvs[0].name if log_csvs else "",
        "dataset_csvs": [path.name for path in dataset_csvs[:18]],
        "latest_dataset": dataset_csvs[0].name if dataset_csvs else "",
    }


def summarize_diagnostics() -> dict:
    ensure_app_weekly_full_backup()
    tracked_paths = current_tracked_paths()
    defaults = tracked_path_defaults()
    path_rows = []
    ready_count = 0
    for item, path in tracked_paths:
        key = str(item["key"])
        label = str(item["label"])
        exists = path.exists()
        readable = exists and os.access(path, os.R_OK)
        writable_target = path if exists else path.parent
        writable = writable_target.exists() and os.access(writable_target, os.W_OK)
        healthy = exists and readable and writable
        ready_count += int(healthy)
        path_rows.append(
            {
                "key": key,
                "label": label,
                "kind": item["kind"],
                "status": "READY" if healthy else "BLOCKED",
                "exists": exists,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exists else "missing",
                "path": str(path),
                "default_path": str(defaults[key]),
                "is_override": path != defaults[key],
            }
        )
    schema_checks = [
        {"csv": "orders", "path": DRAW_ORDERS, "required": ("Status", "Priority", "Preform Number", "Fiber Project")},
        {"csv": "parts_orders", "path": PART_ORDERS, "required": ("Status", "Part Name", "Project Name")},
        {"csv": "schedule", "path": TOWER_SCHEDULE, "required": ("Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence")},
    ]
    schema_rows = []
    for check in schema_checks:
        rows = read_csv_rows(check["path"])
        header = rows[0].keys() if rows else []
        missing = [column for column in check["required"] if column not in header]
        schema_rows.append(
            {
                "csv": check["csv"],
                "ok": not missing,
                "rows": len(rows),
                "missing_columns": ", ".join(missing),
            }
        )
    dataset_files = list_csv_files(DATASET_DIR)
    log_files = list_csv_files(LOGS_DIR)
    path_ok = ready_count == len(tracked_paths)
    schema_ok = all(item["ok"] for item in schema_rows)
    backup_snapshots = len(list_backup_directories())
    full_backup_snapshots = len(list_backup_directories("full_backup_"))
    latest_full_backup = latest_backup_snapshot("full_backup_")
    report_file_count = len(list(REPORTS_DIR.rglob("*"))) if REPORTS_DIR.exists() else 0
    logs_path = next((item for item in path_rows if item["key"] == "logs_dir"), None)
    reports_path = next((item for item in path_rows if item["key"] == "reports_dir"), None)
    health_checks = [
        {
            "key": "paths",
            "label": "Required paths",
            "ok": path_ok,
            "detail": f"{ready_count}/{len(tracked_paths)} tracked paths are ready for read/write work.",
        },
        {
            "key": "schema",
            "label": "CSV schema",
            "ok": schema_ok,
            "detail": f"{sum(1 for item in schema_rows if item['ok'])}/{len(schema_rows)} required CSV structures are complete.",
        },
        {
            "key": "datasets",
            "label": "Dataset workspace",
            "ok": bool(dataset_files),
            "detail": f"{len(dataset_files)} dataset CSVs are available in the draw workspace.",
        },
        {
            "key": "logs",
            "label": "Logs workspace",
            "ok": bool(logs_path and logs_path["status"] == "READY") and bool(log_files),
            "detail": f"{len(log_files)} log CSVs are available in the logs workspace.",
        },
        {
            "key": "backups",
            "label": "Backup coverage",
            "ok": backup_snapshots > 0,
            "detail": f"{backup_snapshots} backup snapshots are available for recovery.",
        },
        {
            "key": "reports",
            "label": "Report output lane",
            "ok": bool(reports_path and reports_path["status"] == "READY"),
            "detail": f"{report_file_count} report files are reachable in the report center.",
        },
    ]
    passed_checks = sum(1 for item in health_checks if item["ok"])
    overall_ok = passed_checks == len(health_checks)
    return {
        "ready_count": ready_count,
        "tracked_count": len(tracked_paths),
        "path_rows": path_rows,
        "schema_rows": schema_rows,
        "backup_snapshots": backup_snapshots,
        "report_file_count": report_file_count,
        "dataset_count": len(dataset_files),
        "log_count": len(log_files),
        "override_count": sum(1 for item in path_rows if item["is_override"]),
        "full_backup_count": full_backup_snapshots,
        "latest_full_backup": latest_full_backup,
        "full_backup_policy_label": FULL_BACKUP_POLICY_LABEL,
        "health_checks": health_checks,
        "passed_checks": passed_checks,
        "total_checks": len(health_checks),
        "overall_ok": overall_ok,
        "overall_label": "All core diagnostics are green." if overall_ok else "Diagnostics need attention.",
        "overall_detail": (
            "Paths, schemas, datasets, logs, backups, and report output are all ready."
            if overall_ok
            else "At least one required diagnostics lane is blocked, missing, or incomplete."
        ),
    }


def latest_csv_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    return rows[-1] if rows else {}


def load_consumables_temp_setpoints(latest_temps: dict[str, str] | None = None) -> dict[str, float]:
    latest_temps = latest_temps or {}
    heater_map = read_json_dict(HEATER_CONFIG)
    setpoints: dict[str, float] = {}
    for field in CONSUMABLE_TEMP_FIELDS:
        default_value = round(to_float(latest_temps.get(field)), 2)
        config_key = CONSUMABLE_TEMP_SETPOINT_KEYS[field]
        csv_key = consumable_temp_setpoint_csv_field(field)
        raw_value = heater_map.get(
            config_key,
            latest_temps.get(csv_key, heater_map.get(field, default_value)),
        )
        setpoints[field] = round(to_float(raw_value), 2)
    return setpoints


def summarize_consumables() -> dict:
    latest_containers = latest_csv_row(TOWER_CONTAINERS)
    latest_temps = latest_csv_row(TOWER_TEMPS)
    stock_map = read_json_value(COATING_STOCK, {})
    dies_map = read_json_value(DIES_CONFIG, {})
    temp_setpoints = load_consumables_temp_setpoints(latest_temps)
    coating_config = read_json_value(COATING_CONFIG, {})
    argon_rows = read_csv_rows(ARGON_MONTHLY_REPORT)
    container_cards = []
    total_level = 0.0
    low_count = 0
    type_counts: dict[str, float] = {}
    for label in ["A", "B", "C", "D"]:
        level = to_float(latest_containers.get(f"{label}_level_kg"))
        kind = str(latest_containers.get(f"{label}_type", "")).strip() or str(read_json_dict(CONTAINER_CONFIG).get(label, {}).get("type", ""))
        total_level += level
        if level < 1.0:
            low_count += 1
        type_counts[kind or f"Container {label}"] = type_counts.get(kind or f"Container {label}", 0.0) + level
        container_cards.append(
            {
                "label": label,
                "level": level,
                "type": kind or "Unassigned",
                "low": level < 1.0,
                "updated_at": latest_containers.get("updated_at", ""),
            }
        )
    temp_holders = []
    for key, label in [
        ("die_holder_primary_c", "Primary holder"),
        ("die_holder_secondary_c", "Secondary holder"),
    ]:
        value = to_float(latest_temps.get(key))
        set_value = temp_setpoints.get(key, round(value, 2))
        if value or value == 0:
            tone = "bad" if value >= 40 else "warn" if value >= 32 else "info"
            temp_holders.append(
                {
                    "label": label,
                    "field": key,
                    "measured_value": round(value, 2),
                    "set_value": round(set_value, 2),
                    "offset": round(value - set_value, 2),
                    "tone": tone,
                }
            )
    temp_stations = []
    for label in ["A", "B", "C", "D"]:
        container_value = to_float(latest_temps.get(f"{label}_container_c"))
        pipe_value = to_float(latest_temps.get(f"{label}_pipe_c"))
        container_field = f"{label}_container_c"
        pipe_field = f"{label}_pipe_c"
        container_set_value = temp_setpoints.get(container_field, round(container_value, 2))
        pipe_set_value = temp_setpoints.get(pipe_field, round(pipe_value, 2))
        values = [value for value in [container_value, pipe_value] if value or value == 0]
        hottest = max(values) if values else 0.0
        tone = "bad" if hottest >= 40 else "warn" if hottest >= 32 else "info"
        temp_stations.append(
            {
                "label": label,
                "container_field": container_field,
                "pipe_field": pipe_field,
                "container_measured_value": round(container_value, 2),
                "container_set_value": round(container_set_value, 2),
                "container_offset": round(container_value - container_set_value, 2),
                "pipe_measured_value": round(pipe_value, 2),
                "pipe_set_value": round(pipe_set_value, 2),
                "pipe_offset": round(pipe_value - pipe_set_value, 2),
                "delta": round(abs(container_value - pipe_value), 2),
                "tone": tone,
            }
        )
    stock_rows = []
    for key, value in sorted((stock_map or {}).items(), key=lambda item: to_float(item[1]), reverse=True):
        numeric = round(to_float(value), 2)
        tone = "bad" if numeric <= 0 else "warn" if numeric < 1.0 else "good"
        stock_rows.append({"label": key, "value": numeric, "unit": "kg", "tone": tone})
    dies_rows = []
    for station, values in (dies_map or {}).items():
        values = values if isinstance(values, dict) else {}
        dies_rows.append(
            {
                "station": station,
                "entry_die_um": to_float(values.get("entry_die_um")),
                "primary_die_um": to_float(values.get("primary_die_um")),
                "primary_on_tower": bool(values.get("primary_on_tower")),
                "secondary_on_tower": bool(values.get("secondary_on_tower")),
            }
        )
    coating_rows = []
    coating_definitions = coating_config.get("coatings", {}) if isinstance(coating_config, dict) else {}
    for label, config in sorted((coating_definitions or {}).items()):
        config = config if isinstance(config, dict) else {}
        stock_kg = round(to_float((stock_map or {}).get(label)), 2)
        coating_rows.append(
            {
                "label": label,
                "description": str(config.get("Description", "")).strip() or "No operator note yet.",
                "density": config.get("Density", ""),
                "viscosity": config.get("Viscosity", ""),
                "refractive_index": config.get("Refractive Index", ""),
                "stock_kg": stock_kg,
                "tone": "bad" if stock_kg <= 0 else "warn" if stock_kg < 1.0 else "good",
            }
        )
    argon_series = [{"label": row.get("month", ""), "value": round(to_float(row.get("total_standard_liters")), 2)} for row in argon_rows[-8:]]
    low_stock_lines = sum(1 for row in stock_rows if row["value"] < 1.0)
    return {
        "metrics": [
            {"label": "Low containers", "value": low_count, "tone": "warn" if low_count else "good"},
            {"label": "Low stock lines", "value": low_stock_lines, "tone": "bad" if low_stock_lines else "good"},
            {"label": "Loaded coatings", "value": len([key for key in type_counts if key]), "tone": "info"},
            {"label": "Die stations", "value": len(dies_rows), "tone": "neutral"},
        ],
        "containers": container_cards,
        "stock_rows": stock_rows,
        "stock_by_type": [{"label": key, "value": round(value, 2)} for key, value in type_counts.items()],
        "temp_rows": temp_holders,
        "temp_holders": temp_holders,
        "temp_stations": temp_stations,
        "temps_updated_at": latest_temps.get("updated_at", ""),
        "dies_rows": dies_rows,
        "coating_rows": coating_rows,
        "argon_rows": argon_rows[-12:],
        "argon_series": argon_series,
    }


def save_consumables_dies_action(payload: dict) -> JsonResponse:
    dies_map = read_json_value(DIES_CONFIG, {})
    stations = payload.get("stations") or []
    if not isinstance(stations, list) or not stations:
        return JsonResponse({"ok": False, "message": "No die station values were supplied."}, 400)
    if not isinstance(dies_map, dict):
        dies_map = {}
    for item in stations:
        if not isinstance(item, dict):
            continue
        station = str(item.get("station", "")).strip()
        if not station:
            continue
        current = dies_map.get(station, {})
        if not isinstance(current, dict):
            current = {}
        current["entry_die_um"] = to_float(item.get("entry_die_um"))
        current["primary_die_um"] = to_float(item.get("primary_die_um"))
        dies_map[station] = current
    DIES_CONFIG.write_text(json.dumps(dies_map, indent=2), encoding="utf-8")
    return JsonResponse({"ok": True, "message": "Die setup saved.", "bootstrap": build_bootstrap_payload().body})


def save_consumables_temps_action(payload: dict) -> JsonResponse:
    raw_setpoints = payload.get("setpoints") or {}
    if not isinstance(raw_setpoints, dict) or not raw_setpoints:
        return JsonResponse({"ok": False, "message": "No temperature set values were supplied."}, 400)
    latest_temps = latest_csv_row(TOWER_TEMPS)
    merged = load_consumables_temp_setpoints(latest_temps)
    for field in CONSUMABLE_TEMP_FIELDS:
        if field in raw_setpoints:
            merged[field] = round(to_float(raw_setpoints.get(field)), 2)
    heater_map = read_json_value(HEATER_CONFIG, {})
    if not isinstance(heater_map, dict):
        heater_map = {}
    for field, config_key in CONSUMABLE_TEMP_SETPOINT_KEYS.items():
        heater_map[config_key] = merged[field]
    heater_map["die_holder_heater_temp_c"] = merged["die_holder_primary_c"]
    write_json_value(HEATER_CONFIG, heater_map)
    temp_rows = read_csv_rows(TOWER_TEMPS)
    temp_fieldnames = read_csv_fieldnames(TOWER_TEMPS)
    if temp_rows and temp_fieldnames:
        for field in CONSUMABLE_TEMP_FIELDS:
            csv_key = consumable_temp_setpoint_csv_field(field)
            if csv_key not in temp_fieldnames:
                temp_fieldnames.append(csv_key)
            temp_rows[-1][csv_key] = merged[field]
        write_csv_rows(TOWER_TEMPS, temp_rows, temp_fieldnames)
    return JsonResponse({"ok": True, "message": "Temperature set values saved.", "bootstrap": build_bootstrap_payload().body})


def summarize_process_setup() -> dict:
    order_draw = summarize_order_draw()
    selected_csv = read_json_dict(SELECTED_CSV_JSON).get("selected_csv", "")
    raw_rows = read_csv_rows(DRAW_ORDERS)
    latest_temps = latest_csv_row(TOWER_TEMPS)
    indexed_rows = [{"source_index": index, **row} for index, row in enumerate(raw_rows)]
    sorted_indexed_rows = sorted(indexed_rows, key=lambda item: parse_dt(item.get("Timestamp")) or datetime.min, reverse=True)

    def process_setup_schedule_key(item: dict[str, str]) -> tuple[int, datetime, datetime]:
        scheduled_dt = (
            parse_dt(item.get("Next Planned Draw Date"))
            or parse_dt(item.get("Desired Date"))
            or parse_dt(item.get("Timestamp"))
        )
        has_date = 0 if scheduled_dt else 1
        return (
            has_date,
            scheduled_dt or datetime.max,
            parse_dt(item.get("Timestamp")) or datetime.min,
        )

    scheduled_order_rows = sorted(
        [
            item for item in indexed_rows
            if str(item.get("Status", "")).strip() == "Scheduled"
        ],
        key=process_setup_schedule_key,
    )
    scheduled_orders = [
        format_order_for_ui(item, int(item.get("source_index", 0)))
        for item in scheduled_order_rows
    ][:12]
    in_progress_orders = [
        format_order_for_ui(item, int(item.get("source_index", 0)))
        for item in sorted_indexed_rows
        if str(item.get("Status", "")).strip() == "In Progress"
    ][:8]
    template_cards = []
    for project_name, template in list(order_draw["templates_by_project"].items())[:8]:
        template_cards.append(
            {
                "project": project_name,
                "geometry": template.get(GEOMETRY_COL, ""),
                "main_coating": template.get("Main Coating", ""),
                "secondary_coating": template.get("Secondary Coating", ""),
                "speed": template.get("Draw Speed (m/min)", ""),
                "tension": template.get("Tension (g)", ""),
            }
        )
    dataset_files = list_csv_files(DATASET_DIR)
    if selected_csv and not any(path.name == selected_csv for path in dataset_files):
        selected_csv = ""
    selected_csv = selected_csv or (dataset_files[0].name if dataset_files else "")
    latest_dataset_name = dataset_files[0].name if dataset_files else ""
    dataset_info = analyze_dataset_file(selected_csv) if selected_csv else {
        "selected_file": "",
        "preview_rows": [],
        "parameter_names": [],
        "groups": [],
        "sections": [],
        "family_counts": [],
        "row_count": 0,
        "numeric_count": 0,
        "latest_modified": "",
    }
    dataset_linked_order_index = find_order_index_by_dataset(selected_csv) if selected_csv else None
    process_rows = [
        row for row in dataset_info.get("preview_rows", [])
        if str(row.get("parameter_name", "")).startswith("Process__")
    ][:24]
    order_rows = [
        row for row in dataset_info.get("preview_rows", [])
        if str(row.get("parameter_name", "")).startswith("Order__")
    ][:16]
    process_map = {
        str(row.get("parameter_name", "")).replace("Process__", "", 1): row.get("value", "")
        for row in dataset_info.get("preview_rows", [])
        if str(row.get("parameter_name", "")).startswith("Process__")
    }
    order_map = {
        str(row.get("parameter_name", "")).replace("Order__", "", 1): row.get("value", "")
        for row in dataset_info.get("preview_rows", [])
        if str(row.get("parameter_name", "")).startswith("Order__")
    }
    preform_options = dedupe_strings(
        [row.get("Preform Number", "") for row in raw_rows]
        + [row.get("Preform Number", "") for row in read_csv_rows(PREFORM_INVENTORY)]
    )
    coating_cfg = read_json_dict(COATING_CONFIG)
    pid_defaults = read_json_dict(PID_CONFIG)
    coating_options = dedupe_strings(list((coating_cfg.get("coatings", {}) or {}).keys()))
    die_options = dedupe_strings(list((coating_cfg.get("dies", {}) or {}).keys()))
    holder_setpoints = load_consumables_temp_setpoints(latest_temps)
    temp_context = {
        "sampled_at": latest_temps.get("updated_at", ""),
        "primary_holder_mv_c": round(to_float(latest_temps.get("die_holder_primary_c")), 2),
        "secondary_holder_mv_c": round(to_float(latest_temps.get("die_holder_secondary_c")), 2),
        "primary_holder_sp_c": round(holder_setpoints.get("die_holder_primary_c", 0.0), 2),
        "secondary_holder_sp_c": round(holder_setpoints.get("die_holder_secondary_c", 0.0), 2),
    }
    return {
        "metrics": [
            {"label": "Scheduled orders", "value": len(scheduled_order_rows)},
        ],
        "scheduled_orders": scheduled_orders,
        "in_progress_orders": in_progress_orders,
        "template_cards": template_cards,
        "selected_csv": selected_csv,
        "dataset_files": [path.name for path in dataset_files],
        "dataset_info": {
            "selected_file": dataset_info.get("selected_file", ""),
            "row_count": dataset_info.get("row_count", 0),
            "numeric_count": dataset_info.get("numeric_count", 0),
            "latest_modified": dataset_info.get("latest_modified", ""),
            "group_count": len(dataset_info.get("groups", [])),
            "section_count": len(dataset_info.get("sections", [])),
            "process_rows": process_rows,
            "order_rows": order_rows,
            "process_map": process_map,
            "order_map": order_map,
        },
        "dataset_context": {
            "selected_file": selected_csv,
            "latest_file": latest_dataset_name,
            "row_count": dataset_info.get("row_count", 0),
            "latest_modified": dataset_info.get("latest_modified", ""),
            "linked_order_index": dataset_linked_order_index,
        },
        "manual_options": {
            "project_names": order_draw["project_names"],
            "preform_options": preform_options,
            "geometry_options": ORDER_DRAW_GEOMETRY_OPTIONS,
            "priorities": ["Low", "Normal", "High"],
        },
        "setup_options": {
            "coatings": coating_options,
            "dies": die_options,
            "pid_defaults": {
                "p_gain": pid_defaults.get("p_gain", 1.0),
                "i_gain": pid_defaults.get("i_gain", 1.0),
                "winder_mode": pid_defaults.get("winder_mode", "Winder"),
                "increment_value": pid_defaults.get("increment_value", 0.5),
            },
            "drums": [f"BN{i}" for i in range(1, 7)],
        },
        "temp_context": temp_context,
        "sap_summary": order_draw["sap_summary"],
    }


def next_process_setup_index(preform_number: str) -> int:
    preform_number = str(preform_number or "").strip()
    if not preform_number:
        return 1
    pattern = f"{preform_number}F_"
    max_index = 0
    for path in list_csv_files(DATASET_DIR):
        name = path.stem
        if not name.startswith(pattern):
            continue
        tail = name[len(pattern):]
        if tail.isdigit():
            max_index = max(max_index, int(tail))
    return max_index + 1


def process_setup_dataset_name(preform_number: str) -> str:
    preform = str(preform_number or "").strip()
    if preform:
        return f"{preform}F_{next_process_setup_index(preform)}.csv"
    return f"draw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def process_setup_order_rows(row: dict[str, object], order_index: int | None, csv_name: str) -> list[dict[str, object]]:
    draw_name = Path(csv_name).stem
    rows = [
        {"Parameter Name": "=== ORDER PARAMETERS ===", "Value": "", "Units": ""},
        {"Parameter Name": "Order__Draw Name", "Value": draw_name, "Units": ""},
        {"Parameter Name": "Order__Draw Date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Units": ""},
    ]
    if order_index is not None:
        rows.append({"Parameter Name": "Order__Order Index", "Value": order_index, "Units": ""})

    def add_row(name: str, value, units: str = "") -> None:
        rows.append({"Parameter Name": f"Order__{name}", "Value": value, "Units": units})

    add_row("Preform Number", row.get("Preform Number", ""))
    add_row("Fiber Project", row.get(PROJECTS_COL, ""))
    add_row("Priority", row.get("Priority", "Normal"))
    add_row("Order Opener", row.get("Order Opener", ""))
    add_row("Fiber Geometry Type", row.get(GEOMETRY_COL, ""))
    add_row("Tiger Cut (%)", row.get(TIGER_CUT_COL, ""), "%")
    add_row("Octagonal F2F (mm)", row.get(OCT_F2F_COL, ""), "mm")
    add_row("Required Length (m) (for T&M+costumer)", row.get(LENGTH_COL, ""), "m")
    add_row("Good Zones Count (required length zones)", row.get(GOOD_ZONES_COL, ""), "count")
    add_row("Fiber Diameter (µm)", row.get("Fiber Diameter (µm)", ""), "µm")
    add_row("Fiber Diameter Tol (± µm)", row.get(FIBER_TOL_COL, ""), "µm")
    add_row("Main Coating Diameter (µm)", row.get("Main Coating Diameter (µm)", ""), "µm")
    add_row("Main Coating Diameter Tol (± µm)", row.get(MAIN_TOL_COL, ""), "µm")
    add_row("Secondary Coating Diameter (µm)", row.get("Secondary Coating Diameter (µm)", ""), "µm")
    add_row("Secondary Coating Diameter Tol (± µm)", row.get(SECONDARY_TOL_COL, ""), "µm")
    add_row("Tension (g)", row.get("Tension (g)", ""), "g")
    add_row("Draw Speed (m/min)", row.get("Draw Speed (m/min)", ""), "m/min")
    add_row("Main Coating", row.get("Main Coating", ""))
    add_row("Secondary Coating", row.get("Secondary Coating", ""))
    add_row("Main Coating Temperature (°C)", row.get(MAIN_TEMP_COL, ""), "°C")
    add_row("Secondary Coating Temperature (°C)", row.get(SECONDARY_TEMP_COL, ""), "°C")
    add_row("Order Notes", row.get("Notes", ""))
    rows.append({"Parameter Name": "", "Value": "", "Units": ""})
    return rows


def write_selected_dataset_csv(csv_name: str) -> None:
    SELECTED_CSV_JSON.write_text(json.dumps({"selected_csv": os.path.basename(str(csv_name or "").strip())}, indent=2), encoding="utf-8")


def create_process_setup_manual_action(payload: dict) -> JsonResponse:
    preform_number = str(payload.get("preformNumber", "")).strip()
    csv_name = os.path.basename(str(payload.get("csvName", "")).strip() or process_setup_dataset_name(preform_number))
    if not csv_name.lower().endswith(".csv"):
        csv_name = f"{csv_name}.csv"
    csv_path = DATASET_DIR / csv_name
    if csv_path.exists():
        return JsonResponse({"ok": False, "message": f"Dataset CSV already exists: {csv_name}"}, 400)

    row = {
        PROJECTS_COL: str(payload.get("project", "")).strip(),
        "Preform Number": preform_number,
        "Order Opener": str(payload.get("opener", "")).strip(),
        "Priority": str(payload.get("priority", "Normal")).strip() or "Normal",
        GEOMETRY_COL: str(payload.get("geometry", "")).strip(),
        LENGTH_COL: str(payload.get("requiredLength", "")).strip(),
        GOOD_ZONES_COL: str(payload.get("goodZones", "")).strip(),
        "Notes": str(payload.get("notes", "")).strip(),
    }
    write_csv_rows(csv_path, process_setup_order_rows(row, None, csv_name), ["Parameter Name", "Value", "Units"])
    write_selected_dataset_csv(csv_name)
    return JsonResponse({"ok": True, "message": f"Created {csv_name} and set it as the active setup dataset.", "bootstrap": build_bootstrap_payload().body})


def create_process_setup_scheduled_action(payload: dict) -> JsonResponse:
    order_index = int(to_float(payload.get("orderIndex"), -1))
    rows = read_csv_rows(DRAW_ORDERS)
    if order_index < 0 or order_index >= len(rows):
        return JsonResponse({"ok": False, "message": "Scheduled order was not found."}, 404)
    row = rows[order_index]
    status = str(row.get("Status", "")).strip()
    if status != "Scheduled":
        return JsonResponse({"ok": False, "message": "Only scheduled orders can start Process Setup here."}, 400)

    preform_number = str(payload.get("preformNumber", "")).strip() or str(row.get("Preform Number", "")).strip()
    if not preform_number or preform_number == "0":
        return JsonResponse({"ok": False, "message": "A valid preform number is required first."}, 400)

    csv_name = os.path.basename(str(payload.get("csvName", "")).strip() or process_setup_dataset_name(preform_number))
    if not csv_name.lower().endswith(".csv"):
        csv_name = f"{csv_name}.csv"
    csv_path = DATASET_DIR / csv_name
    if csv_path.exists():
        return JsonResponse({"ok": False, "message": f"Dataset CSV already exists: {csv_name}"}, 400)

    row["Preform Number"] = preform_number
    write_csv_rows(csv_path, process_setup_order_rows(row, order_index, csv_name), ["Parameter Name", "Value", "Units"])
    row["Active CSV"] = csv_name
    row["Status"] = "In Progress"
    row["Status Updated At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = read_csv_fieldnames(DRAW_ORDERS) or list(row.keys())
    write_csv_rows(DRAW_ORDERS, rows, fieldnames)
    write_selected_dataset_csv(csv_name)
    return JsonResponse({"ok": True, "message": f"Started Process Setup from order #{order_index} and created {csv_name}.", "bootstrap": build_bootstrap_payload().body})


def select_process_setup_dataset_action(payload: dict) -> JsonResponse:
    selected_csv = os.path.basename(str(payload.get("selectedCsv", "")).strip())
    if selected_csv and not (DATASET_DIR / selected_csv).exists():
        return JsonResponse({"ok": False, "message": "Selected dataset CSV was not found."}, 404)
    write_selected_dataset_csv(selected_csv)
    return JsonResponse({"ok": True, "message": "Process Setup dataset changed.", "bootstrap": build_bootstrap_payload().body})


def build_process_setup_save_rows(payload: dict) -> list[dict[str, object]]:
    rows = [
        {"Parameter Name": "=== PROCESS SETUP ===", "Value": "", "Units": ""},
        {"Parameter Name": "Process__Process Setup Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Units": ""},
    ]

    def add(name: str, value, units: str = "") -> None:
        if value is None:
            return
        value_text = str(value).strip()
        if value_text == "":
            return
        rows.append({"Parameter Name": f"Process__{name}", "Value": value, "Units": units})

    iris = payload.get("iris") or {}
    coating = payload.get("coating") or {}
    pid = payload.get("pid") or {}
    drum = payload.get("drum") or {}

    add("Preform Shape", iris.get("shape"))
    add("Preform Diameter", iris.get("preform_diameter_mm"), "mm")
    add("Octagonal Preform", 1 if str(iris.get("shape", "")).strip() == "Octagonal" else 0, "bool")
    add("Octagonal F2F", iris.get("oct_f2f_mm"), "mm")
    add("Tiger Preform", 1 if str(iris.get("shape", "")).strip() == "Tiger Cut" else 0, "bool")
    add("Tiger Cut", iris.get("tiger_cut_pct"), "%")
    add("PM Iris System", 1 if iris.get("pm_system") else 0, "bool")
    add("Iris Mode", iris.get("iris_mode"))
    add("Base Area", iris.get("base_area_mm2"), "mm^2")
    add("Adjusted Area", iris.get("adjusted_area_mm2"), "mm^2")
    add("Effective Preform Diameter", iris.get("effective_preform_diameter_mm"), "mm")
    add("Selected Iris Diameter", iris.get("selected_iris_diameter_mm"), "mm")
    add("Iris Gap Area", iris.get("gap_area_mm2"), "mm^2")

    add("Entry Fiber Diameter", coating.get("entry_fiber_diameter_um"), "µm")
    add("Target First Coating Diameter", coating.get("target_first_coating_diameter_um"), "µm")
    add("Target Second Coating Diameter", coating.get("target_second_coating_diameter_um"), "µm")
    add("Primary Coating", coating.get("primary_coating"))
    add("Secondary Coating", coating.get("secondary_coating"))
    add("Primary Coating Temperature", coating.get("primary_temp_c"), "°C")
    add("Secondary Coating Temperature", coating.get("secondary_temp_c"), "°C")
    add("Primary Die Name", coating.get("primary_die"))
    add("Secondary Die Name", coating.get("secondary_die"))
    add("Coating Die Selection Mode", coating.get("die_mode"))
    add("Draw Speed", coating.get("draw_speed_m_min"), "m/min")

    add("P Gain (Diameter Control)", pid.get("p_gain"))
    add("I Gain (Diameter Control)", pid.get("i_gain"))
    add("TF Mode", pid.get("tf_mode"))
    add("Increment TF Value", pid.get("increment_value_mm"), "mm")

    add("Selected Drum", drum.get("selected_drum"))
    rows.append({"Parameter Name": "", "Value": "", "Units": ""})
    return rows


def save_process_setup_action(payload: dict) -> JsonResponse:
    selected_csv = os.path.basename(str(payload.get("selectedCsv", "")).strip())
    if not selected_csv:
        return JsonResponse({"ok": False, "message": "Choose an active dataset CSV first."}, 400)
    rows = build_process_setup_save_rows(payload)
    ok, message = append_dataset_rows(selected_csv, rows)
    status = 200 if ok else 400
    if not ok:
        return JsonResponse({"ok": False, "message": message}, status)
    write_selected_dataset_csv(selected_csv)
    return JsonResponse({"ok": True, "message": message, "bootstrap": build_bootstrap_payload().body}, status)


def find_order_index_by_dataset(dataset_name: str) -> int | None:
    rows = read_csv_rows(DRAW_ORDERS)
    dataset_name = os.path.basename(str(dataset_name or "").strip())
    for index, row in enumerate(rows):
        for key in ("Assigned Dataset CSV", "Active CSV", "Done CSV", "Failed CSV", "Fail Try Dataset CSV"):
            if os.path.basename(str(row.get(key, "")).strip()) == dataset_name and dataset_name:
                return index
    for index, row in enumerate(rows):
        active_csv = os.path.basename(str(row.get("Active CSV", "")).strip())
        if active_csv == dataset_name and dataset_name:
            return index
    return None


def summarize_draw_finalize(selected_csv_override: str | None = None) -> dict:
    dataset_files = list_csv_files(DATASET_DIR)
    latest_dataset = dataset_files[0].name if dataset_files else ""
    selected_csv = latest_dataset
    if selected_csv_override:
        requested_csv = os.path.basename(str(selected_csv_override).strip())
        if any(path.name == requested_csv for path in dataset_files):
            selected_csv = requested_csv
    orders = read_csv_rows(DRAW_ORDERS)
    fault_rows = read_csv_rows(FAULTS_LOG)
    fault_action_rows = read_csv_rows(FAULTS_ACTIONS_LOG)
    matched_index = find_order_index_by_dataset(selected_csv) if selected_csv else None
    matched_order = format_order_for_ui(orders[matched_index], matched_index) if matched_index is not None and matched_index < len(orders) else {}
    components = dedupe_strings([row.get("fault_component", "") for row in fault_rows] + [row.get("component", "") for row in summarize_maintenance_rebuild()["tasks"]])
    return {
        "dataset_files": [path.name for path in dataset_files],
        "latest_dataset": latest_dataset,
        "selected_csv": selected_csv,
        "matched_order": matched_order,
        "components": components,
        "recent_faults": fault_rows[-8:],
    }


def read_development_tables() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    projects = read_csv_rows(DATA_DIR / "development_projects.csv")
    experiments = read_csv_rows(DATA_DIR / "development_experiments.csv")
    updates = read_csv_rows(DATA_DIR / "experiment_updates.csv")
    return projects, experiments, updates


DEVELOPMENT_PROJECT_FIELDS = [
    "Project Name",
    "Project Purpose",
    "Target",
    "Created At",
    "Archived",
    "Summary Title",
    "Summary Notes",
    "Summary Date",
    "Summary Researcher",
]


def development_project_fieldnames() -> list[str]:
    fieldnames = read_csv_fieldnames(DATA_DIR / "development_projects.csv") or []
    if not fieldnames:
        fieldnames = DEVELOPMENT_PROJECT_FIELDS[:]
    else:
        fieldnames = fieldnames[:]
        for field in DEVELOPMENT_PROJECT_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def ensure_report_center_dir() -> None:
    REPORT_CENTER_DIR.mkdir(parents=True, exist_ok=True)


def build_operations_report_markdown(title: str, start_date: str, end_date: str, sections: list[str]) -> str:
    draws = summarize_draw_orders()
    schedule = summarize_schedule()
    parts = summarize_part_orders()
    maintenance = summarize_maintenance_rebuild()
    included = sections or REPORT_CENTER_SECTIONS
    lines = [
        f"# {title or 'Tower Operations Report'}",
        "",
        f"- Window: `{start_date}` to `{end_date}`",
        f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Snapshot",
        "",
        f"- Active draws: **{draws['active']}**",
        f"- Upcoming events: **{len(schedule['upcoming'])}**",
        f"- Open part orders: **{len(parts['open_orders'])}**",
        f"- Maintenance prep queue: **{len(maintenance['prep_queue'])}**",
        "",
    ]
    for section in included:
        lines.extend(["", f"## {section}", ""])
        if section == "Executive Summary":
            lines.extend(
                [
                    f"- Draw outcomes: Done **{draws['done']}**, Failed **{draws['failed']}**",
                    f"- Schedule event types tracked: **{len(schedule['type_counts'])}**",
                    f"- Part order statuses tracked: **{len(parts['status_counts'])}**",
                    f"- Maintenance tasks tracked: **{len(maintenance['tasks'])}**",
                ]
            )
        elif section == "Resources: Gas + SAP + Preforms":
            sap = summarize_order_draw()["sap_summary"]
            lines.extend(
                [
                    f"- {sap['item']}: **{sap['count']} {sap['units']}**",
                    f"- Last updated: `{sap['last_updated'] or 'Unknown'}`",
                    f"- Notes: {sap['notes'] or 'No notes'}",
                ]
            )
        elif section == "Draw Outcomes (Done/Failed + Notes)":
            recent = draws["recent"][:8]
            if recent:
                for item in recent:
                    lines.append(f"- {item['preform'] or 'Unknown'} | {item['project'] or 'No project'} | {item['status']}")
            else:
                lines.append("- No recent draw rows found.")
        elif section == "Parts Orders Status":
            for item in parts["all_orders"][:10]:
                lines.append(f"- {item['part_name']} | {item['status']} | {item['project'] or item['company'] or 'General'}")
        elif section == "Schedule: Past Week + Next Week":
            for item in schedule["upcoming"][:10]:
                lines.append(f"- {item['start_label']} | {item['event_type']} | {item['description'] or 'No description'}")
        elif section == "Maintenance + Faults":
            for item in maintenance["prep_queue"][:10]:
                lines.append(f"- {item['component']} | {item['task']} | {item['flow_state']}")
        elif section == "Maintenance Tests + Measurements":
            for item in maintenance["recent_actions"][:10]:
                lines.append(f"- {item['done_date']} | {item['component']} | {item['task']}")
        elif section == "Consumables Snapshot":
            inventory = summarize_inventory()
            for item in inventory["low_stock"][:10]:
                lines.append(f"- {item['part_name']} | Qty {item['quantity']} | {item['location'] or 'No location'}")
    return "\n".join(lines).strip() + "\n"


def summarize_development_project(project_name: str) -> dict:
    projects, experiments, updates = read_development_tables()
    project = next((row for row in projects if str(row.get("Project Name", "")).strip() == project_name.strip()), {})
    project_experiments = [row for row in experiments if str(row.get("Project Name", "")).strip() == project_name.strip()]
    project_updates = [row for row in updates if str(row.get("Project Name", "")).strip() == project_name.strip()]
    media_count = 0
    researchers = dedupe_strings([row.get("Researcher", "") for row in project_experiments + project_updates])
    latest_update = ""
    dated_updates = sorted(
        project_updates,
        key=lambda row: str(row.get("Update Date", "")),
        reverse=True,
    )
    if dated_updates:
        latest_update = dated_updates[0].get("Update Date", "")
    project_experiments = sorted(project_experiments, key=lambda row: str(row.get("Date", "")), reverse=True)
    drawing_experiments = [row for row in project_experiments if str(row.get("Is Drawing", "")).strip().lower() in {"true", "1", "yes"}]
    for row in project_experiments:
        for field in ("Result Images", "Result Docs", "Attachments"):
            media_count += len([item for item in str(row.get(field, "")).split(";") if item.strip()])
    return {
        "project": project,
        "experiment_count": len(project_experiments),
        "update_count": len(project_updates),
        "media_count": media_count,
        "latest_update": latest_update,
        "latest_experiment": project_experiments[0] if project_experiments else {},
        "drawing_experiment_count": len(drawing_experiments),
        "archived": str(project.get("Archived", "")).strip().lower() in {"true", "1", "yes"},
        "researchers": researchers,
        "experiments": project_experiments[:18],
        "updates": project_updates[:18],
        "dataset_files": [path.name for path in list_csv_files(DATASET_DIR)[:40]],
    }


def development_project_timeline_entries(details: dict) -> list[dict]:
    dated_items: list[dict] = []
    order = 0
    for item in reversed(details.get("experiments", []) or []):
        dated_items.append(
            {
                "kind": "experiment",
                "order": order,
                "date": str(item.get("Date", "")).strip(),
                "item": item,
            }
        )
        order += 1
    for item in details.get("updates", []) or []:
        dated_items.append(
            {
                "kind": "update",
                "order": order,
                "date": str(item.get("Update Date", "")).strip(),
                "item": item,
            }
        )
        order += 1
    project = details.get("project", {}) or {}
    summary_notes = str(project.get("Summary Notes", "")).strip()
    summary_date = str(project.get("Summary Date", "")).strip()
    summary_title = str(project.get("Summary Title", "")).strip()
    summary_researcher = str(project.get("Summary Researcher", "")).strip()
    if project.get("Project Name") and (summary_notes or summary_title or summary_date or summary_researcher):
        dated_items.append(
            {
                "kind": "summary",
                "order": order,
                "date": summary_date or str(details.get("latest_update", "")).strip(),
                "item": {
                    "title": summary_title or "Project summary",
                    "status": "Archived" if details.get("archived") else "Active",
                    "drawings": details.get("drawing_experiment_count") or 0,
                    "researchers": ", ".join(details.get("researchers") or []) or "Not listed",
                    "target": project.get("Target") or "Not set",
                    "notes": summary_notes,
                    "researcher": summary_researcher,
                    "latestActivity": summary_date or str(details.get("latest_update", "")).strip() or "No activity yet",
                },
            }
        )
    return sorted(
        dated_items,
        key=lambda entry: ((entry.get("date") or "9999-99-99"), entry.get("order", 0)),
    )


def development_project_latest_activity(details: dict) -> str:
    timeline = development_project_timeline_entries(details)
    if timeline:
        return str(timeline[-1].get("date", "")).strip() or "No activity yet"
    return str(details.get("latest_update", "")).strip() or "No activity yet"


def markdown_paragraph(value: str, fallback: str = "Not set.") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def html_multiline(value: str, fallback: str = "Not set.") -> str:
    text = str(value or "").strip()
    safe = escape_html(text or fallback)
    return safe.replace("\n", "<br/>")


def build_development_report_markdown(project_name: str) -> str:
    details = summarize_development_project(project_name)
    project = details["project"]
    if not project:
        return f"# Project Paper: {project_name}\n\nProject was not found.\n"
    timeline = development_project_timeline_entries(details)
    latest_activity = development_project_latest_activity(details)
    lines = [
        f"# Project Paper: {project_name}",
        "",
        "> Structured export from the Tower development workspace.",
        "",
        "## Project Identity",
        "",
        f"- Description: {project.get('Project Purpose', 'Not set')}",
        f"- Target: {project.get('Target', 'Not set')}",
        f"- Created: {project.get('Created At', 'Unknown')}",
        f"- Researchers: {', '.join(details['researchers']) or 'None listed'}",
        f"- Latest activity: {latest_activity}",
        f"- Status: {'Archived' if details.get('archived') else 'Active'}",
        "",
        "## Workspace Metrics",
        "",
        f"- Experiments: **{details['experiment_count']}**",
        f"- Updates: **{details['update_count']}**",
        f"- Drawing runs: **{details['drawing_experiment_count'] or 0}**",
        f"- Media attachments: **{details['media_count']}**",
        "",
        "## Project Summary",
        "",
        f"- Title: {project.get('Summary Title', '') or 'Project summary'}",
        f"- Date: {project.get('Summary Date', '') or 'Not set'}",
        f"- Researcher: {project.get('Summary Researcher', '') or 'Not set'}",
        "",
        markdown_paragraph(project.get("Summary Notes", ""), "No project summary saved yet."),
        "",
        "## Timeline",
        "",
    ]
    if timeline:
        for entry in timeline:
            item = entry.get("item", {}) or {}
            if entry.get("kind") == "summary":
                lines.extend(
                    [
                        f"### {entry.get('date') or 'No date'} · Summary",
                        f"- Title: {item.get('title', 'Project summary')}",
                        f"- Researcher: {item.get('researcher', '') or 'Not set'}",
                        f"- Status: {item.get('status', 'Active')}",
                        f"- Draws: {item.get('drawings', 0)}",
                        "",
                        markdown_paragraph(item.get("notes", ""), "No project summary was written yet."),
                        "",
                    ]
                )
                continue
            if entry.get("kind") == "update":
                lines.extend(
                    [
                        f"### {entry.get('date') or 'No date'} · Update",
                        f"- Title: {item.get('Experiment Title', '') or 'Project update'}",
                        f"- Researcher: {item.get('Researcher', '') or 'Not set'}",
                        "",
                        markdown_paragraph(item.get("Update Notes", ""), "No update notes saved."),
                        "",
                    ]
                )
                continue
            lines.extend(
                [
                    f"### {entry.get('date') or 'No date'} · Experiment",
                    f"- Title: {item.get('Experiment Title', 'Untitled Experiment')}",
                    f"- Researcher: {item.get('Researcher', '') or 'Not set'}",
                    f"- Purpose: {item.get('Purpose', '') or '-'}",
                    f"- Methods: {item.get('Methods', '') or '-'}",
                    f"- Observations: {item.get('Observations', '') or '-'}",
                    f"- Results: {item.get('Results', '') or '-'}",
                    f"- Draw CSV: {item.get('Draw CSV', '') or 'Not linked'}",
                    f"- Drawing details: {item.get('Drawing Details', '') or '-'}",
                    "",
                    markdown_paragraph(item.get("Markdown Notes", ""), "No experiment notes saved."),
                    "",
                ]
            )
    else:
        lines.append("- No project history found.")
    return "\n".join(lines).strip() + "\n"


def build_development_report_html(project_name: str) -> str:
    details = summarize_development_project(project_name)
    project = details["project"]
    if not project:
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Project Paper</title></head>
<body><main><h1>Project Paper</h1><p>Project was not found.</p></main></body></html>"""
    timeline = development_project_timeline_entries(details)
    latest_activity = development_project_latest_activity(details)
    archived_label = "Archived" if details.get("archived") else "Active"
    metrics = [
        ("Experiments", str(details.get("experiment_count", 0))),
        ("Updates", str(details.get("update_count", 0))),
        ("Drawing runs", str(details.get("drawing_experiment_count", 0))),
        ("Status", archived_label),
    ]
    facts = [
        ("Latest activity", latest_activity),
        ("Researchers", ", ".join(details.get("researchers") or []) or "Not listed"),
        ("Created", str(project.get("Created At", "")).strip() or "Unknown"),
    ]
    timeline_markup = []
    for entry in timeline:
        item = entry.get("item", {}) or {}
        kind = str(entry.get("kind", "")).strip()
        date_text = escape_html(str(entry.get("date", "")).strip() or "No date")
        if kind == "summary":
            body = f"""
              <div class="paper-card-body">
                <div class="paper-facet-grid">
                  <div class="paper-facet"><span>Status</span><strong>{escape_html(str(item.get('status', 'Active')))}</strong></div>
                  <div class="paper-facet"><span>Draws</span><strong>{escape_html(str(item.get('drawings', 0)))}</strong></div>
                  <div class="paper-facet"><span>Researchers</span><strong>{escape_html(str(item.get('researchers', 'Not listed')))}</strong></div>
                </div>
                <div class="paper-rich-note">{html_multiline(item.get('notes', ''), 'No project summary was written yet.')}</div>
              </div>
            """
            title = escape_html(str(item.get("title", "Project summary")))
            label = "Summary"
            meta = escape_html(str(item.get("researcher", "")).strip() or "Project note")
            node = "SM"
        elif kind == "update":
            body = f"""
              <div class="paper-card-body">
                <div class="paper-rich-note">{html_multiline(item.get('Update Notes', ''), 'No update notes saved.')}</div>
              </div>
            """
            title = escape_html(str(item.get("Experiment Title", "")).strip() or "Project update")
            label = "Update"
            meta = escape_html(str(item.get("Researcher", "")).strip() or "Not set")
            node = "UP"
        else:
            is_drawing = str(item.get("Is Drawing", "")).strip().lower() in {"true", "1", "yes"}
            body = f"""
              <div class="paper-card-body">
                <div class="paper-facet-grid">
                  <div class="paper-facet"><span>Purpose</span><strong>{escape_html(str(item.get('Purpose', '')).strip() or '-')}</strong></div>
                  <div class="paper-facet"><span>Methods</span><strong>{escape_html(str(item.get('Methods', '')).strip() or '-')}</strong></div>
                  <div class="paper-facet"><span>Observations</span><strong>{escape_html(str(item.get('Observations', '')).strip() or '-')}</strong></div>
                  <div class="paper-facet"><span>Results</span><strong>{escape_html(str(item.get('Results', '')).strip() or '-')}</strong></div>
                </div>
                <div class="paper-rich-note">{html_multiline(item.get('Markdown Notes', ''), 'No experiment notes saved.')}</div>
                <div class="paper-footnote">Draw CSV: {escape_html(str(item.get('Draw CSV', '')).strip() or 'Not linked')} · Drawing details: {escape_html(str(item.get('Drawing Details', '')).strip() or '-')}</div>
              </div>
            """
            title = escape_html(str(item.get("Experiment Title", "")).strip() or "Untitled experiment")
            label = "Draw experiment" if is_drawing else "Experiment"
            meta = escape_html(str(item.get("Researcher", "")).strip() or "Not set")
            node = "DR" if is_drawing else "EX"
        timeline_markup.append(
            f"""
            <article class="paper-timeline-item paper-kind-{kind or 'experiment'}">
              <div class="paper-rail">
                <span class="paper-node">{node}</span>
              </div>
              <div class="paper-card">
                <div class="paper-card-head">
                  <div>
                    <span>{escape_html(label)}</span>
                    <strong>{title}</strong>
                  </div>
                  <div class="paper-card-meta">{date_text} · {meta}</div>
                </div>
                {body}
              </div>
            </article>
            """
        )
    metric_markup = "".join(
        f'<div class="paper-metric"><span>{escape_html(label)}</span><strong>{escape_html(value)}</strong></div>'
        for label, value in metrics
    )
    fact_markup = "".join(
        f'<div class="paper-fact"><span>{escape_html(label)}</span><strong>{escape_html(value)}</strong></div>'
        for label, value in facts
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Paper - {escape_html(project_name)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #061018;
      --panel: rgba(10, 18, 29, 0.92);
      --panel-soft: rgba(12, 24, 36, 0.82);
      --line: rgba(159, 223, 227, 0.14);
      --muted: rgba(183, 214, 220, 0.74);
      --text: rgba(244, 250, 251, 0.96);
      --accent: #72ffe8;
      --accent-soft: rgba(114, 255, 232, 0.18);
      --good: #82ffcf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(114,255,232,0.08), transparent 28%),
        linear-gradient(180deg, #07121d 0%, #050c14 100%);
      color: var(--text);
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .paper-toolbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(6, 16, 24, 0.94);
      backdrop-filter: blur(18px);
    }}
    .paper-toolbar-copy {{
      display: grid;
      gap: 4px;
    }}
    .paper-toolbar-copy span,
    .paper-hero-kicker,
    .paper-section-kicker,
    .paper-card-head span,
    .paper-metric span,
    .paper-fact span,
    .paper-facet span {{
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .paper-toolbar button {{
      border: 1px solid rgba(114,255,232,0.24);
      background: linear-gradient(180deg, rgba(114,255,232,0.16), rgba(114,255,232,0.04));
      color: var(--text);
      padding: 10px 14px;
      cursor: pointer;
    }}
    .paper-shell {{
      width: min(1160px, calc(100% - 40px));
      margin: 0 auto;
      padding: 28px 0 40px;
      display: grid;
      gap: 18px;
    }}
    .paper-hero,
    .paper-panel,
    .paper-progress {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, var(--panel-soft), rgba(7, 14, 22, 0.84));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }}
    .paper-hero {{
      padding: 24px;
      display: grid;
      gap: 12px;
    }}
    .paper-hero h1 {{
      margin: 0;
      font-size: clamp(2.2rem, 5vw, 4rem);
      line-height: 0.95;
      text-transform: uppercase;
      letter-spacing: -0.04em;
    }}
    .paper-description {{
      max-width: 78ch;
      color: rgba(224,238,240,0.9);
      line-height: 1.6;
    }}
    .paper-target {{
      display: grid;
      gap: 4px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      max-width: 48ch;
    }}
    .paper-target strong,
    .paper-fact strong,
    .paper-facet strong,
    .paper-metric strong {{
      font-size: 1.02rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .paper-metrics,
    .paper-facts,
    .paper-facet-grid {{
      display: grid;
      gap: 10px;
    }}
    .paper-metrics {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .paper-facts {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .paper-facet-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .paper-metric,
    .paper-fact,
    .paper-facet {{
      display: grid;
      gap: 6px;
      padding: 14px 16px;
      border: 1px solid rgba(159,223,227,0.08);
      background: linear-gradient(180deg, rgba(8,16,24,0.54), rgba(8,16,24,0.18));
    }}
    .paper-panel {{
      padding: 20px;
      display: grid;
      gap: 14px;
    }}
    .paper-rich-note {{
      color: rgba(224,238,240,0.92);
      line-height: 1.7;
      white-space: normal;
    }}
    .paper-progress {{
      padding: 20px;
      display: grid;
      gap: 18px;
    }}
    .paper-timeline {{
      display: grid;
      gap: 16px;
    }}
    .paper-timeline-item {{
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .paper-rail {{
      position: relative;
      display: flex;
      justify-content: center;
    }}
    .paper-rail::after {{
      content: "";
      position: absolute;
      top: 44px;
      bottom: -16px;
      width: 1px;
      background: linear-gradient(180deg, rgba(114,255,232,0.28), rgba(114,255,232,0));
    }}
    .paper-timeline-item:last-child .paper-rail::after {{ display: none; }}
    .paper-node {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      border-radius: 12px;
      border: 1px solid rgba(114,255,232,0.18);
      background: rgba(13, 28, 40, 0.88);
      color: rgba(233,247,249,0.96);
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.08em;
    }}
    .paper-kind-update .paper-node {{
      border-color: rgba(198, 126, 255, 0.22);
      background: rgba(31, 18, 46, 0.92);
    }}
    .paper-card {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(10,18,29,0.82), rgba(8,14,22,0.92));
    }}
    .paper-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px 14px;
      border-bottom: 1px solid rgba(159,223,227,0.08);
    }}
    .paper-card-head strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.02rem;
      line-height: 1.4;
    }}
    .paper-card-meta {{
      align-self: end;
      color: rgba(214, 233, 236, 0.84);
      white-space: nowrap;
    }}
    .paper-card-body {{
      display: grid;
      gap: 14px;
      padding: 16px 18px 18px;
    }}
    .paper-footnote {{
      color: rgba(183,214,220,0.78);
      line-height: 1.5;
    }}
    @media (max-width: 900px) {{
      .paper-shell {{ width: min(100%, calc(100% - 24px)); }}
      .paper-metrics,
      .paper-facts,
      .paper-facet-grid {{ grid-template-columns: 1fr 1fr; }}
      .paper-card-head {{ flex-direction: column; }}
      .paper-card-meta {{ white-space: normal; }}
    }}
    @media (max-width: 640px) {{
      .paper-metrics,
      .paper-facts,
      .paper-facet-grid {{ grid-template-columns: 1fr; }}
      .paper-timeline-item {{ grid-template-columns: 1fr; }}
      .paper-rail {{ justify-content: flex-start; }}
      .paper-rail::after {{ display: none; }}
    }}
    @media print {{
      @page {{
        size: A4;
        margin: 12mm;
      }}
      .paper-toolbar {{ display: none; }}
      body {{ background: #061018; }}
      .paper-shell {{
        width: 100%;
        padding: 0;
      }}
      .paper-hero,
      .paper-panel,
      .paper-progress,
      .paper-card,
      .paper-timeline-item,
      .paper-metric,
      .paper-fact,
      .paper-facet {{
        break-inside: avoid;
        page-break-inside: avoid;
      }}
      .paper-timeline {{
        gap: 12px;
      }}
      .paper-card-body,
      .paper-rich-note,
      .paper-footnote {{
        orphans: 3;
        widows: 3;
      }}
    }}
  </style>
</head>
<body>
  <div class="paper-toolbar">
    <div class="paper-toolbar-copy">
      <span>Project paper export</span>
      <strong>{escape_html(project_name)}</strong>
    </div>
    <button type="button" onclick="window.print()">Print / Save PDF</button>
  </div>
  <main class="paper-shell">
    <section class="paper-hero">
      <span class="paper-hero-kicker">Development</span>
      <h1>{escape_html(project_name)}</h1>
      <div class="paper-description">{html_multiline(project.get("Project Purpose", ""), "No project description saved yet.")}</div>
      <div class="paper-target">
        <span>Target</span>
        <strong>{escape_html(str(project.get("Target", "")).strip() or "Not set")}</strong>
      </div>
    </section>
    <section class="paper-metrics">{metric_markup}</section>
    <section class="paper-panel">
      <span class="paper-section-kicker">Project facts</span>
      <div class="paper-facts">{fact_markup}</div>
    </section>
    <section class="paper-panel">
      <span class="paper-section-kicker">Project summary</span>
      <strong>{escape_html(str(project.get("Summary Title", "")).strip() or "Project summary")}</strong>
      <div class="paper-footnote">Date: {escape_html(str(project.get("Summary Date", "")).strip() or "Not set")} · Researcher: {escape_html(str(project.get("Summary Researcher", "")).strip() or "Not set")}</div>
      <div class="paper-rich-note">{html_multiline(project.get("Summary Notes", ""), "No project summary saved yet.")}</div>
    </section>
    <section class="paper-progress">
      <div>
        <span class="paper-section-kicker">Project progress</span>
        <h2>Append lab work in one timeline</h2>
      </div>
      <div class="paper-timeline">
        {''.join(timeline_markup) if timeline_markup else '<div class="paper-panel"><div class="paper-rich-note">No project history found yet.</div></div>'}
      </div>
    </section>
  </main>
</body>
</html>"""


def summarize_report_center() -> dict:
    ensure_report_center_dir()
    projects, experiments, updates = read_development_tables()
    pdf_exports = list_recent_files(REPORT_CENTER_DIR, suffixes=(".pdf",), limit=20)
    md_exports = list_recent_files(REPORT_CENTER_DIR, suffixes=(".md",), limit=20)
    project_names = dedupe_strings([row.get("Project Name", "") for row in projects if row.get("Project Name", "")])
    return {
        "metrics": [
            {"label": "Operations exports", "value": len(pdf_exports)},
            {"label": "Markdown exports", "value": len(md_exports)},
            {"label": "Dev projects", "value": len(project_names)},
            {"label": "Experiments", "value": len(experiments)},
        ],
        "modes": ["Operations Report", "Development Process", "Recent Exports"],
        "sections": REPORT_CENTER_SECTIONS,
        "project_names": project_names,
        "projects_count": len(project_names),
        "experiments_count": len(experiments),
        "updates_count": len(updates),
        "recent_pdf_exports": pdf_exports,
        "recent_md_exports": md_exports,
        "latest_export": (pdf_exports + md_exports)[0] if (pdf_exports or md_exports) else {},
        "default_project": project_names[0] if project_names else "",
    }


def normalize_dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    current_section = "General"
    normalized = []
    for index, row in enumerate(rows):
        parameter_name = str(row.get("Parameter Name", "")).strip()
        value = str(row.get("Value", "")).strip()
        units = str(row.get("Units", "")).strip()
        if parameter_name.startswith("===") and parameter_name.endswith("==="):
            current_section = parameter_name.strip("= ").title()
            continue
        group_name = ""
        key_name = parameter_name
        if "__" in parameter_name:
            group_name, key_name = parameter_name.split("__", 1)
        value_num = None
        try:
            value_num = float(value)
        except ValueError:
            value_num = None
        normalized.append(
            {
                "row_index": index,
                "section": current_section,
                "parameter_name": parameter_name,
                "group_name": group_name or current_section,
                "key_name": key_name,
                "value": value,
                "units": units,
                "value_num": value_num,
            }
        )
    return normalized


def dataset_param_family(name: str) -> str:
    s = str(name or "").lower()
    if s.startswith("order__"):
        return "Order"
    if s.startswith("process__"):
        return "Process"
    if "zone " in s or s.startswith("zone ") or s.startswith("marked zone"):
        return "Zones"
    if "t&m" in s or "good zone" in s or "cut/save" in s or "fiber length" in s or "drum |" in s:
        return "Winder + T&M"
    return "General"


def infer_dataset_event_ts(normalized_rows: list[dict[str, object]], path: Path) -> str:
    candidates = []
    for row in normalized_rows:
        param = str(row.get("parameter_name", "")).strip().lower()
        value = str(row.get("value", "")).strip()
        if not value:
            continue
        if param in {"draw date", "draw datetime"} or "draw date" in param or "draw time" in param or "datetime" in param:
            candidates.append(value)
    for value in candidates:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def load_dataset_scope_records(dataset_name: str | None = None) -> tuple[list[Path], list[dict[str, object]]]:
    files = list_csv_files(DATASET_DIR)
    selected_name = str(dataset_name or "").strip()
    if selected_name and selected_name != "__ALL__":
        files = [path for path in files if path.name == selected_name]
    records: list[dict[str, object]] = []
    for path in files:
        normalized = normalize_dataset_rows(read_csv_rows(path))
        draw_id = path.stem
        event_ts = infer_dataset_event_ts(normalized, path)
        for row in normalized:
            records.append(
                {
                    **row,
                    "_draw": draw_id,
                    "filename": path.name,
                    "event_ts": event_ts,
                }
            )
    return files, records


def load_dataset_records_for_filenames(file_names: list[str] | None = None) -> tuple[list[Path], list[dict[str, object]]]:
    requested = {str(name or "").strip() for name in (file_names or []) if str(name or "").strip()}
    files = list_csv_files(DATASET_DIR)
    if requested:
        files = [path for path in files if path.name in requested]
    records: list[dict[str, object]] = []
    for path in files:
        normalized = normalize_dataset_rows(read_csv_rows(path))
        draw_id = path.stem
        event_ts = infer_dataset_event_ts(normalized, path)
        for row in normalized:
            records.append(
                {
                    **row,
                    "_draw": draw_id,
                    "filename": path.name,
                    "event_ts": event_ts,
                }
            )
    return files, records


def collect_sql_lab_components() -> dict[str, list[str]]:
    maintenance_components = dedupe_strings(
        [row.get("component", "") for row in read_csv_rows(MAINTENANCE_ACTIONS)]
        + [row.get("Component", "") for row in read_maintenance_tracker_rows()]
    )
    fault_components = dedupe_strings([row.get("fault_component", "") for row in read_csv_rows(FAULTS_LOG)] + maintenance_components)
    return {
        "maintenance": maintenance_components,
        "faults": fault_components,
    }


def analyze_dataset_file(dataset_name: str | None = None) -> dict:
    files, normalized = load_dataset_scope_records(dataset_name)
    selected_name = str(dataset_name or "").strip()
    selected_label = selected_name if selected_name and selected_name != "__ALL__" else "__ALL__"
    if not files:
        return {
            "selected_file": "",
            "available_files": [],
            "groups": [],
            "sections": [],
            "parameter_names": [],
            "family_counts": [],
            "preview_rows": [],
            "row_count": 0,
            "numeric_count": 0,
        }
    group_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    value_map: dict[str, list[str]] = {}
    for row in normalized:
        group = str(row["group_name"] or "General")
        section = str(row["section"] or "General")
        family = dataset_param_family(str(row["parameter_name"] or ""))
        group_counts[group] = group_counts.get(group, 0) + 1
        section_counts[section] = section_counts.get(section, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        param = str(row["parameter_name"] or "")
        value = str(row["value"] or "")
        if param and value and len(value_map.get(param, [])) < 20 and value not in value_map.setdefault(param, []):
            value_map[param].append(value)
    return {
        "selected_file": selected_label,
        "available_files": ["__ALL__", *[path.name for path in files]],
        "groups": [{"label": key, "value": value} for key, value in sorted(group_counts.items(), key=lambda item: item[0].lower())],
        "sections": [{"label": key, "value": value} for key, value in sorted(section_counts.items(), key=lambda item: item[0].lower())],
        "family_counts": [{"label": key, "value": value} for key, value in sorted(family_counts.items(), key=lambda item: item[0].lower())],
        "parameter_names": sorted({str(row["parameter_name"]) for row in normalized if str(row.get("parameter_name", "")).strip()}, key=str.lower),
        "parameter_values": value_map,
        "preview_rows": normalized[:160],
        "row_count": len(normalized),
        "numeric_count": len([row for row in normalized if row["value_num"] is not None]),
        "latest_modified": max(datetime.fromtimestamp(path.stat().st_mtime) for path in files).strftime("%Y-%m-%d %H:%M"),
    }


def summarize_sql_lab() -> dict:
    dataset_files = list_csv_files(DATASET_DIR)
    latest_payload = analyze_dataset_file("__ALL__") if dataset_files else {}
    components = collect_sql_lab_components()
    return {
        "metrics": [
            {"label": "Dataset CSVs", "value": len(dataset_files)},
            {"label": "Rows in focus", "value": latest_payload.get("row_count", 0)},
            {"label": "Numeric values", "value": latest_payload.get("numeric_count", 0)},
            {"label": "Groups", "value": len(latest_payload.get("groups", []))},
        ],
        "dataset_files": ["__ALL__", *[path.name for path in dataset_files]],
        "latest_dataset": "__ALL__",
        "operators": ["any", "=", "!=", ">", ">=", "<", "<=", "between", "contains"],
        "include_modes": ["Draws", "Maintenance", "Faults"],
        "event_scope_modes": [
            "All events",
            "Only within time filter window",
            "Only within matched draws window",
        ],
        "maintenance_components": components["maintenance"],
        "fault_components": components["faults"],
        "query_templates": [
            {"key": "overview", "label": "Overview", "sql": "SELECT section, COUNT(*) AS rows FROM dataset GROUP BY section ORDER BY rows DESC;"},
            {"key": "order", "label": "Order Params", "sql": "SELECT parameter_name, value, units FROM dataset WHERE group_name = 'Order' ORDER BY parameter_name;"},
            {"key": "numeric", "label": "Numeric", "sql": "SELECT parameter_name, value_num, units FROM dataset WHERE value_num IS NOT NULL ORDER BY ABS(value_num) DESC LIMIT 25;"},
        ],
    }


def sql_value_matches(row: dict[str, object], operator: str, value1: str, value2: str) -> bool:
    operator = str(operator or "any").strip().lower()
    raw_value = str(row.get("value", "")).strip()
    numeric_value = row.get("value_num")
    value1 = str(value1 or "").strip()
    value2 = str(value2 or "").strip()
    if operator == "any":
        return True
    if operator == "contains":
        return bool(value1) and value1.lower() in raw_value.lower()
    if operator == "between":
        if not value1 or not value2:
            return False
        if numeric_value is not None and value1.replace(".", "", 1).replace("-", "", 1).isdigit() and value2.replace(".", "", 1).replace("-", "", 1).isdigit():
            low, high = sorted([float(value1), float(value2)])
            return low <= float(numeric_value) <= high
        return value1 <= raw_value <= value2
    if not value1:
        return False
    if numeric_value is not None and value1.replace(".", "", 1).replace("-", "", 1).isdigit():
        left = float(numeric_value)
        right = float(value1)
    else:
        left = raw_value
        right = value1
    if operator == "=":
        return left == right
    if operator == "!=":
        return left != right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    return False


def sql_eval_group_condition(draw_rows: list[dict[str, object]], condition: dict[str, object]) -> bool:
    params = dedupe_strings(condition.get("params", []) or [])
    if not params:
        return False
    param_results = []
    for param in params:
        matching_rows = [row for row in draw_rows if str(row.get("parameter_name", "")).strip() == param]
        result = any(sql_value_matches(row, str(condition.get("op", "any")), str(condition.get("v1", "")), str(condition.get("v2", ""))) for row in matching_rows)
        param_results.append(result)
    group_logic = str(condition.get("groupLogic", "ANY (OR)"))
    outcome = all(param_results) if group_logic.startswith("ALL") else any(param_results)
    if condition.get("negate"):
        return not outcome
    return outcome


def sql_filter_dataset_records(payload: dict) -> dict:
    dataset_name = str(payload.get("dataset", "__ALL__")).strip() or "__ALL__"
    _, records = load_dataset_scope_records(dataset_name)
    by_draw: dict[str, list[dict[str, object]]] = {}
    for row in records:
        by_draw.setdefault(str(row.get("_draw", "")), []).append(row)

    conditions = payload.get("conditions", []) or []
    time_enabled = bool(payload.get("timeEnabled"))
    time_from = str(payload.get("timeFrom", "")).strip()
    time_to = str(payload.get("timeTo", "")).strip()
    include_draws = bool(payload.get("includeDraws", True))
    include_maintenance = bool(payload.get("includeMaintenance"))
    include_faults = bool(payload.get("includeFaults"))
    event_scope = str(payload.get("eventScope", "Only within matched draws window")).strip()

    matched_draws = []
    matched_values = []
    for draw_id, draw_rows in by_draw.items():
        draw_ts = parse_dt(str(draw_rows[0].get("event_ts", "")))
        if time_enabled and time_from and time_to and draw_ts:
            start_dt = datetime.fromisoformat(time_from)
            end_dt = datetime.fromisoformat(time_to) + timedelta(days=1) - timedelta(seconds=1)
            if not (start_dt <= draw_ts <= end_dt):
                continue
        if conditions:
            overall = None
            for index, condition in enumerate(conditions):
                result = sql_eval_group_condition(draw_rows, condition)
                joiner = str(condition.get("joiner", "AND")).upper()
                if index == 0 or overall is None:
                    overall = result
                elif joiner == "OR":
                    overall = overall or result
                else:
                    overall = overall and result
            if not overall:
                continue
        event_ts = str(draw_rows[0].get("event_ts", ""))
        matched_draws.append(
            {
                "_draw": draw_id,
                "event_ts": event_ts,
                "filename": str(draw_rows[0].get("filename", "")),
            }
        )
        if conditions:
            for condition in conditions:
                params = dedupe_strings(condition.get("params", []) or [])
                for row in draw_rows:
                    if params and str(row.get("parameter_name", "")) not in params:
                        continue
                    if sql_value_matches(row, str(condition.get("op", "any")), str(condition.get("v1", "")), str(condition.get("v2", ""))):
                        matched_values.append(
                            {
                                "_draw": draw_id,
                                "parameter_name": str(row.get("parameter_name", "")),
                                "value": str(row.get("value", "")),
                                "units": str(row.get("units", "")),
                                "event_ts": event_ts,
                            }
                        )

    matched_draws.sort(key=lambda item: (item["event_ts"], item["_draw"]))
    matched_draw_ids = {item["_draw"] for item in matched_draws}
    matched_values = matched_values[:400]

    scope_start = None
    scope_end = None
    if matched_draws and event_scope == "Only within matched draws window":
        scope_times = [parse_dt(item["event_ts"]) for item in matched_draws if parse_dt(item["event_ts"])]
        if scope_times:
            scope_start, scope_end = min(scope_times), max(scope_times)
    elif time_enabled and time_from and time_to and event_scope == "Only within time filter window":
        scope_start = datetime.fromisoformat(time_from)
        scope_end = datetime.fromisoformat(time_to) + timedelta(days=1) - timedelta(seconds=1)

    maintenance_rows = []
    if include_maintenance:
        maint_text = str(payload.get("maintenanceText", "")).strip().lower()
        maint_component = str(payload.get("maintenanceComponent", "")).strip().lower()
        for row in read_csv_rows(MAINTENANCE_ACTIONS):
            ts = parse_dt(str(row.get("action_ts", "")))
            if scope_start and scope_end and ts and not (scope_start <= ts <= scope_end):
                continue
            haystack = " ".join(
                [
                    str(row.get("component", "")),
                    str(row.get("task", "")),
                    str(row.get("note", "")),
                    str(row.get("source_file", "")),
                ]
            ).lower()
            if maint_text and maint_text not in haystack:
                continue
            if maint_component and maint_component not in str(row.get("component", "")).lower():
                continue
            maintenance_rows.append(
                {
                    "event_id": str(row.get("action_id", "")),
                    "event_ts": str(row.get("action_ts", "")),
                    "component": str(row.get("component", "")),
                    "title": str(row.get("task", "")),
                    "note": str(row.get("note", "")),
                    "source_file": str(row.get("source_file", "")),
                }
            )

    fault_rows = []
    if include_faults:
        fault_text = str(payload.get("faultText", "")).strip().lower()
        fault_component = str(payload.get("faultComponent", "")).strip().lower()
        fault_severity = str(payload.get("faultSeverity", "")).strip().lower()
        for row in read_csv_rows(FAULTS_LOG):
            ts = parse_dt(str(row.get("fault_ts", "")))
            if scope_start and scope_end and ts and not (scope_start <= ts <= scope_end):
                continue
            haystack = " ".join(
                [
                    str(row.get("fault_component", "")),
                    str(row.get("fault_title", "")),
                    str(row.get("fault_description", "")),
                    str(row.get("fault_source_file", "")),
                ]
            ).lower()
            if fault_text and fault_text not in haystack:
                continue
            if fault_component and fault_component not in str(row.get("fault_component", "")).lower():
                continue
            if fault_severity and fault_severity != str(row.get("fault_severity", "")).strip().lower():
                continue
            fault_rows.append(
                {
                    "event_id": str(row.get("fault_id", "")),
                    "event_ts": str(row.get("fault_ts", "")),
                    "component": str(row.get("fault_component", "")),
                    "title": str(row.get("fault_title", "")),
                    "severity": str(row.get("fault_severity", "")),
                    "description": str(row.get("fault_description", "")),
                    "source_file": str(row.get("fault_source_file", "")),
                }
            )

    return {
        "ok": True,
        "summary": {
            "matched_draws": len(matched_draws),
            "matched_values": len(matched_values),
            "maintenance_events": len(maintenance_rows),
            "fault_events": len(fault_rows),
            "draw_scope": len(matched_draw_ids),
        },
        "matched_draws": matched_draws[:200] if include_draws else [],
        "matched_values": matched_values,
        "maintenance_events": maintenance_rows[:120],
        "fault_events": fault_rows[:120],
    }


def run_sql_lab_filter_action(payload: dict) -> JsonResponse:
    try:
        return JsonResponse(sql_filter_dataset_records(payload))
    except Exception as exc:
        return JsonResponse({"ok": False, "message": str(exc)}, 400)


def run_sql_lab_analysis_scope_action(payload: dict) -> JsonResponse:
    filenames = dedupe_strings([str(item).strip() for item in (payload.get("filenames") or []) if str(item).strip()])
    if not filenames:
        return JsonResponse(
            {
                "ok": True,
                "records": [],
                "draw_count": 0,
                "row_count": 0,
                "grouped_labels": [],
            }
        )
    files, records = load_dataset_records_for_filenames(filenames[:120])
    return JsonResponse(
        {
            "ok": True,
            "records": records[:30000],
            "draw_count": len(files),
            "row_count": len(records),
            "grouped_labels": sorted(
                {
                    str(row.get("parameter_name", "")).strip()
                    for row in records
                    if str(row.get("parameter_name", "")).strip()
                },
                key=str.lower,
            )[:2000],
        }
    )


def summarize_development() -> dict:
    projects, experiments, updates = read_development_tables()
    active_projects = [row for row in projects if str(row.get("Archived", "")).strip().lower() not in {"true", "1", "yes"}]
    archived_projects = [row for row in projects if str(row.get("Archived", "")).strip().lower() in {"true", "1", "yes"}]
    project_names = [row.get("Project Name", "") for row in active_projects if row.get("Project Name", "")]
    latest_updates = sorted(updates, key=lambda row: str(row.get("Update Date", "")), reverse=True)[:12]
    return {
        "metrics": [
            {"label": "Projects", "value": len(project_names)},
            {"label": "Active", "value": len(active_projects)},
        ],
        "project_names": project_names,
        "archived_project_names": [row.get("Project Name", "") for row in archived_projects if row.get("Project Name", "")],
        "default_project": project_names[0] if project_names else "",
        "latest_updates": latest_updates,
        "dataset_files": [path.name for path in list_csv_files(DATASET_DIR)[:60]],
    }


def run_sql_lab_query_action(payload: dict) -> JsonResponse:
    dataset_name = str(payload.get("dataset", "")).strip()
    sql = str(payload.get("sql", "")).strip()
    if not dataset_name:
        return JsonResponse({"ok": False, "message": "Choose a dataset first."}, 400)
    if not sql:
        return JsonResponse({"ok": False, "message": "SQL is empty."}, 400)
    normalized_sql = sql.lstrip().lower()
    if not (normalized_sql.startswith("select") or normalized_sql.startswith("with")):
        return JsonResponse({"ok": False, "message": "Only SELECT queries are allowed in the rebuild SQL Lab."}, 400)
    _, normalized_rows = load_dataset_scope_records(dataset_name)
    if not normalized_rows:
        return JsonResponse({"ok": False, "message": "Selected dataset scope was not found."}, 404)
    conn = duckdb.connect(":memory:") if duckdb else sqlite3.connect(":memory:")
    try:
        df = pd.DataFrame(normalized_rows)
        if duckdb:
            conn.register("dataset_df", df)
            conn.execute("CREATE TABLE dataset AS SELECT * FROM dataset_df")
        else:
            conn.execute(
                """
                CREATE TABLE dataset (
                    row_index INTEGER,
                    section TEXT,
                    parameter_name TEXT,
                    group_name TEXT,
                    key_name TEXT,
                    value TEXT,
                    units TEXT,
                    value_num REAL,
                    _draw TEXT,
                    filename TEXT,
                    event_ts TEXT
                )
                """
            )
            conn.executemany(
                "INSERT INTO dataset VALUES (:row_index, :section, :parameter_name, :group_name, :key_name, :value, :units, :value_num, :_draw, :filename, :event_ts)",
                normalized_rows,
            )
        limited_sql = sql.rstrip().rstrip(";")
        if " limit " not in normalized_sql:
            limited_sql = f"{limited_sql} LIMIT 200"
        cursor = conn.execute(limited_sql)
        columns = [item[0] for item in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse({"ok": True, "columns": columns, "rows": rows, "row_count": len(rows)})
    except Exception as exc:
        return JsonResponse({"ok": False, "message": str(exc)}, 400)
    finally:
        conn.close()


def append_project_if_missing(project_name: str) -> None:
    project_name = str(project_name or "").strip()
    if not project_name:
        return
    rows = read_csv_rows(PROJECTS_FIBER)
    existing = dedupe_strings([row.get(PROJECTS_COL, "") for row in rows])
    if project_name in existing:
        return
    fieldnames = read_csv_fieldnames(PROJECTS_FIBER) or [PROJECTS_COL]
    rows.append({PROJECTS_COL: project_name})
    write_csv_rows(PROJECTS_FIBER, rows, fieldnames)


def save_project_template(payload: dict) -> None:
    project_name = str(payload.get(PROJECTS_COL, "")).strip()
    if not project_name:
        return
    rows = read_csv_rows(PROJECTS_TEMPLATES)
    fieldnames = read_csv_fieldnames(PROJECTS_TEMPLATES) or TEMPLATE_FIELDS[:]
    fieldnames = fieldnames + [field for field in TEMPLATE_FIELDS if field not in fieldnames]
    template_row = {field: payload.get(field, "") for field in fieldnames}
    replaced = False
    for index, row in enumerate(rows):
        if str(row.get(PROJECTS_COL, "")).strip() == project_name:
            rows[index] = {field: template_row.get(field, row.get(field, "")) for field in fieldnames}
            replaced = True
            break
    if not replaced:
        rows.append(template_row)
    write_csv_rows(PROJECTS_TEMPLATES, rows, fieldnames)


def validate_order_payload(order: dict) -> list[str]:
    errors = []
    if not str(order.get("preformNumber", "")).strip():
        errors.append("Preform Number")
    if not str(order.get("project", "")).strip():
        errors.append("Fiber Project")
    if not str(order.get("opener", "")).strip():
        errors.append("Order Opened By")
    if to_float(order.get("requiredLength")) <= 0:
        errors.append("Required Length (m)")
    if int(to_float(order.get("goodZones") or 0)) <= 0:
        errors.append("Good Zones Count")
    geometry = str(order.get("geometry", "")).strip()
    if not geometry:
        errors.append(GEOMETRY_COL)
    if geometry == "TIGER - PM" and to_float(order.get("tigerCut")) <= 0:
        errors.append("Tiger Cut (%)")
    if geometry == "Octagonal" and to_float(order.get("octF2f")) <= 0:
        errors.append("Octagonal F2F (mm)")
    return errors


def write_schedule_entry(order_row: dict[str, str], order_index: int, schedule_payload: dict, preform_number: str) -> None:
    date_value = str(schedule_payload.get("date", "")).strip()
    time_value = str(schedule_payload.get("startTime", "")).strip()
    duration_minutes = int(to_float(schedule_payload.get("durationMin") or 0))
    start = datetime.fromisoformat(f"{date_value}T{time_value}")
    end = start + timedelta(minutes=duration_minutes)
    existing = read_csv_rows(TOWER_SCHEDULE)
    fieldnames = read_csv_fieldnames(TOWER_SCHEDULE) or SCHEDULE_REQUIRED_COLS[:]
    fieldnames = fieldnames + [field for field in SCHEDULE_REQUIRED_COLS if field not in fieldnames]
    existing.append(
        {
            "Event Type": "Drawing",
            "Start DateTime": start.strftime("%Y-%m-%d %H:%M:%S"),
            "End DateTime": end.strftime("%Y-%m-%d %H:%M:%S"),
            "Description": build_schedule_description(order_row, order_index, preform_number),
            "Recurrence": "None",
        }
    )
    write_csv_rows(TOWER_SCHEDULE, existing, fieldnames)


def create_order_draw_action(payload: dict) -> JsonResponse:
    order = payload.get("order", {})
    schedule_now = bool(payload.get("scheduleNow"))
    schedule_payload = payload.get("schedule", {}) or {}
    save_template = bool(payload.get("saveTemplate"))
    errors = validate_order_payload(order)
    if errors:
        return JsonResponse({"ok": False, "message": "Missing required fields: " + ", ".join(errors)}, 400)
    project_name = str(order.get("project", "")).strip()
    append_project_if_missing(project_name)
    existing = read_csv_rows(DRAW_ORDERS)
    fieldnames = read_csv_fieldnames(DRAW_ORDERS) or []
    new_row = {
        "Status": "Pending",
        "Priority": str(order.get("priority", "Normal")).strip() or "Normal",
        "Order Opener": str(order.get("opener", "")).strip(),
        "Preform Number": str(order.get("preformNumber", "")).strip(),
        PROJECTS_COL: project_name,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "Fiber Diameter (µm)": to_float(order.get("fiberDiameter")),
        "Main Coating Diameter (µm)": to_float(order.get("mainCoatingDiameter")),
        "Secondary Coating Diameter (µm)": to_float(order.get("secondaryCoatingDiameter")),
        "Tension (g)": to_float(order.get("tension")),
        "Draw Speed (m/min)": to_float(order.get("drawSpeed")),
        LENGTH_COL: to_float(order.get("requiredLength")),
        GOOD_ZONES_COL: int(to_float(order.get("goodZones") or 0)),
        "Main Coating": str(order.get("mainCoating", "")).strip(),
        "Secondary Coating": str(order.get("secondaryCoating", "")).strip(),
        "Notes": str(order.get("notes", "")).strip(),
        "Desired Date": str(order.get("desiredDate", "")).strip(),
        "Next Planned Draw Date": "",
        GEOMETRY_COL: str(order.get("geometry", "")).strip(),
        TIGER_CUT_COL: to_float(order.get("tigerCut")),
        OCT_F2F_COL: to_float(order.get("octF2f")),
        MAIN_TEMP_COL: to_float(order.get("mainCoatingTemp")),
        SECONDARY_TEMP_COL: to_float(order.get("secondaryCoatingTemp")),
        FIBER_TOL_COL: to_float(order.get("fiberTol")),
        MAIN_TOL_COL: to_float(order.get("mainTol")),
        SECONDARY_TOL_COL: to_float(order.get("secondaryTol")),
        "Active CSV": "",
        "Done CSV": "",
        "Done Description": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }
    fieldnames = fieldnames or list(new_row.keys())
    fieldnames = fieldnames + [field for field in new_row.keys() if field not in fieldnames]
    existing.append({field: new_row.get(field, "") for field in fieldnames})
    order_index = len(existing) - 1
    message = "Draw order saved."
    if schedule_now:
        schedule_ready = (
            str(schedule_payload.get("password", "")).strip() == SCHEDULE_PASSWORD
            and str(schedule_payload.get("date", "")).strip()
            and str(schedule_payload.get("startTime", "")).strip()
            and int(to_float(schedule_payload.get("durationMin") or 0)) > 0
        )
        if schedule_ready:
            write_schedule_entry(new_row, order_index, schedule_payload, new_row["Preform Number"])
            existing[order_index]["Status"] = "Scheduled"
            message = "Draw order saved and scheduled."
        else:
            message = "Draw order saved, but not scheduled because the schedule details or password were invalid."
    write_csv_rows(DRAW_ORDERS, existing, fieldnames)
    if save_template:
        save_project_template(
            {
                PROJECTS_COL: project_name,
                GEOMETRY_COL: new_row[GEOMETRY_COL],
                TIGER_CUT_COL: new_row[TIGER_CUT_COL],
                OCT_F2F_COL: new_row[OCT_F2F_COL],
                "Fiber Diameter (µm)": new_row["Fiber Diameter (µm)"],
                FIBER_TOL_COL: new_row[FIBER_TOL_COL],
                "Main Coating Diameter (µm)": new_row["Main Coating Diameter (µm)"],
                MAIN_TOL_COL: new_row[MAIN_TOL_COL],
                "Secondary Coating Diameter (µm)": new_row["Secondary Coating Diameter (µm)"],
                SECONDARY_TOL_COL: new_row[SECONDARY_TOL_COL],
                "Tension (g)": new_row["Tension (g)"],
                "Draw Speed (m/min)": new_row["Draw Speed (m/min)"],
                "Main Coating": new_row["Main Coating"],
                "Secondary Coating": new_row["Secondary Coating"],
                MAIN_TEMP_COL: new_row[MAIN_TEMP_COL],
                SECONDARY_TEMP_COL: new_row[SECONDARY_TEMP_COL],
                "Notes Default": new_row["Notes"],
            }
        )
    return JsonResponse({"ok": True, "message": message, "bootstrap": build_bootstrap_payload().body})


def save_order_draw_template_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("project", "")).strip()
    if not project_name:
        return JsonResponse({"ok": False, "message": "Choose a project before saving a template."}, 400)
    append_project_if_missing(project_name)
    save_project_template(
        {
            PROJECTS_COL: project_name,
            GEOMETRY_COL: str(payload.get("geometry", "")).strip(),
            TIGER_CUT_COL: to_float(payload.get("tigerCut")),
            OCT_F2F_COL: to_float(payload.get("octF2f")),
            "Fiber Diameter (µm)": to_float(payload.get("fiberDiameter")),
            FIBER_TOL_COL: to_float(payload.get("fiberTol")),
            "Main Coating Diameter (µm)": to_float(payload.get("mainCoatingDiameter")),
            MAIN_TOL_COL: to_float(payload.get("mainTol")),
            "Secondary Coating Diameter (µm)": to_float(payload.get("secondaryCoatingDiameter")),
            SECONDARY_TOL_COL: to_float(payload.get("secondaryTol")),
            "Tension (g)": to_float(payload.get("tension")),
            "Draw Speed (m/min)": to_float(payload.get("drawSpeed")),
            "Main Coating": str(payload.get("mainCoating", "")).strip(),
            "Secondary Coating": str(payload.get("secondaryCoating", "")).strip(),
            MAIN_TEMP_COL: to_float(payload.get("mainCoatingTemp")),
            SECONDARY_TEMP_COL: to_float(payload.get("secondaryCoatingTemp")),
            "Notes Default": str(payload.get("notes", "")).strip(),
        }
    )
    return JsonResponse({"ok": True, "message": "Template saved.", "bootstrap": build_bootstrap_payload().body})


def schedule_pending_order_action(payload: dict) -> JsonResponse:
    password = str(payload.get("password", "")).strip()
    if password != SCHEDULE_PASSWORD:
        return JsonResponse({"ok": False, "message": "Scheduling password is missing or wrong."}, 400)
    order_index = int(to_float(payload.get("orderIndex")))
    orders = read_csv_rows(DRAW_ORDERS)
    if order_index < 0 or order_index >= len(orders):
        return JsonResponse({"ok": False, "message": "Selected order was not found."}, 404)
    row = orders[order_index]
    if str(row.get("Status", "")).strip() != "Pending":
        return JsonResponse({"ok": False, "message": "Only pending orders can be scheduled here."}, 400)
    preform_number = str(payload.get("preformNumber", "")).strip() or str(row.get("Preform Number", "")).strip()
    if not preform_number or preform_number == "0":
        return JsonResponse({"ok": False, "message": "A real preform number is required before scheduling."}, 400)
    write_schedule_entry(row, order_index, payload, preform_number)
    row["Preform Number"] = preform_number
    row["Status"] = "Scheduled"
    fieldnames = read_csv_fieldnames(DRAW_ORDERS) or list(row.keys())
    write_csv_rows(DRAW_ORDERS, orders, fieldnames)
    return JsonResponse({"ok": True, "message": "Pending order scheduled.", "bootstrap": build_bootstrap_payload().body})


def append_dataset_rows(selected_csv: str, rows: list[dict]) -> tuple[bool, str]:
    if not selected_csv:
        return False, "No dataset CSV selected."
    csv_path = DATA_DIR.parent / "data_set_csv" / os.path.basename(selected_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_csv_rows(csv_path)
    fieldnames = ["Parameter Name", "Value", "Units"]
    normalized_existing = []
    for row in existing:
        normalized_existing.append({field: row.get(field, "") for field in fieldnames})
    normalized_new = [{field: row.get(field, "") for field in fieldnames} for row in rows]
    write_csv_rows(csv_path, normalized_existing + normalized_new, fieldnames)
    return True, f"Saved {len(normalized_new)} rows into {csv_path.name}"


def build_dashboard_zone_rows(log_data: dict, zones: list[dict]) -> list[dict]:
    rows = []
    rows.append({"Parameter Name": "—", "Value": "—", "Units": ""})
    rows.append({"Parameter Name": "Dashboard Log File", "Value": log_data.get("selected_file", ""), "Units": ""})
    rows.append({"Parameter Name": "Good Zones Count", "Value": len(zones), "Units": "count"})
    rows.append({"Parameter Name": "Good Zones X Column", "Value": log_data.get("suggested_x", ""), "Units": ""})
    length_col = log_data.get("length_column", "")
    all_rows = log_data.get("rows", [])
    numeric_columns = log_data.get("numeric_columns", [])
    for zone_number, zone in enumerate(zones, start=1):
        start_index = int(zone.get("startIndex", 0))
        end_index = int(zone.get("endIndex", 0))
        if end_index < start_index:
            start_index, end_index = end_index, start_index
        segment = all_rows[start_index:end_index + 1]
        if not segment:
            continue
        rows.append({"Parameter Name": f"Zone {zone_number} | Start", "Value": segment[0].get("__x", ""), "Units": log_data.get("suggested_x", "")})
        rows.append({"Parameter Name": f"Zone {zone_number} | End", "Value": segment[-1].get("__x", ""), "Units": log_data.get("suggested_x", "")})
        if length_col:
            start_len = to_float(segment[0].get(length_col))
            end_len = to_float(segment[-1].get(length_col))
            rows.append({"Parameter Name": f"Zone {zone_number} | Length span", "Value": abs(end_len - start_len), "Units": "km"})
        for column in numeric_columns[:10]:
            values = [to_float(item.get(column)) for item in segment]
            rows.append({"Parameter Name": f"Zone {zone_number} | {column} | Avg", "Value": sum(values) / max(1, len(values)), "Units": ""})
            rows.append({"Parameter Name": f"Zone {zone_number} | {column} | Min", "Value": min(values), "Units": ""})
            rows.append({"Parameter Name": f"Zone {zone_number} | {column} | Max", "Value": max(values), "Units": ""})
        rows.append({"Parameter Name": "", "Value": "", "Units": ""})
    return rows


def save_dashboard_zones_action(payload: dict) -> JsonResponse:
    log_name = str(payload.get("logName", "")).strip()
    selected_csv = str(payload.get("datasetCsv", "")).strip()
    zones = payload.get("zones", []) or []
    if not zones:
        return JsonResponse({"ok": False, "message": "No saved zones to export."}, 400)
    log_data = analyze_log_file(log_name, sample_limit=1600)
    rows = build_dashboard_zone_rows(log_data, zones)
    ok, message = append_dataset_rows(selected_csv, rows)
    status = 200 if ok else 400
    return JsonResponse({"ok": ok, "message": message}, status)


def export_dashboard_math_plot_action(payload: dict) -> JsonResponse:
    filename = os.path.basename(str(payload.get("filename", "")).strip()) or "tower_math_plot.png"
    content = str(payload.get("content", "")).strip()
    if not content:
        return JsonResponse({"ok": False, "message": "Missing image content."}, 400)
    if content.startswith("data:"):
        _, _, content = content.partition(",")
    try:
        raw = base64.b64decode(content)
    except Exception:
        return JsonResponse({"ok": False, "message": "Invalid image content."}, 400)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("._") or "tower_math_plot.png"
    if not safe_name.lower().endswith(".png"):
        safe_name = f"{safe_name}.png"
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS_DIR / safe_name
    stem = target.stem
    suffix = target.suffix
    version = 2
    while target.exists():
        target = DOWNLOADS_DIR / f"{stem}_{version}{suffix}"
        version += 1
    target.write_bytes(raw)
    return JsonResponse({
        "ok": True,
        "saved_path": str(target),
        "filename": target.name,
        "message": f"Saved plot to {target}",
    })


def build_home_payload() -> JsonResponse:
    draws = summarize_draw_orders()
    schedule = summarize_schedule()
    parts = summarize_part_orders()
    inventory = summarize_inventory()
    maintenance = summarize_maintenance_rebuild()
    payload = {
        "hero": {
            "title": "Tower command deck",
            "subtitle": "Non-Streamlit rebuild foundation",
            "summary": "A real page system with shared shell, page routing, and live Tower data summaries delivered through lightweight Python APIs.",
        },
        "metrics": [
            {"label": "Active Draws", "value": draws["active"]},
            {"label": "Completed", "value": draws["done"]},
            {"label": "Upcoming Events", "value": len(schedule["upcoming"])},
            {"label": "Open Part Orders", "value": len(parts["open_orders"])},
        ],
        "draws": draws,
        "schedule": schedule,
        "parts": parts,
        "inventory": inventory,
        "maintenance": maintenance,
    }
    return JsonResponse(payload)


def build_schedule_payload() -> JsonResponse:
    return JsonResponse(summarize_schedule())


def build_parts_payload() -> JsonResponse:
    payload = summarize_part_orders()
    payload["inventory"] = summarize_inventory()
    payload["manual_lookup"] = summarize_parts_manual_lookup()
    return JsonResponse(payload)


def create_part_order_action(payload: dict) -> JsonResponse:
    rows = read_csv_rows(PART_ORDERS)
    fieldnames = read_csv_fieldnames(PART_ORDERS)
    if not fieldnames:
        fieldnames = [
            "Status", "Part Name", "Serial Number", "Project Name", "Details", "Opened By",
            "Approval Requested From", "Approved", "Approved By", "Approval Date", "Received Date",
            "Received State", "Ordered By", "Date Ordered", "Company", "Inventory Synced",
            "Maintenance Component", "Maintenance Task", "Maintenance Task ID", "Wait ID",
        ]
    part_name = str(payload.get("partName", "")).strip()
    if not part_name:
        return JsonResponse({"ok": False, "message": "Part name is required."}, 400)
    new_row = {
        "Status": "Opened",
        "Part Name": part_name,
        "Serial Number": str(payload.get("serialNumber", "")).strip(),
        "Project Name": str(payload.get("project", "")).strip(),
        "Details": str(payload.get("details", "")).strip(),
        "Opened By": str(payload.get("openedBy", "")).strip(),
        "Approval Requested From": str(payload.get("approvalRequestedFrom", "")).strip(),
        "Approved": "",
        "Approved By": "",
        "Approval Date": "",
        "Received Date": "",
        "Received State": "",
        "Ordered By": "",
        "Date Ordered": "",
        "Company": str(payload.get("company", "")).strip(),
        "Inventory Synced": "",
        "Maintenance Component": str(payload.get("maintenanceComponent", "")).strip(),
        "Maintenance Task": str(payload.get("maintenanceTask", "")).strip(),
        "Maintenance Task ID": str(payload.get("maintenanceTaskId", "")).strip(),
        "Wait ID": str(payload.get("waitId", "")).strip(),
    }
    rows.append(new_row)
    write_csv_rows(PART_ORDERS, rows, fieldnames)
    ensure_parts_company(new_row["Company"])
    return JsonResponse({"ok": True, "message": "Part order saved.", "bootstrap": build_bootstrap_payload().body})


def update_part_order_action(payload: dict) -> JsonResponse:
    rows = read_csv_rows(PART_ORDERS)
    fieldnames = read_csv_fieldnames(PART_ORDERS)
    try:
        index = int(payload.get("index"))
    except (TypeError, ValueError):
        return JsonResponse({"ok": False, "message": "Order selection is invalid."}, 400)
    if index < 0 or index >= len(rows):
        return JsonResponse({"ok": False, "message": "Order not found."}, 400)
    row = rows[index]
    current_status = str(row.get("Status", "Opened")).strip() or "Opened"
    status = str(payload.get("status", current_status)).strip() or current_status
    if status not in PART_STATUS_ORDER:
        status = current_status
    if current_status in PART_STATUS_ORDER and status in PART_STATUS_ORDER:
        cur_idx = PART_STATUS_ORDER.index(current_status)
        allowed = PART_STATUS_ORDER[cur_idx + 1:] or [current_status]
        if status not in allowed:
            status = allowed[0]
    row["Status"] = status
    for source_key, field_name in [
        ("partName", "Part Name"),
        ("serialNumber", "Serial Number"),
        ("project", "Project Name"),
        ("details", "Details"),
        ("openedBy", "Opened By"),
        ("approvalRequestedFrom", "Approval Requested From"),
        ("approvedBy", "Approved By"),
        ("approvalDate", "Approval Date"),
        ("receivedDate", "Received Date"),
        ("receivedState", "Received State"),
        ("orderedBy", "Ordered By"),
        ("dateOrdered", "Date Ordered"),
        ("company", "Company"),
        ("inventorySynced", "Inventory Synced"),
        ("maintenanceComponent", "Maintenance Component"),
        ("maintenanceTask", "Maintenance Task"),
        ("maintenanceTaskId", "Maintenance Task ID"),
        ("waitId", "Wait ID"),
    ]:
        if source_key in payload:
            row[field_name] = str(payload.get(source_key, "")).strip()
    if status in {"Approved", "Ordered", "Received", "Archived"}:
        row["Approved"] = "Yes"
    elif status in {"Opened", "Wait for Approval"}:
        row["Approved"] = "No"
    if status == "Wait for Approval" and not str(row.get("Approval Requested From", "")).strip():
        row["Approval Requested From"] = str(payload.get("approvalRequestedFrom", "")).strip()
    is_approved_step = status in {"Approved", "Ordered", "Received", "Archived"}
    is_ordered_step = status in {"Ordered", "Received", "Archived"}
    is_received_step = status in {"Received", "Archived"}
    if not is_approved_step:
        row["Approved By"] = ""
        row["Approval Date"] = ""
    if not is_ordered_step:
        row["Ordered By"] = ""
        row["Date Ordered"] = ""
        row["Company"] = str(payload.get("company", row.get("Company", ""))).strip() if status == "Wait for Approval" else ""
    if not is_received_step:
        row["Received Date"] = ""
        row["Received State"] = ""
        row["Inventory Synced"] = ""
    inventory_action = str(payload.get("inventoryAction", "")).strip()
    if is_received_step:
        if inventory_action == "Locate in inventory":
            row["Received State"] = "Located in inventory"
            row["Inventory Synced"] = "Yes"
        elif inventory_action == "Mount on machine":
            row["Received State"] = "Mounted on machine"
            row["Inventory Synced"] = "Yes"
        elif not str(row.get("Received State", "")).strip():
            row["Received State"] = "Waiting for inventory action"
            row["Inventory Synced"] = "Pending"
        elif str(row.get("Received State", "")).strip() == "Waiting for inventory action":
            row["Inventory Synced"] = "Pending"
    if status == "Archived" and inventory_action in {"Locate in inventory", "Mount on machine"}:
        sync_part_order_into_inventory(row, payload)
    ensure_parts_company(row.get("Company", ""))
    rows[index] = row
    write_csv_rows(PART_ORDERS, rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Part order updated.", "bootstrap": build_bootstrap_payload().body})


def build_maintenance_payload() -> JsonResponse:
    return JsonResponse(summarize_maintenance_rebuild())


def build_order_draw_payload() -> JsonResponse:
    return JsonResponse(summarize_order_draw())


def build_dashboard_payload() -> JsonResponse:
    return JsonResponse(summarize_dashboard())


def build_dashboard_log_payload(log_name: str | None = None) -> JsonResponse:
    return JsonResponse(analyze_log_file(log_name))


def build_report_center_payload() -> JsonResponse:
    return JsonResponse(summarize_report_center())


def build_report_center_file_payload(file_name: str | None = None) -> Path | None:
    ensure_report_center_dir()
    requested = os.path.basename(str(file_name or "").strip())
    if not requested:
        return None
    candidate = (REPORT_CENTER_DIR / requested).resolve()
    if REPORT_CENTER_DIR.resolve() not in candidate.parents or not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def build_report_center_project_payload(project_name: str | None = None) -> JsonResponse:
    project_name = str(project_name or "").strip()
    return JsonResponse(summarize_development_project(project_name) if project_name else {})


def build_sql_lab_payload() -> JsonResponse:
    return JsonResponse(summarize_sql_lab())


def build_sql_lab_dataset_payload(dataset_name: str | None = None) -> JsonResponse:
    return JsonResponse(analyze_dataset_file(dataset_name))


def build_consumables_payload() -> JsonResponse:
    return JsonResponse(summarize_consumables())


def build_process_setup_payload() -> JsonResponse:
    return JsonResponse(summarize_process_setup())


def build_draw_finalize_payload(selected_csv_override: str | None = None) -> JsonResponse:
    return JsonResponse(summarize_draw_finalize(selected_csv_override))


def build_development_payload() -> JsonResponse:
    return JsonResponse(summarize_development())


def build_development_project_payload(project_name: str | None = None) -> JsonResponse:
    project_name = str(project_name or "").strip()
    return JsonResponse(summarize_development_project(project_name) if project_name else {})


def build_diagnostics_payload() -> JsonResponse:
    return JsonResponse(summarize_diagnostics())


def save_diagnostics_paths_action(payload: dict) -> JsonResponse:
    reset_defaults = str(payload.get("resetDefaults", "")).strip().lower() in {"1", "true", "yes"}
    overrides: dict[str, Path] = {}
    for item in TRACKED_PATH_SPECS:
        key = str(item["key"])
        label = str(item["label"])
        kind = str(item["kind"])
        raw_value = "" if reset_defaults else str(payload.get(key, "")).strip()
        if not reset_defaults and not raw_value:
            return JsonResponse({"ok": False, "message": f"{label} is required."}, 400)
        if reset_defaults:
            continue
        normalized = normalize_tracked_path_value(raw_value)
        if normalized.exists():
            if kind == "dir" and not normalized.is_dir():
                return JsonResponse({"ok": False, "message": f"{label} must point to a folder."}, 400)
            if kind == "file" and not normalized.is_file():
                return JsonResponse({"ok": False, "message": f"{label} must point to a file path, not a folder."}, 400)
        if kind == "dir":
            normalized.mkdir(parents=True, exist_ok=True)
        else:
            normalized.parent.mkdir(parents=True, exist_ok=True)
        overrides[key] = normalized
    save_tracked_path_overrides({} if reset_defaults else overrides)
    apply_tracked_path_overrides({} if reset_defaults else overrides)
    return JsonResponse(
        {
            "ok": True,
            "message": "Tracked paths reset to Tower defaults." if reset_defaults else "Tracked paths saved and applied to the Python app.",
            "bootstrap": build_bootstrap_payload().body,
        }
    )


def create_full_backup_action(payload: dict | None = None) -> JsonResponse:
    snapshot = create_full_backup_snapshot(trigger="manual")
    return JsonResponse(
        {
            "ok": True,
            "message": f"Full backup created: {snapshot['name']}.",
            "snapshot": snapshot,
            "bootstrap": build_bootstrap_payload().body,
        }
    )


def create_operations_report_action(payload: dict) -> JsonResponse:
    ensure_report_center_dir()
    title = str(payload.get("title", "")).strip() or "Tower Operations Report"
    start_date = str(payload.get("startDate", "")).strip()
    end_date = str(payload.get("endDate", "")).strip()
    sections = [str(item).strip() for item in (payload.get("sections") or []) if str(item).strip()]
    filename = str(payload.get("filename", "")).strip() or f"operations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    if not filename.lower().endswith(".md"):
        filename = f"{filename}.md"
    output_path = REPORT_CENTER_DIR / Path(filename).name
    output_path.write_text(build_operations_report_markdown(title, start_date, end_date, sections), encoding="utf-8")
    return JsonResponse(
        {
            "ok": True,
            "message": f"Operations export saved as {output_path.name}.",
            "bootstrap": build_bootstrap_payload().body,
        }
    )


def create_development_report_action(payload: dict) -> JsonResponse:
    ensure_report_center_dir()
    project_name = str(payload.get("projectName", "")).strip()
    if not project_name:
        return JsonResponse({"ok": False, "message": "Choose a development project first."}, 400)
    details = summarize_development_project(project_name)
    if not details.get("project"):
        return JsonResponse({"ok": False, "message": "Selected project was not found."}, 404)
    export_format = str(payload.get("format", "md")).strip().lower() or "md"
    if export_format not in {"md", "html"}:
        return JsonResponse({"ok": False, "message": "Export format is not supported."}, 400)
    default_suffix = "html" if export_format == "html" else "md"
    default_stem = "project_paper" if export_format == "html" else "project_markdown"
    filename = str(payload.get("filename", "")).strip() or f"{default_stem}_{slugify(project_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{default_suffix}"
    if not filename.lower().endswith(f".{default_suffix}"):
        filename = f"{filename}.{default_suffix}"
    output_path = REPORT_CENTER_DIR / Path(filename).name
    content = build_development_report_html(project_name) if export_format == "html" else build_development_report_markdown(project_name)
    output_path.write_text(content, encoding="utf-8")
    encoded_name = quote(output_path.name)
    return JsonResponse(
        {
            "ok": True,
            "message": f"Project export saved as {output_path.name}.",
            "fileName": output_path.name,
            "fileUrl": f"/api/report-center/file?name={encoded_name}&download=1",
            "viewUrl": f"/api/report-center/file?name={encoded_name}&mode=inline",
            "downloadUrl": f"/api/report-center/file?name={encoded_name}&download=1",
            "format": export_format,
            "bootstrap": build_bootstrap_payload().body,
        }
    )


def create_development_project_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("projectName", "")).strip()
    if not project_name:
        return JsonResponse({"ok": False, "message": "Project name is required."}, 400)
    rows = read_csv_rows(DATA_DIR / "development_projects.csv")
    fieldnames = development_project_fieldnames()
    if any(str(row.get("Project Name", "")).strip() == project_name for row in rows):
        return JsonResponse({"ok": False, "message": "Project already exists."}, 400)
    rows.append(
        {
            "Project Name": project_name,
            "Project Purpose": str(payload.get("purpose", "")).strip(),
            "Target": str(payload.get("target", "")).strip(),
            "Created At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Archived": "False",
            "Summary Title": "",
            "Summary Notes": "",
            "Summary Date": "",
            "Summary Researcher": "",
        }
    )
    write_csv_rows(DATA_DIR / "development_projects.csv", rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Development project created.", "bootstrap": build_bootstrap_payload().body})


def save_development_summary_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("projectName", "")).strip()
    summary_notes = str(payload.get("summaryNotes", "")).strip()
    if not project_name or not summary_notes:
        return JsonResponse({"ok": False, "message": "Project and summary notes are required."}, 400)
    rows = read_csv_rows(DATA_DIR / "development_projects.csv")
    fieldnames = development_project_fieldnames()
    index = next((i for i, row in enumerate(rows) if str(row.get("Project Name", "")).strip() == project_name), None)
    if index is None:
        return JsonResponse({"ok": False, "message": "Project not found."}, 404)
    row = rows[index]
    row["Summary Title"] = str(payload.get("summaryTitle", "")).strip()
    row["Summary Notes"] = summary_notes
    row["Summary Date"] = str(payload.get("summaryDate", "")).strip() or datetime.now().strftime("%Y-%m-%d")
    row["Summary Researcher"] = str(payload.get("summaryResearcher", "")).strip()
    rows[index] = row
    write_csv_rows(DATA_DIR / "development_projects.csv", rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Project summary saved.", "bootstrap": build_bootstrap_payload().body})


def create_development_update_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("projectName", "")).strip()
    notes = str(payload.get("updateNotes", "")).strip()
    if not project_name or not notes:
        return JsonResponse({"ok": False, "message": "Project and update notes are required."}, 400)
    rows = read_csv_rows(DATA_DIR / "experiment_updates.csv")
    fieldnames = read_csv_fieldnames(DATA_DIR / "experiment_updates.csv") or ["Experiment Title", "Update Date", "Researcher", "Update Notes", "Project Name"]
    rows.append(
        {
            "Experiment Title": str(payload.get("updateTitle", "")).strip() or str(payload.get("experimentTitle", "")).strip(),
            "Update Date": str(payload.get("updateDate", "")).strip() or datetime.now().strftime("%Y-%m-%d"),
            "Researcher": str(payload.get("researcher", "")).strip(),
            "Update Notes": notes,
            "Project Name": project_name,
        }
    )
    write_csv_rows(DATA_DIR / "experiment_updates.csv", rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Development update saved.", "bootstrap": build_bootstrap_payload().body})


def create_development_experiment_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("projectName", "")).strip()
    experiment_title = str(payload.get("experimentTitle", "")).strip()
    if not project_name or not experiment_title:
        return JsonResponse({"ok": False, "message": "Project and experiment title are required."}, 400)
    rows = read_csv_rows(DATA_DIR / "development_experiments.csv")
    fieldnames = read_csv_fieldnames(DATA_DIR / "development_experiments.csv") or [
        "Project Name",
        "Experiment Title",
        "Date",
        "Researcher",
        "Methods",
        "Purpose",
        "Observations",
        "Results",
        "Is Drawing",
        "Drawing Details",
        "Draw CSV",
        "Attachments",
        "Attachment Captions",
        "Markdown Notes",
    ]
    if any(
        str(row.get("Project Name", "")).strip() == project_name
        and str(row.get("Experiment Title", "")).strip() == experiment_title
        for row in rows
    ):
        return JsonResponse({"ok": False, "message": "That experiment title already exists in this project."}, 400)
    exp_date = str(payload.get("date", "")).strip() or datetime.now().strftime("%Y-%m-%d")
    attachment_paths: list[str] = [item.strip() for item in str(payload.get("attachments", "")).split(";") if item.strip()]
    upload_items = payload.get("attachmentUploads", []) or []
    if upload_items:
        exp_dir = DEVELOPMENT_MEDIA_DIR / slugify(project_name) / f"{slugify(experiment_title)}__{slugify(exp_date)}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for item in upload_items:
            filename = os.path.basename(str(item.get("name", "")).strip())
            content = str(item.get("content", "")).strip()
            if not filename or not content:
                continue
            try:
                raw = base64.b64decode(content)
            except Exception:
                continue
            candidate = exp_dir / filename
            stem = candidate.stem
            suffix = candidate.suffix
            version = 2
            while candidate.exists():
                candidate = exp_dir / f"{stem}__{version}{suffix}"
                version += 1
            candidate.write_bytes(raw)
            attachment_paths.append(str(candidate))
    rows.append(
        {
            "Project Name": project_name,
            "Experiment Title": experiment_title,
            "Date": exp_date,
            "Researcher": str(payload.get("researcher", "")).strip(),
            "Methods": str(payload.get("methods", "")).strip(),
            "Purpose": str(payload.get("purpose", "")).strip(),
            "Observations": str(payload.get("observations", "")).strip(),
            "Results": str(payload.get("results", "")).strip(),
            "Is Drawing": "True" if payload.get("isDrawing") else "False",
            "Drawing Details": str(payload.get("drawingDetails", "")).strip(),
            "Draw CSV": str(payload.get("drawCsv", "")).strip(),
            "Attachments": ";".join(attachment_paths),
            "Attachment Captions": "",
            "Markdown Notes": str(payload.get("markdownNotes", "")).strip(),
        }
    )
    write_csv_rows(DATA_DIR / "development_experiments.csv", rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Experiment saved.", "bootstrap": build_bootstrap_payload().body})


def update_development_experiment_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("projectName", "")).strip()
    original_title = str(payload.get("originalTitle", "")).strip()
    original_date = str(payload.get("originalDate", "")).strip()
    if not project_name or not original_title or not original_date:
        return JsonResponse({"ok": False, "message": "Choose an existing experiment first."}, 400)
    rows = read_csv_rows(DATA_DIR / "development_experiments.csv")
    fieldnames = read_csv_fieldnames(DATA_DIR / "development_experiments.csv") or [
        "Project Name",
        "Experiment Title",
        "Date",
        "Researcher",
        "Methods",
        "Purpose",
        "Observations",
        "Results",
        "Is Drawing",
        "Drawing Details",
        "Draw CSV",
        "Attachments",
        "Attachment Captions",
        "Markdown Notes",
    ]
    index = next(
        (
            i for i, row in enumerate(rows)
            if str(row.get("Project Name", "")).strip() == project_name
            and str(row.get("Experiment Title", "")).strip() == original_title
            and str(row.get("Date", "")).strip() == original_date
        ),
        None,
    )
    if index is None:
        return JsonResponse({"ok": False, "message": "Experiment not found."}, 404)
    row = rows[index]
    for source_key, field_name in [
        ("researcher", "Researcher"),
        ("methods", "Methods"),
        ("purpose", "Purpose"),
        ("observations", "Observations"),
        ("results", "Results"),
        ("drawingDetails", "Drawing Details"),
        ("drawCsv", "Draw CSV"),
        ("markdownNotes", "Markdown Notes"),
        ("attachments", "Attachments"),
    ]:
        if source_key in payload:
            row[field_name] = str(payload.get(source_key, "")).strip()
    row["Is Drawing"] = "True" if payload.get("isDrawing") else "False"
    rows[index] = row
    write_csv_rows(DATA_DIR / "development_experiments.csv", rows, fieldnames)
    return JsonResponse({"ok": True, "message": "Experiment updated.", "bootstrap": build_bootstrap_payload().body})


def manage_development_project_action(payload: dict) -> JsonResponse:
    project_name = str(payload.get("projectName", "")).strip()
    action = str(payload.get("action", "")).strip().lower()
    if not project_name:
        return JsonResponse({"ok": False, "message": "Choose a project first."}, 400)
    rows = read_csv_rows(DATA_DIR / "development_projects.csv")
    fieldnames = development_project_fieldnames()
    index = next((i for i, row in enumerate(rows) if str(row.get("Project Name", "")).strip() == project_name), None)
    if index is None:
        return JsonResponse({"ok": False, "message": "Project not found."}, 404)
    if action == "archive":
        rows[index]["Archived"] = "True"
        write_csv_rows(DATA_DIR / "development_projects.csv", rows, fieldnames)
        return JsonResponse({"ok": True, "message": "Project archived.", "bootstrap": build_bootstrap_payload().body})
    if action == "restore":
        rows[index]["Archived"] = "False"
        write_csv_rows(DATA_DIR / "development_projects.csv", rows, fieldnames)
        return JsonResponse({"ok": True, "message": "Project restored.", "bootstrap": build_bootstrap_payload().body})
    if action == "delete":
        del rows[index]
        write_csv_rows(DATA_DIR / "development_projects.csv", rows, fieldnames)
        exp_rows = [row for row in read_csv_rows(DATA_DIR / "development_experiments.csv") if str(row.get("Project Name", "")).strip() != project_name]
        exp_fields = read_csv_fieldnames(DATA_DIR / "development_experiments.csv")
        write_csv_rows(DATA_DIR / "development_experiments.csv", exp_rows, exp_fields)
        upd_rows = [row for row in read_csv_rows(DATA_DIR / "experiment_updates.csv") if str(row.get("Project Name", "")).strip() != project_name]
        upd_fields = read_csv_fieldnames(DATA_DIR / "experiment_updates.csv")
        write_csv_rows(DATA_DIR / "experiment_updates.csv", upd_rows, upd_fields)
        return JsonResponse({"ok": True, "message": "Project deleted.", "bootstrap": build_bootstrap_payload().body})
    return JsonResponse({"ok": False, "message": "Unknown project action."}, 400)


def finalize_done_action(payload: dict) -> JsonResponse:
    dataset_name = os.path.basename(str(payload.get("dataset", "")).strip())
    done_description = str(payload.get("doneDescription", "")).strip()
    preform_len_cm = to_float(payload.get("preformLengthCm"))
    if not dataset_name:
        return JsonResponse({"ok": False, "message": "Choose a dataset first."}, 400)
    rows = read_csv_rows(DRAW_ORDERS)
    fieldnames = read_csv_fieldnames(DRAW_ORDERS)
    index = find_order_index_by_dataset(dataset_name)
    if index is None or index >= len(rows):
        return JsonResponse({"ok": False, "message": "No draw order matched the selected dataset."}, 404)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = rows[index]
    row["Status"] = "Done"
    row["Done CSV"] = dataset_name
    row["Assigned Dataset CSV"] = dataset_name
    row["Done Description"] = done_description
    row["Done Timestamp"] = now_str
    row["Status Updated At"] = now_str
    row["Failed CSV"] = ""
    row["Failed Description"] = ""
    row["Failed Timestamp"] = ""
    if preform_len_cm > 0:
        row["Preform Length After Draw (cm)"] = str(preform_len_cm)
    rows[index] = row
    write_csv_rows(DRAW_ORDERS, rows, fieldnames)
    append_dataset_rows(dataset_name, [
        {"Parameter Name": "Done Description", "Value": done_description, "Units": ""},
        {"Parameter Name": "Done Timestamp", "Value": now_str, "Units": ""},
        {"Parameter Name": "Preform Length After Draw", "Value": preform_len_cm, "Units": "cm"},
    ])
    return JsonResponse({"ok": True, "message": "Draw marked as done.", "bootstrap": build_bootstrap_payload().body})


def finalize_failed_action(payload: dict) -> JsonResponse:
    dataset_name = os.path.basename(str(payload.get("dataset", "")).strip())
    failed_description = str(payload.get("failedDescription", "")).strip()
    failed_reason = str(payload.get("failedReason", "")).strip()
    preform_left_cm = to_float(payload.get("preformLeftCm"))
    log_fault = bool(payload.get("logFault"))
    if not dataset_name:
        return JsonResponse({"ok": False, "message": "Choose a dataset first."}, 400)
    rows = read_csv_rows(DRAW_ORDERS)
    fieldnames = read_csv_fieldnames(DRAW_ORDERS)
    index = find_order_index_by_dataset(dataset_name)
    if index is None or index >= len(rows):
        return JsonResponse({"ok": False, "message": "No draw order matched the selected dataset."}, 404)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = rows[index]
    row["Status"] = "Failed"
    row["Failed CSV"] = dataset_name
    row["Assigned Dataset CSV"] = dataset_name
    row["Failed Description"] = failed_description
    row["Failed Reason"] = failed_reason
    row["Failed Timestamp"] = now_str
    row["Status Updated At"] = now_str
    row["Fail Try Count"] = str(int(to_float(row.get("Fail Try Count")) + 1))
    row["Fail Try Last Time"] = now_str
    row["Fail Try Dataset CSV"] = dataset_name
    rows[index] = row
    write_csv_rows(DRAW_ORDERS, rows, fieldnames)
    append_dataset_rows(dataset_name, [
        {"Parameter Name": "Failed Description", "Value": failed_description, "Units": ""},
        {"Parameter Name": "Failed Reason", "Value": failed_reason, "Units": ""},
        {"Parameter Name": "Failed Timestamp", "Value": now_str, "Units": ""},
        {"Parameter Name": "Preform Length After Failed Draw", "Value": preform_left_cm, "Units": "cm"},
    ])
    if log_fault:
        fault_rows = read_csv_rows(FAULTS_LOG)
        fault_fieldnames = read_csv_fieldnames(FAULTS_LOG) or ["fault_id", "fault_ts", "fault_component", "fault_title", "fault_description", "fault_severity", "fault_actor", "fault_source_file", "fault_related_draw"]
        fault_id = str(int(datetime.now().timestamp() * 1000))
        fault_rows.append(
            {
                "fault_id": fault_id,
                "fault_ts": now_str,
                "fault_component": str(payload.get("faultComponent", "")).strip(),
                "fault_title": str(payload.get("faultTitle", "")).strip() or "Draw failure",
                "fault_description": str(payload.get("faultDescription", "")).strip() or failed_description,
                "fault_severity": str(payload.get("faultSeverity", "")).strip() or "medium",
                "fault_actor": str(payload.get("actor", "")).strip() or "rebuild",
                "fault_source_file": dataset_name,
                "fault_related_draw": os.path.splitext(dataset_name)[0],
            }
        )
        write_csv_rows(FAULTS_LOG, fault_rows, fault_fieldnames)
    return JsonResponse({"ok": True, "message": "Draw marked as failed.", "bootstrap": build_bootstrap_payload().body})


def compute_next_planned_draw_date(now_dt: datetime | None = None) -> str:
    now_dt = now_dt or datetime.now()
    weekday = now_dt.weekday()
    next_dt = now_dt + timedelta(days=3 if weekday == 3 else 1)
    return next_dt.strftime("%Y-%m-%d")


def reset_failed_draw_action(payload: dict) -> JsonResponse:
    dataset_name = os.path.basename(str(payload.get("dataset", "")).strip())
    mode = str(payload.get("mode", "")).strip().lower()
    if not dataset_name:
        return JsonResponse({"ok": False, "message": "Choose a dataset first."}, 400)
    if mode not in {"next-day", "pending"}:
        return JsonResponse({"ok": False, "message": "Reset mode is invalid."}, 400)
    rows = read_csv_rows(DRAW_ORDERS)
    fieldnames = read_csv_fieldnames(DRAW_ORDERS)
    index = find_order_index_by_dataset(dataset_name)
    if index is None or index >= len(rows):
        return JsonResponse({"ok": False, "message": "No draw order matched the selected dataset."}, 404)
    row = rows[index]
    schedule_date = compute_next_planned_draw_date(datetime.now()) if mode == "next-day" else ""
    row["Status"] = "Scheduled" if schedule_date else "Pending"
    row["Next Planned Draw Date"] = schedule_date
    row["Active CSV"] = ""
    row["Assigned Dataset CSV"] = ""
    row["Done CSV"] = ""
    row["Done Description"] = ""
    row["Done Timestamp"] = ""
    row["Failed CSV"] = ""
    row["Failed Description"] = ""
    row["Failed Reason"] = ""
    row["Failed Timestamp"] = ""
    row["T&M Moved"] = False
    row["T&M Moved Timestamp"] = ""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["Status Updated At"] = now_str
    row["Last Reset Timestamp"] = now_str
    rows[index] = row
    write_csv_rows(DRAW_ORDERS, rows, fieldnames)
    if schedule_date:
        message = f"Reset to Scheduled. Next Planned Draw Date = {schedule_date}."
    else:
        message = "Reset to Pending (no schedule)."
    return JsonResponse({"ok": True, "message": message, "bootstrap": build_bootstrap_payload().body})


def build_bootstrap_payload() -> JsonResponse:
    home = build_home_payload().body
    schedule = summarize_schedule()
    parts = build_parts_payload().body
    maintenance = build_maintenance_payload().body
    consumables = summarize_consumables()
    process_setup = summarize_process_setup()
    order_draw = summarize_order_draw()
    dashboard = summarize_dashboard()
    draw_finalize = summarize_draw_finalize()
    diagnostics = summarize_diagnostics()
    report_center = summarize_report_center()
    sql_lab = summarize_sql_lab()
    development = summarize_development()
    return JsonResponse(
        {
            "home": home,
            "schedule": schedule,
            "parts": parts,
            "maintenance": maintenance,
            "consumables": consumables,
            "processSetup": process_setup,
            "orderDraw": order_draw,
            "dashboard": dashboard,
            "drawFinalize": draw_finalize,
            "diagnostics": diagnostics,
            "reportCenter": report_center,
            "sqlLab": sql_lab,
            "development": development,
        }
    )


API_ROUTES = {
    "/api/bootstrap": build_bootstrap_payload,
    "/api/home": build_home_payload,
    "/api/schedule": build_schedule_payload,
    "/api/parts": build_parts_payload,
    "/api/parts/manual-index": build_parts_manual_index_payload,
    "/api/maintenance": build_maintenance_payload,
    "/api/consumables": build_consumables_payload,
    "/api/process-setup": build_process_setup_payload,
    "/api/order-draw": build_order_draw_payload,
    "/api/dashboard": build_dashboard_payload,
    "/api/draw-finalize": build_draw_finalize_payload,
    "/api/report-center": build_report_center_payload,
    "/api/sql-lab": build_sql_lab_payload,
    "/api/development": build_development_payload,
    "/api/data-diagnostics": build_diagnostics_payload,
}

POST_API_ROUTES = {
    "/api/development/project": create_development_project_action,
    "/api/development/summary": save_development_summary_action,
    "/api/development/update": create_development_update_action,
    "/api/development/experiment": create_development_experiment_action,
    "/api/development/experiment-update": update_development_experiment_action,
    "/api/development/manage": manage_development_project_action,
    "/api/draw-finalize/done": finalize_done_action,
    "/api/draw-finalize/failed": finalize_failed_action,
    "/api/draw-finalize/reset": reset_failed_draw_action,
    "/api/report-center/development-export": create_development_report_action,
    "/api/report-center/operations-export": create_operations_report_action,
    "/api/maintenance/complete": complete_maintenance_task_action,
    "/api/maintenance/create-parts-orders": create_maintenance_part_orders_action,
    "/api/maintenance/schedule": schedule_maintenance_tasks_action,
    "/api/maintenance/runtime": save_maintenance_runtime_action,
    "/api/maintenance/state": set_maintenance_state_action,
    "/api/maintenance/work-package": save_maintenance_work_package_action,
    "/api/schedule/add": add_schedule_event_action,
    "/api/schedule/delete": delete_schedule_event_action,
    "/api/schedule/save-master": save_schedule_master_action,
    "/api/parts/delete": delete_part_order_action,
    "/api/parts/create": create_part_order_action,
    "/api/parts/inventory-stock": update_inventory_stock_action,
    "/api/parts/unmount": unmount_inventory_item_action,
    "/api/parts/update": update_part_order_action,
    "/api/consumables/dies-save": save_consumables_dies_action,
    "/api/consumables/temps-save": save_consumables_temps_action,
    "/api/process-setup/manual-start": create_process_setup_manual_action,
    "/api/process-setup/scheduled-start": create_process_setup_scheduled_action,
    "/api/process-setup/select-dataset": select_process_setup_dataset_action,
    "/api/process-setup/save-all": save_process_setup_action,
    "/api/order-draw/create": create_order_draw_action,
    "/api/order-draw/template": save_order_draw_template_action,
    "/api/order-draw/schedule": schedule_pending_order_action,
    "/api/dashboard/save-zones": save_dashboard_zones_action,
    "/api/dashboard/math-plot-export": export_dashboard_math_plot_action,
    "/api/sql-lab/filter": run_sql_lab_filter_action,
    "/api/sql-lab/analysis-scope": run_sql_lab_analysis_scope_action,
    "/api/sql-lab/query": run_sql_lab_query_action,
    "/api/data-diagnostics/paths": save_diagnostics_paths_action,
    "/api/data-diagnostics/full-backup": create_full_backup_action,
}


class TowerRebuildHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _send_file(self, path: Path, download_name: str | None = None, inline: bool = False) -> None:
        content_type, _ = mimetypes.guess_type(path.name)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        elif inline:
            self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_allowed_file(self, requested: str, roots: list[Path], fallback_names: list[Path] | None = None) -> Path | None:
        text = str(requested or "").strip()
        if not text:
            return None
        raw = Path(text).expanduser()
        candidates: list[Path] = []
        normalized_variants = [text]
        for prefix in ("development_media/", "data/development_media/", "manuals/", "maintenance/"):
            if text.startswith(prefix):
                normalized_variants.append(text[len(prefix):])
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            for variant in normalized_variants:
                for root in roots:
                    candidates.append((root / variant).resolve())
                file_name = Path(variant).name
                for root in (fallback_names or []):
                    candidates.append((root / file_name).resolve())
        allowed_roots = [root.resolve() for root in roots + (fallback_names or [])]
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not any(str(candidate).startswith(str(root)) for root in allowed_roots):
                continue
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/maintenance/manual-prefetch":
            params = parse_qs(parsed.query)
            requested = str((params.get("path") or [""])[0]).strip()
            raw_pages = list(params.get("page") or [])
            raw_pages.extend(params.get("pages") or [])
            if not requested:
                self._send_json({"ok": False, "message": "Missing path."}, 400)
                return
            candidate = self._resolve_allowed_file(
                requested,
                roots=[ROOT_DIR, MAINTENANCE_DIR, MANUALS_DIR, EXTERNAL_MANUALS_DIR],
                fallback_names=[MANUALS_DIR, EXTERNAL_MANUALS_DIR, MAINTENANCE_DIR],
            )
            if candidate is None:
                self._send_json({"ok": False, "message": "File not found."}, 404)
                return
            page_numbers: list[int] = []
            for raw_value in raw_pages:
                for piece in str(raw_value or "").split(","):
                    text = piece.strip()
                    if not text:
                        continue
                    try:
                        page_number = int(text)
                    except ValueError:
                        continue
                    if page_number > 0:
                        page_numbers.append(page_number)
                    if len(page_numbers) >= 24:
                        break
                if len(page_numbers) >= 24:
                    break
            priority = str((params.get("priority") or [""])[0]).strip().lower() in {"1", "true", "yes", "priority"}
            scheduled = schedule_manual_page_prefetch(candidate, page_numbers, priority=priority)
            self._send_json({"ok": True, "scheduled": scheduled, "pages": page_numbers[:24], "priority": priority})
            return
        if route == "/api/maintenance/manual-page":
            params = parse_qs(parsed.query)
            requested = str((params.get("path") or [""])[0]).strip()
            page_number = int(str((params.get("page") or ["1"])[0]).strip() or "1")
            if not requested:
                self._send_json({"ok": False, "message": "Missing path."}, 400)
                return
            if manual_page_render_mode() != "image":
                self._send_json(
                    {
                        "ok": False,
                        "message": "Manual page image rendering is unavailable on this platform. Use the PDF viewer mode instead.",
                    },
                    409,
                )
                return
            candidate = self._resolve_allowed_file(
                requested,
                roots=[ROOT_DIR, MAINTENANCE_DIR, MANUALS_DIR, EXTERNAL_MANUALS_DIR],
                fallback_names=[MANUALS_DIR, EXTERNAL_MANUALS_DIR, MAINTENANCE_DIR],
            )
            if candidate is None:
                self._send_json({"ok": False, "message": "File not found."}, 404)
                return
            try:
                rendered_path = render_manual_page_image(candidate, page_number)
            except Exception as exc:
                self._send_json({"ok": False, "message": f"Manual page render failed: {exc}"}, 500)
                return
            self._send_file(rendered_path)
            return
        if route == "/api/maintenance/manual":
            params = parse_qs(parsed.query)
            requested = str((params.get("path") or [""])[0]).strip()
            if not requested:
                self._send_json({"ok": False, "message": "Missing path."}, 400)
                return
            candidate = self._resolve_allowed_file(
                requested,
                roots=[ROOT_DIR, MAINTENANCE_DIR, MANUALS_DIR, EXTERNAL_MANUALS_DIR],
                fallback_names=[MANUALS_DIR, EXTERNAL_MANUALS_DIR, MAINTENANCE_DIR],
            )
            if candidate is None:
                self._send_json({"ok": False, "message": "File not found."}, 404)
                return
            self._send_file(candidate)
            return
        if route == "/api/development/media":
            params = parse_qs(parsed.query)
            requested = str((params.get("path") or [""])[0]).strip()
            if not requested:
                self._send_json({"ok": False, "message": "Missing path."}, 400)
                return
            candidate = self._resolve_allowed_file(
                requested,
                roots=[DEVELOPMENT_MEDIA_DIR, DATA_DIR / "development_media"],
                fallback_names=[DEVELOPMENT_MEDIA_DIR, DATA_DIR / "development_media"],
            )
            if candidate is None:
                self._send_json({"ok": False, "message": "File not found."}, 404)
                return
            self._send_file(candidate)
            return
        if route == "/api/dashboard/log":
            params = parse_qs(parsed.query)
            response = build_dashboard_log_payload((params.get("name") or [""])[0])
            self._send_json(response.body, response.status)
            return
        if route == "/api/dashboard/math-plot-export":
            params = parse_qs(parsed.query)
            requested = os.path.basename(str((params.get("name") or [""])[0]).strip())
            if not requested:
                self._send_json({"ok": False, "message": "Missing export name."}, 400)
                return
            candidate = (DASHBOARD_EXPORTS_DIR / requested).resolve()
            if DASHBOARD_EXPORTS_DIR.resolve() not in candidate.parents or not candidate.exists() or not candidate.is_file():
                self._send_json({"ok": False, "message": "Export not found."}, 404)
                return
            self._send_file(candidate, requested)
            return
        if route == "/api/report-center/project":
            params = parse_qs(parsed.query)
            response = build_report_center_project_payload((params.get("name") or [""])[0])
            self._send_json(response.body, response.status)
            return
        if route == "/api/report-center/file":
            params = parse_qs(parsed.query)
            candidate = build_report_center_file_payload((params.get("name") or [""])[0])
            if candidate is None:
                self._send_json({"ok": False, "message": "File not found."}, 404)
                return
            download_requested = str((params.get("download") or [""])[0]).strip().lower() in {"1", "true", "yes"}
            inline_requested = str((params.get("mode") or [""])[0]).strip().lower() == "inline"
            self._send_file(candidate, candidate.name if download_requested and not inline_requested else None, inline=inline_requested)
            return
        if route == "/api/sql-lab/dataset":
            params = parse_qs(parsed.query)
            response = build_sql_lab_dataset_payload((params.get("name") or [""])[0])
            self._send_json(response.body, response.status)
            return
        if route == "/api/draw-finalize":
            params = parse_qs(parsed.query)
            response = build_draw_finalize_payload((params.get("name") or [""])[0])
            self._send_json(response.body, response.status)
            return
        if route == "/api/development/project":
            params = parse_qs(parsed.query)
            response = build_development_project_payload((params.get("name") or [""])[0])
            self._send_json(response.body, response.status)
            return
        if route in API_ROUTES:
            response = API_ROUTES[route]()
            self._send_json(response.body, response.status)
            return

        if route in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html")
            return

        static_path = (STATIC_DIR / route.lstrip("/")).resolve()
        if STATIC_DIR in static_path.parents and static_path.exists() and static_path.is_file():
            self._send_file(static_path)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route not in POST_API_ROUTES:
            self._send_json({"ok": False, "message": "Not found"}, 404)
            return
        payload = self._read_json_body()
        response = POST_API_ROUTES[route](payload)
        self._send_json(response.body, response.status)


def run(host: str | None = None, port: int | None = None) -> None:
    ensure_runtime_directories()
    bind_host = str(host or DEFAULT_BIND_HOST).strip() or "127.0.0.1"
    bind_port = int(port or DEFAULT_BIND_PORT or 8010)
    server = HTTPServer((bind_host, bind_port), TowerRebuildHandler)
    print(f"Tower rebuild running on http://{bind_host}:{bind_port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
