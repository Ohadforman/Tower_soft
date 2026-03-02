# ==========================================================
# app_io/paths.py  (FULL)
# ==========================================================
import os
from dataclasses import dataclass

# If you ever want to relocate everything, set TOWER_ROOT env var.
# Fall back to project root (stable), not os.getcwd() (can vary by launcher).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.environ.get("TOWER_ROOT", PROJECT_ROOT)


def _abs(*parts: str) -> str:
    return os.path.join(DEFAULT_ROOT, *parts)


def _pick_path(*candidates: str) -> str:
    """
    Return first existing path; if none exist, return the first candidate.
    This keeps compatibility during folder migrations.
    """
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0] if candidates else ""


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """
    Prevent subfolders and weird paths.
    'ELOP - 130/test.csv' -> 'test.csv'
    """
    s = str(name or "").replace("\\", "/").strip()
    s = os.path.basename(s)
    s = s.replace("/", "_").strip()
    return s


@dataclass(frozen=True)
class Paths:
    # =========================
    # Base directories
    # =========================
    root_dir: str = DEFAULT_ROOT
    data_dir: str = _abs("data")
    config_dir: str = _abs("config")
    state_dir: str = _abs("state")
    assets_dir: str = _abs("assets")
    images_dir: str = _abs("assets", "images")
    protocols_assets_dir: str = _abs("protocols_assets")

    # =========================
    # Core "top-level" files
    # =========================
    orders_csv: str = _abs("data", "draw_orders.csv")
    history_csv: str = _abs("data", "history_log.csv")
    selected_csv_json: str = _abs("data", "selected_csv.json")

    # schedule
    schedule_csv: str = _abs("data", "tower_schedule.csv")

    # development / experiments
    development_csv: str = _abs("data", "development_process.csv")
    experiment_updates_csv: str = _abs("data", "experiment_updates.csv")

    # parts
    parts_orders_csv: str = _abs("data", "part_orders.csv")
    parts_archived_csv: str = _abs("data", "archived_orders.csv")

    # inventory / consumables
    sap_rods_inventory_csv: str = _abs("data", "sap_rods_inventory.csv")
    consumables_csv: str = _abs("data", "consumables.csv")  # optional
    preform_inventory_csv: str = _abs("data", "preforms_inventory.csv")

    # ✅ NEW: live tower temperatures (wide 1-row csv)
    tower_temps_csv: str = _abs("data", "tower_temps.csv")
    tower_containers_csv: str = _abs("data", "tower_containers.csv")
    # projects
    projects_fiber_csv: str = _abs("data", "projects_fiber.csv")
    projects_fiber_templates_csv: str = _abs("data", "projects_fiber_templates.csv")

    # protocols
    protocols_csv: str = _abs("data", "protocols.csv")  # optional if you use one
    protocols_json: str = _abs("config", "protocols.json")

    # =========================
    # Directories
    # =========================
    dataset_dir: str = _abs("data_set_csv")
    logs_dir: str = _abs("logs")
    maintenance_dir: str = _abs("maintenance")
    hooks_dir: str = _abs("hooks")
    parts_dir: str = _abs("parts")

    # ✅ Reports (all reports live here)
    reports_dir: str = _abs("reports")
    gas_reports_dir: str = _abs("reports", "gas")

    # ✅ Gas periodic scheduler state
    gas_reports_state_json: str = _abs("reports", "gas", "_gas_reports_state.json")

    # configs (json/csv that act like configuration)
    coating_config_json: str = _abs("config", "config_coating.json")
    directory_config_json: str = _abs("config", "directory_config.json")
    container_config_json: str = _abs("config", "container_config.json")
    pid_config_json: str = _abs("config", "pid_config.json")
    heater_config_json: str = _abs("config", "heater_config.json")
    dies_config_json: str = _abs("config", "dies_6station.json")
    coating_data_json: str = _abs("config", "coating_data.json")

    # =========================
    # DB
    # =========================
    duckdb_path: str = _abs("data", "tower.duckdb")

    # =========================
    # Hook outputs
    # =========================
    done_snapshots_dir: str = _abs("hooks", "done_csv_snapshots")
    after_done_log: str = _abs("hooks", "after_done_last_run.txt")

    # =========================
    # Assets / state
    # =========================
    home_bg_image: str = _abs("assets", "images", "IMG_1094.JPEG")
    logo_image: str = _abs("assets", "images", "icap.png")
    coating_stock_json: str = _abs("state", "coating_type_stock.json")
    container_levels_prev_json: str = _abs("state", "container_levels_prev.json")


P = Paths()


# ==========================================================
# Dataset CSV helpers
# ==========================================================
def ensure_dataset_dir() -> str:
    return ensure_dir(P.dataset_dir)


def dataset_csv_path(filename: str) -> str:
    """
    Absolute path to a dataset csv in data_set_csv/.
    Safe against 'folder/name.csv' inputs.
    """
    ensure_dataset_dir()
    return os.path.join(P.dataset_dir, safe_filename(filename))


# ==========================================================
# Logs helpers
# ==========================================================
def ensure_logs_dir() -> str:
    return ensure_dir(P.logs_dir)


def log_csv_path(filename: str) -> str:
    ensure_logs_dir()
    return os.path.join(P.logs_dir, safe_filename(filename))


# ==========================================================
# Maintenance helpers
# ==========================================================
def ensure_maintenance_dir() -> str:
    return ensure_dir(P.maintenance_dir)


def maintenance_file_path(filename: str) -> str:
    ensure_maintenance_dir()
    return os.path.join(P.maintenance_dir, safe_filename(filename))


# ==========================================================
# Hooks helpers
# ==========================================================
def ensure_hooks_dir() -> str:
    return ensure_dir(P.hooks_dir)


def ensure_done_snapshots_dir() -> str:
    return ensure_dir(P.done_snapshots_dir)


def hook_file_path(filename: str) -> str:
    ensure_hooks_dir()
    return os.path.join(P.hooks_dir, safe_filename(filename))


# ==========================================================
# Parts helpers
# ==========================================================
def ensure_parts_dir() -> str:
    return ensure_dir(P.parts_dir)


def parts_file_path(filename: str) -> str:
    ensure_parts_dir()
    return os.path.join(P.parts_dir, safe_filename(filename))


# ==========================================================
# ✅ Reports helpers
# ==========================================================
def ensure_reports_dir() -> str:
    return ensure_dir(P.reports_dir)


def ensure_gas_reports_dir() -> str:
    ensure_reports_dir()
    return ensure_dir(P.gas_reports_dir)


def gas_report_path(filename: str) -> str:
    """
    Absolute path to a report file in reports/gas/.
    Safe against 'folder/name.csv' inputs.
    """
    ensure_gas_reports_dir()
    return os.path.join(P.gas_reports_dir, safe_filename(filename))


# ==========================================================
# ✅ Tower temps helpers (optional)
# ==========================================================
def tower_temps_path() -> str:
    """
    Absolute path to the live wide temps CSV (tower_temps.csv).
    """
    ensure_dir(os.path.dirname(P.tower_temps_csv) or ".")
    return P.tower_temps_csv
