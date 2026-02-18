# ==========================================================
# app_io/paths.py  (FULL)
# ==========================================================
import os
from dataclasses import dataclass

# If you ever want to relocate everything, set TOWER_ROOT env var
# Example: export TOWER_ROOT="/Users/ohadformanair/PycharmProjects/Tower_work"
DEFAULT_ROOT = os.environ.get("TOWER_ROOT", os.getcwd())


def _abs(*parts: str) -> str:
    return os.path.join(DEFAULT_ROOT, *parts)


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
    # Core "top-level" files
    # =========================
    orders_csv: str = _abs("draw_orders.csv")
    history_csv: str = _abs("history_log.csv")
    selected_csv_json: str = _abs("selected_csv.json")

    # schedule
    schedule_csv: str = _abs("tower_schedule.csv")

    # development / experiments
    development_csv: str = _abs("development_process.csv")
    experiment_updates_csv: str = _abs("experiment_updates.csv")

    # parts
    parts_orders_csv: str = _abs("part_orders.csv")
    parts_archived_csv: str = _abs("archived_orders.csv")

    # inventory / consumables
    sap_rods_inventory_csv: str = _abs("sap_rods_inventory.csv")
    consumables_csv: str = _abs("consumables.csv")  # optional
    preform_inventory_csv: str = _abs("preforms_inventory.csv")

    # ✅ NEW: live tower temperatures (wide 1-row csv)
    tower_temps_csv: str = _abs("tower_temps.csv")
    tower_containers_csv: str = _abs("tower_containers.csv")
    # projects
    projects_fiber_csv: str = _abs("projects_fiber.csv")
    projects_fiber_templates_csv: str = _abs("projects_fiber_templates.csv")

    # protocols
    protocols_csv: str = _abs("protocols.csv")  # optional if you use one

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
    config_dir: str = _abs(".")  # you keep configs in project root
    coating_config_json: str = _abs("config_coating.json")
    directory_config_json: str = _abs("directory_config.json")
    container_config_json: str = _abs("container_config.json")
    pid_config_json: str = _abs("pid_config.json")

    # =========================
    # DB
    # =========================
    duckdb_path: str = _abs("tower.duckdb")

    # =========================
    # Hook outputs
    # =========================
    done_snapshots_dir: str = _abs("hooks", "done_csv_snapshots")
    after_done_log: str = _abs("hooks", "after_done_last_run.txt")


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