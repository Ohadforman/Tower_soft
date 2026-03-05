from __future__ import annotations

import re
from typing import Dict, List


ERROR_REGISTRY: Dict[str, Dict[str, str]] = {
    "PF-01": {"title": "Folder layout", "fix": "Check `app_io/paths.py` and ensure core files point to `data/`, `config/`, `assets/` folders."},
    "PF-02": {"title": "Required files", "fix": "Restore missing required files in canonical `P` paths (data/config/assets)."},
    "PF-03": {"title": "Backup dir", "fix": "Ensure `P.backups_dir` exists and user has write permission."},
    "PF-04": {"title": "JSON parse", "fix": "Fix invalid JSON syntax in config files (`config_coating.json`, `protocols.json`, etc.)."},
    "PF-05": {"title": "DuckDB local policy", "fix": "Use writable local duckdb parent; check `TOWER_LOCAL_DB_DIR` and permissions."},
    "PF-06": {"title": "CSV schemas", "fix": "Repair missing required CSV columns in the reported file(s)."},
    "AT-01": {"title": "Path health report", "fix": "Run path audit and repair missing/unreadable/unwritable configured paths."},
    "AT-02": {"title": "Paths are folder-based", "fix": "Move path definitions back to canonical folders via `app_io/paths.py`."},
    "AT-03": {"title": "Core files exist", "fix": "Restore required data/config/image files referenced by `P`."},
    "AT-04": {"title": "JSON integrity", "fix": "Validate and fix malformed JSON configs."},
    "AT-05": {"title": "Backup dir writable", "fix": "Grant write permission to backup folder or choose writable location."},
    "AT-06": {"title": "DuckDB local path policy", "fix": "Ensure duckdb path is local/fallback and parent directory is writable."},
    "AT-07": {"title": "Orders schema", "fix": "Add missing required columns to `draw_orders.csv`."},
    "AT-08": {"title": "Parts schema", "fix": "Add missing required columns to `part_orders.csv`."},
    "AT-09": {"title": "Schedule schema", "fix": "Add missing required columns to `tower_schedule.csv`."},
    "AT-10": {"title": "Tower temps schema", "fix": "Add missing required columns to `tower_temps.csv`."},
    "AT-11": {"title": "Tower containers schema", "fix": "Add missing required columns to `tower_containers.csv`."},
    "AT-12": {"title": "CSV roundtrip", "fix": "Check encoding/format issues and normalize CSV write/read behavior."},
    "AT-13": {"title": "Config expected keys", "fix": "Ensure required keys exist in coating/protocol config files."},
    "AT-14": {"title": "No root duplicate files", "fix": "Remove legacy duplicate root files and keep canonical folder files only."},
    "AT-15": {"title": "Legacy redirect runtime", "fix": "Verify `install_legacy_path_compat(P)` is called before app logic."},
    "AT-16": {"title": "dash_try compile", "fix": "Fix syntax/indentation errors in `dash_try.py`."},
    "AT-17": {"title": "Module import smoke", "fix": "Fix import paths/module errors in tabs/components."},
    "AT-18": {"title": "Regression snapshot", "fix": "Run `scripts/cli/run_update_regression_snapshot.py` after intentional data changes."},
    "AT-19": {"title": "Path single-source guardrail", "fix": "Replace hardcoded file names with `P.*` or path helper functions."},
    "EV-01": {"title": "Python version", "fix": "Use Python 3.9+ for this app."},
    "EV-02": {"title": "Virtual environment", "fix": "Activate project venv before running commands (recommended)."},
    "EV-03": {"title": "Streamlit command", "fix": "Install Streamlit and ensure executable is on PATH."},
    "EV-04": {"title": "Required package imports", "fix": "Install missing packages from `requirements.txt` in the active environment."},
    "EV-05": {"title": "Core directories", "fix": "Create missing directories or fix root path/environment configuration."},
    "EV-06": {"title": "Read/write probes", "fix": "Grant write permission to data/logs/backups directories."},
    "EV-07": {"title": "DuckDB parent writable", "fix": "Use writable duckdb parent dir or set `TOWER_LOCAL_DB_DIR`."},
    "EV-08": {"title": "Disk free space", "fix": "Free disk space; keep at least 1GB available."},
    "EV-09": {"title": "Site-packages visibility", "fix": "Ensure Python can see installed packages from active environment."},
    "EV-10": {"title": "Embedded preflight", "fix": "Run preflight and fix failing `PF-*` checks first."},
    "EV-11": {"title": "Storage latency probe", "fix": "If latency is high, avoid slow network mount or move app data to a faster local disk."},
}


_CODE_PATTERN = re.compile(r"\[(PF|AT|EV)-\d{2}\]")


def extract_error_codes(text: str) -> List[str]:
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for full in _CODE_PATTERN.finditer(text):
        code = full.group(0).strip("[]")
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def get_error_help(code: str) -> Dict[str, str]:
    if code in ERROR_REGISTRY:
        info = ERROR_REGISTRY[code]
        prefix = code.split("-", 1)[0]
        if prefix == "PF":
            doc = "docs/OPERATIONS.md"
        elif prefix == "AT":
            doc = "docs/DEVELOPMENT.md"
        elif prefix == "EV":
            doc = "docs/ENV_PRETEST.md"
        else:
            doc = "docs/README.md"
        return {"code": code, "title": info.get("title", ""), "fix": info.get("fix", ""), "doc": doc}
    prefix = code.split("-", 1)[0] if "-" in code else code
    return {
        "code": code,
        "title": f"{prefix} check",
        "fix": "Review check output details and fix path/config/permissions accordingly.",
        "doc": "docs/OPERATIONS.md",
    }
