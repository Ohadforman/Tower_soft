# Tower Management Software

Streamlit application for optical tower operations: draw orders, process setup, consumables, schedule, maintenance, diagnostics, SQL analysis, and development tracking.

## Quick Start

1. Create/activate virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run startup checks:

```bash
python3 scripts/cli/run_preflight.py
```

3. Start the app:

```bash
streamlit run dash_try.py
```

## Core Architecture

Entrypoint:
- `dash_try.py`:
  - installs legacy path compatibility (`install_legacy_path_compat(P)`)
  - configures page/theme
  - runs startup checks (`tests.runners.preflight.run_checks`)
  - builds runtime (`app.bootstrap.build_runtime`)
  - initializes navigation/session state (`app.navigation`)
  - routes to selected tab (`app.router.render_selected_tab`)

App layers:
- `app/`:
  - `bootstrap.py`: startup config and runtime object
  - `navigation.py`: grouped sidebar navigation + tab state memory
  - `router.py`: dispatch to tab renderers
- `app_io/`:
  - `paths.py`: single source of truth for filesystem paths (`P`)
  - `legacy_path_compat.py`: redirects old root filenames to current paths
  - `path_health.py`: health report model for paths/permissions
- `renders/tabs/`: one module per tab
- `renders/components/`: reusable UI blocks used by tabs
- `helpers/`: IO, schema validation, logging, formatting, and utility logic
- `tests/runners/`: preflight, app tests, combined checks, audits

Detailed docs:
- [Architecture](docs/ARCHITECTURE.md)
- [Operations and Runbook](docs/OPERATIONS.md)
- [Maintenance Operator Guide](docs/MAINTENANCE_OPERATOR_GUIDE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Environment Pretest](docs/ENV_PRETEST.md)
- [Path Map](docs/path_map.md)

## Main Tabs

- Home
- Schedule
- Order Draw
- Tower Parts
- Consumables and dies
- Process Setup
- Maintenance
- Dashboard
- Draw Finalize
- Data Diagnostics
- Report Center
- SQL Lab
- Development Process

## Testing and Diagnostics

Run full checks:

```bash
python3 scripts/cli/run_all_checks.py
```

Run release readiness check (checks + backup + final READY/NOT READY):

```bash
python3 scripts/cli/run_release_check.py
```

Run environment pretest (machine/network readiness):

```bash
python3 scripts/cli/run_env_pretest.py
```

Run full health check (all checks + env + release summary):

```bash
python3 scripts/cli/run_full_health_check.py
```

Run V2 deployment protocol (go/no-go + debug hints + artifacts):

```bash
python3 scripts/cli/run_v2_deploy_protocol.py
```

Run app tests only:

```bash
python3 scripts/cli/run_app_tests.py
```

Run permission/path audit:

```bash
python3 scripts/cli/run_path_permissions_audit.py
```

Update regression baseline snapshot:

```bash
python3 scripts/cli/run_update_regression_snapshot.py
```

## Path Management Policy

- Use `app_io.paths.P` in app code for file locations.
- Keep files organized under folders (`data/`, `config/`, `assets/`, `state/`, `reports/`, etc.).
- Avoid hardcoded root-level filenames in tabs/helpers.
- Legacy root names remain supported at runtime through `install_legacy_path_compat(P)`.

## DuckDB Policy (multi-user safe)

- DuckDB defaults to user-local storage (`~/Library/Application Support/Tower_work` on macOS, `%LOCALAPPDATA%\\Tower_work` on Windows, `~/.local/share/Tower_work` on Linux).
- Default mode uses one shared per-user DB filename (`tower.duckdb`) per computer/user.
- Set `TOWER_DUCKDB_ISOLATED=1` (or `TOWER_DUCKDB_SHARED=0`) for per-process DB files (`tower_<pid>.duckdb`) when running multiple local app instances.
- Fallback uses project `data/` only if local user dir cannot be created.

## Environment Variables

- `TOWER_ROOT`: override project root for all path building.
- `TOWER_SAFE_MODE=1`: open app with limited safe tabs when startup checks fail.
- `TOWER_DUCKDB_SHARED=1`: force shared per-user duckdb file.
- `TOWER_DUCKDB_ISOLATED=1`: force per-process duckdb file.
- `TOWER_LOCAL_DB_DIR`: override user-local duckdb directory.

## Notes

- Keep `docs/path_map.md` updated when `app_io/paths.py` changes.
- The app expects canonical files under current folders (not repo root).
