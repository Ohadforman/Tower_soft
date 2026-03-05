# Architecture

## 1) Runtime Flow

Execution order in `dash_try.py`:

1. Import `P` from `app_io.paths`.
2. Install legacy compatibility shim: `install_legacy_path_compat(P)`.
3. Configure Streamlit page (`configure_page`) and global style (`apply_blue_clean_base_theme`).
4. Run startup checks from `tests.runners.preflight.run_checks()`.
5. Fail-fast on startup issues (or continue in SAFE MODE).
6. Validate coating config (`ensure_coating_config`).
7. Build lightweight runtime (`build_runtime`) with shared values.
8. Initialize session state and render grouped sidebar navigation.
9. Route selected tab using `app.router.render_selected_tab`.
10. Re-apply base theme after tab render to prevent per-tab CSS drift.

## 2) Layered Design

### App shell layer

- `app/bootstrap.py`
  - page config
  - startup config validation
  - app runtime object (`AppRuntime`)
- `app/navigation.py`
  - grouped navigation model
  - tab-group persistence in `st.session_state`
  - SAFE MODE navigation subset
- `app/router.py`
  - central dispatch from tab label to tab render function

### IO/path layer

- `app_io/paths.py`
  - canonical path dataclass (`Paths`) and instance (`P`)
  - cross-platform local-data directory resolution
  - duckdb local/shared/fallback policy
  - helper functions for directory-safe file path creation
- `app_io/legacy_path_compat.py`
  - monkeypatch compatibility for old root filenames
  - redirects open/exists/isfile/getmtime/stat to canonical paths
- `app_io/path_health.py`
  - builds health reports for path existence/read/write checks

### UI/render layer

- `renders/tabs/*`
  - page-level rendering logic
- `renders/components/*`
  - shared reusable UI components
- `renders/support/*`
  - styling helpers and asset helpers

### Domain/helper layer

- `helpers/*`
  - CSV schema validation
  - JSON/CSV IO
  - logging
  - format/date/status utilities
  - backup helpers
  - duckdb helpers

### Test/diagnostic layer

- CLI wrappers under `scripts/cli/` call modules under `tests/runners/`.
- `tests/runners/preflight.py`: fast startup gate checks.
- `tests/runners/app_tests.py`: broader functional and regression checks.
- `tests/runners/all_checks.py`: combined pipeline + report artifact.

## 3) Navigation Architecture

Navigation source of truth is in `app/navigation.py`:

- groups:
  - Home & Project Management
  - Operations
  - Monitoring & Research
- state keys:
  - `nav_group_select`
  - `tab_select`
  - `last_tab`
  - `nav_last_tab_by_group`
- behavior:
  - remembers last page per group
  - supports deep jumps via `selected_tab`
  - SAFE MODE limits tabs when startup checks fail

## 4) Data Architecture

Primary storage is filesystem-based CSV/JSON with optional duckdb acceleration.

Canonical roots:

- `data/`: operational CSV files (orders, schedule, containers, temps, protocols CSV, etc.)
- `config/`: JSON configurations (coating/protocols/pid/container/etc.)
- `state/`: state snapshots (stock levels, regression snapshot)
- `logs/`: app and runtime logs
- `reports/`: generated reports/check outputs/path audits
- `assets/images/`: UI images
- `data_set_csv/`: dataset CSV library
- `maintenance/`: maintenance/fault data and templates

Path governance:

- always read/write through `P` paths when possible.
- keep feature logic independent from physical root-level file names.
- support old file names only via compatibility shim.

## 5) DuckDB Concurrency Strategy

From `app_io/paths.py`:

- default: per-process user-local DB (`tower_<pid>.duckdb`) to avoid lock conflicts.
- optional shared per-user DB with `TOWER_DUCKDB_SHARED=1`.
- fallback to `<project>/data` only if local directory cannot be prepared.

This supports multi-user deployments across network shares better than one shared DB file.

## 6) Styling Architecture

Global visual baseline is injected by `apply_blue_clean_base_theme()`:

- button/input/tag/expander/dataframe styling
- blue glow theme tokens and panel backgrounds
- applied at startup and forced again after tab render

Rule of thumb:

- shared look in global theme
- tab-specific CSS only for local layout intent

## 7) Safety/Resilience

Startup safeguards:

- startup checks before full app usage
- safe-mode fallback to limited tabs
- strict coating config validation before run

Operational safeguards:

- preflight verifies file presence/schema/path layout
- app tests verify imports, schema contracts, UI smoke, and snapshot integrity
- path permissions audit for read/write diagnostics
