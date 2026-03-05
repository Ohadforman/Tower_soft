# Development Guide

## Project Structure

- `dash_try.py`: app entrypoint
- `app/`: bootstrap/navigation/router
- `app_io/`: path/config/compatibility infrastructure
- `renders/tabs/`: tab pages
- `renders/components/`: shared UI building blocks
- `helpers/`: domain and IO utilities
- `tests/runners/`: executable test/check pipelines
- `tools/`: maintenance utilities (including path permission audit tool)

## Coding Rules (practical)

- prefer `P.<path_key>` from `app_io.paths` for file access.
- avoid root-level hardcoded filenames in tab logic.
- keep routing decisions in `app/router.py`.
- put reusable UI in `renders/components/`, not duplicated in tabs.
- keep heavy operations lazy (inside tab handlers when possible).

## Adding a New Tab

1. Create renderer in `renders/tabs/<new_tab>.py`.
2. Add navigation label to `NAV_GROUPS` in `app/navigation.py`.
3. Import and route in `app/router.py`.
4. Add smoke/import coverage in `tests/runners/app_tests.py`.

## Changing Paths Safely

1. Update `app_io/paths.py` only.
2. Run:

```bash
python3 scripts/cli/run_preflight.py
python3 scripts/cli/run_app_tests.py
python3 scripts/cli/run_path_permissions_audit.py
```

3. Update docs map (`docs/path_map.md`) if needed.
4. Keep compatibility map in `app_io/legacy_path_compat.py` aligned when migrating legacy names.

## Test Strategy

### Preflight (`scripts/cli/run_preflight.py`)

Purpose: fast startup gate.

Checks include:
- folder layout policy
- required files existence
- backup dir writability
- JSON parsing
- duckdb local policy
- required CSV schema columns

### App tests (`scripts/cli/run_app_tests.py`)

Purpose: broader regression confidence.

Includes:
- path health and path policy checks
- schema and IO roundtrip
- module import smoke
- snapshot baseline regression check
- Streamlit app smoke and tab switching smoke
- legacy redirect runtime behavior

### All checks (`scripts/cli/run_all_checks.py`)

Purpose: single command for preflight + app tests + path permissions audit with JSON report output.

## Performance Guidance

- cache expensive IO/index operations (`@st.cache_data`) with clear invalidation key.
- avoid broad scans of `data_set_csv/` on every render when mtime cache is possible.
- keep startup lightweight; avoid opening heavy DB/session objects globally.

## UI/Theming Guidance

- global style comes from `renders/support/style_utils.py`.
- use blue clean theme defaults for consistency.
- per-tab CSS should be scoped and minimal.

## Commit Hygiene

- run checks before push:

```bash
python3 scripts/cli/run_preflight.py && python3 scripts/cli/run_app_tests.py
```

- include affected docs when changing architecture/path contracts.

