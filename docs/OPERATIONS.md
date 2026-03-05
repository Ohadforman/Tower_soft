# Operations Runbook

## Start App

```bash
streamlit run dash_try.py
```

Startup behavior:

- preflight checks run automatically
- if critical checks fail:
  - default: app stops with clear errors
  - with `TOWER_SAFE_MODE=1`: app opens limited tabs for debugging

## Required Python

- Python 3.9+ recommended.
- Install dependencies from `requirements.txt`.

## Health and Test Commands

Run fast preflight:

```bash
python3 scripts/cli/run_preflight.py
```

Run app test suite:

```bash
python3 scripts/cli/run_app_tests.py
```

Run full checks bundle:

```bash
python3 scripts/cli/run_all_checks.py
```

Run release readiness check:

```bash
python3 scripts/cli/run_release_check.py
```

Run path/permission audit:

```bash
python3 scripts/cli/run_path_permissions_audit.py
```

Update regression snapshot baseline:

```bash
python3 scripts/cli/run_update_regression_snapshot.py
```

Run environment pretest (new machine/network readiness):

```bash
python3 scripts/cli/run_env_pretest.py
```

Run full health check (single command for all readiness checks):

```bash
python3 scripts/cli/run_full_health_check.py
```

Note: if your terminal is inside `docs/`, the same command path still works because `docs/scripts/cli` wrappers forward to the real scripts.

## Generated Artifacts

- full check reports: `reports/checks/all_checks_*.json`
- path permission reports: `reports/path_audit/path_permissions_*.json` and `.csv`
- app logs: `logs/`
- backup snapshots: `backups/`

## Environment Variables

- `TOWER_ROOT`: override base root for path construction.
- `TOWER_SAFE_MODE=1`: allow limited app load when checks fail.
- `TOWER_DUCKDB_SHARED=1`: use shared per-user duckdb file (`tower.duckdb`).
- `TOWER_LOCAL_DB_DIR`: override local duckdb directory.

## DuckDB Multi-user Guidance

For multiple users on networked setups:

- keep default per-process DB naming (`tower_<pid>.duckdb`)
- keep DB in user-local storage
- avoid one shared DB file for concurrent writers

If a lock error occurs, verify whether two app instances are pointing to the same shared duckdb file and disable shared mode.

## Backup Guidance

- backups are written under `P.backups_dir` (`backups/` in project by default).
- include data/config/state/log snapshots when creating operational backups.
- run permission audit periodically to confirm write access.

## Troubleshooting

### FileNotFoundError for config/data files

1. Validate canonical location exists under `data/` or `config/`.
2. Run `python3 scripts/cli/run_preflight.py`.
3. Confirm `app_io/paths.py` values or `docs/path_map.md` entries.

### Streamlit key warnings (Session State default + manual set)

- avoid setting widget defaults both in widget constructor and `st.session_state` for the same key in same run.

### DuckDB lock errors

- expected if same DB file is opened by multiple processes.
- use per-process local DB (default) to avoid collisions.

### UI loads but data sections are empty

1. confirm target CSV exists and has expected schema.
2. run `python3 scripts/cli/run_app_tests.py` to detect schema mismatch.
3. check selected dataset file and `data/selected_csv.json`.
