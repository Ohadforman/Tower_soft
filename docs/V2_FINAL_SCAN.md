# V2 Final Scan (Pre-Air)

This is the final lightweight gate before moving Tower V2 to deployment.

## 1) Run Order (from project root)

```bash
python3 scripts/cli/run_preflight.py
python3 scripts/cli/run_app_tests.py
python3 scripts/cli/run_path_permissions_audit.py
python3 scripts/cli/run_env_pretest.py
python3 scripts/cli/run_full_health_check.py
```

If you are inside `docs/`, run:

```bash
python3 ../scripts/cli/run_full_health_check.py
```

## 2) Pass Criteria

- `run_full_health_check.py` returns `READY ✅`
- `run_app_tests.py` has `Failed: 0`
- No critical path issues in path audit
- Required data/config/assets files are reachable via `app_io/paths.py` only

## 3) V2 Hardening Checks Included

- Navigation/router consistency (active tabs aligned, removed tabs not exposed)
- Base blue theme hook is active from app entry
- Path single-source guardrail (no hardcoded root filenames in app code)

## 4) Manual Spot Checks (UI + Flow)

- Home: section visibility, hover areas, no overlap when sidebar is open
- Process Setup: both flows work
  - from scheduled order
  - manual quick start (no order), with dataset auto-name `{Preform}F_N.csv`
- Maintenance: builder, execute/record, day-pack, no nested expander errors
- Tower Parts: inventory updates, mounted/unmounted counts, finder filters
- SQL Lab: step flow, collapse/expand behavior, run results visible
- Diagnostics: control panel buttons, statuses update after run

## 5) Performance Guardrails

- Keep heavy CSV/PDF readers behind `st.cache_data`/`st.cache_resource`
- Avoid repeated file reads in widget rebuild loops
- Keep optional heavy sections collapsed by default
- Use diagnostics cache-clear only for explicit refresh/debug actions

## 6) Release Artifacts

- `reports/checks/`:
  - release check outputs
  - deploy protocol JSON/MD
  - release bundle ZIP
- `backups/`:
  - latest backup snapshot exists and is readable

## 7) Go/No-Go

- `GO`: all gates pass and manual spot checks are green
- `NO-GO`: any failing gate, schema mismatch, path critical issue, or blocking UI runtime error
