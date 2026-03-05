# Environment Pretest

Run this on a new computer/network location before first app use.

## Command

```bash
python3 scripts/cli/run_env_pretest.py
```

## What It Checks

- `EV-01` Python version (>= 3.9)
- `EV-02` virtual environment active
- `EV-03` `streamlit` command in PATH
- `EV-04` required package imports
- `EV-05` core directories from `P` exist
- `EV-06` read/write probe on `data/`, `logs/`, `backups/`
- `EV-07` DuckDB parent directory writable
- `EV-08` free disk space check
- `EV-09` site-packages visibility
- `EV-10` embedded app preflight checks
- `EV-11` storage latency probe (detects very slow mounted/network storage)

## Outputs

Reports are written to `reports/checks/`:

- `env_pretest_<timestamp>.json`
- `env_pretest_<timestamp>.md`

Result is printed as:

- `READY ✅` when all checks pass
- `NOT READY ❌` when at least one check fails

## Usage Tip

When reporting a problem, share the check code (for example `EV-06`) and its error line.
