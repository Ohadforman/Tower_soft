# V2 Deployment Protocol

This is the controlled go/no-go checklist for Tower V2 deployment.

## 1) Scope and Goal

- Validate paths, files, permissions, schemas, imports, and release snapshot.
- Fail fast with clear debug hints.
- Produce artifacts for audit and rollback readiness.

## 2) One-command Protocol

Run from project root:

```bash
python3 scripts/cli/run_v2_deploy_protocol.py
```

Optional (slower) UI smoke:

```bash
python3 scripts/cli/run_v2_deploy_protocol.py --ui-smoke
```

The protocol writes artifacts to:

- `reports/checks/v2_deploy_protocol_*.json`
- `reports/checks/v2_deploy_protocol_*.md`

Tip: if your terminal is inside `docs/`, run with `../scripts/cli/...` paths.

## 3) Step Order (Control Gates)

1. `Environment pretest`
2. `Preflight`
3. `App tests`
4. `Path permissions audit`
5. `Release check`

Gate rule:

- Deployment is allowed only if all steps pass (`READY ✅`).

## 4) Debug Logic (When NOT READY)

Use the first failing step as the only active blocker. Fix in this order:

1. `Environment pretest` fails:
   - Use project venv and install deps:
   - macOS/Linux: `.venv/bin/pip install -r requirements.txt`
   - Windows: `.venv\\Scripts\\pip install -r requirements.txt`

2. `Preflight` fails:
   - Fix missing/misplaced files from `app_io/paths.py`.
   - Re-check `data/`, `config/`, `assets/`, `logs/`, `maintenance/`, `parts/`.

3. `App tests` fails:
   - Run only app tests and fix first failing code:
   - `python3 scripts/cli/run_app_tests.py`

4. `Path permissions audit` fails:
   - Fix read/write permissions on `data/`, `logs/`, `backups/`, `reports/`.
   - Verify user-local DuckDB parent directory is writable.

5. `Release check` fails:
   - Resolve failed checks and backup snapshot errors.
   - Re-run release check, then re-run full protocol.

## 5) Path Governance (Single Source of Truth)

- All operational paths must come from `app_io/paths.py` (`P`).
- Do not hardcode root filenames in tabs/helpers.
- Legacy root names are runtime-redirected via `app_io/legacy_path_compat.py`.

## 6) Pre-deploy and Post-deploy Commands

Pre-deploy:

```bash
python3 scripts/cli/run_v2_deploy_protocol.py
```

Post-deploy smoke:

```bash
python3 scripts/cli/run_preflight.py
python3 scripts/cli/run_app_tests.py
```

Optional strict UI smoke (slower):

```bash
python3 scripts/cli/run_app_tests.py --ui-smoke --ui-tabs
```

## 8) Included App Test Coverage (high level)

- Path/data/config/file integrity and schemas
- Legacy path redirect compatibility
- Entry compile + module import smoke
- Path single-source guardrail
- Navigation/router consistency
- Base blue theme hook presence

## 7) Rollback Rule

If deployment is `NOT READY ❌` after remediation attempts:

1. Stop rollout.
2. Keep current stable commit/tag active.
3. Open the latest protocol artifact in `reports/checks/`.
4. Fix blockers and rerun protocol.
