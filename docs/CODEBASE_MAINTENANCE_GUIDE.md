# Codebase Maintenance Guide

This guide is for the person who maintains Tower V2 as a working software product, not just as a one-time app.

## Goal

Keep the app:

- stable for operators
- readable for future edits
- safe for offline deployment
- easy to test before every release

## 1) Core Rules

### Paths

- Use `app_io.paths.P` for operational files.
- Do not hardcode `draw_orders.csv`, `tower_schedule.csv`, `config_coating.json`, or similar filenames directly in tabs/helpers.
- If a new file is needed:
  1. add it to `app_io/paths.py`
  2. use `P.<name>` everywhere else
  3. update `docs/path_map.md` if the path model changes

### Runtime data vs code

- Code and docs live in:
  - `app/`
  - `app_io/`
  - `helpers/`
  - `renders/`
  - `scripts/`
  - `tests/`
  - `docs/`
- Runtime data lives in:
  - `data/`
  - `data_set_csv/`
  - `maintenance/`
  - `parts/`
  - `state/`
  - `reports/`
  - `backups/`

Do not mix these casually.

### UI work

- Keep heavy sections collapsed by default.
- Keep blue/glass style consistent with the current app direction.
- Prefer one strong workflow over duplicated controls in multiple places.
- If one action is already handled in a stronger section, remove the weak duplicate.

### Tests

- Every fix that can regress should get a small targeted test.
- Prefer fast checks in `tests/runners/app_tests.py`.
- Use the release scripts before deployment instead of relying on manual memory.

## 2) Folder Intent

### `app/`

- startup wiring
- runtime bootstrap
- navigation
- routing

### `app_io/`

- path ownership
- path health
- path compatibility
- config/dataset helpers

### `renders/tabs/`

- visible product behavior
- one major tab per file

### `renders/components/`

- reusable UI pieces shared by tabs

### `helpers/`

- pure support logic
- IO helpers
- formatting
- calculations
- maintenance/inventory/order support

### `scripts/cli/`

- operational commands
- report generation
- release checks
- environment checks
- bundle creation

### `tests/runners/`

- deployment-safe test runners
- health checks
- app regression checks

## 3) Safe Change Workflow

Use this order for almost every change:

1. identify the single source of truth
2. patch the code in the real source
3. patch tests if behavior changed
4. patch docs if workflow changed
5. run targeted checks
6. run release checks before deployment

## 4) Before Editing a Feature

Check these first:

- does a helper already exist?
- is the path already defined in `P`?
- is there already a stronger workflow in another tab?
- will the change affect docs, tests, or report exports?

If yes, update them in the same pass.

## 5) Release Discipline

Before calling a version ready:

```bash
python3 scripts/cli/run_preflight.py
python3 scripts/cli/run_app_tests.py
python3 scripts/cli/run_path_permissions_audit.py
python3 scripts/cli/run_env_pretest.py
python3 scripts/cli/run_full_health_check.py
python3 scripts/cli/run_v2_deploy_protocol.py
```

If any of those fail, the version is not ready.

## 6) What Not To Commit Lightly

Be careful with:

- `data/*.csv`
- `state/*.json`
- `reports/`
- `backups/`
- generated PDFs/markdown
- machine-specific local state

Commit them only when they are intentionally part of the product or sample data.

## 7) Recommended Commit Types

Use commits that describe intent:

- `Fix report center media embedding`
- `Add maintenance threshold presets`
- `Tighten V2 deploy docs and checks`
- `Refactor consumables stock totals display`

Avoid vague commit messages.

## 8) When To Add A New Test

Add a test when you change:

- path logic
- report generation
- startup behavior
- navigation/routing
- inventory math
- maintenance scheduling logic
- markdown/pdf generation

## 9) Documentation You Should Keep Current

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/OPERATIONS.md`
- `docs/DEVELOPMENT.md`
- `docs/V2_DEPLOY_PROTOCOL.md`
- `docs/V2_FINAL_SCAN.md`
- role guides if responsibilities change

## 10) Practical Ownership Model

Think of the app as 3 working surfaces:

- Project management
- Operations
- Monitoring and research

When editing:

- do not leak research controls into operator workflows
- do not leak diagnostics responsibility into project-management docs
- keep role boundaries clear unless overlap is intentional

## 11) Definition Of “Clean Enough For Deployment”

The codebase is ready when:

- startup checks pass
- app tests pass
- docs match the real scripts
- report exports work
- paths come from `P`
- no known blocking UI/runtime error remains
- the deployment workflow is documented and repeatable

That is the maintenance standard for this project.
