# Offline Transfer and Update Workflow

This guide describes the exact Tower V2 update flow for a closed network:

1. development machine
2. hard disk
3. middle computer with GitHub access
4. hard disk again
5. closed-network target computer

## Goal

Move the app safely between environments while keeping:

- code history clean
- deployment repeatable
- rollback possible
- network-side and closed-network copies aligned

## 1) Working Model

### Development machine

This is the main coding machine.

Use it for:

- code changes
- tests
- docs
- release preparation

### Middle computer

This is the bridge machine with GitHub access.

Use it for:

- pulling/pushing Git history
- staging release bundles
- copying approved versions to/from external storage

### Closed-network computer

This is the target runtime machine.

Use it for:

- running the app
- validating the deployed version
- collecting runtime data and reports

## 2) Recommended Branch Rule

Keep it simple:

- `master` = latest approved stable line
- optional working branch on development machine before merge

For this project, the most important thing is not fancy Git flow. It is traceable, tested movement between machines.

## 3) Prepare a Release on the Development Machine

From project root:

```bash
python3 scripts/cli/run_preflight.py
python3 scripts/cli/run_app_tests.py
python3 scripts/cli/run_path_permissions_audit.py
python3 scripts/cli/run_env_pretest.py
python3 scripts/cli/run_full_health_check.py
python3 scripts/cli/run_v2_deploy_protocol.py
python3 scripts/cli/run_release_bundle.py
```

Expected outputs:

- `reports/checks/release_bundle_*.zip`
- deploy protocol JSON/MD
- release check outputs

## 4) Copy to Hard Disk

Copy these to the hard disk:

- latest approved code folder or clean Git clone copy
- `reports/checks/release_bundle_*.zip`
- any specific migration notes or operator notes for the target update

Do not casually copy:

- `.venv/`
- cache folders
- random local temp outputs

## 5) Use the Middle Computer

On the middle computer:

1. copy the project from the hard disk
2. check Git state
3. push approved commits to GitHub
4. tag or note the release if needed

Recommended commands:

```bash
git status
git log --oneline -5
git push origin master
```

If the closed-network machine later returns data or fixes:

1. copy back through the hard disk
2. inspect carefully
3. commit only what is intentional
4. push again from the middle computer

## 6) Move to the Closed-Network Computer

On the closed-network computer:

1. copy the approved project from the hard disk
2. create/activate venv
3. install from local wheelhouse or approved offline packages
4. run checks
5. start app

Minimum validation:

```bash
python3 scripts/cli/run_preflight.py
python3 scripts/cli/run_app_tests.py
python3 scripts/cli/run_env_pretest.py
python3 scripts/cli/run_full_health_check.py
```

Then:

```bash
streamlit run dash_try.py
```

## 7) Updating Back From Closed Network

When bringing changes/data back:

### Bring back intentionally

- updated code if code was changed there
- relevant CSV/JSON state if meant to become canonical
- useful reports for review

### Do not blindly merge

- local app state
- temporary report outputs
- one-off test files
- machine-local caches

Review everything on the development machine before commit.

## 8) Recommended Approval Rule

Before a closed-network version becomes the new main version:

1. compare changed files
2. confirm path model still uses `P`
3. rerun tests
4. rerun deploy protocol
5. commit with a clear message

## 9) Best Practice for Runtime Data

Treat these carefully:

- `data/`
- `maintenance/`
- `state/`
- `reports/`

Some of them are real operational records.
Some are generated artifacts.
Some are machine-local state.

Do not assume all changes should return to Git.

## 10) Suggested Release Packet

Each approved release should include:

- code at approved commit
- release bundle zip
- deploy protocol artifact
- final scan result
- version note / short summary

## 11) Rollback Rule

If the closed-network deployment is unstable:

1. stop using that new copy
2. restore the last approved release packet
3. keep the failing copy for investigation
4. compare files before reattempting

## 12) Human Workflow Summary

### Forward direction

Development machine -> Hard disk -> Middle computer -> GitHub record -> Hard disk -> Closed network

### Return direction

Closed network -> Hard disk -> Middle computer -> Development machine -> review -> commit -> push

This project is already strong enough that the workflow should stay disciplined. The cleaner the transfer process, the easier updates and debugging will be later.
