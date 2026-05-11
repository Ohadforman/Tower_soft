# Tower Rebuild

Tower Rebuild is the custom web-app version of the Tower control system. It serves a browser UI from `static/` and a Python API from `server.py`.

## What belongs in git

Commit the rebuild source:

- `server.py`
- `static/`
- `tools/`
- `requirements.txt`
- docs like this `README.md`

Do not commit runtime output:

- backups
- reports
- state caches
- logs
- uploaded local media

Those are ignored by the local `.gitignore` in this folder.

## Python dependencies

```bash
cd tower_rebuild
python3 -m pip install -r requirements.txt
```

## Run locally

```bash
cd tower_rebuild
python3 server.py
```

Default app URL:

[http://127.0.0.1:8010/#/home](http://127.0.0.1:8010/#/home)

## Deployment shape

This app expects a runtime root that contains the live Tower folders such as:

- `data/`
- `config/`
- `maintenance/`
- `data_set_csv/`
- `logs/`
- `reports/`
- `backups/`
- `state/`
- `manuals/`

If you deploy the whole `Tower_work` repo together, the default relative paths work as-is.

If you deploy `tower_rebuild` separately, configure the runtime root with environment variables instead of editing code.

## Environment variables

- `TOWER_REBUILD_HOST`
  Default: `127.0.0.1`
- `TOWER_REBUILD_PORT`
  Default: `8010`
- `TOWER_REBUILD_ROOT_DIR`
  Absolute path to the Tower runtime root that contains `data`, `config`, `maintenance`, `logs`, and the other live folders.
- `TOWER_REBUILD_DATA_DIR`
  Optional explicit override for the `data` folder if it is not under `ROOT_DIR/data`.
- `TOWER_REBUILD_HELPER_PYTHON`
  Optional Python path for helper scripts like manual indexing.
- `TOWER_REBUILD_MANUAL_PAGE_MODE`
  Optional override for manual rendering mode. Use `image` or `pdf-inline`.

Example:

```bash
export TOWER_REBUILD_ROOT_DIR="/absolute/path/to/Tower_work"
export TOWER_REBUILD_HOST="0.0.0.0"
export TOWER_REBUILD_PORT="8010"
cd tower_rebuild
python3 server.py
```

## Cross-platform notes

- On macOS, the manual browser can render page images through the Swift helper when available.
- On Windows and other non-mac environments, the manual browser falls back to inline PDF mode.
- Full backups are app-driven. The app can create them from the Diagnostics page, and the weekly backup policy runs only while the app is active.

## Release preflight

Run the preflight before pushing or deploying:

```bash
cd tower_rebuild
python3 tools/release_preflight.py
```

This checks:

- path leakage / hardcoded workstation paths
- helper runtime resolution
- manual render mode
- folder structure
- bootstrap and diagnostics payloads
- route wiring
- project paper export
- backup source coverage

## Suggested git push flow

From the repo root:

```bash
cd <repo-root>
git status --short
git add tower_rebuild
git commit -m "Add Tower rebuild deployment-ready app"
git push origin <branch-name>
```

Because this repo already has other tracked live-data changes outside `tower_rebuild`, keep the commit scoped to the rebuild folder unless you intentionally want to deploy those runtime data changes too.
