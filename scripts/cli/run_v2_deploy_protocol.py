#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P


@dataclass
class StepResult:
    name: str
    cmd: list[str]
    exit_code: int
    ok: bool
    first_line: str
    output: str


def _project_python() -> str:
    if os.name == "nt":
        cand = os.path.join(P.root_dir, ".venv", "Scripts", "python.exe")
    else:
        cand = os.path.join(P.root_dir, ".venv", "bin", "python")
    return cand if os.path.exists(cand) else sys.executable


def _run(cmd: list[str]) -> StepResult:
    proc = subprocess.run(cmd, cwd=P.root_dir, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    msg = out
    if err:
        msg = (msg + "\n" + err).strip()
    first = msg.splitlines()[0] if msg else ""
    return StepResult(
        name="",
        cmd=cmd,
        exit_code=int(proc.returncode),
        ok=proc.returncode == 0,
        first_line=first,
        output=msg,
    )


def _debug_hint(step_name: str) -> str:
    hints = {
        "Environment pretest": "Run `.venv` install and retry: `.venv/bin/pip install -r requirements.txt` (Windows: `.venv\\Scripts\\pip`).",
        "Preflight": "Open paths/config files from `app_io/paths.py` and fix missing/misplaced files first.",
        "App tests": "Run `python3 scripts/cli/run_app_tests.py` and fix first FAIL code before re-running all.",
        "Path permissions audit": "Check OS permissions for `data/`, `logs/`, `backups/`, `reports/`, and local DuckDB parent dir.",
        "Release check": "Inspect backup errors and preflight/app-tests output; fix blockers then re-run release check.",
    }
    return hints.get(step_name, "Inspect step output details and fix root cause, then re-run this protocol.")


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 deployment protocol runner (go/no-go with debug hints).")
    parser.add_argument("--ui-smoke", action="store_true", help="also run streamlit UI smoke test (slower)")
    args = parser.parse_args()

    py = _project_python()
    steps: list[tuple[str, list[str]]] = [
        ("Environment pretest", [py, os.path.join(P.root_dir, "scripts", "cli", "run_env_pretest.py")]),
        ("Preflight", [py, os.path.join(P.root_dir, "scripts", "cli", "run_preflight.py")]),
        ("App tests", [py, os.path.join(P.root_dir, "scripts", "cli", "run_app_tests.py")]),
        ("Path permissions audit", [py, os.path.join(P.root_dir, "scripts", "cli", "run_path_permissions_audit.py")]),
        ("Release check", [py, os.path.join(P.root_dir, "scripts", "cli", "run_release_check.py")]),
    ]
    if args.ui_smoke:
        steps.insert(
            3,
            ("UI smoke (optional)", [py, os.path.join(P.root_dir, "scripts", "cli", "run_app_tests.py"), "--ui-smoke"]),
        )

    started = datetime.now()
    results: list[StepResult] = []
    failed: list[StepResult] = []

    print("=== V2 Deploy Protocol ===")
    print(f"Root: {P.root_dir}")
    print(f"Python used: {py}")
    print(f"Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    for step_name, cmd in steps:
        print(f"[RUN] {step_name}")
        r = _run(cmd)
        r.name = step_name
        results.append(r)
        if r.ok:
            print(f"[PASS] {step_name}")
        else:
            failed.append(r)
            print(f"[FAIL] {step_name}")
            if r.first_line:
                print(f"  {r.first_line}")

    ended = datetime.now()
    ready = len(failed) == 0
    print("\n=== Summary ===")
    print(f"Finished: {ended.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Failed steps: {len(failed)}")
    print(f"Result: {'READY ✅' if ready else 'NOT READY ❌'}")

    if failed:
        print("\n=== Debug Hints ===")
        for f in failed:
            print(f"- {f.name}: {_debug_hint(f.name)}")

    checks_dir = os.path.join(P.reports_dir, "checks")
    os.makedirs(checks_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(checks_dir, f"v2_deploy_protocol_{ts}.json")
    md_path = os.path.join(checks_dir, f"v2_deploy_protocol_{ts}.md")

    payload = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": ended.isoformat(timespec="seconds"),
        "root_dir": P.root_dir,
        "python_used": py,
        "ready": ready,
        "failed_steps": [f.name for f in failed],
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# V2 Deploy Protocol Report",
        "",
        f"- Root: `{P.root_dir}`",
        f"- Python: `{py}`",
        f"- Started: `{started.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Finished: `{ended.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Result: **{'READY ✅' if ready else 'NOT READY ❌'}**",
        "",
        "## Steps",
    ]
    for r in results:
        lines.append(f"- [{'PASS' if r.ok else 'FAIL'}] **{r.name}**")
        if r.first_line:
            lines.append(f"  - `{r.first_line}`")
    if failed:
        lines += ["", "## Debug Hints"]
        for f in failed:
            lines.append(f"- **{f.name}**: {_debug_hint(f.name)}")
    lines += ["", "## Artifacts", f"- JSON: `{json_path}`", f"- MD: `{md_path}`"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== Artifacts ===")
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

