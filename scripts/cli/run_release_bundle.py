#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P


def _project_python() -> str:
    if os.name == "nt":
        cand = os.path.join(P.root_dir, ".venv", "Scripts", "python.exe")
    else:
        cand = os.path.join(P.root_dir, ".venv", "bin", "python")
    return cand if os.path.exists(cand) else sys.executable


def _run_step(name: str, cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=P.root_dir, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    merged = out + ("\n" + err if err else "")
    first = merged.strip().splitlines()[0] if merged.strip() else ""
    return {
        "name": name,
        "cmd": cmd,
        "exit_code": int(proc.returncode),
        "ok": proc.returncode == 0,
        "first_line": first,
        "output": merged.strip(),
    }


def _latest_by_prefix(folder: str, prefix: str, suffix: str) -> str:
    if not os.path.isdir(folder):
        return ""
    cand = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith(prefix) and x.endswith(suffix)]
    if not cand:
        return ""
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def main() -> int:
    py = _project_python()
    started = datetime.now()
    ts = started.strftime("%Y%m%d_%H%M%S")

    checks_dir = os.path.join(P.reports_dir, "checks")
    os.makedirs(checks_dir, exist_ok=True)
    bundle_dir = os.path.join(checks_dir, f"release_bundle_{ts}")
    os.makedirs(bundle_dir, exist_ok=True)

    steps = [
        ("Environment pretest", [py, os.path.join(P.root_dir, "scripts", "cli", "run_env_pretest.py")]),
        ("Full health check", [py, os.path.join(P.root_dir, "scripts", "cli", "run_full_health_check.py")]),
        ("Release check", [py, os.path.join(P.root_dir, "scripts", "cli", "run_release_check.py")]),
        ("V2 deploy protocol", [py, os.path.join(P.root_dir, "scripts", "cli", "run_v2_deploy_protocol.py")]),
    ]

    print("=== Release Bundle ===")
    print(f"Root: {P.root_dir}")
    print(f"Python: {py}")
    print(f"Bundle dir: {bundle_dir}")
    print("")

    results = []
    for name, cmd in steps:
        print(f"[RUN] {name}")
        r = _run_step(name, cmd)
        results.append(r)
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {name}")
        if r["first_line"]:
            print(f"  {r['first_line']}")

    ready = all(x["ok"] for x in results)
    finished = datetime.now()

    # Collect latest artifacts from checks folder.
    refs = {
        "env_pretest_json": _latest_by_prefix(checks_dir, "env_pretest_", ".json"),
        "full_health_json": _latest_by_prefix(checks_dir, "full_health_check_", ".json"),
        "release_check_json": _latest_by_prefix(checks_dir, "release_check_", ".json"),
        "v2_deploy_json": _latest_by_prefix(checks_dir, "v2_deploy_protocol_", ".json"),
        "v2_deploy_md": _latest_by_prefix(checks_dir, "v2_deploy_protocol_", ".md"),
    }

    summary = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "ready": ready,
        "root_dir": P.root_dir,
        "python_used": py,
        "steps": results,
        "artifacts": refs,
    }
    summary_json = os.path.join(bundle_dir, "release_bundle_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_md = os.path.join(bundle_dir, "release_bundle_summary.md")
    lines = [
        "# Release Bundle Summary",
        "",
        f"- Started: `{started.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Finished: `{finished.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Result: **{'READY ✅' if ready else 'NOT READY ❌'}**",
        "",
        "## Steps",
    ]
    for s in results:
        lines.append(f"- [{'PASS' if s['ok'] else 'FAIL'}] **{s['name']}**")
        if s["first_line"]:
            lines.append(f"  - `{s['first_line']}`")
    lines += ["", "## Linked Artifacts"]
    for k, v in refs.items():
        if v:
            lines.append(f"- {k}: `{v}`")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    zip_path = os.path.join(checks_dir, f"release_bundle_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_json, arcname=os.path.basename(summary_json))
        zf.write(summary_md, arcname=os.path.basename(summary_md))
        for p in refs.values():
            if p and os.path.isfile(p):
                zf.write(p, arcname=os.path.basename(p))

    print("\n=== Bundle Result ===")
    print(f"Result: {'READY ✅' if ready else 'NOT READY ❌'}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD  : {summary_md}")
    print(f"Bundle ZIP  : {zip_path}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

