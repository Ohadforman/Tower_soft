#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=P.root_dir, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    msg = out
    if err:
        msg = (msg + "\n" + err).strip()
    return int(proc.returncode), msg


def main() -> int:
    steps = [
        ("Preflight", [sys.executable, os.path.join(P.root_dir, "scripts", "cli", "run_preflight.py")]),
        ("App tests", [sys.executable, os.path.join(P.root_dir, "scripts", "cli", "run_app_tests.py")]),
        (
            "Path permissions audit",
            [sys.executable, os.path.join(P.root_dir, "scripts", "cli", "run_path_permissions_audit.py")],
        ),
        ("Environment pretest", [sys.executable, os.path.join(P.root_dir, "scripts", "cli", "run_env_pretest.py")]),
        ("Release check", [sys.executable, os.path.join(P.root_dir, "scripts", "cli", "run_release_check.py")]),
    ]

    failed = []
    print("=== Full Health Check ===")
    for name, cmd in steps:
        code, msg = _run(cmd)
        status = "PASS" if code == 0 else "FAIL"
        print(f"[{status}] {name}")
        if code != 0:
            failed.append((name, code, msg))

    ok = len(failed) == 0
    print("\n=== Full Health Summary ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Failed steps: {len(failed)}")
    print(f"Result: {'READY ✅' if ok else 'NOT READY ❌'}")

    if failed:
        print("\nFailures:")
        for name, code, msg in failed:
            print(f"- {name} (exit={code})")
            if msg:
                first = msg.splitlines()[0]
                print(f"  {first}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
