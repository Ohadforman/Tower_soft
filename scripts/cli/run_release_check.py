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
from helpers.backup_io import create_backup_snapshot


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
    ]

    failed = []
    print("=== Release Check ===")
    for name, cmd in steps:
        code, msg = _run(cmd)
        status = "PASS" if code == 0 else "FAIL"
        print(f"[{status}] {name}")
        if code != 0:
            failed.append((name, code, msg))

    print("\n[RUN] Backup snapshot")
    backup = create_backup_snapshot(P)
    backup_ok = len(backup.errors) == 0
    print(
        f"[{'PASS' if backup_ok else 'WARN'}] Backup snapshot "
        f"(files={backup.copied_files}, errors={len(backup.errors)})"
    )
    print(f"Snapshot: {backup.snapshot_dir}")

    ready = (len(failed) == 0) and backup_ok
    print("\n=== Release Summary ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checks failed: {len(failed)}")
    print(f"Backup errors: {len(backup.errors)}")
    print(f"Result: {'READY ✅' if ready else 'NOT READY ❌'}")

    if failed:
        print("\nFailed checks:")
        for name, code, msg in failed:
            print(f"- {name} (exit={code})")
            if msg:
                first = msg.splitlines()[0]
                print(f"  {first}")

    if backup.errors:
        print("\nBackup warnings:")
        for e in backup.errors[:10]:
            print(f"- {e}")

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
