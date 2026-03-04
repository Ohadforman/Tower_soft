#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

from app_io.paths import P
from helpers.app_logger import log_event


def _run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=P.root_dir, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "exit_code": int(p.returncode),
        "stdout": (p.stdout or "").strip(),
        "stderr": (p.stderr or "").strip(),
    }


def main() -> int:
    checks = [
        [sys.executable, os.path.join(P.root_dir, "run_preflight.py")],
        [sys.executable, os.path.join(P.root_dir, "run_app_tests.py")],
        [sys.executable, os.path.join(P.root_dir, "run_path_permissions_audit.py")],
    ]

    results = [_run(c) for c in checks]
    failed = [r for r in results if r["exit_code"] != 0]

    out_dir = os.path.join(P.reports_dir, "checks")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(out_dir, f"all_checks_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ok": len(failed) == 0,
                "failed_count": len(failed),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Combined report: {report_path}")
    for r in results:
        status = "PASS" if r["exit_code"] == 0 else "FAIL"
        print(f"[{status}] {r['cmd']}")

    code = 0 if not failed else 1
    log_event("all_checks_run", ok=(code == 0), failed_count=len(failed), report_path=report_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
