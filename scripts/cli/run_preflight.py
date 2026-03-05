#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tests.runners.preflight import CheckResult, run, run_checks

__all__ = ["CheckResult", "run", "run_checks"]

if __name__ == "__main__":
    raise SystemExit(run())
