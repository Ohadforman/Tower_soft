#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys

from app_io.paths import P


def main() -> int:
    cmd = [sys.executable, os.path.join(P.root_dir, "tools", "path_permissions_audit.py")]
    proc = subprocess.run(cmd, cwd=P.root_dir)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
