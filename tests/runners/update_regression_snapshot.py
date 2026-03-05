#!/usr/bin/env python3
from __future__ import annotations

import os

from app_io.paths import P
from helpers.regression_snapshot import compute_snapshot, save_baseline


def main() -> int:
    path = os.path.join(P.state_dir, "regression_snapshot.json")
    snap = compute_snapshot(P)
    save_baseline(path, snap)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
