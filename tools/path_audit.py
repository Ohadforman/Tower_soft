#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_io.paths import P
from app_io.path_health import build_path_health_report


CRITICAL_KEYS = [
    "orders_csv",
    "parts_orders_csv",
    "schedule_csv",
    "tower_temps_csv",
    "tower_containers_csv",
    "coating_config_json",
    "protocols_json",
    "dies_config_json",
    "pid_config_json",
    "duckdb_path",
    "home_bg_image",
    "logo_image",
]

LEGACY_ALIASES = {}


def main() -> int:
    report = build_path_health_report(
        P,
        critical_keys=CRITICAL_KEYS,
        legacy_aliases=LEGACY_ALIASES,
    )
    summary = report.get("summary", {})
    print(json.dumps(summary, indent=2))

    crit = int(summary.get("critical_issues", 0))
    if crit > 0:
        print("\nCritical issues:")
        for x in report.get("critical_issues", []):
            print(f"- {x.get('key')}: {x.get('path')}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
