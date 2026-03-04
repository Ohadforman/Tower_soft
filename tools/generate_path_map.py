#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_io.paths import P


def main() -> int:
    out_path = os.path.join(P.root_dir, "docs", "path_map.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append("# Path Map")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("| Key | Path | Exists |")
    lines.append("|---|---|---|")

    for key, val in sorted(getattr(P, "__dict__", {}).items()):
        if not isinstance(val, str):
            continue
        if not key.endswith(("_dir", "_csv", "_json", "_path", "_image", "_log")):
            continue
        exists = "yes" if os.path.exists(val) else "no"
        lines.append(f"| `{key}` | `{val}` | {exists} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
