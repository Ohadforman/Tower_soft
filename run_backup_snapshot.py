#!/usr/bin/env python3
from __future__ import annotations

from app_io.paths import P, ensure_backups_dir
from helpers.backup_io import create_backup_snapshot


def main() -> int:
    ensure_backups_dir()
    result = create_backup_snapshot(P)
    print(f"Snapshot: {result.snapshot_dir}")
    print(f"Copied files: {result.copied_files}")
    print(f"Copied bytes: {result.copied_bytes}")
    print(f"Manifest: {result.manifest_path}")
    if result.errors:
        print(f"Warnings: {len(result.errors)}")
        for e in result.errors[:20]:
            print(f"- {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
