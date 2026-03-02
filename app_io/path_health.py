from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def _is_path_key(key: str) -> bool:
    return key.endswith(("_dir", "_csv", "_json", "_path", "_image"))


def _infer_kind(key: str, value: str) -> str:
    if key.endswith("_dir"):
        return "dir"
    if os.path.isdir(value):
        return "dir"
    return "file"


def _is_writable(path: str, kind: str) -> bool:
    if kind == "dir":
        return os.path.exists(path) and os.access(path, os.W_OK)
    if os.path.exists(path):
        return os.access(path, os.W_OK)
    parent = os.path.dirname(path) or "."
    return os.path.isdir(parent) and os.access(parent, os.W_OK)


def build_path_health_report(
    paths_obj: Any,
    critical_keys: Optional[List[str]] = None,
    legacy_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    critical_keys = set(critical_keys or [])
    items: List[Dict[str, Any]] = []

    for key, value in sorted(getattr(paths_obj, "__dict__", {}).items()):
        if not isinstance(value, str) or not _is_path_key(key):
            continue
        kind = _infer_kind(key, value)
        exists = os.path.exists(value)
        readable = exists and os.access(value, os.R_OK)
        writable = _is_writable(value, kind)
        healthy = exists and readable and writable
        items.append(
            {
                "key": key,
                "path": value,
                "kind": kind,
                "exists": exists,
                "readable": readable,
                "writable": writable,
                "healthy": healthy,
                "critical": key in critical_keys,
            }
        )

    alias_items: List[Dict[str, Any]] = []
    if legacy_aliases:
        root_dir = getattr(paths_obj, "root_dir", os.getcwd())
        for alias_name, canonical in sorted(legacy_aliases.items()):
            alias_path = os.path.join(root_dir, alias_name)
            canonical_exists = bool(canonical and os.path.exists(canonical))
            alias_exists = os.path.exists(alias_path)
            same_size = False
            if alias_exists and canonical_exists:
                try:
                    same_size = os.path.getsize(alias_path) == os.path.getsize(canonical)
                except Exception:
                    same_size = False
            alias_items.append(
                {
                    "alias": alias_name,
                    "alias_path": alias_path,
                    "canonical_path": canonical,
                    "alias_exists": alias_exists,
                    "canonical_exists": canonical_exists,
                    "same_size": same_size,
                    "healthy": (not canonical_exists) or (alias_exists and same_size),
                }
            )

    issues = [x for x in items if not x["healthy"]]
    critical_issues = [x for x in issues if x["critical"]]
    alias_issues = [x for x in alias_items if not x["healthy"]]

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_paths": len(items),
        "healthy_paths": len(items) - len(issues),
        "issues": len(issues),
        "critical_issues": len(critical_issues),
        "alias_checks": len(alias_items),
        "alias_issues": len(alias_issues),
    }

    return {
        "summary": summary,
        "items": items,
        "issues": issues,
        "critical_issues": critical_issues,
        "alias_items": alias_items,
        "alias_issues": alias_issues,
    }

