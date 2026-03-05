#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import site
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app_io.paths import P
from tests.runners.preflight import run_checks as run_preflight_checks


@dataclass
class EnvCheckResult:
    code: str
    name: str
    ok: bool
    details: str = ""


class EnvRunner:
    def __init__(self) -> None:
        self.results: list[EnvCheckResult] = []

    def check(self, code: str, name: str, fn: Callable[[], str]) -> None:
        try:
            details = fn() or ""
            self.results.append(EnvCheckResult(code=code, name=name, ok=True, details=details))
        except Exception as e:
            self.results.append(EnvCheckResult(code=code, name=name, ok=False, details=f"{type(e).__name__}: {e}"))


_REQUIRED_IMPORTS = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("plotly", "plotly"),
    ("duckdb", "duckdb"),
    ("pyarrow", "pyarrow"),
    ("reportlab", "reportlab"),
    ("PyPDF2", "PyPDF2"),
]


def _check_python() -> str:
    if sys.version_info < (3, 9):
        raise RuntimeError(f"python {platform.python_version()} detected, need >= 3.9")
    return f"python={platform.python_version()}"


def _check_venv() -> str:
    in_venv = (getattr(sys, "base_prefix", sys.prefix) != sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    if not in_venv:
        return "virtual environment not active (recommended, but not mandatory)"
    return f"venv={sys.prefix}"


def _check_streamlit_cmd() -> str:
    path = shutil.which("streamlit")
    if not path:
        raise RuntimeError("streamlit command not found in PATH")
    return f"streamlit={path}"


def _check_required_imports() -> str:
    missing: list[str] = []
    versions: list[str] = []
    for module_name, pretty in _REQUIRED_IMPORTS:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "n/a")
            versions.append(f"{pretty}={ver}")
        except Exception:
            missing.append(pretty)
    if missing:
        raise RuntimeError("missing imports: " + ", ".join(missing))
    return "; ".join(versions)


def _check_paths_base() -> str:
    required_dirs = [
        P.root_dir,
        P.data_dir,
        P.config_dir,
        P.assets_dir,
        P.logs_dir,
        P.reports_dir,
        P.backups_dir,
        P.dataset_dir,
    ]
    missing = [d for d in required_dirs if not os.path.isdir(d)]
    if missing:
        raise RuntimeError("missing dirs: " + ", ".join(missing))
    return "all core dirs exist"


def _rw_probe(path: str, tag: str) -> str:
    os.makedirs(path, exist_ok=True)
    name = f"._pretest_{tag}_{os.getpid()}_{int(time.time()*1000)}.tmp"
    fp = os.path.join(path, name)
    t0 = time.perf_counter()
    with open(fp, "w", encoding="utf-8") as f:
        f.write("tower-pretest")
    with open(fp, "r", encoding="utf-8") as f:
        _ = f.read()
    os.remove(fp)
    ms = int((time.perf_counter() - t0) * 1000)
    return f"{tag} rw ok ({ms} ms)"


def _check_rw_probes() -> str:
    msgs = [
        _rw_probe(P.data_dir, "data"),
        _rw_probe(P.logs_dir, "logs"),
        _rw_probe(P.backups_dir, "backups"),
    ]
    return "; ".join(msgs)


def _check_storage_latency() -> str:
    """
    Lightweight latency check (helps detect slow network-mounted paths).
    Fails only for very high latency to avoid false alarms.
    """
    samples_ms: list[int] = []
    for idx in range(5):
        name = f"._latency_probe_{os.getpid()}_{idx}_{int(time.time()*1000)}.tmp"
        fp = os.path.join(P.logs_dir, name)
        t0 = time.perf_counter()
        with open(fp, "w", encoding="utf-8") as f:
            f.write("latency-probe")
        with open(fp, "r", encoding="utf-8") as f:
            _ = f.read()
        os.remove(fp)
        samples_ms.append(int((time.perf_counter() - t0) * 1000))
    avg_ms = sum(samples_ms) / len(samples_ms)
    mx = max(samples_ms)
    if mx > 1500:
        raise RuntimeError(f"very high storage latency: max={mx}ms avg={avg_ms:.1f}ms")
    return f"storage latency avg={avg_ms:.1f}ms max={mx}ms (logs dir)"


def _check_duckdb_dir() -> str:
    db_parent = os.path.dirname(P.duckdb_path) or "."
    if not os.path.isdir(db_parent):
        raise RuntimeError(f"duckdb parent missing: {db_parent}")
    if not os.access(db_parent, os.W_OK):
        raise RuntimeError(f"duckdb parent not writable: {db_parent}")
    return f"duckdb_parent={db_parent}"


def _check_disk_space() -> str:
    usage = shutil.disk_usage(P.root_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < 1.0:
        raise RuntimeError(f"low free space: {free_gb:.2f} GB")
    return f"free_space={free_gb:.2f} GB"


def _check_python_site_packages() -> str:
    paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    user_site = site.getusersitepackages() if hasattr(site, "getusersitepackages") else ""
    packed = [p for p in paths if p]
    if user_site:
        packed.append(user_site)
    return "site_packages=" + ", ".join(packed[:3])


def _run_embedded_preflight() -> str:
    res = run_preflight_checks()
    failed = [r for r in res if not r.ok]
    if failed:
        first = failed[0]
        raise RuntimeError(f"preflight failed: [{first.code}] {first.name} -> {first.details}")
    return f"preflight_passed={len(res)}"


def _report_paths() -> tuple[str, str]:
    out_dir = os.path.join(P.reports_dir, "checks")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        os.path.join(out_dir, f"env_pretest_{ts}.json"),
        os.path.join(out_dir, f"env_pretest_{ts}.md"),
    )


def _write_reports(results: list[EnvCheckResult]) -> tuple[str, str, bool]:
    ok = all(r.ok for r in results)
    failed = [r for r in results if not r.ok]
    json_path, md_path = _report_paths()

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "root_dir": P.root_dir,
        "ok": ok,
        "failed_count": len(failed),
        "results": [r.__dict__ for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Environment Pretest Report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Host: {payload['host']}",
        f"- Platform: {payload['platform']}",
        f"- Python: {payload['python']}",
        f"- Root: `{payload['root_dir']}`",
        f"- Result: {'READY ✅' if ok else 'NOT READY ❌'}",
        "",
        "## Checks",
        "",
        "| Status | Code | Check | Details |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        details = (r.details or "").replace("|", "\\|")
        lines.append(f"| {status} | {r.code} | {r.name} | {details} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path, ok


def main() -> int:
    runner = EnvRunner()
    checks = [
        ("EV-01", "Python version", _check_python),
        ("EV-02", "Virtual environment", _check_venv),
        ("EV-03", "Streamlit command", _check_streamlit_cmd),
        ("EV-04", "Required package imports", _check_required_imports),
        ("EV-05", "Core directories", _check_paths_base),
        ("EV-06", "Read/write probes (data/logs/backups)", _check_rw_probes),
        ("EV-07", "DuckDB parent writable", _check_duckdb_dir),
        ("EV-08", "Disk free space", _check_disk_space),
        ("EV-09", "Python site-packages visibility", _check_python_site_packages),
        ("EV-10", "Embedded preflight", _run_embedded_preflight),
        ("EV-11", "Storage latency probe", _check_storage_latency),
    ]

    print("=== Environment Pretest ===")
    for code, name, fn in checks:
        runner.check(code, name, fn)
        r = runner.results[-1]
        if r.ok:
            print(f"[PASS] [{r.code}] {r.name} -> {r.details}")
        else:
            print(f"[FAIL] [{r.code}] {r.name} -> {r.details}")

    json_path, md_path, ok = _write_reports(runner.results)
    print("\n=== Report Files ===")
    print(f"JSON: {json_path}")
    print(f"MD  : {md_path}")
    print("\n=== Result ===")
    print(f"{'READY ✅' if ok else 'NOT READY ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
