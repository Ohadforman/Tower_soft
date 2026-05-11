from __future__ import annotations

import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import server  # noqa: E402


APP_JS = ROOT_DIR / "static" / "app" / "app.js"
SERVER_PY = ROOT_DIR / "server.py"
RUNTIME_FILES = [
    SERVER_PY,
    ROOT_DIR / "static" / "app" / "router.js",
    ROOT_DIR / "static" / "app" / "main.js",
    APP_JS,
    ROOT_DIR / "static" / "index.html",
]
PATH_LEAK_PATTERN = re.compile(r"/Users/ohadformanair|PycharmProjects/Tower_work")
API_LITERAL_PATTERN = re.compile(r'["\'`](/api/[A-Za-z0-9._/?=&%+-]+)')


def check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"name": name, "ok": ok, "detail": detail}


def runtime_path_leak_check() -> dict[str, object]:
    hits: list[str] = []
    for file_path in RUNTIME_FILES:
        text = file_path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if PATH_LEAK_PATTERN.search(line):
                hits.append(f"{file_path.name}:{line_number}")
    if hits:
        return check("runtime-paths", False, f"Absolute workstation paths found in runtime files: {', '.join(hits[:8])}")
    return check("runtime-paths", True, "Runtime files do not contain workstation-specific absolute paths.")


def helper_python_check() -> dict[str, object]:
    helper = server.resolve_helper_python(("pypdf",))
    if helper is None:
        return check("helper-python", False, "No helper Python runtime with pypdf was found for manual indexing.")
    return check("helper-python", True, f"Resolved helper runtime: {helper}")


def manual_renderer_check() -> dict[str, object]:
    mode = server.manual_page_render_mode()
    if mode == "image":
        return check("manual-renderer", True, "Manual page viewer can render page images on this platform.")
    if mode == "pdf-inline":
        return check("manual-renderer", True, "Manual page viewer will use inline PDF fallback instead of image rendering.")
    return check("manual-renderer", False, f"Unexpected manual render mode: {mode}")


def bootstrap_check() -> dict[str, object]:
    payload = server.build_bootstrap_payload().body
    required = {
        "home",
        "schedule",
        "parts",
        "maintenance",
        "consumables",
        "processSetup",
        "orderDraw",
        "dashboard",
        "drawFinalize",
        "diagnostics",
        "reportCenter",
        "sqlLab",
        "development",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        return check("bootstrap", False, f"Bootstrap payload is missing keys: {', '.join(missing)}")
    return check("bootstrap", True, f"Bootstrap payload exposes all {len(required)} required sections.")


def diagnostics_check() -> dict[str, object]:
    payload = server.build_diagnostics_payload().body
    expected_keys = {"ready_count", "tracked_count", "health_checks", "overall_ok"}
    missing = sorted(expected_keys.difference(payload.keys()))
    if missing:
        return check("diagnostics", False, f"Diagnostics payload is missing keys: {', '.join(missing)}")
    return check(
        "diagnostics",
        True,
        f"Diagnostics payload built successfully with {payload.get('passed_checks', 0)}/{payload.get('total_checks', 0)} checks passing.",
    )


def development_payload_check() -> dict[str, object]:
    payload = server.build_development_payload().body
    if "project_names" not in payload or "metrics" not in payload:
        return check("development-payload", False, "Development payload is missing project_names or metrics.")
    return check(
        "development-payload",
        True,
        f"Development payload built successfully with {len(payload.get('project_names', []))} project choices.",
    )


def route_wiring_check() -> dict[str, object]:
    app_text = APP_JS.read_text(encoding="utf-8")
    app_routes = sorted({match.split("?", 1)[0] for match in API_LITERAL_PATTERN.findall(app_text)})
    server_routes = set(server.API_ROUTES.keys()) | set(server.POST_API_ROUTES.keys()) | {
        "/api/maintenance/manual",
        "/api/maintenance/manual-page",
        "/api/maintenance/manual-prefetch",
        "/api/development/media",
        "/api/dashboard/log",
        "/api/dashboard/math-plot-export",
        "/api/report-center/project",
        "/api/report-center/file",
        "/api/sql-lab/dataset",
        "/api/draw-finalize",
        "/api/development/project",
    }
    missing = [route for route in app_routes if route not in server_routes]
    if missing:
        return check("route-wiring", False, f"App references routes with no backend handler: {', '.join(missing[:10])}")
    return check("route-wiring", True, f"Verified {len(app_routes)} app API references against backend route handlers.")


def folder_structure_check() -> dict[str, object]:
    required_dirs = [
        ROOT_DIR / "static" / "app",
        ROOT_DIR / "static" / "styles",
        ROOT_DIR / "tools",
        server.STATE_DIR,
        server.REPORT_CENTER_DIR,
        server.BACKUPS_DIR,
    ]
    missing = [str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path) for path in required_dirs if not path.exists()]
    if missing:
        return check("folder-structure", False, f"Missing runtime or source folders: {', '.join(missing)}")
    return check("folder-structure", True, "Core source and runtime folders are present.")


@contextmanager
def temporary_report_output():
    temp_root = Path(tempfile.mkdtemp(prefix="tower-rebuild-preflight-"))
    original_reports = server.REPORTS_DIR
    original_center = server.REPORT_CENTER_DIR
    try:
        server.REPORTS_DIR = temp_root / "reports"
        server.REPORT_CENTER_DIR = server.REPORTS_DIR / "report_center"
        server.ensure_runtime_directories()
        yield temp_root
    finally:
        server.REPORTS_DIR = original_reports
        server.REPORT_CENTER_DIR = original_center
        shutil.rmtree(temp_root, ignore_errors=True)


def report_export_check() -> dict[str, object]:
    development = server.build_development_payload().body
    project_name = str(development.get("default_project") or (development.get("project_names") or [""])[0]).strip()
    if not project_name:
        return check("report-export", True, "Skipped project export exercise because no development projects were found.")
    with temporary_report_output():
        html_result = server.create_development_report_action({"projectName": project_name, "format": "html"}).body
        markdown_result = server.create_development_report_action({"projectName": project_name, "format": "md"}).body
        html_name = str(html_result.get("fileName") or "")
        markdown_name = str(markdown_result.get("fileName") or "")
        html_path = server.REPORT_CENTER_DIR / html_name
        markdown_path = server.REPORT_CENTER_DIR / markdown_name
        if not html_name or not markdown_name or not html_path.exists() or not markdown_path.exists():
            return check("report-export", False, "Project export did not create both HTML and markdown files in the report center.")
    return check("report-export", True, f"Project export succeeds for {project_name} in both HTML paper and markdown modes.")


def backup_sources_check() -> dict[str, object]:
    sources = server.build_full_backup_sources()
    missing_labels = [str(item["label"]) for item in sources if not Path(item["path"]).exists()]
    if missing_labels:
        return check("backup-sources", False, f"Backup source paths are missing: {', '.join(missing_labels[:8])}")
    return check("backup-sources", True, f"Backup manifest resolves {len(sources)} source locations for full backups.")


def run_preflight() -> int:
    checks = [
        runtime_path_leak_check(),
        helper_python_check(),
        manual_renderer_check(),
        folder_structure_check(),
        bootstrap_check(),
        diagnostics_check(),
        development_payload_check(),
        route_wiring_check(),
        report_export_check(),
        backup_sources_check(),
    ]
    failures = [item for item in checks if not item["ok"]]
    for item in checks:
        status = "PASS" if item["ok"] else "FAIL"
        print(f"[{status}] {item['name']}: {item['detail']}")
    print()
    print(f"Checks passed: {len(checks) - len(failures)}/{len(checks)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_preflight())
