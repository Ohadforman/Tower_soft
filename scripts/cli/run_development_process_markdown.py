#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P, ensure_dir


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _parse_path_list(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def _parse_caption_map(value: object) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        raw = json.loads(text)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _is_image(path: str) -> bool:
    ext = os.path.splitext(str(path).lower())[1]
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.lower() in {"nan", "none", "null"}:
        return "-"
    return text if text else "-"


def _project_bundle(project_name: str) -> dict:
    data_dir = P.data_dir
    projects = _read_csv(os.path.join(data_dir, "development_projects.csv"))
    experiments = _read_csv(os.path.join(data_dir, "development_experiments.csv"))
    updates = _read_csv(os.path.join(data_dir, "experiment_updates.csv"))

    project = {}
    if not projects.empty and "Project Name" in projects.columns:
        match = projects[projects["Project Name"].astype(str).str.strip() == project_name].copy()
        if not match.empty:
            project = match.iloc[0].to_dict()

    if not experiments.empty and "Project Name" in experiments.columns:
        experiments = experiments[experiments["Project Name"].astype(str).str.strip() == project_name].copy()
    else:
        experiments = pd.DataFrame()

    if not updates.empty and "Project Name" in updates.columns:
        updates = updates[updates["Project Name"].astype(str).str.strip() == project_name].copy()
    else:
        updates = pd.DataFrame()

    return {"project": project, "experiments": experiments, "updates": updates}


def _attachment_blocks(row: pd.Series) -> tuple[list[str], list[str]]:
    data_dir = P.data_dir
    image_lines: list[str] = []
    file_lines: list[str] = []

    all_paths = []
    for field in ["Result Images", "Result Docs", "Attachments"]:
        if field in row.index:
            all_paths.extend(_parse_path_list(row.get(field, "")))

    captions = {}
    for field in ["Image Captions", "Doc Captions", "Attachment Captions"]:
        if field in row.index:
            captions.update(_parse_caption_map(row.get(field, "")))

    for rel_path in all_paths:
        if os.path.isabs(rel_path):
            abs_path = rel_path
        else:
            candidate_data = os.path.join(data_dir, rel_path)
            candidate_root = os.path.join(ROOT_DIR, rel_path)
            abs_path = candidate_data if os.path.exists(candidate_data) else candidate_root
        file_name = os.path.basename(rel_path)
        caption = captions.get(file_name, "") or captions.get(rel_path, "")
        label = f"{file_name} - {caption}" if caption else file_name
        if _is_image(abs_path) and os.path.exists(abs_path):
            image_lines.append(f"#### {label}")
            image_lines.append("")
            image_lines.append(f"![{label}]({abs_path})")
            image_lines.append("")
        else:
            file_lines.append(f"- [{label}]({abs_path})")
    return image_lines, file_lines


def build_markdown(project_name: str) -> str:
    bundle = _project_bundle(project_name)
    project = bundle["project"]
    experiments = bundle["experiments"]
    updates = bundle["updates"]

    lines: list[str] = []
    lines.append(f"# Development Process Report: {project_name}")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Project Summary")
    lines.append("")
    lines.append(f"- Project Name: `{project.get('Project Name', project_name) or project_name}`")
    lines.append(f"- Project Purpose: {_clean_text(project.get('Project Purpose', '-'))}")
    lines.append(f"- Target: {_clean_text(project.get('Target', '-'))}")
    lines.append(f"- Created At: `{_clean_text(project.get('Created At', '-'))}`")
    lines.append(f"- Archived: `{_clean_text(project.get('Archived', 'False'))}`")
    lines.append(f"- Experiments Count: `{len(experiments)}`")
    lines.append(f"- Updates Count: `{len(updates)}`")
    lines.append("")

    if experiments.empty:
        lines.append("## Experiments")
        lines.append("")
        lines.append("No experiments found for this project.")
        lines.append("")
    else:
        lines.append("## Experiments Overview")
        lines.append("")
        lines.append("| Experiment | Date | Researcher | Is Drawing | Draw CSV |")
        lines.append("|---|---|---|---|---|")
        for _, row in experiments.iterrows():
            lines.append(
                f"| {_clean_text(row.get('Experiment Title', '-'))} | `{_clean_text(row.get('Date', '-'))}` | {_clean_text(row.get('Researcher', '-'))} | `{_clean_text(row.get('Is Drawing', 'False'))}` | {_clean_text(row.get('Draw CSV', '-'))} |"
            )
        lines.append("")

        for idx, row in experiments.iterrows():
            exp_title = _clean_text(row.get("Experiment Title", f"Experiment {idx + 1}"))
            lines.append(f"## Experiment {idx + 1}: {exp_title}")
            lines.append("")
            lines.append(f"- Date: `{_clean_text(row.get('Date', '-'))}`")
            lines.append(f"- Researcher: {_clean_text(row.get('Researcher', '-'))}")
            lines.append(f"- Is Drawing: `{_clean_text(row.get('Is Drawing', 'False'))}`")
            lines.append(f"- Draw CSV: {_clean_text(row.get('Draw CSV', '-'))}")
            lines.append(f"- Drawing Details: {_clean_text(row.get('Drawing Details', '-'))}")
            lines.append("")
            for label, field in [
                ("Purpose", "Purpose"),
                ("Methods", "Methods"),
                ("Observations", "Observations"),
                ("Results", "Results"),
                ("Markdown Notes", "Markdown Notes"),
            ]:
                value = str(row.get(field, "") or "").strip()
                if value:
                    lines.append(f"### {label}")
                    lines.append("")
                    lines.append(value)
                    lines.append("")
            image_lines, file_lines = _attachment_blocks(row)
            if image_lines:
                lines.append("### Images")
                lines.append("")
                lines.extend(image_lines)
            if file_lines:
                lines.append("### Files")
                lines.append("")
                lines.extend(file_lines)
                lines.append("")

    lines.append("## Experiment Updates")
    lines.append("")
    if updates.empty:
        lines.append("No updates found for this project.")
        lines.append("")
    else:
        updates = updates.sort_values("Update Date", kind="stable")
        for _, row in updates.iterrows():
            title = _clean_text(row.get("Experiment Title", "-"))
            update_date = _clean_text(row.get("Update Date", "-"))
            researcher = _clean_text(row.get("Researcher", "-"))
            note = _clean_text(row.get("Update Notes", "-"))
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"- Update Date: `{update_date}`")
            lines.append(f"- Researcher: {researcher}")
            lines.append("")
            lines.append(note)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    projects = _read_csv(os.path.join(P.data_dir, "development_projects.csv"))
    if projects.empty or "Project Name" not in projects.columns:
        raise SystemExit("No development projects found.")

    project_name = ""
    if len(sys.argv) > 1:
        project_name = str(sys.argv[1]).strip()
    if not project_name:
        names = [str(x).strip() for x in projects["Project Name"].dropna().tolist() if str(x).strip()]
        if not names:
            raise SystemExit("No project names found.")
        project_name = names[0]

    out_dir = P.report_center_dir
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"development_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    content = build_markdown(project_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
