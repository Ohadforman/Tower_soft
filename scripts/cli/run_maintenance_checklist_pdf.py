#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P, ensure_dir
from scripts.cli.run_maintenance_flow_playbook import load_maintenance_tasks


def build_checklist_pdf(output_pdf: str) -> int:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as e:
        raise RuntimeError(f"reportlab required: {e}") from e

    ensure_dir(os.path.dirname(output_pdf) or ".")
    tasks = load_maintenance_tasks().copy()

    if tasks.empty:
        raise RuntimeError("No maintenance tasks found.")

    # Keep only rows that are actual tasks and have some manual reference data.
    tasks["Task"] = tasks["Task"].astype(str).fillna("").str.strip()
    tasks["Procedure_Summary"] = tasks["Procedure_Summary"].astype(str).fillna("").str.strip()
    tasks["Manual_Name"] = tasks["Manual_Name"].astype(str).fillna("").str.strip()
    tasks["Page"] = tasks["Page"].astype(str).fillna("").str.strip()
    tasks = tasks[tasks["Task"].ne("")].copy()
    tasks = tasks.sort_values(["Component", "Task"], kind="stable").reset_index(drop=True)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="Maintenance Checklist",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=18)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7, leading=9)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=8, leading=10)

    story = []
    story.append(Paragraph("Maintenance Checklist", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Paragraph("Fields: task name, task description, manual, manual page, and a blank notes column for handwriting.", small))
    story.append(Spacer(1, 4 * mm))

    data = [[
        "#",
        "Task Name",
        "Task Description",
        "Manual",
        "Page",
        "Notes",
    ]]

    for i, (_, row) in enumerate(tasks.iterrows(), start=1):
        task_name = str(row.get("Task", "")).strip()
        task_desc = str(row.get("Procedure_Summary", "")).strip() or task_name
        manual_name = str(row.get("Manual_Name", "")).strip() or str(row.get("Source_File", "")).strip()
        page = str(row.get("Page", "")).strip()
        data.append([
            str(i),
            Paragraph(task_name, body),
            Paragraph(task_desc, body),
            Paragraph(manual_name, body),
            page,
            "",
        ])

    table = Table(
        data,
        colWidths=[10 * mm, 65 * mm, 108 * mm, 36 * mm, 14 * mm, 44 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9eb6cf")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (4, 1), (4, -1), "CENTER"),
                ("ROWHEIGHT", (0, 1), (-1, -1), 15 * mm),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return len(tasks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a maintenance checklist PDF.")
    parser.add_argument(
        "--output",
        default=os.path.join(
            P.reports_dir,
            "maintenance_todo",
            f"maintenance_checklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        ),
        help="Output PDF path",
    )
    args = parser.parse_args()

    count = build_checklist_pdf(args.output)
    print("=== Maintenance Checklist PDF ===")
    print(f"Tasks: {count}")
    print(f"PDF: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
