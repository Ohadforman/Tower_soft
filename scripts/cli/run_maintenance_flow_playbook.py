#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P, ensure_dir


NORMALIZE_MAP: Dict[str, str] = {
    "equipment": "Component",
    "task name": "Task",
    "task": "Task",
    "task id": "Task_ID",
    "group": "Task_Group",
    "maintenance group": "Task_Group",
    "todo group": "Task_Group",
    "groups": "Task_Groups",
    "task groups": "Task_Groups",
    "maintenance groups": "Task_Groups",
    "required parts": "Required_Parts",
    "parts needed": "Required_Parts",
    "needed parts": "Required_Parts",
    "tracking mode": "Tracking_Mode",
    "hours source": "Hours_Source",
    "interval value": "Interval_Value",
    "interval unit": "Interval_Unit",
    "manual page": "Page",
    "document name": "Manual_Name",
    "document file/link": "Document",
    "procedure summary": "Procedure_Summary",
    "safety/notes": "Notes",
    "owner": "Owner",
    "source_file": "Source_File",
}


def _clean(v: object) -> str:
    return str(v or "").strip()


def _mode_norm(v: object) -> str:
    s = _clean(v).lower()
    if s in {"draw", "draws", "draw_count", "draws_count"}:
        return "draws"
    return s


def _split_groups(row: pd.Series) -> List[str]:
    out: List[str] = []
    for col in ("Task_Group", "Task_Groups"):
        raw = _clean(row.get(col, ""))
        if not raw:
            continue
        for p in raw.replace("|", ",").replace(";", ",").split(","):
            g = _clean(p)
            if g and g.lower() not in {x.lower() for x in out}:
                out.append(g)
    return out


def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    return pd.read_excel(path)


def _normalize_df(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    renamed = {}
    for c in df.columns:
        key = _clean(c).lower()
        renamed[c] = NORMALIZE_MAP.get(key, _clean(c))
    out = df.rename(columns=renamed).copy()
    for c in [
        "Component",
        "Task",
        "Task_ID",
        "Task_Group",
        "Task_Groups",
        "Tracking_Mode",
        "Hours_Source",
        "Interval_Value",
        "Interval_Unit",
        "Required_Parts",
        "Manual_Name",
        "Page",
        "Procedure_Summary",
        "Notes",
    ]:
        if c not in out.columns:
            out[c] = ""
    out["Source_File"] = source_file
    out = out[out["Component"].astype(str).str.strip().ne("") & out["Task"].astype(str).str.strip().ne("")].copy()
    return out


def load_maintenance_tasks() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(P.maintenance_dir, "*.*")))
    blocks: List[pd.DataFrame] = []
    for fp in files:
        base = os.path.basename(fp)
        if base.startswith("_"):
            continue
        if not base.lower().endswith((".xlsx", ".xls", ".csv")):
            continue
        if any(k in base.lower() for k in ["log", "state", "work_package", "reservation", "action"]):
            continue
        try:
            raw = _read_any(fp)
            norm = _normalize_df(raw, base)
            blocks.append(norm)
        except Exception:
            continue
    if not blocks:
        return pd.DataFrame(columns=["Component", "Task", "Task_ID", "Task_Group", "Task_Groups", "Tracking_Mode", "Hours_Source", "Interval_Value", "Interval_Unit", "Required_Parts", "Source_File"])
    all_df = pd.concat(blocks, ignore_index=True)
    for c in ["Component", "Task", "Task_ID", "Task_Group", "Task_Groups", "Tracking_Mode", "Hours_Source", "Interval_Value", "Interval_Unit", "Required_Parts", "Source_File"]:
        all_df[c] = all_df[c].astype(str).fillna("")

    # Drop obvious empty placeholders
    empty_task = all_df["Task"].str.strip().str.lower().isin({"", "-", "--", "nan", "none", "null", "(blank)"})
    all_df = all_df[~empty_task].copy()

    # De-dup by ID if exists, otherwise by component+task+source
    all_df["_dedup_key"] = all_df.apply(
        lambda r: _clean(r.get("Task_ID", "")).lower()
        or f"{_clean(r.get('Component', '')).lower()}|{_clean(r.get('Task', '')).lower()}|{_clean(r.get('Source_File','')).lower()}",
        axis=1,
    )
    all_df = all_df.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])
    return all_df.reset_index(drop=True)


def build_non_draw_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["Mode"] = work["Tracking_Mode"].apply(_mode_norm)
    work["Groups"] = work.apply(lambda r: ", ".join(_split_groups(r)), axis=1)

    # Non-draw-only means not pure per-draw/startup cadence.
    g_lower = work["Groups"].str.lower()
    draw_group_mask = g_lower.str.contains("per-draw/startup|per draw|startup", na=False)
    draw_mode_mask = work["Mode"].eq("draws")
    non_draw = work[~(draw_group_mask | draw_mode_mask)].copy()

    # Keep only practical columns for planning/building.
    non_draw = non_draw[
        [
            "Component",
            "Task",
            "Task_ID",
            "Groups",
            "Mode",
            "Hours_Source",
            "Interval_Value",
            "Interval_Unit",
            "Required_Parts",
            "Source_File",
        ]
    ].copy()

    # Sort by component/task
    non_draw = non_draw.sort_values(["Component", "Task"], kind="stable").reset_index(drop=True)
    return non_draw


def _p(v: object):
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    return Paragraph(str(v or ""), styles["BodyText"])


def build_pdf(all_tasks: pd.DataFrame, non_draw: pd.DataFrame, out_pdf: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
    except Exception as e:
        raise RuntimeError(f"reportlab required: {e}") from e

    ensure_dir(os.path.dirname(out_pdf) or ".")
    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="Maintenance Flow Playbook",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=18)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=13, textColor=colors.HexColor("#0d2f57"))
    b = ParagraphStyle("b", parent=styles["BodyText"], fontName="Helvetica", fontSize=8, leading=10)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7, leading=9)

    story = []
    story.append(Paragraph("Tower Maintenance Flow - From Start to Actual Task Build", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Paragraph(f"Source folder: {P.maintenance_dir}", small))
    story.append(Spacer(1, 3 * mm))

    summary = [
        ["Total maintenance tasks loaded", str(len(all_tasks))],
        ["Non-draw-only tasks (planning cadence)", str(len(non_draw))],
        ["Purpose", "Operational flow until building actual executable tasks"],
    ]
    s_tbl = Table(summary, colWidths=[70 * mm, 190 * mm])
    s_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9eb6cf")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4fb")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(s_tbl)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Flow Steps", h2))
    flow_steps = [
        "1) Fill inventory baseline: stock, mounted qty, storage location, serial, min level, and item type.",
        "2) Correlate inventory with app data: component mapping, tools vs consumables, and mounted vs stock.",
        "3) Build maintenance tasks in Builder: task groups, timing triggers (hours/draw/calendar), BOM parts + tools, and work package.",
        "4) Readiness check: pick tasks with available parts first; for missing parts create orders from the same flow.",
        "5) Build actual executable tasks/day packs: preparation, safety, manual pages, and procedure are ready before execution.",
        "6) Execute and record: start task, mark done/wait-for-part, update inventory, history, and reporting.",
    ]
    for step in flow_steps:
        story.append(Paragraph(step, b))
        story.append(Spacer(1, 1.5 * mm))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("App Maintenance Workflow (Implemented)", h2))
    app_flow = [
        "- Unified Maintenance Flow in app: Builder -> Prepare Day Pack -> Schedule + Forecast -> Execute + Records.",
        "- Inventory Center supports stock/mounted model, locations, tools classification, and low-stock ordering.",
        "- Work Package contains Preparation, Safety protocol, Procedure, stop plan, and completion criteria.",
        "- Manual context links task pages and BOM pages for technician guidance during execution.",
        "- Execute lane supports done/wait-for-part actions, readiness checks, and reservation/order bridging.",
        "- Diagnostics + path governance ensure stable paths/config for app deployment and debugging.",
    ]
    for line in app_flow:
        story.append(Paragraph(line, b))
        story.append(Spacer(1, 1.0 * mm))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Quality Gate Before Execution", h2))
    quality = [
        "- Every task has Task_ID, Group(s), Trigger mode, and interval.",
        "- Every task has BOM split into parts + tools.",
        "- Safety protocol is present and high-fall-risk tasks force NO ENTRY.",
        "- Missing parts generate order candidates before task execution.",
        "- Manual context opens task procedure pages and relevant BOM pages.",
    ]
    for q in quality:
        story.append(Paragraph(q, b))

    story.append(PageBreak())
    story.append(Paragraph("Table: Tasks Not Only Per-Draw / Startup (Regular Cadence)", h2))
    story.append(Paragraph("This table excludes draw-only startup tasks so planning can focus on weekly/monthly/3M/6M/hours/calendar/on-condition work.", small))
    story.append(Spacer(1, 2 * mm))

    headers = [
        "#",
        "Component",
        "Task",
        "Task ID",
    ]
    data = [headers]
    if non_draw.empty:
        data.append(["-", "No tasks found", "", ""])
    else:
        for i, (_, r) in enumerate(non_draw.iterrows(), start=1):
            data.append(
                [
                    str(i),
                    Paragraph(_clean(r.get("Component", "")), small),
                    Paragraph(_clean(r.get("Task", "")), small),
                    _clean(r.get("Task_ID", "")),
                ]
            )

    col_widths = [10 * mm, 52 * mm, 170 * mm, 35 * mm]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9eb6cf")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
            ]
        )
    )
    story.append(tbl)

    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate maintenance flow playbook PDF.")
    parser.add_argument(
        "--output",
        default=os.path.join(
            P.reports_dir,
            "maintenance_todo",
            f"maintenance_flow_playbook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        ),
        help="Output PDF file path",
    )
    args = parser.parse_args()

    all_tasks = load_maintenance_tasks()
    non_draw = build_non_draw_table(all_tasks)
    build_pdf(all_tasks, non_draw, args.output)

    print("=== Maintenance Flow Playbook ===")
    print(f"All tasks loaded : {len(all_tasks)}")
    print(f"Non-draw tasks  : {len(non_draw)}")
    print(f"PDF: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
