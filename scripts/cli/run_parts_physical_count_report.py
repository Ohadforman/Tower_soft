#!/usr/bin/env python3
"""
Build a PDF for physical parts counting from all available project sources.

Sources scanned:
- maintenance tracker templates (.xlsx/.xls/.csv)
- data/part_orders.csv
- data/parts_inventory.csv
- manuals/ + parts/ document filenames (reference only)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P, ensure_dir


@dataclass
class PartEntry:
    name: str
    components: Set[str] = field(default_factory=set)
    used_for: Set[str] = field(default_factory=set)
    manual_refs: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    confidence: str = "High"


def _clean(s: object) -> str:
    return str(s or "").strip()


def _norm_part_name(s: str) -> str:
    s = _clean(s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" -;,.")
    return s


def _is_empty_like(s: str) -> bool:
    return _clean(s).lower() in {"", "nan", "none", "n/a", "na", "-"}


def _split_required_parts(raw: str) -> List[str]:
    s = _clean(raw)
    if _is_empty_like(s):
        return []
    # Normalize separators first.
    s = s.replace("\n", ";").replace("|", ";").replace(",", ";")
    chunks = [x.strip() for x in s.split(";") if x.strip()]
    out: List[str] = []
    for c in chunks:
        # Keep slash as secondary split to capture lists like "A / B".
        subs = [x.strip() for x in c.split("/") if x.strip()]
        if len(subs) > 1:
            out.extend(subs)
        else:
            out.append(c)
    # Deduplicate preserve order.
    seen = set()
    uniq = []
    for p in out:
        pn = _norm_part_name(p)
        if _is_empty_like(pn):
            continue
        k = pn.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(pn)
    return uniq


def _infer_parts_from_task(task_name: str) -> List[str]:
    """
    Low-confidence heuristic extraction from task text when Required Parts is empty.
    """
    t = _clean(task_name)
    if not t:
        return []
    patterns = [
        r"replace\s+([^;,.()]+)",
        r"inspect/replace\s+([^;,.()]+)",
        r"clean/replace\s+([^;,.()]+)",
        r"check\s+([^;,.()]+)\s+for",
    ]
    found: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            p = _norm_part_name(m.group(1))
            if p and not _is_empty_like(p):
                found.append(p)
    # Lightweight cleanup for generic noise.
    bad = {"if needed", "needed", "damage", "wear", "contamination", "alignment"}
    out = []
    for p in found:
        if p.lower() in bad:
            continue
        out.append(p)
    # Dedup
    seen = set()
    uniq = []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    return pd.read_excel(path)


def _add_part(
    bag: Dict[str, PartEntry],
    *,
    part_name: str,
    component: str = "",
    used_for: str = "",
    manual_ref: str = "",
    source: str = "",
    confidence: str = "High",
) -> None:
    pn = _norm_part_name(part_name)
    if _is_empty_like(pn):
        return
    key = pn.lower()
    if key not in bag:
        bag[key] = PartEntry(name=pn, confidence=confidence)
    ent = bag[key]
    # Preserve strongest confidence.
    if ent.confidence != "High" and confidence == "High":
        ent.confidence = "High"
    if component and not _is_empty_like(component):
        ent.components.add(_clean(component))
    if used_for and not _is_empty_like(used_for):
        ent.used_for.add(_clean(used_for))
    if manual_ref and not _is_empty_like(manual_ref):
        ent.manual_refs.add(_clean(manual_ref))
    if source and not _is_empty_like(source):
        ent.sources.add(_clean(source))


def collect_parts() -> List[PartEntry]:
    bag: Dict[str, PartEntry] = {}

    # 1) Maintenance trackers (primary source).
    maint_files = sorted(glob.glob(os.path.join(P.maintenance_dir, "*.*")))
    for fp in maint_files:
        base = os.path.basename(fp)
        if base.startswith("_"):
            continue
        if not base.lower().endswith((".xlsx", ".xls", ".csv")):
            continue
        if "log" in base.lower():
            continue
        try:
            df = _read_any_table(fp)
        except Exception:
            continue
        cols = {str(c).strip(): c for c in df.columns}
        c_equipment = cols.get("Equipment") or cols.get("Component")
        c_task = cols.get("Task Name") or cols.get("Task")
        c_req = cols.get("Required Parts") or cols.get("Required_Parts")
        c_doc = cols.get("Document Name")
        c_page = cols.get("Manual Page") or cols.get("Page")
        c_group = cols.get("Group") or cols.get("Task_Group")
        c_proc = cols.get("Procedure Summary")

        for _, r in df.iterrows():
            equipment = _clean(r.get(c_equipment, "")) if c_equipment else ""
            task = _clean(r.get(c_task, "")) if c_task else ""
            req = _clean(r.get(c_req, "")) if c_req else ""
            doc = _clean(r.get(c_doc, "")) if c_doc else ""
            page = _clean(r.get(c_page, "")) if c_page else ""
            group = _clean(r.get(c_group, "")) if c_group else ""
            proc = _clean(r.get(c_proc, "")) if c_proc else ""
            manual_ref = f"{doc} p.{page}".strip() if (doc or page) else ""
            used_for = " | ".join([x for x in [equipment, task, group, proc] if x][:3])

            parts = _split_required_parts(req)
            if parts:
                for p in parts:
                    _add_part(
                        bag,
                        part_name=p,
                        component=equipment,
                        used_for=used_for,
                        manual_ref=manual_ref,
                        source=f"maintenance/{base}",
                        confidence="High",
                    )
            else:
                # Heuristic candidates from task text.
                inferred = _infer_parts_from_task(task)
                for p in inferred:
                    _add_part(
                        bag,
                        part_name=p,
                        component=equipment,
                        used_for=f"{equipment} | {task}",
                        manual_ref=manual_ref,
                        source=f"maintenance/{base} (heuristic)",
                        confidence="Medium",
                    )

    # 2) Existing parts orders.
    if os.path.exists(P.parts_orders_csv):
        try:
            odf = pd.read_csv(P.parts_orders_csv, keep_default_na=False)
        except Exception:
            odf = pd.DataFrame()
        for _, r in odf.iterrows():
            p = _clean(r.get("Part Name", ""))
            d = _clean(r.get("Details", ""))
            comp = _clean(r.get("Project Name", ""))
            _add_part(
                bag,
                part_name=p,
                component=comp,
                used_for=f"Order details: {d}" if d else "Order history",
                source="data/part_orders.csv",
                confidence="High",
            )

    # 3) Existing inventory.
    if os.path.exists(P.parts_inventory_csv):
        try:
            idf = pd.read_csv(P.parts_inventory_csv, keep_default_na=False)
        except Exception:
            idf = pd.DataFrame()
        for _, r in idf.iterrows():
            p = _clean(r.get("Part Name", ""))
            comp = _clean(r.get("Component", ""))
            notes = _clean(r.get("Notes", ""))
            _add_part(
                bag,
                part_name=p,
                component=comp,
                used_for=f"Inventory note: {notes}" if notes else "Inventory record",
                source="data/parts_inventory.csv",
                confidence="High",
            )

    # 4) Manual/doc references (filename-only hints).
    for root in [os.path.join(P.root_dir, "manuals"), os.path.join(P.root_dir, "parts")]:
        if not os.path.isdir(root):
            continue
        for fp in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
            if not os.path.isfile(fp):
                continue
            if not fp.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                continue
            base = os.path.basename(fp)
            label = os.path.splitext(base)[0].replace("_", " ").replace("-", " ")
            label = re.sub(r"\s+", " ", label).strip()
            if len(label) < 4:
                continue
            # Keep this as low-confidence reference item.
            _add_part(
                bag,
                part_name=label,
                used_for="Manual/datasheet reference",
                source=os.path.relpath(fp, P.root_dir),
                confidence="Low",
            )

    items = list(bag.values())
    items.sort(key=lambda x: x.name.lower())
    return items


def build_pdf(items: List[PartEntry], out_pdf: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as e:
        raise RuntimeError(
            "reportlab is required. Run from project venv. "
            f"Import error: {e}"
        ) from e

    ensure_dir(os.path.dirname(out_pdf) or ".")

    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="Tower Parts Physical Count List",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "h1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#0d2f57"),
        spaceAfter=5,
    )
    normal = ParagraphStyle(
        "normal",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    )
    small = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
    )

    story = []
    story.append(Paragraph("Tower Parts Physical Count List", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(
        Paragraph(
            "Scope: consolidated from maintenance templates, current orders, inventory records, and manuals/datasheet references.",
            small,
        )
    )
    story.append(Spacer(1, 4 * mm))

    hi = sum(1 for x in items if x.confidence == "High")
    mi = sum(1 for x in items if x.confidence == "Medium")
    lo = sum(1 for x in items if x.confidence == "Low")
    summary_data = [
        ["Total unique parts", str(len(items))],
        ["High confidence", str(hi)],
        ["Medium confidence", str(mi)],
        ["Low confidence (manual name hints)", str(lo)],
    ]
    summary_tbl = Table(summary_data, colWidths=[70 * mm, 35 * mm])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9eb6cf")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4fb")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_tbl)
    story.append(Spacer(1, 5 * mm))

    table_rows = [
        [
            "#",
            "Part",
            "Used For (sample)",
            "Component(s)",
            "Manual Ref(s)",
            "Confidence",
            "Qty Counted",
            "Count Notes",
        ]
    ]
    for i, it in enumerate(items, start=1):
        used = "; ".join(sorted(it.used_for))[:260]
        comps = "; ".join(sorted(it.components))[:120]
        mrefs = "; ".join(sorted(it.manual_refs))[:140]
        table_rows.append(
            [
                str(i),
                Paragraph(it.name, normal),
                Paragraph(used or "-", small),
                Paragraph(comps or "-", small),
                Paragraph(mrefs or "-", small),
                it.confidence,
                "",
                "",
            ]
        )

    col_widths = [8 * mm, 56 * mm, 90 * mm, 42 * mm, 48 * mm, 16 * mm, 20 * mm, 28 * mm]
    tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9eb6cf")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
            ]
        )
    )
    story.append(tbl)

    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build parts physical count PDF from all available sources.")
    parser.add_argument(
        "--output",
        default=os.path.join(P.reports_dir, "maintenance_todo", f"parts_physical_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()

    items = collect_parts()
    if not items:
        print("No parts found from available sources.")
        return 1
    build_pdf(items, args.output)
    print(f"Saved PDF: {args.output}")
    print(f"Total parts: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
