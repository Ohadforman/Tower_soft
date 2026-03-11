#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P, ensure_dir


@dataclass
class BomRow:
    manual: str
    page: int
    item_no: str
    part: str
    part_number: str
    qty_per_assembly: str
    assembly: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _is_number_line(s: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d+)?", _clean(s)))


def _is_part_number_line(s: str) -> bool:
    t = _clean(s).upper().rstrip(".")
    # Typical codes: 284531, 284330A, H04478, K00558, etc.
    return bool(re.fullmatch(r"[A-Z]?\d{4,6}[A-Z]?", t))


def _looks_like_desc(s: str) -> bool:
    t = _clean(s)
    if not t:
        return False
    if _is_number_line(t) or _is_part_number_line(t):
        return False
    # Avoid pure drawing marks.
    if re.fullmatch(r"[A-Z]-[A-Z]", t):
        return False
    if len(t) < 3:
        return False
    return True


def _extract_parts_list_from_lines(lines: List[str], manual_name: str, page_no: int) -> List[BomRow]:
    out: List[BomRow] = []
    # Find start near PARTS LIST.
    start = 0
    for i, l in enumerate(lines):
        if "PARTS LIST" in _clean(l).upper():
            start = i + 1
            break
    # remove obvious headers/noise
    tokens = []
    for raw in lines[start:]:
        t = _clean(raw)
        if not t:
            continue
        up = t.upper()
        if up in {"DESCRIPTION", "PART NUMBER", "PART", "NUMBER", "QTY", "ITEM"}:
            continue
        if up.startswith("THIS DOCUMENT BELONGS"):
            break
        if up in {"SG CONTROLS", "DRAWN", "DATE"}:
            continue
        tokens.append(t)

    i = 0
    while i + 3 < len(tokens):
        d, pn, qty, item = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
        if _looks_like_desc(d) and _is_part_number_line(pn) and _is_number_line(qty) and _is_number_line(item):
            out.append(
                BomRow(
                    manual=manual_name,
                    page=page_no,
                    item_no=item,
                    part=d,
                    part_number=pn.rstrip("."),
                    qty_per_assembly=qty,
                    assembly=os.path.splitext(manual_name)[0],
                )
            )
            i += 4
            continue
        # fallback: if desc spans 2 lines before part number.
        if i + 4 < len(tokens):
            d2 = f"{d} {pn}"
            pn2, qty2, item2 = tokens[i + 2], tokens[i + 3], tokens[i + 4]
            if _looks_like_desc(d2) and _is_part_number_line(pn2) and _is_number_line(qty2) and _is_number_line(item2):
                out.append(
                    BomRow(
                        manual=manual_name,
                        page=page_no,
                        item_no=item2,
                        part=_clean(d2),
                        part_number=pn2.rstrip("."),
                        qty_per_assembly=qty2,
                        assembly=os.path.splitext(manual_name)[0],
                    )
                )
                i += 5
                continue
        i += 1
    return out


def collect_manual_bom() -> List[BomRow]:
    try:
        import fitz
    except Exception as e:
        raise RuntimeError(f"PyMuPDF (fitz) is required: {e}") from e

    rows: List[BomRow] = []
    pdfs = sorted(glob.glob(os.path.join(P.root_dir, "manuals", "*.pdf")))
    key_pat = re.compile(r"PARTS?\s+LIST|BILL OF MATERIALS|BOM|PART NUMBER|ITEM", re.IGNORECASE)
    for pdf in pdfs:
        manual_name = os.path.basename(pdf)
        try:
            doc = fitz.open(pdf)
        except Exception:
            continue
        for pidx in range(len(doc)):
            txt = doc.load_page(pidx).get_text("text") or ""
            if not key_pat.search(txt):
                continue
            lines = [x for x in txt.splitlines() if _clean(x)]
            ext = _extract_parts_list_from_lines(lines, manual_name, pidx + 1)
            rows.extend(ext)
        doc.close()

    # Dedup by manual/page/item/part_number.
    seen = set()
    uniq = []
    for r in rows:
        k = (r.manual.lower(), r.page, r.item_no, r.part_number.upper(), r.part.lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda x: (x.manual.lower(), x.page, int(x.item_no) if x.item_no.isdigit() else 9999, x.part.lower()))
    return uniq


def build_pdf(rows: List[BomRow], out_pdf: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
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
        title="Manuals BOM Physical Count",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#0d2f57"))
    small = ParagraphStyle("small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7, leading=9)

    story = []
    story.append(Paragraph("Manuals BOM Physical Count List", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Paragraph("Source: parts-list/BOM pages found inside manuals/*.pdf", small))
    story.append(Spacer(1, 4 * mm))

    manuals = sorted({r.manual for r in rows})
    summary = [
        ["Rows extracted", str(len(rows))],
        ["Manuals with BOM rows", str(len(manuals))],
        ["Manual list", "Grouped below by each manual"],
    ]
    s_tbl = Table(summary, colWidths=[45 * mm, 220 * mm])
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

    by_manual = {}
    for r in rows:
        by_manual.setdefault(r.manual, []).append(r)

    n = 1
    for manual in sorted(by_manual.keys()):
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(f"Manual: {manual}", h2))
        hdr = ["#", "Page", "Item", "Part", "Part Number", "Qty/Asm", "Serial", "Qty Counted", "Count Notes"]
        data = [hdr]
        for r in sorted(by_manual[manual], key=lambda x: (x.page, int(x.item_no) if x.item_no.isdigit() else 9999, x.part)):
            data.append([
                str(n),
                str(r.page),
                r.item_no,
                Paragraph(r.part, small),
                r.part_number,
                r.qty_per_assembly,
                "",
                "",
                "",
            ])
            n += 1

        col_widths = [8 * mm, 11 * mm, 10 * mm, 96 * mm, 28 * mm, 14 * mm, 16 * mm, 18 * mm, 35 * mm]
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
    parser = argparse.ArgumentParser(description="Generate manuals BOM physical count PDF.")
    parser.add_argument(
        "--output",
        default=os.path.join(P.reports_dir, "maintenance_todo", f"manuals_bom_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"),
        help="Output PDF file path",
    )
    args = parser.parse_args()

    rows = collect_manual_bom()
    if not rows:
        print("No BOM/parts-list rows found in manuals PDFs.")
        return 1
    build_pdf(rows, args.output)
    print(f"Saved PDF: {args.output}")
    print(f"Rows extracted: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
