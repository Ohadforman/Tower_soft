#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_io.paths import P, ensure_dir


def _import_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception as e:
        raise RuntimeError(
            "reportlab is required. Run with the project venv, for example: .venv/bin/python scripts/cli/run_role_guides_pdf.py"
        ) from e
    return colors, TA_LEFT, TA_RIGHT, A4, ParagraphStyle, getSampleStyleSheet, mm, pdfmetrics, TTFont, Paragraph, SimpleDocTemplate, Spacer


def _looks_hebrew(text: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", text or ""))


def _pick_hebrew_font() -> str | None:
    candidates = [
        "/System/Library/Fonts/SFHebrew.ttf",
        "/System/Library/Fonts/ArialHB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _register_font(pdfmetrics, TTFont, font_name: str, font_path: str | None) -> str:
    if font_path and os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            return font_name
        except Exception:
            pass
    return "Helvetica"


def _md_to_story(md_path: str, out_pdf: str, title: str) -> str:
    colors, TA_LEFT, TA_RIGHT, A4, ParagraphStyle, getSampleStyleSheet, mm, pdfmetrics, TTFont, Paragraph, SimpleDocTemplate, Spacer = _import_reportlab()

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    is_hebrew = any(_looks_hebrew(line) for line in lines)
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if is_hebrew:
        font_path = _pick_hebrew_font()
        body_font = _register_font(pdfmetrics, TTFont, "TowerGuideHebrew", font_path)
        bold_font = body_font

    ensure_dir(os.path.dirname(out_pdf) or ".")
    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    align = TA_RIGHT if is_hebrew else TA_LEFT
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=bold_font, fontSize=17, leading=20, textColor=colors.HexColor("#0d2f57"), alignment=align)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold_font, fontSize=12, leading=15, textColor=colors.HexColor("#124b8a"), alignment=align, spaceBefore=6, spaceAfter=2)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName=bold_font, fontSize=10, leading=13, textColor=colors.HexColor("#15623a"), alignment=align, spaceBefore=4, spaceAfter=1)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=body_font, fontSize=9, leading=12, alignment=align)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12 if not is_hebrew else 0, rightIndent=0 if not is_hebrew else 12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#4b6480"))

    story = [
        Paragraph(title, h1),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small),
        Spacer(1, 4 * mm),
    ]

    for raw in lines:
        line = (raw or "").strip()
        if not line:
            story.append(Spacer(1, 1.6 * mm))
            continue
        safe = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if line.startswith("# "):
            story.append(Paragraph(safe[2:].strip(), h1))
        elif line.startswith("## "):
            story.append(Paragraph(safe[3:].strip(), h2))
        elif line.startswith("### "):
            story.append(Paragraph(safe[4:].strip(), h3))
        elif line.startswith("- "):
            story.append(Paragraph(f"• {safe[2:].strip()}", bullet))
        elif re.match(r"^\d+\.\s", line):
            story.append(Paragraph(safe, bullet))
        else:
            story.append(Paragraph(safe, body))

    doc.build(story)
    return out_pdf


def main() -> int:
    reports_dir = os.path.join(P.reports_dir, "guides")
    ensure_dir(reports_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    en_md = os.path.join(ROOT_DIR, "docs", "APP_ROLE_GUIDE_EN.md")
    he_md = os.path.join(ROOT_DIR, "docs", "APP_ROLE_GUIDE_HE.md")
    en_pdf = os.path.join(reports_dir, f"tower_role_guide_en_{ts}.pdf")
    he_pdf = os.path.join(reports_dir, f"tower_role_guide_he_{ts}.pdf")

    _md_to_story(en_md, en_pdf, "Tower App Role Guide (English)")
    _md_to_story(he_md, he_pdf, "מדריך תפקידי עבודה באפליקציית המגדל")

    print("Generated guides:")
    print(f"EN MD : {en_md}")
    print(f"HE MD : {he_md}")
    print(f"EN PDF: {en_pdf}")
    print(f"HE PDF: {he_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
