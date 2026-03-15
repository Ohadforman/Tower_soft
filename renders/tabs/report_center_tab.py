from __future__ import annotations

from datetime import datetime
import os
import glob
import json
import re

import pandas as pd
import streamlit as st

from app_io.paths import P as APP_PATHS, ensure_report_center_dir, report_center_path
from scripts.cli.run_weekly_report import (
    _build_fault_summary,
    _build_gas_summary,
    _build_maintenance_summary,
    _build_sap_summary,
    _df_for_pdf_table,
    _expand_schedule_for_window,
    _parse_dt_robust,
    _read_csv_safe,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SECTIONS = [
    "Executive Summary",
    "Resources: Gas + SAP + Preforms",
    "Draw Outcomes (Done/Failed + Notes)",
    "Parts Orders Status",
    "Schedule: Past Week + Next Week",
    "Maintenance + Faults",
    "Maintenance Tests + Measurements",
    "Consumables Snapshot",
]


def _mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _glob_reports_cached(report_dir: str, pattern: str, dir_mtime: float) -> list[str]:
    return sorted(glob.glob(os.path.join(report_dir, pattern)), key=os.path.getmtime, reverse=True)


def _parse_path_list(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


def _parse_caption_map(value: str) -> dict[str, str]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        raw = json.loads(value)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _is_image_path(path: str) -> bool:
    ext = os.path.splitext(str(path).lower())[1]
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def _resolve_report_media_path(data_dir: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    if os.path.isabs(raw_path):
        return raw_path
    primary = os.path.join(data_dir, raw_path)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(ROOT_DIR, raw_path)
    if os.path.exists(fallback):
        return fallback
    return primary


def _looks_hebrew(text: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", str(text or "")))


def _pick_pdf_unicode_font() -> str | None:
    candidates = [
        "/System/Library/Fonts/SFHebrew.ttf",
        "/System/Library/Fonts/ArialHB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _safe_para_text(value: object) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")


def _plain_text(value: object) -> str:
    s = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.MULTILINE)
    s = s.replace("$$", "").replace("$", "")
    return s.strip()


def _safe_report_slug(value: str, fallback: str = "report") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text, flags=re.UNICODE).strip("_")
    return text or fallback


def _next_available_report_filename(base_name: str, ext: str, directory: str) -> str:
    clean_base = str(base_name or "").strip() or "Report"
    clean_ext = ext if str(ext).startswith(".") else f".{ext}"
    candidate = f"{clean_base}{clean_ext}"
    if not os.path.exists(os.path.join(directory, candidate)):
        return candidate
    idx = 2
    while True:
        candidate = f"{clean_base} ({idx}){clean_ext}"
        if not os.path.exists(os.path.join(directory, candidate)):
            return candidate
        idx += 1


def _prepare_development_project_bundle(data_dir: str, project_name: str) -> dict:
    projects_fp = os.path.join(data_dir, "development_projects.csv")
    experiments_fp = os.path.join(data_dir, "development_experiments.csv")
    updates_fp = os.path.join(data_dir, "experiment_updates.csv")

    projects = _read_csv_safe(projects_fp)
    experiments = _read_csv_safe(experiments_fp)
    updates = _read_csv_safe(updates_fp)

    project_row = {}
    if not projects.empty and "Project Name" in projects.columns:
        match = projects[projects["Project Name"].astype(str).str.strip() == str(project_name).strip()].copy()
        if not match.empty:
            project_row = match.iloc[0].to_dict()

    if not experiments.empty and "Project Name" in experiments.columns:
        experiments = experiments[experiments["Project Name"].astype(str).str.strip() == str(project_name).strip()].copy()
    else:
        experiments = pd.DataFrame()

    if not updates.empty and "Project Name" in updates.columns:
        updates = updates[updates["Project Name"].astype(str).str.strip() == str(project_name).strip()].copy()
    else:
        updates = pd.DataFrame()

    for col in [
        "Experiment Title", "Date", "Researcher", "Methods", "Purpose", "Observations",
        "Results", "Is Drawing", "Drawing Details", "Draw CSV", "Attachments",
        "Attachment Captions", "Markdown Notes"
    ]:
        if col not in experiments.columns:
            experiments[col] = ""

    for col in ["Experiment Title", "Update Date", "Researcher", "Update Notes"]:
        if col not in updates.columns:
            updates[col] = ""

    attachment_summary_rows = []
    image_paths = []
    for _, row in experiments.iterrows():
        exp_title = str(row.get("Experiment Title", "")).strip()
        all_paths = []
        for field in ["Result Images", "Result Docs", "Attachments"]:
            if field in row.index:
                all_paths.extend(_parse_path_list(str(row.get(field, ""))))
        captions = {}
        for field in ["Image Captions", "Doc Captions", "Attachment Captions"]:
            if field in row.index:
                captions.update(_parse_caption_map(str(row.get(field, ""))))
        for p in all_paths:
            cap = captions.get(os.path.basename(p), "") or captions.get(p, "")
            abs_p = _resolve_report_media_path(data_dir, p)
            attachment_summary_rows.append(
                {
                    "Experiment": exp_title,
                    "File": os.path.basename(p),
                    "Type": os.path.splitext(p)[1].lower().lstrip("."),
                    "Caption": cap,
                    "Path": abs_p,
                }
            )
            if _is_image_path(abs_p) and os.path.exists(abs_p):
                image_paths.append({"experiment": exp_title, "path": abs_p, "caption": cap or os.path.basename(p)})

    return {
        "project": project_row,
        "experiments": experiments,
        "updates": updates,
        "attachments": pd.DataFrame(attachment_summary_rows),
        "images": image_paths,
    }


def _build_development_project_pdf(out_pdf: str, project_name: str, data_dir: str) -> None:
    try:
        return _build_development_project_pdf_pil(out_pdf, project_name, data_dir)
    except Exception:
        pass

    return _build_development_project_pdf_reportlab(out_pdf, project_name, data_dir)


def _build_development_project_pdf_reportlab(out_pdf: str, project_name: str, data_dir: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    bundle = _prepare_development_project_bundle(data_dir, project_name)
    project = bundle["project"]
    experiments = bundle["experiments"]
    updates = bundle["updates"]
    attachments = bundle["attachments"]
    images = bundle["images"]

    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"Development Process Report - {project_name}",
    )
    styles = getSampleStyleSheet()
    sample_text = " ".join(
        [
            str(project.get("Project Name", "")),
            str(project.get("Project Purpose", "")),
            str(project.get("Target", "")),
            " ".join(experiments.get("Experiment Title", pd.Series(dtype=str)).astype(str).tolist()) if not experiments.empty else "",
            " ".join(experiments.get("Markdown Notes", pd.Series(dtype=str)).astype(str).tolist()) if not experiments.empty else "",
            " ".join(updates.get("Update Notes", pd.Series(dtype=str)).astype(str).tolist()) if not updates.empty else "",
        ]
    )
    use_hebrew = _looks_hebrew(sample_text)
    base_font = "Helvetica"
    unicode_font = _pick_pdf_unicode_font()
    if unicode_font:
        try:
            pdfmetrics.registerFont(TTFont("TowerUnicode", unicode_font))
            base_font = "TowerUnicode"
        except Exception:
            pass

    align = TA_RIGHT if use_hebrew else TA_LEFT
    h1 = ParagraphStyle("DEV_H1", parent=styles["Heading1"], fontName=base_font, fontSize=17, leading=21, textColor=colors.HexColor("#0A3A66"), alignment=align)
    h2 = ParagraphStyle("DEV_H2", parent=styles["Heading2"], fontName=base_font, fontSize=12.5, leading=15, textColor=colors.HexColor("#114E86"), alignment=align)
    h3 = ParagraphStyle("DEV_H3", parent=styles["Heading3"], fontName=base_font, fontSize=10.5, leading=13, textColor=colors.HexColor("#15623a"), alignment=align)
    body = ParagraphStyle("DEV_BODY", parent=styles["BodyText"], fontName=base_font, fontSize=9.2, leading=12, alignment=align)
    small = ParagraphStyle("DEV_SMALL", parent=styles["BodyText"], fontName=base_font, fontSize=8, leading=10, textColor=colors.HexColor("#56708D"), alignment=align)
    callout = ParagraphStyle("DEV_CALLOUT", parent=styles["BodyText"], fontName=base_font, fontSize=9.2, leading=12, alignment=align, textColor=colors.HexColor("#113A63"))

    story = [
        Paragraph(_safe_para_text(f"Development Process Report: {project_name}"), h1),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small),
        Spacer(1, 6),
    ]

    project_rows = [
        ["Project Name", str(project.get("Project Name", project_name) or project_name)],
        ["Project Purpose", str(project.get("Project Purpose", "") or "-")],
        ["Target", str(project.get("Target", "") or "-")],
        ["Created At", str(project.get("Created At", "") or "-")],
        ["Archived", str(project.get("Archived", "") or "False")],
        ["Experiments", str(int(len(experiments)))],
        ["Updates", str(int(len(updates)))],
    ]
    story.append(Paragraph("Project Summary", h2))
    p_tbl = Table(project_rows, colWidths=[45 * mm, 130 * mm])
    p_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A5C9E8")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5FF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), base_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story += [p_tbl, Spacer(1, 8)]

    summary_parts = []
    if str(project.get("Project Purpose", "")).strip():
        summary_parts.append(f"<b>Project Purpose:</b> {_safe_para_text(_plain_text(project.get('Project Purpose', '')))}")
    if str(project.get("Target", "")).strip():
        summary_parts.append(f"<b>Target:</b> {_safe_para_text(_plain_text(project.get('Target', '')))}")
    if summary_parts:
        callout_tbl = Table([[Paragraph("<br/>".join(summary_parts), callout)]], colWidths=[175 * mm])
        callout_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4FAFF")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B7D7F2")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story += [callout_tbl, Spacer(1, 10)]

    story.append(Paragraph("Experiments", h2))
    if experiments.empty:
        story.append(Paragraph("No experiments found for this project.", body))
        story.append(Spacer(1, 6))
    else:
        overview_rows = [["Experiment", "Date", "Researcher", "Has Images", "Has Files"]]
        for _, row in experiments.iterrows():
            all_paths = []
            for field in ["Result Images", "Result Docs", "Attachments"]:
                if field in row.index:
                    all_paths.extend(_parse_path_list(str(row.get(field, ""))))
            has_images = "Yes" if any(_is_image_path(p) for p in all_paths) else "No"
            has_files = "Yes" if all_paths else "No"
            overview_rows.append([
                str(row.get("Experiment Title", "") or "-"),
                str(row.get("Date", "") or "-"),
                str(row.get("Researcher", "") or "-"),
                has_images,
                has_files,
            ])
        story.append(_styled_table(overview_rows, header_font_size=8, body_font_size=8))
        story.append(Spacer(1, 8))

        for idx, row in experiments.iterrows():
            exp_title = str(row.get("Experiment Title", "") or "-")
            if idx > 0:
                story.append(PageBreak())
            story.append(Paragraph(_safe_para_text(exp_title), h3))
            meta_rows = [
                ["Date", str(row.get("Date", "") or "-")],
                ["Researcher", str(row.get("Researcher", "") or "-")],
                ["Is Drawing", str(row.get("Is Drawing", "") or "False")],
                ["Draw CSV", str(row.get("Draw CSV", "") or "-")],
                ["Drawing Details", str(row.get("Drawing Details", "") or "-")],
            ]
            meta_tbl = Table(meta_rows, colWidths=[34 * mm, 140 * mm])
            meta_tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C4D6EA")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7FBFF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story += [meta_tbl, Spacer(1, 4)]
            rich_fields = []
            for label, field in [
                ("Purpose", "Purpose"),
                ("Methods", "Methods"),
                ("Observations", "Observations"),
                ("Results", "Results"),
                ("Markdown Notes", "Markdown Notes"),
            ]:
                value = _plain_text(row.get(field, ""))
                if value:
                    rich_fields.append((label, value))
            if rich_fields:
                for label, value in rich_fields:
                    story.append(Paragraph(f"<b>{_safe_para_text(label)}:</b>", body))
                    text_box = Table([[Paragraph(_safe_para_text(value), body)]], colWidths=[175 * mm])
                    text_box.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBFDFF")),
                        ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5E4F2")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    story += [text_box, Spacer(1, 4)]

            row_attachments = attachments[attachments["Experiment"].astype(str) == exp_title].copy() if not attachments.empty else pd.DataFrame()
            if not row_attachments.empty:
                story.append(Spacer(1, 2))
                story.append(Paragraph("Files attached to this experiment", small))
                for _, arow in row_attachments.iterrows():
                    line = f"• {str(arow.get('File', '') or '-')} [{str(arow.get('Type', '') or '-')}]"
                    cap = _plain_text(arow.get("Caption", ""))
                    if cap:
                        line += f" - {cap}"
                    story.append(Paragraph(_safe_para_text(line), body))
                story.append(Spacer(1, 4))

            row_images = [img for img in images if img["experiment"] == exp_title]
            if row_images:
                story.append(Paragraph("Photos / result images", small))
                for img in row_images[:3]:
                    try:
                        ir = ImageReader(img["path"])
                        iw, ih = ir.getSize()
                        max_w = 175 * mm
                        max_h = 105 * mm
                        scale = min(max_w / float(iw), max_h / float(ih))
                        story.append(Paragraph(_safe_para_text(img["caption"]), small))
                        story.append(Image(img["path"], width=iw * scale, height=ih * scale))
                        story.append(Spacer(1, 6))
                    except Exception:
                        continue
            story.append(Spacer(1, 6))

    story.append(Paragraph("Experiment Updates", h2))
    if updates.empty:
        story.append(Paragraph("No updates found for this project.", body))
        story.append(Spacer(1, 8))
    else:
        upd_show = updates[[c for c in ["Experiment Title", "Update Date", "Researcher", "Update Notes"] if c in updates.columns]].copy()
        story.append(_styled_table(_df_for_pdf_table(upd_show, limit=200), header_font_size=7, body_font_size=7))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Attachments + Files", h2))
    if attachments.empty:
        story.append(Paragraph("No attachments/files found for this project.", body))
        story.append(Spacer(1, 8))
    else:
        grouped = {}
        for _, row in attachments.iterrows():
            grouped.setdefault(str(row.get("Experiment", "") or "-"), []).append(row.to_dict())
        for exp_name, rows in grouped.items():
            story.append(Paragraph(_safe_para_text(exp_name), h3))
            for row in rows:
                cap = _plain_text(row.get("Caption", ""))
                line = f"• {str(row.get('File', '') or '-')} [{str(row.get('Type', '') or '-')}]"
                if cap:
                    line += f" - {cap}"
                story.append(Paragraph(_safe_para_text(line), body))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))

    doc.build(story)


def _build_development_project_pdf_pil(out_pdf: str, project_name: str, data_dir: str) -> None:
    from PIL import Image as PILImage, ImageDraw, ImageFont

    bundle = _prepare_development_project_bundle(data_dir, project_name)
    project = bundle["project"]
    experiments = bundle["experiments"]
    updates = bundle["updates"]
    attachments = bundle["attachments"]
    images = bundle["images"]

    PAGE_W, PAGE_H = 1654, 2339
    MARGIN = 90
    CONTENT_W = PAGE_W - (2 * MARGIN)
    BG = (248, 251, 255)
    PANEL = (255, 255, 255)
    BLUE = (17, 78, 134)
    GREEN = (21, 98, 58)
    TEXT = (20, 34, 52)
    MUTED = (92, 113, 136)
    BORDER = (185, 212, 235)

    font_regular_path = _pick_pdf_unicode_font() or "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if not os.path.exists(font_bold_path):
        font_bold_path = font_regular_path

    f_title = ImageFont.truetype(font_bold_path, 42)
    f_h1 = ImageFont.truetype(font_bold_path, 30)
    f_h2 = ImageFont.truetype(font_bold_path, 24)
    f_body = ImageFont.truetype(font_regular_path, 20)
    f_small = ImageFont.truetype(font_regular_path, 16)

    def rtl_fix(text: str) -> str:
        s = str(text or "")
        return s[::-1] if _looks_hebrew(s) else s

    def measure(font, text: str) -> int:
        box = font.getbbox(text or " ")
        return box[2] - box[0]

    def wrap_text(text: str, font, max_width: int) -> list[str]:
        raw = _plain_text(text)
        if not raw:
            return []
        lines = []
        for para in raw.split("\n"):
            para = para.strip()
            if not para:
                lines.append("")
                continue
            words = para.split()
            cur = ""
            for word in words:
                trial = word if not cur else f"{cur} {word}"
                if measure(font, rtl_fix(trial)) <= max_width:
                    cur = trial
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
        return lines

    pages = []
    page = PILImage.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(page)
    y = MARGIN

    def new_page():
        nonlocal page, draw, y
        pages.append(page)
        page = PILImage.new("RGB", (PAGE_W, PAGE_H), BG)
        draw = ImageDraw.Draw(page)
        y = MARGIN

    def ensure_space(height_needed: int):
        nonlocal y
        if y + height_needed > PAGE_H - MARGIN:
            new_page()

    def draw_text_block(text: str, font, fill, *, header=False, max_width=CONTENT_W):
        nonlocal y
        lines = wrap_text(text, font, max_width)
        if not lines:
            return
        line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 8
        ensure_space(len(lines) * line_h + 12)
        for line in lines:
            shown = rtl_fix(line)
            if _looks_hebrew(line):
                w = measure(font, shown)
                draw.text((PAGE_W - MARGIN - w, y), shown, font=font, fill=fill)
            else:
                draw.text((MARGIN, y), shown, font=font, fill=fill)
            y += line_h
        y += 6 if header else 2

    def draw_panel(title: str, rows: list[tuple[str, str]]):
        nonlocal y
        row_h = 34
        height = 56 + (len(rows) * row_h)
        ensure_space(height + 10)
        draw.rounded_rectangle((MARGIN, y, PAGE_W - MARGIN, y + height), radius=18, fill=PANEL, outline=BORDER, width=2)
        draw.text((MARGIN + 18, y + 14), title, font=f_h2, fill=BLUE)
        yy = y + 54
        for k, v in rows:
            draw.text((MARGIN + 20, yy), f"{k}:", font=f_small, fill=MUTED)
            vv = rtl_fix(v)
            if _looks_hebrew(v):
                w = measure(f_body, vv)
                draw.text((PAGE_W - MARGIN - 20 - w, yy), vv, font=f_body, fill=TEXT)
            else:
                draw.text((MARGIN + 240, yy), vv, font=f_body, fill=TEXT)
            yy += row_h
        y += height + 16

    def draw_bullets(title: str, items: list[str]):
        nonlocal y
        if not items:
            return
        draw_text_block(title, f_h2, BLUE, header=True)
        for item in items:
            line = f"• {item}"
            draw_text_block(line, f_body, TEXT, max_width=CONTENT_W - 20)

    def draw_image(path: str, caption: str):
        nonlocal y
        if not os.path.exists(path):
            return
        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            return
        max_w = CONTENT_W
        max_h = 620
        scale = min(max_w / img.width, max_h / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        ensure_space(nh + 70)
        img = img.resize((nw, nh))
        draw_text_block(caption, f_small, MUTED, max_width=CONTENT_W)
        page.paste(img, (MARGIN, y))
        y += nh + 18

    draw_text_block(f"Development Process Report", f_title, BLUE, header=True)
    draw_text_block(project_name, f_h1, GREEN, header=True)
    draw_text_block(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f_small, MUTED)
    draw_panel(
        "Project Summary",
        [
            ("Project Name", str(project.get("Project Name", project_name) or project_name)),
            ("Project Purpose", str(project.get("Project Purpose", "") or "-")),
            ("Target", str(project.get("Target", "") or "-")),
            ("Created At", str(project.get("Created At", "") or "-")),
            ("Experiments", str(int(len(experiments)))),
            ("Updates", str(int(len(updates)))),
        ],
    )

    if not experiments.empty:
        for idx, row in experiments.iterrows():
            if idx > 0:
                new_page()
            exp_title = str(row.get("Experiment Title", "") or "-")
            draw_text_block("Experiment", f_h2, BLUE, header=True)
            draw_text_block(exp_title, f_h1, GREEN, header=True)
            draw_panel(
                "Experiment Meta",
                [
                    ("Date", str(row.get("Date", "") or "-")),
                    ("Researcher", str(row.get("Researcher", "") or "-")),
                    ("Is Drawing", str(row.get("Is Drawing", "") or "False")),
                    ("Draw CSV", str(row.get("Draw CSV", "") or "-")),
                ],
            )

            for label, field in [
                ("Purpose", "Purpose"),
                ("Methods", "Methods"),
                ("Observations", "Observations"),
                ("Results", "Results"),
                ("Markdown Notes", "Markdown Notes"),
            ]:
                value = _plain_text(row.get(field, ""))
                if value:
                    draw_text_block(label, f_h2, BLUE, header=True)
                    draw_text_block(value, f_body, TEXT)

            exp_name = str(row.get("Experiment Title", "") or "-")
            row_attach = attachments[attachments["Experiment"].astype(str) == exp_name].copy() if not attachments.empty else pd.DataFrame()
            if not row_attach.empty:
                items = []
                for _, arow in row_attach.iterrows():
                    cap = _plain_text(arow.get("Caption", ""))
                    item = f"{str(arow.get('File', '') or '-')} [{str(arow.get('Type', '') or '-')}]"
                    if cap:
                        item += f" - {cap}"
                    items.append(item)
                draw_bullets("Files", items)

            row_images = [img for img in images if img["experiment"] == exp_name]
            for img in row_images[:2]:
                draw_image(img["path"], img["caption"] or os.path.basename(img["path"]))

    if not updates.empty:
        new_page()
        draw_text_block("Experiment Updates", f_h1, BLUE, header=True)
        for _, row in updates.iterrows():
            title = f"{str(row.get('Experiment Title', '') or '-') } | {str(row.get('Update Date', '') or '-')}"
            note = _plain_text(row.get("Update Notes", ""))
            draw_text_block(title, f_h2, GREEN, header=True)
            if str(row.get("Researcher", "")).strip():
                draw_text_block(f"Researcher: {row.get('Researcher')}", f_small, MUTED)
            if note:
                draw_text_block(note, f_body, TEXT)
            y += 10

    pages.append(page)
    first, rest = pages[0], pages[1:]
    first.save(out_pdf, "PDF", resolution=150.0, save_all=True, append_images=rest)


def _build_development_project_markdown(out_md: str, project_name: str, data_dir: str) -> None:
    def _md_path(path: str) -> str:
        return f"<{path}>"

    bundle = _prepare_development_project_bundle(data_dir, project_name)
    project = bundle["project"]
    experiments = bundle["experiments"]
    updates = bundle["updates"]
    attachments = bundle["attachments"]
    images = bundle["images"]

    lines: list[str] = []
    lines.append(f"# Development Process Report: {project_name}")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("> Structured project report from Development Process records, updates, notes, and attached media.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Project Summary")
    lines.append("")
    lines.append(f"- Project Name: `{project.get('Project Name', project_name) or project_name}`")
    lines.append(f"- Project Purpose: {_plain_text(project.get('Project Purpose', '-')) or '-'}")
    lines.append(f"- Target: {_plain_text(project.get('Target', '-')) or '-'}")
    lines.append(f"- Created At: `{_plain_text(project.get('Created At', '-')) or '-'}`")
    lines.append(f"- Archived: `{_plain_text(project.get('Archived', 'False')) or '-'}`")
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
                f"| {_plain_text(row.get('Experiment Title', '-')) or '-'} | `{_plain_text(row.get('Date', '-')) or '-'}` | {_plain_text(row.get('Researcher', '-')) or '-'} | `{_plain_text(row.get('Is Drawing', '-')) or '-'}` | {_plain_text(row.get('Draw CSV', '-')) or '-'} |"
            )
        lines.append("")

        for idx, row in experiments.iterrows():
            exp_title = _plain_text(row.get("Experiment Title", f"Experiment {idx + 1}")) or f"Experiment {idx + 1}"
            lines.append("---")
            lines.append("")
            lines.append(f"## Experiment {idx + 1}: {exp_title}")
            lines.append("")
            lines.append("> This section groups the experiment identity, notes, images, and supporting files in one place.")
            lines.append("")
            lines.append(f"- Date: `{_plain_text(row.get('Date', '-')) or '-'}`")
            lines.append(f"- Researcher: {_plain_text(row.get('Researcher', '-')) or '-'}")
            lines.append(f"- Is Drawing: `{_plain_text(row.get('Is Drawing', '-')) or '-'}`")
            lines.append(f"- Draw CSV: {_plain_text(row.get('Draw CSV', '-')) or '-'}")
            lines.append(f"- Drawing Details: {_plain_text(row.get('Drawing Details', '-')) or '-'}")
            lines.append("")
            for label, field in [
                ("Purpose", "Purpose"),
                ("Methods", "Methods"),
                ("Observations", "Observations"),
                ("Results", "Results"),
                ("Markdown Notes", "Markdown Notes"),
            ]:
                value = _plain_text(row.get(field, ""))
                if value and value != "-":
                    lines.append(f"### {label}")
                    lines.append("")
                    lines.append(value)
                    lines.append("")

            row_images = [img for img in images if img["experiment"] == exp_title]
            if row_images:
                lines.append("### Images")
                lines.append("")
                for img in row_images:
                    cap = _plain_text(img.get("caption", "")) or os.path.basename(img["path"])
                    lines.append(f"#### {cap}")
                    lines.append("")
                    lines.append(f"![{cap}]({_md_path(str(img['path']))})")
                    lines.append("")

            row_attachments = attachments[attachments["Experiment"].astype(str) == exp_title].copy() if not attachments.empty else pd.DataFrame()
            if not row_attachments.empty:
                lines.append("### Files")
                lines.append("")
                for _, arow in row_attachments.iterrows():
                    file_name = _plain_text(arow.get("File", "")) or "-"
                    cap = _plain_text(arow.get("Caption", ""))
                    label = f"{file_name} - {cap}" if cap and cap != "-" else file_name
                    abs_path = _resolve_report_media_path(data_dir, str(arow.get("Path", "")))
                    lines.append(f"- [{label}]({_md_path(str(abs_path))})")
                lines.append("")

    lines.append("## Experiment Updates")
    lines.append("")
    if updates.empty:
        lines.append("No updates found for this project.")
        lines.append("")
    else:
        updates = updates.sort_values("Update Date", kind="stable")
        for _, row in updates.iterrows():
            lines.append(f"### {_plain_text(row.get('Experiment Title', '-')) or '-'}")
            lines.append("")
            lines.append(f"- Update Date: `{_plain_text(row.get('Update Date', '-')) or '-'}`")
            lines.append(f"- Researcher: {_plain_text(row.get('Researcher', '-')) or '-'}")
            lines.append("")
            lines.append(_plain_text(row.get("Update Notes", "-")) or "-")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## End of Report")
    lines.append("")
    lines.append(f"- Project: `{project_name}`")
    lines.append(f"- Generated by: `Report Center / Development Process Markdown`")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _styled_table(table_data: list[list[str]], *, header_bg: str = "#EDF5FF", header_font_size: int = 8, body_font_size: int = 8):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from reportlab.platypus import Table, TableStyle

    if not table_data:
        table_data = [["No data"]]

    ncols = max(1, len(table_data[0]))
    usable_width = 510.0
    col_w = max(48.0, min(140.0, usable_width / float(ncols)))
    col_widths = [col_w] * ncols

    # Keep very long path-like or note-like cells inside page bounds.
    max_chars = 64
    if ncols >= 8:
        max_chars = 26
    elif ncols >= 6:
        max_chars = 34
    elif ncols >= 4:
        max_chars = 46

    h_style = ParagraphStyle("rc_tbl_h", fontSize=header_font_size, leading=header_font_size + 2)
    b_style = ParagraphStyle("rc_tbl_b", fontSize=body_font_size, leading=body_font_size + 2)

    wrapped = []
    for ridx, row in enumerate(table_data):
        out_row = []
        for c in row:
            s = "" if c is None else str(c)
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            if len(s) > max_chars:
                s = s[: max_chars - 1] + "…"
            s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            out_row.append(Paragraph(s, h_style if ridx == 0 else b_style))
        wrapped.append(out_row)

    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C0C0C0")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("FONTSIZE", (0, 0), (-1, 0), header_font_size),
                ("FONTSIZE", (0, 1), (-1, -1), body_font_size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
            ]
        )
    )
    return t


def _prepare_orders_period(orders_csv_path: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    df = _read_csv_safe(orders_csv_path)
    if df.empty:
        return df
    defaults = {
        "Status": "",
        "Timestamp": "",
        "Fiber Project": "",
        "Preform Number": "",
        "Good Zones Count (required length zones)": 0,
        "Required Length (m) (for T&M+costumer)": 0.0,
        "Done Description": "",
        "Failed Description": "",
        "Failed Reason": "",
        "Notes": "",
        "Done CSV": "",
        "Failed CSV": "",
        "Preform Length After Draw (cm)": 0.0,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    df["_ts"] = df["Timestamp"].apply(_parse_dt_robust)
    mask = df["_ts"].notna() & (df["_ts"] >= start_dt) & (df["_ts"] <= end_dt)
    out = df.loc[mask].copy()
    if out.empty:
        return out

    out["Required Length (m) (for T&M+costumer)"] = pd.to_numeric(
        out["Required Length (m) (for T&M+costumer)"], errors="coerce"
    ).fillna(0.0)
    out["Good Zones Count (required length zones)"] = pd.to_numeric(
        out["Good Zones Count (required length zones)"], errors="coerce"
    ).fillna(0).astype(int)
    out["Preform Length After Draw (cm)"] = pd.to_numeric(
        out["Preform Length After Draw (cm)"], errors="coerce"
    ).fillna(0.0)
    return out


def _prepare_schedule_windows(schedule_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sched = _read_csv_safe(schedule_csv_path)
    if sched.empty:
        return pd.DataFrame(), pd.DataFrame()
    now = pd.Timestamp.now()
    past = _expand_schedule_for_window(sched, (now - pd.Timedelta(days=7)).normalize(), now)
    nxt = _expand_schedule_for_window(sched, now, now + pd.Timedelta(days=7))
    keep = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
    if not past.empty:
        past = past[keep].sort_values("Start DateTime")
    if not nxt.empty:
        nxt = nxt[keep].sort_values("Start DateTime")
    return past, nxt


def _prepare_parts_orders(parts_orders_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_csv_safe(parts_orders_csv_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for c in [
        "Status",
        "Part Name",
        "Serial Number",
        "Project Name",
        "Details",
        "Opened By",
        "Approved",
        "Approved By",
        "Approval Date",
        "Ordered By",
        "Date Ordered",
        "Company",
    ]:
        if c not in df.columns:
            df[c] = ""

    df["Status"] = (
        df["Status"]
        .astype(str)
        .str.strip()
        .replace({"Needed": "Opened", "needed": "Opened"})
    )
    # Remove truly blank rows.
    keep_mask = (
        df["Status"].ne("")
        | df["Part Name"].astype(str).str.strip().ne("")
        | df["Serial Number"].astype(str).str.strip().ne("")
        | df["Project Name"].astype(str).str.strip().ne("")
        | df["Details"].astype(str).str.strip().ne("")
    )
    df = df.loc[keep_mask].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    status_order = ["Opened", "Approved", "Ordered", "Shipped", "Received", "Installed"]
    counts = (
        df.assign(_status=df["Status"].where(df["Status"].isin(status_order), "Opened"))
        .groupby("_status", as_index=False)
        .size()
        .rename(columns={"_status": "status", "size": "count"})
    )
    # Keep consistent order.
    counts["status"] = pd.Categorical(counts["status"], categories=status_order, ordered=True)
    counts = counts.sort_values("status")
    return counts, df


def _prepare_maintenance_tests(maintenance_dir: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    fp = os.path.join(maintenance_dir, "maintenance_test_records.csv")
    df = _read_csv_safe(fp)
    if df.empty:
        return df
    for c in [
        "test_ts",
        "task_id",
        "component",
        "task",
        "test_preset",
        "result_mode",
        "condition_met",
        "auto_threshold_met",
        "threshold_hits",
        "values_json",
        "condition_text",
        "action_text",
        "notes",
        "actor",
    ]:
        if c not in df.columns:
            df[c] = ""
    df["_ts"] = df["test_ts"].apply(_parse_dt_robust)
    mask = df["_ts"].notna() & (df["_ts"] >= start_dt) & (df["_ts"] <= end_dt)
    out = df.loc[mask].copy()
    if out.empty:
        return out
    out["values_summary"] = out["values_json"].astype(str).str.slice(0, 160)
    cols = [
        "test_ts",
        "component",
        "task_id",
        "test_preset",
        "result_mode",
        "condition_met",
        "auto_threshold_met",
        "threshold_hits",
        "values_summary",
        "notes",
        "actor",
    ]
    return out[[c for c in cols if c in out.columns]].copy()


def _latest_consumables_totals_snapshot() -> pd.DataFrame:
    try:
        with open(APP_PATHS.coating_stock_json, "r") as f:
            warehouse_stock = json.load(f)
            if not isinstance(warehouse_stock, dict):
                warehouse_stock = {}
    except Exception:
        warehouse_stock = {}

    container_df = _read_csv_safe(APP_PATHS.tower_containers_csv)
    container_last = container_df.tail(1).iloc[0].to_dict() if not container_df.empty else {}

    try:
        with open(APP_PATHS.coating_config_json, "r") as f:
            cfg = json.load(f)
            cfg_types = list((cfg.get("coatings") or {}).keys())
    except Exception:
        cfg_types = []

    container_totals: dict[str, float] = {}
    for lab in ["A", "B", "C", "D"]:
        ctype = str(container_last.get(f"{lab}_type", "") or "").strip()
        clevel = pd.to_numeric(container_last.get(f"{lab}_level_kg", 0.0), errors="coerce")
        clevel = float(0.0 if pd.isna(clevel) else clevel)
        if ctype:
            container_totals[ctype] = container_totals.get(ctype, 0.0) + clevel

    all_types = []
    for t in list(cfg_types) + list(warehouse_stock.keys()) + list(container_totals.keys()):
        t = str(t or "").strip()
        if t and t not in all_types:
            all_types.append(t)

    rows = []
    updated_at = str(container_last.get("updated_at", "") or "")
    for ctype in all_types:
        warehouse_kg = float(pd.to_numeric(warehouse_stock.get(ctype, 0.0), errors="coerce") or 0.0)
        containers_kg = float(pd.to_numeric(container_totals.get(ctype, 0.0), errors="coerce") or 0.0)
        rows.append(
            {
                "updated_at": updated_at,
                "coating_type": ctype,
                "warehouse_kg": warehouse_kg,
                "containers_kg": containers_kg,
                "total_kg": warehouse_kg + containers_kg,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["total_kg", "coating_type"], ascending=[False, True]).reset_index(drop=True)


def _build_custom_pdf(
    out_pdf: str,
    title: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    sections: list[str],
    orders_csv_path: str,
    parts_orders_csv_path: str,
    schedule_csv_path: str,
    preforms_csv_path: str,
    maintenance_dir: str,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    orders = _prepare_orders_period(orders_csv_path, start_dt, end_dt)
    gas = _build_gas_summary(start_dt, end_dt, dt_cap_s=2.0)
    sap = _build_sap_summary(start_dt, end_dt)
    maintenance = _build_maintenance_summary(start_dt, end_dt)
    faults = _build_fault_summary(start_dt, end_dt)
    containers = _latest_consumables_totals_snapshot()
    preforms = _read_csv_safe(preforms_csv_path)
    maint_tests = _prepare_maintenance_tests(maintenance_dir, start_dt, end_dt)
    past_sched, next_sched = _prepare_schedule_windows(schedule_csv_path)
    parts_counts, parts_rows = _prepare_parts_orders(parts_orders_csv_path)

    done = orders[orders["Status"].astype(str).str.strip().str.lower().eq("done")].copy() if not orders.empty else pd.DataFrame()
    failed = orders[orders["Status"].astype(str).str.strip().str.lower().eq("failed")].copy() if not orders.empty else pd.DataFrame()

    done_rows = pd.DataFrame()
    if not done.empty:
        done_rows = done[
            [
                "Timestamp",
                "Fiber Project",
                "Preform Number",
                "Good Zones Count (required length zones)",
                "Required Length (m) (for T&M+costumer)",
                "Done Description",
                "Notes",
                "Done CSV",
                "Preform Length After Draw (cm)",
            ]
        ].rename(
            columns={
                "Timestamp": "order_ts",
                "Fiber Project": "project",
                "Preform Number": "preform",
                "Good Zones Count (required length zones)": "zones_count",
                "Required Length (m) (for T&M+costumer)": "required_length_m",
                "Done Description": "done_description",
                "Notes": "order_notes",
                "Done CSV": "dataset_csv",
                "Preform Length After Draw (cm)": "preform_left_cm",
            }
        )

    failed_rows = pd.DataFrame()
    if not failed.empty:
        failed_rows = failed[
            [
                "Timestamp",
                "Fiber Project",
                "Preform Number",
                "Required Length (m) (for T&M+costumer)",
                "Failed Description",
                "Failed Reason",
                "Notes",
                "Failed CSV",
            ]
        ].rename(
            columns={
                "Timestamp": "order_ts",
                "Fiber Project": "project",
                "Preform Number": "preform",
                "Required Length (m) (for T&M+costumer)": "required_length_m",
                "Failed Description": "failed_description",
                "Failed Reason": "failed_reason",
                "Notes": "order_notes",
                "Failed CSV": "dataset_csv",
            }
        )

    # Preforms used only in selected period.
    used_preforms = pd.DataFrame()
    if not orders.empty:
        used = orders.copy()
        used["preform"] = used["Preform Number"].astype(str).str.strip()
        used = used[used["preform"] != ""].copy()
        if not used.empty:
            used_preforms = (
                used.groupby("preform", as_index=False)
                .agg(
                    project=("Fiber Project", lambda s: ", ".join(sorted({str(x).strip() for x in s if str(x).strip()}))[:80]),
                    orders_count=("preform", "count"),
                    total_required_m=("Required Length (m) (for T&M+costumer)", "sum"),
                    avg_zones=("Good Zones Count (required length zones)", "mean"),
                    avg_preform_left_cm=("Preform Length After Draw (cm)", "mean"),
                )
                .sort_values("orders_count", ascending=False)
            )
            used_preforms["total_required_m"] = pd.to_numeric(used_preforms["total_required_m"], errors="coerce").fillna(0.0)
            used_preforms["avg_zones"] = pd.to_numeric(used_preforms["avg_zones"], errors="coerce").fillna(0.0)
            used_preforms["avg_preform_left_cm"] = pd.to_numeric(used_preforms["avg_preform_left_cm"], errors="coerce").fillna(0.0)

            # Join inventory left-length only for used preforms.
            if not preforms.empty and "Preform Name" in preforms.columns:
                inv = preforms.copy()
                inv["preform"] = inv["Preform Name"].astype(str).str.strip()
                if "Length" in inv.columns:
                    inv["Length"] = pd.to_numeric(inv["Length"], errors="coerce").fillna(0.0)
                inv = inv[["preform", "Length"]].drop_duplicates(subset=["preform"], keep="last")
                used_preforms = used_preforms.merge(inv, on="preform", how="left")
                used_preforms = used_preforms.rename(columns={"Length": "inventory_length_left"})

    maint_next_week = pd.DataFrame()
    if not next_sched.empty:
        s = next_sched.copy()
        mmask = s["Event Type"].astype(str).str.lower().str.contains("maint") | s["Description"].astype(str).str.lower().str.contains("maint")
        maint_next_week = s.loc[mmask].copy()

    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("RC_H1", parent=styles["Heading1"], fontSize=17, leading=21, textColor=colors.HexColor("#0A3A66"))
    h2 = ParagraphStyle("RC_H2", parent=styles["Heading2"], fontSize=12.5, leading=15, textColor=colors.HexColor("#114E86"))
    body = ParagraphStyle("RC_BODY", parent=styles["BodyText"], fontSize=9.5, leading=12)

    story = [
        Paragraph(title, h1),
        Paragraph(f"Period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}", body),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body),
        Spacer(1, 8),
    ]

    if "Executive Summary" in sections:
        rows = [
            ["Done draws", str(int(len(done_rows)))],
            ["Failed draws", str(int(len(failed_rows)))],
            ["Gas used (SL)", f"{gas.total_sl:.2f}"],
            ["SAP used / left", f"{sap.events_count} / {sap.current_count:.0f}"],
            ["Maintenance done", str(maintenance.actions_count)],
            ["Faults opened / closed", f"{faults.faults_opened_count} / {faults.faults_closed_count}"],
        ]
        story.append(Paragraph("Executive Summary", h2))
        t = Table(rows, colWidths=[80 * mm, 44 * mm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A5C9E8")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ]))
        story += [t, Spacer(1, 8)]

    if "Resources: Gas + SAP + Preforms" in sections:
        story.append(Paragraph("Resources: Gas + SAP + Preforms", h2))
        resources = [
            ["Argon used in period (SL)", f"{gas.total_sl:.2f}"],
            ["Gas weighted avg flow (SLPM)", f"{gas.avg_slpm_weighted:.2f}"],
            ["SAP sets used in period", str(sap.events_count)],
            ["SAP sets left now", f"{sap.current_count:.0f}"],
            ["Preforms used (done draws)", str(int(len(done_rows)))],
            ["Unique preforms used", str(int(done_rows['preform'].astype(str).str.strip().replace('', pd.NA).dropna().nunique()) if not done_rows.empty else 0)],
            ["Avg preform left after draw (cm)", f"{float(done_rows['preform_left_cm'].mean() if not done_rows.empty else 0.0):.2f}"],
        ]
        r_tbl = Table(resources, colWidths=[80 * mm, 44 * mm])
        r_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A5C9E8")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ]))
        story += [r_tbl, Spacer(1, 5)]

        story.append(Paragraph("Preforms used in selected period (not full inventory)", body))
        p_cols = ["preform", "project", "orders_count", "total_required_m", "avg_zones", "avg_preform_left_cm", "inventory_length_left"]
        p_show = used_preforms[[c for c in p_cols if c in used_preforms.columns]] if not used_preforms.empty else pd.DataFrame()
        p_tbl = _styled_table(_df_for_pdf_table(p_show, limit=24))
        story += [p_tbl, Spacer(1, 8)]

    if "Draw Outcomes (Done/Failed + Notes)" in sections:
        story.append(Paragraph("Draw Outcomes (Done/Failed + Notes)", h2))
        story.append(Paragraph("Done draws: project + zones + length + done description + order notes", body))
        d_tbl = _styled_table(_df_for_pdf_table(done_rows, limit=18))
        story += [d_tbl, Spacer(1, 6)]

        story.append(Paragraph("Failed draws: failed description/reason + order notes", body))
        f_tbl = _styled_table(_df_for_pdf_table(failed_rows, limit=18), header_bg="#FFEDEE")
        story += [f_tbl, Spacer(1, 8)]

    if "Parts Orders Status" in sections:
        story.append(Paragraph("Parts Orders Status", h2))
        pc_tbl = _styled_table(_df_for_pdf_table(parts_counts, limit=16))
        story += [pc_tbl, Spacer(1, 5)]
        story.append(Paragraph("Current parts orders list", body))
        cols = [
            "Status",
            "Part Name",
            "Serial Number",
            "Project Name",
            "Details",
            "Opened By",
            "Approved",
            "Ordered By",
            "Date Ordered",
            "Company",
        ]
        p_show = parts_rows[[c for c in cols if c in parts_rows.columns]] if not parts_rows.empty else pd.DataFrame()
        pr_tbl = _styled_table(_df_for_pdf_table(p_show, limit=24))
        story += [pr_tbl, Spacer(1, 8)]

    if "Schedule: Past Week + Next Week" in sections:
        story.append(Paragraph("Schedule: Past Week + Next Week", h2))
        story.append(Paragraph("Past week", body))
        p_tbl = _styled_table(_df_for_pdf_table(past_sched, limit=18))
        story += [p_tbl, Spacer(1, 6)]

        story.append(Paragraph("Next week", body))
        n_tbl = _styled_table(_df_for_pdf_table(next_sched, limit=18))
        story += [n_tbl, Spacer(1, 8)]

    if "Maintenance + Faults" in sections:
        story.append(Paragraph("Maintenance + Faults", h2))
        mkpi = [
            ["Maintenance done (period)", str(maintenance.actions_count)],
            ["Maintenance planned next week", str(int(len(maint_next_week)))],
            ["Faults opened / closed", f"{faults.faults_opened_count} / {faults.faults_closed_count}"],
            ["Open critical faults now", str(faults.open_critical_now_count)],
        ]
        mk_tbl = Table(mkpi, colWidths=[80 * mm, 44 * mm])
        mk_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A5C9E8")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ]))
        story += [mk_tbl, Spacer(1, 5)]

        story.append(Paragraph("Maintenance actions done", body))
        ma_tbl = _styled_table(_df_for_pdf_table(maintenance.actions_rows, limit=18))
        story += [ma_tbl, Spacer(1, 5)]

        story.append(Paragraph("Maintenance planned next week (from schedule)", body))
        mn_tbl = _styled_table(_df_for_pdf_table(maint_next_week, limit=18))
        story += [mn_tbl, Spacer(1, 5)]

        story.append(Paragraph("Fault events", body))
        fe_tbl = _styled_table(_df_for_pdf_table(faults.faults_rows, limit=18), header_bg="#FFEDEE")
        story += [fe_tbl, Spacer(1, 5)]

        story.append(Paragraph("Fault actions", body))
        fa_tbl = _styled_table(_df_for_pdf_table(faults.fault_actions_rows, limit=18))
        story += [fa_tbl, Spacer(1, 8)]

    if "Maintenance Tests + Measurements" in sections:
        story.append(Paragraph("Maintenance Tests + Measurements", h2))
        summary_rows = [
            ["Saved test records", str(int(len(maint_tests)))],
            ["Condition met", str(int(maint_tests["condition_met"].astype(str).str.strip().str.lower().eq("yes").sum()) if not maint_tests.empty else 0)],
            ["Auto threshold hit", str(int(maint_tests["auto_threshold_met"].astype(str).str.strip().str.lower().eq("yes").sum()) if not maint_tests.empty else 0)],
        ]
        mt_tbl = Table(summary_rows, colWidths=[80 * mm, 44 * mm])
        mt_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A5C9E8")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ]))
        story += [mt_tbl, Spacer(1, 5)]
        story.append(Paragraph("Recorded maintenance measurements/tests in selected period", body))
        mt_rows_tbl = _styled_table(_df_for_pdf_table(maint_tests, limit=24))
        story += [mt_rows_tbl, Spacer(1, 8)]

    if "Consumables Snapshot" in sections:
        story.append(Paragraph("Consumables Snapshot", h2))
        story.append(Paragraph("Per coating type: warehouse + containers total.", body))
        c_tbl = _styled_table(_df_for_pdf_table(containers, limit=24))
        story += [c_tbl, Spacer(1, 8)]

    doc.build(story)


def render_report_center_tab(P) -> None:
    st.markdown(
        """
        <style>
          .report-center-hero{
            padding: 16px 18px 14px 18px;
            border-radius: 18px;
            border: 1px solid rgba(110,196,255,0.28);
            background:
              linear-gradient(135deg, rgba(9,27,52,0.94), rgba(8,18,36,0.86)),
              radial-gradient(circle at top right, rgba(56,138,219,0.20), transparent 34%);
            box-shadow: 0 18px 32px rgba(4,16,34,0.22);
            margin-bottom: 12px;
          }
          .report-center-hero h2{
            margin: 0 0 6px 0;
            color: rgba(244,250,255,0.98);
            font-size: 1.9rem;
            line-height: 1.05;
          }
          .report-center-hero p{
            margin: 0;
            color: rgba(195,220,241,0.92);
            font-size: 0.98rem;
          }
          .report-center-note{
            margin: 10px 0 18px 0;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(90,176,255,0.22);
            background: linear-gradient(180deg, rgba(9,27,52,0.58), rgba(5,16,31,0.54));
            color: rgba(198,229,250,0.92);
          }
          .report-center-subtle{
            color: rgba(168,202,230,0.82);
            font-size: 0.93rem;
            margin-top: 4px;
          }
          div[data-testid="stButton"] > button{
            border-radius: 12px !important;
            border: 1px solid rgba(138,214,255,0.58) !important;
            background: linear-gradient(180deg, rgba(28,74,120,0.72), rgba(12,36,68,0.66)) !important;
            color: rgba(236,248,255,0.98) !important;
            box-shadow: 0 8px 18px rgba(8,30,58,0.32), 0 0 12px rgba(74,170,255,0.18) !important;
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease !important;
          }
          div[data-testid="stButton"] > button:hover{
            transform: translateY(-1px);
            border-color: rgba(188,238,255,0.86) !important;
            box-shadow: 0 12px 24px rgba(8,30,58,0.36), 0 0 16px rgba(96,194,255,0.30) !important;
          }
          div[data-testid="stButton"] > button[kind="primary"]{
            border-color: rgba(170,232,255,0.84) !important;
            background: linear-gradient(180deg, rgba(76,168,255,0.90), rgba(32,98,172,0.88)) !important;
            box-shadow: 0 14px 24px rgba(12, 68, 124, 0.40), 0 0 18px rgba(96,194,255,0.34) !important;
          }
          div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
          div[data-testid="stMultiSelect"] span[data-baseweb="tag"]{
            background: linear-gradient(180deg, rgba(70,160,238,0.94), rgba(32,96,168,0.92)) !important;
            border: 1px solid rgba(170,232,255,0.82) !important;
            color: rgba(244,252,255,0.99) !important;
            box-shadow: 0 0 0 1px rgba(108,198,255,0.24), 0 4px 10px rgba(10,46,84,0.30) !important;
            max-width: none !important;
            width: auto !important;
            height: auto !important;
          }
          div[data-testid="stMultiSelect"] div[data-baseweb="tag"] *,
          div[data-testid="stMultiSelect"] span[data-baseweb="tag"] *{
            color: rgba(244,252,255,0.99) !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
          }
          div[data-testid="stMultiSelect"] div[data-baseweb="tag"] svg,
          div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg{
            fill: rgba(238,250,255,0.98) !important;
          }
          div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div{
            min-height: 52px !important;
            height: auto !important;
          }
          div[role="radiogroup"]{
            gap: 10px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="report-center-hero">
          <h2>🗂️ Report Center</h2>
          <p>Build clean handover reports for operations and structured markdown reports for development projects.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Report mode",
        ["Operations Report", "Development Process", "Recent Exports"],
        key="report_center_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="report-center-note">
          <strong>Current workspace:</strong> {mode}
          <div class="report-center-subtle">
            {"Operations handover PDF with schedule, maintenance, consumables, and outcomes." if mode == "Operations Report" else
             "Markdown-first development report with experiments, updates, notes, files, and inline photos." if mode == "Development Process" else
             "Quick access to the latest generated report files from this tab."}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dev_projects_fp = os.path.join(P.data_dir, "development_projects.csv")
    dev_projects = _read_csv_safe(dev_projects_fp)
    project_options = []
    if not dev_projects.empty and "Project Name" in dev_projects.columns:
        project_options = sorted([str(x).strip() for x in dev_projects["Project Name"].dropna().tolist() if str(x).strip()])
    ensure_report_center_dir()

    if mode == "Development Process":
        st.markdown("### 🧪 Development Process Report")
        st.caption("Choose one project, preview its content, then export a clean markdown report with inline images.")

        d1, d2 = st.columns([2, 1])
        selected_dev_project = d1.selectbox(
            "Choose development project",
            [""] + project_options,
            key="report_center_dev_project",
            format_func=lambda x: "Select project..." if x == "" else x,
        )
        if selected_dev_project:
            suggested_name = _next_available_report_filename(
                f"{selected_dev_project} - Development Report",
                ".md",
                P.report_center_dir,
            )
        else:
            suggested_name = _next_available_report_filename(
                "Development Report",
                ".md",
                P.report_center_dir,
            )
        current_name = st.session_state.get("report_center_dev_pdf_name", "").strip()
        if (
            not current_name
            or current_name.startswith("development_process_")
            or current_name == "Development Report.md"
            or current_name.endswith(" - Development Report.md")
            or re.match(r"^.+ - Development Report \(\d+\)\.md$", current_name)
        ):
            st.session_state["report_center_dev_pdf_name"] = suggested_name
        dev_pdf_name = d2.text_input(
            "Report filename",
            value=st.session_state.get("report_center_dev_pdf_name", suggested_name),
            key="report_center_dev_pdf_name",
        )

        if selected_dev_project:
            bundle = _prepare_development_project_bundle(P.data_dir, selected_dev_project)
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Experiments", int(len(bundle["experiments"])))
            dc2.metric("Updates", int(len(bundle["updates"])))
            dc3.metric("Media rows", int(len(bundle["attachments"])))

            with st.expander("Preview project content", expanded=False):
                proj = bundle["project"]
                if proj:
                    st.markdown(f"**Project Purpose:** {proj.get('Project Purpose', '-')}")
                    st.markdown(f"**Target:** {proj.get('Target', '-')}")
                if not bundle["experiments"].empty:
                    exp_prev = bundle["experiments"][[c for c in ["Experiment Title", "Date", "Researcher", "Purpose"] if c in bundle["experiments"].columns]]
                    st.dataframe(exp_prev, use_container_width=True, hide_index=True)
                if not bundle["attachments"].empty:
                    att_prev = bundle["attachments"][[c for c in ["Experiment", "File", "Type", "Caption"] if c in bundle["attachments"].columns]]
                    st.dataframe(att_prev, use_container_width=True, hide_index=True)

        if st.button("Generate Development Process Markdown", key="report_center_generate_dev_md", use_container_width=True):
            if not selected_dev_project:
                st.warning("Choose a development project first.")
                return
            base_name = (
                dev_pdf_name.strip()
                or _next_available_report_filename(
                    f"{selected_dev_project} - Development Report",
                    ".md",
                    P.report_center_dir,
                )
            )
            if not base_name.lower().endswith(".md"):
                base_name += ".md"
            out_md = report_center_path(base_name)
            if os.path.exists(out_md):
                stem, ext = os.path.splitext(base_name)
                base_name = _next_available_report_filename(stem, ext or ".md", P.report_center_dir)
            out_md = report_center_path(base_name)
            with st.spinner("Building development process markdown..."):
                try:
                    _build_development_project_markdown(
                        out_md=out_md,
                        project_name=selected_dev_project,
                        data_dir=P.data_dir,
                    )
                    st.success(f"Development markdown saved: {out_md}")
                except Exception as e:
                    st.error(f"Failed to build development markdown: {e}")

        st.markdown("---")
        st.markdown("**Recent Development Reports**")
        md_files = _glob_reports_cached(P.report_center_dir, "*.md", _mtime(P.report_center_dir))
        if not md_files:
            st.info("No development markdown reports yet.")
        else:
            for p in md_files[:12]:
                st.code(p)

    elif mode == "Operations Report":
        st.markdown("### 📊 Operations Report")
        st.caption("Build the operations handover PDF for the selected period, with schedule, maintenance, outcomes, resources, and tests.")

        if "report_center_start" not in st.session_state:
            st.session_state["report_center_start"] = (pd.Timestamp.now() - pd.Timedelta(days=7)).date()
        if "report_center_end" not in st.session_state:
            st.session_state["report_center_end"] = pd.Timestamp.now().date()

        p1, p2 = st.columns([1, 1])
        if p1.button("📆 Week Before + Week After", key="report_center_range_prev_next", use_container_width=True):
            st.session_state["report_center_start"] = (pd.Timestamp.now() - pd.Timedelta(days=7)).date()
            st.session_state["report_center_end"] = (pd.Timestamp.now() + pd.Timedelta(days=7)).date()
            st.rerun()
        if p2.button("📅 Last 7 Days", key="report_center_range_last7", use_container_width=True):
            st.session_state["report_center_start"] = (pd.Timestamp.now() - pd.Timedelta(days=7)).date()
            st.session_state["report_center_end"] = pd.Timestamp.now().date()
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", key="report_center_start")
        with c2:
            end_date = st.date_input("End date", key="report_center_end")

        title = st.text_input("Report title", value="Tower Operations Report", key="report_center_title")
        selected_sections = st.multiselect(
            "Choose sections",
            SECTIONS,
            default=SECTIONS,
            key="report_center_sections",
        )

        start_dt = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
        end_dt = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
        prev = _prepare_orders_period(P.orders_csv, start_dt, end_dt)
        done_count = int(prev["Status"].astype(str).str.strip().str.lower().eq("done").sum()) if not prev.empty else 0
        failed_count = int(prev["Status"].astype(str).str.strip().str.lower().eq("failed").sum()) if not prev.empty else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Orders in period", int(len(prev)))
        c2.metric("Done", done_count)
        c3.metric("Failed", failed_count)

        default_name = f"report_center_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        if st.button("Generate PDF Report", key="report_center_generate", use_container_width=True):
            if not selected_sections:
                st.warning("Choose at least one section.")
                return
            if end_dt < start_dt:
                st.error("End date must be after start date.")
                return

            out_pdf = report_center_path(default_name)
            with st.spinner("Building report..."):
                try:
                    _build_custom_pdf(
                        out_pdf=out_pdf,
                        title=title.strip() or "Tower Operations Report",
                        start_dt=start_dt,
                        end_dt=end_dt,
                        sections=selected_sections,
                        orders_csv_path=P.orders_csv,
                        parts_orders_csv_path=P.parts_orders_csv,
                        schedule_csv_path=P.schedule_csv,
                        preforms_csv_path=P.preform_inventory_csv,
                        maintenance_dir=P.maintenance_dir,
                    )
                    st.success(f"Report saved: {out_pdf}")
                except Exception as e:
                    st.error(f"Failed to build report: {e}")

        with st.expander("Quick preview for selected period", expanded=False):
            st.markdown("Use this as a fast sanity check before you export.")
            pca, pcb, pcc = st.columns(3)
            pca.metric("Orders in period", int(len(prev)))
            pcb.metric("Done", done_count)
            pcc.metric("Failed", failed_count)

        st.markdown("---")
        st.markdown("**Recent Operations PDFs**")
        files = _glob_reports_cached(P.report_center_dir, "*.pdf", _mtime(P.report_center_dir))
        if not files:
            st.info("No operations PDFs yet.")
        else:
            for p in files[:12]:
                st.code(p)

    else:
        st.markdown("### 🗃️ Recent Exports")
        st.caption("Fast access to the latest outputs generated from Report Center.")
        pdf_files = _glob_reports_cached(P.report_center_dir, "*.pdf", _mtime(P.report_center_dir))
        md_files = _glob_reports_cached(P.report_center_dir, "*.md", _mtime(P.report_center_dir))
        c1, c2 = st.columns(2)
        c1.metric("Operations PDFs", len(pdf_files))
        c2.metric("Development Markdown", len(md_files))

        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**PDF exports**")
            if not pdf_files:
                st.info("No PDF exports yet.")
            else:
                for p in pdf_files[:15]:
                    st.code(p)
        with l2:
            st.markdown("**Markdown exports**")
            if not md_files:
                st.info("No markdown exports yet.")
            else:
                for p in md_files[:15]:
                    st.code(p)
