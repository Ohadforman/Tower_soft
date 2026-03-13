from __future__ import annotations

from datetime import datetime
import os
import glob

import pandas as pd
import streamlit as st

from app_io.paths import ensure_report_center_dir, report_center_path
from scripts.cli.run_weekly_report import (
    _build_fault_summary,
    _build_gas_summary,
    _build_maintenance_summary,
    _build_sap_summary,
    _df_for_pdf_table,
    _expand_schedule_for_window,
    _latest_containers_snapshot,
    _parse_dt_robust,
    _read_csv_safe,
)

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
    containers = _latest_containers_snapshot()
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
        c_tbl = _styled_table(_df_for_pdf_table(containers, limit=1))
        story += [c_tbl, Spacer(1, 8)]

    doc.build(story)


def render_report_center_tab(P) -> None:
    st.markdown(
        """
        <style>
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="dash-title">🗂️ Report Center</div>', unsafe_allow_html=True)
    st.caption("Build custom PDF reports with operations-focused data for team handover.")

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

    ensure_report_center_dir()
    default_name = f"report_center_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    if st.button("Generate PDF Report", key="report_center_generate", use_container_width=True):
        if not selected_sections:
            st.warning("Choose at least one section.")
            return
        start_dt = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
        end_dt = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
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

    st.markdown("**Quick preview (selected period)**")
    prev = _prepare_orders_period(P.orders_csv, pd.Timestamp(datetime.combine(start_date, datetime.min.time())), pd.Timestamp(datetime.combine(end_date, datetime.max.time())))
    done_count = int(prev["Status"].astype(str).str.strip().str.lower().eq("done").sum()) if not prev.empty else 0
    failed_count = int(prev["Status"].astype(str).str.strip().str.lower().eq("failed").sum()) if not prev.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders in period", int(len(prev)))
    c2.metric("Done", done_count)
    c3.metric("Failed", failed_count)

    st.markdown("---")
    st.markdown("**Recent Report Center PDFs**")
    files = sorted(glob.glob(os.path.join(P.report_center_dir, "*.pdf")), key=os.path.getmtime, reverse=True)
    if not files:
        st.info("No reports yet. Generate your first custom PDF.")
    else:
        for p in files[:20]:
            st.code(p)
