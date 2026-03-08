import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_io.paths import P, dataset_csv_path
from helpers.params_io import get_float_param
from helpers.dataset_param_parsers import _parse_steps, _parse_zones_from_end, _parse_marked_zone_lengths


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, keep_default_na: bool, mtime: float):
    return pd.read_csv(path, keep_default_na=keep_default_na)


def render_home_draw_orders_overview(
        orders_file: str = P.orders_csv,
        title: str = "🚀 Draws Monitor",
        height: int = 360,
):
    # ---------- Title ----------
    st.markdown(
        """
        <style>
        .home-balloon {
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
            transform-origin: center center;
            will-change: transform;
        }
        .home-balloon:hover {
            transform: translateY(-4px) scale(1.012);
            box-shadow: 0 14px 30px rgba(0,0,0,0.35) !important;
        }
        .home-now-box.home-balloon:hover {
            transform: translateY(-4px) scale(1.008);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Title ----------
    st.markdown(
        f"""
        <div style="
            font-size: 1.5rem;
            font-weight: 700;
            color: rgba(255,255,255,0.95);
            margin-bottom: 0.6em;
            text-align: left;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Small card helper ----------
    def _card(title_txt, value, border_color, emoji=""):
        st.markdown(
            f"""
            <div style="
                width: 100%;
                min-height: 140px;
                background: rgba(0,0,0,0.52);
                border: 2px solid {border_color};
                border-radius: 18px;
                padding: 14px;
                text-align: center;
                box-shadow: 0 6px 18px rgba(0,0,0,0.25);
                backdrop-filter: blur(6px);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                gap: 6px;
            " class="home-balloon">
                <div style="font-size:18px;font-weight:800;color:white;">
                    {emoji} {title_txt}
                </div>
                <div style="font-size:44px;font-weight:900;color:white;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- NOW DRAWING BOX ----------
    def _now_box(text_left, text_right="", border_color="#6F87A6", emoji="🟣"):
        st.markdown(
            f"""
            <div style="
                width:100%;
                padding:18px 22px;
                border-radius:20px;
                border:2px solid {border_color};
                background:rgba(18,28,44,0.70);
                box-shadow:0 10px 30px rgba(0,0,0,0.35);
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:18px;
            " class="home-now-box home-balloon">
                <div style="font-size:22px;font-weight:900;color:white;">
                    {emoji} {text_left}
                </div>
                <div style="font-size:14px;color:#e6d9ff;">
                    {text_right}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not os.path.exists(orders_file):
        st.info("No orders submitted yet.")
        return

    # ---------- Load orders (safe) ----------
    try:
        df = _read_csv_cached(orders_file, keep_default_na=False, mtime=_mtime(orders_file))
    except Exception:
        st.info("No orders submitted yet.")
        return

    if df.empty and len(df.columns) == 0:
        st.info("No orders submitted yet.")
        return

    # Ensure columns
    for col, default in {
        "Status": "Pending",
        "Priority": "Normal",
        "Fiber Project": "",
        "Preform Number": "",
        "Timestamp": "",
        "Desired Date": "",
        "Length (m)": "",
        "Spools": "",
        "Notes": "",
        "Done CSV": "",
        "Done Description": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }.items():
        if col not in df.columns:
            df[col] = default

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Desired Date"] = pd.to_datetime(df["Desired Date"], errors="coerce").dt.date

    # Normalize text
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""}).fillna("").str.strip()

    # Normalize T&M flag
    df["T&M Moved"] = df["T&M Moved"].apply(
        lambda x: str(x).strip().lower() in ("true", "1", "yes", "y", "moved")
    )

    df_visible = df[~df["T&M Moved"]].copy()

    def _norm_draw_status(x: str) -> str:
        s = str(x).strip().lower()
        if s in ("pending", "open", "new", "opened"):
            return "Pending"
        if s in ("scheduled", "schedule", "planned"):
            return "Scheduled"
        if s in ("done", "completed", "complete", "finished"):
            return "Done"
        if s in ("failed", "fail", "error", "aborted"):
            return "Failed"
        if s in ("in progress", "in-progress", "progress", "in prograss", "drawing"):
            return "In Progress"
        return str(x).strip() if str(x).strip() else "Pending"

    df_visible["Status_Norm"] = df_visible["Status"].apply(_norm_draw_status)

    # ==========================================================
    # 🟣 NOW DRAWING (Status == In Progress)
    # ==========================================================
    df_prog = df_visible[df_visible["Status_Norm"].eq("In Progress")].copy()

    if not df_prog.empty:
        df_prog = df_prog.sort_values("Timestamp", ascending=False)
        row = df_prog.iloc[0]

        preform = row.get("Preform Number", "")
        fiber = row.get("Fiber Project", "")

        if preform:
            now_text = f"Now drawing: F{preform}"
        elif fiber:
            now_text = f"Now drawing: {fiber}"
        else:
            now_text = "Now drawing: In progress"

        n_ip = len(df_prog)
        ts = row.get("Timestamp")

        if pd.notna(ts):
            now_right = f"In progress: {n_ip} | {ts.strftime('%Y-%m-%d %H:%M')}"
        else:
            now_right = f"In progress: {n_ip}"

        # ✅ Only here
        _now_box(now_text, now_right)

    # ==========================================================
    # KPI COUNTS (visible only)
    # ==========================================================
    def _count(s):
        return int(df_visible["Status_Norm"].eq(s).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card("Pending", _count("Pending"), "#B98A52", "🟠")
    with c2:
        _card("Scheduled", _count("Scheduled"), "#5B83AA", "🗓️")
    with c3:
        _card("Done", _count("Done"), "#5E947B", "✅")
    with c4:
        _card("Failed", _count("Failed"), "#9A5A5A", "❌")


def render_schedule_home_minimal():
    import plotly.express as px
    st.subheader("📅 Schedule")

    SCHEDULE_FILE = P.schedule_csv
    required_columns = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    # Ensure file exists (works even if empty / no events)
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=required_columns).to_csv(SCHEDULE_FILE, index=False)

    schedule_df = _read_csv_cached(SCHEDULE_FILE, keep_default_na=False, mtime=_mtime(SCHEDULE_FILE))
    schedule_df.columns = schedule_df.columns.str.strip()

    # Ensure required columns exist (auto-fix)
    for col in required_columns:
        if col not in schedule_df.columns:
            schedule_df[col] = pd.Series(dtype="object")
    schedule_df = schedule_df[required_columns]

    # Parse datetimes safely
    schedule_df["Start DateTime"] = pd.to_datetime(schedule_df["Start DateTime"], errors="coerce")
    schedule_df["End DateTime"] = pd.to_datetime(schedule_df["End DateTime"], errors="coerce")

    # Clean Description/Recurrence strings
    schedule_df["Description"] = (
        schedule_df["Description"]
        .fillna("")
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Description=", "", regex=False)
        .str.strip()
    )

    schedule_df["Recurrence"] = (
        schedule_df["Recurrence"]
        .fillna("None")
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Recurrence=", "", regex=False)
        .str.strip()
    )
    schedule_df.loc[schedule_df["Recurrence"].eq(""), "Recurrence"] = "None"

    # -------------------------
    # Home view is fixed to weekly horizon (no range buttons here)
    # -------------------------
    today = pd.Timestamp.today().normalize()
    start_filter = today
    end_filter = today + pd.DateOffset(weeks=1)

    # Filter to valid rows first
    valid = schedule_df.dropna(subset=["Start DateTime", "End DateTime"]).copy()

    # -------------------------
    # Expand recurrences so Home shows ALL occurrences in the selected range
    # -------------------------
    def _expand_recurrences(df_in: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        if df_in.empty:
            return df_in

        out_rows = []
        max_instances = 2000  # safety cap

        for _, r in df_in.iterrows():
            st0 = r.get("Start DateTime")
            en0 = r.get("End DateTime")
            if pd.isna(st0) or pd.isna(en0):
                continue

            rec = str(r.get("Recurrence", "None")).strip()
            rec_low = rec.lower()

            duration = en0 - st0
            if pd.isna(duration) or duration <= pd.Timedelta(seconds=0):
                duration = pd.Timedelta(minutes=1)

            # Non-recurring
            if rec_low in ("", "none", "nan"):
                d0 = r.to_dict()
                d0["Recurrence"] = "None"
                out_rows.append(d0)
                continue

            def _add_step(ts: pd.Timestamp) -> pd.Timestamp:
                if "week" in rec_low:
                    return ts + pd.DateOffset(weeks=1)
                if "every 3 months" in rec_low or "3 months" in rec_low or "quarterly" in rec_low:
                    return ts + pd.DateOffset(months=3)
                if "every 6 months" in rec_low or "6 months" in rec_low or "semiannual" in rec_low or "semi-annually" in rec_low:
                    return ts + pd.DateOffset(months=6)
                if "month" in rec_low:
                    return ts + pd.DateOffset(months=1)
                if "year" in rec_low:
                    return ts + pd.DateOffset(years=1)
                return pd.NaT  # unknown

            # Unknown recurrence -> treat as non-recurring
            if pd.isna(_add_step(pd.Timestamp(st0))):
                d0 = r.to_dict()
                d0["Recurrence"] = "None"
                out_rows.append(d0)
                continue

            cur_start = pd.Timestamp(st0)
            cur_end = cur_start + duration

            # Fast-forward to near the window
            guard = 0
            while cur_end < window_start and guard < max_instances:
                nxt = _add_step(cur_start)
                if pd.isna(nxt):
                    break
                cur_start = nxt
                cur_end = cur_start + duration
                guard += 1

            # Generate occurrences that intersect the window
            gen = 0
            while cur_start <= window_end and gen < max_instances:
                cur_end = cur_start + duration
                if (cur_end >= window_start) and (cur_start <= window_end):
                    d = r.to_dict()
                    d["Start DateTime"] = cur_start
                    d["End DateTime"] = cur_end
                    d["Recurrence"] = rec if rec else "None"
                    out_rows.append(d)

                nxt = _add_step(cur_start)
                if pd.isna(nxt):
                    break
                cur_start = nxt
                gen += 1

        if not out_rows:
            return pd.DataFrame(columns=df_in.columns)

        out = pd.DataFrame(out_rows)
        for c in df_in.columns:
            if c not in out.columns:
                out[c] = ""
        out = out[df_in.columns]
        out["Start DateTime"] = pd.to_datetime(out["Start DateTime"], errors="coerce")
        out["End DateTime"] = pd.to_datetime(out["End DateTime"], errors="coerce")

        # Ensure Recurrence always has a nice value
        out["Recurrence"] = out["Recurrence"].fillna("None").astype(str).str.strip()
        out.loc[out["Recurrence"].isin(["", "none", "None", "nan", "NaN"]), "Recurrence"] = "None"
        return out

    win_start = pd.to_datetime(start_filter)
    win_end = pd.to_datetime(end_filter)

    expanded = _expand_recurrences(valid, win_start, win_end)

    # Now filter (overlap logic)
    filtered = expanded[
        (expanded["End DateTime"] >= win_start) &
        (expanded["Start DateTime"] <= win_end)
    ].copy()

    st.write("### Schedule Timeline")

    event_colors = {
        "Maintenance": "#4C78A8",
        "Drawing": "#5F9E89",
        "Stop": "#9A5A5A",
        "Management Event": "#7A6D8F",
    }

    if filtered.empty:
        st.info("No events in the selected range (or schedule is empty).")
        return

    # =========================================================
    # ✅ HOVER FIX (ONLY CHANGE YOU ASKED FOR)
    # px.timeline hover can't reliably format x_end, so we precompute strings
    # and force a clean hovertemplate (like the Schedule tab)
    # =========================================================
    filtered["StartStr"] = pd.to_datetime(filtered["Start DateTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    filtered["EndStr"] = pd.to_datetime(filtered["End DateTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    filtered["RecurrenceDisp"] = filtered["Recurrence"].fillna("None").astype(str).str.strip()
    filtered.loc[filtered["RecurrenceDisp"].isin(["", "none", "None", "nan", "NaN"]), "RecurrenceDisp"] = "None"

    fig = px.timeline(
        filtered,
        x_start="Start DateTime",
        x_end="End DateTime",
        y="Event Type",
        color="Event Type",
        color_discrete_map=event_colors,
        title="Tower Schedule",
        custom_data=["StartStr", "EndStr", "RecurrenceDisp", "Description"],
    )

    # Force clean hover (NO weird %{customdata[0]} text leaks, NO broken x_end formatting)
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Start: %{customdata[0]}<br>"
            "End: %{customdata[1]}<br>"
            "Recurrence: %{customdata[2]}<br>"
            "Description: %{customdata[3]}"
            "<extra></extra>"
        ),
        marker_line_color="rgba(224,236,248,0.28)",
        marker_line_width=1.0,
        opacity=0.86,
    )

    # Keep your layout exactly as you had it
    fig.update_layout(
        paper_bgcolor="rgba(15,15,20,0.92)",
        plot_bgcolor="rgba(15,15,20,0.70)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Event Type",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    st.plotly_chart(fig, use_container_width=True)


def render_parts_orders_home_all():
    st.subheader("🧩 Parts Orders")

    ORDER_FILE = P.parts_orders_csv

    # ✅ NEW canonical statuses (Needed -> Opened)
    status_order = ["Opened", "Approved", "Ordered", "Shipped", "Received", "Installed"]

    # ✅ display columns (remove Purpose, keep single Details)
    column_order = [
        "Status",
        "Part Name",
        "Serial Number",
        "Project Name",
        "Details",
        "Approved",
        "Approved By",
        "Approval Date",
        "Ordered By",
        "Date Ordered",
        "Company",
    ]

    # ---------------- Load / Safety ----------------
    if not os.path.exists(ORDER_FILE):
        st.info("No orders file yet (part_orders.csv).")
        return

    orders_df = _read_csv_cached(ORDER_FILE, keep_default_na=False, mtime=_mtime(ORDER_FILE))
    orders_df.columns = orders_df.columns.str.strip()

    # Ensure required columns exist
    for col in column_order:
        if col not in orders_df.columns:
            orders_df[col] = ""

    orders_df = orders_df[column_order].copy().fillna("")
    orders_df["Status"] = orders_df["Status"].astype(str).str.strip()

    # ✅ Backward-compat: map old "Needed" to "Opened"
    orders_df["Status"] = orders_df["Status"].replace({"Needed": "Opened", "needed": "Opened"})

    if orders_df.empty:
        st.warning("No orders have been placed yet.")
        return

    # ---------------- Counts ----------------
    status_lower = orders_df["Status"].astype(str).str.lower()
    opened_count = int((status_lower == "opened").sum())
    approved_count = int((status_lower == "approved").sum())
    ordered_count = int((status_lower == "ordered").sum())
    received_count = int((status_lower == "received").sum())

    # ---------------- KPI Cards + GLASS TABLE CSS ----------------
    st.markdown(
        """
        <style>
        .parts-home-shell{
            border: 1px solid rgba(150, 220, 255, 0.20);
            border-radius: 16px;
            padding: 10px 12px 12px 12px;
            background: linear-gradient(160deg, rgba(6, 14, 26, 0.58), rgba(6, 12, 22, 0.42));
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.28);
        }
        /* ================= KPI CARDS ================= */
        .kpi-card{
            background: rgba(0,0,0,0.52);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px;
            padding: 14px 16px;
            text-align: center;
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-title{
            font-size: 16px;
            font-weight: 800;
            color: rgba(255,255,255,0.92);
        }
        .kpi-value{
            font-size: 34px;
            font-weight: 900;
            color: rgba(255,255,255,0.98);
            margin-top: 6px;
            line-height: 1;
        }
        .kpi-opened{
            border: 2px solid rgba(255, 80, 80, 0.95) !important;
            box-shadow: 0 0 18px rgba(255, 80, 80, 0.85);
            background: rgba(255, 80, 80, 0.22);
        }
        .kpi-received{
            border: 2px solid rgba(80, 255, 120, 0.95) !important;
            box-shadow: 0 0 18px rgba(80, 255, 120, 0.85);
            background: rgba(80, 255, 120, 0.22);
        }

        /* ================= FULL GLASS TABLE (AG-GRID) ================= */

        div[data-testid="stDataFrame"]{
            background: transparent !important;
        }

        div[data-testid="stDataFrame"] > div{
            background: rgba(0,0,0,0.28) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 18px !important;
            padding: 10px !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25) !important;
        }

        .ag-root-wrapper, .ag-root, .ag-body-viewport, .ag-center-cols-viewport,
        .ag-center-cols-container, .ag-floating-top, .ag-floating-bottom,
        .ag-pinned-left-cols-container, .ag-pinned-right-cols-container,
        .ag-row, .ag-row-odd, .ag-row-even{
            background: transparent !important;
        }

        .ag-header{
            background: rgba(0,0,0,0.30) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border-bottom: 1px solid rgba(255,255,255,0.10) !important;
        }
        .ag-header-cell, .ag-header-group-cell{
            background: transparent !important;
            color: rgba(255,255,255,0.85) !important;
            font-weight: 800 !important;
            border-right: 1px solid rgba(255,255,255,0.07) !important;
        }

        .ag-cell{
            background: rgba(0,0,0,0.14) !important;
            color: rgba(255,255,255,0.92) !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
            border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        }

        .ag-row-hover .ag-cell{
            background: rgba(255,255,255,0.06) !important;
        }

        .ag-body-viewport{
            background: rgba(0,0,0,0.14) !important;
        }

        ::-webkit-scrollbar { height: 10px; width: 10px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- KPI Cards (symmetric) ----------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    opened_class = "kpi-card kpi-opened" if opened_count > 0 else "kpi-card"
    with c1:
        st.markdown(
            f"""
            <div class="{opened_class}">
                <div class="kpi-title">🔴 Opened</div>
                <div class="kpi-value">{opened_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🟢 Approved</div>
                <div class="kpi-value">{approved_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🟡 Ordered</div>
                <div class="kpi-value">{ordered_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    received_class = "kpi-card kpi-received" if received_count > 0 else "kpi-card"
    with c4:
        st.markdown(
            f"""
            <div class="{received_class}">
                <div class="kpi-title">✅ Received</div>
                <div class="kpi-value">{received_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- Opened-only compact cards ----------------
    opened_only = orders_df[
        orders_df["Status"].astype(str).str.strip().str.lower().eq("opened")
    ].copy()

    st.markdown(
        """
        <style>
        .parts-opened-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: rgba(246,252,255,0.98);
            text-shadow: 0 1px 10px rgba(0,0,0,0.45);
            margin: 12px 0 8px 0;
        }
        .parts-opened-card {
            border: 1px solid rgba(255, 120, 120, 0.42);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(76, 22, 22, 0.62);
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }
        .parts-opened-card:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 10px 20px rgba(0,0,0,0.24);
            border-color: rgba(255, 146, 146, 0.70);
        }
        .parts-opened-main {
            font-size: 1.0rem;
            font-weight: 800;
            color: rgba(255,244,244,0.99);
            text-shadow: 0 1px 8px rgba(0,0,0,0.42);
            margin-bottom: 6px;
        }
        .parts-opened-sub {
            font-size: 0.86rem;
            color: rgba(245,245,245,0.95);
            text-shadow: 0 1px 8px rgba(0,0,0,0.40);
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='parts-home-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='parts-opened-title'>🔴 Opened Parts</div>", unsafe_allow_html=True)

    if opened_only.empty:
        st.info("No parts currently in Opened status.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    opened_only["Date Ordered"] = pd.to_datetime(opened_only["Date Ordered"], errors="coerce")
    opened_only = opened_only.sort_values("Date Ordered", ascending=False, na_position="last")

    for _, r in opened_only.iterrows():
        part_name = str(r.get("Part Name", "")).strip() or "—"
        details = str(r.get("Details", "")).strip() or "No details"
        ordered_by = str(r.get("Ordered By", "")).strip() or "—"
        company = str(r.get("Company", "")).strip() or "—"
        dt_val = r.get("Date Ordered")
        dt_txt = dt_val.strftime("%Y-%m-%d") if pd.notna(dt_val) else "—"

        st.markdown(
            f"""
            <div class="parts-opened-card">
                <div class="parts-opened-main">{part_name}</div>
                <div class="parts-opened-sub">{details}</div>
                <div class="parts-opened-sub" style="margin-top:6px;">
                    Ordered by: <b>{ordered_by}</b> &nbsp;|&nbsp; Date: <b>{dt_txt}</b> &nbsp;|&nbsp; Company: <b>{company}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_tm_drum_fiber_visual_from_csv(df_params: pd.DataFrame, dataset_name: str):
    """
    Draws drum + fiber + zones/segments.
    Uses ONLY what exists in the dataset CSV (no log needed).
    Priority:
      1) Zone i Start/End (from end)  -> shows ALL zones accurately
      2) Marked Zone i Length         -> shows ALL zones sequentially
      3) STEP i Action/Length         -> fallback merged plan
    """
    total_km = get_float_param(df_params, "Fiber Total Length (Log End)", 0.0)
    total_save = get_float_param(df_params, "Total Saved Length", 0.0)
    total_cut = get_float_param(df_params, "Total Cut Length", 0.0)

    # Try to get explicit zone positions
    zones = _parse_zones_from_end(df_params)

    # If no positions, try to get marked zone lengths and place sequentially
    if not zones:
        marked_lengths = _parse_marked_zone_lengths(df_params)
        if marked_lengths:
            a = 0.0
            zones = []
            for i, L in enumerate(marked_lengths, start=1):
                zones.append({"i": i, "a": a, "b": a + L, "len": L})
                a += L

    # If still none, fallback to steps (SAVE segments)
    if not zones:
        steps = _parse_steps(df_params)
        if steps:
            a = 0.0
            zones = []
            zi = 1
            for action, L in steps:
                if action == "SAVE":
                    zones.append({"i": zi, "a": a, "b": a + L, "len": L})
                    zi += 1
                a += L

    if not zones:
        st.info(
            "No zone information found in dataset CSV (no Zone-from-end, no Marked Zone Lengths, no STEP SAVE segments).")
        return

    # If total length missing, infer from max(b)
    if total_km <= 0:
        total_km = float(max(z["b"] for z in zones))

    # ---- draw
    fig = go.Figure()

    # Drum
    fig.add_shape(
        type="circle",
        xref="paper", yref="paper",
        x0=0.02, y0=0.35, x1=0.18, y1=0.65,
        line=dict(width=3),
        fillcolor="rgba(255,255,255,0.06)",
    )

    # Fiber baseline
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.18, y0=0.50, x1=0.98, y1=0.50,
        line=dict(width=6),
    )

    # Zones (green blocks)
    for z in sorted(zones, key=lambda r: r["a"]):
        x0 = max(0.0, min(1.0, z["a"] / total_km))
        x1 = max(0.0, min(1.0, z["b"] / total_km))
        x0p = 0.18 + 0.80 * x0
        x1p = 0.18 + 0.80 * x1

        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x0p, x1=x1p,
            y0=0.46, y1=0.54,
            fillcolor="rgba(0,180,0,0.40)",
            line=dict(width=1),
        )

        # label if visible enough
        if (x1 - x0) > 0.05:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5 * (x0p + x1p),
                y=0.58,
                text=f"Zone {z['i']}  {z['len']:.3f} km",
                showarrow=False
            )

        fig.add_trace(go.Scatter(
            x=[0.5 * (x0p + x1p)],
            y=[0.50],
            mode="markers",
            marker=dict(size=18, opacity=0),
            hovertemplate=(
                f"<b>Zone {z['i']}</b><br>"
                f"From end: {z['a']:.6f} → {z['b']:.6f} km<br>"
                f"Length: {z['len']:.6f} km"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    # Header + metrics inside popup
    st.markdown("### 🧵 Drum & Fiber – Good Zones Map")
    st.caption(f"Dataset: **{dataset_name}**  |  0 km = fiber end")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total length (km)", f"{total_km:.6f}")
    c2.metric("Total SAVE (km)", f"{total_save:.6f}" if total_save else "—")
    c3.metric("Total CUT (km)", f"{total_cut:.6f}" if total_cut else "—")

    st.plotly_chart(fig, use_container_width=True)


def render_done_home_section(show_header: bool = True):
    import os
    import pandas as pd
    import streamlit as st
    import datetime as dt

    if show_header:
        st.subheader("✅ DONE – Recent Draws (last 4 days)")
        st.caption("Summarizes finished draws from the dataset CSV. After 4 days, they auto-move to T&M.")

    ORDERS_FILE = P.orders_csv
    AUTO_MOVE_DAYS = 4

    if not os.path.exists(ORDERS_FILE):
        st.info("No draw_orders.csv found.")
        return

    df = _read_csv_cached(ORDERS_FILE, keep_default_na=False, mtime=_mtime(ORDERS_FILE))
    df.columns = [c.strip() for c in df.columns]

    # Ensure columns exist
    needed_cols = [
        "Status", "Active CSV", "Done CSV", "Done Description",
        "T&M Moved", "T&M Moved Timestamp",
        "Done Timestamp"
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = ""

    # Normalize boolean
    df["T&M Moved"] = df["T&M Moved"].astype(str).str.lower().isin(["true", "1", "yes"])

    # -----------------------------
    # Helpers
    # -----------------------------
    def _read_dataset_df(csv_path: str):
        try:
            dfx = _read_csv_cached(csv_path, keep_default_na=False, mtime=_mtime(csv_path))
        except Exception:
            return None
        if dfx is None or dfx.empty:
            return None
        dfx.columns = [str(c).strip() for c in dfx.columns]
        for c in ["Parameter Name", "Value", "Units"]:
            if c not in dfx.columns:
                dfx[c] = ""
        return dfx

    def _read_dataset_kv(csv_path: str) -> dict:
        """
        Reads dataset CSV and returns a dict:
          key = normalized Parameter Name (lower, stripped)
          val = Value (string)
        NOTE: keeps first occurrence only (older files may have duplicates).
        """
        dfx = _read_dataset_df(csv_path)
        if dfx is None:
            return {}
        dfx["Parameter Name"] = dfx["Parameter Name"].astype(str).str.strip()
        dfx["Value"] = dfx["Value"].astype(str).str.strip()

        out = {}
        for _, r in dfx.iterrows():
            k = str(r.get("Parameter Name", "")).strip().lower()
            v = str(r.get("Value", "")).strip()
            if k and k not in out and v.lower() != "nan":
                out[k] = v
        return out

    def _pick(kv: dict, aliases: list) -> str:
        """
        Tries aliases in order. Aliases must be exact strings as stored in CSV.
        We normalize to lowercase to match kv keys.
        """
        for a in aliases:
            k = str(a).strip().lower()
            if k in kv and str(kv[k]).strip() and str(kv[k]).strip().lower() != "nan":
                return str(kv[k]).strip()
        return ""

    def _to_dt(s: str):
        s = str(s or "").strip()
        if not s:
            return None
        try:
            x = pd.to_datetime(s, errors="coerce")
            if pd.isna(x):
                return None
            return x.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None

    def _infer_done_dt(row, kv: dict):
        """
        Priority:
          1) draw_orders.csv "Done Timestamp" (if exists)
          2) dataset CSV "Order__Draw Date" (new order scheme)
          3) dataset CSV "Draw Date" (legacy)
          4) dataset CSV "Process__Process Setup Timestamp" / "Process Setup Timestamp"
          5) None
        """
        dt1 = _to_dt(row.get("Done Timestamp", ""))
        if dt1:
            return dt1

        dt2 = _to_dt(_pick(kv, ["Order__Draw Date"]))
        if dt2:
            return dt2

        dt3 = _to_dt(_pick(kv, ["Draw Date"]))
        if dt3:
            return dt3

        dt4 = _to_dt(_pick(kv, ["Process__Process Setup Timestamp", "Process Setup Timestamp"]))
        if dt4:
            return dt4

        return None

    def fmt_float(x, nd=2):
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return "—"
            return f"{float(s):.{nd}f}"
        except Exception:
            return "—"

    def fmt_int(x):
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return "—"
            return str(int(float(s)))
        except Exception:
            return "—"

    # -----------------------------
    # Filter "Done and not moved"
    # -----------------------------
    done_not_moved = df[
        (df["Status"].astype(str).str.strip().str.lower() == "done")
        & (~df["T&M Moved"])
    ].copy()

    if done_not_moved.empty:
        st.success("✅ No recent DONE draws waiting here (everything is already moved to T&M).")
        return

    # -----------------------------
    # Auto-move after 4 days
    # -----------------------------
    now = dt.datetime.now()
    changed = False
    recent_rows = []

    for idx, row in done_not_moved.iterrows():
        done_csv = str(row.get("Done CSV") or "").strip()
        active_csv = str(row.get("Active CSV") or "").strip()
        csv_name = done_csv if done_csv else active_csv

        csv_path = dataset_csv_path(csv_name) if csv_name else ""
        kv = _read_dataset_kv(csv_path) if (csv_name and os.path.exists(csv_path)) else {}

        done_dt = _infer_done_dt(row, kv)

        if done_dt and not str(row.get("Done Timestamp", "")).strip():
            df.loc[idx, "Done Timestamp"] = done_dt.strftime("%Y-%m-%d %H:%M:%S")
            changed = True

        if done_dt:
            age_days = (now - done_dt).total_seconds() / 86400.0
        else:
            age_days = 0.0

        if age_days >= AUTO_MOVE_DAYS:
            df.loc[idx, "T&M Moved"] = True
            if not str(df.loc[idx, "T&M Moved Timestamp"]).strip():
                df.loc[idx, "T&M Moved Timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
            changed = True
        else:
            recent_rows.append((idx, row, kv, csv_name, csv_path, done_dt, age_days))

    if changed:
        df.to_csv(ORDERS_FILE, index=False)

    if not recent_rows:
        st.success("✅ No recent DONE draws (older than 4 days were auto-moved to T&M).")
        return

    # -----------------------------
    # UI styling (cards)
    # -----------------------------
    st.markdown(
        """
        <style>
        .done-card {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(10,10,10,0.60);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            margin-bottom: 12px;
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
            will-change: transform;
        }
        .done-card:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 14px 30px rgba(0,0,0,0.32);
            border-color: rgba(140,255,185,0.35);
        }
        .done-meta {
            color: rgba(255,255,255,0.82);
            font-size: 0.92rem;
            margin-top: -6px;
            margin-bottom: 10px;
        }
        .done-main {
            font-size: 1.08rem;
            font-weight: 800;
            color: rgba(255,255,255,0.97);
            margin-bottom: 6px;
        }
        .done-sub {
            color: rgba(255,255,255,0.78);
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .done-pill {
            display:inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.06);
            margin-left: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Cards
    # -----------------------------
    for (idx, row, kv, csv_name, csv_path, done_dt, age_days) in recent_rows:
        done_desc = str(row.get("Done Description") or "").strip()

        # ✅ NEW: Prefer Order__/Process__ names first, then legacy aliases
        project = _pick(kv, [
            "Order__Fiber Project",
            "Fiber Project",
            "Project",
            "Project Name",
            "Fiber name and number",
            "Fiber Name and Number",
        ]) or str(row.get("Project Name") or "").strip()

        preform = _pick(kv, [
            "Order__Preform Number",
            "Preform Number",
            "Preform Name",
            "Preform",
        ]) or str(row.get("Preform Name") or row.get("Preform Number") or "").strip()

        fiber = _pick(kv, [
            "Order__Fiber Geometry Type",
            "Fiber Geometry Type",
            "Fiber Type",
        ]) or str(row.get("Fiber Type") or row.get("Fiber Project") or "").strip()

        # Drum: prefer dashboard group, then process setup, then legacy
        drum = _pick(kv, [
            "Drum | Selected",
            "Process__Selected Drum",
            "Selected Drum",
            "Drum",
        ])

        # Lengths: new dashboard name first
        total_km = _pick(kv, [
            "Fiber Length | End (log end)",
            "Fiber Length End (log end)",
            "Fibre Length End (log end)",
        ])

        save_km = _pick(kv, [
            "Total Saved Length",
            "Total Saved Length (km)",
        ])

        cut_km = _pick(kv, [
            "Total Cut Length",
            "Total Cut Length (km)",
        ])

        zones_n = _pick(kv, [
            "Good Zones Count",
            "Order__Good Zones Count (required length zones)",
            "Good Zones Count (required length zones)",
        ])

        done_str = done_dt.strftime("%Y-%m-%d %H:%M:%S") if done_dt else "—"
        age_str = f"{age_days:.1f} days" if done_dt else "—"

        st.markdown("<div class='done-card'>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='done-meta'>
                <div class="done-main">
                    Project: {project or "—"} &nbsp;&nbsp;|&nbsp;&nbsp; Preform: {preform or "—"}
                </div>
                <div class="done-sub">
                    Fiber type: {fiber or "—"} &nbsp;|&nbsp; Drum: {drum or "—"}
                    &nbsp;|&nbsp; CSV: <code>{csv_name or "—"}</code>
                </div>
                <div style="margin-top:6px;">
                    <span class="done-pill">Done: {done_str}</span>
                    <span class="done-pill">Age: {age_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total (km)", fmt_float(total_km, 2))
        c2.metric("SAVE (km)", fmt_float(save_km, 2))
        c3.metric("CUT (km)", fmt_float(cut_km, 2))
        c4.metric("Zones", fmt_int(zones_n))

        if done_desc:
            st.caption(f"Done notes: {done_desc}")

        if done_dt:
            st.caption(
                f"Auto-move to T&M after **{AUTO_MOVE_DAYS} days** "
                f"(this one moves in ~{max(0.0, AUTO_MOVE_DAYS - age_days):.1f} days)."
            )
        else:
            st.caption(f"Auto-move to T&M after **{AUTO_MOVE_DAYS} days** (done time unknown).")

        st.markdown("</div>", unsafe_allow_html=True)
