def render_home_tab(
    P,
    image_base64,
    STATUS_COL,
    STATUS_UPDATED_COL,
    FAILED_REASON_COL,
    parse_dt_safe,
    now_str,
    safe_str,
    render_home_draw_orders_overview,
    render_done_home_section,
    render_schedule_home_minimal,
    render_parts_orders_home_all,
):
    import os
    import pandas as pd
    import streamlit as st

    st.title("️ Tower Management Software")

    # =========================================================
    # 🎨 CSS (yours + small dialog polish)
    # =========================================================
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .css-1aumxhk {{ background-color: rgba(20, 20, 20, 0.90) !important; }}
        div[data-testid="stDialog"] {{
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # ❌ FAILED → AUTO BACK TO PENDING AFTER 4 DAYS
    # =========================================================
    ORDERS_FILE = P.orders_csv

    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        if STATUS_COL not in df.columns:
            df[STATUS_COL] = "Pending"
        if STATUS_UPDATED_COL not in df.columns:
            df[STATUS_UPDATED_COL] = ""
        if FAILED_REASON_COL not in df.columns:
            df[FAILED_REASON_COL] = ""
        return df

    def auto_move_failed_to_pending(days: int = 4):
        if not os.path.exists(ORDERS_FILE):
            return

        try:
            df = pd.read_csv(ORDERS_FILE)
        except Exception:
            return

        df = _ensure_cols(df)
        if df.empty:
            return

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days)
        changed = False

        for i in range(len(df)):
            if str(df.at[i, STATUS_COL]).strip().lower() != "failed":
                continue

            t = parse_dt_safe(df.at[i, STATUS_UPDATED_COL])

            # stamp missing timestamps so the 4-day timer works
            if t is None:
                df.at[i, STATUS_UPDATED_COL] = now_str()
                changed = True
                continue

            if t < cutoff:
                df.at[i, STATUS_COL] = "Pending"
                df.at[i, STATUS_UPDATED_COL] = now_str()
                changed = True

        if changed:
            df.to_csv(ORDERS_FILE, index=False)


    # =========================================================
    # 🚨 CRITICAL OPEN FAULTS (Home indicator)
    # - Reads maintenance/faults_log.csv
    # - Counts severity == "critical" AND not closed
    # =========================================================
    FAULTS_CSV = os.path.join(P.maintenance_dir, "faults_log.csv")


    def compute_open_critical_faults(faults_csv: str) -> int:
        if not os.path.isfile(faults_csv):
            return 0
        try:
            df = pd.read_csv(faults_csv)
        except Exception:
            return 0
        if df.empty:
            return 0

        # normalize columns
        cols = {c.lower().strip(): c for c in df.columns}
        sev_col = cols.get("fault_severity", None)
        if not sev_col:
            return 0

        # Optional "Status" / "Closed" support (if you add it later)
        status_col = cols.get("fault_status", None)
        closed_col = cols.get("fault_closed", None)

        sev = df[sev_col].astype(str).str.strip().str.lower()

        # If no status info exists → treat everything as open
        is_open = pd.Series(True, index=df.index)

        if status_col:
            stt = df[status_col].astype(str).str.strip().str.lower()
            is_open = ~stt.isin(["closed", "done", "resolved", "fixed"])
        elif closed_col:
            # supports True/False or yes/no
            cl = df[closed_col].astype(str).str.strip().str.lower()
            is_open = ~cl.isin(["true", "1", "yes", "y", "closed"])

        return int((sev == "critical")[is_open].sum())

    # =========================================================
    # ❌ FAILED (last 4 days) — compact list + POPUP reason
    # =========================================================
    def render_failed_home_popup(days_visible: int = 4):
        st.subheader("❌ Failed (last 4 days)")

        if not os.path.exists(ORDERS_FILE):
            st.info("No orders file found.")
            return

        try:
            df = pd.read_csv(ORDERS_FILE)
        except Exception as e:
            st.error(f"Failed to read {ORDERS_FILE}: {e}")
            return

        df = _ensure_cols(df)

        if df.empty:
            st.info("No orders.")
            return

        failed = df[df[STATUS_COL].astype(str).str.strip().str.lower().eq("failed")].copy()
        if failed.empty:
            st.success("No Failed orders 👍")
            return

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days_visible)

        failed["_dt"] = failed[STATUS_UPDATED_COL].apply(parse_dt_safe)
        failed["_dt"] = failed["_dt"].fillna(now)
        failed = failed[failed["_dt"] >= cutoff].copy().sort_values("_dt", ascending=False)

        if failed.empty:
            st.info("No recent Failed orders.")
            return

        def _open_failed_dialog(title: str, reason: str, updated: str, extra_lines: list):
            @st.dialog(title)
            def _dlg():
                if reason:
                    st.error(reason)
                else:
                    st.info("No failed description recorded.")

                if updated:
                    st.caption(f"Updated: {updated}")

                if extra_lines:
                    st.markdown("**Info**")
                    for line in extra_lines:
                        if line:
                            st.write(f"• {line}")

            _dlg()

        for i, (_, row) in enumerate(failed.iterrows()):
            oid = safe_str(row.get("Order ID"))
            pf = safe_str(row.get("Preform Number"))
            ftype = safe_str(row.get("Fiber Type"))
            proj = safe_str(row.get("Fiber Project"))
            updated = safe_str(row.get(STATUS_UPDATED_COL))
            reason = safe_str(row.get(FAILED_REASON_COL))

            left = " | ".join([p for p in [
                f"#{oid}" if oid else "",
                f"PF {pf}" if pf else "",
                ftype if ftype else "",
                proj if proj else ""
            ] if p])

            extra = []
            if "Required Length (m) (for T&M+costumer)" in failed.columns:
                val = safe_str(row.get("Required Length (m) (for T&M+costumer)"))
                if val:
                    extra.append(f"Required Length: {val} m")
            elif "Required Length (m)" in failed.columns:
                val = safe_str(row.get("Required Length (m)"))
                if val:
                    extra.append(f"Required Length: {val} m")

            if "Priority" in failed.columns:
                val = safe_str(row.get("Priority"))
                if val:
                    extra.append(f"Priority: {val}")

            if "Notes" in failed.columns:
                val = safe_str(row.get("Notes"))
                if val:
                    extra.append(f"Notes: {val}")

            c1, c2 = st.columns([3.2, 1.2])
            with c1:
                st.markdown(f"**{left if left else 'Failed Order'}**")
                if updated:
                    st.caption(f"Updated: {updated}")
            with c2:
                btn_key = f"failed_reason_btn_{i}_{oid}_{pf}"
                if st.button("View reason", key=btn_key, use_container_width=True):
                    dlg_title = left if left else "Failed Order"
                    _open_failed_dialog(
                        title=f"❌ Failed: {dlg_title}",
                        reason=reason,
                        updated=updated,
                        extra_lines=extra
                    )

            st.markdown("---")

    # =========================================================
    # 🔁 AUTO CLEANUP FIRST
    # =========================================================
    auto_move_failed_to_pending(days=4)

    # =========================================================
    # ✅ 1) DRAW ORDERS (keep as-is)
    # =========================================================
    render_home_draw_orders_overview()
    st.markdown("---")

    # =========================================================
    # ✅ 2) DONE
    # =========================================================
    render_done_home_section()
    st.markdown("---")

    # =========================================================
    # ✅ 3) FAILED
    # =========================================================
    render_failed_home_popup(days_visible=4)
    st.markdown("---")

    # =========================================================
    # ✅ 4) CALENDAR / SCHEDULE (MOVED HERE ✅)
    # =========================================================
    render_schedule_home_minimal()
    st.markdown("---")

    # =========================================================
    # 5) MAINTENANCE OVERVIEW (unchanged below)
    # =========================================================
    def compute_maintenance_counts_for_home(
            maint_folder: str,
            dataset_dir: str,
            base_dir: str = None,
    ):
        # (your existing function unchanged)
        import os
        import json
        import datetime as dt
        import pandas as pd
        import numpy as np

        base_dir = base_dir or P.root_dir

        def get_draw_csv_count(folder: str) -> int:
            if not os.path.isdir(folder):
                return 0
            return sum(1 for f in os.listdir(folder) if f.lower().endswith(".csv") and not f.startswith("~$"))

        def parse_date(x):
            if pd.isna(x) or x == "":
                return None
            d = pd.to_datetime(x, errors="coerce")
            if pd.isna(d):
                return None
            return d.date()

        def parse_float(x):
            if pd.isna(x) or x == "":
                return None
            try:
                return float(x)
            except Exception:
                return None

        def parse_int(x):
            if pd.isna(x) or x == "":
                return None
            try:
                return int(float(x))
            except Exception:
                return None

        def norm_source(s) -> str:
            s = "" if s is None or pd.isna(s) else str(s)
            return s.strip().lower()

        def mode_norm(x: str) -> str:
            s = "" if x is None or pd.isna(x) else str(x).strip().lower()
            if s in ("draw", "draws", "draws_count", "draw_count"):
                return "draws"
            return s

        def load_state(path: str) -> dict:
            try:
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
            return {}

        state_path = os.path.join(maint_folder, "_app_state.json")
        state = load_state(state_path)

        current_date = dt.date.today()
        furnace_hours = float(state.get("furnace_hours", 0.0) or 0.0)
        uv1_hours = float(state.get("uv1_hours", 0.0) or 0.0)
        uv2_hours = float(state.get("uv2_hours", 0.0) or 0.0)
        warn_days = int(state.get("warn_days", 14) or 14)
        warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

        current_draw_count = get_draw_csv_count(dataset_dir)

        if not os.path.isdir(maint_folder):
            return 0, 0

        files = [f for f in os.listdir(maint_folder) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        if not files:
            return 0, 0

        normalize_map = {
            "equipment": "Component",
            "task name": "Task",
            "task id": "Task_ID",
            "interval type": "Interval_Type",
            "interval value": "Interval_Value",
            "interval unit": "Interval_Unit",
            "tracking mode": "Tracking_Mode",
            "hours source": "Hours_Source",
            "calendar rule": "Calendar_Rule",
            "due threshold (days)": "Due_Threshold_Days",
            "document name": "Manual_Name",
            "document file/link": "Document",
            "manual page": "Page",
            "procedure summary": "Procedure_Summary",
            "safety/notes": "Notes",
            "owner": "Owner",
            "last done date": "Last_Done_Date",
            "last done hours": "Last_Done_Hours",
            "last done draw": "Last_Done_Draw",
        }

        REQUIRED = ["Component", "Task", "Tracking_Mode"]
        OPTIONAL = [
            "Task_ID",
            "Interval_Type", "Interval_Value", "Interval_Unit",
            "Due_Threshold_Days",
            "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
            "Manual_Name", "Page", "Document",
            "Procedure_Summary", "Notes", "Owner",
            "Hours_Source", "Calendar_Rule",
        ]

        def read_file(path: str) -> pd.DataFrame:
            if path.lower().endswith(".csv"):
                return pd.read_csv(path)
            return pd.read_excel(path)

        def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.rename(columns={c: normalize_map.get(str(c).strip().lower(), c) for c in df.columns}, inplace=True)
            for r in REQUIRED:
                if r not in df.columns:
                    df[r] = np.nan
            for c in OPTIONAL:
                if c not in df.columns:
                    df[c] = np.nan
            return df

        frames = []
        for fname in sorted(files):
            fpath = os.path.join(maint_folder, fname)
            try:
                raw = read_file(fpath)
                if raw is None or raw.empty:
                    continue
                dfm = normalize_df(raw)
                dfm["Source_File"] = fname
                frames.append(dfm)
            except Exception:
                continue

        if not frames:
            return 0, 0

        dfm = pd.concat(frames, ignore_index=True)

        def pick_current_hours(hours_source: str) -> float:
            hs = norm_source(hours_source)
            if hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
                return float(uv2_hours)
            if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
                return float(uv1_hours)
            return float(furnace_hours)

        dfm["Last_Done_Date_parsed"] = dfm["Last_Done_Date"].apply(parse_date)
        dfm["Last_Done_Hours_parsed"] = dfm["Last_Done_Hours"].apply(parse_float)
        dfm["Last_Done_Draw_parsed"] = dfm["Last_Done_Draw"].apply(parse_int)
        dfm["Current_Hours_For_Task"] = dfm["Hours_Source"].apply(pick_current_hours)
        dfm["Tracking_Mode_norm"] = dfm["Tracking_Mode"].apply(mode_norm)

        def next_due_date(row):
            if row.get("Tracking_Mode_norm") != "calendar":
                return None
            last = row.get("Last_Done_Date_parsed", None)
            if last is None:
                return None
            try:
                v = int(float(row.get("Interval_Value", np.nan)))
            except Exception:
                return None
            unit = str(row.get("Interval_Unit", "")).strip().lower()
            base = pd.Timestamp(last)
            if pd.isna(base) or base is pd.NaT:
                return None
            if "day" in unit:
                out = base + pd.DateOffset(days=v)
            elif "week" in unit:
                out = base + pd.DateOffset(weeks=v)
            elif "month" in unit:
                out = base + pd.DateOffset(months=v)
            elif "year" in unit:
                out = base + pd.DateOffset(years=v)
            else:
                out = base + pd.DateOffset(days=v)
            if pd.isna(out) or out is pd.NaT:
                return None
            return out.date()

        def next_due_hours(row):
            if row.get("Tracking_Mode_norm") != "hours":
                return None
            last_h = row.get("Last_Done_Hours_parsed", None)
            if last_h is None:
                return None
            try:
                v = float(row.get("Interval_Value", np.nan))
            except Exception:
                return None
            if pd.isna(v):
                return None
            return float(last_h) + float(v)

        def next_due_draw(row):
            if row.get("Tracking_Mode_norm") != "draws":
                return None
            last_d = row.get("Last_Done_Draw_parsed", None)
            if last_d is None:
                return None
            try:
                v = int(float(row.get("Interval_Value", np.nan)))
            except Exception:
                return None
            return int(last_d) + int(v)

        dfm["Next_Due_Date"] = dfm.apply(next_due_date, axis=1)
        dfm["Next_Due_Hours"] = dfm.apply(next_due_hours, axis=1)
        dfm["Next_Due_Draw"] = dfm.apply(next_due_draw, axis=1)

        def status_row(row):
            mode = row.get("Tracking_Mode_norm", "")
            if mode == "event":
                return "ROUTINE"

            overdue = False
            due_soon = False

            nd = row.get("Next_Due_Date", None)
            nh = row.get("Next_Due_Hours", None)
            ndr = row.get("Next_Due_Draw", None)

            if nd is not None and not pd.isna(nd):
                if nd < current_date:
                    overdue = True
                else:
                    thresh = row.get("Due_Threshold_Days", np.nan)
                    try:
                        thresh = int(float(thresh)) if not pd.isna(thresh) else int(warn_days)
                    except Exception:
                        thresh = int(warn_days)
                    if (nd - current_date).days <= thresh:
                        due_soon = True

            if nh is not None and not pd.isna(nh):
                nh = float(nh)
                cur_h = float(row.get("Current_Hours_For_Task", 0.0))
                if nh < cur_h:
                    overdue = True
                elif (nh - cur_h) <= float(warn_hours):
                    due_soon = True

            if ndr is not None and not pd.isna(ndr):
                ndr = int(ndr)
                if ndr < int(current_draw_count):
                    overdue = True
                elif (ndr - int(current_draw_count)) <= 5:
                    due_soon = True

            if overdue:
                return "OVERDUE"
            if due_soon:
                return "DUE SOON"
            return "OK"

        dfm["Status"] = dfm.apply(status_row, axis=1)

        overdue = int((dfm["Status"] == "OVERDUE").sum())
        due_soon = int((dfm["Status"] == "DUE SOON").sum())
        return overdue, due_soon

    st.subheader("🧰 Maintenance Overview")

    MAINT_FOLDER = P.maintenance_dir
    DATASET_DIR = P.dataset_dir

    overdue, due_soon = compute_maintenance_counts_for_home(
        maint_folder=MAINT_FOLDER,
        dataset_dir=DATASET_DIR,
    )

    st.session_state["maint_overdue"] = overdue
    st.session_state["maint_due_soon"] = due_soon

    c1, c2 = st.columns(2)
    c1.metric("🔴 Overdue", overdue)
    c2.metric("🟠 Due soon", due_soon)

    st.subheader("🚨 Faults Overview")

    open_critical = compute_open_critical_faults(FAULTS_CSV)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("🟥 Critical open faults", open_critical)

    with c2:
        if open_critical == 0:
            st.success("No critical faults ✅")
        else:
            st.warning("Check Maintenance → Faults")

    with c3:
        if open_critical > 0:
            st.caption("Tip: open 🧰 Maintenance → Faults / Incidents to review.")
    st.markdown("---")
    # =========================================================
    # 6) PARTS NEEDED
    # =========================================================

    render_parts_orders_home_all()
