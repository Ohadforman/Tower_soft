def render_schedule_tab(P):
    import os
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    
    st.markdown(
        """
        <style>
          .sched-top-spacer{ height: 8px; }
          .sched-title{
            font-size: 1.62rem;
            font-weight: 900;
            margin: 0;
            padding-top: 4px;
            line-height: 1.2;
            color: rgba(236,248,255,0.98);
            text-shadow: 0 0 14px rgba(86,178,255,0.22);
          }
          .sched-sub{
            margin: 4px 0 8px 0;
            font-size: 0.92rem;
            color: rgba(188,224,248,0.88);
          }
          .sched-line{
            height: 1px;
            margin: 0 0 12px 0;
            background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
          }
          .sched-section{
            margin-top: 8px;
            margin-bottom: 8px;
            padding-left: 8px;
            border-left: 3px solid rgba(120,200,255,0.62);
            font-size: 1.04rem;
            font-weight: 820;
            color: rgba(230,246,255,0.98);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sched-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sched-title">📅 Tower Schedule</div>', unsafe_allow_html=True)
    st.markdown('<div class="sched-sub">Plan, view, and manage recurring/non-recurring tower events.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sched-line"></div>', unsafe_allow_html=True)
    
    SCHEDULE_FILE = P.schedule_csv
    required_columns = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
    
    # =========================================================
    # Ensure schedule file exists + required columns
    # =========================================================
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=required_columns).to_csv(SCHEDULE_FILE, index=False)
        st.warning("Schedule file was missing. New file with required columns created.")
    
    schedule_df = pd.read_csv(SCHEDULE_FILE)

    # Ensure required columns exist
    needs_persist = False
    missing_columns = [c for c in required_columns if c not in schedule_df.columns]
    for c in missing_columns:
        schedule_df[c] = ""

    if missing_columns:
        needs_persist = True

    # Enforce column order and persist only when changed (avoid write on each rerun)
    if list(schedule_df.columns) != required_columns:
        needs_persist = True
    schedule_df = schedule_df[required_columns]
    if needs_persist:
        schedule_df.to_csv(SCHEDULE_FILE, index=False)
    
    # Parse datetimes safely
    schedule_df["Start DateTime"] = pd.to_datetime(schedule_df["Start DateTime"], errors="coerce")
    schedule_df["End DateTime"] = pd.to_datetime(schedule_df["End DateTime"], errors="coerce")
    
    # Clean leaked Plotly template text from Description
    schedule_df["Description"] = (
        schedule_df["Description"]
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Description=", "", regex=False)
        .str.strip()
    )
    
    # Normalize recurrence display in MASTER (so empty shows "None")
    def _norm_recur(v) -> str:
        r = str(v).strip()
        return "None" if r in ["", "None", "none", "NONE", "nan", "NaN"] else r
    
    schedule_df["Recurrence"] = schedule_df["Recurrence"].apply(_norm_recur)
    
    # =========================================================
    # Quick range presets (SAFE: apply before widgets instantiate)
    # =========================================================
    if "schedule_apply_preset" not in st.session_state:
        st.session_state["schedule_apply_preset"] = None  # None / "w1" / "m1" / "m3"

    # Initialize widget-backed state once; avoid passing `value=` with keyed widgets.
    if "schedule_start_date" not in st.session_state:
        st.session_state["schedule_start_date"] = pd.Timestamp.now().date()
    if "schedule_end_date" not in st.session_state:
        st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(weeks=1)).date()
    
    preset = st.session_state.get("schedule_apply_preset")
    if preset:
        today = pd.Timestamp.now().date()
        if preset == "w1":
            st.session_state["schedule_start_date"] = today
            st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(weeks=1)).date()
        elif preset == "m1":
            st.session_state["schedule_start_date"] = today
            st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(months=1)).date()
        elif preset == "m3":
            st.session_state["schedule_start_date"] = today
            st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(months=3)).date()
    
        st.session_state["schedule_apply_preset"] = None
    
    # =========================================================
    # Main-page Filters
    # =========================================================
    st.markdown('<div class="sched-section">🗓️ View Range</div>', unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns([1.1, 1.1, 1.4])
    
    with f1:
        start_filter = st.date_input(
            "Start Date",
            key="schedule_start_date",
        )
    
    with f2:
        end_filter = st.date_input(
            "End Date",
            key="schedule_end_date",
        )
    
    with f3:
        st.markdown("#### Quick ranges")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("1 Week", use_container_width=True, key="sched_preset_w1"):
                st.session_state["schedule_apply_preset"] = "w1"
                st.rerun()
        with b2:
            if st.button("1 Month", use_container_width=True, key="sched_preset_m1"):
                st.session_state["schedule_apply_preset"] = "m1"
                st.rerun()
        with b3:
            if st.button("3 Months", use_container_width=True, key="sched_preset_m3"):
                st.session_state["schedule_apply_preset"] = "m3"
                st.rerun()
    
    range_start = pd.to_datetime(start_filter)
    range_end = pd.to_datetime(end_filter) + pd.to_timedelta(1, unit="day")  # include end day
    
    base = schedule_df.dropna(subset=["Start DateTime", "End DateTime"]).copy()
    
    # =========================================================
    # Expand recurring events so they "show all"
    # =========================================================
    def _next_dt(dt: pd.Timestamp, recurrence: str) -> pd.Timestamp:
        r = str(recurrence).strip().lower()
        if r == "weekly":
            return dt + pd.DateOffset(weeks=1)
        if r == "monthly":
            return dt + pd.DateOffset(months=1)
        if r in ("every 3 months", "3 months", "quarterly"):
            return dt + pd.DateOffset(months=3)
        if r in ("every 6 months", "6 months", "semiannual", "semi-annually"):
            return dt + pd.DateOffset(months=6)
        if r == "yearly":
            return dt + pd.DateOffset(years=1)
        return dt
    
    expanded_rows = []
    for _, row in base.iterrows():
        rec = _norm_recur(row.get("Recurrence", "None"))
        start_dt = row["Start DateTime"]
        end_dt = row["End DateTime"]
    
        if pd.isna(start_dt) or pd.isna(end_dt):
            continue
    
        # If no recurrence -> keep single
        if rec == "None":
            rdict = row.to_dict()
            rdict["Recurrence"] = "None"
            expanded_rows.append(rdict)
            continue
    
        duration = end_dt - start_dt
        occ_start = start_dt
        occ_end = occ_start + duration
    
        safety = 0
        while occ_end < range_start and safety < 5000:
            occ_start = _next_dt(occ_start, rec)
            occ_end = occ_start + duration
            safety += 1
    
        safety = 0
        while occ_start <= range_end and safety < 5000:
            new_row = row.to_dict()
            new_row["Start DateTime"] = occ_start
            new_row["End DateTime"] = occ_end
            new_row["Recurrence"] = rec  # keep as display-ready
            expanded_rows.append(new_row)
    
            occ_start = _next_dt(occ_start, rec)
            occ_end = occ_start + duration
            safety += 1
    
    expanded_df = pd.DataFrame(expanded_rows)
    if not expanded_df.empty:
        expanded_df["Start DateTime"] = pd.to_datetime(expanded_df["Start DateTime"], errors="coerce")
        expanded_df["End DateTime"] = pd.to_datetime(expanded_df["End DateTime"], errors="coerce")
        expanded_df = expanded_df.dropna(subset=["Start DateTime", "End DateTime"])
    
    # =========================================================
    # Filter by overlap (on expanded)
    # =========================================================
    if expanded_df.empty:
        filtered_schedule = expanded_df
    else:
        filtered_schedule = expanded_df[
            (expanded_df["End DateTime"] >= range_start) &
            (expanded_df["Start DateTime"] <= range_end)
        ].copy()
    
    # =========================================================
    # Timeline (FIXED hover formatting)
    #   px.timeline does NOT support %{x_end|...} in hovertemplate.
    #   So we precompute formatted strings and show them via custom_data.
    # =========================================================
    st.markdown('<div class="sched-section">📈 Timeline</div>', unsafe_allow_html=True)
    
    event_colors = {
        "Maintenance": "#4C78A8",      # muted steel blue
        "Drawing": "#5F9E89",          # desaturated teal-green
        "Stop": "#9A5A5A",             # muted brick red
        "Management Event": "#7A6D8F", # dusty violet-gray
    }
    
    if not filtered_schedule.empty:
        # Precompute clean display strings
        filtered_schedule["StartStr"] = filtered_schedule["Start DateTime"].dt.strftime("%Y-%m-%d %H:%M")
        filtered_schedule["EndStr"] = filtered_schedule["End DateTime"].dt.strftime("%Y-%m-%d %H:%M")
        filtered_schedule["RecurrenceDisp"] = filtered_schedule["Recurrence"].apply(_norm_recur)
    
        # Also ensure description is clean for hover
        filtered_schedule["Description"] = (
            filtered_schedule["Description"]
            .astype(str)
            .str.replace(r"%\{.*?\}", "", regex=True)
            .str.replace("Description=", "", regex=False)
            .str.strip()
        )
    
        fig = px.timeline(
            filtered_schedule,
            x_start="Start DateTime",
            x_end="End DateTime",
            y="Event Type",
            color="Event Type",
            color_discrete_map=event_colors,
            custom_data=["StartStr", "EndStr", "RecurrenceDisp", "Description"],
            title="Tower Schedule",
        )
    
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
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,16,28,0.32)",
            font=dict(color="rgba(220,238,252,0.96)"),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(color="rgba(212,232,248,0.94)"),
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events in the selected date range.")
    
    st.divider()
    
    # =========================================================
    # Management area (Main page, no sidebar)
    # =========================================================
    st.markdown('<div class="sched-section">🧩 Manage Schedule</div>', unsafe_allow_html=True)
    
    left, right = st.columns([1.05, 0.95], gap="large")
    
    # -----------------------------
    # LEFT: Master table editor
    # -----------------------------
    with left:
        with st.expander("🧩 Current Schedule (Master)", expanded=False):
            st.caption("Master stores recurrence once; timeline shows expanded occurrences.")
            st.data_editor(schedule_df, height=320, use_container_width=True, key="sched_master_editor")

            csave, creload = st.columns([1, 1])
            with csave:
                if st.button("💾 Save Master Table", use_container_width=True, key="sched_save_master"):
                    edited = st.session_state.get("sched_master_editor", schedule_df)

                    for c in required_columns:
                        if c not in edited.columns:
                            edited[c] = ""
                    edited = edited[required_columns].copy()

                    edited["Start DateTime"] = pd.to_datetime(edited["Start DateTime"], errors="coerce")
                    edited["End DateTime"] = pd.to_datetime(edited["End DateTime"], errors="coerce")

                    edited["Description"] = (
                        edited["Description"]
                        .astype(str)
                        .str.replace(r"%\{.*?\}", "", regex=True)
                        .str.replace("Description=", "", regex=False)
                        .str.strip()
                    )

                    edited["Recurrence"] = edited["Recurrence"].apply(_norm_recur)

                    edited.to_csv(SCHEDULE_FILE, index=False)
                    st.success("Saved master schedule.")
                    st.rerun()

            with creload:
                if st.button("🔄 Reload From File", use_container_width=True, key="sched_reload"):
                    st.rerun()
    
    # -----------------------------
    # RIGHT: Add / Delete
    # -----------------------------
    with right:
        with st.expander("➕ Add New Event", expanded=False):
            event_type = st.selectbox(
                "Event Type",
                ["Maintenance", "Drawing", "Stop", "Management Event"],
                key="sched_type",
            )
            event_description = st.text_area("Description", key="sched_desc", height=90)
    
            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input("Start Date", pd.Timestamp.now().date(), key="sched_start_date2")
                start_time = st.time_input("Start Time", key="sched_start_time")
            with d2:
                end_date = st.date_input("End Date", pd.Timestamp.now().date(), key="sched_end_date2")
                end_time = st.time_input("End Time", key="sched_end_time")
    
            recurrence = st.selectbox(
                "Recurrence",
                ["None", "Weekly", "Monthly", "Every 3 Months", "Every 6 Months", "Yearly"],
                key="sched_recur",
            )
    
            start_datetime = pd.to_datetime(f"{start_date} {start_time}")
            end_datetime = pd.to_datetime(f"{end_date} {end_time}")
    
            if end_datetime < start_datetime:
                st.warning("End DateTime is before Start DateTime.")
    
            if st.button("Add Event", use_container_width=True, key="sched_add_btn"):
                new_event = pd.DataFrame([{
                    "Event Type": event_type,
                    "Start DateTime": start_datetime,
                    "End DateTime": end_datetime,
                    "Description": str(event_description).strip(),
                    "Recurrence": _norm_recur(recurrence),
                }])
    
                full_schedule_df = pd.read_csv(SCHEDULE_FILE)
                for c in required_columns:
                    if c not in full_schedule_df.columns:
                        full_schedule_df[c] = ""
                full_schedule_df = full_schedule_df[required_columns]
    
                full_schedule_df = pd.concat([full_schedule_df, new_event], ignore_index=True)
                full_schedule_df.to_csv(SCHEDULE_FILE, index=False)
                st.success("Event added to schedule!")
                st.rerun()
    
        with st.expander("🗑️ Delete Event", expanded=False):
            if schedule_df.empty:
                st.info("No events available for deletion.")
            else:
                # show Recurrence too (helps users pick the right one)
                delete_options = [
                    f"{i}: {schedule_df.loc[i, 'Event Type']} | "
                    f"{schedule_df.loc[i, 'Start DateTime']} | "
                    f"{_norm_recur(schedule_df.loc[i, 'Recurrence'])} | "
                    f"{str(schedule_df.loc[i, 'Description'])[:60]}"
                    for i in schedule_df.index
                ]
                to_delete = st.selectbox("Select event to delete", delete_options, key="sched_del_select")
                del_idx = int(to_delete.split(":")[0])
    
                if st.button("Delete Selected Event", use_container_width=True, key="sched_del_btn"):
                    schedule_df2 = schedule_df.drop(index=del_idx).reset_index(drop=True)
                    schedule_df2.to_csv(SCHEDULE_FILE, index=False)
                    st.success("Event deleted successfully!")
                    st.rerun()
