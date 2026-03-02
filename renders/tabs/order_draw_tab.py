def render_order_draw_tab(P):
    import os
    import datetime as dt
    import pandas as pd
    import streamlit as st
    import json
    from helpers.text_utils import to_float
    from renders.support.style_utils import color_priority, color_status
    from renders.support.ui_state import safe_str_from_state
    
    st.title("📦 Order Draw")
    
    orders_file = P.orders_csv
    SCHEDULE_FILE = P.schedule_csv
    schedule_required_cols = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
    
    SCHEDULE_PASSWORD = "DORON"
    
    GOOD_ZONES_COL = "Good Zones Count (required length zones)"
    FIBER_GEOMETRY_COL = "Fiber Geometry Type"
    
    SAP_INVENTORY_FILE = P.sap_rods_inventory_csv
    
    PROJECTS_FILE = P.projects_fiber_csv
    PROJECTS_COL = "Fiber Project"
    PROJECT_TEMPLATES_FILE = P.projects_fiber_templates_csv
    
    # ✅ coating temperature columns
    MAIN_COAT_TEMP_COL = "Main Coating Temperature (°C)"
    SEC_COAT_TEMP_COL = "Secondary Coating Temperature (°C)"
    
    # ✅ geometry-specific columns
    TIGER_CUT_COL = "Tiger Cut (%)"
    OCT_F2F_COL = "Octagonal F2F (mm)"
    
    # ✅ config_coating.json path (coatings list must match this!)
    COATING_CFG_PATH = P.coating_config_json
    
    # ✅ tolerance columns
    FIBER_D_TOL_COL = "Fiber Diameter Tol (± µm)"
    MAIN_D_TOL_COL = "Main Coating Diameter Tol (± µm)"
    SEC_D_TOL_COL = "Secondary Coating Diameter Tol (± µm)"
    
    FIBER_GEOMETRY_OPTIONS = [
        "",
        "PANDA - PM",
        "TIGER - PM",
        "Octagonal",
        "ROUND",
        "STEP INDEX",
        "Ring Core",
        "Hollow Core",
        "Photonic Crystal",
        "Custom (write in Notes)",
    ]
    
    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    
    
    def __from_state(key: str, default=0.0) -> float:
        return to_float(st.session_state.get(key, default), default)
    
    
    def _fmt_pm(val: float, tol: float, unit: str = "µm") -> str:
        try:
            val = float(val)
            tol = float(tol)
        except Exception:
            return ""
        if val <= 0:
            return ""
        if tol > 0:
            return f"{val:g} ± {tol:g} {unit}"
        return f"{val:g} {unit}"
    
    # ---------------------------------------------------------
    # Load coating options from config_coating.json
    # ---------------------------------------------------------
    def load_config_coating_json(path: str = COATING_CFG_PATH) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def coating_options_from_cfg(cfg: dict) -> list:
        coats = (cfg or {}).get("coatings", {})
        if isinstance(coats, dict):
            return [str(k).strip() for k in coats.keys() if str(k).strip() != ""]
        return []
    
    
    coating_cfg = load_config_coating_json()
    COATING_OPTIONS = coating_options_from_cfg(coating_cfg) or [""]
    if not COATING_OPTIONS:
        st.warning("⚠️ No coatings found in config_coating.json → using empty list.")
        COATING_OPTIONS = []
    
    # ---------------------------------------------------------
    # Ensure schedule file exists
    # ---------------------------------------------------------
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=schedule_required_cols).to_csv(SCHEDULE_FILE, index=False)
    
    # ---------------------------------------------------------
    # Projects list helpers
    # ---------------------------------------------------------
    def ensure_projects_file():
        if not os.path.exists(PROJECTS_FILE):
            pd.DataFrame(columns=[PROJECTS_COL]).to_csv(PROJECTS_FILE, index=False)
    
    def load_projects() -> list:
        ensure_projects_file()
        try:
            d = pd.read_csv(PROJECTS_FILE, keep_default_na=False)
        except Exception:
            return []
        if PROJECTS_COL not in d.columns:
            return []
        items = (
            d[PROJECTS_COL].astype(str)
            .replace({"nan": "", "None": ""})
            .fillna("")
            .map(lambda x: x.strip())
        )
        items = [x for x in items.tolist() if x]
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    def add_project(new_name: str):
        new_name = str(new_name or "").strip()
        if not new_name:
            return False, "Project name is empty."
        ensure_projects_file()
        existing = load_projects()
        if new_name in existing:
            return False, "Project already exists."
        dfp = pd.read_csv(PROJECTS_FILE, keep_default_na=False) if os.path.exists(PROJECTS_FILE) else pd.DataFrame()
        if PROJECTS_COL not in dfp.columns:
            dfp[PROJECTS_COL] = ""
        dfp = pd.concat([dfp, pd.DataFrame([{PROJECTS_COL: new_name}])], ignore_index=True)
        dfp.to_csv(PROJECTS_FILE, index=False)
        return True, f"Added project: {new_name}"
    
    # ---------------------------------------------------------
    # Project template helpers (includes tolerances)
    # ---------------------------------------------------------
    TEMPLATE_FIELDS = [
        PROJECTS_COL,
        FIBER_GEOMETRY_COL,
        TIGER_CUT_COL,
        OCT_F2F_COL,
        "Fiber Diameter (µm)",
        FIBER_D_TOL_COL,
        "Main Coating Diameter (µm)",
        MAIN_D_TOL_COL,
        "Secondary Coating Diameter (µm)",
        SEC_D_TOL_COL,
        "Tension (g)",
        "Draw Speed (m/min)",
        "Main Coating",
        "Secondary Coating",
        MAIN_COAT_TEMP_COL,
        SEC_COAT_TEMP_COL,
        "Notes Default",
    ]
    
    TEMPLATE_TO_WIDGET_KEY = {
        FIBER_GEOMETRY_COL: "order_fiber_geometry_required",
        TIGER_CUT_COL: "order_tiger_cut_pct",
        OCT_F2F_COL: "order_oct_f2f_mm",
        "Fiber Diameter (µm)": "order_fiber_diam",
        FIBER_D_TOL_COL: "order_fiber_diam_tol",
        "Main Coating Diameter (µm)": "order_main_diam",
        MAIN_D_TOL_COL: "order_main_diam_tol",
        "Secondary Coating Diameter (µm)": "order_sec_diam",
        SEC_D_TOL_COL: "order_sec_diam_tol",
        "Tension (g)": "order_tension",
        "Draw Speed (m/min)": "order_speed",
        "Main Coating": "order_coating_main",
        "Secondary Coating": "order_coating_secondary",
        MAIN_COAT_TEMP_COL: "order_main_coat_temp_c",
        SEC_COAT_TEMP_COL: "order_sec_coat_temp_c",
        "Notes Default": "order_notes",
    }
    
    NUMERIC_WIDGET_KEYS = {
        "order_fiber_diam",
        "order_fiber_diam_tol",
        "order_main_diam",
        "order_main_diam_tol",
        "order_sec_diam",
        "order_sec_diam_tol",
        "order_tension",
        "order_speed",
        "order_main_coat_temp_c",
        "order_sec_coat_temp_c",
        "order_tiger_cut_pct",
        "order_oct_f2f_mm",
    }
    
    def ensure_templates_file():
        if not os.path.exists(PROJECT_TEMPLATES_FILE):
            pd.DataFrame(columns=TEMPLATE_FIELDS).to_csv(PROJECT_TEMPLATES_FILE, index=False)
    
    def load_templates_df() -> pd.DataFrame:
        ensure_templates_file()
        try:
            d = pd.read_csv(PROJECT_TEMPLATES_FILE, keep_default_na=False)
        except Exception:
            d = pd.DataFrame(columns=TEMPLATE_FIELDS)
        for c in TEMPLATE_FIELDS:
            if c not in d.columns:
                d[c] = ""
        return d[TEMPLATE_FIELDS].copy()
    
    def get_template_for_project(project_name: str) -> dict:
        project_name = str(project_name or "").strip()
        if not project_name:
            return {}
        d = load_templates_df()
        m = d[PROJECTS_COL].astype(str).str.strip() == project_name
        if not m.any():
            return {}
        return d.loc[m].iloc[-1].to_dict()
    
    def save_or_update_template(project_name: str, template_payload: dict):
        project_name = str(project_name or "").strip()
        if not project_name:
            return False, "No project selected."
    
        d = load_templates_df()
        m = d[PROJECTS_COL].astype(str).str.strip() == project_name
    
        row = {k: "" for k in TEMPLATE_FIELDS}
        row[PROJECTS_COL] = project_name
        for k, v in (template_payload or {}).items():
            if k in row:
                row[k] = v
    
        if m.any():
            d.loc[m, :] = pd.DataFrame([row]).iloc[0].values
        else:
            d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    
        d.to_csv(PROJECT_TEMPLATES_FILE, index=False)
        return True, f"✅ Template saved for project: {project_name}"
    
    def apply_template_to_form(project_name: str):
        tpl = get_template_for_project(project_name)
        if not tpl:
            return False
    
        for col, widget_key in TEMPLATE_TO_WIDGET_KEY.items():
            val = tpl.get(col, "")
            if widget_key in NUMERIC_WIDGET_KEYS:
                num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                st.session_state[widget_key] = float(num) if pd.notna(num) else 0.0
            else:
                st.session_state[widget_key] = str(val)
        return True
    
    # ---------------------------------------------------------
    # Auto-apply state
    # ---------------------------------------------------------
    if "order_last_project_applied" not in st.session_state:
        st.session_state["order_last_project_applied"] = ""
    
    # ---------------------------------------------------------
    # SAP inventory helper (read-only)
    # ---------------------------------------------------------
    def get_sap_rods_set_count() -> float:
        if not os.path.exists(SAP_INVENTORY_FILE):
            return 0.0
        try:
            inv = pd.read_csv(SAP_INVENTORY_FILE, keep_default_na=False)
        except Exception:
            return 0.0
        if inv.empty or "Item" not in inv.columns or "Count" not in inv.columns:
            return 0.0
        m = inv["Item"].astype(str).str.strip().str.lower() == "sap rods set"
        if not m.any():
            return 0.0
        val = inv.loc[m, "Count"].iloc[-1]
        num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        return float(num) if pd.notna(num) else 0.0
    
    def render_sap_inventory_banner():
        sap_cnt = get_sap_rods_set_count()
        if sap_cnt < 1:
            st.warning(f"⚠️ SAP Rods Set inventory is LOW: **{sap_cnt:g}** sets (under 1).")
        else:
            st.success(f"🧪 SAP Rods Set inventory available: **{sap_cnt:g}** sets.")
    
    # =========================================================
    # 1) TABLE FIRST (with colors)
    # =========================================================
    st.subheader("📋 Existing Draw Orders")
    
    if not os.path.exists(orders_file):
        st.info("No orders submitted yet.")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(orders_file, keep_default_na=False)
    
    if not df.empty:
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    
        for col, default in {
            "Status": "Pending",
            "Priority": "Normal",
            PROJECTS_COL: "",
            "Order Opener": "",
            "Preform Number": "",
            FIBER_GEOMETRY_COL: "",
            TIGER_CUT_COL: "",
            OCT_F2F_COL: "",
            "Done CSV": "",
            "Done Description": "",
            "Active CSV": "",
            "T&M Moved": False,
            "T&M Moved Timestamp": "",
            "Required Length (m) (for T&M+costumer)": "",
            GOOD_ZONES_COL: "",
            "Notes": "",
            "Main Coating": "",
            "Secondary Coating": "",
            MAIN_COAT_TEMP_COL: "",
            SEC_COAT_TEMP_COL: "",
            "Fiber Diameter (µm)": "",
            FIBER_D_TOL_COL: "",
            "Main Coating Diameter (µm)": "",
            MAIN_D_TOL_COL: "",
            "Secondary Coating Diameter (µm)": "",
            SEC_D_TOL_COL: "",
            "Tension (g)": "",
            "Draw Speed (m/min)": "",
        }.items():
            if col not in df.columns:
                df[col] = default
    
        if "T&M Moved" in df.columns:
            df["T&M Moved"] = df["T&M Moved"].apply(
                lambda x: str(x).strip().lower() in ("true", "1", "yes", "y", "moved")
            )
        df_visible = df[~df["T&M Moved"]].copy() if "T&M Moved" in df.columns else df.copy()
    
    
        styled_df = (
            df_visible.style
            .applymap(color_status, subset=["Status"] if "Status" in df_visible.columns else None)
            .applymap(color_priority, subset=["Priority"] if "Priority" in df_visible.columns else None)
        )
    
        st.dataframe(styled_df, use_container_width=True)
    else:
        df_visible = pd.DataFrame()
    
    # =========================================================
    # ✅ Pending → Schedule (quick)
    # =========================================================
    st.markdown("---")
    st.subheader("🕒 Pending → Schedule (quick)")
    
    if df_visible is None or df_visible.empty:
        st.info("No orders to schedule.")
    else:
        df_pending = df_visible[df_visible["Status"].astype(str).str.strip() == "Pending"].copy()
    
        if df_pending.empty:
            st.info("No Pending orders.")
        else:
            pending_indices = df_pending.index.tolist()
    
            def _fmt_pending(i: int) -> str:
                try:
                    prj = str(df_pending.loc[i, PROJECTS_COL]).strip()
                    pref = str(df_pending.loc[i, "Preform Number"]).strip()
                    pri = str(df_pending.loc[i, "Priority"]).strip()
                    ts = df_pending.loc[i, "Timestamp"] if "Timestamp" in df_pending.columns else ""
                    return f"#{i} | {prj} | Preform: {pref} | Priority: {pri} | {ts}"
                except Exception:
                    return f"#{i}"
    
            selected_idx = st.selectbox(
                "Select Pending order",
                options=pending_indices,
                format_func=_fmt_pending,
                key="pending_to_schedule_selectbox",
            )
    
            sel_row = df.loc[selected_idx]  # ORIGINAL df row
    
            with st.expander("📅 Schedule selected Pending order", expanded=True):
                preform_now = str(sel_row.get("Preform Number", "")).strip()
                need_preform = (preform_now == "" or preform_now == "0" or preform_now.lower() == "none")
    
                preform_real = ""
                if need_preform:
                    preform_real = st.text_input(
                        "Preform Number (required for scheduling — cannot be 0)",
                        placeholder="e.g., P0888",
                        key="pending_sched_real_preform_input",
                    )
    
                pwd2 = st.text_input("Scheduling password", type="password", key="pending_sched_pwd2")
                sched_ok2 = (pwd2 == SCHEDULE_PASSWORD)
                if pwd2.strip():
                    (st.success if sched_ok2 else st.error)("Password OK ✅" if sched_ok2 else "Wrong password ❌")
    
                default_date2 = pd.Timestamp.today().date()
                preset2 = st.radio(
                    "Preset",
                    ["All day (08:00–16:00)", "Before lunch (08:00–12:00)", "After lunch (12:00–16:00)"],
                    horizontal=True,
                    key="pending_sched_preset2",
                    label_visibility="collapsed",
                )
    
                if preset2.startswith("All day"):
                    preset_start2 = dt.time(8, 0)
                    preset_duration2 = 8 * 60
                elif preset2.startswith("Before lunch"):
                    preset_start2 = dt.time(8, 0)
                    preset_duration2 = 4 * 60
                else:
                    preset_start2 = dt.time(12, 0)
                    preset_duration2 = 4 * 60
    
                cA2, cB2, cC2 = st.columns([1, 1, 1], vertical_alignment="bottom")
                with cA2:
                    sched_date2 = st.date_input("Schedule Date", value=default_date2, key="pending_sched_date2")
                with cB2:
                    sched_start2 = st.time_input("Start Time", value=preset_start2, key="pending_sched_start2")
                with cC2:
                    sched_dur2 = st.number_input(
                        "Duration (min)",
                        min_value=1,
                        step=5,
                        value=int(preset_duration2),
                        key="pending_sched_dur2",
                    )
    
                start_dt2 = pd.to_datetime(f"{sched_date2} {sched_start2}")
                end_dt2 = start_dt2 + pd.to_timedelta(int(sched_dur2), unit="m")
    
                if st.button("✅ Schedule this Pending Order", key="pending_schedule_confirm_btn"):
                    if not sched_ok2:
                        st.error("Not scheduled: password missing/wrong.")
                        st.stop()
    
                    if need_preform and not str(preform_real).strip():
                        st.error("Please enter a real Preform Number (cannot schedule with 0).")
                        st.stop()
    
                    existing2 = pd.read_csv(SCHEDULE_FILE) if os.path.exists(SCHEDULE_FILE) else pd.DataFrame()
                    for c in schedule_required_cols:
                        if c not in existing2.columns:
                            existing2[c] = ""
                    existing2 = existing2[schedule_required_cols]
    
                    geom2 = str(sel_row.get(FIBER_GEOMETRY_COL, "")).strip()
                    prj2 = str(sel_row.get(PROJECTS_COL, "")).strip()
                    pri2 = str(sel_row.get("Priority", "")).strip()
                    pref2 = str(preform_real).strip() if need_preform else preform_now
    
                    length2 = sel_row.get("Required Length (m) (for T&M+costumer)", "")
                    zones2 = sel_row.get(GOOD_ZONES_COL, "")
    
                    tiger2 = to_float(sel_row.get(TIGER_CUT_COL, 0.0), 0.0)
                    oct2 = to_float(sel_row.get(OCT_F2F_COL, 0.0), 0.0)
    
                    mtemp2 = to_float(sel_row.get(MAIN_COAT_TEMP_COL, 0.0), 0.0)
                    stemp2 = to_float(sel_row.get(SEC_COAT_TEMP_COL, 0.0), 0.0)
    
                    notes2 = str(sel_row.get("Notes", "")).strip()
    
                    fd2 = to_float(sel_row.get("Fiber Diameter (µm)", 0.0), 0.0)
                    md2 = to_float(sel_row.get("Main Coating Diameter (µm)", 0.0), 0.0)
                    sd2 = to_float(sel_row.get("Secondary Coating Diameter (µm)", 0.0), 0.0)
                    fdt2 = to_float(sel_row.get(FIBER_D_TOL_COL, 0.0), 0.0)
                    mdt2 = to_float(sel_row.get(MAIN_D_TOL_COL, 0.0), 0.0)
                    sdt2 = to_float(sel_row.get(SEC_D_TOL_COL, 0.0), 0.0)
    
                    diam_bits2 = []
                    s_fd2 = _fmt_pm(fd2, fdt2)
                    s_md2 = _fmt_pm(md2, mdt2)
                    s_sd2 = _fmt_pm(sd2, sdt2)
                    if s_fd2:
                        diam_bits2.append(f"Fiber {s_fd2}")
                    if s_md2:
                        diam_bits2.append(f"Coat1 {s_md2}")
                    if s_sd2:
                        diam_bits2.append(f"Coat2 {s_sd2}")
    
                    desc_lines2 = [
                        f"ORDER #{selected_idx} | Priority: {pri2}",
                        f"Fiber: {prj2} | Geometry: {geom2} | Preform: {pref2}",
                        f"Required Length: {length2} m | Good Zones Count: {zones2}",
                    ]
                    if diam_bits2:
                        desc_lines2.append("Diameters: " + " | ".join(diam_bits2))
    
                    if geom2 == "TIGER - PM" and tiger2 > 0:
                        desc_lines2.append(f"Tiger Cut: {tiger2:.1f}%")
                    if geom2 == "Octagonal" and oct2 > 0:
                        desc_lines2.append(f"Oct F2F: {oct2:.2f} mm")
    
                    if mtemp2 > 0:
                        desc_lines2.append(f"Main Coat Temp: {mtemp2:.0f}°C")
                    if stemp2 > 0:
                        desc_lines2.append(f"Sec Coat Temp: {stemp2:.0f}°C")
                    if notes2:
                        desc_lines2.append(f"Notes: {notes2}")
    
                    event_description2 = " | ".join([x for x in desc_lines2 if str(x).strip()])
    
                    new_event2 = pd.DataFrame([{
                        "Event Type": "Drawing",
                        "Start DateTime": start_dt2,
                        "End DateTime": end_dt2,
                        "Description": event_description2,
                        "Recurrence": "None",
                    }])
    
                    pd.concat([existing2, new_event2], ignore_index=True).to_csv(SCHEDULE_FILE, index=False)
    
                    if need_preform:
                        df.at[selected_idx, "Preform Number"] = pref2
                    df.at[selected_idx, "Status"] = "Scheduled"
                    df.to_csv(orders_file, index=False)
    
                    st.success("✅ Scheduled + moved Status to Scheduled.")
                    st.rerun()
    
    # =========================================================
    # 2) CREATE NEW ORDER (UI + tolerances + notes recommended)
    #   IMPORTANT FIX: Schedule UI is OUTSIDE the form
    # =========================================================
    st.markdown("---")
    st.markdown("### ➕ Create New Order")
    
    if "show_new_order_form" not in st.session_state:
        st.session_state["show_new_order_form"] = False
    
    show_form = st.checkbox(
        "Create new order",
        value=bool(st.session_state["show_new_order_form"]),
        key="order_create_new_cb",
    )
    st.session_state["show_new_order_form"] = bool(show_form)
    
    if st.session_state["show_new_order_form"]:
        st.markdown(
            """
            <style>
            div[data-testid="stForm"] { padding-top: 0.25rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
        with st.container(border=True):
            projects = load_projects()
    
            # Project row (outside form to support template auto-apply)
            selA, selB = st.columns([2.4, 1.0], vertical_alignment="bottom")
            with selA:
                selected_project = st.selectbox(
                    "Project * (auto-fills if template exists)",
                    options=[""] + projects,
                    index=0,
                    key="order_project_select",
                    placeholder="Select project...",
                )
            with selB:
                with st.popover("➕ Add project", use_container_width=True):
                    new_proj = st.text_input("New project name", key="order_new_project_name")
                    if st.button("Add", key="order_add_project_btn", use_container_width=True):
                        okp, msgp = add_project(new_proj)
                        (st.success if okp else st.warning)(msgp)
                        if okp:
                            st.rerun()
    
            # Auto-apply template when project changes
            if (
                str(selected_project).strip()
                and st.session_state.get("order_last_project_applied", "") != str(selected_project).strip()
            ):
                applied = apply_template_to_form(selected_project)
                st.session_state["order_last_project_applied"] = str(selected_project).strip()
                if applied:
                    st.toast("Template auto-applied ✅", icon="✅")
                    st.rerun()
    
            if str(selected_project).strip():
                tpl_exists = bool(get_template_for_project(selected_project))
                st.caption("✅ Template exists for this project." if tpl_exists else "ℹ️ No template yet for this project.")
    
            # -----------------------------
            # FORM (single submit)
            # -----------------------------
            save_tpl = False
            submit = False
            cancel = False
    
            with st.form("order_create_form", clear_on_submit=False):
                tab_req, tab_targets, tab_materials, tab_template = st.tabs(
                    ["✅ Required", "🧪 Targets", "🧴 Materials", "💾 Template"]
                )
    
                # ✅ REQUIRED TAB
                with tab_req:
                    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.0, 1.4], vertical_alignment="bottom")
                    with c1:
                        st.text_input(
                            "Preform Number *",
                            key="order_preform_name",
                            placeholder="0 (if not exist yet) or P0888",
                            help="Use 0 if preform does not exist yet.",
                        )
                    with c2:
                        st.text_input(
                            "Fiber Project *",
                            value=str(selected_project),
                            disabled=True,
                            key="order_fiber_project_disabled",
                        )
                    with c3:
                        st.selectbox("Priority *", ["Low", "Normal", "High"], index=1, key="order_priority")
                    with c4:
                        st.selectbox(
                            f"{FIBER_GEOMETRY_COL} *",
                            options=FIBER_GEOMETRY_OPTIONS,
                            index=0,
                            key="order_fiber_geometry_required",
                        )
    
                    if "order_tiger_cut_pct" not in st.session_state:
                        st.session_state["order_tiger_cut_pct"] = 0.0
                    if "order_oct_f2f_mm" not in st.session_state:
                        st.session_state["order_oct_f2f_mm"] = 0.0
    
                    geom = str(st.session_state.get("order_fiber_geometry_required", "")).strip()
                    g1, g2, g3 = st.columns([1.2, 1.2, 1.6], vertical_alignment="bottom")
    
                    with g1:
                        if geom == "TIGER - PM":
                            st.number_input(
                                "Tiger Cut (%) *",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.5,
                                value=__from_state("order_tiger_cut_pct", 0.0),
                                key="order_tiger_cut_pct",
                            )
                        else:
                            st.caption("Tiger Cut (%) — only for TIGER")
    
                    with g2:
                        if geom == "Octagonal":
                            st.number_input(
                                "Octagonal F2F (mm) *",
                                min_value=0.0,
                                step=0.01,
                                value=__from_state("order_oct_f2f_mm", 0.0),
                                format="%.2f",
                                key="order_oct_f2f_mm",
                            )
                        else:
                            st.caption("Octagonal F2F — only for Octagonal")
    
                    with g3:
                        if geom == "PANDA - PM":
                            st.markdown("**🧪 SAP Inventory**")
                            render_sap_inventory_banner()
                        else:
                            st.caption("SAP inventory — only for PANDA - PM")
    
                    r5, r6, r7 = st.columns([1.3, 1.1, 1.6], vertical_alignment="bottom")
                    with r5:
                        st.number_input(
                            "Required Length (m) *",
                            min_value=0.0,
                            key="order_length_required_required",
                            help="Required Length (m) (for T&M+costumer)",
                        )
                    with r6:
                        st.number_input(
                            "Good Zones Count *",
                            min_value=1,
                            step=1,
                            value=int(st.session_state.get("order_good_zones_required", 1) or 1),
                            key="order_good_zones_required",
                            help=GOOD_ZONES_COL,
                        )
                    with r7:
                        st.text_input("Order Opened By *", key="order_opener", placeholder="Name / initials")
    
                    # Notes shown in required, recommended but NOT blocking submit
                    st.markdown("##### Notes (recommended)")
                    st.text_area(
                        "Additional Notes / Instructions",
                        key="order_notes",
                        height=120,
                        placeholder="Optional but recommended (special instructions, customer notes, risks, etc.)",
                    )
    
                # 🧪 TARGETS TAB
                with tab_targets:
                    st.caption("Optional targets. Leave 0 if unknown.")
                    d1, d2, d3 = st.columns(3, vertical_alignment="bottom")
                    with d1:
                        st.number_input("Fiber Diameter (µm)", min_value=0.0, key="order_fiber_diam")
                        st.number_input(FIBER_D_TOL_COL, min_value=0.0, step=0.1, format="%.2f", key="order_fiber_diam_tol")
                    with d2:
                        st.number_input("Main Coating Diameter (µm)", min_value=0.0, key="order_main_diam")
                        st.number_input(MAIN_D_TOL_COL, min_value=0.0, step=0.1, format="%.2f", key="order_main_diam_tol")
                    with d3:
                        st.number_input("Secondary Coating Diameter (µm)", min_value=0.0, key="order_sec_diam")
                        st.number_input(SEC_D_TOL_COL, min_value=0.0, step=0.1, format="%.2f", key="order_sec_diam_tol")
    
                    st.markdown("---")
                    t1, t2 = st.columns(2, vertical_alignment="bottom")
                    with t1:
                        st.number_input("Tension (g)", min_value=0.0, key="order_tension")
                    with t2:
                        st.number_input("Draw Speed (m/min)", min_value=0.0, key="order_speed")
    
                # 🧴 MATERIALS TAB
                with tab_materials:
                    st.caption("Coating names are loaded from config_coating.json.")
                    m1, m2 = st.columns(2, vertical_alignment="bottom")
                    with m1:
                        st.selectbox("Main Coating", options=[""] + COATING_OPTIONS, index=0, key="order_coating_main")
                    with m2:
                        st.selectbox("Secondary Coating", options=[""] + COATING_OPTIONS, index=0, key="order_coating_secondary")
    
                    tt1, tt2 = st.columns(2, vertical_alignment="bottom")
                    with tt1:
                        st.number_input(
                            "Main Coating Temperature (°C)",
                            value=__from_state("order_main_coat_temp_c", 25.0),
                            step=0.5,
                            format="%.1f",
                            key="order_main_coat_temp_c",
                        )
                    with tt2:
                        st.number_input(
                            "Secondary Coating Temperature (°C)",
                            value=__from_state("order_sec_coat_temp_c", 25.0),
                            step=0.5,
                            format="%.1f",
                            key="order_sec_coat_temp_c",
                        )
    
                # 💾 TEMPLATE TAB (only template save button)
                with tab_template:
                    st.markdown("#### 💾 Project Template (auto-fill defaults)")
                    tA, tB = st.columns([1.2, 2.8], vertical_alignment="center")
                    with tA:
                        save_tpl = st.form_submit_button(
                            "💾 Save / Update Template",
                            disabled=(not str(selected_project).strip()),
                            use_container_width=True,
                        )
                    with tB:
                        st.caption("Saves geometry + tiger/f2f + diameters+tolerances + tension + speed + coatings + temps + notes.")
    
                st.markdown("---")
                a1, a2 = st.columns([1, 1], vertical_alignment="center")
                with a1:
                    submit = st.form_submit_button("📤 Submit Draw Order", use_container_width=True)
                with a2:
                    cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
    
            # =========================================================
            # ✅ Scheduling UI OUTSIDE the form (so checkbox works instantly)
            # =========================================================
            st.markdown("---")
            st.markdown("#### 📅 Optional: schedule immediately (password protected)")
    
            schedule_now = st.checkbox("Schedule now", value=False, key="order_schedule_now_cb")
    
            sched_ok = False
            start_dt_new = None
            end_dt_new = None
    
            if schedule_now:
                pwd = st.text_input("Scheduling password", type="password", key="order_sched_pwd")
                if pwd == SCHEDULE_PASSWORD:
                    sched_ok = True
                    st.success("Password OK ✅")
                elif pwd.strip():
                    st.error("Wrong password ❌")
    
                default_date = pd.Timestamp.today().date()
                preset = st.radio(
                    "Preset",
                    ["All day (08:00–16:00)", "Before lunch (08:00–12:00)", "After lunch (12:00–16:00)"],
                    horizontal=True,
                    key="order_create_sched_preset",
                    label_visibility="collapsed",
                )
    
                if preset.startswith("All day"):
                    preset_start = dt.time(8, 0)
                    preset_duration = 8 * 60
                elif preset.startswith("Before lunch"):
                    preset_start = dt.time(8, 0)
                    preset_duration = 4 * 60
                else:
                    preset_start = dt.time(12, 0)
                    preset_duration = 4 * 60
    
                sA, sB, sC = st.columns([1.2, 1.0, 1.0], vertical_alignment="bottom")
                with sA:
                    sched_date_new = st.date_input("Schedule Date", value=default_date, key="order_create_sched_date")
                with sB:
                    sched_start_new = st.time_input("Start Time", value=preset_start, key="order_create_sched_start")
                with sC:
                    sched_dur_new = st.number_input(
                        "Duration (min)",
                        min_value=1,
                        step=5,
                        value=int(preset_duration),
                        key="order_create_sched_dur",
                    )
    
                start_dt_new = pd.to_datetime(f"{sched_date_new} {sched_start_new}")
                end_dt_new = start_dt_new + pd.to_timedelta(int(sched_dur_new), unit="m")
    
            # =========================================================
            # Handle Template Save
            # =========================================================
            if save_tpl:
                payload = {
                    FIBER_GEOMETRY_COL: safe_str_from_state("order_fiber_geometry_required", ""),
                    TIGER_CUT_COL: __from_state("order_tiger_cut_pct", 0.0),
                    OCT_F2F_COL: __from_state("order_oct_f2f_mm", 0.0),
    
                    "Fiber Diameter (µm)": __from_state("order_fiber_diam", 0.0),
                    FIBER_D_TOL_COL: __from_state("order_fiber_diam_tol", 0.0),
    
                    "Main Coating Diameter (µm)": __from_state("order_main_diam", 0.0),
                    MAIN_D_TOL_COL: __from_state("order_main_diam_tol", 0.0),
    
                    "Secondary Coating Diameter (µm)": __from_state("order_sec_diam", 0.0),
                    SEC_D_TOL_COL: __from_state("order_sec_diam_tol", 0.0),
    
                    "Tension (g)": __from_state("order_tension", 0.0),
                    "Draw Speed (m/min)": __from_state("order_speed", 0.0),
                    "Main Coating": safe_str_from_state("order_coating_main", ""),
                    "Secondary Coating": safe_str_from_state("order_coating_secondary", ""),
                    MAIN_COAT_TEMP_COL: __from_state("order_main_coat_temp_c", 25.0),
                    SEC_COAT_TEMP_COL: __from_state("order_sec_coat_temp_c", 25.0),
                    "Notes Default": safe_str_from_state("order_notes", ""),
                }
                ok_s, msg_s = save_or_update_template(selected_project, payload)
                (st.success if ok_s else st.warning)(msg_s)
    
            # Cancel
            if cancel and not submit:
                st.session_state["show_new_order_form"] = False
                st.rerun()
    
            # =========================================================
            # Submit order
            # =========================================================
            if submit:
                missing = []
                geom = str(st.session_state.get("order_fiber_geometry_required", "")).strip()
    
                if not str(st.session_state.get("order_preform_name", "")).strip():
                    missing.append("Preform Number")
                if not str(selected_project).strip():
                    missing.append("Fiber Project")
                if not str(st.session_state.get("order_opener", "")).strip():
                    missing.append("Order Opened By")
    
                length_required_val = to_float(st.session_state.get("order_length_required_required", 0.0), 0.0)
                if length_required_val <= 0:
                    missing.append("Required Length (m)")
    
                good_zones_val = int(st.session_state.get("order_good_zones_required", 1) or 1)
                if good_zones_val <= 0:
                    missing.append("Good Zones Count")
    
                if not geom:
                    missing.append(FIBER_GEOMETRY_COL)
    
                if geom == "TIGER - PM" and __from_state("order_tiger_cut_pct", 0.0) <= 0:
                    missing.append("Tiger Cut (%)")
                if geom == "Octagonal" and __from_state("order_oct_f2f_mm", 0.0) <= 0:
                    missing.append("Octagonal F2F (mm)")
    
                if missing:
                    st.error("Please fill required fields: " + ", ".join(missing))
                    st.stop()
    
                # sanitize geometry fields
                if geom != "TIGER - PM":
                    st.session_state["order_tiger_cut_pct"] = 0.0
                if geom != "Octagonal":
                    st.session_state["order_oct_f2f_mm"] = 0.0
    
                tiger_cut_val = __from_state("order_tiger_cut_pct", 0.0) if geom == "TIGER - PM" else 0.0
                oct_f2f_val = __from_state("order_oct_f2f_mm", 0.0) if geom == "Octagonal" else 0.0
    
                # tolerances
                fiber_diam_tol = __from_state("order_fiber_diam_tol", 0.0)
                main_diam_tol = __from_state("order_main_diam_tol", 0.0)
                sec_diam_tol = __from_state("order_sec_diam_tol", 0.0)
    
                order_data = {
                    "Status": "Pending",
                    "Priority": str(st.session_state.get("order_priority", "Normal")).strip(),
                    "Order Opener": str(st.session_state.get("order_opener", "")).strip(),
                    "Preform Number": str(st.session_state.get("order_preform_name", "")).strip(),
                    PROJECTS_COL: str(selected_project).strip(),
                    FIBER_GEOMETRY_COL: geom,
                    TIGER_CUT_COL: tiger_cut_val,
                    OCT_F2F_COL: oct_f2f_val,
                    "Timestamp": pd.Timestamp.now(),
    
                    "Fiber Diameter (µm)": __from_state("order_fiber_diam", 0.0),
                    FIBER_D_TOL_COL: float(fiber_diam_tol),
    
                    "Main Coating Diameter (µm)": __from_state("order_main_diam", 0.0),
                    MAIN_D_TOL_COL: float(main_diam_tol),
    
                    "Secondary Coating Diameter (µm)": __from_state("order_sec_diam", 0.0),
                    SEC_D_TOL_COL: float(sec_diam_tol),
    
                    "Tension (g)": __from_state("order_tension", 0.0),
                    "Draw Speed (m/min)": __from_state("order_speed", 0.0),
    
                    "Required Length (m) (for T&M+costumer)": float(length_required_val),
                    GOOD_ZONES_COL: int(good_zones_val),
    
                    "Main Coating": safe_str_from_state("order_coating_main", ""),
                    "Secondary Coating": safe_str_from_state("order_coating_secondary", ""),
                    MAIN_COAT_TEMP_COL: __from_state("order_main_coat_temp_c", 25.0),
                    SEC_COAT_TEMP_COL: __from_state("order_sec_coat_temp_c", 25.0),
    
                    "Notes": safe_str_from_state("order_notes", ""),
    
                    "Active CSV": "",
                    "Done CSV": "",
                    "Done Description": "",
                    "T&M Moved": False,
                    "T&M Moved Timestamp": "",
                }
    
                old = pd.read_csv(orders_file, keep_default_na=False) if os.path.exists(orders_file) else pd.DataFrame()
                new_df = pd.concat([old, pd.DataFrame([order_data])], ignore_index=True)
                new_df.to_csv(orders_file, index=False)
                new_idx = int(len(new_df) - 1)
    
                # Optional schedule now
                if st.session_state.get("order_schedule_now_cb", False):
                    if not sched_ok or start_dt_new is None or end_dt_new is None:
                        st.error("Order saved, but NOT scheduled (password missing/wrong or schedule details missing).")
                    else:
                        existing = pd.read_csv(SCHEDULE_FILE) if os.path.exists(SCHEDULE_FILE) else pd.DataFrame()
                        for c in schedule_required_cols:
                            if c not in existing.columns:
                                existing[c] = ""
                        existing = existing[schedule_required_cols]
    
                        priority = str(st.session_state.get("order_priority", "Normal")).strip()
                        preform_name = str(st.session_state.get("order_preform_name", "")).strip()
    
                        fd = __from_state("order_fiber_diam", 0.0)
                        md = __from_state("order_main_diam", 0.0)
                        sd = __from_state("order_sec_diam", 0.0)
    
                        diam_bits = []
                        s_fd = _fmt_pm(fd, fiber_diam_tol)
                        s_md = _fmt_pm(md, main_diam_tol)
                        s_sd = _fmt_pm(sd, sec_diam_tol)
                        if s_fd:
                            diam_bits.append(f"Fiber {s_fd}")
                        if s_md:
                            diam_bits.append(f"Coat1 {s_md}")
                        if s_sd:
                            diam_bits.append(f"Coat2 {s_sd}")
    
                        desc_lines = [
                            f"ORDER #{new_idx} | Priority: {priority}",
                            f"Fiber: {selected_project} | Geometry: {geom} | Preform: {preform_name}",
                            f"Required Length: {length_required_val} m | Good Zones Count: {int(good_zones_val)}",
                        ]
                        if diam_bits:
                            desc_lines.append("Diameters: " + " | ".join(diam_bits))
    
                        if geom == "TIGER - PM":
                            desc_lines.append(f"Tiger Cut: {tiger_cut_val:.1f}%")
                        if geom == "Octagonal":
                            desc_lines.append(f"Oct F2F: {oct_f2f_val:.2f} mm")
    
                        mtemp = __from_state("order_main_coat_temp_c", 0.0)
                        stemp = __from_state("order_sec_coat_temp_c", 0.0)
                        if mtemp > 0:
                            desc_lines.append(f"Main Coat Temp: {mtemp:.0f}°C")
                        if stemp > 0:
                            desc_lines.append(f"Sec Coat Temp: {stemp:.0f}°C")
    
                        notes = safe_str_from_state("order_notes", "")
                        if notes:
                            desc_lines.append(f"Notes: {notes}")
    
                        event_description = " | ".join([x for x in desc_lines if str(x).strip()])
    
                        new_event = pd.DataFrame([{
                            "Event Type": "Drawing",
                            "Start DateTime": start_dt_new,
                            "End DateTime": end_dt_new,
                            "Description": event_description,
                            "Recurrence": "None",
                        }])
    
                        pd.concat([existing, new_event], ignore_index=True).to_csv(SCHEDULE_FILE, index=False)
    
                        new_df.at[new_idx, "Status"] = "Scheduled"
                        new_df.to_csv(orders_file, index=False)
                        st.success("✅ Order saved + scheduled (status set to Scheduled).")
    
                st.session_state["show_new_order_form"] = False
                st.success("✅ Draw order submitted!")
                st.rerun()
    # ------------------ Tower Parts Tab ------------------
