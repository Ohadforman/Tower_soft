def render_tower_parts_tab(P):
    import os
    import base64
    import pandas as pd
    import streamlit as st
    from helpers.parts_inventory import (
        load_inventory as _raw_load_inventory,
        save_inventory,
        increment_part,
        ensure_inventory_file,
        ensure_general_tools_seed,
        load_locations as _raw_load_locations,
        save_locations,
        ensure_locations_file,
        decrement_part,
        set_part_quantity,
        is_non_consumable_part,
    )
    
    st.markdown(
        """
        <style>
          .tp-top-spacer{ height: 8px; }
          .tp-title{
            font-size: 1.62rem;
            font-weight: 900;
            margin: 0;
            padding-top: 4px;
            line-height: 1.2;
            color: rgba(236,248,255,0.98);
            text-shadow: 0 0 14px rgba(86,178,255,0.22);
          }
          .tp-sub{
            margin: 4px 0 8px 0;
            font-size: 0.92rem;
            color: rgba(188,224,248,0.88);
          }
          .tp-line{
            height: 1px;
            margin: 0 0 12px 0;
            background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
          }
          .tp-section{
            margin-top: 8px;
            margin-bottom: 8px;
            padding-left: 8px;
            border-left: 3px solid rgba(120,200,255,0.62);
            font-size: 1.04rem;
            font-weight: 820;
            color: rgba(230,246,255,0.98);
          }
          .tp-action-card{
            border: 1px solid rgba(128,206,255,0.22);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(14,32,56,0.26), rgba(8,16,28,0.20));
            padding: 10px 12px;
            margin-bottom: 12px;
          }
          .tp-action-help{
            color: rgba(194,228,248,0.90);
            font-size: 0.84rem;
            margin-top: 6px;
          }
          .tp-green-text{
            color: rgba(126, 255, 190, 0.98);
            font-size: 0.88rem;
            font-weight: 650;
            margin: 4px 0 8px 0;
            text-shadow: 0 0 8px rgba(46, 208, 132, 0.20);
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
          div[data-testid="stButton"] > button:disabled{
            opacity: 0.78 !important;
            color: rgba(212,238,255,0.92) !important;
            border-color: rgba(128,206,255,0.32) !important;
            background: linear-gradient(180deg, rgba(24,62,102,0.52), rgba(12,34,64,0.48)) !important;
            box-shadow: 0 4px 10px rgba(8,30,58,0.20) !important;
          }
          div[data-baseweb="tag"],
          span[data-baseweb="tag"],
          div[data-baseweb="select"] div[data-baseweb="tag"],
          div[data-baseweb="select"] span[data-baseweb="tag"]{
            background: linear-gradient(180deg, rgba(72,160,248,0.94), rgba(38,102,182,0.92)) !important;
            border: 1px solid rgba(178,232,255,0.80) !important;
            color: rgba(244,252,255,0.99) !important;
            box-shadow: 0 0 10px rgba(74,170,255,0.24) !important;
          }
          div[data-baseweb="tag"] *,
          span[data-baseweb="tag"] *,
          div[data-baseweb="select"] div[data-baseweb="tag"] *,
          div[data-baseweb="select"] span[data-baseweb="tag"] *{
            color: rgba(244,252,255,0.99) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tp-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-title">🛠️ Tower Parts Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-sub">Track parts orders, update statuses, archive installed items, and browse docs.</div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-line"></div>', unsafe_allow_html=True)
    
    ORDER_FILE = P.parts_orders_csv
    inventory_file = P.parts_inventory_csv
    locations_file = P.parts_locations_csv
    coating_stock_file = P.coating_stock_json
    containers_csv = P.tower_containers_csv
    PARTS_DIRECTORY = P.parts_dir

    def _mtime(path: str) -> float:
        try:
            return float(os.path.getmtime(path))
        except Exception:
            return 0.0

    @st.cache_data(show_spinner=False)
    def _read_csv_cached(path: str, keep_default_na: bool, file_mtime: float) -> pd.DataFrame:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=keep_default_na)

    @st.cache_data(show_spinner=False)
    def _load_inventory_cached(path: str, file_mtime: float) -> pd.DataFrame:
        return _raw_load_inventory(path)

    def load_inventory(path: str) -> pd.DataFrame:
        return _load_inventory_cached(path, _mtime(path))

    @st.cache_data(show_spinner=False)
    def _load_locations_cached(path: str, file_mtime: float) -> pd.DataFrame:
        return _raw_load_locations(path)

    def load_locations(path: str) -> pd.DataFrame:
        return _load_locations_cached(path, _mtime(path))

    @st.cache_data(show_spinner=False)
    def _manual_pdf_signature_cached(manuals_dir: str, dir_mtime: float) -> tuple:
        sig = []
        if os.path.isdir(manuals_dir):
            for fn in sorted(os.listdir(manuals_dir)):
                if fn.lower().endswith(".pdf"):
                    fp = os.path.join(manuals_dir, fn)
                    sig.append((fn, _mtime(fp)))
        return tuple(sig)
    
    # ✅ Status rename (Needed -> Opened)
    STATUS_ORDER = ["Opened", "Approved", "Ordered", "Shipped", "Received", "Installed"]
    ITEM_TYPE_OPTIONS = ["Part", "Tool", "Consumable"]
    
    # ✅ Single description field (remove Purpose completely)
    BASE_COLUMNS = [
        "Status", "Part Name", "Serial Number",
        "Project Name", "Details",
        "Opened By",
        "Approved", "Approved By", "Approval Date",
        "Ordered By", "Date Ordered", "Company"
    ]
    
    # ---------------- Load / init ----------------
    if os.path.exists(ORDER_FILE):
        orders_df = _read_csv_cached(ORDER_FILE, False, _mtime(ORDER_FILE))
    else:
        orders_df = pd.DataFrame(columns=BASE_COLUMNS)
    
    orders_df.columns = orders_df.columns.str.strip()
    
    # Backward compat: ensure columns exist + map old "Needed" to "Opened"
    for col in BASE_COLUMNS:
        if col not in orders_df.columns:
            orders_df[col] = ""
    if "Inventory Synced" not in orders_df.columns:
        orders_df["Inventory Synced"] = ""
    
    # Drop old Purpose if exists
    orders_df = orders_df.drop(columns=["Purpose"], errors="ignore")

    # Remove truly blank rows first (prevents empty lines from appearing as fake "Opened" rows).
    _raw_status = orders_df["Status"].fillna("").astype(str).str.strip()
    _row_has_content = (
        _raw_status.ne("")
        | orders_df["Part Name"].fillna("").astype(str).str.strip().ne("")
        | orders_df["Serial Number"].fillna("").astype(str).str.strip().ne("")
        | orders_df["Project Name"].fillna("").astype(str).str.strip().ne("")
        | orders_df["Details"].fillna("").astype(str).str.strip().ne("")
        | orders_df["Opened By"].fillna("").astype(str).str.strip().ne("")
        | orders_df["Company"].fillna("").astype(str).str.strip().ne("")
    )
    orders_df = orders_df[_row_has_content].copy()

    orders_df["Status"] = orders_df["Status"].fillna("").astype(str).str.strip()
    orders_df["Status"] = orders_df["Status"].replace({"Needed": "Opened", "needed": "Opened"})
    
    # Unknown / empty -> Opened
    orders_df["Status"] = orders_df["Status"].apply(lambda s: s if s in STATUS_ORDER else "Opened")

    ensure_inventory_file(inventory_file)
    ensure_locations_file(locations_file)
    seeded_tools_count = ensure_general_tools_seed(inventory_file)
    if seeded_tools_count > 0:
        st.info(f"Seeded {seeded_tools_count} General Tools template rows in inventory.")

    def _normalize_received_sync_state(df_orders: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        marked_pending = 0
        moved_installed = 0
        out = df_orders.copy()
        inv_now = load_inventory(inventory_file)

        def _has_inventory_location(part_name: str, serial_number: str) -> bool:
            pn = str(part_name or "").strip().lower()
            sn = str(serial_number or "").strip().lower()
            if not pn:
                return False
            m = inv_now[inv_now["Part Name"].astype(str).str.strip().str.lower().eq(pn)].copy()
            if sn:
                m_sn = m[m["Serial Number"].astype(str).str.strip().str.lower().eq(sn)]
                if not m_sn.empty:
                    return m_sn["Location"].astype(str).str.strip().ne("").any()
            if m.empty:
                return False
            return m["Location"].astype(str).str.strip().ne("").any()

        for i, r in out.iterrows():
            status = str(r.get("Status", "")).strip().lower()
            inv_synced = str(r.get("Inventory Synced", "")).strip().lower()
            if status == "received":
                part_name = str(r.get("Part Name", "")).strip()
                serial_number = str(r.get("Serial Number", "")).strip()
                if _has_inventory_location(part_name, serial_number):
                    out.at[i, "Status"] = "Installed"
                    out.at[i, "Inventory Synced"] = ""
                    moved_installed += 1
                elif inv_synced != "pending":
                    out.at[i, "Inventory Synced"] = "Pending"
                    marked_pending += 1
            elif status != "received":
                out.at[i, "Inventory Synced"] = ""
        return out, marked_pending, moved_installed

    orders_df, pending_new_count, moved_installed_count = _normalize_received_sync_state(orders_df)
    if pending_new_count > 0 or moved_installed_count > 0:
        orders_df.to_csv(ORDER_FILE, index=False)
    if pending_new_count > 0:
        st.info(f"{pending_new_count} received order(s) are waiting for intake/location.")
    if moved_installed_count > 0:
        st.success(f"{moved_installed_count} received order(s) moved to Installed (location found in inventory).")
    
    # ---------------- Projects list (match 📦 Order Draw) ----------------
    PROJECTS_FILE = P.projects_fiber_csv
    PROJECTS_COL = "Fiber Project"
    
    project_options = ["None"]
    try:
        if os.path.exists(PROJECTS_FILE):
            projects_df = _read_csv_cached(PROJECTS_FILE, False, _mtime(PROJECTS_FILE))
            projects_df.columns = [str(c).strip() for c in projects_df.columns]
            if PROJECTS_COL in projects_df.columns:
                vals = (
                    projects_df[PROJECTS_COL]
                    .astype(str)
                    .fillna("")
                    .map(lambda x: x.strip())
                )
                vals = [v for v in vals.tolist() if v and v.lower() != "nan"]
                project_options += sorted(list(pd.Series(vals).unique()))
    except Exception:
        pass
    
    # =========================
    # TABLE (FIRST)
    # =========================
    st.markdown('<div class="tp-section">📋 Orders Table</div>', unsafe_allow_html=True)
    
    column_order = [
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
    ]
    for col in column_order:
        if col not in orders_df.columns:
            orders_df[col] = ""
    
    # Sort by status
    tmp = orders_df.copy()
    tmp["__status_sort__"] = pd.Categorical(tmp["Status"], categories=STATUS_ORDER, ordered=True)
    tmp = tmp.sort_values(["__status_sort__", "Part Name"], na_position="last").drop(columns="__status_sort__")
    
    # Color status cell only (cleaner dark-friendly colors)
    def highlight_status(row):
        color_map = {
            "Opened": "background-color: rgba(255,102,102,0.28); color: #ffd4d4; font-weight: 800;",
            "Approved": "background-color: rgba(105,240,174,0.24); color: #c8ffd8; font-weight: 800;",
            "Ordered": "background-color: rgba(255,214,102,0.24); color: #ffe9b8; font-weight: 800;",
            "Shipped": "background-color: rgba(126,182,255,0.24); color: #d5e8ff; font-weight: 800;",
            "Received": "background-color: rgba(92,214,122,0.30); color: #d7ffe1; font-weight: 800;",
            "Installed": "background-color: rgba(190,198,210,0.22); color: #ecf0f6; font-weight: 800;",
        }
        s = str(row.get("Status", "")).strip()
        styles = [""] * len(row)
        # Status column index after adding row number
        if "Status" in row.index:
            styles[list(row.index).index("Status")] = color_map.get(s, "")
        return styles
    
    if not tmp.empty:
        tmp_display = tmp[column_order].fillna("").copy()
        tmp_display.insert(0, "#", range(1, len(tmp_display) + 1))
        styled = (
            tmp_display.style
            .apply(highlight_status, axis=1)
            .set_properties(subset=["#"], **{"color": "rgba(180,210,230,0.90)", "font-weight": "700"})
        )
        table_height = max(170, min(420, 48 + 36 * len(tmp_display)))
        st.dataframe(
            styled,
            height=table_height,
            use_container_width=True,
        )
    else:
        st.info("No orders have been placed yet.")
    
    st.divider()
    
    # =========================
    # CLEAN POP AREA (AFTER TABLE)
    # =========================
    st.markdown('<div class="tp-section">✍️ Manage Orders</div>', unsafe_allow_html=True)

    if "parts_manage_action" not in st.session_state:
        st.session_state["parts_manage_action"] = ""

    st.markdown('<div class="tp-action-card">', unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="small")
    with a1:
        if st.button(
            "➕ Open New Order",
            use_container_width=True,
            type="primary" if st.session_state["parts_manage_action"] == "Add New Order" else "secondary",
            key="parts_open_add_btn",
        ):
            if st.session_state["parts_manage_action"] == "Add New Order":
                st.session_state["parts_manage_action"] = ""
            else:
                st.session_state["parts_manage_action"] = "Add New Order"
    with a2:
        if st.button(
            "🛠️ Open Edit Order",
            use_container_width=True,
            type="primary" if st.session_state["parts_manage_action"] == "Update Existing Order" else "secondary",
            key="parts_open_edit_btn",
        ):
            if st.session_state["parts_manage_action"] == "Update Existing Order":
                st.session_state["parts_manage_action"] = ""
            else:
                st.session_state["parts_manage_action"] = "Update Existing Order"
    st.markdown(
        f"<div class='tp-action-help'>Active panel: <b>{st.session_state['parts_manage_action'] or 'None'}</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    action = st.session_state["parts_manage_action"]
    
    if action == "":
        st.info("Choose `Open New Order` or `Open Edit Order` to expand a panel.")

    # ---------- Add New ----------
    if action == "Add New Order":
        st.markdown("#### ➕ Add New Order")
        with st.container(border=True):
            with st.form("add_new_order_form", clear_on_submit=True, enter_to_submit=False):
                c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    
                with c1:
                    part_name = st.text_input("Part Name")
                    serial_number = st.text_input("Serial Number")
                    status = st.selectbox("Status", STATUS_ORDER, index=0)
    
                with c2:
                    opened_by = st.text_input("Opened By")
                    selected_project = st.selectbox("Fiber Project", project_options)
                    company = st.text_input("Company (optional)")
    
                with c3:
                    approved = st.selectbox("Approved", ["No", "Yes"], index=0)
                    approved_by = st.text_input("Approved By (optional)")
                    approval_date = st.date_input("Approval Date", value=pd.Timestamp.today())
    
                details = st.text_area("Details", height=120)
    
                save = st.form_submit_button("💾 Save Order", use_container_width=True)
    
                if save:
                    if not part_name.strip():
                        st.error("Part Name is required.")
                    else:
                        new_row = {
                            "Status": status,
                            "Part Name": part_name.strip(),
                            "Serial Number": serial_number.strip(),
                            "Project Name": "" if selected_project == "None" else str(selected_project),
                            "Details": details.strip(),
                            "Opened By": opened_by.strip(),
                            "Company": company.strip(),
                            "Approved": approved,
                            "Approved By": approved_by.strip(),
                            "Approval Date": approval_date.strftime("%Y-%m-%d") if approved == "Yes" else "",
                            "Ordered By": "",
                            "Date Ordered": "",
                        }
                        orders_df = pd.concat([orders_df, pd.DataFrame([new_row])], ignore_index=True)
                        orders_df.to_csv(ORDER_FILE, index=False)
                        st.success("✅ Order saved.")
    
    # ---------- Update Existing ----------
    elif action == "Update Existing Order":
        st.markdown("#### 🛠️ Update Existing Order")
        with st.container(border=True):
            if orders_df.empty:
                st.warning("No orders to update.")
            else:
                labels = (orders_df["Part Name"].astype(str).fillna("") + "  |  " +
                          orders_df["Serial Number"].astype(str).fillna(""))
                label_to_idx = {labels.iloc[i]: i for i in range(len(labels))}
                selected_label = st.selectbox("Select an order", list(label_to_idx.keys()), key="order_update_select")
                order_index = label_to_idx[selected_label]
                cur = orders_df.loc[order_index].to_dict()
    
                with st.form("update_order_form", enter_to_submit=False):
                    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    
                    with c1:
                        updated_part_name = st.text_input("Part Name", value=str(cur.get("Part Name", "")))
                        updated_serial_number = st.text_input("Serial Number", value=str(cur.get("Serial Number", "")))
                        cur_status = str(cur.get("Status", "Opened")).strip()
                        new_status = st.selectbox(
                            "Status",
                            STATUS_ORDER,
                            index=STATUS_ORDER.index(cur_status) if cur_status in STATUS_ORDER else 0,
                        )
    
                    with c2:
                        cur_proj = str(cur.get("Project Name", ""))
                        updated_project = st.selectbox(
                            "Fiber Project",
                            project_options,
                            index=project_options.index(cur_proj) if cur_proj in project_options else 0,
                        )
                        updated_opened_by = st.text_input("Opened By", value=str(cur.get("Opened By", "")))
                        company = st.text_input("Company", value=str(cur.get("Company", "")))
    
                    with c3:
                        approved_value = str(cur.get("Approved", "No"))
                        approved = st.selectbox("Approved", ["No", "Yes"], index=0 if approved_value != "Yes" else 1)
                        approved_by = st.text_input("Approved By", value=str(cur.get("Approved By", "")))
    
                        date_ordered_raw = str(cur.get("Date Ordered", ""))
                        date_ordered_dt = pd.to_datetime(date_ordered_raw, errors="coerce")
                        if pd.isna(date_ordered_dt):
                            date_ordered_dt = pd.Timestamp.today()
                        date_ordered = st.date_input("Date Ordered", value=date_ordered_dt)
    
                        approval_raw = str(cur.get("Approval Date", ""))
                        approval_dt = pd.to_datetime(approval_raw, errors="coerce")
                        if pd.isna(approval_dt):
                            approval_dt = pd.Timestamp.today()
                        approval_date = st.date_input("Approval Date", value=approval_dt)
    
                    details = st.text_area("Details", value=str(cur.get("Details", "")), height=120)
                    ordered_by = st.text_input("Ordered By", value=str(cur.get("Ordered By", "")))
    
                    do_update = st.form_submit_button("✅ Update Order", use_container_width=True)
    
                    if do_update:
                        orders_df.at[order_index, "Part Name"] = updated_part_name.strip()
                        orders_df.at[order_index, "Serial Number"] = updated_serial_number.strip()
                        orders_df.at[order_index, "Status"] = new_status
    
                        orders_df.at[order_index, "Project Name"] = "" if updated_project == "None" else str(updated_project)
                        orders_df.at[order_index, "Opened By"] = updated_opened_by.strip()
    
                        orders_df.at[order_index, "Details"] = details.strip()
                        orders_df.at[order_index, "Company"] = company.strip()
                        orders_df.at[order_index, "Ordered By"] = ordered_by.strip()
    
                        orders_df.at[order_index, "Approved"] = approved
                        orders_df.at[order_index, "Approved By"] = approved_by.strip()
                        orders_df.at[order_index, "Approval Date"] = approval_date.strftime("%Y-%m-%d") if approved == "Yes" else ""
    
                        orders_df.at[order_index, "Date Ordered"] = date_ordered.strftime("%Y-%m-%d")

                        if new_status.lower() == "received":
                            orders_df.at[order_index, "Inventory Synced"] = "Pending"
                        elif new_status.lower() != "received":
                            orders_df.at[order_index, "Inventory Synced"] = ""

                        orders_df.to_csv(ORDER_FILE, index=False)
                        st.success("✅ Order updated.")

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                st.caption("Danger zone")
                if st.button("🗑️ Delete This Order", use_container_width=True, key="delete_part_inside_edit"):
                    orders_df = orders_df.drop(index=order_index).reset_index(drop=True)
                    orders_df.to_csv(ORDER_FILE, index=False)
                    st.success("✅ Deleted.")
    
    st.divider()

    # =========================
    # Parts Inventory Center (collapsed)
    # =========================
    show_inventory_center = st.toggle(
        "📦 Open Inventory Center",
        value=False,
        key="parts_inventory_open_toggle",
    )
    if show_inventory_center:
        st.caption("Manage part stock and location intake for received orders. Intake with location auto-moves order to Installed.")

        def _safe_mtime(p: str) -> float:
            try:
                return float(os.path.getmtime(p))
            except Exception:
                return 0.0

        @st.cache_data(show_spinner=False)
        def _load_tower_components_cached(maintenance_dir: str, signature: tuple) -> list:
            comps = set()
            if os.path.isdir(maintenance_dir):
                for fn in sorted(os.listdir(maintenance_dir)):
                    if not fn.lower().endswith((".xlsx", ".xls", ".csv")):
                        continue
                    if "log" in fn.lower() or fn.startswith("_"):
                        continue
                    fp = os.path.join(maintenance_dir, fn)
                    try:
                        df = pd.read_csv(fp, keep_default_na=False) if fn.lower().endswith(".csv") else pd.read_excel(fp)
                    except Exception:
                        continue
                    for col in ["Equipment", "Component"]:
                        if col in df.columns:
                            vals = df[col].astype(str).fillna("").map(lambda x: x.strip())
                            for v in vals.tolist():
                                if v and v.lower() != "nan":
                                    comps.add(v)
            return sorted(list(comps))

        def _load_tower_components() -> list:
            mdir = P.maintenance_dir
            sig = []
            if os.path.isdir(mdir):
                for fn in sorted(os.listdir(mdir)):
                    if not fn.lower().endswith((".xlsx", ".xls", ".csv")):
                        continue
                    if "log" in fn.lower() or fn.startswith("_"):
                        continue
                    fp = os.path.join(mdir, fn)
                    sig.append((fn, _safe_mtime(fp)))
            return _load_tower_components_cached(mdir, tuple(sig))

        def _sync_coating_from_consumables() -> int:
            import json

            warehouse = {}
            if os.path.exists(coating_stock_file):
                try:
                    with open(coating_stock_file, "r", encoding="utf-8") as f:
                        raw = json.load(f) or {}
                    for k, v in raw.items():
                        try:
                            warehouse[str(k).strip()] = float(v)
                        except Exception:
                            warehouse[str(k).strip()] = 0.0
                except Exception:
                    pass

            container_sum = {}
            if os.path.exists(containers_csv):
                try:
                    cdf = pd.read_csv(containers_csv, keep_default_na=False)
                    if not cdf.empty:
                        row = cdf.iloc[-1]
                        for lab in ["A", "B", "C", "D"]:
                            t = str(row.get(f"{lab}_type", "")).strip()
                            lv = pd.to_numeric(row.get(f"{lab}_level_kg", 0.0), errors="coerce")
                            lvl = 0.0 if pd.isna(lv) else float(lv)
                            if t:
                                container_sum[t] = float(container_sum.get(t, 0.0)) + max(0.0, lvl)
                except Exception:
                    pass

            all_types = sorted(set(list(warehouse.keys()) + list(container_sum.keys())))
            touched = 0
            for ctype in all_types:
                total_kg = max(0.0, float(warehouse.get(ctype, 0.0)) + float(container_sum.get(ctype, 0.0)))
                set_part_quantity(
                    inventory_file,
                    f"Coating::{ctype}",
                    qty=total_kg,
                    component="Consumables",
                    location="Consumables",
                    location_serial="COAT-STOCK",
                    notes="Auto sync from consumables (warehouse + containers)",
                )
                touched += 1
            return touched

        source_sig = (_safe_mtime(coating_stock_file), _safe_mtime(containers_csv))
        last_sig = st.session_state.get("parts_coating_sync_sig")
        synced_types = 0
        if last_sig != source_sig:
            synced_types = _sync_coating_from_consumables()
            st.session_state["parts_coating_sync_sig"] = source_sig
        if synced_types > 0:
            st.caption(f"🧪 Coating quantities synced dynamically ({synced_types} types, KG).")

        active_locations_df = load_locations(locations_file)
        active_locations_df = active_locations_df[
            active_locations_df["Active"].astype(str).str.strip().str.lower().ne("no")
        ].copy()
        location_options = sorted(
            [str(x).strip() for x in active_locations_df["Location Name"].tolist() if str(x).strip()]
        )
        loc_serial_map = {}
        for _, lr in active_locations_df.iterrows():
            ln = str(lr.get("Location Name", "")).strip()
            ls = str(lr.get("Location Serial", "")).strip()
            if ln:
                loc_serial_map[ln] = ls

        def _upsert_storage_location(loc_name: str, loc_serial: str = "") -> None:
            ln = str(loc_name).strip()
            if not ln:
                return
            ls = str(loc_serial).strip()
            ldf = load_locations(locations_file)
            now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if ldf.empty:
                ldf = pd.DataFrame(columns=["Location Name", "Location Serial", "Description", "Active", "Last Updated"])
            mask = ldf["Location Name"].astype(str).str.strip().str.lower().eq(ln.lower())
            if mask.any():
                idx = ldf[mask].index[0]
                if ls:
                    ldf.at[idx, "Location Serial"] = ls
                if not str(ldf.at[idx, "Active"]).strip():
                    ldf.at[idx, "Active"] = "Yes"
                ldf.at[idx, "Last Updated"] = now_ts
            else:
                ldf = pd.concat(
                    [
                        ldf,
                        pd.DataFrame(
                            [
                                {
                                    "Location Name": ln,
                                    "Location Serial": ls,
                                    "Description": "Auto-added from Manual Inventory Update",
                                    "Active": "Yes",
                                    "Last Updated": now_ts,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            save_locations(locations_file, ldf)

        def _manual_component_guess(manual_name: str) -> str:
            nm = str(manual_name or "").strip().lower()
            if "furnace" in nm:
                return "63mm Furnace System"
            if "capstan" in nm:
                return "Capstan System"
            if "winder" in nm:
                return "Take Up Winder"
            if "coating" in nm:
                return "Wet-on-Dry Coating System"
            if "uv" in nm or "ultra violet" in nm:
                return "UV Curing System"
            if "clean air" in nm:
                return "Clean Air System"
            if "cane puller" in nm:
                return "Cane Puller"
            if "preform" in nm:
                return "Preform Feed Assembly"
            if "guide pulley" in nm or "tension gauge" in nm:
                return "Guide Pulley / Tension Gauge"
            return "Manual BOM"

        @st.cache_data(show_spinner=False)
        def _extract_manual_bom_catalog(manuals_dir: str, signature: tuple):
            import glob
            import fitz
            import re

            def _clean(s):
                return re.sub(r"\s+", " ", str(s or "")).strip()

            def _is_num(s):
                return bool(re.fullmatch(r"\d+(\.\d+)?", _clean(s)))

            def _is_pn(s):
                # Accept wide engineering PN formats:
                # 286491, EE0031166, EL82610, EE006003.EE006064, A12-BC34
                t = _clean(s).upper().rstrip(".")
                if not t or " " in t:
                    return False
                if not re.fullmatch(r"[A-Z0-9][A-Z0-9._/\-]*", t):
                    return False
                if not re.search(r"\d", t):
                    return False
                # Avoid matching short pure numbers like ITEM/QTY cells.
                if re.fullmatch(r"\d{1,2}", t):
                    return False
                return True

            out = []
            key_pat = re.compile(r"PARTS?\s+LIST|BILL OF MATERIALS|BOM|PART NUMBER|ITEM", re.IGNORECASE)
            for pdf in sorted(glob.glob(os.path.join(manuals_dir, "*.pdf"))):
                mname = os.path.basename(pdf)
                try:
                    doc = fitz.open(pdf)
                except Exception:
                    continue
                for pidx in range(len(doc)):
                    txt = doc.load_page(pidx).get_text("text") or ""
                    if not key_pat.search(txt):
                        continue
                    tokens = [_clean(x) for x in txt.splitlines() if _clean(x)]
                    i = 0
                    while i + 3 < len(tokens):
                        d, pn, qty, item = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
                        if _is_pn(pn) and _is_num(qty) and _is_num(item) and len(d) >= 3 and not _is_num(d):
                            out.append((d, pn.rstrip("."), mname, float(qty)))
                            i += 4
                            continue
                        i += 1
                doc.close()
            # dedup + qty aggregate (keep max qty per unique part/PN to avoid duplicate drawing rows)
            agg = {}
            for p, pn, mn, q in out:
                key = (p.strip().lower(), pn.strip().lower())
                if key not in agg:
                    agg[key] = (p.strip(), pn.strip(), mn, float(q))
                else:
                    old = agg[key]
                    agg[key] = (old[0], old[1], old[2], max(float(old[3]), float(q)))
            return list(agg.values())

        manuals_dir = os.path.join(P.root_dir, "manuals")
        manual_sig = _manual_pdf_signature_cached(manuals_dir, _mtime(manuals_dir))
        manual_catalog = _extract_manual_bom_catalog(manuals_dir, manual_sig) if manual_sig else []

        inv_df = load_inventory(inventory_file)
        sheet_components = _load_tower_components()
        inv_components = sorted(list({str(x).strip() for x in inv_df.get("Component", pd.Series([], dtype=str)).tolist() if str(x).strip()}))
        component_options = sorted(list({*sheet_components, *inv_components, "Tower Parts", "Consumables", "General Tools"}))

        # Intake queue from received orders not yet organized in inventory.
        received_pending = orders_df[
            orders_df["Status"].astype(str).str.strip().str.lower().eq("received")
            & orders_df["Inventory Synced"].astype(str).str.strip().str.lower().ne("yes")
        ].copy()
        if not received_pending.empty:
            st.warning("Received orders waiting for intake. Assign location and confirm stock add.")
            show_cols = [c for c in ["Part Name", "Serial Number", "Project Name", "Company", "Details"] if c in received_pending.columns]
            st.dataframe(received_pending[show_cols], use_container_width=True, height=180)

            rp_labels = []
            rp_map = {}
            for idx, rr in received_pending.iterrows():
                label = f"{str(rr.get('Part Name','')).strip()} | SN:{str(rr.get('Serial Number','')).strip() or '-'} | row:{idx}"
                rp_labels.append(label)
                rp_map[label] = int(idx)

            i1, i2, i3, i4, i5 = st.columns([1.2, 1.0, 0.8, 1.0, 1.1])
            with i1:
                intake_pick = st.selectbox("Received order", options=[""] + rp_labels, key="parts_intake_pick")
            with i2:
                intake_loc = st.selectbox("Store in location", options=[""] + location_options, key="parts_intake_loc")
            with i3:
                intake_qty = st.number_input("Qty", min_value=0.01, max_value=10000.0, value=1.0, step=0.1, key="parts_intake_qty")
            with i4:
                intake_override_sn = st.text_input("Serial override", key="parts_intake_sn")
            with i5:
                intake_component = st.selectbox("Component", options=component_options or ["Tower Parts"], key="parts_intake_component")

            if st.button("✅ Intake + Locate Received Order", use_container_width=True, key="parts_intake_apply_btn", type="primary"):
                if not intake_pick or not intake_loc:
                    st.error("Select received order and storage location.")
                else:
                    ridx = rp_map[intake_pick]
                    row = orders_df.loc[ridx]
                    part_name = str(row.get("Part Name", "")).strip()
                    serial_num = intake_override_sn.strip() or str(row.get("Serial Number", "")).strip()
                    increment_part(
                        inventory_file,
                        part_name,
                        qty=float(intake_qty),
                        component=intake_component,
                        serial_number=serial_num,
                        location=intake_loc,
                        location_serial=loc_serial_map.get(intake_loc, ""),
                        notes="Intake from received order",
                    )
                    orders_df.at[ridx, "Status"] = "Installed"
                    orders_df.at[ridx, "Inventory Synced"] = "Yes"
                    orders_df.to_csv(ORDER_FILE, index=False)
                    st.success("Received order organized, added to inventory, and moved to Installed.")

        inv_df = load_inventory(inventory_file)
        if inv_df.empty:
            st.info("Inventory is empty. Add rows below or intake received orders.")

        with st.expander("🔎 Inventory Finder", expanded=False):
            st.caption("Unified view: each part shows stock and mounted quantities.")
            fc1, fc2, fc3, fc4 = st.columns([1.6, 1.2, 0.8, 0.8])
            with fc1:
                filter_components = st.multiselect(
                    "Filter by component",
                    options=component_options,
                    default=[],
                    key="parts_filter_components",
                )
            with fc2:
                part_query = st.text_input("Part search", key="parts_filter_part_query", placeholder="type part name...")
            with fc3:
                only_missing_loc = st.checkbox("Only missing location", value=False, key="parts_filter_missing_loc")
            with fc4:
                only_tools = st.checkbox("Tools only", value=False, key="parts_filter_tools_only")

        # Build one-row-per-part(+serial) finder table with split quantities.
        finder_src = inv_df.copy()
        finder_src["Part Name"] = finder_src["Part Name"].astype(str).fillna("").str.strip()
        finder_src["Item Type"] = finder_src.get("Item Type", "").astype(str).fillna("").str.strip()
        finder_src["Serial Number"] = finder_src["Serial Number"].astype(str).fillna("").str.strip()
        finder_src["Component"] = finder_src["Component"].astype(str).fillna("").str.strip()
        finder_src["Location"] = finder_src["Location"].astype(str).fillna("").str.strip()
        finder_src["Location Serial"] = finder_src["Location Serial"].astype(str).fillna("").str.strip()
        finder_src["Quantity"] = pd.to_numeric(finder_src["Quantity"], errors="coerce").fillna(0.0)
        finder_src["Min Level"] = pd.to_numeric(finder_src["Min Level"], errors="coerce").fillna(0.0)
        finder_src["Notes"] = finder_src["Notes"].astype(str).fillna("").str.strip()
        finder_src["_is_mounted"] = (
            finder_src["Location"].str.lower().eq("mounted")
            | finder_src["Component"].str.lower().eq("mounted")
        )

        # Apply filters at source level first (more reliable than filtering aggregated labels).
        if filter_components:
            finder_src = finder_src[finder_src["Component"].astype(str).isin(filter_components)].copy()
        if part_query.strip():
            q = part_query.strip().lower()
            finder_src = finder_src[
                finder_src["Part Name"].astype(str).str.lower().str.contains(q, na=False)
                | finder_src["Serial Number"].astype(str).str.lower().str.contains(q, na=False)
            ].copy()
        if only_missing_loc:
            finder_src = finder_src[finder_src["Location"].astype(str).str.strip().eq("")].copy()
        if only_tools:
            finder_src = finder_src[finder_src["Item Type"].astype(str).str.strip().str.lower().eq("tool")].copy()

        def _summarize_part_group(g):
            g_non_m = g[~g["_is_mounted"]]
            g_m = g[g["_is_mounted"]]
            qty_stock = float(g_non_m["Quantity"].sum()) if not g_non_m.empty else 0.0
            qty_mounted = float(g_m["Quantity"].sum()) if not g_m.empty else 0.0
            qty_total = float(g["Quantity"].sum())
            comps = [x for x in g_non_m["Component"].tolist() if str(x).strip() and str(x).strip().lower() != "mounted"]
            if not comps:
                comps = [x for x in g["Component"].tolist() if str(x).strip() and str(x).strip().lower() != "mounted"]
            comp_show = ", ".join(sorted(set(comps)))[:80]
            locs = [x for x in g_non_m["Location"].tolist() if str(x).strip()]
            loc_serials = [x for x in g_non_m["Location Serial"].tolist() if str(x).strip()]
            notes = [x for x in g["Notes"].tolist() if str(x).strip()]
            return pd.Series(
                {
                    "Item Type": (
                        str(g["Item Type"].dropna().astype(str).iloc[0]).strip()
                        if ("Item Type" in g.columns and not g.empty)
                        else "Part"
                    ),
                    "Component": comp_show,
                    "Location": ", ".join(sorted(set(locs)))[:80],
                    "Location Serial": ", ".join(sorted(set(loc_serials)))[:80],
                    "Qty Stock": qty_stock,
                    "Qty Mounted": qty_mounted,
                    "Quantity": qty_total,
                    "Min Level": float(g["Min Level"].max()) if not g.empty else 0.0,
                    "Notes": (" | ".join(sorted(set(notes))))[:160],
                }
            )

        finder_df = (
            finder_src
            .groupby(["Part Name", "Serial Number"], as_index=False, dropna=False)
            .apply(_summarize_part_group)
            .reset_index(drop=True)
        )
        finder_df = finder_df.sort_values(
            ["Component", "Part Name", "Serial Number"],
            ascending=[True, True, True],
            na_position="last",
        )
        st.caption(f"Finder rows: {len(finder_df)}")
        st.dataframe(
            finder_df[
                [
                    c
                    for c in [
                        "Part Name",
                        "Item Type",
                        "Component",
                        "Serial Number",
                        "Location",
                        "Location Serial",
                        "Qty Stock",
                        "Qty Mounted",
                        "Quantity",
                        "Min Level",
                        "Notes",
                    ]
                    if c in finder_df.columns
                ]
            ],
            use_container_width=True,
            height=220,
        )

        # Maintenance reservations visibility (read-only): stock already deducted while ACTIVE.
        with st.expander("🧷 Maintenance Reservations (Execution Hold)", expanded=False):
            res_file = os.path.join(P.maintenance_dir, "maintenance_parts_reservations.csv")
            try:
                if os.path.exists(res_file):
                    res_all = _read_csv_cached(res_file, False, _mtime(res_file))
                else:
                    res_all = pd.DataFrame()
            except Exception:
                res_all = pd.DataFrame()

            if res_all.empty:
                st.caption("No maintenance reservations found.")
            else:
                for c in ["state", "part_name", "qty", "task_id", "component", "task", "updated_ts"]:
                    if c not in res_all.columns:
                        res_all[c] = ""
                active_res = res_all[res_all["state"].astype(str).str.upper().eq("ACTIVE")].copy()
                c_r1, c_r2 = st.columns(2)
                c_r1.metric("Active Reservations", int(len(active_res)))
                c_r2.metric("Total Reserved Qty", float(pd.to_numeric(active_res.get("qty", 0), errors="coerce").fillna(0.0).sum()))
                view_res = active_res.copy()
                if view_res.empty:
                    st.caption("No ACTIVE reservations.")
                else:
                    st.dataframe(
                        view_res[
                            [
                                c for c in [
                                    "reservation_ts",
                                    "task_id",
                                    "component",
                                    "task",
                                    "part_name",
                                    "qty",
                                    "state",
                                    "actor",
                                    "note",
                                ] if c in view_res.columns
                            ]
                        ],
                        use_container_width=True,
                        height=190,
                    )
                    st.caption("ACTIVE reservation means consumable stock is held for execution (tools stay non-consumable).")

        # Live low-stock visibility (red highlight when Quantity <= Min Level).
        inv_status = inv_df.copy()
        inv_status["Quantity"] = pd.to_numeric(inv_status["Quantity"], errors="coerce").fillna(0.0)
        inv_status["Min Level"] = pd.to_numeric(inv_status["Min Level"], errors="coerce").fillna(0.0)
        inv_status["Item Type"] = inv_status.get("Item Type", "").astype(str).str.strip()
        is_coating = inv_status["Part Name"].astype(str).str.startswith("Coating::")
        is_tool = inv_status["Item Type"].str.lower().eq("tool")
        # Coating rows use default low threshold=1.0 when Min Level is not set (<=0),
        # matching consumables low-stock behavior.
        effective_min = inv_status["Min Level"].copy()
        effective_min = effective_min.where(~(is_coating & (effective_min <= 0)), 1.0)
        inv_status["Effective Min"] = effective_min
        inv_status["_low"] = (
            inv_status["Part Name"].astype(str).str.strip().ne("")
            & (effective_min > 0)
            & (inv_status["Quantity"] <= effective_min)
            & (~is_tool)
        )
        low_count = int(inv_status["_low"].sum())
        low_total_unique = int(inv_status.loc[inv_status["_low"], "Part Name"].astype(str).str.strip().nunique())
        active_status = {"opened", "approved", "ordered", "shipped"}
        low_parts_global = sorted(list({str(x).strip() for x in inv_status.loc[inv_status["_low"], "Part Name"].tolist() if str(x).strip()}))
        low_need_order_unique = 0
        if low_parts_global:
            for pn in low_parts_global:
                has_active = (
                    orders_df["Part Name"].astype(str).str.strip().str.lower().eq(pn.lower())
                    & orders_df["Status"].astype(str).str.strip().str.lower().isin(active_status)
                ).any()
                if not has_active:
                    low_need_order_unique += 1
        m1, m2 = st.columns(2)
        m1.metric("Low Stock Total", int(low_total_unique))
        m2.metric("Low Stock Need Order", int(low_need_order_unique))
        if low_count > 0:
            st.warning(f"Low stock alerts: {low_count} part(s) at/below Min Level.")
        else:
            st.success("No low-stock alerts.")

        with st.expander("Low Stock Details + Order Actions", expanded=False):
            st.markdown("#### Low Stock Only")
            view_cols = [
                c for c in [
                    "Part Name", "Item Type", "Serial Number", "Location", "Location Serial",
                    "Quantity", "Min Level", "Effective Min", "Notes"
                ] if c in inv_status.columns
            ]
            view_df = inv_status[inv_status["_low"]][view_cols + ["_low"]].copy()

            def _low_style(row):
                if bool(row.get("_low", False)):
                    return ["background-color: rgba(255, 77, 77, 0.22); color: #ffd8d8; font-weight: 700;"] * len(row)
                return [""] * len(row)

            if view_df.empty:
                st.info("No low-stock rows.")
            else:
                part_has_active_order = {}
                for pn in sorted(list({str(x).strip() for x in view_df["Part Name"].tolist() if str(x).strip()})):
                    has_active = (
                        orders_df["Part Name"].astype(str).str.strip().str.lower().eq(pn.lower())
                        & orders_df["Status"].astype(str).str.strip().str.lower().isin(active_status)
                    ).any()
                    part_has_active_order[pn] = bool(has_active)

                view_df["Has Active Order"] = view_df["Part Name"].map(lambda x: part_has_active_order.get(str(x).strip(), False))
                view_df["Needs Order"] = ~view_df["Has Active Order"]

                styled = (
                    view_df.style
                    .apply(_low_style, axis=1)
                    .format({"Quantity": "{:.2f}", "Min Level": "{:.2f}", "Effective Min": "{:.2f}"})
                )
                st.dataframe(styled, use_container_width=True, height=220)

                # Quick order creation from low-stock list.
                low_parts = sorted(list({str(x).strip() for x in view_df["Part Name"].tolist() if str(x).strip()}))
                st.markdown("#### 🧾 Create Orders From Low Stock")
                selected_low_parts = st.multiselect(
                    "Pick low-stock parts",
                    options=low_parts,
                    default=[],
                    key="parts_low_order_selected",
                )

                def _create_orders_for_parts(part_names):
                    nonlocal orders_df
                    create_rows = []
                    skipped = []
                    for pn in part_names:
                        pn_clean = str(pn).strip()
                        if not pn_clean:
                            continue
                        exists_active = (
                            orders_df["Part Name"].astype(str).str.strip().str.lower().eq(pn_clean.lower())
                            & orders_df["Status"].astype(str).str.strip().str.lower().isin(active_status)
                        ).any()
                        if exists_active:
                            skipped.append(pn_clean)
                            continue
                        create_rows.append({
                            "Status": "Opened",
                            "Part Name": pn_clean,
                            "Serial Number": "",
                            "Project Name": "Maintenance",
                            "Details": "Auto-created from low stock alert",
                            "Opened By": str(st.session_state.get("maint_actor", "operator")),
                            "Approved": "No",
                            "Approved By": "",
                            "Approval Date": "",
                            "Ordered By": "",
                            "Date Ordered": "",
                            "Company": "",
                            "Inventory Synced": "",
                        })
                    if create_rows:
                        orders_df = pd.concat([orders_df, pd.DataFrame(create_rows)], ignore_index=True)
                        orders_df.to_csv(ORDER_FILE, index=False)
                    return len(create_rows), skipped

                o1, o2 = st.columns(2)
                with o1:
                    if st.button("🧾 Order Selected Low Parts", use_container_width=True, key="parts_order_selected_low_btn"):
                        if not selected_low_parts:
                            st.error("Select at least one part.")
                        else:
                            created, skipped = _create_orders_for_parts(selected_low_parts)
                            if created > 0:
                                st.success(f"Created {created} order(s).")
                            if skipped:
                                st.info("Skipped (active order exists): " + ", ".join(skipped))
                with o2:
                    if st.button("🧾 Order ALL Low Parts", use_container_width=True, key="parts_order_all_low_btn"):
                        created, skipped = _create_orders_for_parts(low_parts)
                        if created > 0:
                            st.success(f"Created {created} order(s).")
                        if skipped:
                            st.info("Skipped (active order exists): " + ", ".join(skipped))

        st.caption("Coating rows are auto-synced from Consumables totals (warehouse + containers, KG).")

        with st.expander("✍️ Manual Inventory Update", expanded=False):
            st.caption("Update existing stock or create a new part in one place.")
            st.markdown('<div class="tp-action-card">', unsafe_allow_html=True)
            q1, q1b, q2, q3, q3b = st.columns([1.1, 1.2, 0.8, 1.0, 0.9])
            inv_names = sorted([str(x).strip() for x in inv_df["Part Name"].tolist() if str(x).strip()])

            def _sync_manual_update_from_part():
                sel = str(st.session_state.get("parts_quick_part", "")).strip()
                if not sel:
                    return
                m = inv_df[inv_df["Part Name"].astype(str).str.strip().str.lower().eq(sel.lower())]
                if m.empty:
                    return
                r0 = m.iloc[0]
                comp = str(r0.get("Component", "")).strip()
                if comp:
                    st.session_state["parts_quick_component"] = comp
                item_type = str(r0.get("Item Type", "")).strip()
                if item_type in ITEM_TYPE_OPTIONS:
                    st.session_state["parts_quick_item_type"] = item_type

            with q1:
                quick_part = st.selectbox(
                    "Part",
                    options=[""] + inv_names,
                    key="parts_quick_part",
                    on_change=_sync_manual_update_from_part,
                )
            with q1b:
                quick_new_part = st.text_input("or New Part", key="parts_quick_new_part", placeholder="create new...")
            with q2:
                quick_delta = st.number_input("Qty", min_value=0.01, max_value=10000.0, value=1.0, step=0.1, key="parts_quick_qty")
            with q3:
                quick_component = st.selectbox("Component", options=component_options or ["Tower Parts"], key="parts_quick_component")
            with q3b:
                quick_item_type = st.selectbox("Type", options=ITEM_TYPE_OPTIONS, key="parts_quick_item_type")

            q4, q5, q6, q7 = st.columns([1.1, 0.9, 0.9, 0.9])
            with q4:
                quick_loc_pick = st.selectbox(
                    "Location",
                    options=[""] + location_options + ["Other (custom)"],
                    key="parts_quick_loc_pick",
                )
            quick_loc_custom = ""
            if quick_loc_pick == "Other (custom)":
                quick_loc_custom = st.text_input("Custom Location", key="parts_quick_loc_custom", placeholder="e.g. Rack C-12")
            quick_loc = quick_loc_custom.strip() if quick_loc_pick == "Other (custom)" else quick_loc_pick
            with q5:
                quick_loc_serial = st.text_input("Loc Serial", key="parts_quick_loc_serial")
            with q6:
                quick_serial = st.text_input("Serial", key="parts_quick_serial")
            with q7:
                quick_min = st.number_input("Min Level", min_value=0.0, max_value=10000.0, value=0.0, step=0.1, key="parts_quick_min")

            quick_target = quick_new_part.strip() or quick_part.strip()

            bq1, bq2 = st.columns(2)
            with bq1:
                if st.button("➕ Add Stock", use_container_width=True, key="parts_quick_add"):
                    if quick_target:
                        # If custom/new location was entered, register it in Storage Locations list.
                        if str(quick_loc).strip():
                            _upsert_storage_location(str(quick_loc).strip(), quick_loc_serial.strip())
                        increment_part(
                            inventory_file,
                            quick_target,
                            qty=float(quick_delta),
                            component=quick_component,
                            serial_number=quick_serial.strip(),
                            location=str(quick_loc).strip(),
                            location_serial=quick_loc_serial.strip(),
                            notes="Quick +",
                            item_type=quick_item_type,
                        )
                        if float(quick_min) > 0:
                            cur = load_inventory(inventory_file)
                            m = (
                                cur["Part Name"].astype(str).str.strip().str.lower().eq(quick_target.lower())
                                & cur["Serial Number"].astype(str).str.strip().str.lower().eq(quick_serial.strip().lower())
                            )
                            if m.any():
                                cur.loc[m, "Min Level"] = float(quick_min)
                                save_inventory(inventory_file, cur)
                        st.success("Stock increased.")
                    else:
                        st.error("Select a part or type a new part.")
            with bq2:
                if st.button("➖ Use Stock", use_container_width=True, key="parts_quick_sub"):
                    if quick_target:
                        ok = decrement_part(
                            inventory_file,
                            quick_target,
                            qty=float(quick_delta),
                            serial_number=quick_serial.strip(),
                        )
                        if ok:
                            if is_non_consumable_part(inventory_file, quick_target, quick_serial.strip()):
                                st.success("Tool usage recorded (non-consumable, stock unchanged).")
                            else:
                                st.success("Stock decreased.")
                        else:
                            st.warning("Part was not found in inventory.")
                    else:
                        st.error("Select a part or type a new part.")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📌 Mounted Inventory Controls", expanded=False):
            st.caption("Mounted parts are real inventory rows (Component/Location = Mounted). You can unmount anytime.")
            mounted_df_inv = inv_df[
                inv_df["Location"].astype(str).str.strip().str.lower().eq("mounted")
                | inv_df["Component"].astype(str).str.strip().str.lower().eq("mounted")
            ].copy()

        def _unmount_part_inventory_center(part_name: str, part_no: str, qty: float) -> bool:
            pname = str(part_name or "").strip()
            pno = str(part_no or "").strip()
            q = max(0.01, float(qty))
            if not pname:
                return False
            cur = load_inventory(inventory_file)
            cur["Part Name"] = cur["Part Name"].astype(str).fillna("")
            cur["Serial Number"] = cur["Serial Number"].astype(str).fillna("")
            cur["Location"] = cur["Location"].astype(str).fillna("")
            cur["Quantity"] = pd.to_numeric(cur["Quantity"], errors="coerce").fillna(0.0)
            mask = (
                cur["Part Name"].str.strip().str.lower().eq(pname.lower())
                & cur["Location"].str.strip().str.lower().eq("mounted")
            )
            if pno:
                m_sn = mask & cur["Serial Number"].str.strip().str.lower().eq(pno.lower())
                if m_sn.any():
                    mask = m_sn
            if not mask.any():
                return False
            idx = cur[mask].index[0]
            new_qty = max(0.0, float(cur.at[idx, "Quantity"]) - q)
            cur.at[idx, "Quantity"] = new_qty
            cur.at[idx, "Last Updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.at[idx, "Notes"] = "Unmounted from machine"
            if new_qty == 0.0:
                cur.at[idx, "Location"] = ""
                cur.at[idx, "Location Serial"] = ""
                cur.at[idx, "Component"] = "Tower Parts"
            save_inventory(inventory_file, cur)
            return True

            if mounted_df_inv.empty:
                st.info("No mounted parts in inventory yet.")
            else:
                show_m_cols2 = [c for c in ["Part Name", "Serial Number", "Quantity", "Location", "Location Serial", "Component"] if c in mounted_df_inv.columns]
                st.dataframe(mounted_df_inv[show_m_cols2], use_container_width=True, height=180)
                mlabels2 = []
                mmap2 = {}
                for i, mr in mounted_df_inv.iterrows():
                    lbl = f"{mr.get('Part Name','')} | SN:{mr.get('Serial Number','') or '-'} | Qty:{mr.get('Quantity',0)}"
                    mlabels2.append(lbl)
                    mmap2[lbl] = i
                u1, u2 = st.columns([2, 1])
                with u1:
                    m_pick2 = st.selectbox("Select mounted part to unmount", [""] + mlabels2, key="parts_unmount_pick_inv")
                with u2:
                    u_qty2 = st.number_input("Unmount qty", min_value=0.01, max_value=10000.0, value=1.0, step=0.1, key="parts_unmount_qty_inv")
                if st.button("↩️ Unmount selected part", use_container_width=True, key="parts_unmount_btn_inv"):
                    if not m_pick2:
                        st.error("Select a mounted part first.")
                    else:
                        rsel2 = mounted_df_inv.loc[mmap2[m_pick2]]
                        ok2 = _unmount_part_inventory_center(
                            str(rsel2.get("Part Name", "")),
                            str(rsel2.get("Serial Number", "")),
                            float(u_qty2),
                        )
                        if ok2:
                            st.success("Mounted quantity updated (part unmounted).")
                            st.rerun()
                        else:
                            st.warning("Could not unmount this part.")

        with st.expander("🗂️ Storage Locations", expanded=False):
            st.caption("Create/edit storage places with location serials (used by intake).")
            loc_df = load_locations(locations_file)
            loc_edit = st.data_editor(
                loc_df,
                use_container_width=True,
                height=220,
                num_rows="dynamic",
                column_config={
                    "Location Name": st.column_config.TextColumn("Location Name", required=True),
                    "Location Serial": st.column_config.TextColumn("Location Serial"),
                    "Description": st.column_config.TextColumn("Description"),
                    "Active": st.column_config.SelectboxColumn("Active", options=["Yes", "No"]),
                    "Last Updated": st.column_config.TextColumn("Last Updated"),
                },
                key="parts_locations_editor",
            )
            if st.button("💾 Save Locations", use_container_width=True, key="parts_locations_save_btn", type="primary"):
                now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                loc_save = loc_edit.copy()
                if "Last Updated" in loc_save.columns:
                    touch = loc_save["Location Name"].astype(str).str.strip().ne("")
                    loc_save.loc[touch, "Last Updated"] = now_ts
                save_locations(locations_file, loc_save)
                st.success("Storage locations saved.")

    else:
        st.caption("Inventory Center is collapsed. Open the toggle to manage stock and intake.")

    st.divider()
    
    # =========================
    # Parts Datasheet (OLD FLOW) + NICE VIEWER
    # =========================
    show_parts_datasheet = st.toggle(
        "📚 Open Parts Datasheet",
        value=False,
        key="parts_datasheet_open_toggle",
    )
    if not show_parts_datasheet:
        st.caption("Parts Datasheet is collapsed. Open the toggle to view manual/BOM tools.")
        return

    st.markdown("## 📚 Parts Datasheet")
    st.caption("Manual BOM tools in one clean area: find parts, inspect pages, compare with inventory, and apply actions.")

    def _clean_txt(s):
        import re
        return re.sub(r"\s+", " ", str(s or "")).strip()

    def _is_num_line(s: str) -> bool:
        import re
        return bool(re.fullmatch(r"\d+(\.\d+)?", _clean_txt(s)))

    def _is_part_num_line(s: str) -> bool:
        import re
        t = _clean_txt(s).upper().rstrip(".")
        # Accept wide engineering PN formats:
        # 286491, EE0031166, EL82610, EE006003.EE006064, A12-BC34
        if not t or " " in t:
            return False
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9._/\\-]*", t):
            return False
        if not re.search(r"\d", t):
            return False
        if re.fullmatch(r"\d{1,2}", t):
            return False
        return True

    def _looks_like_desc(s: str) -> bool:
        import re
        t = _clean_txt(s)
        if not t:
            return False
        if _is_num_line(t) or _is_part_num_line(t):
            return False
        if re.fullmatch(r"[A-Z]-[A-Z]", t):
            return False
        return len(t) >= 3

    def _extract_parts_rows_from_lines(lines, manual_name: str, page_no: int):
        rows = []
        start = 0
        for i, l in enumerate(lines):
            if "PARTS LIST" in _clean_txt(l).upper():
                start = i + 1
                break
        tokens = []
        for raw in lines[start:]:
            t = _clean_txt(raw)
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

        # Remove obvious stray direction markers that break row parsing
        # (common in drawing exports where RH/LH appears outside table cells).
        cleaned = []
        for idx, tok in enumerate(tokens):
            up = tok.upper()
            if up in {"RH", "LH"}:
                prev_tok = tokens[idx - 1] if idx > 0 else ""
                next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                if _is_part_num_line(prev_tok) and _is_num_line(next_tok):
                    continue
            cleaned.append(tok)
        tokens = cleaned

        i = 0
        while i + 3 < len(tokens):
            d, pn, qty, item = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
            # Layout A: DESCRIPTION, PART NUMBER, QTY, ITEM
            if _looks_like_desc(d) and _is_part_num_line(pn) and _is_num_line(qty) and _is_num_line(item):
                rows.append(
                    {
                        "Manual": manual_name,
                        "Page": int(page_no),
                        "Item": item,
                        "Part": d,
                        "Part Number": pn.rstrip("."),
                        "Qty/Asm": qty,
                    }
                )
                i += 4
                continue
            # Layout B: ITEM, QTY, PART NUMBER, DESCRIPTION
            item2, qty2, pn2, d2 = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
            if _is_num_line(item2) and _is_num_line(qty2) and _is_part_num_line(pn2) and _looks_like_desc(d2):
                rows.append(
                    {
                        "Manual": manual_name,
                        "Page": int(page_no),
                        "Item": item2,
                        "Part": d2,
                        "Part Number": pn2.rstrip("."),
                        "Qty/Asm": qty2,
                    }
                )
                i += 4
                continue
            if i + 4 < len(tokens):
                d2 = f"{d} {pn}"
                pn2, qty2, item2 = tokens[i + 2], tokens[i + 3], tokens[i + 4]
                if _looks_like_desc(d2) and _is_part_num_line(pn2) and _is_num_line(qty2) and _is_num_line(item2):
                    rows.append(
                        {
                            "Manual": manual_name,
                            "Page": int(page_no),
                            "Item": item2,
                            "Part": _clean_txt(d2),
                            "Part Number": pn2.rstrip("."),
                            "Qty/Asm": qty2,
                        }
                    )
                    i += 5
                    continue
            i += 1
        return rows

    @st.cache_data(show_spinner=False)
    def _build_manual_bom_index(manuals_dir: str, signature: tuple):
        import glob
        import fitz
        import pandas as pd
        import re

        key_pat = re.compile(r"PARTS?\s+LIST|BILL OF MATERIALS|BOM|PART NUMBER|ITEM", re.IGNORECASE)
        rows = []
        for pdf in sorted(glob.glob(os.path.join(manuals_dir, "*.pdf"))):
            mname = os.path.basename(pdf)
            try:
                doc = fitz.open(pdf)
            except Exception:
                continue
            for pidx in range(len(doc)):
                txt = doc.load_page(pidx).get_text("text") or ""
                if not key_pat.search(txt):
                    continue
                lines = [x for x in txt.splitlines() if _clean_txt(x)]
                rows.extend(_extract_parts_rows_from_lines(lines, mname, pidx + 1))
            doc.close()
        if not rows:
            return pd.DataFrame(columns=["Manual", "Page", "Item", "Part", "Part Number", "Qty/Asm"])
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Manual", "Page", "Item", "Part Number", "Part"]).reset_index(drop=True)
        return df

    @st.cache_data(show_spinner=False)
    def _render_pdf_page_png(path: str, page_no: int, zoom: float = 1.6):
        import fitz
        doc = fitz.open(path)
        pidx = max(0, min(int(page_no) - 1, len(doc) - 1))
        page = doc.load_page(pidx)
        pix = page.get_pixmap(matrix=fitz.Matrix(float(zoom), float(zoom)), alpha=False)
        doc.close()
        return pix.tobytes("png")

    manuals_dir = os.path.join(P.root_dir, "manuals")
    if os.path.isdir(manuals_dir):
        sig = _manual_pdf_signature_cached(manuals_dir, _mtime(manuals_dir))
        bom_df = _build_manual_bom_index(manuals_dir, tuple(sig))
    else:
        bom_df = None

    st.markdown("#### 🔎 Part → Manual Page Finder")
    st.caption("Search a part and open the exact manual page from BOM/parts-list sections.")
    if bom_df is None or bom_df.empty:
        st.info("No manuals BOM index found.")
    else:
        def _create_or_open_part_order(part_name: str, part_no: str, details: str = ""):
            nonlocal orders_df
            pn = str(part_name or "").strip()
            if not pn:
                return "missing", ""
            pno = str(part_no or "").strip()
            active_status = {"opened", "approved", "ordered", "shipped", "received"}
            mask = (
                orders_df["Part Name"].astype(str).str.strip().str.lower().eq(pn.lower())
                & orders_df["Status"].astype(str).str.strip().str.lower().isin(active_status)
            )
            if mask.any():
                idx = orders_df[mask].index[0]
                return "exists", f"row {idx}"
            new_row = {
                "Status": "Opened",
                "Part Name": pn,
                "Serial Number": pno,
                "Project Name": "Maintenance",
                "Details": str(details or "").strip()[:300],
                "Opened By": str(st.session_state.get("maint_actor", "operator")),
                "Approved": "No",
                "Approved By": "",
                "Approval Date": "",
                "Ordered By": "",
                "Date Ordered": "",
                "Company": "SG",
                "Inventory Synced": "",
            }
            orders_df = pd.concat([orders_df, pd.DataFrame([new_row])], ignore_index=True)
            orders_df.to_csv(ORDER_FILE, index=False)
            return "created", ""

        def _mark_part_mounted(part_name: str, part_no: str, qty: float, component: str = ""):
            pname = str(part_name or "").strip()
            if not pname:
                return False
            pno = str(part_no or "").strip()
            cpt = str(component or "").strip()
            # Keep "Mounted" as real inventory location for installed assembly parts.
            ldf = load_locations(locations_file)
            has_mounted = ldf["Location Name"].astype(str).str.strip().str.lower().eq("mounted").any()
            if not has_mounted:
                now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                ldf = pd.concat(
                    [
                        ldf,
                        pd.DataFrame([{
                            "Location Name": "Mounted",
                            "Location Serial": "MOUNTED",
                            "Description": "Installed on machine",
                            "Active": "Yes",
                            "Last Updated": now_ts,
                        }]),
                    ],
                    ignore_index=True,
                )
                save_locations(locations_file, ldf)
            q = max(0.01, float(qty))
            cur = load_inventory(inventory_file)
            cur["Part Name"] = cur["Part Name"].astype(str).fillna("")
            cur["Serial Number"] = cur["Serial Number"].astype(str).fillna("")
            cur["Component"] = cur["Component"].astype(str).fillna("")
            cur["Location"] = cur["Location"].astype(str).fillna("")
            cur["Location Serial"] = cur["Location Serial"].astype(str).fillna("")
            cur["Notes"] = cur["Notes"].astype(str).fillna("")
            cur["Quantity"] = pd.to_numeric(cur["Quantity"], errors="coerce").fillna(0.0)

            # 1) Upsert dedicated mounted row (never merge into regular stock rows).
            m_mask = (
                cur["Part Name"].str.strip().str.lower().eq(pname.lower())
                & cur["Location"].str.strip().str.lower().eq("mounted")
            )
            if pno:
                m_mask = m_mask & cur["Serial Number"].str.strip().str.lower().eq(pno.lower())

            now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if m_mask.any():
                midx = cur[m_mask].index[0]
                cur.at[midx, "Quantity"] = float(cur.at[midx, "Quantity"]) + q
                if cpt and not str(cur.at[midx, "Component"]).strip():
                    cur.at[midx, "Component"] = cpt
                cur.at[midx, "Last Updated"] = now_ts
                cur.at[midx, "Notes"] = "Mounted on machine"
            else:
                # derive component from existing stock row when possible
                comp_guess = cpt
                if not comp_guess:
                    src = cur[cur["Part Name"].str.strip().str.lower().eq(pname.lower())]
                    if pno:
                        src_sn = src[src["Serial Number"].str.strip().str.lower().eq(pno.lower())]
                        if not src_sn.empty:
                            src = src_sn
                    if not src.empty:
                        comp_guess = str(src.iloc[0].get("Component", "")).strip()
                if not comp_guess:
                    comp_guess = "Manual BOM"
                new_row = pd.DataFrame(
                    [
                        {
                            "Part Name": pname,
                            "Component": comp_guess,
                            "Serial Number": pno,
                            "Location": "Mounted",
                            "Location Serial": "MOUNTED",
                            "Quantity": q,
                            "Min Level": 0.0,
                            "Notes": "Mounted on machine",
                            "Last Updated": now_ts,
                        }
                    ]
                )
                cur = pd.concat([cur, new_row], ignore_index=True)

            # 2) Optional stock movement: reduce non-mounted stock rows first.
            remain = q
            stock_mask = (
                cur["Part Name"].str.strip().str.lower().eq(pname.lower())
                & cur["Location"].str.strip().str.lower().ne("mounted")
            )
            if pno:
                stock_mask = stock_mask & cur["Serial Number"].str.strip().str.lower().eq(pno.lower())
            stock_rows = cur[stock_mask].index.tolist()
            for sidx in stock_rows:
                if remain <= 0:
                    break
                have = float(cur.at[sidx, "Quantity"])
                take = min(have, remain)
                if take > 0:
                    cur.at[sidx, "Quantity"] = have - take
                    cur.at[sidx, "Last Updated"] = now_ts
                    remain -= take

            save_inventory(inventory_file, cur)
            return True

        def _unmount_part(part_name: str, part_no: str, qty: float) -> bool:
            pname = str(part_name or "").strip()
            pno = str(part_no or "").strip()
            q = max(0.01, float(qty))
            if not pname:
                return False
            cur = load_inventory(inventory_file)
            cur["Part Name"] = cur["Part Name"].astype(str).fillna("")
            cur["Serial Number"] = cur["Serial Number"].astype(str).fillna("")
            cur["Location"] = cur["Location"].astype(str).fillna("")
            cur["Quantity"] = pd.to_numeric(cur["Quantity"], errors="coerce").fillna(0.0)
            mask = (
                cur["Part Name"].str.strip().str.lower().eq(pname.lower())
                & cur["Location"].str.strip().str.lower().eq("mounted")
            )
            if pno:
                m_sn = mask & cur["Serial Number"].str.strip().str.lower().eq(pno.lower())
                if m_sn.any():
                    mask = m_sn
            if not mask.any():
                return False
            idx = cur[mask].index[0]
            new_qty = max(0.0, float(cur.at[idx, "Quantity"]) - q)
            cur.at[idx, "Quantity"] = new_qty
            cur.at[idx, "Last Updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.at[idx, "Notes"] = "Unmounted from machine"
            if new_qty == 0.0:
                # Keep history row but clear mounted location when empty.
                cur.at[idx, "Location"] = ""
                cur.at[idx, "Location Serial"] = ""
            save_inventory(inventory_file, cur)
            return True

        def _sort_bom_rows(df: pd.DataFrame) -> pd.DataFrame:
            """Keep BOM rows in stable human order (item, then page/PN)."""
            if df is None or df.empty:
                return df
            out = df.copy()
            out["_item_num"] = pd.to_numeric(out.get("Item"), errors="coerce")
            out = out.sort_values(by=["_item_num", "Page", "Part Number"], na_position="last").drop(
                columns=["_item_num"], errors="ignore"
            )
            return out

        def _filter_bom_rows_by_pn(df: pd.DataFrame, query: str) -> pd.DataFrame:
            """Optional quick filter by Part Number token (EE/EL/K/etc)."""
            if df is None or df.empty:
                return df
            q = str(query or "").strip()
            if not q:
                return df
            return df[df["Part Number"].astype(str).str.contains(q, case=False, na=False)].copy()

        def _render_bom_row_actions(rr: pd.Series, key_suffix: str, detail_prefix: str = "From manual") -> None:
            """Single row actions reused across manual/BOM views."""
            mfile = str(rr.get("Manual", "")).strip()
            pno = int(rr.get("Page", 1))
            a1, a2 = st.columns(2)
            with a1:
                if st.button("🧾 Create/Open part order", key=f"parts_bom_order_btn_{key_suffix}", use_container_width=True):
                    stt, msg = _create_or_open_part_order(
                        str(rr.get("Part", "")),
                        str(rr.get("Part Number", "")),
                        details=f"{detail_prefix} {mfile} p.{pno}".strip(),
                    )
                    if stt == "created":
                        st.success("Part order created (Opened).")
                    elif stt == "exists":
                        st.info(f"Active order already exists ({msg}).")
                    else:
                        st.warning("Part is empty.")
            with a2:
                if st.button("📌 Mark as Mounted in inventory", key=f"parts_bom_mount_btn_{key_suffix}", use_container_width=True):
                    qty_asm = pd.to_numeric(rr.get("Qty/Asm", 1), errors="coerce")
                    qty_asm = 1.0 if pd.isna(qty_asm) or float(qty_asm) <= 0 else float(qty_asm)
                    ok = _mark_part_mounted(
                        str(rr.get("Part", "")),
                        str(rr.get("Part Number", "")),
                        qty=qty_asm,
                        component="",
                    )
                    if ok:
                        st.success("Part added/updated in inventory at location: Mounted.")
                        st.rerun()

        def _locate_part_stock(part_name: str, part_no: str, location_name: str, qty_hint: float = 0.0) -> bool:
            pname = str(part_name or "").strip()
            pno = str(part_no or "").strip()
            loc_name = str(location_name or "").strip()
            if not pname or not loc_name:
                return False
            cur = load_inventory(inventory_file)
            cur["Part Name"] = cur["Part Name"].astype(str).fillna("")
            cur["Serial Number"] = cur["Serial Number"].astype(str).fillna("")
            cur["Location"] = cur["Location"].astype(str).fillna("")
            cur["Location Serial"] = cur.get("Location Serial", "").astype(str).fillna("")
            cur["Quantity"] = pd.to_numeric(cur["Quantity"], errors="coerce").fillna(0.0)
            loc_df = load_locations(locations_file)
            ls = ""
            if not loc_df.empty:
                mloc = loc_df["Location Name"].astype(str).str.strip().str.lower().eq(loc_name.lower())
                if mloc.any():
                    ls = str(loc_df.loc[mloc, "Location Serial"].iloc[0]).strip()

            stock_mask = (
                cur["Part Name"].str.strip().str.lower().eq(pname.lower())
                & cur["Location"].str.strip().str.lower().ne("mounted")
            )
            if pno:
                by_sn = stock_mask & cur["Serial Number"].str.strip().str.lower().eq(pno.lower())
                if by_sn.any():
                    stock_mask = by_sn
            now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if stock_mask.any():
                idx = cur[stock_mask].index[0]
                cur.at[idx, "Location"] = loc_name
                cur.at[idx, "Location Serial"] = ls
                cur.at[idx, "Last Updated"] = now_ts
            else:
                cur = pd.concat(
                    [
                        cur,
                        pd.DataFrame(
                            [
                                {
                                    "Part Name": pname,
                                    "Component": "Manual BOM",
                                    "Serial Number": pno,
                                    "Location": loc_name,
                                    "Location Serial": ls,
                                    "Quantity": max(0.0, float(qty_hint)),
                                    "Min Level": 0.0,
                                    "Notes": "Located from manual actions",
                                    "Last Updated": now_ts,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            save_inventory(inventory_file, cur)
            return True

        def _stock_not_mounted_qty(part_name: str, part_no: str) -> float:
            cur = load_inventory(inventory_file)
            cur["Part Name"] = cur["Part Name"].astype(str).fillna("")
            cur["Serial Number"] = cur["Serial Number"].astype(str).fillna("")
            cur["Location"] = cur["Location"].astype(str).fillna("")
            cur["Quantity"] = pd.to_numeric(cur["Quantity"], errors="coerce").fillna(0.0)
            mask = (
                cur["Part Name"].str.strip().str.lower().eq(str(part_name or "").strip().lower())
                & cur["Location"].str.strip().str.lower().ne("mounted")
            )
            pno = str(part_no or "").strip().lower()
            if pno:
                by_sn = mask & cur["Serial Number"].str.strip().str.lower().eq(pno)
                if by_sn.any():
                    mask = by_sn
            return float(cur.loc[mask, "Quantity"].sum()) if mask.any() else 0.0

        st.session_state.setdefault("parts_action_log", [])
        selected_batch = []

        st.markdown("### 🔀 Manual Tools")
        st.markdown(
            '<div class="tp-green-text">Pick one mode, select one or more rows, then use Action Center below.</div>',
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "View mode",
            ["Part → Manual Page Finder", "Manual → Parts List + Inventory Correlation"],
            horizontal=True,
            key="parts_manual_tools_mode",
        )

        selected_ctx = None
        location_df = load_locations(locations_file)
        location_opts_action = sorted({str(x).strip() for x in location_df.get("Location Name", pd.Series([], dtype=str)).tolist() if str(x).strip()})

        if mode == "Part → Manual Page Finder":
            f1, f2 = st.columns([1.8, 1.2])
            with f1:
                bom_query = st.text_input(
                    "Search part / part number",
                    key="parts_bom_query",
                    placeholder="e.g. clamp washer, 284531...",
                )
            with f2:
                manual_opts = ["All"] + sorted(bom_df["Manual"].astype(str).unique().tolist())
                bom_manual = st.selectbox("Manual", manual_opts, key="parts_bom_manual")

            match_df = bom_df.copy()
            if bom_manual != "All":
                match_df = match_df[match_df["Manual"].astype(str).eq(bom_manual)].copy()
            if bom_query.strip():
                q = bom_query.strip().lower()
                match_df = match_df[
                    match_df["Part"].astype(str).str.lower().str.contains(q, na=False)
                    | match_df["Part Number"].astype(str).str.lower().str.contains(q, na=False)
                ].copy()
            match_df = _sort_bom_rows(match_df)

            st.caption(f"Matches: {len(match_df)} (showing first 250)")
            st.dataframe(
                match_df[["Part", "Part Number", "Qty/Asm", "Manual", "Page", "Item"]].head(250),
                use_container_width=True,
                height=220,
            )

            if not match_df.empty:
                labels = []
                idx_map = {}
                for i, r in match_df.head(250).iterrows():
                    lb = f"{r.get('Part','')} | PN:{r.get('Part Number','')} | {r.get('Manual','')} p.{r.get('Page','')}"
                    labels.append(lb)
                    idx_map[lb] = i
                pick = st.selectbox("Open result", [""] + labels, key="parts_bom_pick")
                if pick:
                    rr = match_df.loc[idx_map[pick]]
                    mfile = str(rr.get("Manual", "")).strip()
                    pno = int(rr.get("Page", 1))
                    mpath = os.path.join(manuals_dir, mfile)
                    if os.path.exists(mpath):
                        st.markdown(f"**Selected:** `{rr.get('Part','')}` | **Manual:** `{mfile}` | **Page:** `{pno}`")
                        try:
                            png = _render_pdf_page_png(mpath, pno, 1.6)
                            st.image(png, caption=f"{mfile} — page {pno}", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Page preview failed: {e}")
                        selected_ctx = {
                            "part": str(rr.get("Part", "")).strip(),
                            "part_no": str(rr.get("Part Number", "")).strip(),
                            "qty": float(pd.to_numeric(rr.get("Qty/Asm", 1), errors="coerce") or 1.0),
                            "manual": mfile,
                            "page": pno,
                        }
                    else:
                        st.warning(f"Manual file not found: {mfile}")

                batch_pick = st.multiselect(
                    "Batch select rows (for group actions)",
                    options=labels,
                    default=[],
                    key="parts_bom_batch_pick",
                )
                for lb in batch_pick:
                    rr = match_df.loc[idx_map[lb]]
                    selected_batch.append(
                        {
                            "part": str(rr.get("Part", "")).strip(),
                            "part_no": str(rr.get("Part Number", "")).strip(),
                            "qty": float(pd.to_numeric(rr.get("Qty/Asm", 1), errors="coerce") or 1.0),
                            "manual": str(rr.get("Manual", "")).strip(),
                            "page": int(rr.get("Page", 1)),
                        }
                    )

        else:
            st.markdown(
                '<div class="tp-green-text">Selected page mode is strict: only rows parsed from that exact page are shown.</div>',
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns([1.5, 1.0, 0.8])
            with m1:
                man_pick = st.selectbox(
                    "Manual to inspect",
                    sorted(bom_df["Manual"].astype(str).unique().tolist()),
                    key="parts_manual_rev_pick",
                )
            man_df = bom_df[bom_df["Manual"].astype(str).eq(man_pick)].copy()
            page_opts = sorted(man_df["Page"].astype(int).unique().tolist()) if not man_df.empty else []
            with m2:
                page_mode = st.selectbox("BOM scope", ["Selected page", "All BOM pages in manual"], key="parts_manual_scope_mode")
            with m3:
                assemblies_plan = st.number_input(
                    "Assemblies planned",
                    min_value=1,
                    max_value=1000,
                    value=1,
                    step=1,
                    key="parts_manual_asm_plan",
                )

            selected_page = None
            if page_mode == "Selected page":
                if page_opts:
                    st.session_state.setdefault("parts_manual_rev_page_nav", int(page_opts[0]))
                    if st.session_state["parts_manual_rev_page_nav"] not in page_opts:
                        st.session_state["parts_manual_rev_page_nav"] = int(page_opts[0])
                    n1, n2, n3 = st.columns([0.6, 0.6, 1.6])
                    cur_idx = page_opts.index(int(st.session_state["parts_manual_rev_page_nav"]))
                    with n1:
                        if st.button("◀ Prev", key="parts_manual_prev_btn", use_container_width=True, disabled=(cur_idx == 0)):
                            st.session_state["parts_manual_rev_page_nav"] = int(page_opts[max(0, cur_idx - 1)])
                    with n2:
                        if st.button("Next ▶", key="parts_manual_next_btn", use_container_width=True, disabled=(cur_idx >= len(page_opts) - 1)):
                            st.session_state["parts_manual_rev_page_nav"] = int(page_opts[min(len(page_opts) - 1, cur_idx + 1)])
                    with n3:
                        selected_page = st.selectbox(
                            "Page",
                            page_opts,
                            index=page_opts.index(int(st.session_state["parts_manual_rev_page_nav"])),
                            key="parts_manual_rev_page",
                        )
                    st.session_state["parts_manual_rev_page_nav"] = int(selected_page)
                else:
                    selected_page = 1
                scope_df = man_df[man_df["Page"].astype(int).eq(int(selected_page))].copy() if not man_df.empty else man_df
            else:
                scope_df = man_df.copy()

            scope_df = _sort_bom_rows(scope_df)
            mpath2 = os.path.join(manuals_dir, man_pick)
            if os.path.exists(mpath2):
                preview_page = int(selected_page) if selected_page else (int(page_opts[0]) if page_opts else 1)
                try:
                    png2 = _render_pdf_page_png(mpath2, preview_page, 1.4)
                    st.image(png2, caption=f"{man_pick} — page {preview_page}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Manual page preview failed: {e}")

            if scope_df.empty:
                st.info("No BOM rows found for selected scope.")
            else:
                qpn = st.text_input(
                    "Filter Part Number (optional)",
                    value="",
                    placeholder="e.g. EE, EL, K00...",
                    key="parts_bom_scope_pn_filter",
                )
                show_df = _filter_bom_rows_by_pn(scope_df.copy(), qpn)
                st.caption(f"Rows on selected scope: {len(scope_df)} | Rows after filter: {len(show_df)}")
                st.dataframe(
                    show_df[["Part", "Part Number", "Qty/Asm", "Page", "Item"]],
                    use_container_width=True,
                    height=220,
                )
                if not show_df.empty:
                    row_labels = []
                    row_map = {}
                    for i, br in show_df.iterrows():
                        lb = f"Item {br.get('Item','')} | {br.get('Part','')} | PN:{br.get('Part Number','')}"
                        row_labels.append(lb)
                        row_map[lb] = i
                    row_pick = st.selectbox("Pick row for actions", [""] + row_labels, key="parts_corr_row_pick")
                    if row_pick:
                        rr = show_df.loc[row_map[row_pick]]
                        qty_asm = float(pd.to_numeric(rr.get("Qty/Asm", 1), errors="coerce") or 1.0)
                        selected_ctx = {
                            "part": str(rr.get("Part", "")).strip(),
                            "part_no": str(rr.get("Part Number", "")).strip(),
                            "qty": max(0.01, qty_asm * float(assemblies_plan)),
                            "manual": str(rr.get("Manual", "")).strip(),
                            "page": int(rr.get("Page", 1)),
                        }

                    batch_corr = st.multiselect(
                        "Batch select rows (for group actions)",
                        options=row_labels,
                        default=[],
                        key="parts_corr_batch_pick",
                    )
                    for lb in batch_corr:
                        rr = show_df.loc[row_map[lb]]
                        qty_asm = float(pd.to_numeric(rr.get("Qty/Asm", 1), errors="coerce") or 1.0)
                        selected_batch.append(
                            {
                                "part": str(rr.get("Part", "")).strip(),
                                "part_no": str(rr.get("Part Number", "")).strip(),
                                "qty": max(0.01, qty_asm * float(assemblies_plan)),
                                "manual": str(rr.get("Manual", "")).strip(),
                                "page": int(rr.get("Page", 1)),
                            }
                        )

        st.markdown("### 🧰 Action Center")
        st.markdown(
            '<div class="tp-green-text">Run actions on current selection. Orders from manuals are created with Company=SG.</div>',
            unsafe_allow_html=True,
        )
        if (not selected_ctx) and (not selected_batch):
            st.info("Select a row in current mode to enable actions.")
        else:
            # Auto-detect mode: if batch has any rows -> group mode, else single row mode.
            if selected_batch:
                dedup = {}
                for it in selected_batch:
                    dedup[(it["part"].lower(), it["part_no"].lower())] = it
                targets = list(dedup.values())
                st.success(f"Group mode (auto): {len(targets)} item(s) selected.")
            else:
                targets = [selected_ctx] if selected_ctx else []
                if selected_ctx:
                    st.success(
                        f"Single mode (auto): {selected_ctx['part']} | PN: {selected_ctx['part_no']} | "
                        f"Manual: {selected_ctx['manual']} p.{selected_ctx['page']}"
                    )

            if targets:
                preview_rows = []
                for t in targets:
                    preview_rows.append(
                        {
                            "Part": t["part"],
                            "PN": t["part_no"],
                            "Qty Suggest": round(float(t.get("qty", 0.0)), 3),
                            "Stock (not mounted)": round(_stock_not_mounted_qty(t["part"], t["part_no"]), 3),
                        }
                    )
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=160)

            a1, a2, a3 = st.columns([1.0, 1.2, 1.4])
            with a1:
                action_qty = st.number_input(
                    "Action qty",
                    min_value=0.01,
                    max_value=10000.0,
                    value=float(max(0.01, targets[0].get("qty", 1.0) if targets else 1.0)),
                    step=0.1,
                    key="parts_action_qty",
                )
            with a2:
                action_location = st.selectbox("Locate in", [""] + location_opts_action, key="parts_action_location")
            with a3:
                action_details = st.text_input(
                    "Order details",
                    value=f"From manual {selected_ctx['manual']} p.{selected_ctx['page']}",
                    key="parts_action_details",
                )

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🧾 Create/Open Order (Company: SG)", use_container_width=True, key="parts_action_order_btn"):
                    created_n = 0
                    exists_n = 0
                    for t in targets:
                        stt, _msg = _create_or_open_part_order(t["part"], t["part_no"], details=action_details)
                        if stt == "created":
                            created_n += 1
                        elif stt == "exists":
                            exists_n += 1
                        st.session_state["parts_action_log"].append(
                            {
                                "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Action": "Create/Open Order",
                                "Part": t["part"],
                                "PN": t["part_no"],
                                "Result": stt.upper(),
                            }
                        )
                    st.success(f"Orders created: {created_n} | already open: {exists_n}")
            with b2:
                if st.button("📌 Mount Qty", use_container_width=True, key="parts_action_mount_btn"):
                    ok_n = 0
                    for t in targets:
                        ok = _mark_part_mounted(t["part"], t["part_no"], qty=float(action_qty), component="")
                        if ok:
                            ok_n += 1
                        st.session_state["parts_action_log"].append(
                            {
                                "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Action": "Mount Qty",
                                "Part": t["part"],
                                "PN": t["part_no"],
                                "Result": "OK" if ok else "FAILED",
                            }
                        )
                    st.success(f"Mounted updated for {ok_n}/{len(targets)} item(s).")
                    st.rerun()
            with b3:
                if st.button("📍 Locate In Place", use_container_width=True, key="parts_action_locate_btn"):
                    if not action_location:
                        st.error("Select a location first.")
                    else:
                        ok_n = 0
                        for t in targets:
                            ok = _locate_part_stock(
                                t["part"],
                                t["part_no"],
                                action_location,
                                qty_hint=float(action_qty),
                            )
                            if ok:
                                ok_n += 1
                            st.session_state["parts_action_log"].append(
                                {
                                    "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Action": "Locate In Place",
                                    "Part": t["part"],
                                    "PN": t["part_no"],
                                    "Result": "OK" if ok else "FAILED",
                                }
                            )
                        st.success(f"Location updated for {ok_n}/{len(targets)} item(s).")
                        st.rerun()

            with st.expander("🧾 Action Log", expanded=False):
                logs = st.session_state.get("parts_action_log", [])
                if not logs:
                    st.info("No actions yet.")
                else:
                    st.dataframe(pd.DataFrame(logs).tail(200), use_container_width=True, height=180)
                    if st.button("Clear action log", key="parts_action_log_clear_btn"):
                        st.session_state["parts_action_log"] = []
                        st.rerun()

        st.markdown("---")

    # NOTE: PARTS_DIRECTORY must exist in your app globals/config.
    # Example: PARTS_DIRECTORY = "tower_parts_docs"
    
    def render_pdf_embed(path, height=760):
        """Nice in-app PDF viewer (like other tabs)."""
        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            st.markdown(
                f"""
                <iframe
                    src="data:application/pdf;base64,{b64}"
                    width="100%"
                    height="{height}"
                    style="border:none; border-radius: 12px; background: rgba(0,0,0,0.04);"
                ></iframe>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Failed to render PDF: {e}")
    
    def display_directory(current_path, level=0):
        try:
            items = sorted(os.listdir(current_path))
        except Exception as e:
            st.error(f"Error accessing {current_path}: {e}")
            return None
    
        folder_options = []
        files = []
        for item in items:
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                folder_options.append(item)
            else:
                files.append(full_path)
    
        selected_folder = st.selectbox(
            f"📂 Select folder in {os.path.basename(current_path)}:",
            [""] + folder_options,
            key=f"parts_folder_{level}",
        )
    
        selected_file = None
        if selected_folder:
            selected_file = display_directory(os.path.join(current_path, selected_folder), level + 1)
    
        # old style file buttons -> now we just set selected_file for preview
        for file_path in files:
            file_name = os.path.basename(file_path)
            if st.button(f"📄 Select {file_name}", key=f"select_{file_path}"):
                selected_file = file_path
    
        return selected_file
    
    if PARTS_DIRECTORY and os.path.isdir(PARTS_DIRECTORY) and os.listdir(PARTS_DIRECTORY):
        st.write("Pick folder(s), then select a file to preview:")
    
        selected_file = display_directory(PARTS_DIRECTORY)
    
        st.divider()
        st.write("### 👁️ Preview")
    
        if not selected_file:
            st.info("Select a file above to preview it here.")
        else:
            ext = os.path.splitext(selected_file)[1].lower()
    
            # Always allow download
            try:
                with open(selected_file, "rb") as f:
                    data = f.read()
                st.download_button(
                    "⬇️ Download file",
                    data=data,
                    file_name=os.path.basename(selected_file),
                    use_container_width=True,
                    key=f"parts_dl_{selected_file}"
                )
            except Exception:
                pass
    
            if ext == ".pdf":
                render_pdf_embed(selected_file, height=780)
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
                st.image(selected_file, use_container_width=True)
            elif ext in [".txt", ".log", ".csv", ".json", ".md"]:
                try:
                    txt = open(selected_file, "r", encoding="utf-8", errors="ignore").read()
                    st.code(txt if len(txt) < 80_000 else (txt[:80_000] + "\n\n... (truncated)"), language="text")
                except Exception as e:
                    st.error(f"Failed to preview text: {e}")
            else:
                st.info("Preview not supported for this file type. Use Download and open locally.")
                st.write(f"**Path:** `{selected_file}`")
    else:
        st.info(f"No parts documents found in: {PARTS_DIRECTORY}")
    # ------------------ Development Tab ------------------
