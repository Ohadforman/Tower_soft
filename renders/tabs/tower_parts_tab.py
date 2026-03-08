def render_tower_parts_tab(P):
    import os
    import base64
    import pandas as pd
    import streamlit as st
    from helpers.parts_inventory import (
        load_inventory,
        save_inventory,
        increment_part,
        ensure_inventory_file,
        load_locations,
        save_locations,
        ensure_locations_file,
        decrement_part,
        set_part_quantity,
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tp-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-title">🛠️ Tower Parts Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-sub">Track parts orders, update statuses, archive installed items, and browse docs.</div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-line"></div>', unsafe_allow_html=True)
    
    ORDER_FILE = P.parts_orders_csv
    archive_file = P.parts_archived_csv
    inventory_file = P.parts_inventory_csv
    locations_file = P.parts_locations_csv
    coating_stock_file = P.coating_stock_json
    containers_csv = P.tower_containers_csv
    PARTS_DIRECTORY = P.parts_dir
    
    # ✅ Status rename (Needed -> Opened)
    STATUS_ORDER = ["Opened", "Approved", "Ordered", "Shipped", "Received", "Installed"]
    
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
        orders_df = pd.read_csv(ORDER_FILE, keep_default_na=False)
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

    def _normalize_received_sync_state(df_orders: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        marked_pending = 0
        out = df_orders.copy()
        for i, r in out.iterrows():
            status = str(r.get("Status", "")).strip().lower()
            inv_synced = str(r.get("Inventory Synced", "")).strip().lower()
            if status == "received":
                if inv_synced not in ("yes", "pending"):
                    out.at[i, "Inventory Synced"] = "Pending"
                    marked_pending += 1
            elif status != "received":
                out.at[i, "Inventory Synced"] = ""
        return out, marked_pending

    orders_df, pending_new_count = _normalize_received_sync_state(orders_df)
    if pending_new_count > 0:
        orders_df.to_csv(ORDER_FILE, index=False)
        st.info(f"{pending_new_count} received order(s) are waiting for intake/location.")
    
    # ---------------- Projects list (match 📦 Order Draw) ----------------
    PROJECTS_FILE = P.projects_fiber_csv
    PROJECTS_COL = "Fiber Project"
    
    project_options = ["None"]
    try:
        if os.path.exists(PROJECTS_FILE):
            projects_df = pd.read_csv(PROJECTS_FILE, keep_default_na=False)
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
            st.rerun()
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
            st.rerun()
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
            with st.form("add_new_order_form", clear_on_submit=True):
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
                        st.rerun()
    
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
    
                with st.form("update_order_form"):
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
                        st.rerun()

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                st.caption("Danger zone")
                if st.button("🗑️ Delete This Order", use_container_width=True, key="delete_part_inside_edit"):
                    orders_df = orders_df.drop(index=order_index).reset_index(drop=True)
                    orders_df.to_csv(ORDER_FILE, index=False)
                    st.success("✅ Deleted.")
                    st.rerun()
    
    st.divider()

    # =========================
    # Parts Inventory Center (collapsed)
    # =========================
    with st.expander("📦 Inventory Center", expanded=False):
        st.caption("Manage part stock and location intake for received orders.")

        def _load_tower_components() -> list:
            comps = set()
            mdir = P.maintenance_dir
            if os.path.isdir(mdir):
                for fn in sorted(os.listdir(mdir)):
                    if not fn.lower().endswith((".xlsx", ".xls", ".csv")):
                        continue
                    if "log" in fn.lower() or fn.startswith("_"):
                        continue
                    fp = os.path.join(mdir, fn)
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

        synced_types = _sync_coating_from_consumables()
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

        inv_df = load_inventory(inventory_file)
        sheet_components = _load_tower_components()
        inv_components = sorted(list({str(x).strip() for x in inv_df.get("Component", pd.Series([], dtype=str)).tolist() if str(x).strip()}))
        component_options = sorted(list({*sheet_components, *inv_components, "Tower Parts", "Consumables"}))

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
                intake_qty = st.number_input("Qty (kg)", min_value=0.01, max_value=10000.0, value=1.0, step=0.1, key="parts_intake_qty")
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
                    orders_df.at[ridx, "Inventory Synced"] = "Yes"
                    orders_df.to_csv(ORDER_FILE, index=False)
                    st.success("Received order organized and added to inventory.")
                    st.rerun()

        inv_df = load_inventory(inventory_file)
        if inv_df.empty:
            st.info("Inventory is empty. Add rows below or intake received orders.")

        st.markdown("#### 🔎 Inventory Finder")
        fc1, fc2, fc3 = st.columns([1.6, 1.2, 1.0])
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

        finder_df = inv_df.copy()
        if filter_components:
            finder_df = finder_df[finder_df["Component"].astype(str).isin(filter_components)]
        if part_query.strip():
            q = part_query.strip().lower()
            finder_df = finder_df[finder_df["Part Name"].astype(str).str.lower().str.contains(q, na=False)]
        if only_missing_loc:
            finder_df = finder_df[finder_df["Location"].astype(str).str.strip().eq("")]
        st.caption(f"Finder rows: {len(finder_df)}")
        st.dataframe(
            finder_df[[c for c in ["Part Name", "Component", "Serial Number", "Location", "Location Serial", "Quantity", "Min Level"] if c in finder_df.columns]],
            use_container_width=True,
            height=180,
        )

        sort_mode = st.selectbox(
            "Inventory order",
            ["Component -> Part Name", "Part Name", "Quantity (low first)"],
            key="parts_inventory_sort_mode",
        )
        inv_view = inv_df.copy()
        inv_view["Component"] = inv_view["Component"].astype(str)
        inv_view["Part Name"] = inv_view["Part Name"].astype(str)
        if sort_mode == "Component -> Part Name":
            inv_view = inv_view.sort_values(["Component", "Part Name"], ascending=[True, True], na_position="last")
        elif sort_mode == "Part Name":
            inv_view = inv_view.sort_values(["Part Name"], ascending=[True], na_position="last")
        else:
            inv_view["Quantity"] = pd.to_numeric(inv_view["Quantity"], errors="coerce").fillna(0.0)
            inv_view = inv_view.sort_values(["Quantity", "Part Name"], ascending=[True, True], na_position="last")

        inv_edit = st.data_editor(
            inv_view,
            use_container_width=True,
            height=320,
            num_rows="dynamic",
            column_config={
                "Part Name": st.column_config.TextColumn("Part Name", required=True),
                "Component": st.column_config.SelectboxColumn("Component", options=component_options or ["Tower Parts"]),
                "Serial Number": st.column_config.TextColumn("Serial Number"),
                "Location": st.column_config.TextColumn("Location"),
                "Location Serial": st.column_config.TextColumn("Location Serial"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.1, format="%.2f"),
                "Min Level": st.column_config.NumberColumn("Min Level", min_value=0.0, step=0.1, format="%.2f"),
                "Notes": st.column_config.TextColumn("Notes"),
                "Last Updated": st.column_config.TextColumn("Last Updated"),
            },
            key="parts_inventory_editor",
        )
        c_inv1, c_inv2 = st.columns(2)
        with c_inv1:
            if st.button("💾 Save Inventory", use_container_width=True, key="parts_inv_save_btn", type="primary"):
                save_inventory(inventory_file, inv_edit)
                st.success("Inventory saved.")
                st.rerun()
        with c_inv2:
            if st.button("🔎 Refresh Intake Queue", use_container_width=True, key="parts_intake_refresh_btn"):
                st.rerun()

        # Live low-stock visibility (red highlight when Quantity <= Min Level).
        inv_status = inv_edit.copy()
        inv_status["Quantity"] = pd.to_numeric(inv_status["Quantity"], errors="coerce").fillna(0.0)
        inv_status["Min Level"] = pd.to_numeric(inv_status["Min Level"], errors="coerce").fillna(0.0)
        is_coating = inv_status["Part Name"].astype(str).str.startswith("Coating::")
        # Coating rows use default low threshold=1.0 when Min Level is not set (<=0),
        # matching consumables low-stock behavior.
        effective_min = inv_status["Min Level"].copy()
        effective_min = effective_min.where(~(is_coating & (effective_min <= 0)), 1.0)
        inv_status["Effective Min"] = effective_min
        inv_status["_low"] = (
            inv_status["Part Name"].astype(str).str.strip().ne("")
            & (effective_min > 0)
            & (inv_status["Quantity"] <= effective_min)
        )
        low_count = int(inv_status["_low"].sum())
        if low_count > 0:
            st.warning(f"Low stock alerts: {low_count} part(s) at/below Min Level.")
        else:
            st.success("No low-stock alerts.")

        st.markdown("#### Low Stock Only")
        view_cols = [
            c for c in [
                "Part Name", "Serial Number", "Location", "Location Serial",
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
            # Order coverage status for low-stock parts.
            active_status = {"opened", "approved", "ordered", "shipped"}
            part_has_active_order = {}
            for pn in sorted(list({str(x).strip() for x in view_df["Part Name"].tolist() if str(x).strip()})):
                has_active = (
                    orders_df["Part Name"].astype(str).str.strip().str.lower().eq(pn.lower())
                    & orders_df["Status"].astype(str).str.strip().str.lower().isin(active_status)
                ).any()
                part_has_active_order[pn] = bool(has_active)

            view_df["Has Active Order"] = view_df["Part Name"].map(lambda x: part_has_active_order.get(str(x).strip(), False))
            view_df["Needs Order"] = ~view_df["Has Active Order"]

            low_total_unique = len(part_has_active_order)
            low_need_order_unique = len([p for p, has_o in part_has_active_order.items() if not has_o])
            m1, m2 = st.columns(2)
            m1.metric("Low Stock Total", int(low_total_unique))
            m2.metric("Low Stock Need Order", int(low_need_order_unique))

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
                        if created > 0:
                            st.rerun()
            with o2:
                if st.button("🧾 Order ALL Low Parts", use_container_width=True, key="parts_order_all_low_btn"):
                    created, skipped = _create_orders_for_parts(low_parts)
                    if created > 0:
                        st.success(f"Created {created} order(s).")
                    if skipped:
                        st.info("Skipped (active order exists): " + ", ".join(skipped))
                    if created > 0:
                        st.rerun()

        st.caption("Coating rows are auto-synced from Consumables totals (warehouse + containers, KG).")

        st.markdown("#### Quick Stock Update")
        q1, q2, q3, q4, q5, q6 = st.columns([1.1, 0.8, 0.8, 0.9, 0.9, 1.0])
        inv_names = sorted([str(x).strip() for x in inv_df["Part Name"].tolist() if str(x).strip()])
        with q1:
            quick_part = st.selectbox("Part", options=[""] + inv_names, key="parts_quick_part")
        with q2:
            quick_delta = st.number_input("Qty (kg)", min_value=0.01, max_value=10000.0, value=1.0, step=0.1, key="parts_quick_qty")
        with q3:
            quick_component = st.selectbox("Component", options=component_options or ["Tower Parts"], key="parts_quick_component")
        with q4:
            quick_loc = st.text_input("Location", key="parts_quick_loc")
        with q5:
            quick_loc_serial = st.text_input("Loc Serial", key="parts_quick_loc_serial")
        with q6:
            quick_serial = st.text_input("Serial", key="parts_quick_serial")

        bq1, bq2 = st.columns(2)
        with bq1:
            if st.button("➕ Add Stock", use_container_width=True, key="parts_quick_add"):
                if quick_part.strip():
                    increment_part(
                        inventory_file,
                        quick_part.strip(),
                        qty=float(quick_delta),
                        component=quick_component,
                        serial_number=quick_serial.strip(),
                        location=quick_loc.strip(),
                        location_serial=quick_loc_serial.strip(),
                        notes="Quick +",
                    )
                    st.success("Stock increased.")
                    st.rerun()
                else:
                    st.error("Select a part first.")
        with bq2:
            if st.button("➖ Use Stock", use_container_width=True, key="parts_quick_sub"):
                if quick_part.strip():
                    ok = decrement_part(
                        inventory_file,
                        quick_part.strip(),
                        qty=float(quick_delta),
                        serial_number=quick_serial.strip(),
                    )
                    if ok:
                        st.success("Stock decreased.")
                        st.rerun()
                    else:
                        st.warning("Part was not found in inventory.")
                else:
                    st.error("Select a part first.")

        st.markdown("#### 🗂️ Storage Locations")
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
            st.rerun()

    st.divider()

    # =========================
    # ARCHIVE / VIEW ARCHIVE
    # =========================
    st.markdown('<div class="tp-section">🗃️ Archive</div>', unsafe_allow_html=True)
    cA, cB = st.columns([1, 1])
    
    with cA:
        if st.button("📦 Archive Installed Orders", use_container_width=True):
            installed_df = orders_df[orders_df["Status"].astype(str).str.strip().str.lower() == "installed"]
            remaining_df = orders_df[orders_df["Status"].astype(str).str.strip().str.lower() != "installed"]
    
            if installed_df.empty:
                st.info("No installed parts to archive.")
            else:
                if os.path.exists(archive_file):
                    archived_df = pd.read_csv(archive_file, keep_default_na=False)
                    archived_df.columns = archived_df.columns.str.strip()
                    for col in BASE_COLUMNS:
                        if col not in archived_df.columns:
                            archived_df[col] = ""
                    archived_df = pd.concat([archived_df, installed_df], ignore_index=True)
                else:
                    archived_df = installed_df.copy()
    
                archived_df.to_csv(archive_file, index=False)
                remaining_df.to_csv(ORDER_FILE, index=False)
                st.success(f"✅ {len(installed_df)} installed order(s) archived.")
                st.rerun()
    
    with cB:
        show_archive = st.button("📂 View Archived Orders", use_container_width=True)
    
    if show_archive:
        if os.path.exists(archive_file):
            archived_df = pd.read_csv(archive_file, keep_default_na=False)
            archived_df.columns = archived_df.columns.str.strip()
            for col in BASE_COLUMNS:
                if col not in archived_df.columns:
                    archived_df[col] = ""
            if archived_df.empty:
                st.info("The archive is currently empty.")
            else:
                st.write("#### Archived Orders")
                show_cols = [c for c in column_order if c in archived_df.columns]
                st.dataframe(archived_df[show_cols], height=320, use_container_width=True)
        else:
            st.info("Archive file does not exist yet.")
    
    st.divider()
    
    # =========================
    # Parts Datasheet (OLD FLOW) + NICE VIEWER
    # =========================
    st.markdown('<div class="tp-section">📚 Parts Datasheet (Hierarchical View)</div>', unsafe_allow_html=True)
    
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
