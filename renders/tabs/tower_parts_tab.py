def render_tower_parts_tab(P):
    import os
    import base64
    import pandas as pd
    import streamlit as st
    
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
    
    # Drop old Purpose if exists
    orders_df = orders_df.drop(columns=["Purpose"], errors="ignore")
    
    orders_df["Status"] = orders_df["Status"].fillna("").astype(str).str.strip()
    orders_df["Status"] = orders_df["Status"].replace({"Needed": "Opened", "needed": "Opened"})
    
    # Unknown / empty -> Opened
    orders_df["Status"] = orders_df["Status"].apply(lambda s: s if s in STATUS_ORDER else "Opened")
    
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
        st.dataframe(
            styled,
            height=420,
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
    
                        orders_df.to_csv(ORDER_FILE, index=False)
                        st.success("✅ Order updated.")
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
    # DELETE
    # =========================
    st.markdown('<div class="tp-section">🗑️ Delete</div>', unsafe_allow_html=True)
    if orders_df.empty:
        st.info("Nothing to delete.")
    else:
        del_labels = (orders_df["Part Name"].astype(str).fillna("") + "  |  " +
                      orders_df["Serial Number"].astype(str).fillna(""))
        del_map = {del_labels.iloc[i]: i for i in range(len(del_labels))}
        del_choice = st.selectbox("Select an order to delete", list(del_map.keys()), key="delete_part_main")
    
        if st.button("Delete Selected Order", use_container_width=True):
            idx = del_map[del_choice]
            orders_df = orders_df.drop(index=idx).reset_index(drop=True)
            orders_df.to_csv(ORDER_FILE, index=False)
            st.success("✅ Deleted.")
            st.rerun()
    
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
    
    if "PARTS_DIRECTORY" in globals() and os.path.exists(PARTS_DIRECTORY) and os.listdir(PARTS_DIRECTORY):
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
        st.info("No parts documents found in PARTS_DIRECTORY (or PARTS_DIRECTORY not set).")
    # ------------------ Development Tab ------------------
