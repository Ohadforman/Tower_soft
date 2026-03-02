def render_tower_parts_tab(P):
    import os
    import base64
    import pandas as pd
    import streamlit as st
    
    st.title("🛠️ Tower Parts Management")
    
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
    st.write("### 📋 Orders Table")
    
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
    
    # Color status cell only
    def highlight_status(row):
        color_map = {
            "Opened": "background-color: lightcoral; color: black; font-weight: 900;",
            "Approved": "background-color: lightgreen; color: black; font-weight: 900;",
            "Ordered": "background-color: lightyellow; color: black; font-weight: 900;",
            "Shipped": "background-color: lightblue; color: black; font-weight: 900;",
            "Received": "background-color: green; color: black; font-weight: 900;",
            "Installed": "background-color: lightgray; color: black; font-weight: 900;",
        }
        s = str(row.get("Status", "")).strip()
        return [color_map.get(s, "")] + [""] * (len(row) - 1)
    
    if not tmp.empty:
        st.dataframe(
            tmp[column_order].fillna("").style.apply(highlight_status, axis=1),
            height=420,
            use_container_width=True,
        )
    else:
        st.info("No orders have been placed yet.")
    
    st.divider()
    
    # =========================
    # CLEAN POP AREA (AFTER TABLE)
    # =========================
    st.write("### ✍️ Manage Orders")
    
    action = st.radio(
        "Choose action",
        ["Add New Order", "Update Existing Order"],
        horizontal=True,
        key="order_action_main",
    )
    
    # ---------- Add New ----------
    if action == "Add New Order":
        with st.expander("➕ Add New Order", expanded=True):
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
    else:
        with st.expander("🛠️ Update Existing Order", expanded=True):
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
    st.write("### 🗃️ Archive")
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
    st.write("### 🗑️ Delete")
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
    st.write("### 📚 Parts Datasheet (Hierarchical View)")
    
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
