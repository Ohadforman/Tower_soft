def render_draw_finalize_tab(P):
    # ==========================================================
    # Imports (local)
    # ==========================================================
    import os, re, time
    import datetime as dt
    from datetime import datetime
    import pandas as pd
    import streamlit as st
    import duckdb

    from helpers.text_utils import safe_str
    from helpers.dataset_io import (
        append_rows_to_dataset_csv,
        resolve_dataset_csv_path,
        ensure_dataset_dir,
    )
    from helpers.dates import compute_next_planned_draw_date
    from helpers.match_utils import norm_str, alt_names
    from helpers.tm_ops import (
        append_preform_length,
        get_most_recent_dataset_csv,
        is_pm_draw_from_dataset_csv,
        decrement_sap_rods_set_by_one,
        mark_draw_order_failed_by_dataset_csv,
        reset_failed_order_to_beginning_and_schedule,
    )
    from hooks.after_done import run_after_done_hook

    st.title("✅ Draw Finalize")
    st.caption("Mark orders as ✅ Done / ❌ Failed from a selected dataset CSV. (Moved out of Dashboard)")

    # ==========================================================
    # Constants
    # ==========================================================
    ORDERS_FILE = P.orders_csv
    DATASET_DIR = P.dataset_dir
    SAP_INVENTORY_FILE = P.sap_rods_inventory_csv
    PREFORMS_FILE = P.preform_inventory_csv

    MAINT_FOLDER = P.maintenance_dir
    DB_PATH = P.duckdb_path

    # ✅ Fault logs (same as Maintenance tab)
    os.makedirs(MAINT_FOLDER, exist_ok=True)
    FAULTS_CSV = os.path.join(MAINT_FOLDER, "faults_log.csv")
    FAULTS_ACTIONS_CSV = os.path.join(MAINT_FOLDER, "faults_actions_log.csv")

    FAULTS_COLS = [
        "fault_id",
        "fault_ts",
        "fault_component",
        "fault_title",
        "fault_description",
        "fault_severity",
        "fault_actor",
        "fault_source_file",
        "fault_related_draw",
    ]

    FAULTS_ACTIONS_COLS = [
        "fault_action_id",
        "fault_id",
        "action_ts",
        "action_type",     # close / reopen / note
        "actor",
        "note",
        "fix_summary",
    ]

    ensure_dataset_dir(DATASET_DIR)

    # ==========================================================
    # DuckDB connection (shared with SQL Lab + Maintenance)
    # ==========================================================
    if "sql_duck_con" not in st.session_state:
        st.session_state["sql_duck_con"] = duckdb.connect(DB_PATH)
    con = st.session_state["sql_duck_con"]
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass

    # Create tables if missing (same schema as Maintenance tab)
    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_events (
        fault_id        BIGINT,
        fault_ts        TIMESTAMP,
        component       VARCHAR,
        title           VARCHAR,
        description     VARCHAR,
        severity        VARCHAR,
        actor           VARCHAR,
        source_file     VARCHAR,
        related_draw    VARCHAR
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_actions (
        fault_action_id  BIGINT,
        fault_id         BIGINT,
        action_ts        TIMESTAMP,
        action_type      VARCHAR,
        actor            VARCHAR,
        note             VARCHAR,
        fix_summary      VARCHAR
    );
    """)

    # ==========================================================
    # CSV helpers (append-only)
    # ==========================================================
    def _ensure_csv(path: str, cols: list):
        if not os.path.isfile(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)

    def _append_csv(path: str, cols: list, df_rows: pd.DataFrame):
        _ensure_csv(path, cols)
        df = df_rows.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

        # stringify time fields to avoid dtype crash
        for tcol in [c for c in cols if c.endswith("_ts")]:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        df.to_csv(path, mode="a", header=False, index=False)

    def _read_csv_safe(path: str, cols: list) -> pd.DataFrame:
        if not os.path.isfile(path):
            return pd.DataFrame(columns=cols)
        try:
            df = pd.read_csv(path)
            if df is None:
                return pd.DataFrame(columns=cols)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols].copy()
        except Exception:
            return pd.DataFrame(columns=cols)

    def _latest_fault_state(actions_df: pd.DataFrame) -> dict:
        """
        fault_id -> last action (close/reopen/note) ; closed if last action is 'close'
        """
        out = {}
        if actions_df is None or actions_df.empty:
            return out

        a = actions_df.copy()
        a["action_ts"] = pd.to_datetime(a["action_ts"], errors="coerce")
        a["fault_id"] = pd.to_numeric(a["fault_id"], errors="coerce")
        a = a.dropna(subset=["fault_id"]).copy()
        a["fault_id"] = a["fault_id"].astype(int)

        a = a.sort_values(["fault_id", "action_ts"], ascending=[True, True])
        last = a.groupby("fault_id").tail(1)

        for _, r in last.iterrows():
            fid = int(r["fault_id"])
            typ = safe_str(r.get("action_type", "")).strip().lower()
            out[fid] = {
                "is_closed": (typ == "close"),
                "last_ts": r.get("action_ts", None),
                "last_note": safe_str(r.get("note", "")),
                "last_fix": safe_str(r.get("fix_summary", "")),
                "last_type": typ,
                "last_actor": safe_str(r.get("actor", "")),
            }
        return out

    def _write_fault_action(con, *, fault_id: int, action_type: str, actor: str, note: str = "", fix_summary: str = ""):
        now_dt = dt.datetime.now()
        aid = int(time.time() * 1000)

        try:
            con.execute("""
                INSERT INTO faults_actions
                (fault_action_id, fault_id, action_ts, action_type, actor, note, fix_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [aid, int(fault_id), now_dt, str(action_type), str(actor), str(note), str(fix_summary)])
        except Exception as e:
            st.warning(f"Fault action DB insert failed (still saving CSV): {e}")

        row = pd.DataFrame([{
            "fault_action_id": aid,
            "fault_id": int(fault_id),
            "action_ts": now_dt,
            "action_type": str(action_type),
            "actor": str(actor),
            "note": str(note),
            "fix_summary": str(fix_summary),
        }])
        _append_csv(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS, row)

    # ==========================================================
    # Actor (same key as Maintenance so you type once)
    # ==========================================================
    st.session_state.setdefault("maint_actor", "operator")
    st.text_input("Actor / operator name (for history)", key="maint_actor")
    actor = st.session_state.get("maint_actor", "operator")

    # ==========================================================
    # Component list helper (try to reuse Maintenance task components)
    # ==========================================================
    normalize_map = {
        "equipment": "Component",
        "component": "Component",
        "task name": "Task",
        "task": "Task",
    }

    def _norm_colname(c: str) -> str:
        return str(c).strip().lower()

    def _load_components_from_maintenance_folder(folder: str) -> list:
        comps = set()

        # 1) from existing faults log
        fdf = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
        if not fdf.empty and "fault_component" in fdf.columns:
            for x in fdf["fault_component"].astype(str).fillna("").tolist():
                x = str(x).strip()
                if x:
                    comps.add(x)

        # 2) from maintenance task files (best)
        if not os.path.isdir(folder):
            return sorted(comps)

        files = [f for f in os.listdir(folder) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        for fname in files:
            p = os.path.join(folder, fname)
            try:
                if p.lower().endswith(".csv"):
                    df = pd.read_csv(p)
                else:
                    df = pd.read_excel(p)
                if df is None or df.empty:
                    continue
                # normalize minimal
                df = df.rename(columns={c: normalize_map.get(_norm_colname(c), c) for c in df.columns})
                if "Component" not in df.columns:
                    continue
                for x in df["Component"].astype(str).fillna("").tolist():
                    x = str(x).strip()
                    if x:
                        comps.add(x)
            except Exception:
                continue

        return sorted(comps)

    @st.cache_data(show_spinner=False)
    def _cached_components(folder: str) -> list:
        return _load_components_from_maintenance_folder(folder)

    # ==========================================================
    # Short-lived message window (under Done / Failed)
    # ==========================================================
    FLASH_SECONDS = 6  # window visible for N seconds

    def _set_flash(level: str, title: str, details: str = ""):
        st.session_state["_finalize_flash"] = {
            "ts": time.time(),
            "level": level,      # "success" | "warning" | "info" | "error"
            "title": title,
            "details": details or "",
            "just_set": True,
        }

    def _render_flash_window(where: str):
        flash = st.session_state.get("_finalize_flash")
        if not flash:
            return

        now = time.time()
        if flash.get("just_set"):
            flash["just_set"] = False
            st.session_state["_finalize_flash"] = flash
        else:
            age = now - float(flash.get("ts", 0))
            if age > FLASH_SECONDS:
                st.session_state.pop("_finalize_flash", None)
                return

        try:
            st.autorefresh(interval=1000, limit=FLASH_SECONDS + 2, key=f"finalize_flash_refresh_{where}")
        except Exception:
            pass

        with st.container(border=True):
            lvl = flash.get("level", "info")
            title = flash.get("title", "")
            details = flash.get("details", "")

            if lvl == "success":
                st.success(title)
            elif lvl == "warning":
                st.warning(title)
            elif lvl == "error":
                st.error(title)
            else:
                st.info(title)

            if details:
                st.caption(details)

    # ==========================================================
    # Dataset CSV context
    # ==========================================================
    recent_csv_files = (
        sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")])
        if os.path.exists(DATASET_DIR) else []
    )

    latest_csv = get_most_recent_dataset_csv(DATASET_DIR)
    st.caption(f"Most recent dataset CSV: **{latest_csv if latest_csv else 'None'}**")

    # ==========================================================
    # Helpers (shared)
    # ==========================================================
    def dataset_csv_path(name_or_path: str, dataset_dir: str) -> str:
        return resolve_dataset_csv_path(name_or_path, dataset_dir=dataset_dir)

    def _norm_col(series: pd.Series) -> pd.Series:
        return (
            series.astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
        )

    def _ensure_orders_schema(orders: pd.DataFrame) -> pd.DataFrame:
        orders.columns = [str(c).replace("\ufeff", "").strip() for c in orders.columns]
        for col, default in {
            "Status": "Pending",
            "Active CSV": "",
            "Done CSV": "",
            "Done Description": "",
            "Done Timestamp": "",
            "Failed CSV": "",
            "Failed Description": "",
            "Failed Timestamp": "",
            "Preform Length After Draw (cm)": "",
            "Next Planned Draw Date": "",
            "T&M Moved": False,
            "T&M Moved Timestamp": "",
            "Status Updated At": "",
            "Assigned Dataset CSV": "",
        }.items():
            if col not in orders.columns:
                orders[col] = default
        return orders

    def _match_order_row(orders: pd.DataFrame, dataset_csv_filename: str) -> pd.Series:
        target = norm_str(dataset_csv_filename)
        target_alts = alt_names(target)

        cols_to_check = []
        for c in ["Assigned Dataset CSV", "Active CSV", "Done CSV", "Failed CSV"]:
            if c in orders.columns:
                cols_to_check.append(c)

        if not cols_to_check:
            return pd.Series([False] * len(orders))

        m = pd.Series([False] * len(orders))
        for c in cols_to_check:
            normed = _norm_col(orders[c])
            for t in target_alts:
                m = m | (normed == t) | normed.str.endswith(t, na=False) | normed.str.contains(re.escape(t), na=False)

        return m

    def _read_dataset_params(target_csv: str):
        p = dataset_csv_path(target_csv, DATASET_DIR)
        if not p or not os.path.exists(p):
            return None, f"Dataset CSV not found: {p}"
        try:
            dfp = pd.read_csv(p, keep_default_na=False)
            return dfp, ""
        except Exception as e:
            return None, f"Failed reading dataset CSV: {e}"

    # ==========================================================
    # Orders CSV: mark DONE / FAILED
    # ==========================================================
    def mark_draw_order_done_by_dataset_csv(dataset_csv_filename: str, done_desc: str, preform_len_after_cm: float):
        if not os.path.exists(ORDERS_FILE):
            return False, f"{ORDERS_FILE} not found (couldn't mark order done)."

        try:
            orders = pd.read_csv(ORDERS_FILE, keep_default_na=False)
        except Exception as e:
            return False, f"Failed reading {ORDERS_FILE}: {e}"

        orders = _ensure_orders_schema(orders)
        match = _match_order_row(orders, dataset_csv_filename)

        if not match.any():
            sample_active = _norm_col(orders.get("Active CSV", pd.Series([], dtype=str))).dropna().unique()[:12].tolist()
            return False, (
                f"No matching row found in draw_orders.csv for '{dataset_csv_filename}'.\n"
                f"Sample Done/Active CSV values: {sample_active}"
            )

        if match.sum() > 1:
            return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        orders.loc[match, "Status"] = "Done"
        orders.loc[match, "Done CSV"] = os.path.basename(dataset_csv_filename)
        orders.loc[match, "Done Description"] = str(done_desc).strip()
        orders.loc[match, "Preform Length After Draw (cm)"] = float(preform_len_after_cm)
        orders.loc[match, "Done Timestamp"] = now_str
        orders.loc[match, "Status Updated At"] = now_str

        orders.loc[match, "Failed CSV"] = ""
        orders.loc[match, "Failed Description"] = ""
        orders.loc[match, "Failed Timestamp"] = ""
        orders.loc[match, "Next Planned Draw Date"] = ""

        if "Assigned Dataset CSV" in orders.columns:
            cur = orders.loc[match, "Assigned Dataset CSV"].astype(str).iloc[0].strip()
            if cur == "" or cur.lower() == "nan":
                orders.loc[match, "Assigned Dataset CSV"] = os.path.basename(dataset_csv_filename)

        orders.to_csv(ORDERS_FILE, index=False)
        return True, "Order marked as Done."

    def append_failed_metadata_to_dataset_csv(dataset_csv_filename: str, failed_desc: str, preform_left_cm: float):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = [
            {"Parameter Name": "Failed Description", "Value": str(failed_desc).strip(), "Units": ""},
            {"Parameter Name": "Preform Length After Failed Draw", "Value": float(preform_left_cm), "Units": "cm"},
            {"Parameter Name": "Failed Timestamp", "Value": now_str, "Units": ""},
        ]
        return append_rows_to_dataset_csv(dataset_csv_filename, rows, dataset_dir=DATASET_DIR)

    # ==========================================================
    # Fault logging helper (used in Failed tab)
    # ==========================================================
    def log_fault_event_for_draw(
        *,
        con,
        actor: str,
        fault_component: str,
        severity: str,
        title: str,
        description: str,
        source_file: str,
        related_draw: str,
    ):
        now_dt = dt.datetime.now()
        fid = int(time.time() * 1000)

        # DuckDB
        try:
            con.execute("""
                INSERT INTO faults_events
                (fault_id, fault_ts, component, title, description, severity, actor, source_file, related_draw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fid, now_dt,
                str(fault_component), str(title), str(description),
                str(severity), str(actor), str(source_file), str(related_draw)
            ])
        except Exception as e:
            st.warning(f"DuckDB fault insert failed (still saving CSV): {e}")

        # CSV
        row = pd.DataFrame([{
            "fault_id": fid,
            "fault_ts": now_dt,
            "fault_component": str(fault_component),
            "fault_title": str(title),
            "fault_description": str(description),
            "fault_severity": str(severity),
            "fault_actor": str(actor),
            "fault_source_file": str(source_file),
            "fault_related_draw": str(related_draw),
        }])
        _append_csv(FAULTS_CSV, FAULTS_COLS, row)
        return fid

    # ==========================================================
    # Dataset picker (shared UI)
    # ==========================================================
    st.markdown("---")
    st.subheader("🎯 Select Target Dataset CSV")

    pick_mode = st.radio(
        "Target selection",
        options=["Most recent", "Choose from list"],
        horizontal=True,
        key="finalize_pick_mode",
    )

    if pick_mode == "Most recent":
        target_csv = latest_csv
    else:
        target_csv = st.selectbox(
            "Choose a dataset CSV",
            options=[""] + recent_csv_files,
            index=0,
            key="finalize_choose_csv",
        ) or None

    if not target_csv:
        st.info("Select a dataset CSV to enable Done/Failed actions.")
        st.stop()

    st.success(f"Target: **{target_csv}**")

    related_draw_default = os.path.splitext(os.path.basename(target_csv))[0]

    # Show matched order preview
    if os.path.exists(ORDERS_FILE):
        try:
            orders_preview = pd.read_csv(ORDERS_FILE, keep_default_na=False)
            orders_preview = _ensure_orders_schema(orders_preview)
            match = _match_order_row(orders_preview, target_csv)
            if match.any() and match.sum() == 1:
                r = orders_preview.loc[match].iloc[0]
                st.caption(
                    f"Matched order: **PF {r.get('Preform Number','')}** | "
                    f"Project: **{r.get('Fiber Project','')}** | "
                    f"Status: **{r.get('Status','')}**"
                )
            elif match.sum() > 1:
                st.warning("⚠️ Multiple order rows match this dataset CSV (duplicates).")
            else:
                st.warning("⚠️ No order row matches this dataset CSV in draw_orders.csv.")
        except Exception as e:
            st.warning(f"Order preview error: {e}")

    # ==========================================================
    # Inner tabs: Done / Failed
    # ==========================================================
    tab_done, tab_failed = st.tabs(["✅ Done", "❌ Failed"])

    with tab_done:
        st.subheader("✅ Mark Done")

        done_desc = st.text_area(
            "Done description (what happened / notes)",
            value=st.session_state.get("final_done_desc", ""),
            key="final_done_desc",
            height=100
        )

        preform_len_after_cm = st.number_input(
            "Preform length after draw (cm) — can be 0",
            min_value=0.0,
            value=float(st.session_state.get("final_preform_len_after_cm", 0.0)),
            step=0.5,
            format="%.1f",
            key="final_preform_len_after_cm",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            do_mark_done = st.button(
                "✅ Mark DONE",
                use_container_width=True,
                disabled=(not str(done_desc).strip()),
                key="final_mark_done_btn",
            )
        with c2:
            st.caption("Also appends Done info into dataset CSV.")

        _render_flash_window(where="done")

        if do_mark_done:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            summary_lines = []
            final_level = "success"

            ok_csv, msg_csv = append_rows_to_dataset_csv(
                target_csv,
                [
                    {"Parameter Name": "Preform Length After Draw", "Value": float(preform_len_after_cm), "Units": "cm"},
                    {"Parameter Name": "Done Description", "Value": str(done_desc).strip(), "Units": ""},
                    {"Parameter Name": "Done Timestamp", "Value": now_str, "Units": ""},
                ],
                dataset_dir=DATASET_DIR
            )
            if ok_csv:
                st.toast("✅ Dataset CSV updated", icon="✅")
                summary_lines.append("✅ Dataset CSV updated")
            else:
                st.toast("⚠️ Dataset CSV update failed", icon="⚠️")
                summary_lines.append("⚠️ Dataset CSV update failed")
                final_level = "warning"

            ok, msg = mark_draw_order_done_by_dataset_csv(target_csv, done_desc, float(preform_len_after_cm))
            if not ok:
                st.toast("❌ Failed to mark DONE", icon="❌")
                _set_flash("error", "Finalize FAILED", msg)
                st.rerun()

            st.toast("✅ Order marked DONE", icon="✅")
            summary_lines.append("✅ Order marked DONE")

            try:
                hook_ok, hook_msg = run_after_done_hook(
                    target_csv=target_csv,
                    done_desc=done_desc,
                    preform_len_after_cm=float(preform_len_after_cm),
                    hook_dir=P.hooks_dir,
                    timeout_sec=120,
                )
                if hook_ok:
                    st.toast("✅ After-done hook executed", icon="✅")
                    summary_lines.append("✅ After-done hook executed")
                else:
                    st.toast("⚠️ After-done hook failed", icon="⚠️")
                    summary_lines.append("⚠️ After-done hook failed")
                    final_level = "warning"
            except Exception:
                st.toast("ℹ️ After-done hook skipped", icon="ℹ️")
                summary_lines.append("ℹ️ After-done hook skipped")

            try:
                df_params, err = _read_dataset_params(target_csv)
                if df_params is None:
                    st.toast("⚠️ SAP check failed", icon="⚠️")
                    summary_lines.append("⚠️ SAP check failed")
                    final_level = "warning"
                else:
                    pm_detected = is_pm_draw_from_dataset_csv(df_params)
                    if not pm_detected:
                        st.toast("ℹ️ SAP not updated (not PM)", icon="ℹ️")
                        summary_lines.append("ℹ️ SAP not updated (not PM)")
                    else:
                        inv_ok, inv_msg = decrement_sap_rods_set_by_one(
                            sap_inventory_file=SAP_INVENTORY_FILE,
                            source_draw=related_draw_default,
                            when_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        if inv_ok:
                            st.toast("🧪 SAP inventory updated", icon="✅")
                            summary_lines.append("✅ SAP inventory updated")
                        else:
                            st.toast("⚠️ SAP NOT decremented", icon="⚠️")
                            summary_lines.append("⚠️ SAP NOT decremented")
                            final_level = "warning"
            except Exception:
                st.toast("ℹ️ SAP update skipped", icon="ℹ️")
                summary_lines.append("ℹ️ SAP update skipped")

            try:
                df_params, err = _read_dataset_params(target_csv)
                if df_params is not None and "Parameter Name" in df_params.columns:
                    m = df_params["Parameter Name"].astype(str).str.strip() == "Preform Number"
                    if m.any():
                        pf_name = df_params.loc[m, "Value"].iloc[0]
                        append_preform_length(
                            preforms_file=PREFORMS_FILE,
                            preform_name=str(pf_name),
                            length_cm=float(preform_len_after_cm),
                            source_draw=related_draw_default,
                        )
                        st.toast("📏 Preform length saved", icon="✅")
                        summary_lines.append(f"✅ Preform saved ({pf_name})")
            except Exception:
                st.toast("ℹ️ Preform registry skipped", icon="ℹ️")
                summary_lines.append("ℹ️ Preform registry skipped")

            _set_flash(final_level, "Finalize DONE", "\n".join(summary_lines))
            st.rerun()

    with tab_failed:
        st.subheader("❌ Mark Failed")

        failed_desc = st.text_area(
            "Failed description (why it failed)",
            value=st.session_state.get("final_failed_desc", ""),
            key="final_failed_desc",
            height=100
        )

        preform_left_cm = st.number_input(
            "Preform length after failed draw (cm) — can be 0",
            min_value=0.0,
            value=float(st.session_state.get("final_preform_left_cm", 0.0)),
            step=0.5,
            format="%.1f",
            key="final_preform_left_cm",
        )

        # ======================================================
        # ✅ Fault insert (same idea as Maintenance tab)
        # ======================================================
        st.markdown("### 🚨 Failure → optionally log a Fault / Incident")

        log_as_fault = st.toggle(
            "This failure is a fault / incident (log to Faults)",
            value=bool(st.session_state.get("final_failed_is_fault", False)),
            key="final_failed_is_fault",
        )

        # Defaults/inputs (only when toggle is ON)
        fault_payload = None
        if log_as_fault:
            comps = _cached_components(MAINT_FOLDER)
            comp_options = (comps if comps else []) + ["Other (custom)"]

            c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
            with c1:
                selected_comp = st.selectbox(
                    "Fault component",
                    options=comp_options,
                    key="final_fault_component_select",
                )
                if selected_comp == "Other (custom)":
                    fault_component = st.text_input("Custom component name", key="final_fault_component_custom")
                else:
                    fault_component = selected_comp
            with c2:
                fault_severity = st.selectbox(
                    "Severity",
                    ["low", "medium", "high", "critical"],
                    index=1,
                    key="final_fault_sev_in",
                )
            with c3:
                st.text_input(
                    "Related draw",
                    value=related_draw_default,
                    disabled=True,
                    key="final_fault_related_draw_view",
                )

            fault_title = st.text_input(
                "Fault title",
                placeholder="Short title",
                value=st.session_state.get("final_fault_title_in", ""),
                key="final_fault_title_in",
            )

            fault_desc_extra = st.text_area(
                "Fault description (details / what to check next time)",
                placeholder="You can paste the failed description + more details…",
                value=st.session_state.get("final_fault_desc_in", ""),
                height=120,
                key="final_fault_desc_in",
            )

            fault_source_file = st.text_input(
                "Source file (optional)",
                placeholder="e.g. photo.jpg / email.pdf / log screenshot",
                value=st.session_state.get("final_fault_src_in", ""),
                key="final_fault_src_in",
            )

            # Build payload (we’ll validate right before saving)
            fault_payload = {
                "fault_component": safe_str(fault_component).strip(),
                "fault_severity": safe_str(fault_severity).strip().lower(),
                "fault_title": safe_str(fault_title).strip(),
                "fault_description": safe_str(fault_desc_extra).strip(),
                "fault_source_file": safe_str(fault_source_file).strip(),
                "fault_related_draw": related_draw_default,
            }

            st.caption("Will be saved to **DuckDB faults_events** + **maintenance/faults_log.csv**")

        st.markdown("---")

        do_mark_failed = st.button(
            "❌ Mark FAILED",
            use_container_width=True,
            disabled=(not str(failed_desc).strip()),
            key="final_mark_failed_btn",
        )

        _render_flash_window(where="failed")

        if do_mark_failed:
            summary_lines = []
            final_level = "success"

            # 1) Mark failed in orders
            okf, msgf = mark_draw_order_failed_by_dataset_csv(
                orders_file=ORDERS_FILE,
                dataset_csv_filename=target_csv,
                failed_desc=failed_desc,
                preform_len_after_cm=float(preform_left_cm),
            )
            if not okf:
                st.toast("⚠️ Failed to mark order FAILED", icon="⚠️")
                _set_flash("warning", "Failed to mark order FAILED", msgf)
                st.rerun()

            st.toast("❌ Order marked FAILED", icon="✅")
            summary_lines.append("✅ Order marked FAILED")

            # 2) Append failed metadata into dataset CSV
            ok_csv, msg_csv = append_failed_metadata_to_dataset_csv(target_csv, failed_desc, float(preform_left_cm))
            if ok_csv:
                st.toast("✅ Failed metadata saved to dataset CSV", icon="✅")
                summary_lines.append("✅ Failed metadata saved to dataset CSV")
            else:
                st.toast("⚠️ Failed metadata NOT saved to dataset CSV", icon="⚠️")
                summary_lines.append("⚠️ Failed metadata NOT saved to dataset CSV")
                final_level = "warning"

            # 3) Optional: log fault (same style as Maintenance)
            if log_as_fault:
                try:
                    if not fault_payload:
                        raise RuntimeError("Fault payload missing.")

                    comp = fault_payload["fault_component"]
                    title = fault_payload["fault_title"]
                    desc = fault_payload["fault_description"]

                    # If user didn’t type extra desc, reuse failed_desc
                    if not desc:
                        desc = str(failed_desc).strip()

                    # If user didn’t give title, auto-create from failed_desc
                    if not title:
                        title = (str(failed_desc).strip()[:80] + "…") if len(str(failed_desc).strip()) > 80 else str(failed_desc).strip()

                    if not comp:
                        st.toast("⚠️ Fault component is required", icon="⚠️")
                        summary_lines.append("⚠️ Fault NOT logged (missing component)")
                        final_level = "warning"
                    elif not title and not desc:
                        st.toast("⚠️ Give fault title or description", icon="⚠️")
                        summary_lines.append("⚠️ Fault NOT logged (missing title/desc)")
                        final_level = "warning"
                    else:
                        fid = log_fault_event_for_draw(
                            con=con,
                            actor=actor,
                            fault_component=comp,
                            severity=fault_payload["fault_severity"] or "medium",
                            title=title,
                            description=desc,
                            source_file=fault_payload["fault_source_file"],
                            related_draw=related_draw_default,
                        )
                        st.toast("🚨 Fault logged", icon="✅")
                        summary_lines.append(f"✅ Fault logged (ID {fid})")
                except Exception as e:
                    st.toast("⚠️ Fault logging failed", icon="⚠️")
                    summary_lines.append(f"⚠️ Fault logging failed: {e}")
                    final_level = "warning"

            # ✅ One flash
            _set_flash(final_level, "Finalize FAILED", "\n".join(summary_lines))

            # Next step buttons
            st.info("Next step:")
            c1, c2 = st.columns(2)

            with c1:
                if st.button("📅 Draw next day (reset + schedule)", key="final_failed_schedule_nextday", use_container_width=True):
                    schedule_date = compute_next_planned_draw_date(datetime.now())
                    oks, msgs = reset_failed_order_to_beginning_and_schedule(
                        orders_file=ORDERS_FILE,
                        dataset_csv_filename=target_csv,
                        schedule_date=schedule_date,
                        scheduled_status="Scheduled",
                    )
                    if oks:
                        st.toast("✅ Reset + scheduled", icon="✅")
                        _set_flash("success", "Reset + scheduled", msgs)
                    else:
                        st.toast("⚠️ Reset failed", icon="⚠️")
                        _set_flash("warning", "Reset failed", msgs)
                    st.rerun()

            with c2:
                if st.button("↩ Return to Pending (no schedule)", key="final_failed_return_pending", use_container_width=True):
                    oks, msgs = reset_failed_order_to_beginning_and_schedule(
                        orders_file=ORDERS_FILE,
                        dataset_csv_filename=target_csv,
                        schedule_date="",
                        scheduled_status="Scheduled",
                    )
                    if oks:
                        st.toast("✅ Reset to Pending", icon="✅")
                        _set_flash("success", "Reset to Pending", msgs)
                    else:
                        st.toast("⚠️ Reset failed", icon="⚠️")
                        _set_flash("warning", "Reset failed", msgs)
                    st.rerun()

            # If user doesn’t click next-step, still rerun to show flash cleanly
            st.rerun()
