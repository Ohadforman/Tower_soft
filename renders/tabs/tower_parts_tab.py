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
            border: 0;
            background: transparent;
            padding: 0;
            margin-bottom: 12px;
          }
          .tp-action-help{
            color: rgba(170,204,228,0.78);
            font-size: 0.80rem;
            margin-top: 2px;
          }
          .tp-green-text{
            color: rgba(126, 255, 190, 0.98);
            font-size: 0.88rem;
            font-weight: 650;
            margin: 4px 0 8px 0;
            text-shadow: 0 0 8px rgba(46, 208, 132, 0.20);
          }
          .tp-soft-note{
            color: rgba(186,216,232,0.82);
            font-size: 0.84rem;
            font-weight: 560;
            margin: 4px 0 8px 0;
          }
          .tp-focus-title{
            color: rgba(126, 255, 190, 0.98);
            font-size: 0.90rem;
            font-weight: 760;
            margin: 0 0 8px 0;
            text-shadow: 0 0 10px rgba(46,208,132,0.18);
          }
          .tp-step-shell{
            border: 0;
            border-radius: 0;
            background: transparent;
            padding: 4px 0;
            margin: 8px 0 10px 0;
          }
          .tp-step-shell.is-focus{
            box-shadow: none;
          }
          .tp-step-sub{
            color: rgba(184,220,242,0.84);
            font-size: 0.83rem;
            margin: 2px 0 8px 0;
          }
          .tp-chip-grid{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
          }
          .tp-status-overview{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(108px, 1fr));
            gap: 10px;
            margin: 6px 0 12px 0;
          }
          .tp-status-pill{
            padding: 8px 10px;
            border-radius: 12px;
            border: 1px solid rgba(128,206,255,0.16);
            background: linear-gradient(180deg, rgba(13,26,42,0.32), rgba(8,15,24,0.18));
          }
          .tp-status-pill.is-active{
            border-color: rgba(164,230,255,0.54);
            box-shadow: 0 0 14px rgba(74,170,255,0.14);
            background: linear-gradient(180deg, rgba(24,58,96,0.44), rgba(10,28,52,0.34));
          }
          .tp-status-pill-label{
            color: rgba(190,224,244,0.88);
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 2px;
          }
          .tp-status-pill-value{
            color: rgba(237,248,255,0.98);
            font-size: 1rem;
            font-weight: 850;
          }
          .tp-queue-board{
            display:grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 6px 0 12px 0;
          }
          .tp-queue-card{
            border: 1px solid rgba(128,206,255,0.10);
            border-radius: 10px;
            background: rgba(8,16,28,0.08);
            padding: 10px;
          }
          .tp-queue-title{
            color: rgba(232,246,255,0.98);
            font-size: 0.88rem;
            font-weight: 850;
            margin-bottom: 2px;
          }
          .tp-queue-count{
            color: rgba(126,198,255,0.98);
            font-size: 1.6rem;
            line-height: 1.05;
            font-weight: 900;
            margin-bottom: 4px;
            text-shadow: 0 0 12px rgba(86,180,255,0.24);
          }
          .tp-queue-sub{
            color: rgba(190,224,244,0.86);
            font-size: 0.78rem;
            min-height: 34px;
          }
          .tp-context-card{
            border: 1px solid rgba(128,206,255,0.14);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(10,20,32,0.18), rgba(8,14,22,0.12));
            padding: 10px 12px;
            margin: 8px 0 12px 0;
          }
          .tp-context-card.is-quiet{
            border-color: rgba(128,206,255,0.10);
            background: linear-gradient(180deg, rgba(8,16,26,0.12), rgba(8,14,22,0.08));
            opacity: 0.92;
          }
          .tp-context-grid{
            display:grid;
            grid-template-columns: repeat(5, minmax(0,1fr));
            gap: 8px 12px;
          }
          .tp-context-label{
            color: rgba(166,194,214,0.68);
            font-size: 0.70rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 2px;
          }
          .tp-context-value{
            color: rgba(214,232,246,0.84);
            font-size: 0.84rem;
            font-weight: 600;
          }
          .tp-current-goal{
            border-left: 3px solid rgba(164,230,255,0.48);
            border-radius: 0;
            background: transparent;
            padding: 2px 0 2px 10px;
            margin: 6px 0 8px 0;
            box-shadow: none;
          }
          .tp-current-goal b{
            color: rgba(244,252,255,0.99);
          }
          .tp-current-goal span{
            color: rgba(208,236,250,0.95);
          }
          @media (max-width: 1100px){
            .tp-queue-board{
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .tp-context-grid{
              grid-template-columns: repeat(2, minmax(0,1fr));
            }
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
          div[data-testid="stSegmentedControl"]{
            background: linear-gradient(180deg, rgba(10,20,36,0.32), rgba(8,14,24,0.22));
            border: 1px solid rgba(128,206,255,0.18);
            border-radius: 12px;
            padding: 6px;
            box-shadow: inset 0 0 0 1px rgba(120,200,255,0.06);
          }
          div[data-testid="stSegmentedControl"] button{
            border-radius: 10px !important;
            border: 1px solid rgba(138,214,255,0.26) !important;
            background: linear-gradient(180deg, rgba(24,58,96,0.42), rgba(10,28,52,0.36)) !important;
            color: rgba(224,242,255,0.96) !important;
            box-shadow: none !important;
          }
          div[data-testid="stSegmentedControl"] button:hover{
            border-color: rgba(182,234,255,0.60) !important;
            background: linear-gradient(180deg, rgba(34,80,128,0.58), rgba(14,40,72,0.48)) !important;
          }
          div[data-testid="stSegmentedControl"] button[aria-pressed="true"]{
            border-color: rgba(178,236,255,0.90) !important;
            background: linear-gradient(180deg, rgba(76,168,255,0.96), rgba(30,96,170,0.92)) !important;
            color: rgba(248,252,255,0.99) !important;
            box-shadow: 0 0 14px rgba(88,186,255,0.28) !important;
          }
          div[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
          div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p,
          div[data-testid="stSegmentedControl"] button[aria-pressed="true"] span{
            color: rgba(248,252,255,0.99) !important;
            fill: rgba(248,252,255,0.99) !important;
          }
          div[data-testid="stPills"] [role="radiogroup"]{
            gap: 8px;
          }
          div[data-testid="stPills"] [role="radio"]{
            border-radius: 999px !important;
            border: 1px solid rgba(138,214,255,0.34) !important;
            background: linear-gradient(180deg, rgba(20,50,84,0.42), rgba(10,28,52,0.34)) !important;
            color: rgba(224,242,255,0.96) !important;
            box-shadow: none !important;
          }
          div[data-testid="stPills"] [role="radio"][aria-checked="true"]{
            border-color: rgba(178,236,255,0.90) !important;
            background: linear-gradient(180deg, rgba(76,168,255,0.96), rgba(30,96,170,0.92)) !important;
            color: rgba(248,252,255,0.99) !important;
            box-shadow: 0 0 12px rgba(88,186,255,0.24) !important;
          }
          div[data-testid="stPills"] [role="radio"][aria-checked="true"] *,
          div[data-testid="stPills"] [role="radio"][aria-checked="true"] p,
          div[data-testid="stPills"] [role="radio"][aria-checked="true"] span{
            color: rgba(248,252,255,0.99) !important;
            fill: rgba(248,252,255,0.99) !important;
          }
          div[data-testid="stPills"] [role="radio"]:focus-visible,
          div[data-testid="stSegmentedControl"] button:focus-visible{
            outline: 2px solid rgba(178,236,255,0.68) !important;
            outline-offset: 1px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tp-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-title">🛠️ Tower Parts Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="tp-sub">Track parts orders, move them through approval and ordering, and manage received-item intake.</div>', unsafe_allow_html=True)
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

    def _location_serial_from_name(name: str) -> str:
        import re
        base = str(name or "").strip().upper()
        if not base:
            return ""
        base = re.sub(r"[^A-Z0-9]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_")
        return base[:40]

    @st.cache_data(show_spinner=False)
    def _manual_pdf_signature_cached(manuals_dir: str, dir_mtime: float) -> tuple:
        sig = []
        if os.path.isdir(manuals_dir):
            for fn in sorted(os.listdir(manuals_dir)):
                if fn.lower().endswith(".pdf"):
                    fp = os.path.join(manuals_dir, fn)
                    sig.append((fn, _mtime(fp)))
        return tuple(sig)

    def _render_choice_buttons(
        *,
        label: str,
        options: list[str],
        selected: str,
        key_prefix: str,
        format_func=None,
        per_row: int = 4,
        compact: bool = False,
        show_label: bool = True,
    ) -> str:
        if show_label:
            st.markdown(f"**{label}**")
        current = selected if selected in options else ""
        if compact:
            st.markdown("<div class='tp-chip-grid'>", unsafe_allow_html=True)
        for row_start in range(0, len(options), max(1, per_row)):
            row_opts = options[row_start: row_start + max(1, per_row)]
            cols = st.columns(len(row_opts))
            for idx, opt in enumerate(row_opts):
                with cols[idx]:
                    if st.button(
                        (format_func(opt) if format_func else opt),
                        key=f"{key_prefix}_{opt}",
                        use_container_width=True,
                        type="primary" if opt == current else "secondary",
                    ):
                        current = opt
                        st.session_state[key_prefix] = opt
                        st.rerun()
        if compact:
            st.markdown("</div>", unsafe_allow_html=True)
        return current

    def _status_rank(status: str) -> int:
        try:
            return STATUS_ORDER.index(str(status).strip())
        except Exception:
            return -1
    
    # ✅ Status rename (Needed -> Opened)
    STATUS_ORDER = ["Opened", "Wait for Approval", "Approved", "Ordered", "Received", "Archived"]
    ITEM_TYPE_OPTIONS = ["Part", "Tool", "Consumable"]
    
    # ✅ Single description field (remove Purpose completely)
    BASE_COLUMNS = [
        "Status", "Part Name", "Serial Number",
        "Project Name", "Details",
        "Opened By",
        "Approval Requested From",
        "Approved", "Approved By", "Approval Date",
        "Received Date",
        "Received State",
        "Ordered By", "Date Ordered", "Company",
        "Inventory Synced",
        "Maintenance Component", "Maintenance Task", "Maintenance Task ID", "Wait ID",
    ]
    
    # ---------------- Load / init ----------------
    if os.path.exists(ORDER_FILE):
        orders_df = _read_csv_cached(ORDER_FILE, False, _mtime(ORDER_FILE))
    else:
        orders_df = pd.DataFrame(columns=BASE_COLUMNS)

    orders_df.columns = orders_df.columns.str.strip()
    _orders_schema_changed = False

    # Backward compat: ensure columns exist + map old "Needed" to "Opened"
    for col in BASE_COLUMNS:
        if col not in orders_df.columns:
            orders_df[col] = ""
            _orders_schema_changed = True
    if "Inventory Synced" not in orders_df.columns:
        orders_df["Inventory Synced"] = ""
        _orders_schema_changed = True
    
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
    orders_df["Status"] = orders_df["Status"].replace({
        "Needed": "Opened",
        "needed": "Opened",
        "Shipped": "Ordered",
        "shipped": "Ordered",
        "Installed": "Received",
        "installed": "Received",
        "Archived": "Archived",
        "archived": "Archived",
    })
    
    # Unknown / empty -> Opened
    orders_df["Status"] = orders_df["Status"].apply(lambda s: s if s in STATUS_ORDER else "Opened")

    ensure_inventory_file(inventory_file)
    ensure_locations_file(locations_file)
    seeded_tools_count = ensure_general_tools_seed(inventory_file)
    if seeded_tools_count > 0:
        st.info(f"Seeded {seeded_tools_count} General Tools template rows in inventory.")

    def _normalize_received_sync_state(df_orders: pd.DataFrame) -> tuple[pd.DataFrame, int, int, bool]:
        marked_pending = 0
        marked_intake_ready = 0
        changed = False
        out = df_orders.copy()
        inv_now = load_inventory(inventory_file)

        def _inventory_received_state(part_name: str, serial_number: str, *, strict: bool = False) -> str:
            pn = str(part_name or "").strip().lower()
            sn = str(serial_number or "").strip().lower()
            if not pn:
                return ""
            m = inv_now[inv_now["Part Name"].astype(str).str.strip().str.lower().eq(pn)].copy()
            if sn:
                m_sn = m[m["Serial Number"].astype(str).str.strip().str.lower().eq(sn)]
                if not m_sn.empty:
                    locs = m_sn["Location"].astype(str).str.strip().str.lower()
                    if locs.eq("mounted").any():
                        return "Mounted on machine"
                    if locs.ne("").any():
                        return "Located in inventory"
                    return ""
            elif strict and len(m) != 1:
                return ""
            if m.empty:
                return ""
            locs = m["Location"].astype(str).str.strip().str.lower()
            if locs.eq("mounted").any():
                return "Mounted on machine"
            if locs.ne("").any():
                return "Located in inventory"
            return ""

        for i, r in out.iterrows():
            status = str(r.get("Status", "")).strip().lower()
            inv_synced = str(r.get("Inventory Synced", "")).strip().lower()
            if status == "received":
                part_name = str(r.get("Part Name", "")).strip()
                serial_number = str(r.get("Serial Number", "")).strip()
                recv_state = _inventory_received_state(
                    part_name,
                    serial_number,
                    strict=bool(
                        str(r.get("Maintenance Task ID", "")).strip()
                        or str(r.get("Wait ID", "")).strip()
                    ),
                )
                if recv_state:
                    if str(out.at[i, "Received State"]).strip() != recv_state:
                        out.at[i, "Received State"] = recv_state
                        changed = True
                    if inv_synced != "yes":
                        out.at[i, "Inventory Synced"] = "Yes"
                        marked_intake_ready += 1
                        changed = True
                else:
                    desired_state = "Waiting for inventory action"
                    if str(out.at[i, "Received State"]).strip() != desired_state:
                        out.at[i, "Received State"] = desired_state
                        changed = True
                    if inv_synced != "pending":
                        out.at[i, "Inventory Synced"] = "Pending"
                        marked_pending += 1
                        changed = True
            elif status != "received":
                if str(out.at[i, "Inventory Synced"]).strip() != "":
                    out.at[i, "Inventory Synced"] = ""
                    changed = True
                if str(out.at[i, "Received State"]).strip() != "":
                    out.at[i, "Received State"] = ""
                    changed = True
        return out, marked_pending, marked_intake_ready, changed

    orders_df, pending_new_count, intake_marked_count, normalized_changed = _normalize_received_sync_state(orders_df)
    if _orders_schema_changed or normalized_changed:
        orders_df.to_csv(ORDER_FILE, index=False)
    if pending_new_count > 0:
        st.info(f"{pending_new_count} received order(s) are waiting for inventory action.")
    if intake_marked_count > 0:
        st.success(f"{intake_marked_count} received order(s) already have inventory location and were marked as organized.")
    
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
    
    queue_counts = {
        "Opened": int(orders_df["Status"].astype(str).str.strip().eq("Opened").sum()) if not orders_df.empty else 0,
        "Wait for Approval": int(orders_df["Status"].astype(str).str.strip().eq("Wait for Approval").sum()) if not orders_df.empty else 0,
        "Approved": int(orders_df["Status"].astype(str).str.strip().eq("Approved").sum()) if not orders_df.empty else 0,
        "Received Pending": int(
            (
                orders_df["Status"].astype(str).str.strip().eq("Received")
                & orders_df["Inventory Synced"].astype(str).str.strip().str.lower().ne("yes")
            ).sum()
        ) if not orders_df.empty else 0,
    }
    maintenance_urgent_n = int(
        (
            orders_df["Status"].astype(str).str.strip().isin(["Opened", "Wait for Approval", "Approved", "Ordered", "Received"])
            & orders_df.get("Maintenance Task ID", pd.Series("", index=orders_df.index)).astype(str).str.strip().ne("")
        ).sum()
    ) if not orders_df.empty else 0

    st.markdown('<div class="tp-section">⚡ Orders Requiring Action</div>', unsafe_allow_html=True)
    st.caption("Start here: approval queue, ready-to-order items, received items still waiting for inventory, and live maintenance-linked orders.")
    st.markdown(
        f"""
        <div class="tp-queue-board">
          <div class="tp-queue-card">
            <div class="tp-queue-title">Wait for Approval</div>
            <div class="tp-queue-count">{queue_counts["Wait for Approval"]}</div>
            <div class="tp-queue-sub">Requests opened and now waiting for approval.</div>
          </div>
          <div class="tp-queue-card">
            <div class="tp-queue-title">Ready to Order</div>
            <div class="tp-queue-count">{queue_counts["Approved"]}</div>
            <div class="tp-queue-sub">Approved items that should move into a real order.</div>
          </div>
          <div class="tp-queue-card">
            <div class="tp-queue-title">Received Pending Inventory</div>
            <div class="tp-queue-count">{queue_counts["Received Pending"]}</div>
            <div class="tp-queue-sub">Already received, but inventory action still not done.</div>
          </div>
          <div class="tp-queue-card">
            <div class="tp-queue-title">Maintenance-Linked Live</div>
            <div class="tp-queue-count">{maintenance_urgent_n}</div>
            <div class="tp-queue-sub">Live orders that came from maintenance tasks and waits.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    qa1, qa2, qa3, qa4, qa5 = st.columns(5)
    with qa1:
        if st.button("Open Opened", key="tp_queue_opened", use_container_width=True):
            st.session_state["parts_manage_action"] = "Update Existing Order"
            st.session_state["parts_manage_status_filter"] = "Opened"
            st.rerun()
    with qa2:
        if st.button("Open Approval Queue", key="tp_queue_wait_approval", use_container_width=True):
            st.session_state["parts_manage_action"] = "Update Existing Order"
            st.session_state["parts_manage_status_filter"] = "Wait for Approval"
            st.rerun()
    with qa3:
        if st.button("Open Ready To Order", key="tp_queue_approved", use_container_width=True):
            st.session_state["parts_manage_action"] = "Update Existing Order"
            st.session_state["parts_manage_status_filter"] = "Approved"
            st.rerun()
    with qa4:
        if st.button("Open Received Pending", key="tp_queue_received_pending", use_container_width=True):
            st.session_state["parts_manage_action"] = "Update Existing Order"
            st.session_state["parts_manage_status_filter"] = "Received"
            st.rerun()
    with qa5:
        if st.button("Open All Orders", key="tp_queue_all", use_container_width=True):
            st.session_state["parts_manage_action"] = "Update Existing Order"
            st.session_state["parts_manage_status_filter"] = "All"
            st.rerun()

    # =========================
    # TABLE (FIRST)
    # =========================
    st.markdown('<div class="tp-section">📋 Orders Table</div>', unsafe_allow_html=True)

    def _order_origin_label(row) -> str:
        task_id = str(row.get("Maintenance Task ID", "")).strip()
        wait_id = str(row.get("Wait ID", "")).strip()
        comp = str(row.get("Maintenance Component", "")).strip()
        if task_id:
            base = "Maintenance"
            if comp:
                base = f"Maintenance · {comp}"
            return f"{base} · {task_id}"
        if wait_id:
            return f"Maintenance wait · {wait_id}"
        return "Manual / PM"

    def _order_flow_state_label(row) -> str:
        status = str(row.get("Status", "")).strip()
        inv_synced = str(row.get("Inventory Synced", "")).strip().lower()
        recv_state = str(row.get("Received State", "")).strip()
        if status == "Opened":
            return "Request opened, not yet sent for approval"
        if status == "Wait for Approval":
            return "Waiting for approval decision"
        if status == "Approved":
            return "Approved and ready to order"
        if status == "Ordered":
            return "Order placed, waiting to receive"
        if status == "Received":
            if inv_synced == "yes":
                return "Received and inventory-synced"
            if recv_state == "Waiting for inventory action":
                return "Received, waiting inventory action"
            return "Received, inventory follow-up needed"
        if status == "Archived":
            return "Closed and archived"
        return "Review order state"
    
    column_order = [
        "Status",
        "Flow State",
        "Origin",
        "Part Name",
        "Qty",
        "Serial Number",
        "Project Name",
        "Details",
        "Opened By",
        "Approved",
        "Approved By",
        "Approval Date",
        "Received Date",
        "Received State",
        "Ordered By",
        "Date Ordered",
        "Company",
    ]
    for col in column_order:
        if col not in orders_df.columns:
            orders_df[col] = ""
    
    tmp = orders_df.copy()
    tmp["Origin"] = tmp.apply(_order_origin_label, axis=1)
    tmp["Flow State"] = tmp.apply(_order_flow_state_label, axis=1)
    tmp["Qty"] = 1
    tmp["__status_sort__"] = pd.Categorical(tmp["Status"], categories=STATUS_ORDER, ordered=True)
    if "Date Ordered" in tmp.columns:
        tmp["__date_sort__"] = pd.to_datetime(tmp["Date Ordered"], errors="coerce")
        tmp = tmp.sort_values(["__status_sort__", "__date_sort__", "Part Name"], ascending=[True, False, True], na_position="last")
        tmp = tmp.drop(columns=["__status_sort__", "__date_sort__"])
    else:
        tmp = tmp.sort_values(["__status_sort__", "Part Name"], na_position="last").drop(columns="__status_sort__")

    # PM-facing table: collapse maintenance-linked duplicate rows from the same wait/task into one visible line.
    if not tmp.empty:
        tmp["_group_key"] = ""
        maint_group_mask = (
            tmp["Wait ID"].astype(str).str.strip().ne("")
            & tmp["Maintenance Task ID"].astype(str).str.strip().ne("")
            & tmp["Part Name"].astype(str).str.strip().ne("")
        )
        tmp.loc[maint_group_mask, "_group_key"] = (
            tmp.loc[maint_group_mask, "Status"].astype(str).str.strip()
            + "||" + tmp.loc[maint_group_mask, "Wait ID"].astype(str).str.strip()
            + "||" + tmp.loc[maint_group_mask, "Maintenance Task ID"].astype(str).str.strip()
            + "||" + tmp.loc[maint_group_mask, "Part Name"].astype(str).str.strip().str.lower()
            + "||" + tmp.loc[maint_group_mask, "Received State"].astype(str).str.strip()
        )
        grouped_rows = []
        used_keys = set()
        for _, row in tmp.iterrows():
            gk = str(row.get("_group_key", "")).strip()
            if not gk:
                grouped_rows.append(row.drop(labels=["_group_key"]).to_dict())
                continue
            if gk in used_keys:
                continue
            used_keys.add(gk)
            grp = tmp[tmp["_group_key"].astype(str).eq(gk)].copy()
            base = grp.iloc[0].drop(labels=["_group_key"]).to_dict()
            base["Qty"] = int(len(grp))
            base["Flow State"] = _order_flow_state_label(base)
            detail_list = []
            for dv in grp["Details"].astype(str).tolist():
                dv_s = str(dv).strip()
                if dv_s and dv_s not in detail_list:
                    detail_list.append(dv_s)
            if detail_list:
                base["Details"] = detail_list[0]
            grouped_rows.append(base)
        tmp = pd.DataFrame(grouped_rows)
        for col in column_order:
            if col not in tmp.columns:
                tmp[col] = ""

    # Color status cell only (cleaner dark-friendly colors)
    def highlight_status(row):
        color_map = {
            "Opened": "background-color: rgba(255,102,102,0.28); color: #ffd4d4; font-weight: 800;",
            "Wait for Approval": "background-color: rgba(255,196,87,0.22); color: #ffe0a6; font-weight: 800;",
            "Approved": "background-color: rgba(105,240,174,0.24); color: #c8ffd8; font-weight: 800;",
            "Ordered": "background-color: rgba(255,214,102,0.24); color: #ffe9b8; font-weight: 800;",
            "Received": "background-color: rgba(92,214,122,0.30); color: #d7ffe1; font-weight: 800;",
            "Archived": "background-color: rgba(160,168,182,0.20); color: #e8edf6; font-weight: 800;",
        }
        s = str(row.get("Status", "")).strip()
        styles = [""] * len(row)
        # Status column index after adding row number
        if "Status" in row.index:
            styles[list(row.index).index("Status")] = color_map.get(s, "")
        if "Origin" in row.index:
            origin = str(row.get("Origin", "")).strip()
            if origin.startswith("Maintenance"):
                styles[list(row.index).index("Origin")] = "background-color: rgba(82,196,255,0.16); color: #d9f4ff; font-weight: 760;"
            else:
                styles[list(row.index).index("Origin")] = "color: rgba(204,224,238,0.84);"
        if "Received State" in row.index:
            rstate = str(row.get("Received State", "")).strip()
            state_style = {
                "Waiting for inventory action": "background-color: rgba(255,196,87,0.18); color: #ffe2aa; font-weight: 740;",
                "Located in inventory": "background-color: rgba(92,214,122,0.18); color: #d8ffe2; font-weight: 740;",
                "Mounted on machine": "background-color: rgba(92,196,255,0.18); color: #d7f3ff; font-weight: 740;",
            }.get(rstate, "")
            if state_style:
                styles[list(row.index).index("Received State")] = state_style
        if "Flow State" in row.index:
            flow_state = str(row.get("Flow State", "")).strip().lower()
            flow_style = ""
            if "waiting for approval" in flow_state or "waiting inventory action" in flow_state or "follow-up needed" in flow_state:
                flow_style = "background-color: rgba(255,184,77,0.18); color: #ffe0a9; font-weight: 760;"
            elif "ready to order" in flow_state or "inventory-synced" in flow_state:
                flow_style = "background-color: rgba(92,214,122,0.18); color: #d8ffe2; font-weight: 760;"
            elif "not yet sent" in flow_state or "waiting to receive" in flow_state:
                flow_style = "background-color: rgba(92,196,255,0.18); color: #d7f3ff; font-weight: 740;"
            elif "archived" in flow_state:
                flow_style = "background-color: rgba(160,168,182,0.16); color: #e8edf6; font-weight: 740;"
            if flow_style:
                styles[list(row.index).index("Flow State")] = flow_style
        return styles
    
    if not tmp.empty:
        st.caption("Showing all orders. Use the update workspace below to focus on a specific order if needed.")
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
        st.info("No orders yet.")
    
    st.divider()
    
    # =========================
    # CLEAN POP AREA (AFTER TABLE)
    # =========================
    st.markdown('<div class="tp-section">✍️ Manage Orders</div>', unsafe_allow_html=True)

    if "parts_manage_action" not in st.session_state:
        st.session_state["parts_manage_action"] = None
    if "parts_manage_status_filter" not in st.session_state:
        st.session_state["parts_manage_status_filter"] = "All"
    if st.session_state["parts_manage_action"] not in ["Add New Order", "Update Existing Order", None]:
        st.session_state["parts_manage_action"] = None
    if st.session_state["parts_manage_status_filter"] not in ["All"] + STATUS_ORDER:
        st.session_state["parts_manage_status_filter"] = "All"

    status_counts = (
        orders_df["Status"].astype(str).str.strip().value_counts().to_dict()
        if not orders_df.empty else {}
    )

    st.markdown('<div class="tp-action-card">', unsafe_allow_html=True)
    action = str(st.session_state.get("parts_manage_action") or "")
    action_c1, action_c2 = st.columns(2)
    with action_c1:
        if st.button(
            "➕ New Order",
            key="parts_manage_action_new",
            use_container_width=True,
            type="primary" if action == "Add New Order" else "secondary",
        ):
            st.session_state["parts_manage_action"] = None if action == "Add New Order" else "Add New Order"
            st.rerun()
    with action_c2:
        if st.button(
            "🛠️ Update Order",
            key="parts_manage_action_update",
            use_container_width=True,
            type="primary" if action == "Update Existing Order" else "secondary",
        ):
            st.session_state["parts_manage_action"] = None if action == "Update Existing Order" else "Update Existing Order"
            st.rerun()
    action = str(st.session_state.get("parts_manage_action") or "")
    if not action:
        st.markdown("<div class='tp-action-help'>Choose a workspace to continue.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Add New ----------
    if action == "Add New Order":
        st.markdown("#### ➕ Add New Order")
        with st.container(border=True):
            with st.form("add_new_order_form", clear_on_submit=True, enter_to_submit=False):
                c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    
                with c1:
                    part_name = st.text_input("Part Name")
                    serial_number = st.text_input("Serial Number")
                with c2:
                    opened_by = st.text_input("Opened By")
                    selected_project = st.selectbox("Fiber Project", project_options)
                    company = st.text_input("Company (optional)")
                with c3:
                    st.markdown('<div class="tp-green-text">New orders start as Opened. Move them to Wait for Approval in the next step.</div>', unsafe_allow_html=True)

                details = st.text_area("Details", height=120)
    
                save = st.form_submit_button("💾 Save Order", use_container_width=True)
    
                if save:
                    if not part_name.strip():
                        st.error("Part Name is required.")
                    else:
                        new_row = {
                            "Status": "Opened",
                            "Part Name": part_name.strip(),
                            "Serial Number": serial_number.strip(),
                            "Project Name": "" if selected_project == "None" else str(selected_project),
                            "Details": details.strip(),
                            "Opened By": opened_by.strip(),
                            "Approval Requested From": "",
                            "Company": company.strip(),
                            "Approved": "No",
                            "Approved By": "",
                            "Approval Date": "",
                            "Received Date": "",
                            "Received State": "",
                            "Ordered By": "",
                            "Date Ordered": "",
                            "Inventory Synced": "",
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
                st.markdown("<div class='tp-step-shell'>", unsafe_allow_html=True)
                st.markdown("<div class='tp-focus-title'>Find the order</div>", unsafe_allow_html=True)
                st.markdown("<div class='tp-step-sub'>Filter lightly, then pick one order to update.</div>", unsafe_allow_html=True)
                status_filter_options = STATUS_ORDER + ["All"]
                status_labels = {
                    "Opened": "Opened",
                    "Wait for Approval": "Wait Approval",
                    "Approved": "Approved",
                    "Ordered": "Ordered",
                    "Received": "Received",
                    "Archived": "Archived",
                    "All": "All",
                }
                status_focus = _render_choice_buttons(
                    label="Status filter",
                    options=status_filter_options,
                    selected=str(st.session_state.get("parts_manage_status_filter", "All")),
                    key_prefix="parts_manage_status_filter",
                    format_func=lambda s: f"{status_labels.get(s, s)} ({len(orders_df) if s == 'All' else int(status_counts.get(s, 0))})",
                    per_row=4,
                    compact=True,
                )
                filter_c2, filter_c3 = st.columns([1.2, 2.0])
                with filter_c2:
                    order_search = st.text_input(
                        "Search",
                        value="",
                        key="parts_manage_order_search",
                        placeholder="part / serial / project...",
                    ).strip().lower()

                filtered_orders = orders_df.copy()
                if status_focus != "All":
                    filtered_orders = filtered_orders[
                        filtered_orders["Status"].astype(str).str.strip().eq(status_focus)
                    ].copy()
                if order_search:
                    search_blob = (
                        filtered_orders["Part Name"].astype(str).str.lower()
                        + " "
                        + filtered_orders["Serial Number"].astype(str).str.lower()
                        + " "
                        + filtered_orders["Project Name"].astype(str).str.lower()
                        + " "
                        + filtered_orders["Details"].astype(str).str.lower()
                        + " "
                        + filtered_orders.get("Maintenance Component", pd.Series("", index=filtered_orders.index)).astype(str).str.lower()
                        + " "
                        + filtered_orders.get("Maintenance Task", pd.Series("", index=filtered_orders.index)).astype(str).str.lower()
                        + " "
                        + filtered_orders.get("Maintenance Task ID", pd.Series("", index=filtered_orders.index)).astype(str).str.lower()
                        + " "
                        + filtered_orders.get("Wait ID", pd.Series("", index=filtered_orders.index)).astype(str).str.lower()
                    )
                    filtered_orders = filtered_orders[search_blob.str.contains(order_search, na=False)].copy()
                if filtered_orders.empty:
                    msg = "No orders match the current filter."
                    if status_focus != "All":
                        msg = f"No orders currently in `{status_focus}`."
                    st.info(msg)
                    st.session_state["parts_order_filter_sig"] = f"{status_focus}::{order_search}"
                    return
                labels = (
                    filtered_orders["Part Name"].astype(str).fillna("").str.strip()
                    + "  |  "
                    + filtered_orders["Serial Number"].astype(str).fillna("").str.strip()
                    + "  |  "
                    + filtered_orders["Status"].astype(str).fillna("").str.strip()
                )
                label_to_idx = {labels.iloc[i]: filtered_orders.index[i] for i in range(len(labels))}
                order_options = list(label_to_idx.keys())
                filter_sig = f"{status_focus}::{order_search}"
                prev_filter_sig = str(st.session_state.get("parts_order_filter_sig", ""))
                st.session_state["parts_order_filter_sig"] = filter_sig
                select_widget_key = f"order_update_select_widget::{filter_sig}"
                if prev_filter_sig != filter_sig and select_widget_key in st.session_state:
                    del st.session_state[select_widget_key]
                current_pick = str(st.session_state.get(select_widget_key, "") or "")
                if current_pick not in order_options and order_options:
                    st.session_state[select_widget_key] = order_options[0]
                with filter_c3:
                    selected_label = st.selectbox(
                        "Select an order",
                        order_options,
                        key=select_widget_key,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                order_index = label_to_idx[selected_label]
                cur = orders_df.loc[order_index].to_dict()
                cur_status = str(cur.get("Status", "Opened")).strip()
                if cur_status not in STATUS_ORDER:
                    cur_status = "Opened"
                if cur_status in STATUS_ORDER:
                    cur_idx = STATUS_ORDER.index(cur_status)
                    allowed_targets = STATUS_ORDER[cur_idx + 1:]
                else:
                    allowed_targets = []
                if not allowed_targets:
                    allowed_targets = [cur_status]
                target_state_key = f"parts_order_target_status::{order_index}"
                target_status = str(st.session_state.get(target_state_key, cur_status)).strip()
                if target_status not in allowed_targets:
                    target_status = allowed_targets[0] if allowed_targets else cur_status
                    st.session_state[target_state_key] = target_status

                st.markdown("<div class='tp-step-shell is-focus'>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="tp-green-text">Current workflow step: <b>{cur_status}</b></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='tp-step-sub'>Choose the destination step for this update.</div>", unsafe_allow_html=True)
                target_status = _render_choice_buttons(
                    label="Destination step",
                    options=allowed_targets,
                    selected=str(st.session_state.get(target_state_key, cur_status)),
                    key_prefix=target_state_key,
                    format_func=lambda s: s,
                    per_row=len(allowed_targets) if allowed_targets else 1,
                )
                target_status = str(target_status or cur_status).strip()
                st.caption(f"Transition: {cur_status} -> {target_status}")
                step_goal_map = {
                    "Wait for Approval": "Ask for approval and record who needs to approve this order.",
                    "Approved": "Confirm approval so the order becomes ready for purchasing.",
                    "Ordered": "Record supplier and ordering details so the PM queue moves into real purchasing.",
                    "Received": "Capture the receive date first, then finish inventory action once the item is physically handled.",
                    "Archived": "Close the order after the received item has been handled and no more PM action is needed.",
                }
                step_goal = step_goal_map.get(target_status, "Move this order to the next workflow step.")
                st.markdown(
                    f"<div class='tp-current-goal'><b>Current step goal</b><span>{step_goal}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                current_inv_df = load_inventory(inventory_file)
                current_locations_df = load_locations(locations_file)
                current_locations_df = current_locations_df[
                    current_locations_df["Active"].astype(str).str.strip().str.lower().ne("no")
                ].copy()
                current_location_options = sorted(
                    [str(x).strip() for x in current_locations_df["Location Name"].tolist() if str(x).strip()]
                )
                current_loc_serial_map = {
                    str(r.get("Location Name", "")).strip(): str(r.get("Location Serial", "")).strip()
                    for _, r in current_locations_df.iterrows()
                    if str(r.get("Location Name", "")).strip()
                }
                current_component_options = sorted(
                    list({*[str(x).strip() for x in current_inv_df.get("Component", pd.Series([], dtype=str)).tolist() if str(x).strip()], "Tower Parts", "Consumables", "General Tools"})
                )
                received_action_key = f"parts_received_action::{order_index}"
                matched_inv = pd.DataFrame()
                show_received_action_panel = cur_status in ["Received", "Archived"]
                if show_received_action_panel:
                    sel_part = str(cur.get("Part Name", "")).strip()
                    sel_sn = str(cur.get("Serial Number", "")).strip()
                    if not current_inv_df.empty and sel_part:
                        matched_inv = current_inv_df[
                            current_inv_df["Part Name"].astype(str).str.strip().str.lower().eq(sel_part.lower())
                        ].copy()
                        if sel_sn:
                            matched_sn = matched_inv[
                                matched_inv["Serial Number"].astype(str).str.strip().str.lower().eq(sel_sn.lower())
                            ].copy()
                            if not matched_sn.empty:
                                matched_inv = matched_sn

                    default_received_action = "No inventory action"
                    cur_received_state = str(cur.get("Received State", "")).strip()
                    if cur_received_state == "Located in inventory":
                        default_received_action = "Locate in inventory"
                    elif cur_received_state == "Mounted on machine":
                        default_received_action = "Mount on machine"
                    current_received_action = str(
                        st.session_state.get(received_action_key, default_received_action)
                    ).strip()
                    if current_received_action not in ["No inventory action", "Locate in inventory", "Mount on machine"]:
                        current_received_action = default_received_action
                        st.session_state[received_action_key] = current_received_action

                    if not matched_inv.empty:
                        st.markdown(
                            f'<div class="tp-soft-note">Inventory match found: {len(matched_inv)} row(s), qty {float(pd.to_numeric(matched_inv["Quantity"], errors="coerce").fillna(0).sum()):g}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="tp-soft-note">No matching inventory row found yet. The selected action can create a new inventory item.</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('<div class="tp-focus-title">Inventory action</div>', unsafe_allow_html=True)
                    current_received_action = _render_choice_buttons(
                        label="Inventory action",
                        options=["No inventory action", "Locate in inventory", "Mount on machine"],
                        selected=str(st.session_state.get(received_action_key, default_received_action)),
                        key_prefix=received_action_key,
                        per_row=3,
                    )
                    current_received_action = str(current_received_action or default_received_action).strip()
                else:
                    current_received_action = "No inventory action"

                context_pairs = [
                    ("Part", str(cur.get("Part Name", "")).strip() or "—"),
                    ("Serial", str(cur.get("Serial Number", "")).strip() or "—"),
                    ("Origin", _order_origin_label(cur)),
                    ("Opened By", str(cur.get("Opened By", "")).strip() or "—"),
                    ("Ordered From", str(cur.get("Company", "")).strip() or "—"),
                    ("Approval Requested", str(cur.get("Approval Requested From", "")).strip() or "—"),
                    ("Approved By", str(cur.get("Approved By", "")).strip() or "—"),
                    ("Received Date", str(cur.get("Received Date", "")).strip() or "—"),
                    ("Received State", str(cur.get("Received State", "")).strip() or "—"),
                    ("Target Step", target_status or "—"),
                ]
                context_pairs = [pair for pair in context_pairs if str(pair[1]).strip() and str(pair[1]).strip() != "—"]
                context_html = "".join(
                    f"<div><div class='tp-context-label'>{label}</div><div class='tp-context-value'>{value}</div></div>"
                    for label, value in context_pairs
                )
                st.markdown(
                    f"<div class='tp-context-card is-quiet'><div class='tp-context-grid'>{context_html}</div></div>",
                    unsafe_allow_html=True,
                )
                maint_task_id_ctx = str(cur.get("Maintenance Task ID", "")).strip()
                if maint_task_id_ctx:
                    jc1, jc2 = st.columns([1.2, 2.0])
                    with jc1:
                        if st.button("🧰 Open linked maintenance task", key=f"parts_jump_maint_{order_index}", use_container_width=True):
                            st.session_state["selected_tab"] = "🧰 Maintenance"
                            st.session_state["maint_main_group"] = "maintenance"
                            st.session_state["maint_flow_step"] = "2) Execute + Resolve Blocks"
                            st.session_state["maint_open_task_id"] = maint_task_id_ctx
                            st.rerun()
                    with jc2:
                        st.markdown(
                            f"<div class='tp-soft-note'>Linked maintenance task: <b>{maint_task_id_ctx}</b></div>",
                            unsafe_allow_html=True,
                        )

                with st.form("update_order_form", enter_to_submit=False):
                    updated_part_name = str(cur.get("Part Name", ""))
                    updated_serial_number = str(cur.get("Serial Number", ""))
                    cur_proj = str(cur.get("Project Name", ""))
                    updated_project = cur_proj if cur_proj in project_options else "None"
                    updated_opened_by = str(cur.get("Opened By", ""))
                    company = str(cur.get("Company", ""))

                    with st.expander("Optional: edit base order info", expanded=False):
                        c_base1, c_base2 = st.columns(2)
                        with c_base1:
                            updated_part_name = st.text_input("Part Name", value=str(cur.get("Part Name", "")))
                            updated_serial_number = st.text_input("Serial Number", value=str(cur.get("Serial Number", "")))
                        with c_base2:
                            updated_project = st.selectbox(
                                "Fiber Project",
                                project_options,
                                index=project_options.index(cur_proj) if cur_proj in project_options else 0,
                            )
                            updated_opened_by = st.text_input("Opened By", value=str(cur.get("Opened By", "")))

                    st.markdown("<div class='tp-step-shell is-focus'>", unsafe_allow_html=True)
                    st.markdown('<div class="tp-focus-title">Current step input</div>', unsafe_allow_html=True)
                    st.markdown("<div class='tp-step-sub'>Only fields needed for the selected step are shown here.</div>", unsafe_allow_html=True)
                    c_step1, c_step2 = st.columns([1.1, 1.1])

                    with c_step1:
                        pass

                    with c_step2:
                        approval_raw = str(cur.get("Approval Date", ""))
                        approval_dt = pd.to_datetime(approval_raw, errors="coerce")
                        if pd.isna(approval_dt):
                            approval_dt = pd.Timestamp.today()
                        date_ordered_raw = str(cur.get("Date Ordered", ""))
                        date_ordered_dt = pd.to_datetime(date_ordered_raw, errors="coerce")
                        if pd.isna(date_ordered_dt):
                            date_ordered_dt = pd.Timestamp.today()
                        received_raw = str(cur.get("Received Date", ""))
                        received_dt = pd.to_datetime(received_raw, errors="coerce")
                        if pd.isna(received_dt):
                            received_dt = pd.Timestamp.today()

                        cur_rank = _status_rank(cur_status)
                        target_rank = _status_rank(target_status)
                        approved_by = str(cur.get("Approved By", "")).strip()
                        approval_date = approval_dt
                        ordered_by = str(cur.get("Ordered By", "")).strip()
                        date_ordered = date_ordered_dt
                        received_date = received_dt
                        approval_requested_from = str(cur.get("Approval Requested From", "")).strip()
                        received_state = str(cur.get("Received State", "")).strip() or "Waiting for inventory action"
                        inventory_action = "No inventory action"
                        inventory_location = ""
                        inventory_qty = 1.0
                        inventory_serial = str(cur.get("Serial Number", "")).strip()
                        inventory_component = "Tower Parts"

                        needs_approval_request = target_rank >= _status_rank("Wait for Approval") and (not approval_requested_from or cur_rank < _status_rank("Wait for Approval"))
                        needs_approval_confirmation = target_rank >= _status_rank("Approved") and ((not approved_by) or pd.isna(approval_dt) or cur_rank < _status_rank("Approved"))
                        needs_order_data = target_rank >= _status_rank("Ordered") and ((not company.strip()) or (not ordered_by) or pd.isna(date_ordered_dt) or cur_rank < _status_rank("Ordered"))
                        needs_received_date = target_rank >= _status_rank("Received") and (pd.isna(received_dt) or cur_rank < _status_rank("Received"))

                        if needs_approval_request:
                            approval_requested_from = st.text_input(
                                "Approval Requested From",
                                value=approval_requested_from,
                            )

                        if needs_approval_confirmation:
                            approved_by = st.text_input("Approved By", value=approved_by)
                            approval_date = st.date_input("Approval Date", value=approval_dt)

                        if needs_order_data:
                            existing_company = str(cur.get("Company", "")).strip()
                            if existing_company and not needs_order_data:
                                company = existing_company
                            elif existing_company and target_status == "Ordered":
                                st.markdown(
                                    f"<div class='tp-soft-note'>Ordered from: <b>{existing_company}</b></div>",
                                    unsafe_allow_html=True,
                                )
                            company = st.text_input(
                                "Company / Ordered From",
                                value=company.strip() or existing_company,
                                key=f"parts_company_input_{order_index}",
                            )
                            ordered_by = st.text_input("Ordered By", value=ordered_by)
                            date_ordered = st.date_input("Date Ordered", value=date_ordered_dt)

                        if target_status in ["Received", "Archived"] and needs_received_date:
                            received_date = st.date_input("Received Date", value=received_dt)
                        elif target_status == "Archived":
                            st.markdown('<div class="tp-action-help">Received date is already captured from the previous step.</div>', unsafe_allow_html=True)

                        if show_received_action_panel:
                            sel_sn = str(cur.get("Serial Number", "")).strip()
                            if not matched_inv.empty:
                                existing_components = matched_inv["Component"].astype(str).str.strip().replace("", pd.NA).dropna()
                                if existing_components.shape[0]:
                                    inventory_component = str(existing_components.iloc[0])

                            if current_received_action == "No inventory action":
                                st.markdown('<div class="tp-action-help">Keep the order in this step until you are ready to place it in inventory or mount it.</div>', unsafe_allow_html=True)
                            else:
                                with st.container(border=True):
                                    st.markdown('<div class="tp-focus-title">Current inventory action</div>', unsafe_allow_html=True)
                                    if current_received_action == "Locate in inventory":
                                        inventory_location = st.selectbox(
                                            "Inventory location",
                                            options=[""] + current_location_options,
                                            key=f"parts_received_loc_{order_index}",
                                        )
                                    elif current_received_action == "Mount on machine":
                                        inventory_location = "Mounted"
                                        st.markdown('<div class="tp-action-help">This will register the part directly under Mounted.</div>', unsafe_allow_html=True)
                                    inventory_qty = st.number_input(
                                        "Inventory qty",
                                        min_value=0.01,
                                        max_value=10000.0,
                                        value=1.0,
                                        step=0.1,
                                        key=f"parts_received_qty_{order_index}",
                                    )
                                    inventory_serial = st.text_input(
                                        "Inventory serial",
                                        value=sel_sn,
                                        key=f"parts_received_serial_{order_index}",
                                    )
                                    inventory_component = st.selectbox(
                                        "Inventory component",
                                        options=current_component_options or ["Tower Parts"],
                                        index=(current_component_options.index(inventory_component) if inventory_component in current_component_options else 0),
                                        key=f"parts_received_component_{order_index}",
                                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    details = st.text_area("Details", value=str(cur.get("Details", "")), height=120)

                    do_update = st.form_submit_button("✅ Update Order", use_container_width=True)

                    if do_update:
                        orders_df.at[order_index, "Part Name"] = updated_part_name.strip()
                        orders_df.at[order_index, "Serial Number"] = updated_serial_number.strip()
                        orders_df.at[order_index, "Status"] = target_status
    
                        orders_df.at[order_index, "Project Name"] = "" if updated_project == "None" else str(updated_project)
                        orders_df.at[order_index, "Opened By"] = updated_opened_by.strip()
    
                        orders_df.at[order_index, "Details"] = details.strip()
                        orders_df.at[order_index, "Company"] = company.strip() if target_status in ["Ordered", "Received", "Archived"] else str(cur.get("Company", "")).strip()
                        orders_df.at[order_index, "Approval Requested From"] = approval_requested_from.strip() if target_status in ["Wait for Approval", "Approved", "Ordered", "Received", "Archived"] else ""

                        is_approved_step = target_status in ["Approved", "Ordered", "Received", "Archived"]
                        is_ordered_step = target_status in ["Ordered", "Received", "Archived"]

                        orders_df.at[order_index, "Approved"] = "Yes" if is_approved_step else "No"
                        orders_df.at[order_index, "Approved By"] = approved_by.strip() if is_approved_step else ""
                        orders_df.at[order_index, "Approval Date"] = approval_date.strftime("%Y-%m-%d") if is_approved_step else ""

                        orders_df.at[order_index, "Ordered By"] = ordered_by.strip() if is_ordered_step else ""
                        orders_df.at[order_index, "Date Ordered"] = date_ordered.strftime("%Y-%m-%d") if is_ordered_step else ""
                        orders_df.at[order_index, "Received Date"] = received_date.strftime("%Y-%m-%d") if target_status in ["Received", "Archived"] else ""

                        if target_status in ["Received", "Archived"]:
                            if current_received_action == "Locate in inventory":
                                received_state = "Located in inventory"
                            elif current_received_action == "Mount on machine":
                                received_state = "Mounted on machine"
                            elif show_received_action_panel:
                                received_state = "Waiting for inventory action"
                        orders_df.at[order_index, "Received State"] = received_state if target_status in ["Received", "Archived"] else ""

                        if target_status.lower() in ["received", "archived"]:
                            if show_received_action_panel and current_received_action in ["Locate in inventory", "Mount on machine"]:
                                if current_received_action == "Locate in inventory" and not str(inventory_location).strip():
                                    st.error("Choose an inventory location for the received item.")
                                    st.stop()
                                increment_part(
                                    inventory_file,
                                    updated_part_name.strip(),
                                    qty=float(inventory_qty),
                                    component=inventory_component,
                                    serial_number=inventory_serial.strip(),
                                    location=inventory_location if current_received_action == "Locate in inventory" else "Mounted",
                                    location_serial=current_loc_serial_map.get(inventory_location, "MOUNTED" if current_received_action == "Mount on machine" else ""),
                                    notes="Inventory action from received order",
                                )
                                orders_df.at[order_index, "Inventory Synced"] = "Yes"
                            elif show_received_action_panel:
                                orders_df.at[order_index, "Inventory Synced"] = "Pending"
                        elif target_status.lower() not in ["received", "archived"]:
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
        st.caption("Manage part stock and inventory actions for received orders.")

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

        def _upsert_storage_location(loc_name: str, loc_serial: str = "", description: str = "") -> None:
            ln = str(loc_name).strip()
            if not ln:
                return
            ls = str(loc_serial).strip() or _location_serial_from_name(ln)
            desc = str(description).strip()
            ldf = load_locations(locations_file)
            now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if ldf.empty:
                ldf = pd.DataFrame(columns=["Location Name", "Location Serial", "Description", "Active", "Last Updated"])
            mask = ldf["Location Name"].astype(str).str.strip().str.lower().eq(ln.lower())
            if mask.any():
                idx = ldf[mask].index[0]
                if ls:
                    ldf.at[idx, "Location Serial"] = ls
                if desc:
                    ldf.at[idx, "Description"] = desc
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
                                    "Description": desc or "Auto-added from inventory flow",
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

        inv_df = load_inventory(inventory_file)
        if inv_df.empty:
            st.info("Inventory is empty. Add rows below to start organizing tower parts.")

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
        if "Lead Time Days" not in finder_src.columns:
            finder_src["Lead Time Days"] = 0.0
        finder_src["Lead Time Days"] = pd.to_numeric(finder_src["Lead Time Days"], errors="coerce").fillna(0.0)
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
                    "Lead Time Days": float(g["Lead Time Days"].max()) if ("Lead Time Days" in g.columns and not g.empty) else 0.0,
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
                        "Lead Time Days",
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
        if "Lead Time Days" not in inv_status.columns:
            inv_status["Lead Time Days"] = 0.0
        inv_status["Lead Time Days"] = pd.to_numeric(inv_status["Lead Time Days"], errors="coerce").fillna(0.0)
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
        active_status = {"opened", "wait for approval", "approved", "ordered"}
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
                    "Quantity", "Min Level", "Lead Time Days", "Effective Min", "Notes"
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
                    .format({"Quantity": "{:.2f}", "Min Level": "{:.2f}", "Lead Time Days": "{:.0f}", "Effective Min": "{:.2f}"})
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
                            "Approval Requested From": "",
                            "Approved": "No",
                            "Approved By": "",
                            "Approval Date": "",
                            "Received Date": "",
                            "Received State": "",
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
                quick_serial = st.text_input("Serial", key="parts_quick_serial")
            with q6:
                quick_loc_desc = st.text_input("Location note", key="parts_quick_loc_desc", placeholder="optional")
            with q7:
                quick_min = st.number_input("Min Level", min_value=0.0, max_value=10000.0, value=0.0, step=0.1, key="parts_quick_min")
            q8c1, q8c2 = st.columns([1.0, 2.0])
            with q8c1:
                quick_lead_days = st.number_input("Lead Time Days", min_value=0.0, max_value=3650.0, value=0.0, step=1.0, key="parts_quick_lead_days")
            with q8c2:
                st.caption("How many days it usually takes to bring this part after ordering.")

            quick_target = quick_new_part.strip() or quick_part.strip()

            bq1, bq2 = st.columns(2)
            with bq1:
                if st.button("➕ Add Stock", use_container_width=True, key="parts_quick_add"):
                    if quick_target:
                        # If custom/new location was entered, register it in Storage Locations list.
                        if str(quick_loc).strip():
                            _upsert_storage_location(str(quick_loc).strip(), description=quick_loc_desc.strip())
                        quick_loc_serial = _location_serial_from_name(str(quick_loc).strip())
                        increment_part(
                            inventory_file,
                            quick_target,
                            qty=float(quick_delta),
                            component=quick_component,
                            serial_number=quick_serial.strip(),
                            location=str(quick_loc).strip(),
                            location_serial=quick_loc_serial,
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
                                cur.loc[m, "Lead Time Days"] = float(quick_lead_days)
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
            st.caption("Create/edit storage places by name and description. Internal serials are generated automatically.")
            loc_df = load_locations(locations_file)
            loc_edit_src = loc_df[["Location Name", "Description"]].copy() if not loc_df.empty else pd.DataFrame(columns=["Location Name", "Description"])
            loc_edit = st.data_editor(
                loc_edit_src,
                use_container_width=True,
                height=220,
                num_rows="dynamic",
                column_config={
                    "Location Name": st.column_config.TextColumn("Location Name", required=True),
                    "Description": st.column_config.TextColumn("Description"),
                },
                key="parts_locations_editor",
            )
            if st.button("💾 Save Locations", use_container_width=True, key="parts_locations_save_btn", type="primary"):
                now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                existing = load_locations(locations_file)
                loc_save = loc_edit.copy()
                loc_save["Location Name"] = loc_save["Location Name"].astype(str).fillna("").str.strip()
                loc_save["Description"] = loc_save["Description"].astype(str).fillna("").str.strip()
                loc_save = loc_save[loc_save["Location Name"].ne("")].copy()
                rows = []
                for _, rr in loc_save.iterrows():
                    ln = str(rr.get("Location Name", "")).strip()
                    desc = str(rr.get("Description", "")).strip()
                    existing_match = existing[
                        existing["Location Name"].astype(str).str.strip().str.lower().eq(ln.lower())
                    ] if not existing.empty else pd.DataFrame()
                    old = existing_match.iloc[0] if not existing_match.empty else {}
                    rows.append(
                        {
                            "Location Name": ln,
                            "Location Serial": str(getattr(old, "get", lambda *_: "")("Location Serial", "") or _location_serial_from_name(ln)).strip(),
                            "Description": desc,
                            "Active": str(getattr(old, "get", lambda *_: "Yes")("Active", "Yes")).strip() or "Yes",
                            "Last Updated": now_ts,
                        }
                    )
                loc_save = pd.DataFrame(rows, columns=["Location Name", "Location Serial", "Description", "Active", "Last Updated"])
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
            active_status = {"opened", "wait for approval", "approved", "ordered", "received"}
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
                "Approval Requested From": "",
                "Approved": "No",
                "Approved By": "",
                "Approval Date": "",
                "Received Date": "",
                "Received State": "",
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
