def render_maintenance_tab(P):
    import os, json, glob, time
    import datetime as dt
    
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    import duckdb

    st.markdown(
        """
        <style>
          .maint-title{
            font-size: 1.62rem;
            font-weight: 900;
            margin: 0 0 2px 0;
            padding-top: 6px;
            line-height: 1.2;
            color: rgba(236,248,255,0.98);
            text-shadow: 0 0 14px rgba(86,178,255,0.22);
          }
          .maint-sub{
            margin: 0 0 8px 0;
            font-size: 0.90rem;
            color: rgba(188,224,248,0.88);
          }
          .maint-top-spacer{
            height: 6px;
          }
          .maint-line{
            height: 1px;
            margin: 0 0 12px 0;
            background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
          }
          .maint-help{
            border: 1px solid rgba(128,206,255,0.22);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(14,32,56,0.30), rgba(8,16,28,0.22));
            padding: 8px 10px;
            margin: 4px 0 12px 0;
            color: rgba(201,230,249,0.90);
            font-size: 0.84rem;
          }
          .maint-help b{
            color: rgba(226,245,255,0.97);
          }
          .maint-section-title{
            margin: 10px 0 8px 0;
            padding-left: 8px;
            border-left: 3px solid rgba(120,200,255,0.62);
            font-size: 1.22rem;
            font-weight: 850;
            color: rgba(230,246,255,0.98);
            text-shadow: 0 0 10px rgba(84,174,255,0.18);
          }
          .st-key-maint_focus_status div[data-baseweb="tag"],
          .st-key-maint_focus_status span[data-baseweb="tag"]{
            background: linear-gradient(180deg, rgba(70,160,238,0.94), rgba(32,96,168,0.92)) !important;
            background-color: rgba(44,124,206,0.94) !important;
            border: 1px solid rgba(170,232,255,0.82) !important;
            color: rgba(244,252,255,0.99) !important;
            box-shadow: 0 0 0 1px rgba(108,198,255,0.26), 0 4px 10px rgba(10,46,84,0.32) !important;
          }
          .st-key-maint_focus_status div[data-baseweb="tag"] > *,
          .st-key-maint_focus_status span[data-baseweb="tag"] > *{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            color: rgba(244,252,255,0.99) !important;
          }
          .st-key-maint_focus_status div[data-baseweb="tag"] svg,
          .st-key-maint_focus_status span[data-baseweb="tag"] svg{
            fill: rgba(238,250,255,0.98) !important;
          }
          div[data-testid="stButton"] > button{
            border-radius: 12px !important;
            border: 1px solid rgba(138,214,255,0.58) !important;
            background: linear-gradient(180deg, rgba(28,74,120,0.72), rgba(12,36,68,0.66)) !important;
            color: rgba(236,248,255,0.98) !important;
            box-shadow: 0 8px 18px rgba(8,30,58,0.32), 0 0 12px rgba(74,170,255,0.18) !important;
          }
          div[data-testid="stButton"] > button:hover{
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
          }
          div[data-testid="stExpander"] details{
            border: 1px solid rgba(132,214,255,0.22) !important;
            border-radius: 12px !important;
            background: linear-gradient(165deg, rgba(12,24,42,0.56), rgba(10,18,30,0.40)) !important;
          }
          div[data-testid="stDataFrame"]{
            border: 1px solid rgba(132,214,255,0.22) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="maint-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="maint-title">🧰 Maintenance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="maint-sub">Maintenance planning, fault handling, usage analytics, and manuals in one place.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="maint-line"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="maint-help"><b>Tip:</b> Update status inputs first, then use Dashboard/Future Schedule and mark tasks done.</div>',
        unsafe_allow_html=True,
    )
    
    # =========================================================
    # Small utils
    # =========================================================
    def safe_str(x) -> str:
        try:
            if x is None:
                return ""
            if isinstance(x, float) and np.isnan(x):
                return ""
            return str(x)
        except Exception:
            return ""
    
    # =========================================================
    # Paths
    # =========================================================
    BASE_DIR = P.root_dir
    MAINT_FOLDER = P.maintenance_dir
    DRAW_FOLDER = P.dataset_dir   # dataset CSVs (summary)
    LOGS_FOLDER = P.logs_dir      # ✅ LOG CSVs (MFC actual)
    STATE_PATH = os.path.join(MAINT_FOLDER, "_app_state.json")
    os.makedirs(MAINT_FOLDER, exist_ok=True)
    
    # ✅ Append-only CSV logs (for SQL Lab line-search)
    MAINT_ACTIONS_CSV = os.path.join(MAINT_FOLDER, "maintenance_actions_log.csv")
    FAULTS_CSV = os.path.join(MAINT_FOLDER, "faults_log.csv")
    FAULTS_ACTIONS_CSV = os.path.join(MAINT_FOLDER, "faults_actions_log.csv")
    
    MAINT_ACTIONS_COLS = [
        "maintenance_id",
        "maintenance_ts",
        "maintenance_component",
        "maintenance_task",
        "maintenance_task_id",
        "maintenance_mode",
        "maintenance_hours_source",
        "maintenance_done_date",
        "maintenance_done_hours",
        "maintenance_done_draw",
        "maintenance_source_file",
        "maintenance_actor",
        "maintenance_note",
    ]
    
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
    
    # ✅ Fault actions (close/reopen/notes) — append-only
    FAULTS_ACTIONS_COLS = [
        "fault_action_id",
        "fault_id",
        "action_ts",
        "action_type",     # close / reopen / note
        "actor",
        "note",
        "fix_summary",
    ]
    
    # =========================================================
    # DuckDB connection (shared with SQL Lab)
    # =========================================================
    con = st.session_state.get("sql_duck_con")
    if con is None:
        try:
            con = duckdb.connect(P.duckdb_path)
            st.session_state["sql_duck_con"] = con
        except Exception as e:
            msg = str(e)
            if "Could not set lock on file" in msg:
                st.warning(
                    "DuckDB is locked by another Tower process on this computer. "
                    "Close the other running app instance and retry this tab."
                )
                st.caption(f"DB path: `{P.duckdb_path}`")
                return
            raise
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass
    
    # =========================================================
    # Create DB tables
    # =========================================================
    con.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_tasks (
        task_key            VARCHAR,
        task_id             VARCHAR,
        component           VARCHAR,
        task                VARCHAR,
        tracking_mode       VARCHAR,
        hours_source        VARCHAR,
        interval_value      VARCHAR,
        interval_unit       VARCHAR,
        due_threshold_days  VARCHAR,
        manual_name         VARCHAR,
        page                VARCHAR,
        document            VARCHAR,
        procedure_summary   VARCHAR,
        notes               VARCHAR,
        owner               VARCHAR,
        source_file         VARCHAR,
        loaded_at           TIMESTAMP
    );
    """)
    
    con.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_actions (
        action_id       BIGINT,
        action_ts       TIMESTAMP,
        component       VARCHAR,
        task            VARCHAR,
        task_id         VARCHAR,
        tracking_mode   VARCHAR,
        hours_source    VARCHAR,
        done_date       DATE,
        done_hours      DOUBLE,
        done_draw       INTEGER,
        source_file     VARCHAR,
        actor           VARCHAR,
        note            VARCHAR
    );
    """)
    
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
    
    # =========================================================
    # Persistent state helpers
    # =========================================================
    def load_state(path: str) -> dict:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_state(path: str, state: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
        def _sanitize(o):
            if isinstance(o, (dt.date, dt.datetime)):
                return o.isoformat()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o
    
        clean = {k: _sanitize(v) for k, v in state.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
    
    state = load_state(STATE_PATH)
    
    # =========================================================
    # CSV helpers (append-only)
    # =========================================================
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
        for dcol in [c for c in cols if c.endswith("_date")]:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
    
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
    
        # DuckDB
        try:
            con.execute("""
                INSERT INTO faults_actions
                (fault_action_id, fault_id, action_ts, action_type, actor, note, fix_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [aid, int(fault_id), now_dt, str(action_type), str(actor), str(note), str(fix_summary)])
        except Exception as e:
            st.warning(f"Fault action DB insert failed (still saving CSV): {e}")
    
        # CSV
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
    
    # =========================================================
    # Draw count helper
    # =========================================================
    def get_draw_csv_count(folder: str) -> int:
        if not os.path.isdir(folder):
            return 0
        return sum(
            1 for f in os.listdir(folder)
            if f.lower().endswith(".csv") and not f.startswith("~$")
        )
    
    current_draw_count = get_draw_csv_count(DRAW_FOLDER)
    
    # =========================================================
    # Maintenance file loading
    # =========================================================
    files = [f for f in os.listdir(MAINT_FOLDER) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
    if not files:
        st.warning("No maintenance files found in /maintenance folder.")
        st.stop()
    
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
    inverse_map = {v: k for k, v in normalize_map.items()}
    
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
    
    def write_file(path: str, df: pd.DataFrame):
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
    
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
    
    def templateize_df(df_internal: pd.DataFrame, original_cols: list) -> pd.DataFrame:
        df = df_internal.copy()
        rename_back = {}
        for internal_col, template_key_lower in inverse_map.items():
            match = None
            for oc in original_cols:
                if str(oc).strip().lower() == template_key_lower:
                    match = oc
                    break
            if match is not None and internal_col in df.columns:
                rename_back[internal_col] = match
        df.rename(columns=rename_back, inplace=True)
        return df
    
    frames = []
    load_errors = []
    for fname in sorted(files):
        fpath = os.path.join(MAINT_FOLDER, fname)
        try:
            raw = read_file(fpath)
            if raw is None or raw.empty:
                continue
            df = normalize_df(raw)
            df["Source_File"] = fname
            frames.append(df)
        except ImportError as e:
            st.error("Excel engine missing. Install openpyxl in your .venv:")
            st.code("pip install openpyxl", language="bash")
            st.exception(e)
            st.stop()
        except Exception as e:
            load_errors.append((fname, str(e)))
    
    if not frames:
        st.error("No valid maintenance data could be loaded.")
        if load_errors:
            st.dataframe(pd.DataFrame(load_errors, columns=["File", "Error"]), use_container_width=True)
        st.stop()
    
    dfm = pd.concat(frames, ignore_index=True)
    
    # =========================================================
    # Persisted inputs (hours + settings)
    # =========================================================
    def _persist_inputs():
        state["current_date"] = dt.date.today().isoformat()
        state["furnace_hours"] = float(st.session_state.get("maint_furnace_hours", 0.0))
        state["uv1_hours"] = float(st.session_state.get("maint_uv1_hours", 0.0))
        state["uv2_hours"] = float(st.session_state.get("maint_uv2_hours", 0.0))
        state["warn_days"] = int(st.session_state.get("maint_warn_days", 14))
        state["warn_hours"] = float(st.session_state.get("maint_warn_hours", 50.0))
        save_state(STATE_PATH, state)
        st.session_state["furnace_hours"] = state["furnace_hours"]
        st.session_state["uv1_hours"] = state["uv1_hours"]
        st.session_state["uv2_hours"] = state["uv2_hours"]
    
    default_furnace = float(state.get("furnace_hours", 0.0) or 0.0)
    default_uv1 = float(state.get("uv1_hours", 0.0) or 0.0)
    default_uv2 = float(state.get("uv2_hours", 0.0) or 0.0)
    default_warn_days = int(state.get("warn_days", 14) or 14)
    default_warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

    # Compact weekly status view + folded editor.
    # Keep editor defaults synced with last saved weekly snapshot (without overriding active unsaved typing every rerun).
    saved_stamp = safe_str(state.get("status_weekly_updated_at", ""))
    if st.session_state.get("maint_weekly_loaded_stamp", None) != saved_stamp:
        st.session_state["maint_furnace_hours"] = default_furnace
        st.session_state["maint_uv1_hours"] = default_uv1
        st.session_state["maint_uv2_hours"] = default_uv2
        st.session_state["maint_warn_days"] = default_warn_days
        st.session_state["maint_warn_hours"] = default_warn_hours
        st.session_state["maint_weekly_loaded_stamp"] = saved_stamp
    else:
        st.session_state.setdefault("maint_furnace_hours", default_furnace)
        st.session_state.setdefault("maint_uv1_hours", default_uv1)
        st.session_state.setdefault("maint_uv2_hours", default_uv2)
        st.session_state.setdefault("maint_warn_days", default_warn_days)
        st.session_state.setdefault("maint_warn_hours", default_warn_hours)

    current_date = dt.date.today()
    weekly_updated_raw = safe_str(state.get("status_weekly_updated_at", ""))
    weekly_updated_dt = pd.to_datetime(weekly_updated_raw, errors="coerce")
    is_weekly_fresh = pd.notna(weekly_updated_dt) and ((pd.Timestamp(current_date) - weekly_updated_dt.normalize()).days <= 7)

    st.markdown('<div class="maint-section-title">📌 Current Tower Status (Weekly)</div>', unsafe_allow_html=True)
    draw_logs_count = get_draw_csv_count(LOGS_FOLDER)
    c0, c1, c2, c3, c4, c5, c6 = st.columns(7)
    c0.metric("Today", str(current_date))
    c1.metric("Furnace h", f"{float(st.session_state.get('maint_furnace_hours', 0.0)):.1f}")
    c2.metric("UV1 h", f"{float(st.session_state.get('maint_uv1_hours', 0.0)):.1f}")
    c3.metric("UV2 h", f"{float(st.session_state.get('maint_uv2_hours', 0.0)):.1f}")
    c4.metric("Draws (logs)", int(draw_logs_count))
    c5.metric("Warn days", int(st.session_state.get("maint_warn_days", 14)))
    c6.metric("Warn hours", f"{float(st.session_state.get('maint_warn_hours', 50.0)):.1f}")

    if is_weekly_fresh:
        st.success(f"Weekly status updated: {weekly_updated_dt.strftime('%Y-%m-%d %H:%M')}")
    else:
        show_ts = weekly_updated_dt.strftime("%Y-%m-%d %H:%M") if pd.notna(weekly_updated_dt) else "never"
        st.warning(f"Weekly status update is due. Last update: {show_ts}")

    with st.expander("🛠️ Edit Weekly Status Inputs", expanded=False):
        c2, c3, c4, c5 = st.columns([1, 1, 1, 1])
        with c2:
            st.number_input(
                "Furnace hours", min_value=0.0, step=1.0,
                key="maint_furnace_hours"
            )
        with c3:
            st.number_input(
                "UV System 1 hours", min_value=0.0, step=1.0,
                key="maint_uv1_hours"
            )
        with c4:
            st.number_input(
                "UV System 2 hours", min_value=0.0, step=1.0,
                key="maint_uv2_hours"
            )
        with c5:
            st.number_input(
                "Warn if due within (days)", min_value=0, step=1,
                key="maint_warn_days"
            )

        st.number_input(
            "Warn if due within (hours)", min_value=0.0, step=1.0,
            key="maint_warn_hours"
        )
        if st.button("💾 Save Weekly Status", key="maint_save_weekly_status_btn", type="primary", use_container_width=True):
            state["status_weekly_updated_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _persist_inputs()
            st.session_state["maint_weekly_loaded_stamp"] = safe_str(state.get("status_weekly_updated_at", ""))
            st.success("Weekly status saved.")
            st.rerun()

    furnace_hours = float(st.session_state.get("maint_furnace_hours", default_furnace))
    uv1_hours = float(st.session_state.get("maint_uv1_hours", default_uv1))
    uv2_hours = float(st.session_state.get("maint_uv2_hours", default_uv2))
    warn_days = int(st.session_state.get("maint_warn_days", default_warn_days))
    warn_hours = float(st.session_state.get("maint_warn_hours", default_warn_hours))
    st.caption("Hours-based tasks use **Hours Source**: FURNACE / UV1 / UV2. If empty -> defaults to FURNACE.")
    
    # =========================================================
    # Actor
    # =========================================================
    st.session_state.setdefault("maint_actor", "operator")
    st.text_input("Actor / operator name (for history)", key="maint_actor")
    actor = st.session_state.get("maint_actor", "operator")
    
    # =========================================================
    # Helpers
    # =========================================================
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
    
    def pick_current_hours(hours_source: str) -> float:
        hs = norm_source(hours_source)
        if hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
            return float(uv2_hours)
        if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
            return float(uv1_hours)
        return float(furnace_hours)
    
    def mode_norm(x: str) -> str:
        s = "" if x is None or pd.isna(x) else str(x).strip().lower()
        if s in ("draw", "draws", "draws_count", "draw_count"):
            return "draws"
        return s
    
    # =========================================================
    # Compute Next Due + Status
    # =========================================================
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
    
        # calendar
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
    
        # hours
        if nh is not None and not pd.isna(nh):
            nh = float(nh)
            cur_h = float(row.get("Current_Hours_For_Task", 0.0))
            if nh < cur_h:
                overdue = True
            elif (nh - cur_h) <= float(warn_hours):
                due_soon = True
    
        # draws
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
    
    st.session_state["maint_overdue"] = int((dfm["Status"] == "OVERDUE").sum())
    st.session_state["maint_due_soon"] = int((dfm["Status"] == "DUE SOON").sum())
    
    # =========================================================
    # Dashboard metrics + Open Critical Faults
    # =========================================================
    def get_open_faults_counts():
        faults_csv = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
        actions_csv = _read_csv_safe(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS)
        smap = _latest_fault_state(actions_csv)
    
        if faults_csv.empty:
            return 0, 0
    
        faults_csv["fault_id"] = pd.to_numeric(faults_csv["fault_id"], errors="coerce")
        faults_csv = faults_csv.dropna(subset=["fault_id"]).copy()
        faults_csv["fault_id"] = faults_csv["fault_id"].astype(int)
    
        faults_csv["_is_closed"] = faults_csv["fault_id"].apply(lambda fid: bool(smap.get(int(fid), {}).get("is_closed", False)))
        open_df = faults_csv[~faults_csv["_is_closed"]].copy()
    
        crit_open = int((open_df["fault_severity"].astype(str).str.lower() == "critical").sum()) if not open_df.empty else 0
        open_total = int(len(open_df))
        return open_total, crit_open
    
    def render_maintenance_dashboard_metrics(dfm):
        st.markdown(
            """
            <style>
              .maint-metrics-grid{
                display:grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 10px;
                margin: 8px 0 10px 0;
              }
              .maint-metric-card{
                border-radius: 12px;
                padding: 10px 12px;
                border: 1px solid rgba(128,206,255,0.24);
                background: linear-gradient(180deg, rgba(14,32,56,0.36), rgba(8,16,28,0.26));
              }
              .maint-metric-k{
                font-size: 0.78rem;
                color: rgba(188,224,248,0.92);
                margin-bottom: 4px;
                font-weight: 700;
                letter-spacing: 0.2px;
              }
              .maint-metric-v{
                font-size: 2.0rem;
                line-height: 1.0;
                font-weight: 900;
              }
              .maint-v-red{ color:#ff5f5f; text-shadow:0 0 12px rgba(255,72,72,0.26); }
              .maint-v-orange{ color:#ffb84d; text-shadow:0 0 12px rgba(255,168,48,0.24); }
              .maint-v-green{ color:#6dff95; text-shadow:0 0 12px rgba(88,246,126,0.22); }
              .maint-v-blue{ color:#7ec6ff; text-shadow:0 0 12px rgba(86,180,255,0.22); }
              @media (max-width: 1100px){
                .maint-metrics-grid{ grid-template-columns: repeat(3, minmax(0, 1fr)); }
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        overdue = int((dfm["Status"] == "OVERDUE").sum())
        due_soon = int((dfm["Status"] == "DUE SOON").sum())
        routine = int((dfm["Status"] == "ROUTINE").sum())
        ok = int((dfm["Status"] == "OK").sum())
        open_faults, crit_faults = get_open_faults_counts()
        overdue_cls = "maint-v-red" if overdue > 0 else "maint-v-green"
        due_cls = "maint-v-orange" if due_soon > 0 else "maint-v-green"
        routine_cls = "maint-v-blue"
        ok_cls = "maint-v-green"
        open_cls = "maint-v-orange" if open_faults > 0 else "maint-v-green"
        crit_cls = "maint-v-red" if crit_faults > 0 else "maint-v-green"

        st.markdown(
            f"""
            <div class="maint-metrics-grid">
              <div class="maint-metric-card">
                <div class="maint-metric-k">OVERDUE</div>
                <div class="maint-metric-v {overdue_cls}">{overdue}</div>
              </div>
              <div class="maint-metric-card">
                <div class="maint-metric-k">DUE SOON</div>
                <div class="maint-metric-v {due_cls}">{due_soon}</div>
              </div>
              <div class="maint-metric-card">
                <div class="maint-metric-k">ROUTINE</div>
                <div class="maint-metric-v {routine_cls}">{routine}</div>
              </div>
              <div class="maint-metric-card">
                <div class="maint-metric-k">OK</div>
                <div class="maint-metric-v {ok_cls}">{ok}</div>
              </div>
              <div class="maint-metric-card">
                <div class="maint-metric-k">🚨 Open Faults</div>
                <div class="maint-metric-v {open_cls}">{open_faults}</div>
              </div>
              <div class="maint-metric-card">
                <div class="maint-metric-k">🟥 Critical Open</div>
                <div class="maint-metric-v {crit_cls}">{crit_faults}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.setdefault("maint_dash_focus", "")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        if b1.button("View OVERDUE", key="maint_dash_btn_overdue", use_container_width=True):
            st.session_state["maint_dash_focus"] = "OVERDUE"
        if b2.button("View DUE SOON", key="maint_dash_btn_due_soon", use_container_width=True):
            st.session_state["maint_dash_focus"] = "DUE SOON"
        if b3.button("View ROUTINE", key="maint_dash_btn_routine", use_container_width=True):
            st.session_state["maint_dash_focus"] = "ROUTINE"
        if b4.button("View OK", key="maint_dash_btn_ok", use_container_width=True):
            st.session_state["maint_dash_focus"] = "OK"
        if b5.button("View Open Faults", key="maint_dash_btn_open_faults", use_container_width=True):
            st.session_state["maint_dash_focus"] = "OPEN_FAULTS"
        if b6.button("View Critical", key="maint_dash_btn_critical_faults", use_container_width=True):
            st.session_state["maint_dash_focus"] = "CRITICAL_FAULTS"

        focus = st.session_state.get("maint_dash_focus", "")
        if focus:
            c_left, c_right = st.columns([5, 1])
            c_left.markdown(f"**Focused List: {focus}**")
            if c_right.button("Clear", key="maint_dash_focus_clear", use_container_width=True):
                st.session_state["maint_dash_focus"] = ""
                st.rerun()

            if focus in {"OVERDUE", "DUE SOON", "ROUTINE", "OK"}:
                show_cols = [
                    c for c in [
                        "Status", "Component", "Task", "Task_ID", "Tracking_Mode",
                        "Hours_Source", "Next_Due_Date", "Next_Due_Hours", "Next_Due_Draw", "Source_File"
                    ] if c in dfm.columns
                ]
                view_df = dfm[dfm["Status"] == focus].copy()
                st.dataframe(view_df[show_cols], use_container_width=True, hide_index=True, height=280)
            else:
                faults_csv = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
                actions_csv = _read_csv_safe(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS)
                smap = _latest_fault_state(actions_csv)
                if faults_csv.empty:
                    st.info("No faults data.")
                    return
                faults_csv["fault_id"] = pd.to_numeric(faults_csv["fault_id"], errors="coerce")
                faults_csv = faults_csv.dropna(subset=["fault_id"]).copy()
                faults_csv["fault_id"] = faults_csv["fault_id"].astype(int)
                faults_csv["_is_closed"] = faults_csv["fault_id"].apply(lambda fid: bool(smap.get(int(fid), {}).get("is_closed", False)))
                open_df = faults_csv[~faults_csv["_is_closed"]].copy()

                if focus == "OPEN_FAULTS":
                    out = open_df[[
                        "fault_ts", "fault_id", "fault_component", "fault_severity",
                        "fault_title", "fault_description", "fault_related_draw"
                    ]] if not open_df.empty else pd.DataFrame()
                    st.dataframe(out, use_container_width=True, hide_index=True, height=280)
                elif focus == "CRITICAL_FAULTS":
                    crit = open_df[open_df["fault_severity"].astype(str).str.lower().eq("critical")].copy()
                    out = crit[[
                        "fault_ts", "fault_id", "fault_component", "fault_severity",
                        "fault_title", "fault_description", "fault_related_draw"
                    ]] if not crit.empty else pd.DataFrame()
                    st.dataframe(out, use_container_width=True, hide_index=True, height=280)
    
    # =========================================================
    # Horizon selector + roadmaps
    # =========================================================
    def render_maintenance_horizon_selector(current_draw_count: int):
        st.markdown("#### Horizon Setup")
    
        st.markdown(
            """
            <style>
            div.stButton > button {
                width: 100%;
                height: 44px;
                border-radius: 12px;
                font-weight: 600;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
        st.session_state.setdefault("maint_horizon_hours", 10)
        st.session_state.setdefault("maint_horizon_days", 7)
        st.session_state.setdefault("maint_horizon_draws", 5)
    
        def button_group(title, options, value, key):
            st.caption(title)
            cols = st.columns(len(options))
            for col, (label, v) in zip(cols, options):
                if col.button(label, key=f"{key}_{v}", type="primary" if v == value else "secondary"):
                    return v
            return value
    
        c1, c2, c3 = st.columns(3)
    
        with c1:
            st.session_state["maint_horizon_hours"] = button_group(
                "Hours horizon",
                [("10", 10), ("50", 50), ("100", 100)],
                st.session_state["maint_horizon_hours"],
                "mh"
            )
    
        with c2:
            st.session_state["maint_horizon_days"] = button_group(
                "Calendar horizon",
                [("Week", 7), ("Month", 30), ("3 Months", 90)],
                st.session_state["maint_horizon_days"],
                "md"
            )
    
        with c3:
            st.session_state["maint_horizon_draws"] = button_group(
                "Draw horizon",
                [("5", 5), ("10", 10), ("50", 50)],
                st.session_state["maint_horizon_draws"],
                "mD"
            )
    
        st.caption(
            f"📦 Now: **{current_draw_count}** → "
            f"Horizon: **{st.session_state['maint_horizon_draws']}** → "
            f"Up to draw **#{current_draw_count + st.session_state['maint_horizon_draws']}**"
        )
    
        return (
            st.session_state["maint_horizon_hours"],
            st.session_state["maint_horizon_days"],
            st.session_state["maint_horizon_draws"],
        )

    def render_future_schedule_focus_selector():
        st.session_state.setdefault("maint_future_focus", "all")
        st.caption("Focus by type")
        picked = st.radio(
            "Timeline type",
            options=["all", "hours", "draws", "calendar"],
            format_func=lambda v: {
                "all": "🌐 All",
                "hours": "🔥 Hours",
                "draws": "🧵 Draws",
                "calendar": "🗓️ Calendar",
            }.get(v, v),
            horizontal=True,
            key="maint_future_focus",
            label_visibility="collapsed",
        )
        return picked
    
    def render_maintenance_roadmaps(
        dfm: pd.DataFrame,
        current_date,
        current_draw_count: int,
        furnace_hours: float,
        uv1_hours: float,
        uv2_hours: float,
        horizon_hours: int,
        horizon_days: int,
        horizon_draws: int,
        focus: str = "all",
    ):
        def status_color(s):
            s = str(s).upper()
            if s == "OVERDUE":
                return "#ff4d4d"
            if s == "DUE SOON":
                return "#ffcc00"
            return "#66ff99"
    
        def roadmap(x0, x1, title, xlabel, df, xcol, hover):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[0, 0], mode="lines",
                line=dict(width=6, color="rgba(180,180,180,0.2)"),
                hoverinfo="skip"
            ))
            fig.add_vline(x=x0, line_dash="dash")
    
            if df is not None and not df.empty:
                fig.add_trace(go.Scatter(
                    x=df[xcol],
                    y=[0] * len(df),
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=[status_color(s) for s in df["Status"]],
                        line=dict(width=1, color="rgba(255,255,255,0.5)")
                    ),
                    text=df[hover],
                    hovertemplate="%{text}<extra></extra>",
                ))
            else:
                mid = x0 + (x1 - x0) / 2
                fig.add_annotation(x=mid, y=0, text="No tasks in horizon", showarrow=False)
    
            fig.update_layout(
                title=title,
                height=220,
                yaxis=dict(visible=False),
                xaxis=dict(title=xlabel),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return fig
    
        def norm_group(src):
            s = str(src).lower()
            if "uv1" in s:
                return "UV1"
            if "uv2" in s:
                return "UV2"
            return "FURNACE"
    
        hours_df = dfm[dfm["Tracking_Mode_norm"] == "hours"].copy()
        hours_df["Due"] = pd.to_numeric(hours_df["Next_Due_Hours"], errors="coerce")
        hours_df = hours_df.dropna(subset=["Due"])
        hours_df["Group"] = hours_df["Hours_Source"].apply(norm_group)
        hours_df["Hover"] = hours_df["Component"] + " — " + hours_df["Task"] + "<br>Status: " + hours_df["Status"]
    
        cal_df = dfm[dfm["Tracking_Mode_norm"] == "calendar"].copy()
        cal_df["Due"] = pd.to_datetime(cal_df["Next_Due_Date"], errors="coerce")
        cal_df = cal_df.dropna(subset=["Due"])
        cal_df["Hover"] = cal_df["Component"] + " — " + cal_df["Task"] + "<br>Status: " + cal_df["Status"]
    
        draw_df = dfm[dfm["Tracking_Mode_norm"] == "draws"].copy()
        draw_df["Due"] = pd.to_numeric(draw_df["Next_Due_Draw"], errors="coerce")
        draw_df = draw_df.dropna(subset=["Due"])
        draw_df["Hover"] = draw_df["Component"] + " — " + draw_df["Task"] + "<br>Status: " + draw_df["Status"]
    
        if focus in ("all", "hours"):
            st.markdown("### 🔥 Furnace / 💡 UV timelines")
            c1, c2, c3 = st.columns(3)

            with c1:
                x0, x1 = furnace_hours, furnace_hours + horizon_hours
                st.plotly_chart(
                    roadmap(x0, x1, "FURNACE", "Hours",
                            hours_df[(hours_df["Group"] == "FURNACE") & hours_df["Due"].between(x0, x1)],
                            "Due", "Hover"),
                    use_container_width=True
                )

            with c2:
                x0, x1 = uv1_hours, uv1_hours + horizon_hours
                st.plotly_chart(
                    roadmap(x0, x1, "UV1", "Hours",
                            hours_df[(hours_df["Group"] == "UV1") & hours_df["Due"].between(x0, x1)],
                            "Due", "Hover"),
                    use_container_width=True
                )

            with c3:
                x0, x1 = uv2_hours, uv2_hours + horizon_hours
                st.plotly_chart(
                    roadmap(x0, x1, "UV2", "Hours",
                            hours_df[(hours_df["Group"] == "UV2") & hours_df["Due"].between(x0, x1)],
                            "Due", "Hover"),
                    use_container_width=True
                )

        if focus in ("all", "draws"):
            st.markdown("### 🧵 Draw timeline")
            d0, d1 = current_draw_count, current_draw_count + horizon_draws
            st.plotly_chart(
                roadmap(d0, d1, "Draw-based tasks", "Draw #",
                        draw_df[draw_df["Due"].between(d0, d1)],
                        "Due", "Hover"),
                use_container_width=True
            )

        if focus in ("all", "calendar"):
            st.markdown("### 🗓️ Calendar timeline")
            t0 = pd.Timestamp(current_date)
            t1 = t0 + pd.Timedelta(days=horizon_days)
            st.plotly_chart(
                roadmap(t0, t1, "Calendar tasks", "Date",
                        cal_df[(cal_df["Due"] >= t0) & (cal_df["Due"] <= t1)],
                        "Due", "Hover"),
                use_container_width=True
            )
    
    # =========================================================
    # Done editor + apply done (updates + logs DB + CSV)
    # =========================================================
    def render_maintenance_done_editor(dfm):
        focus_default = ["OVERDUE", "DUE SOON", "ROUTINE"]
        focus_status = st.multiselect(
            "Work on these statuses",
            ["OVERDUE", "DUE SOON", "ROUTINE", "OK"],
            default=focus_default,
            key="maint_focus_status"
        )
    
        work = (
            dfm[dfm["Status"].isin(focus_status)]
            .copy()
            .sort_values(["Status", "Component", "Task"])
        )
        work["Done_Now"] = False
    
        cols = [
            "Done_Now",
            "Status", "Component", "Task", "Task_ID",
            "Tracking_Mode", "Hours_Source", "Current_Hours_For_Task",
            "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
            "Next_Due_Date", "Next_Due_Hours", "Next_Due_Draw",
            "Manual_Name", "Page", "Document",
            "Owner", "Source_File"
        ]
        cols = [c for c in cols if c in work.columns]
    
        edited = st.data_editor(
            work[cols],
            use_container_width=True,
            height=420,
            column_config={
                "Done_Now": st.column_config.CheckboxColumn("Done now", help="Tick tasks you completed")
            },
            disabled=[c for c in cols if c != "Done_Now"],
            key="maint_editor"
        )
        return edited

    def render_quick_reschedule_panel(
        *,
        dfm,
        MAINT_FOLDER,
        current_draw_count,
        furnace_hours,
        uv1_hours,
        uv2_hours,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
    ):
        st.caption("Push a maintenance task quickly: +draws or +hours, without editing full tables.")

        def _task_label(r: dict) -> str:
            return (
                f"[{safe_str(r.get('Status',''))}] "
                f"{safe_str(r.get('Component',''))} — {safe_str(r.get('Task',''))} "
                f"(ID: {safe_str(r.get('Task_ID',''))})"
            )

        def _update_task_in_source(task_row: dict, *, by_mode: str, shift_value: float):
            src = safe_str(task_row.get("Source_File", ""))
            if not src:
                st.error("Task has no Source_File; cannot update.")
                return
            path = os.path.join(MAINT_FOLDER, src)
            if not os.path.exists(path):
                st.error(f"Source file missing: {path}")
                return

            raw = read_file(path)
            df_src = normalize_df(raw)
            mask = (
                df_src["Component"].astype(str).eq(str(task_row.get("Component", "")))
                & df_src["Task"].astype(str).eq(str(task_row.get("Task", "")))
            )
            if not mask.any():
                st.warning("Task not found in source file.")
                return

            if by_mode == "draws":
                base = parse_int(task_row.get("Last_Done_Draw", None))
                if base is None:
                    base = int(current_draw_count)
                df_src.loc[mask, "Last_Done_Draw"] = int(base + int(shift_value))
            elif by_mode == "hours":
                hs = norm_source(task_row.get("Hours_Source", ""))
                current_ref = float(furnace_hours)
                if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
                    current_ref = float(uv1_hours)
                elif hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
                    current_ref = float(uv2_hours)

                base = parse_float(task_row.get("Last_Done_Hours", None))
                if base is None:
                    base = current_ref
                df_src.loc[mask, "Last_Done_Hours"] = float(base + float(shift_value))
            else:
                return

            out = templateize_df(df_src, list(raw.columns))
            write_file(path, out)
            st.success(f"Rescheduled: {safe_str(task_row.get('Component',''))} — {safe_str(task_row.get('Task',''))}")
            st.rerun()

        # Draw-based quick push
        draw_tasks = (
            dfm[dfm["Tracking_Mode_norm"] == "draws"]
            .copy()
            .sort_values(["Status", "Component", "Task"])
        )
        hour_tasks = (
            dfm[dfm["Tracking_Mode_norm"] == "hours"]
            .copy()
            .sort_values(["Status", "Component", "Task"])
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🧵 Draw-based tasks**")
            if draw_tasks.empty:
                st.info("No draw-based tasks found.")
            else:
                d_opts = draw_tasks.to_dict("records")
                d_pick = st.selectbox(
                    "Select draw task",
                    options=d_opts,
                    format_func=_task_label,
                    key="maint_resched_draw_pick",
                )
                d_shift = st.number_input(
                    "Push by draws",
                    min_value=1,
                    value=5,
                    step=1,
                    key="maint_resched_draw_shift",
                )
                if st.button("⏩ Schedule +Draws", use_container_width=True, type="primary", key="maint_resched_draw_apply"):
                    _update_task_in_source(d_pick, by_mode="draws", shift_value=float(d_shift))

        with c2:
            st.markdown("**🔥 Hour-based tasks**")
            if hour_tasks.empty:
                st.info("No hour-based tasks found.")
            else:
                h_opts = hour_tasks.to_dict("records")
                h_pick = st.selectbox(
                    "Select hours task",
                    options=h_opts,
                    format_func=_task_label,
                    key="maint_resched_hours_pick",
                )
                h_shift = st.number_input(
                    "Push by hours",
                    min_value=1.0,
                    value=5.0,
                    step=1.0,
                    key="maint_resched_hours_shift",
                )
                if st.button("⏩ Schedule +Hours", use_container_width=True, type="primary", key="maint_resched_hours_apply"):
                    _update_task_in_source(h_pick, by_mode="hours", shift_value=float(h_shift))

    def render_maintenance_scheduler_bridge(
        *,
        dfm,
        current_date,
        current_draw_count,
        furnace_hours,
        uv1_hours,
        uv2_hours,
    ):
        st.caption("Auto-plan maintenance tasks into free schedule slots, based on urgency and current tower state.")
        st.caption("Default rule: prioritize Thursday. Friday/Saturday are non-working days.")

        # ---- Controls
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            horizon_days = st.number_input("Plan horizon (days)", min_value=3, max_value=30, value=7, step=1, key="maint_sched_horizon_days")
        with c2:
            slot_min = st.selectbox("Slot length (min)", options=[30, 45, 60, 90, 120], index=2, key="maint_sched_slot_minutes")
        with c3:
            day_start_h = st.number_input("Day start (hour)", min_value=0, max_value=23, value=8, step=1, key="maint_sched_day_start")
        with c4:
            day_end_h = st.number_input("Day end (hour)", min_value=1, max_value=23, value=18, step=1, key="maint_sched_day_end")

        weekday_options = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        weekday_to_idx = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Sunday": 6,
        }
        preferred_days = st.multiselect(
            "Preferred maintenance day(s)",
            options=weekday_options,
            default=["Thursday"],
            key="maint_sched_preferred_days",
            help="Planner will prefer these days first (still by urgency).",
        )
        if not preferred_days:
            preferred_days = ["Thursday"]
        preferred_idx = [weekday_to_idx[d] for d in preferred_days if d in weekday_to_idx]

        if int(day_end_h) <= int(day_start_h):
            st.warning("Day end must be after day start.")
            return

        # ---- Candidate tasks (only actionable statuses)
        cand = dfm[dfm["Status"].isin(["OVERDUE", "DUE SOON"])].copy()
        if cand.empty:
            st.info("No OVERDUE / DUE SOON tasks to schedule.")
            return

        def _urgency_score(row):
            score = 0.0
            status = str(row.get("Status", "")).upper()
            mode = str(row.get("Tracking_Mode_norm", "")).lower()
            if status == "OVERDUE":
                score += 100.0
            elif status == "DUE SOON":
                score += 50.0

            if mode == "calendar":
                nd = row.get("Next_Due_Date", None)
                if nd is not None and not pd.isna(nd):
                    try:
                        dd = (pd.Timestamp(current_date) - pd.Timestamp(nd)).days
                        score += max(0.0, float(dd))
                    except Exception:
                        pass
            elif mode == "hours":
                nh = row.get("Next_Due_Hours", None)
                cur = float(row.get("Current_Hours_For_Task", 0.0) or 0.0)
                try:
                    if nh is not None and not pd.isna(nh):
                        score += max(0.0, cur - float(nh))
                except Exception:
                    pass
            elif mode == "draws":
                nd = row.get("Next_Due_Draw", None)
                try:
                    if nd is not None and not pd.isna(nd):
                        score += max(0.0, float(current_draw_count) - float(nd))
                except Exception:
                    pass
            return score

        cand["_urgency"] = cand.apply(_urgency_score, axis=1)
        cand = cand.sort_values(["_urgency", "Component", "Task"], ascending=[False, True, True]).reset_index(drop=True)

        max_tasks = st.number_input("Max tasks to plan", min_value=1, max_value=100, value=min(12, len(cand)), step=1, key="maint_sched_max_tasks")
        cand = cand.head(int(max_tasks)).copy()

        # ---- Read schedule and build busy intervals
        sched_path = P.schedule_csv
        if os.path.exists(sched_path):
            try:
                sched_df = pd.read_csv(sched_path, keep_default_na=False)
            except Exception:
                sched_df = pd.DataFrame()
        else:
            sched_df = pd.DataFrame()

        for col in ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]:
            if col not in sched_df.columns:
                sched_df[col] = ""

        start_window = pd.Timestamp(current_date)
        end_window = start_window + pd.Timedelta(days=int(horizon_days))
        sched_df["_start"] = pd.to_datetime(sched_df["Start DateTime"], errors="coerce")
        sched_df["_end"] = pd.to_datetime(sched_df["End DateTime"], errors="coerce")

        busy = []
        for _, r in sched_df.iterrows():
            s = r.get("_start")
            e = r.get("_end")
            if pd.isna(s) or pd.isna(e):
                continue
            if e < start_window or s > end_window:
                continue
            busy.append((pd.Timestamp(s), pd.Timestamp(e)))

        def _overlaps(a0, a1, b0, b1):
            return (a0 < b1) and (a1 > b0)

        # ---- Build free slots
        slots = []
        slot_delta = pd.Timedelta(minutes=int(slot_min))
        d = start_window.normalize()
        while d < end_window:
            # Friday/Saturday are non-working days.
            if d.weekday() in (4, 5):
                d = d + pd.Timedelta(days=1)
                continue
            day_s = d + pd.Timedelta(hours=int(day_start_h))
            day_e = d + pd.Timedelta(hours=int(day_end_h))
            t = day_s
            while t + slot_delta <= day_e:
                t2 = t + slot_delta
                conflict = any(_overlaps(t, t2, b0, b1) for b0, b1 in busy)
                if not conflict:
                    # Priority: preferred selected days first, then other working days.
                    # weekday(): Monday=0 ... Thursday=3, Sunday=6.
                    if d.weekday() in preferred_idx:
                        # keep user-selected preferred day order
                        pref_rank = preferred_idx.index(d.weekday())
                        slots.append((0, pref_rank, t, t2))
                    else:
                        slots.append((1, 999, t, t2))
                t = t + slot_delta
            d = d + pd.Timedelta(days=1)

        if not slots:
            st.warning("No free slots found in selected horizon/day window.")
            return

        slots.sort(key=lambda x: (x[0], x[1], x[2]))

        # ---- Auto assign highest urgency tasks to earliest free slots
        plan_rows = []
        for i, (_, task) in enumerate(cand.iterrows()):
            if i >= len(slots):
                break
            _, _, s, e = slots[i]
            comp = safe_str(task.get("Component", "")).strip()
            tname = safe_str(task.get("Task", "")).strip()
            tid = safe_str(task.get("Task_ID", "")).strip()
            status = safe_str(task.get("Status", "")).strip()
            mode = safe_str(task.get("Tracking_Mode", "")).strip()
            hs = safe_str(task.get("Hours_Source", "")).strip()
            desc = f"[AUTO-MAINT] {comp} - {tname} (ID:{tid}) | status={status} | mode={mode} | source={hs or 'FURNACE'}"
            plan_rows.append(
                {
                    "Event Type": "Maintenance",
                    "Start DateTime": s.strftime("%Y-%m-%d %H:%M:%S"),
                    "End DateTime": e.strftime("%Y-%m-%d %H:%M:%S"),
                    "Description": desc,
                    "Recurrence": "",
                    "Component": comp,
                    "Task": tname,
                    "Task_ID": tid,
                    "Status": status,
                    "Urgency": float(task.get("_urgency", 0.0)),
                }
            )

        plan_df = pd.DataFrame(plan_rows)
        st.markdown("**Suggested maintenance schedule events**")
        st.dataframe(plan_df, use_container_width=True, hide_index=True, height=280)
        st.caption(f"Preferred day order: {', '.join(preferred_days)}")

        # Visual schedule plot for suggestions.
        if not plan_df.empty:
            try:
                import plotly.express as px

                vis = plan_df.copy()
                vis["start_ts"] = pd.to_datetime(vis["Start DateTime"], errors="coerce")
                vis["end_ts"] = pd.to_datetime(vis["End DateTime"], errors="coerce")
                vis["Task Label"] = vis["Component"].astype(str) + " - " + vis["Task"].astype(str)
                vis = vis.dropna(subset=["start_ts", "end_ts"]).copy()

                if not vis.empty:
                    fig = px.timeline(
                        vis,
                        x_start="start_ts",
                        x_end="end_ts",
                        y="Task Label",
                        color="Status",
                        hover_data=["Task_ID", "Urgency", "Start DateTime", "End DateTime"],
                        title="Suggested Maintenance Timeline (Thu prioritized, Fri/Sat excluded)",
                    )
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(height=360, margin=dict(l=8, r=8, t=42, b=8))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        st.caption(f"Current state used: draw_count={current_draw_count}, furnace={furnace_hours:.1f}, uv1={uv1_hours:.1f}, uv2={uv2_hours:.1f}")

        if st.button("💾 Add Suggested Events to Tower Schedule", key="maint_sched_apply_btn", type="primary", use_container_width=True):
            if plan_df.empty:
                st.info("No suggestions to save.")
                return

            to_save = plan_df[["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]].copy()
            existing_keys = set(
                zip(
                    sched_df["Event Type"].astype(str),
                    sched_df["Start DateTime"].astype(str),
                    sched_df["End DateTime"].astype(str),
                    sched_df["Description"].astype(str),
                    sched_df["Recurrence"].astype(str),
                )
            )
            add_rows = []
            for _, r in to_save.iterrows():
                k = (
                    str(r["Event Type"]),
                    str(r["Start DateTime"]),
                    str(r["End DateTime"]),
                    str(r["Description"]),
                    str(r["Recurrence"]),
                )
                if k not in existing_keys:
                    add_rows.append(r)
                    existing_keys.add(k)

            if not add_rows:
                st.info("All suggested events already exist in schedule.")
                return

            out = pd.concat([sched_df[["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]], pd.DataFrame(add_rows)], ignore_index=True)
            out.to_csv(sched_path, index=False)
            st.success(f"Added {len(add_rows)} maintenance event(s) to schedule.")
            st.rerun()
    
    def render_maintenance_apply_done(
        edited,
        *,
        dfm,
        current_date,
        current_draw_count,
        actor,
        MAINT_FOLDER,
        con,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
        pick_current_hours,
        mode_norm,
    ):
        if not st.button("✅ Apply 'Done Now' updates", type="primary"):
            return
    
        done_rows = edited[edited["Done_Now"] == True].copy()
        if done_rows.empty:
            st.info("No tasks selected.")
            return
    
        updated = 0
        problems = []
    
        # ---- Update source files ----
        for src, grp in done_rows.groupby("Source_File"):
            path = os.path.join(MAINT_FOLDER, src)
            try:
                raw = read_file(path)
                df_src = normalize_df(raw)
    
                for _, r in grp.iterrows():
                    mode = mode_norm(r.get("Tracking_Mode", ""))
    
                    mask = (
                        df_src["Component"].astype(str).eq(str(r.get("Component", ""))) &
                        df_src["Task"].astype(str).eq(str(r.get("Task", "")))
                    )
                    if not mask.any():
                        continue
    
                    df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()
    
                    if mode == "hours":
                        df_src.loc[mask, "Last_Done_Hours"] = float(pick_current_hours(r.get("Hours_Source", "")))
                    elif mode == "draws":
                        df_src.loc[mask, "Last_Done_Draw"] = int(current_draw_count)
    
                    updated += int(mask.sum())
    
                out = templateize_df(df_src, list(raw.columns))
                write_file(path, out)
    
            except Exception as e:
                problems.append((src, str(e)))
    
        st.success(f"Updated {updated} task(s).")
    
        # ---- Log to DuckDB + CSV line log ----
        now_dt = dt.datetime.now()
        csv_rows = []
    
        for _, r in done_rows.iterrows():
            action_id = int(time.time() * 1000)
            mode = mode_norm(r.get("Tracking_Mode", ""))
    
            hs_raw = r.get("Hours_Source", "")
            hs_str = "" if hs_raw is None or (isinstance(hs_raw, float) and np.isnan(hs_raw)) else str(hs_raw).strip()
            if hs_str == "":
                hs_str = "FURNACE"
    
            # ALWAYS snapshot hours (for filtering/search)
            hours_snapshot = float(pick_current_hours(hs_str))
    
            done_hours_db = None
            done_draw = None
            if mode == "hours":
                done_hours_db = hours_snapshot
            elif mode == "draws":
                done_draw = int(current_draw_count)
    
            try:
                con.execute("""
                    INSERT INTO maintenance_actions
                    (action_id, action_ts, component, task, task_id, tracking_mode, hours_source,
                     done_date, done_hours, done_draw, source_file, actor, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    action_id,
                    now_dt,
                    str(r.get("Component", "")),
                    str(r.get("Task", "")),
                    str(r.get("Task_ID", "")),
                    str(r.get("Tracking_Mode", "")),
                    hs_str,
                    current_date,
                    done_hours_db,
                    done_draw,
                    str(r.get("Source_File", "")),
                    str(actor),
                    "",
                ])
            except Exception as e:
                st.warning(f"DuckDB insert failed (still saving CSV log): {e}")
    
            csv_rows.append({
                "maintenance_id": action_id,
                "maintenance_ts": now_dt,
                "maintenance_component": str(r.get("Component", "")),
                "maintenance_task": str(r.get("Task", "")),
                "maintenance_task_id": str(r.get("Task_ID", "")),
                "maintenance_mode": str(r.get("Tracking_Mode", "")),
                "maintenance_hours_source": hs_str,
                "maintenance_done_date": current_date,
                "maintenance_done_hours": hours_snapshot,  # ✅ always filled
                "maintenance_done_draw": done_draw if done_draw is not None else "",
                "maintenance_source_file": str(r.get("Source_File", "")),
                "maintenance_actor": str(actor),
                "maintenance_note": "",
            })
    
        if csv_rows:
            try:
                _append_csv(MAINT_ACTIONS_CSV, MAINT_ACTIONS_COLS, pd.DataFrame(csv_rows))
                st.caption("✅ Logged maintenance lines to maintenance_actions_log.csv")
            except Exception as e:
                st.error(f"Failed writing maintenance_actions_log.csv: {e}")
    
        if problems:
            st.warning("Some files had issues:")
            st.dataframe(pd.DataFrame(problems, columns=["File", "Error"]), use_container_width=True)
    
        st.rerun()
    
    # =========================================================
    # History viewer (DuckDB + CSV)
    # =========================================================
    def render_maintenance_history(con, limit: int = 200, height: int = 320):
        with st.expander("🗃️ Maintenance history (DuckDB)", expanded=False):
            try:
                recent = con.execute(f"""
                    SELECT action_ts, component, task, tracking_mode, hours_source,
                           done_date, done_hours, done_draw, actor, source_file
                    FROM maintenance_actions
                    ORDER BY action_ts DESC
                    LIMIT {int(limit)}
                """).fetchdf()
    
                if not recent.empty:
                    recent["done_date"] = pd.to_datetime(recent["done_date"], errors="coerce").dt.date
                    recent["action_ts"] = pd.to_datetime(recent["action_ts"], errors="coerce")
    
                st.dataframe(recent, use_container_width=True, height=int(height))
            except Exception as e:
                st.warning(f"DB read failed: {e}")
    
        with st.expander("🧾 Maintenance lines (CSV log)", expanded=False):
            if not os.path.isfile(MAINT_ACTIONS_CSV):
                st.info("No maintenance_actions_log.csv yet (mark something done first).")
            else:
                try:
                    df = pd.read_csv(MAINT_ACTIONS_CSV)
                    st.dataframe(df.tail(250), use_container_width=True, height=360)
                except Exception as e:
                    st.warning(f"CSV read failed: {e}")
    
    def render_gas_report(LOGS_FOLDER: str):
        """
        Gas usage report (MFC ACTUAL)
        Assumptions:
        - MFC columns contain BOTH 'MFC' and 'Actual'
        - Units are SLM (Standard Liters per Minute)
        - Integration: SL = Σ(SLM × dt_minutes)
        """
    
        st.markdown("---")
        st.subheader("🧪 Gas usage report (MFC actual, SLM)")
    
        show = st.toggle("Show gas report", value=False, key="gasrep_show")
        if not show:
            st.caption("(Hidden by default to keep UI light)")
            return
    
        if not os.path.isdir(LOGS_FOLDER):
            st.warning(f"Logs folder not found: {LOGS_FOLDER}")
            return
    
        # --------------------------------------------------
        # Collect log files
        # --------------------------------------------------
        csv_files = sorted(
            [os.path.join(LOGS_FOLDER, f)
             for f in os.listdir(LOGS_FOLDER)
             if f.lower().endswith(".csv") and not f.startswith("~$")],
            key=lambda p: os.path.getmtime(p),
        )
    
        if not csv_files:
            st.info("No log CSV files found.")
            return
    
        st.caption(f"Found {len(csv_files)} log files.")
    
        # --------------------------------------------------
        # Reports folder (auto-save)
        # --------------------------------------------------
        REPORT_DIR = P.gas_reports_dir
        os.makedirs(REPORT_DIR, exist_ok=True)
        st.caption(f"Reports folder: {REPORT_DIR}")
    
        # --------------------------------------------------
        # Time window selector
        # --------------------------------------------------
        st.markdown("#### Time window")
        c1, c2, c3, c4 = st.columns([1,1,1,2])
    
        st.session_state.setdefault("gasrep_window_days", 30)
    
        with c1:
            if st.button("Last 7 days", key="gasrep_btn_7", use_container_width=True):
                st.session_state["gasrep_window_days"] = 7
        with c2:
            if st.button("Last 30 days", key="gasrep_btn_30", use_container_width=True):
                st.session_state["gasrep_window_days"] = 30
        with c3:
            if st.button("Last 90 days", key="gasrep_btn_90", use_container_width=True):
                st.session_state["gasrep_window_days"] = 90
        with c4:
            st.caption(f"Selected: {st.session_state['gasrep_window_days']} days")
    
        window_days = int(st.session_state.get("gasrep_window_days", 30))
    
        # --------------------------------------------------
        # Helpers
        # --------------------------------------------------
        def _norm(s):
            return str(s).strip().lower()
    
        def _find_time_col(cols):
            for c in cols:
                if _norm(c) in {"date/time","datetime","timestamp","date time"}:
                    return c
            for c in cols:
                if "date" in _norm(c) and "time" in _norm(c):
                    return c
            return None
    
        def _is_mfc_actual(c):
            s = _norm(c)
            return ("mfc" in s) and ("actual" in s)
    
        # --------------------------------------------------
        # Scan logs and integrate usage
        # --------------------------------------------------
        rows = []
    
        for p in csv_files:
            try:
                df = pd.read_csv(p)
                if df is None or df.empty:
                    continue
    
                time_col = _find_time_col(df.columns)
                if not time_col:
                    continue
    
                t = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
                if t.isna().all():
                    continue
    
                df["__t"] = t
                df = df.dropna(subset=["__t"]).sort_values("__t").reset_index(drop=True)
                if len(df) < 2:
                    continue
    
                # dt in minutes
                dt_min = df["__t"].diff().dt.total_seconds() / 60.0
                dt_min = dt_min.fillna(0.0).clip(lower=0.0)
    
                mfc_cols = [c for c in df.columns if _is_mfc_actual(c)]
                if not mfc_cols:
                    continue
    
                total_sl = 0.0
                for c in mfc_cols:
                    flow_slm = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    total_sl += float((flow_slm * dt_min).sum())
    
                rows.append({
                    "log_file": os.path.basename(p),
                    "start_time": df["__t"].iloc[0],
                    "end_time": df["__t"].iloc[-1],
                    "duration_min": float(dt_min.sum()),
                    "Total SL": total_sl,
                })
            except Exception:
                continue
    
        if not rows:
            st.info("No usable MFC ACTUAL data detected in logs.")
            return
    
        usage = pd.DataFrame(rows)
        usage["start_time"] = pd.to_datetime(usage["start_time"], errors="coerce")
        usage = usage.sort_values("start_time").reset_index(drop=True)
    
        # --------------------------------------------------
        # Apply time window
        # --------------------------------------------------
        latest = usage["start_time"].max()
        if pd.isna(latest):
            latest = pd.Timestamp.now()
    
        t0 = latest - pd.Timedelta(days=window_days)
        usage = usage[usage["start_time"] >= t0]
    
        if usage.empty:
            st.warning("No logs in selected window.")
            return
    
        # --------------------------------------------------
        # Summary metrics
        # --------------------------------------------------
        total_sl = float(usage["Total SL"].sum())
        total_hours = float(usage["duration_min"].sum()) / 60.0
        avg_slm = (total_sl / usage["duration_min"].sum()) if usage["duration_min"].sum() > 0 else 0.0
    
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Gas Used (SL)", f"{total_sl:,.2f}")
        m2.metric("Total Duration (hours)", f"{total_hours:,.2f}")
        m3.metric("Average Flow (SLM)", f"{avg_slm:,.3f}")
    
        # --------------------------------------------------
        # Period reports
        # --------------------------------------------------
        tmp = usage.copy()
        tmp["Week"] = tmp["start_time"].dt.to_period("W").astype(str)
        tmp["Month"] = tmp["start_time"].dt.to_period("M").astype(str)
        tmp["Quarter"] = tmp["start_time"].dt.to_period("Q").astype(str)
    
        week_rep = tmp.groupby("Week", as_index=False)["Total SL"].sum().sort_values("Week")
        month_rep = tmp.groupby("Month", as_index=False)["Total SL"].sum().sort_values("Month")
        quarter_rep = tmp.groupby("Quarter", as_index=False)["Total SL"].sum().sort_values("Quarter")
    
        t1, t2, t3 = st.tabs(["Weekly", "Monthly", "3 Months"])
        with t1:
            st.dataframe(week_rep, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(month_rep, use_container_width=True, hide_index=True)
        with t3:
            st.dataframe(quarter_rep, use_container_width=True, hide_index=True)
    
        # --------------------------------------------------
        # Per log breakdown
        # --------------------------------------------------
        st.markdown("#### Per log file breakdown")
        st.dataframe(usage.tail(250), use_container_width=True, height=350)
    
        # --------------------------------------------------
        # Auto-save reports (FULL history from folder, not only selected window)
        # --------------------------------------------------
        try:
            full_usage = pd.DataFrame(rows)
            full_usage["start_time"] = pd.to_datetime(full_usage["start_time"], errors="coerce")
            full_usage["end_time"] = pd.to_datetime(full_usage["end_time"], errors="coerce")
            full_usage = full_usage.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    
            if not full_usage.empty:
                full_usage["Week"] = full_usage["start_time"].dt.to_period("W").astype(str)
                full_usage["Month"] = full_usage["start_time"].dt.to_period("M").astype(str)
                full_usage["Quarter"] = full_usage["start_time"].dt.to_period("Q").astype(str)
    
                # 1) Per-log summary
                out_all_logs = full_usage[[
                    "log_file", "start_time", "end_time", "duration_min", "Total SL", "Week", "Month", "Quarter"
                ]].copy()
                p1 = os.path.join(REPORT_DIR, "gas_summary_all_logs.csv")
                out_all_logs.to_csv(p1, index=False)
    
                # 2) Weekly totals
                week_agg = full_usage.groupby("Week", as_index=False).agg(total_sl=("Total SL", "sum"))
                week_agg = week_agg.sort_values("Week").reset_index(drop=True)
                p2 = os.path.join(REPORT_DIR, "gas_weekly_totals.csv")
                week_agg.to_csv(p2, index=False)
    
                # 3) Monthly totals + avg SLM for month
                month_agg = full_usage.groupby("Month", as_index=False).agg(
                    total_sl=("Total SL", "sum"),
                    total_minutes=("duration_min", "sum"),
                    n_logs=("log_file", "count"),
                    first_start=("start_time", "min"),
                    last_end=("end_time", "max"),
                )
                month_agg["avg_slm"] = month_agg.apply(
                    lambda r: (float(r["total_sl"]) / float(r["total_minutes"])) if float(r["total_minutes"]) > 0 else 0.0,
                    axis=1,
                )
                month_agg = month_agg.sort_values("Month").reset_index(drop=True)
                p3 = os.path.join(REPORT_DIR, "gas_monthly_totals.csv")
                month_agg.to_csv(p3, index=False)
    
                # 4) Quarterly totals
                q_agg = full_usage.groupby("Quarter", as_index=False).agg(total_sl=("Total SL", "sum"))
                q_agg = q_agg.sort_values("Quarter").reset_index(drop=True)
                p4 = os.path.join(REPORT_DIR, "gas_quarterly_totals.csv")
                q_agg.to_csv(p4, index=False)
    
                # Missing months detection (between first and last month)
                first_m = pd.Period(full_usage["start_time"].min(), freq="M")
                last_m = pd.Period(full_usage["start_time"].max(), freq="M")
                expected = [str(p) for p in pd.period_range(first_m, last_m, freq="M")]
                present = set(month_agg["Month"].astype(str).tolist())
                missing = [m for m in expected if m not in present]
    
                st.success("✅ Gas reports saved automatically")
                st.code("\n".join([p1, p2, p3, p4]))
    
                if missing:
                    st.warning("Missing months (no logs found): " + ", ".join(missing))
                else:
                    st.caption("No missing months detected between first and last log month.")
            else:
                st.info("No full-history rows available to save reports.")
        except Exception as e:
            st.warning(f"Auto-save failed: {e}")
    
        st.caption("Units: MFC Actual assumed SLM. Integrated to SL via SLM × dt(minutes).")
    
    # =========================================================
    # ✅ Faults section
    def render_faults_section(con, MAINT_FOLDER, actor):
        st.subheader("🚨 Faults / Incidents")
    
        faults_csv = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
        actions_csv = _read_csv_safe(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS)
        state_map = _latest_fault_state(actions_csv)
    
        if not faults_csv.empty:
            faults_csv["fault_id"] = pd.to_numeric(faults_csv["fault_id"], errors="coerce")
            faults_csv = faults_csv.dropna(subset=["fault_id"]).copy()
            faults_csv["fault_id"] = faults_csv["fault_id"].astype(int)
            faults_csv["fault_ts"] = pd.to_datetime(faults_csv["fault_ts"], errors="coerce")
    
            faults_csv["_is_closed"] = faults_csv["fault_id"].apply(
                lambda fid: bool(state_map.get(int(fid), {}).get("is_closed", False))
            )
            faults_csv["_last_action_ts"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_ts", None)
            )
            faults_csv["_last_action_type"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_type", "")
            )
            faults_csv["_last_action_actor"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_actor", "")
            )
            faults_csv["_last_fix"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_fix", "")
            )
        else:
            faults_csv = pd.DataFrame(columns=FAULTS_COLS + ["_is_closed", "_last_action_ts", "_last_action_type", "_last_action_actor", "_last_fix"])
    
        # ---- Log a new fault ----
        with st.expander("➕ Log a new fault", expanded=False):
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                comp_list = (
                    dfm["Component"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
                comp_list = sorted([c for c in comp_list if c])
                comp_options = comp_list + ["Other (custom)"]
    
                selected_comp = st.selectbox(
                    "Fault component",
                    options=comp_options,
                    key="fault_component_select"
                )
    
                if selected_comp == "Other (custom)":
                    fault_component = st.text_input(
                        "Custom component name",
                        key="fault_component_custom"
                    )
                else:
                    fault_component = selected_comp
            with c2:
                severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], index=1, key="fault_sev_in")
            with c3:
                related_draw = st.text_input("Related draw (optional)", placeholder="e.g. FP0888_1", key="fault_draw_in")
    
            title = st.text_input("Fault title", placeholder="Short title", key="fault_title_in")
            desc = st.text_area("Fault description", placeholder="What happened? what did you do? what to check next time?", height=120, key="fault_desc_in")
    
            cA, cB = st.columns([1, 1])
            with cA:
                src_file = st.text_input("Source file (optional)", placeholder="e.g. faults.xlsx / email.pdf / photo.jpg", key="fault_src_in")
            with cB:
                st.caption("Saved as BOTH DuckDB + faults_log.csv")
    
            if st.button("➕ Log fault", type="primary", use_container_width=True, key="fault_add_btn"):
                if not str(fault_component).strip():
                    st.warning("Fault component is required.")
                    st.stop()
                if not str(title).strip() and not str(desc).strip():
                    st.warning("Give at least a title or description.")
                    st.stop()
    
                now_dt = dt.datetime.now()
                fid = int(time.time() * 1000)
    
                try:
                    con.execute("""
                        INSERT INTO faults_events
                        (fault_id, fault_ts, component, title, description, severity, actor, source_file, related_draw)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        fid, now_dt,
                        str(fault_component), str(title), str(desc),
                        str(severity), str(actor), str(src_file), str(related_draw)
                    ])
                except Exception as e:
                    st.warning(f"DuckDB insert failed (still saving CSV log): {e}")
    
                row = pd.DataFrame([{
                    "fault_id": fid,
                    "fault_ts": now_dt,
                    "fault_component": str(fault_component),
                    "fault_title": str(title),
                    "fault_description": str(desc),
                    "fault_severity": str(severity),
                    "fault_actor": str(actor),
                    "fault_source_file": str(src_file),
                    "fault_related_draw": str(related_draw),
                }])
                try:
                    _append_csv(FAULTS_CSV, FAULTS_COLS, row)
                    st.success("Fault logged.")
                except Exception as e:
                    st.error(f"Failed writing faults_log.csv: {e}")
    
                st.rerun()
    
        # ---- Open faults list ----
        with st.expander("🔓 Open faults", expanded=False):
            open_df = faults_csv[faults_csv["_is_closed"] == False].copy()
            open_df = open_df.sort_values("fault_ts", ascending=False)

            if open_df.empty:
                st.success("No open faults 👍")
            else:
                for _, r in open_df.iterrows():
                    fid = int(r["fault_id"])
                    comp = safe_str(r.get("fault_component", ""))
                    sev = safe_str(r.get("fault_severity", ""))
                    title = safe_str(r.get("fault_title", "")) or "Fault"
                    ts = safe_str(r.get("fault_ts", ""))

                    c1, c2, c3 = st.columns([3.4, 1.1, 1.1])
                    with c1:
                        st.markdown(f"**[{sev.upper()}] {comp} — {title}**")
                        st.caption(f"ID: `{fid}`  |  Time: {ts}")

                    with c2:
                        @st.dialog(f"Close fault: {comp} — {title} (#{fid})")
                        def _dlg_close():
                            fix = st.text_input("Fix summary (short)", key=f"fix_sum__{fid}")
                            note = st.text_area("Closure notes", height=120, key=f"fix_note__{fid}")
                            if st.button("✅ Close fault", type="primary", use_container_width=True, key=f"close_do__{fid}"):
                                _write_fault_action(con, fault_id=fid, action_type="close", actor=actor, note=note, fix_summary=fix)
                                st.success("Closed.")
                                st.rerun()

                        if st.button("✅ Close", use_container_width=True, key=f"btn_close__{fid}"):
                            _dlg_close()

                    with c3:
                        @st.dialog(f"Add note: #{fid}")
                        def _dlg_note():
                            note = st.text_area("Note", height=120, key=f"note_txt__{fid}")
                            if st.button("➕ Save note", type="primary", use_container_width=True, key=f"note_do__{fid}"):
                                _write_fault_action(con, fault_id=fid, action_type="note", actor=actor, note=note, fix_summary="")
                                st.success("Saved note.")
                                st.rerun()

                        if st.button("📝 Note", use_container_width=True, key=f"btn_note__{fid}"):
                            _dlg_note()

                    desc_txt = safe_str(r.get("fault_description", "")) or "—"
                    st.caption(f"Details: {desc_txt}")
                    st.caption(f"Source file: {safe_str(r.get('fault_source_file',''))} | Related draw: {safe_str(r.get('fault_related_draw',''))}")

                    st.divider()
    
        # ---- All faults table + reopen ----
        with st.expander("📜 All faults (table)", expanded=False):
            df_all = faults_csv.copy()
            if df_all.empty:
                st.info("No faults yet.")
            else:
                df_all["Status"] = np.where(df_all["_is_closed"], "Closed", "Open")
                df_all["Last Action"] = df_all["_last_action_type"]
                df_all["Last Action By"] = df_all["_last_action_actor"]
                df_all["Last Fix Summary"] = df_all["_last_fix"]
                show = df_all[[
                    "fault_ts", "Status", "fault_id", "fault_component", "fault_severity",
                    "fault_title", "fault_actor", "fault_related_draw",
                    "Last Action", "Last Action By", "Last Fix Summary"
                ]].copy()
                st.dataframe(show, use_container_width=True, height=360, hide_index=True)
    
                closed_ids = df_all[df_all["_is_closed"] == True]["fault_id"].astype(int).tolist()
                if closed_ids:
                    st.markdown("##### Reopen a fault")
                    pick = st.selectbox("Closed fault ID", options=[""] + [str(x) for x in closed_ids], key="reopen_pick")
                    if pick and st.button("♻️ Reopen", use_container_width=True, key="reopen_btn"):
                        _write_fault_action(con, fault_id=int(pick), action_type="reopen", actor=actor, note="Reopened", fix_summary="")
                        st.success("Reopened.")
                        st.rerun()
    
        with st.expander("🧾 Fault actions (CSV log)", expanded=False):
            if not os.path.isfile(FAULTS_ACTIONS_CSV):
                st.info("No faults_actions_log.csv yet (close/reopen/note first).")
            else:
                try:
                    df = pd.read_csv(FAULTS_ACTIONS_CSV)
                    st.dataframe(df.tail(300), use_container_width=True, height=360)
                except Exception as e:
                    st.warning(f"Fault actions CSV read failed: {e}")
    
    # =========================================================
    # Load report + tasks editor
    # =========================================================
    def render_maintenance_load_report(files, load_errors):
        with st.expander("Load report", expanded=False):
            try:
                st.write("Loaded files:", sorted(list(files or [])))
            except Exception:
                st.write("Loaded files:", files)
    
            if load_errors:
                st.warning("Some files failed to load:")
                st.dataframe(pd.DataFrame(load_errors, columns=["File", "Error"]), use_container_width=True)
    
    def render_maintenance_tasks_editor(
        MAINT_FOLDER,
        files,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
    ):
        with st.expander("📝 Maintenance tasks editor (source files)", expanded=False):
            st.caption("Edits the selected maintenance file (Excel/CSV) and saves back.")
            pick = st.selectbox("Select maintenance file", options=sorted(files), key="maint_edit_file_pick")
            if not pick:
                return
            path = os.path.join(MAINT_FOLDER, pick)
            try:
                raw = read_file(path)
                if raw is None or raw.empty:
                    st.info("File is empty.")
                    return
                df = normalize_df(raw)
    
                show_cols = [c for c in df.columns if c != "Source_File"]
                edited = st.data_editor(df[show_cols], use_container_width=True, height=420, key="maint_tasks_editor_grid")
    
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("💾 Save file", type="primary", use_container_width=True, key="maint_save_file_btn"):
                        out = templateize_df(edited, list(raw.columns))
                        write_file(path, out)
                        st.success("Saved.")
                        st.rerun()
                with c2:
                    st.caption("Saved back in the original template columns.")
            except Exception as e:
                st.warning(f"Tasks editor failed: {e}")
    
    # =========================================================
    # Manuals / Documents browser (same preview style)
    # =========================================================
    def render_manuals_browser(BASE_DIR):
        MANUALS_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    
        def _ext(p: str) -> str:
            return os.path.splitext(str(p).lower())[1]
    
        def _is_pdf(p: str) -> bool:
            return str(p).lower().endswith(".pdf")
    
        def _is_img(p: str) -> bool:
            return _ext(p) in MANUALS_IMG_EXTS
    
        def _short_name(fn: str, max_len: int = 42) -> str:
            fn = str(fn)
            if len(fn) <= max_len:
                return fn
            keep_tail = 16
            head = max_len - keep_tail - 3
            return fn[:head] + "..." + fn[-keep_tail:]
    
        @st.cache_data(show_spinner=False)
        def _read_bytes(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()
    
        def _download_btn(path: str, label: str, key: str):
            if not os.path.exists(path):
                st.warning(f"Missing file: {os.path.basename(path)}")
                return
            data = _read_bytes(path)
            st.download_button(
                label=label,
                data=data,
                file_name=os.path.basename(path),
                mime=None,
                key=key,
                use_container_width=True,
            )
    
        @st.cache_data(show_spinner=False)
        def _pdf_render_pages(path: str, max_pages: int = 1, zoom: float = 1.6):
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            n = min(len(doc), int(max_pages))
            out = []
            mat = fitz.Matrix(float(zoom), float(zoom))
            for i in range(n):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out.append(pix.tobytes("png"))
            doc.close()
            return out
    
        def _render_pdf_preview(path: str, *, key_prefix: str):
            if not os.path.exists(path):
                st.warning("PDF file not found.")
                return
    
            state_key = f"{key_prefix}__show_all"
            st.session_state.setdefault(state_key, False)
    
            c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
            with c1:
                st.markdown("**PDF preview (rendered)**")
                st.caption("Default shows page 1. Click to render more pages.")
            with c2:
                zoom = st.selectbox("Quality", [1.3, 1.6, 2.0], index=1, key=f"{key_prefix}__zoom")
            with c3:
                max_pages = st.number_input(
                    "Pages (when expanded)", min_value=1, max_value=200, value=30, step=1, key=f"{key_prefix}__pages"
                )
    
            b1, b2 = st.columns([1, 1])
            with b1:
                if not st.session_state[state_key]:
                    if st.button("📄 Render more pages", use_container_width=True, key=f"{key_prefix}__more"):
                        st.session_state[state_key] = True
                        st.rerun()
                else:
                    if st.button("⬅️ Back to page 1", use_container_width=True, key=f"{key_prefix}__less"):
                        st.session_state[state_key] = False
                        st.rerun()
            with b2:
                _download_btn(path, "⬇️ Download PDF", key=f"{key_prefix}__dl")
    
            try:
                if st.session_state[state_key]:
                    imgs = _pdf_render_pages(path, max_pages=int(max_pages), zoom=float(zoom))
                    st.caption(f"Showing **{len(imgs)}** page(s).")
                    for i, b in enumerate(imgs, start=1):
                        st.image(b, caption=f"Page {i}", use_container_width=True)
                else:
                    imgs = _pdf_render_pages(path, max_pages=1, zoom=float(zoom))
                    if imgs:
                        st.image(imgs[0], caption="Page 1", use_container_width=True)
            except Exception as e:
                st.error(f"PDF render failed. Install PyMuPDF: `pip install pymupdf`  |  Error: {e}")
    
        with st.expander("📚 Manuals / Documents browser", expanded=False):
            st.caption("Tight checklist view: select manuals, then preview one.")
    
            candidate_dirs = [
                os.path.join(BASE_DIR, "manuals"),
                os.path.join(BASE_DIR, "docs"),
                os.path.join(BASE_DIR, "maintenance", "manuals"),
                os.path.join(BASE_DIR, "maintenance", "docs"),
            ]
            existing = [d for d in candidate_dirs if os.path.isdir(d)]
            if not existing:
                st.info("No manuals/docs folder found. (Create /manuals or /docs).")
                return
    
            root = st.selectbox("Folder", existing, key="maint_manuals_root_pick")
    
            paths = sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True))
            paths = [p for p in paths if os.path.isfile(p)]
            if not paths:
                st.info("No files found.")
                return
    
            c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
            with c1:
                q = st.text_input("Search", placeholder="type filename…", key="maint_manuals_search")
            with c2:
                kind = st.selectbox("Type", ["All", "PDF", "Images", "Other"], key="maint_manuals_type")
            with c3:
                limit = st.number_input("Show (max)", 10, 500, 120, 10, key="maint_manuals_limit")
    
            def _match(p):
                fn = os.path.basename(p).lower()
                if q and q.lower().strip() not in fn:
                    return False
                if kind == "PDF" and not _is_pdf(p):
                    return False
                if kind == "Images" and not _is_img(p):
                    return False
                if kind == "Other" and (_is_pdf(p) or _is_img(p)):
                    return False
                return True
    
            shown = [p for p in paths if _match(p)]
            st.caption(f"Files found: **{len(shown)}** (total in folder: {len(paths)})")
            shown = shown[: int(limit)]
    
            st.session_state.setdefault("maint_manuals_checked", [])
            st.session_state.setdefault("maint_manuals_active", "")
    
            st.markdown("#### ✅ Select manuals")
            checked = set(st.session_state.get("maint_manuals_checked", []))
    
            for i, p in enumerate(shown):
                fn = os.path.basename(p)
                col0, col1, col2 = st.columns([0.35, 5.0, 1.0], gap="small")
                with col0:
                    is_on = st.checkbox("", value=(p in checked), key=f"maint_manuals_chk__{i}")
                with col1:
                    st.markdown(f"**{_short_name(fn)}**")
                with col2:
                    _download_btn(p, "⬇️", key=f"maint_manuals_dl__{i}__{fn}")
    
                if is_on:
                    checked.add(p)
                else:
                    checked.discard(p)
    
            st.session_state["maint_manuals_checked"] = sorted(list(checked))
    
            st.divider()
    
            picked_list = st.session_state["maint_manuals_checked"]
            if not picked_list:
                st.info("Select at least one manual to preview.")
                return
    
            if st.session_state["maint_manuals_active"] not in picked_list:
                st.session_state["maint_manuals_active"] = picked_list[0]
    
            labels = {p: os.path.basename(p) for p in picked_list}
            active = st.selectbox(
                "👁️ Preview selected manual",
                options=picked_list,
                format_func=lambda p: labels.get(p, p),
                key="maint_manuals_active",
            )
    
            st.markdown("### Preview")
            st.caption(os.path.basename(active))
    
            if _is_pdf(active):
                _render_pdf_preview(active, key_prefix=f"maint_manuals_pdf__{os.path.basename(active)}")
            elif _is_img(active):
                st.image(active, use_container_width=True)
            else:
                st.info("No preview for this file type (use Download).")
    
            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("🧹 Clear selection", use_container_width=True, key="maint_manuals_clear"):
                    st.session_state["maint_manuals_checked"] = []
                    st.session_state["maint_manuals_active"] = ""
                    st.rerun()
            with cB:
                _download_btn(active, "⬇️ Download active", key="maint_manuals_dl_active")
    
    # =========================================================
    # UI flow
    # =========================================================
    st.markdown('<div class="maint-section-title">📊 Dashboard</div>', unsafe_allow_html=True)
    render_maintenance_dashboard_metrics(dfm)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
          .maint-group-wrap{
            display:grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap:10px;
            margin: 4px 0 10px 0;
          }
          .maint-group-card{
            border-radius:12px;
            border:1px solid rgba(132,214,255,0.28);
            background: linear-gradient(180deg, rgba(14,32,56,0.34), rgba(8,16,28,0.24));
            padding:8px 10px;
          }
          .maint-group-title{
            font-size:0.95rem;
            font-weight:800;
            color:rgba(230,246,255,0.98);
          }
          .maint-group-sub{
            font-size:0.78rem;
            color:rgba(188,224,248,0.88);
            margin-top:3px;
          }
          @media (max-width: 1100px){
            .maint-group-wrap{ grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="maint-group-wrap">
          <div class="maint-group-card"><div class="maint-group-title">🧰 Maintenance</div><div class="maint-group-sub">Tasks, schedule, reschedule, planning, history, editors</div></div>
          <div class="maint-group-card"><div class="maint-group-title">🚨 Faults</div><div class="maint-group-sub">Open incidents, close/reopen, notes, actions log</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("maint_main_group", "maintenance")
    if st.session_state.get("maint_main_group") == "gas":
        st.session_state["maint_main_group"] = "maintenance"
    g1, g2 = st.columns(2)
    if g1.button("🧰 Maintenance", key="maint_group_btn_maint", use_container_width=True, type="primary" if st.session_state["maint_main_group"] == "maintenance" else "secondary"):
        st.session_state["maint_main_group"] = "maintenance"
        st.rerun()
    if g2.button("🚨 Faults", key="maint_group_btn_faults", use_container_width=True, type="primary" if st.session_state["maint_main_group"] == "faults" else "secondary"):
        st.session_state["maint_main_group"] = "faults"
        st.rerun()

    group = st.session_state.get("maint_main_group", "maintenance")

    if group == "maintenance":
        with st.expander("📅 Future Schedule", expanded=False):
            st.caption("Choose type to show timeline.")
            horizon_hours, horizon_days, horizon_draws = render_maintenance_horizon_selector(current_draw_count)
            focus = render_future_schedule_focus_selector()
            render_maintenance_roadmaps(
                dfm,
                current_date,
                current_draw_count,
                furnace_hours,
                uv1_hours,
                uv2_hours,
                horizon_hours,
                horizon_days,
                horizon_draws,
                focus=focus,
            )

        with st.expander("✅ Mark Tasks Done", expanded=False):
            edited = render_maintenance_done_editor(dfm)
            render_maintenance_apply_done(
                edited,
                dfm=dfm,
                current_date=current_date,
                current_draw_count=current_draw_count,
                actor=actor,
                MAINT_FOLDER=MAINT_FOLDER,
                con=con,
                read_file=read_file,
                write_file=write_file,
                normalize_df=normalize_df,
                templateize_df=templateize_df,
                pick_current_hours=pick_current_hours,
                mode_norm=mode_norm,
            )

        with st.expander("⏩ Quick Reschedule", expanded=False):
            render_quick_reschedule_panel(
                dfm=dfm,
                MAINT_FOLDER=MAINT_FOLDER,
                current_draw_count=current_draw_count,
                furnace_hours=furnace_hours,
                uv1_hours=uv1_hours,
                uv2_hours=uv2_hours,
                read_file=read_file,
                write_file=write_file,
                normalize_df=normalize_df,
                templateize_df=templateize_df,
            )

        with st.expander("🗓️ Maintenance Scheduler Bridge", expanded=False):
            render_maintenance_scheduler_bridge(
                dfm=dfm,
                current_date=current_date,
                current_draw_count=current_draw_count,
                furnace_hours=furnace_hours,
                uv1_hours=uv1_hours,
                uv2_hours=uv2_hours,
            )
        # These renderers already contain their own expanders; avoid wrapping to prevent nested-expander errors.
        render_maintenance_history(con)
        render_maintenance_load_report(files, load_errors)
        render_maintenance_tasks_editor(
            MAINT_FOLDER=MAINT_FOLDER,
            files=files,
            read_file=read_file,
            write_file=write_file,
            normalize_df=normalize_df,
            templateize_df=templateize_df,
        )
        render_manuals_browser(BASE_DIR)

    elif group == "faults":
        render_faults_section(
            con=con,
            MAINT_FOLDER=MAINT_FOLDER,
            actor=actor,
        )
    # ------------------ Correlation & Outliers ------------------
