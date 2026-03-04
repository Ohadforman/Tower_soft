def render_sql_lab_tab(P):
    import os, glob, re
    import pandas as pd
    import numpy as np
    import streamlit as st
    import duckdb
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from plotly.subplots import make_subplots
    
    # NOTE:
    # Best practice is to call st.set_page_config(layout="wide") ONCE at the top of the app.
    # We guard it here so re-runs / other tabs won't break the app.
    if "_page_config_set" not in st.session_state:
        try:
            st.set_page_config(layout="wide")
        except Exception:
            pass
        st.session_state["_page_config_set"] = True
    
    # Force wide-looking layout even if page_config is set elsewhere
    st.markdown(
        """
        <style>
        .block-container { max-width: 98% !important; padding-left: 1.8rem; padding-right: 1.8rem; padding-top: 2.35rem; }
        .sql-top-spacer{ height: 8px; }
        .sql-title{
            font-size: 1.62rem;
            font-weight: 900;
            margin: 0;
            padding-top: 4px;
            line-height: 1.2;
            color: rgba(236,248,255,0.98);
            text-shadow: 0 0 14px rgba(86,178,255,0.22);
        }
        .sql-sub{
            margin: 4px 0 8px 0;
            font-size: 0.92rem;
            color: rgba(188,224,248,0.88);
        }
        .sql-line{
            height: 1px;
            margin: 0 0 12px 0;
            background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
        }
        .sql-section{
            margin-top: 8px;
            margin-bottom: 8px;
            padding-left: 8px;
            border-left: 3px solid rgba(120,200,255,0.62);
            font-size: 1.04rem;
            font-weight: 820;
            color: rgba(230,246,255,0.98);
        }
        .st-key-sql_ui_step_btn_1 button,
        .st-key-sql_ui_step_btn_2 button,
        .st-key-sql_ui_step_btn_3 button,
        .st-key-sql_ui_step_btn_4 button {
            border: 1px solid rgba(120, 205, 255, 0.44) !important;
            border-radius: 12px !important;
            background: linear-gradient(145deg, rgba(16, 28, 44, 0.66), rgba(10, 18, 30, 0.48)) !important;
            color: rgba(232, 245, 255, 0.98) !important;
            min-height: 58px !important;
            font-weight: 800 !important;
        }
        .st-key-sql_ui_step_btn_1 button:hover,
        .st-key-sql_ui_step_btn_2 button:hover,
        .st-key-sql_ui_step_btn_3 button:hover,
        .st-key-sql_ui_step_btn_4 button:hover {
            transform: translateY(-2px) scale(1.015) !important;
            border-color: rgba(156, 224, 255, 0.72) !important;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.30), 0 0 18px rgba(92, 190, 255, 0.34) !important;
        }
        .st-key-sql_ui_step_btn_1 button[kind="primary"],
        .st-key-sql_ui_step_btn_2 button[kind="primary"],
        .st-key-sql_ui_step_btn_3 button[kind="primary"],
        .st-key-sql_ui_step_btn_4 button[kind="primary"] {
            border-color: rgba(172, 232, 255, 0.90) !important;
            background: linear-gradient(145deg, rgba(34, 70, 108, 0.88), rgba(18, 42, 72, 0.80)) !important;
            box-shadow: 0 14px 28px rgba(24, 88, 138, 0.40), 0 0 24px rgba(96, 196, 255, 0.45) !important;
        }
        .st-key-sql_run button {
            border-radius: 12px !important;
            border: 1px solid rgba(172, 232, 255, 0.90) !important;
            background: linear-gradient(145deg, rgba(34, 70, 108, 0.88), rgba(18, 42, 72, 0.80)) !important;
            color: rgba(238, 250, 255, 0.98) !important;
            box-shadow: 0 14px 28px rgba(24, 88, 138, 0.40), 0 0 24px rgba(96, 196, 255, 0.45) !important;
            font-weight: 800 !important;
        }
        .st-key-sql_run button:hover {
            border-color: rgba(192, 240, 255, 0.98) !important;
            box-shadow: 0 16px 32px rgba(24, 88, 138, 0.44), 0 0 28px rgba(96, 196, 255, 0.52) !important;
            transform: translateY(-1px) scale(1.01) !important;
        }
        body:has(.st-key-sql_ui_step_btn_1) div[data-testid="stTextInput"] input,
        body:has(.st-key-sql_ui_step_btn_1) div[data-testid="stTextArea"] textarea,
        body:has(.st-key-sql_ui_step_btn_1) div[data-testid="stNumberInput"] input,
        body:has(.st-key-sql_ui_step_btn_1) div[data-testid="stDateInput"] input {
            background: rgba(10, 18, 30, 0.66) !important;
            border: 1px solid rgba(132, 210, 255, 0.30) !important;
            border-radius: 10px !important;
            color: rgba(236, 248, 255, 0.98) !important;
        }
        body:has(.st-key-sql_ui_step_btn_1) div[data-baseweb="select"] > div,
        body:has(.st-key-sql_ui_step_btn_1) div[data-baseweb="select"] input {
            background: rgba(10, 18, 30, 0.66) !important;
            border-color: rgba(132, 210, 255, 0.30) !important;
            color: rgba(236, 248, 255, 0.98) !important;
        }
        body:has(.st-key-sql_ui_step_btn_1) div[data-baseweb="tag"] {
            background: rgba(34, 66, 102, 0.66) !important;
            border: 1px solid rgba(160, 228, 255, 0.44) !important;
            color: rgba(240, 251, 255, 0.98) !important;
        }
        .sql-help {
            border: 1px solid rgba(132, 214, 255, 0.24);
            border-radius: 10px;
            padding: 8px 10px;
            margin: 6px 0 10px 0;
            background: rgba(10, 20, 34, 0.42);
            color: rgba(230, 246, 255, 0.95);
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .sql-subhead {
            font-size: 0.90rem;
            font-weight: 760;
            color: rgba(170, 226, 255, 0.96);
            margin: 4px 0 6px 0;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(145, 214, 255, 0.18) !important;
            border-radius: 12px !important;
            background: rgba(8, 14, 24, 0.35) !important;
            overflow: hidden !important;
        }
        div[data-testid="stExpander"] details[open] {
            border: 1px solid rgba(150, 222, 255, 0.44) !important;
            box-shadow: 0 14px 28px rgba(0,0,0,0.28), 0 0 18px rgba(84, 182, 255, 0.24) !important;
            background: linear-gradient(165deg, rgba(10, 20, 34, 0.72), rgba(8, 14, 24, 0.52)) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sql-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sql-title">🧪 SQL Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sql-sub">Filter draw CSVs with AND/OR/NOT, then overlay Maintenance and Faults in a separate events lane. Click any point to inspect.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sql-line"></div>', unsafe_allow_html=True)
    st.session_state.setdefault("sql_ui_step", 1)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        if st.button("STEP 1 · Pick parameter / group", key="sql_ui_step_btn_1", use_container_width=True, type="primary" if st.session_state["sql_ui_step"] == 1 else "secondary"):
            st.session_state["sql_ui_step"] = 1
            st.rerun()
    with s2:
        if st.button("STEP 2 · Set condition + scope", key="sql_ui_step_btn_2", use_container_width=True, type="primary" if st.session_state["sql_ui_step"] == 2 else "secondary"):
            st.session_state["sql_ui_step"] = 2
            st.rerun()
    with s3:
        if st.button("STEP 3 · Add conditions to filter", key="sql_ui_step_btn_3", use_container_width=True, type="primary" if st.session_state["sql_ui_step"] == 3 else "secondary"):
            st.session_state["sql_ui_step"] = 3
            st.rerun()
    with s4:
        if st.button("STEP 4 · Run and inspect results", key="sql_ui_step_btn_4", use_container_width=True, type="primary" if st.session_state["sql_ui_step"] == 4 else "secondary"):
            st.session_state["sql_ui_step"] = 4
            st.rerun()
    active_step = st.session_state.get("sql_ui_step", 1)

    def _step_label(base: str, step_num: int) -> str:
        return f"{base}  •  ACTIVE" if active_step == step_num else base
    
    DATASET_DIR = P.dataset_dir
    DB_PATH = P.duckdb_path
    
    # =========================================================
    # Persistent DuckDB connection
    # =========================================================
    if "sql_duck_con" not in st.session_state:
        st.session_state["sql_duck_con"] = duckdb.connect(DB_PATH)
    con = st.session_state["sql_duck_con"]
    
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass
    
    # =========================================================
    # Ensure required DB tables exist
    # =========================================================
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
    
    # =========================================================
    # Helpers
    # =========================================================
    def _esc(s: str) -> str:
        return (s or "").replace("'", "''")
    
    def _lit(x) -> str:
        return "'" + _esc(str(x)) + "'"
    
    def _is_num(x) -> bool:
        try:
            float(str(x))
            return True
        except Exception:
            return False
    
    def _dedupe_keep_order(seq):
        out, seen = [], set()
        for x in seq or []:
            s = str(x).strip() if x is not None else ""
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    
    def _mtime_ts(path: str):
        try:
            return pd.to_datetime(os.path.getmtime(path), unit="s")
        except Exception:
            return pd.NaT
    
    def _tokenize_search(q: str):
        q = (q or "").strip().lower()
        if not q:
            return []
        return [t.strip() for t in re.split(r"[,\s]+", q) if t.strip()]
    
    def _match_params_by_tokens(params, tokens):
        if not tokens:
            return list(params)
        out = []
        for p in params:
            pl = str(p).lower()
            if all(t in pl for t in tokens):
                out.append(p)
        return out
    
    def _extract_zone_num(pname: str):
        s = str(pname or "")
        m = re.search(r"(?i)\bzone\D*([0-9]{1,3})\b", s)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None
    
    def _filters_summary_for_draws(used_params_list, human_lines_list) -> str:
        used_params_list = _dedupe_keep_order(used_params_list or [])
        parts = []
        if used_params_list:
            parts.append("Params: " + ", ".join(used_params_list))
        if human_lines_list:
            hl = [str(x).strip() for x in (human_lines_list or []) if str(x).strip()]
            if hl:
                parts.append(" | ".join(hl[:6]) + (" …" if len(hl) > 6 else ""))
        return " || ".join(parts).strip()
    
    # =========================================================
    # Build DuckDB view for dataset CSVs (KV)
    # =========================================================
    def build_datasets_kv_view_from_disk() -> int:
        files = glob.glob(os.path.join(DATASET_DIR, "**", "*.csv"), recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            con.execute("""
                CREATE OR REPLACE VIEW datasets_kv AS
                SELECT
                    'dataset'::VARCHAR AS source_kind,
                    NULL::TIMESTAMP AS event_ts,
                    NULL::VARCHAR AS event_id,
                    NULL::VARCHAR AS _draw,
                    NULL::VARCHAR AS _file,
                    NULL::VARCHAR AS filename,
                    NULL::VARCHAR AS "Parameter Name",
                    NULL::VARCHAR AS "Value",
                    NULL::VARCHAR AS "Units"
                WHERE FALSE;
            """)
            return 0
    
        files = [f.replace("\\", "/") for f in files]
        files_sql = "[" + ",".join(_lit(f) for f in files) + "]"
    
        con.execute(f"""
            CREATE OR REPLACE VIEW datasets_kv AS
            WITH raw AS (
                SELECT
                    *,
                    filename,
                    regexp_extract(filename, '[^/]+$', 0) AS _file,
                    regexp_extract(filename, '([^/]+)\\.csv$', 1) AS _draw
                FROM read_csv_auto(
                    {files_sql},
                    filename=true,
                    union_by_name=true,
                    ALL_VARCHAR=TRUE,
                    ignore_errors=true
                )
            ),
            ts AS (
                SELECT
                    _draw,
                    MAX(TRY_CAST("Value" AS TIMESTAMP)) FILTER (
                        WHERE lower(trim("Parameter Name")) IN ('draw date','draw datetime')
                           OR lower(trim("Parameter Name")) LIKE '%draw date%'
                           OR lower(trim("Parameter Name")) LIKE '%draw time%'
                           OR lower(trim("Parameter Name")) LIKE '%datetime%'
                    ) AS draw_ts
                FROM raw
                GROUP BY _draw
            )
            SELECT
                'dataset'::VARCHAR AS source_kind,
                ts.draw_ts AS event_ts,
                raw._draw::VARCHAR AS event_id,
                raw._draw,
                raw._file,
                raw.filename,
                raw."Parameter Name",
                raw."Value",
                raw."Units"
            FROM raw
            LEFT JOIN ts USING (_draw);
        """)
        return len(files)
    
    def ensure_views():
        try:
            con.execute("SELECT COUNT(*) FROM datasets_kv").fetchone()
        except Exception:
            n = build_datasets_kv_view_from_disk()
            st.caption(f"Indexed dataset CSVs: {n}")
    
    # =========================================================
    # Indexing controls
    # =========================================================
    with st.expander(_step_label("📁 Indexing", 1), expanded=(active_step == 1)):
        st.markdown(
            """
            <div class="sql-help">
              <b>Indexing control:</b> Rebuild after adding/removing dataset CSV files.
              Use <b>Reset SQL state</b> only when the tab behaves unexpectedly.
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("🔄 Rebuild dataset index", use_container_width=True, key="sql_rebuild_kv"):
            for k in [
                "sql_df_all", "sql_matched_draws", "sql_selected_event_key", "math_selected_event_key",
                "ds_conditions", "ds_conditions_human", "ds_conditions_struct",
                "sql_filter_params_seq", "sql_group_selected_params",
                "sql_values_found_long", "sql_values_found_wide", "sql_last_filters_summary",
                "sql_matched_params_only", "sql_plot_params",
                "sql_group_defs_for_plot",
            ]:
                st.session_state.pop(k, None)
            n = build_datasets_kv_view_from_disk()
            st.success(f"Rebuilt dataset index. Files: {n}")
    
        if c2.button("🧹 Reset SQL state", use_container_width=True, key="sql_reset_state"):
            for k in list(st.session_state.keys()):
                if k.startswith(("sql_", "math_", "ds_")):
                    st.session_state.pop(k, None)
            st.success("Reset done.")
            st.stop()
    
    ensure_views()
    
    # =========================================================
    # Filter builder state
    # =========================================================
    st.session_state.setdefault("ds_conditions", [])
    st.session_state.setdefault("ds_conditions_human", [])
    st.session_state.setdefault("ds_conditions_struct", [])
    st.session_state.setdefault("sql_filter_params_seq", [])
    st.session_state.setdefault("sql_last_human_lines", [])
    st.session_state.setdefault("sql_last_filters_summary", "")
    st.session_state.setdefault("sql_group_selected_params", [])
    st.session_state.setdefault("sql_matched_params_only", [])
    st.session_state.setdefault("sql_group_defs_for_plot", [])
    st.session_state.setdefault("sql_plot_params", [])
    
    # =========================================================
    # Available params list
    # =========================================================
    params_df = con.execute("""
        SELECT DISTINCT "Parameter Name"
        FROM datasets_kv
        WHERE "Parameter Name" IS NOT NULL AND trim("Parameter Name") <> ''
        ORDER BY 1
    """).fetchdf()
    all_params = params_df["Parameter Name"].astype(str).tolist() if not params_df.empty else []
    
    if not all_params:
        st.warning("No dataset parameters were found (datasets_kv empty).")
        st.stop()
    
    # =========================================================
    # FILTER UI
    # =========================================================
    with st.expander(_step_label("1) 🔎 Pick parameter / group", 1), expanded=(active_step == 1)):
        st.markdown(
            """
            <div class="sql-help">
              <b>Goal:</b> choose the parameter(s) you want to filter.<br>
              Start with <b>Single parameter</b> for quick runs, or switch to <b>Group</b> for multi-zone filtering.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sql-subhead'>🔎 Parameter Search</div>", unsafe_allow_html=True)
        search_q = st.text_input(
            "Search parameters",
            placeholder="Examples: zone avg diameter | tension | furnace",
            key="sql_param_search2",
            help="Type multiple words — results must contain ALL words.",
        )
        tokens = _tokenize_search(search_q)
        matches_all = _match_params_by_tokens(all_params, tokens)
        matches = matches_all[:500]
    
        zone_map = {}
        for nm in matches:
            zi = _extract_zone_num(nm)
            if zi is not None:
                zone_map.setdefault(zi, []).append(nm)
        zone_nums = sorted(zone_map.keys())
        has_zone_matches = len(zone_nums) >= 2
    
        cA, cB, cC = st.columns([1.2, 1.2, 1])
        with cA:
            select_mode = st.radio(
                "Pick mode",
                ["Single parameter", "Group from search results"],
                horizontal=True,
                key="sql_pick_mode_builder",
            )
            st.caption("Single = one parameter rule | Group = one rule over many selected parameters.")
        with cB:
            group_mode = st.radio(
                "Group logic",
                ["ALL (AND)", "ANY (OR)"],
                horizontal=True,
                key="sql_group_logic",
                help="ALL = every selected parameter must match. ANY = at least one matches.",
            )
            st.caption("ALL = each selected parameter must pass | ANY = at least one passes.")
        with cC:
            st.caption(f"Matches: **{len(matches_all):,}**" + (" (showing first 500)" if len(matches_all) > 500 else ""))
    
        param_search = (search_q or "").strip().lower()
        shown_params = [pp for pp in all_params if param_search in pp.lower()] if param_search else all_params
        p = st.selectbox(
            "Parameter Name (single)",
            shown_params,
            key="sql_param_name",
            disabled=(select_mode != "Single parameter"),
        )
    
        st.markdown("<div class='sql-subhead'>✅ Group Selection (from search results)</div>", unsafe_allow_html=True)
        if select_mode != "Group from search results":
            st.info("Switch **Pick mode** to **Group from search results** to select many parameters at once.")
            selected_group = []
        else:
            if not matches:
                st.warning("No matches. Type a search above (e.g. `zone avg diameter`).")
                selected_group = []
            else:
                zone_filtered_matches = list(matches)
    
                if has_zone_matches:
                    zmin, zmax = min(zone_nums), max(zone_nums)
                    z1, z2 = st.slider(
                        "Zone range helper (only affects matches that contain “Zone <n>”)",
                        min_value=zmin,
                        max_value=zmax,
                        value=(zmin, zmax),
                        step=1,
                        key="sql_zone_range",
                    )
                    zone_keep = set([z for z in zone_nums if z1 <= z <= z2])
                    tmp = []
                    for nm in zone_filtered_matches:
                        zi = _extract_zone_num(nm)
                        if zi is None or zi in zone_keep:
                            tmp.append(nm)
                    zone_filtered_matches = tmp
    
                cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
                with cc1:
                    quick_avg = st.checkbox("Only Avg", value=False, key="sql_quick_avg")
                with cc2:
                    quick_min = st.checkbox("Only Min", value=False, key="sql_quick_min")
                with cc3:
                    quick_max = st.checkbox("Only Max", value=False, key="sql_quick_max")
                with cc4:
                    quick_diam = st.checkbox("Only ‘diameter’", value=False, key="sql_quick_diam")
    
                def _metric_filter_list(lst):
                    out = list(lst)
                    metric_flags = []
                    if quick_avg: metric_flags.append("avg")
                    if quick_min: metric_flags.append("min")
                    if quick_max: metric_flags.append("max")
                    if metric_flags:
                        out = [nm for nm in out if any(f in nm.lower() for f in metric_flags)]
                    if quick_diam:
                        out = [nm for nm in out if "diameter" in nm.lower()]
                    return out
    
                zone_filtered_matches = _metric_filter_list(zone_filtered_matches)
    
                csel1, csel2 = st.columns([1, 1])
                with csel1:
                    if st.button("✅ Select all (shown)", use_container_width=True, key="sql_sel_all_shown"):
                        st.session_state["sql_group_selected_params"] = list(zone_filtered_matches)
                        st.rerun()
                with csel2:
                    if st.button("🧼 Clear selection", use_container_width=True, key="sql_clear_group_sel"):
                        st.session_state["sql_group_selected_params"] = []
                        st.rerun()
                st.caption("Tip: search → select all shown → use 'Add group' in Step 3.")
    
                selected_group = st.multiselect(
                    "Selected parameters",
                    options=zone_filtered_matches,
                    default=[x for x in st.session_state.get("sql_group_selected_params", []) if x in zone_filtered_matches],
                    key="sql_group_selected_params",
                    help="Tip: search → Select all → add one condition for all zones.",
                )
    
    # ---- detect numeric vs categorical based on param_for_type ----
    is_param_numeric = False
    param_values = []
    param_for_type = p
    if select_mode == "Group from search results":
        gg = st.session_state.get("sql_group_selected_params", []) or []
        if gg:
            param_for_type = gg[0]
    
    try:
        df_param_sample = con.execute(f"""
            SELECT "Value"
            FROM datasets_kv
            WHERE "Parameter Name" = {_lit(param_for_type)}
            LIMIT 200
        """).fetchdf()
        if not df_param_sample.empty:
            sample_series = pd.to_numeric(df_param_sample["Value"], errors="coerce")
            is_param_numeric = sample_series.notna().sum() > 0
            if not is_param_numeric:
                df_opts = con.execute(f"""
                    SELECT DISTINCT trim("Value") AS val
                    FROM datasets_kv
                    WHERE "Parameter Name" = {_lit(param_for_type)}
                      AND trim(COALESCE("Value",'')) <> ''
                    ORDER BY val
                    LIMIT 200
                """).fetchdf()
                param_values = df_opts["val"].astype(str).tolist() if not df_opts.empty else []
    except Exception:
        pass
    
    with st.expander(_step_label("2) ⚙️ Condition", 2), expanded=(active_step == 2)):
        st.markdown(
            """
            <div class="sql-help">
              <b>Goal:</b> define the comparison rule for your selected parameter(s).<br>
              Tip: for text use <b>contains</b>; for ranges use <b>between</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_op, c_v1, c_v2 = st.columns([1.2, 2, 2])
    
        with c_op:
            op = st.selectbox(
                "Operator",
                ["any", "=", "!=", ">", ">=", "<", "<=", "between", "contains"],
                key="sql_op",
            )
    
        with c_v1:
            if (not is_param_numeric) and op not in ["any", "contains"] and param_values:
                v1 = st.selectbox("Value", options=[""] + param_values, key="sql_v1")
            else:
                v1 = st.text_input("Value", key="sql_v1")
    
        with c_v2:
            if op == "between":
                if (not is_param_numeric) and param_values:
                    v2 = st.selectbox("Second value (between)", options=[""] + param_values, key="sql_v2")
                else:
                    v2 = st.text_input("Second value (between)", key="sql_v2")
            else:
                v2 = st.text_input("Second value (between)", key="sql_v2")
    
        c_join, c_not = st.columns([1, 1])
        with c_join:
            joiner = st.radio("Join", ["AND", "OR"], horizontal=True, key="sql_joiner")
            st.caption("`AND` = must match all rules | `OR` = can match any rule.")
        with c_not:
            negate = st.checkbox("NOT", value=False, key="sql_negate")
            st.caption("`NOT` flips the selected rule (exclude matches).")
    
        st.markdown("<div class='sql-subhead'>🗓️ Time Filter (optional)</div>", unsafe_allow_html=True)
        time_on = st.checkbox("Enable time filter", value=False, key="sql_time_on")
        t1, t2 = st.columns(2)
        with t1:
            d_from = st.date_input("From", value=None, key="sql_time_from")
        with t2:
            d_to = st.date_input("To", value=None, key="sql_time_to")
    
        st.markdown("#### 🧩 Include")
        inc1, inc2, inc3 = st.columns(3)
        with inc1:
            include_draws = st.checkbox("Draws", value=True, key="sql_inc_draws")
        with inc2:
            include_maint = st.checkbox("Maintenance", value=False, key="sql_inc_maint")
        with inc3:
            include_faults = st.checkbox("Faults", value=False, key="sql_inc_faults")
    
        if not (include_draws or include_maint or include_faults):
            st.warning("Pick at least one: Draws / Maintenance / Faults.")
            st.stop()
    
    # =========================================================
    # Maintenance + Fault filters (collapsed by default)
    # =========================================================
    with st.expander(_step_label("🛠 Maintenance & Fault Filters (optional)", 2), expanded=(active_step == 2)):
        st.markdown(
            """
            <div class="sql-help">
              <b>Quick recipe:</b> enable one section → add text/component → keep scope as <b>matched draws window</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if include_maint or include_faults:
            st.markdown("#### ⏱️ Event scope")
            event_scope = st.radio(
                "How to constrain Maintenance/Faults relative to your draw filter?",
                [
                    "All events (respect only Maintenance/Fault filters)",
                    "Only within time filter window",
                    "Only within matched draws window",
                ],
                index=2,
                key="sql_event_scope",
                help="Matched draws window = min/max timestamp of the draws you matched (after timestamp fallback).",
            )
            st.markdown("---")
        else:
            event_scope = st.session_state.get("sql_event_scope", "Only within matched draws window")
    
        st.markdown("##### 🛠 Maintenance")
        maint_on = st.checkbox("Enable maintenance filter", value=False, key="sql_maint_on")
        maint_text = ""
        maint_component = ""
        if maint_on:
            m1, m2 = st.columns(2)
            with m1:
                maint_text = st.text_input(
                    "Maintenance text contains",
                    key="sql_maint_text",
                    placeholder="task / note / source_file",
                )
            with m2:
                comps = []
                try:
                    comps = (
                        con.execute("""
                            SELECT DISTINCT component
                            FROM maintenance_tasks
                            WHERE component IS NOT NULL AND TRIM(component) <> ''
                            ORDER BY component
                        """)
                        .fetchdf()["component"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    comps = []
    
                if not comps:
                    try:
                        comps = (
                            con.execute("""
                                SELECT DISTINCT component
                                FROM maintenance_actions
                                WHERE component IS NOT NULL AND TRIM(component) <> ''
                                ORDER BY component
                            """)
                            .fetchdf()["component"]
                            .astype(str)
                            .tolist()
                        )
                    except Exception:
                        comps = []
    
                pick = st.selectbox(
                    "Maintenance component",
                    options=["All"] + comps + ["Custom contains…"],
                    key="sql_maint_comp_pick",
                )
                if pick == "All":
                    maint_component = ""
                elif pick == "Custom contains…":
                    maint_component = st.text_input(
                        "Maintenance component contains",
                        key="sql_maint_comp",
                        placeholder="type part of component name…",
                    )
                else:
                    maint_component = pick
    
        st.markdown("---")
    
        st.markdown("##### 🚨 Faults")
        fault_on = st.checkbox("Enable faults filter", value=False, key="sql_fault_on")
        fault_text = ""
        fault_component = ""
        fault_sev = ""
        if fault_on:
            f1, f2, f3 = st.columns([1.2, 1.2, 1])
            with f1:
                fault_text = st.text_input(
                    "Fault text contains",
                    key="sql_fault_text",
                    placeholder="title / description / source_file",
                )
            with f2:
                fault_comps = []
                try:
                    fault_comps = (
                        con.execute("""
                            SELECT DISTINCT component
                            FROM faults_events
                            WHERE component IS NOT NULL AND TRIM(component) <> ''
                            ORDER BY component
                        """)
                        .fetchdf()["component"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    fault_comps = []
    
                maint_comps = []
                try:
                    maint_comps = (
                        con.execute("""
                            SELECT DISTINCT component
                            FROM maintenance_tasks
                            WHERE component IS NOT NULL AND TRIM(component) <> ''
                            ORDER BY component
                        """)
                        .fetchdf()["component"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    maint_comps = []
    
                comp_pool = sorted(set([c for c in (fault_comps + maint_comps) if str(c).strip()]))
    
                pick = st.selectbox(
                    "Fault component",
                    options=["All"] + comp_pool + ["Custom contains…"],
                    key="sql_fault_comp_pick",
                )
                if pick == "All":
                    fault_component = ""
                elif pick == "Custom contains…":
                    fault_component = st.text_input(
                        "Fault component contains",
                        key="sql_fault_comp",
                        placeholder="type part of component name…",
                    )
                else:
                    fault_component = pick
            with f3:
                fault_sev = st.selectbox(
                    "Severity",
                    ["", "low", "medium", "high", "critical"],
                    index=0,
                    key="sql_fault_sev",
                )
    
    # =========================================================
    # Condition SQL builder (against draws)
    # =========================================================
    def build_cond_sql(p, op, v1, v2):
        p = (p or "").strip()
        v1 = (v1 or "").strip()
        v2 = (v2 or "").strip()
        if not p:
            return None
    
        base = f'kv."Parameter Name" = {_lit(p)} AND kv._draw = d._draw'
    
        if op == "any":
            return f"EXISTS (SELECT 1 FROM datasets_kv kv WHERE {base})"
    
        if op == "contains":
            if not v1:
                return None
            return (
                "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                f"{base} AND CAST(kv.\"Value\" AS VARCHAR) ILIKE '%{_esc(v1)}%')"
            )
    
        if op == "between":
            if not v1 or not v2:
                return None
            if _is_num(v1) and _is_num(v2):
                return (
                    "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                    f"{base} AND TRY_CAST(kv.\"Value\" AS DOUBLE) BETWEEN {v1} AND {v2})"
                )
            return (
                "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                f"{base} AND kv.\"Value\" BETWEEN {_lit(v1)} AND {_lit(v2)})"
            )
    
        if not v1:
            return None
    
        if _is_num(v1):
            return (
                "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                f"{base} AND TRY_CAST(kv.\"Value\" AS DOUBLE) {op} {v1})"
            )
    
        return (
            "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
            f"{base} AND kv.\"Value\" {op} {_lit(v1)})"
        )
    
    def build_cond_human(p, op, v1, v2):
        v1s = (v1 or "").strip()
        v2s = (v2 or "").strip()
        if op == "any":
            return f"{p}: is present"
        if op == "contains":
            if not v1s:
                return None
            return f"{p}: contains “{v1s}”"
        if op == "between":
            if not v1s or not v2s:
                return None
            return f"{p}: between {v1s} and {v2s}"
        if not v1s:
            return None
        op_map = {"=": "=", "!=": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤"}
        return f"{p}: {op_map.get(op, op)} {v1s}"
    
    def wrap_not(sql, human, negate):
        if not sql or not human:
            return None, None
        if negate:
            return f"(NOT ({sql}))", f"NOT {human}"
        return f"({sql})", human
    
    def build_group_cond_sql(params, op, v1, v2, group_logic):
        params = _dedupe_keep_order(params or [])
        if not params:
            return None
        parts = []
        for pp in params:
            s = build_cond_sql(pp, op, v1, v2)
            if s:
                parts.append(f"({s})")
        if not parts:
            return None
        glue = " AND " if str(group_logic).startswith("ALL") else " OR "
        return "(" + glue.join(parts) + ")"
    
    def build_group_cond_human(params, op, v1, v2, group_logic):
        params = _dedupe_keep_order(params or [])
        if not params:
            return None
        preview = ", ".join(params[:5]) + (" …" if len(params) > 5 else "")
        v1s = (v1 or "").strip()
        v2s = (v2 or "").strip()
    
        if op == "any":
            cond = "is present"
        elif op == "contains":
            if not v1s:
                return None
            cond = f"contains “{v1s}”"
        elif op == "between":
            if not v1s or not v2s:
                return None
            cond = f"between {v1s} and {v2s}"
        else:
            if not v1s:
                return None
            op_map = {"=": "=", "!=": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤"}
            cond = f"{op_map.get(op, op)} {v1s}"
    
        return f"{group_logic}: {len(params)} params [{preview}] → {cond}"
    
    def _kv_predicate_sql(op, v1, v2):
        v1 = (v1 or "").strip()
        v2 = (v2 or "").strip()
    
        if op == "any":
            return "TRUE"
    
        if op == "contains":
            if not v1:
                return None
            return f"CAST(kv.\"Value\" AS VARCHAR) ILIKE '%{_esc(v1)}%'"
    
        if op == "between":
            if not v1 or not v2:
                return None
            if _is_num(v1) and _is_num(v2):
                return f"TRY_CAST(kv.\"Value\" AS DOUBLE) BETWEEN {v1} AND {v2}"
            return f"kv.\"Value\" BETWEEN {_lit(v1)} AND {_lit(v2)}"
    
        if not v1:
            return None
    
        if _is_num(v1):
            return f"TRY_CAST(kv.\"Value\" AS DOUBLE) {op} {v1}"
    
        return f"kv.\"Value\" {op} {_lit(v1)}"
    
    # =========================================================
    # Add/remove conditions (UI)
    # =========================================================
    with st.expander(_step_label("3) ➕ Build filter", 3), expanded=(active_step == 3)):
        st.markdown(
            """
            <div class="sql-help">
              <b>Build your query:</b> add one or more conditions, then review the active filter summary.
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    
        if b1.button("➕ Add condition", use_container_width=True, key="sql_add_cond"):
            sql_raw = build_cond_sql(p, op, v1, v2)
            human_raw = build_cond_human(p, op, v1, v2)
            sql_cond, human_cond = wrap_not(sql_raw, human_raw, negate)
    
            if not sql_cond or not human_cond:
                st.warning("Condition not complete.")
            else:
                if st.session_state.ds_conditions:
                    st.session_state.ds_conditions.append(f"{joiner} {sql_cond}")
                else:
                    st.session_state.ds_conditions.append(sql_cond)
    
                if st.session_state.ds_conditions_human:
                    st.session_state.ds_conditions_human.append(f"{joiner} {human_cond}")
                else:
                    st.session_state.ds_conditions_human.append(human_cond)
    
                st.session_state.sql_filter_params_seq.append(p)
    
                st.session_state.ds_conditions_struct.append({
                    "params": [p],
                    "op": op,
                    "v1": v1,
                    "v2": v2,
                    "negate": bool(negate),
                })
    
        if b2.button("🧩 Add group (from search)", use_container_width=True, key="sql_add_group"):
            params_group = st.session_state.get("sql_group_selected_params", []) if select_mode == "Group from search results" else []
            sql_raw = build_group_cond_sql(params_group, op, v1, v2, st.session_state.get("sql_group_logic", "ALL (AND)"))
            human_raw = build_group_cond_human(params_group, op, v1, v2, st.session_state.get("sql_group_logic", "ALL (AND)"))
            sql_cond, human_cond = wrap_not(sql_raw, human_raw, negate)
    
            if not sql_cond or not human_cond:
                st.warning("Group condition not complete (select params + set values).")
            else:
                if st.session_state.ds_conditions:
                    st.session_state.ds_conditions.append(f"{joiner} {sql_cond}")
                else:
                    st.session_state.ds_conditions.append(sql_cond)
    
                if st.session_state.ds_conditions_human:
                    st.session_state.ds_conditions_human.append(f"{joiner} {human_cond}")
                else:
                    st.session_state.ds_conditions_human.append(human_cond)
    
                for pp in (_dedupe_keep_order(params_group) or []):
                    st.session_state.sql_filter_params_seq.append(pp)
    
                st.session_state.ds_conditions_struct.append({
                    "params": _dedupe_keep_order(params_group),
                    "op": op,
                    "v1": v1,
                    "v2": v2,
                    "negate": bool(negate),
                    "group_logic": st.session_state.get("sql_group_logic", "ALL (AND)"),
                })
    
        if b3.button("↩ Remove last", use_container_width=True, key="sql_pop_cond"):
            if st.session_state.ds_conditions:
                st.session_state.ds_conditions.pop()
            if st.session_state.ds_conditions_human:
                st.session_state.ds_conditions_human.pop()
            if st.session_state.sql_filter_params_seq:
                st.session_state.sql_filter_params_seq.pop()
            if st.session_state.get("ds_conditions_struct"):
                st.session_state.ds_conditions_struct.pop()
    
        if b4.button("🧹 Clear", use_container_width=True, key="sql_clear_cond"):
            st.session_state.ds_conditions = []
            st.session_state.ds_conditions_human = []
            st.session_state.ds_conditions_struct = []
            st.session_state.sql_filter_params_seq = []
            st.session_state.sql_group_defs_for_plot = []
            st.session_state.sql_matched_params_only = []
            st.session_state.sql_values_found_long = pd.DataFrame()
            st.session_state.sql_values_found_wide = pd.DataFrame()
            st.session_state.sql_plot_params = []
    
        human_lines = list(st.session_state.ds_conditions_human)
        if time_on and d_from and d_to:
            human_lines.append(f"Time: {d_from} → {d_to}")
        if maint_on and (maint_text.strip() or maint_component.strip()):
            human_lines.append(f"Maintenance: {maint_component.strip()} {maint_text.strip()}".strip())
        if fault_on and (fault_text.strip() or fault_component.strip() or fault_sev.strip()):
            human_lines.append(f"Faults: {fault_component.strip()} {fault_text.strip()} {fault_sev.strip()}".strip())
    
        if human_lines:
            st.success("**Active filter:**\n" + "\n".join([f"- {x}" for x in human_lines]))
        else:
            st.info("No filter set (will include all selected event types).")
    
        st.session_state["sql_last_human_lines"] = list(human_lines)
    
    # =========================================================
    # Build draw WHERE (conditions only; time applied after fallback)
    # =========================================================
    where_sql_draws = ""
    if st.session_state.ds_conditions:
        conds = list(st.session_state.ds_conditions)
        if conds:
            first = str(conds[0]).lstrip()
            if first.upper().startswith("OR "):
                conds[0] = first[3:].lstrip()
            elif first.upper().startswith("AND "):
                conds[0] = first[4:].lstrip()
        where_sql_draws = "WHERE " + "\n  ".join(conds)
    
    # =========================================================
    # Build maint/fault WHERE (text/component/sev only; scope applied later)
    # =========================================================
    maint_where_base = ""
    if maint_on and (maint_text.strip() or maint_component.strip()):
        conds = []
        if maint_text.strip():
            s = maint_text.strip()
            conds.append(
                f"(COALESCE(task,'') ILIKE '%{_esc(s)}%' OR COALESCE(note,'') ILIKE '%{_esc(s)}%' OR COALESCE(source_file,'') ILIKE '%{_esc(s)}%')"
            )
        if maint_component.strip():
            s2 = maint_component.strip()
            if st.session_state.get("sql_maint_comp_pick", "") not in ("All", "Custom contains…"):
                conds.append(f"(COALESCE(component,'') = {_lit(s2)})")
            else:
                conds.append(f"(COALESCE(component,'') ILIKE '%{_esc(s2)}%')")
        maint_where_base = "WHERE " + " AND ".join(conds)
    
    fault_where_base = ""
    if fault_on and (fault_text.strip() or fault_component.strip() or fault_sev.strip()):
        conds = []
        if fault_text.strip():
            s = fault_text.strip()
            conds.append(
                f"(COALESCE(title,'') ILIKE '%{_esc(s)}%' OR COALESCE(description,'') ILIKE '%{_esc(s)}%' OR COALESCE(source_file,'') ILIKE '%{_esc(s)}%')"
            )
        if fault_component.strip():
            s2 = fault_component.strip()
            if st.session_state.get("sql_fault_comp_pick", "") not in ("All", "Custom contains…"):
                conds.append(f"(COALESCE(component,'') = {_lit(s2)})")
            else:
                conds.append(f"(COALESCE(component,'') ILIKE '%{_esc(s2)}%')")
        if fault_sev.strip():
            conds.append(f"(COALESCE(severity,'') = {_lit(fault_sev.strip())})")
        fault_where_base = "WHERE " + " AND ".join(conds)
    
    # =========================================================
    # RUN FILTER
    # =========================================================
    st.markdown('<div class="sql-section">▶ Run</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sql-help">
          <b>Run filter</b> executes your active conditions and loads matching Draws / Maintenance / Faults.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    sql_draws = f"""
    WITH draws AS (
        SELECT _draw, MAX(event_ts) AS event_ts
        FROM datasets_kv
        GROUP BY _draw
    )
    SELECT d._draw, d.event_ts
    FROM draws d
    {where_sql_draws}
    ORDER BY COALESCE(d.event_ts, TIMESTAMP '1900-01-01') ASC, d._draw;
    """
    
    if st.button("▶ Run filter", type="primary", use_container_width=True, key="sql_run"):
        used_params_run = _dedupe_keep_order(st.session_state.get("sql_filter_params_seq", []))
        human_lines_run = list(st.session_state.get("sql_last_human_lines", []) or [])
        filters_summary_run = _filters_summary_for_draws(used_params_run, human_lines_run)
        st.session_state["sql_last_filters_summary"] = filters_summary_run
    
        # ---- draws ----
        df_draws = pd.DataFrame(columns=["_draw", "event_ts"])
        if include_draws:
            df_draws = con.execute(sql_draws).fetchdf()
    
            # Fill missing event_ts using file mtime
            if not df_draws.empty:
                draw_list = df_draws["_draw"].dropna().astype(str).unique().tolist()
                draws_sql = "(" + ",".join(_lit(d) for d in draw_list) + ")"
                df_files = con.execute(f"""
                    SELECT _draw, ANY_VALUE(filename) AS filename
                    FROM datasets_kv
                    WHERE CAST(_draw AS VARCHAR) IN {draws_sql}
                    GROUP BY _draw
                """).fetchdf()
                if not df_files.empty:
                    df_draws = df_draws.merge(df_files, on="_draw", how="left")
                    df_draws["event_ts"] = pd.to_datetime(df_draws["event_ts"], errors="coerce")
                    df_draws["event_ts"] = df_draws["event_ts"].fillna(df_draws["filename"].astype(str).apply(_mtime_ts))
                    df_draws = df_draws.drop(columns=["filename"], errors="ignore")
    
            # Apply time filter AFTER fallback
            if time_on and d_from and d_to and not df_draws.empty:
                start_ts = pd.Timestamp(d_from)
                end_ts = pd.Timestamp(d_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df_draws["event_ts"] = pd.to_datetime(df_draws["event_ts"], errors="coerce")
                df_draws = df_draws[df_draws["event_ts"].between(start_ts, end_ts)]
    
            if not df_draws.empty:
                df_draws = df_draws.sort_values(["event_ts", "_draw"], na_position="last")
    
        # ---- supporting “values found” ----
        values_long = pd.DataFrame()
        values_wide = pd.DataFrame()
        matched_params_only = []
        group_defs = []
    
        if df_draws is not None and not df_draws.empty:
            df_draws = df_draws.copy()
            df_draws["matched_by"] = filters_summary_run if filters_summary_run else (
                ", ".join(used_params_run) if used_params_run else "(no draw filter)"
            )
    
            draw_list = df_draws["_draw"].dropna().astype(str).unique().tolist()
            draws_sql = "(" + ",".join(_lit(d) for d in draw_list) + ")"
    
            match_terms = []
            for cond in (st.session_state.get("ds_conditions_struct") or []):
                pred = _kv_predicate_sql(cond.get("op"), cond.get("v1"), cond.get("v2"))
                if not pred:
                    continue
                if cond.get("negate"):
                    pred = f"NOT ({pred})"
                for pp in (cond.get("params") or []):
                    match_terms.append(f'(kv."Parameter Name" = {_lit(pp)} AND ({pred}))')
    
            if match_terms:
                match_where = " OR ".join(match_terms)
                values_long = con.execute(f"""
                    SELECT
                        CAST(kv._draw AS VARCHAR) AS _draw,
                        CAST(kv."Parameter Name" AS VARCHAR) AS "Parameter Name",
                        CAST(kv."Value" AS VARCHAR) AS "Value",
                        CAST(COALESCE(kv."Units",'') AS VARCHAR) AS "Units"
                    FROM datasets_kv kv
                    WHERE CAST(kv._draw AS VARCHAR) IN {draws_sql}
                      AND TRIM(COALESCE(CAST(kv."Value" AS VARCHAR),'')) <> ''
                      AND ({match_where})
                """).fetchdf()
    
            if values_long is not None and not values_long.empty:
                matched_params_only = _dedupe_keep_order(values_long["Parameter Name"].astype(str).tolist())
    
                values_long = (
                    values_long.drop_duplicates(subset=["_draw", "Parameter Name"], keep="first")
                    .sort_values(["_draw", "Parameter Name"])
                    .reset_index(drop=True)
                )
    
                values_wide = (
                    values_long.pivot_table(
                        index="_draw",
                        columns="Parameter Name",
                        values="Value",
                        aggfunc="first",
                    )
                    .reset_index()
                )
    
                join_cols = [c for c in values_wide.columns if c != "_draw"][:12]
                if join_cols:
                    df_draws = df_draws.merge(values_wide[["_draw"] + join_cols], on="_draw", how="left")
    
            # group defs for plotting
            for i, cond in enumerate(st.session_state.get("ds_conditions_struct") or []):
                params = _dedupe_keep_order(cond.get("params") or [])
                if len(params) < 2:
                    continue
    
                op_i = str(cond.get("op", "any"))
                v1_i = str(cond.get("v1", "")).strip()
                v2_i = str(cond.get("v2", "")).strip()
                if op_i == "any":
                    rhs = "present"
                elif op_i == "contains":
                    rhs = f"contains '{v1_i}'" if v1_i else "contains"
                elif op_i == "between":
                    rhs = f"between {v1_i}..{v2_i}" if (v1_i and v2_i) else "between"
                else:
                    rhs = f"{op_i} {v1_i}" if v1_i else op_i
    
                gl = str(cond.get("group_logic", st.session_state.get("sql_group_logic", "ALL (AND)")))
                neg = " NOT" if bool(cond.get("negate")) else ""
                label = f"🧩 Group {i+1}: {gl}{neg} ({len(params)} params) → {rhs}"
    
                group_defs.append({"label": label, "params": params})
    
        # ---- event scope window (for maint/fault overlay) ----
        scope_start = None
        scope_end = None
    
        if (include_maint or include_faults) and event_scope == "Only within time filter window" and time_on and d_from and d_to:
            scope_start = pd.Timestamp(d_from)
            scope_end = pd.Timestamp(d_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
        if (include_maint or include_faults) and event_scope == "Only within matched draws window" and include_draws and df_draws is not None and not df_draws.empty:
            tmin = pd.to_datetime(df_draws["event_ts"], errors="coerce").min()
            tmax = pd.to_datetime(df_draws["event_ts"], errors="coerce").max()
            if pd.notna(tmin) and pd.notna(tmax):
                scope_start = tmin
                scope_end = tmax
    
        def _add_scope(where_base: str, ts_col: str):
            if scope_start is None or scope_end is None:
                return where_base
            add = f"({ts_col} BETWEEN {_lit(scope_start)} AND {_lit(scope_end)})"
            return (where_base + " AND " if where_base else "WHERE ") + add
    
        maint_where = _add_scope(maint_where_base, "action_ts")
        fault_where = _add_scope(fault_where_base, "fault_ts")
    
        # ---- Load draw KV rows for matched draws ----
        df_kv = pd.DataFrame()
        if include_draws and df_draws is not None and not df_draws.empty:
            draw_list = df_draws["_draw"].dropna().astype(str).unique().tolist()
            draws_sql = "(" + ",".join(_lit(d) for d in draw_list) + ")"
            df_kv = con.execute(f"""
                SELECT
                    'dataset'::VARCHAR AS source_kind,
                    NULL::VARCHAR AS source_file,
                    event_ts,
                    event_id,
                    _draw,
                    filename,
                    "Parameter Name",
                    "Value",
                    "Units"
                FROM datasets_kv
                WHERE CAST(_draw AS VARCHAR) IN {draws_sql}
            """).fetchdf()
    
        # ---- Maintenance events ----
        df_m = pd.DataFrame()
        if include_maint:
            df_m = con.execute(f"""
                SELECT
                    'maintenance'::VARCHAR AS source_kind,
                    COALESCE(source_file,'')::VARCHAR AS source_file,
                    action_ts AS event_ts,
                    CAST(action_id AS VARCHAR) AS event_id,
                    NULL::VARCHAR AS _draw,
                    NULL::VARCHAR AS filename,
                    COALESCE(component,'')::VARCHAR AS "Parameter Name",
                    (trim(COALESCE(task,'')) || CASE WHEN COALESCE(note,'') <> '' THEN (' | ' || note) ELSE '' END)::VARCHAR AS "Value",
                    ''::VARCHAR AS "Units"
                FROM maintenance_actions
                {maint_where}
            """).fetchdf()
    
        # ---- Fault events ----
        df_f = pd.DataFrame()
        if include_faults:
            df_f = con.execute(f"""
                SELECT
                    'fault'::VARCHAR AS source_kind,
                    COALESCE(source_file,'')::VARCHAR AS source_file,
                    fault_ts AS event_ts,
                    CAST(fault_id AS VARCHAR) AS event_id,
                    NULL::VARCHAR AS _draw,
                    NULL::VARCHAR AS filename,
                    COALESCE(component,'')::VARCHAR AS "Parameter Name",
                    (trim(COALESCE(title,'')) || CASE WHEN COALESCE(severity,'') <> '' THEN (' | severity=' || severity) ELSE '' END ||
                     CASE WHEN COALESCE(description,'') <> '' THEN (' | ' || description) ELSE '' END)::VARCHAR AS "Value",
                    ''::VARCHAR AS "Units"
                FROM faults_events
                {fault_where}
            """).fetchdf()
    
        df_all = pd.concat([df_kv, df_m, df_f], ignore_index=True)
    
        st.session_state["sql_df_all"] = df_all
        st.session_state["sql_matched_draws"] = df_draws
        st.session_state["sql_values_found_long"] = values_long if values_long is not None else pd.DataFrame()
        st.session_state["sql_values_found_wide"] = values_wide if values_wide is not None else pd.DataFrame()
        st.session_state["sql_matched_params_only"] = matched_params_only
        st.session_state["sql_group_defs_for_plot"] = group_defs
    
        st.session_state.pop("sql_selected_event_key", None)
        st.session_state.pop("math_selected_event_key", None)
    
        with st.expander("✅ Results summary", expanded=True):
            if include_draws:
                md = st.session_state.get("sql_matched_draws", pd.DataFrame())
                st.success(f"Matched draws: {0 if md is None else len(md):,}")
    
                fs = st.session_state.get("sql_last_filters_summary", "")
                if fs:
                    show_filters = st.checkbox(
                        "🔎 Show filters used to match these draws",
                        value=False,
                        key="sql_show_filters_summary",
                    )
                    if show_filters:
                        st.write(fs)
    
                show_values = st.checkbox(
                    "📌 Show values that actually matched your filter",
                    value=True,
                    key="sql_show_matched_values",
                )
                if show_values:
                    values_long2 = st.session_state.get("sql_values_found_long", pd.DataFrame())
                    values_wide2 = st.session_state.get("sql_values_found_wide", pd.DataFrame())
                    if values_long2 is None or values_long2.empty:
                        st.info("No supporting matched values found (maybe NOT-only logic or 'any' conditions).")
                    else:
                        st.caption("Long view (only rows that satisfied your conditions):")
                        st.dataframe(values_long2, use_container_width=True, height=260)
                        if values_wide2 is not None and not values_wide2.empty:
                            st.caption("Wide view (one row per draw):")
                            st.dataframe(values_wide2, use_container_width=True, height=260)
    
                st.caption("Matched draws table:")
                st.dataframe(md, use_container_width=True, height=260)
            else:
                st.info("Draws excluded (timeline can show only Maintenance/Faults).")
    
    # =========================================================
    # Guards
    # =========================================================
    if "sql_df_all" not in st.session_state:
        st.stop()
    
    df_all = st.session_state["sql_df_all"]
    if df_all is None or df_all.empty:
        st.warning("No results loaded. Run filter.")
        st.stop()
    
    # =========================================================
    # Event details
    # =========================================================
    def render_event_details(event_key: str):
        if not event_key or ":" not in event_key:
            return
        kind, eid = event_key.split(":", 1)
        kind = (kind or "").strip()
    
        st.markdown("### 🔎 Event details")
        st.caption(f"Selected: **{event_key}**")
    
        if kind == "dataset":
            df_kv2 = con.execute(f"""
                SELECT
                    event_ts,
                    event_id,
                    _draw,
                    filename,
                    "Parameter Name",
                    "Value",
                    "Units"
                FROM datasets_kv
                WHERE CAST(_draw AS VARCHAR) = {_lit(eid)}
                   OR CAST(event_id AS VARCHAR) = {_lit(eid)}
                ORDER BY "Parameter Name"
            """).fetchdf()
            if df_kv2.empty:
                st.warning("No KV rows found for this draw.")
                return
            top = df_kv2.head(1)
            st.markdown(f"**Draw:** `{top['event_id'].iloc[0]}`")
            if "event_ts" in top.columns:
                st.caption(f"Time: {top['event_ts'].iloc[0]}")
            st.dataframe(df_kv2[["Parameter Name", "Value", "Units"]], use_container_width=True, height=460)
    
        elif kind == "maintenance":
            df_act = con.execute(f"""
                SELECT
                    action_ts,
                    component,
                    task,
                    task_id,
                    tracking_mode,
                    hours_source,
                    done_date,
                    done_hours,
                    done_draw,
                    actor,
                    note,
                    source_file
                FROM maintenance_actions
                WHERE CAST(action_id AS VARCHAR) = {_lit(eid)}
                LIMIT 1
            """).fetchdf()
            if df_act.empty:
                st.warning("Maintenance action not found.")
                return
            st.dataframe(df_act, use_container_width=True, height=180)
    
        elif kind == "fault":
            df_fault = con.execute(f"""
                SELECT
                    fault_ts,
                    component,
                    severity,
                    title,
                    description,
                    actor,
                    source_file,
                    related_draw
                FROM faults_events
                WHERE CAST(fault_id AS VARCHAR) = {_lit(eid)}
                LIMIT 1
            """).fetchdf()
            if df_fault.empty:
                st.warning("Fault not found.")
                return
            st.dataframe(df_fault, use_container_width=True, height=180)
        else:
            st.info("Unknown event type.")
    
    # =========================================================
    # VISUAL LAB  (FIXED: main plot always shows numeric/group;
    # text is a categorical Y axis on main plot; maint/fault only below)
    # =========================================================
    # =========================================================
    # VISUAL LAB  (Main plot: numeric/group + auto text categorical axes;
    # Events lane: maintenance+fault only)
    # =========================================================
    # =========================================================
    # VISUAL LAB  (Main plot: numeric/group + auto text categorical axes;
    # Events lane: maintenance+fault only)
    # =========================================================
    st.subheader("📈 Visual Lab")
    
    df = df_all.copy()
    df["event_ts"] = pd.to_datetime(df.get("event_ts"), errors="coerce")
    
    ds_kv = df[df["source_kind"].astype(str).eq("dataset")].copy() if "source_kind" in df.columns else pd.DataFrame()
    maint_kv = df[
        df["source_kind"].astype(str).eq("maintenance")].copy() if "source_kind" in df.columns else pd.DataFrame()
    fault_kv = df[df["source_kind"].astype(str).eq("fault")].copy() if "source_kind" in df.columns else pd.DataFrame()
    
    show_draw_traces = st.toggle(
        "Show Draw traces",
        value=True,
        key="sql_show_draw_traces",
        help="Turn OFF if you want only Maintenance/Faults.",
    )
    
    wide = pd.DataFrame()
    numeric_all = []
    if not ds_kv.empty:
        if "filename" in ds_kv.columns:
            ds_kv["event_ts"] = ds_kv["event_ts"].fillna(ds_kv["filename"].astype(str).apply(_mtime_ts))
    
        ds_kv = ds_kv[ds_kv["event_ts"].notna()].copy()
        ds_kv["event_key"] = "dataset:" + ds_kv["event_id"].astype(str)
    
        wide = (
            ds_kv.pivot_table(
                index=["event_ts", "event_key"],
                columns="Parameter Name",
                values="Value",
                aggfunc="first",
            )
            .reset_index()
            .sort_values("event_ts")
        )
    
        META = {"event_ts", "event_key"}
        all_plot_params = [c for c in wide.columns if c not in META]
        numeric_all = [c for c in all_plot_params if pd.to_numeric(wide[c], errors="coerce").notna().sum() > 0]
    
    # ------------------- Plot picker -------------------
    numeric_chosen, cat_chosen, group_chosen, chosen_all = [], [], [], []
    
    if wide is not None and not wide.empty:
        matched_params_only = st.session_state.get("sql_matched_params_only", []) or []
    
        st.markdown("#### 🎛 Plot helper")
    
        pick_mode = st.radio(
            "Parameter picker",
            [
                "Only parameters used in the filter",
                "Only parameters that actually matched (supporting rows)",
                "Any parameter from matched draws",
            ],
            horizontal=True,
            key="sql_pick_mode",
        )
    
        used_params2 = _dedupe_keep_order(st.session_state.get("sql_filter_params_seq", []))
        if pick_mode.startswith("Only parameters used"):
            pool = [pp for pp in used_params2 if pp in wide.columns]
        elif pick_mode.startswith("Only parameters that actually matched"):
            pool = [pp for pp in matched_params_only if pp in wide.columns]
        else:
            pool = [c for c in wide.columns if c not in ("event_ts", "event_key")]
    
        group_defs = st.session_state.get("sql_group_defs_for_plot", []) or []
        group_labels = [g.get("label") for g in group_defs if str(g.get("label", "")).strip()]
        pool = list(group_labels) + [pp for pp in pool if pp not in group_labels]
    
        prev = [x for x in (st.session_state.get("sql_plot_params") or []) if x in pool]
        if prev != (st.session_state.get("sql_plot_params") or []):
            st.session_state["sql_plot_params"] = prev
    
        cbtn1, cbtn2 = st.columns([1, 1])
        with cbtn1:
            if st.button("🎯 Set plot = all matched params", use_container_width=True, key="sql_apply_plot_matched"):
                st.session_state["sql_plot_params"] = [pp for pp in matched_params_only if pp in pool]
                st.rerun()
        with cbtn2:
            if st.button("🧹 Clear plot selection", use_container_width=True, key="sql_clear_plot_sel"):
                st.session_state["sql_plot_params"] = []
                st.rerun()
    
        chosen_all = st.multiselect(
            "Draw parameters to plot",
            pool,
            key="sql_plot_params",
        )
    
        group_label_set = set(group_labels)
        group_chosen = [x for x in (chosen_all or []) if x in group_label_set]
        normal_chosen = [x for x in (chosen_all or []) if x not in group_label_set]
    
        for c in (normal_chosen or []):
            if c in ("event_ts", "event_key"):
                continue
            s_num = pd.to_numeric(wide[c], errors="coerce")
            if s_num.notna().sum() > 0:
                numeric_chosen.append(c)
            else:
                # Treat as text/categorical if it has any non-empty string values
                if wide[c].astype(str).replace("nan", "").str.strip().ne("").any():
                    cat_chosen.append(c)
    else:
        st.info("No draw timestamps available. Maintenance/Faults can still show.")
    
    # =========================================================
    # Plot: row1 main draw plot (MULTI-Y + auto TEXT categorical axes),
    # row2 events lane (maintenance+fault only)
    # =========================================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.06,
    )
    
    palette = list(getattr(qualitative, "Plotly", [])) or [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    
    st.markdown("#### ⚙️ Main plot axes")
    
    # One axis per requested numeric/group parameter (no sharing)
    total_requested = len(numeric_chosen) + len(group_chosen)
    numeric_axis_count = max(1, int(total_requested))
    
    st.caption(
        f"Numeric/group Y-axes: **{numeric_axis_count}** (one per selected numeric/group parameter). "
        "Text parameters automatically get their own categorical Y-axes."
    )
    
    
    # IMPORTANT:
    # make_subplots(rows=2, cols=1) uses:
    #   row 1 -> y  (layout.yaxis)
    #   row 2 -> y2 (layout.yaxis2)
    # We must NOT use y2 for the main plot. So main-plot overlays start at y3.
    
    def _yaxis_id(i: int) -> str:
        # i is 0-based axis index for MAIN PLOT axes.
        # make_subplots uses y for row1 and y2 for row2, so we skip y2.
        return "y" if i == 0 else f"y{i + 2}"  # y, y3, y4... (skip y2)
    
    
    def _yaxis_layout_name(i: int) -> str:
        return "yaxis" if i == 0 else f"yaxis{i + 2}"  # yaxis, yaxis3, yaxis4... (skip yaxis2)
    
    
    axis_owner = {}  # numeric axis index -> {name,color}
    
    # ---- MAIN PLOT traces ----
    if show_draw_traces and wide is not None and not wide.empty:
    
        # ---------- numeric traces ----------
        for i, col in enumerate(list(numeric_chosen or [])):
            y = pd.to_numeric(wide[col], errors="coerce")
            if y.notna().sum() == 0:
                continue
    
            colr = palette[i % len(palette)]
            ax_i = i  # one axis per numeric parameter
            yaxis_id = _yaxis_id(ax_i)
    
            if ax_i not in axis_owner:
                axis_owner[ax_i] = {"name": col, "color": colr}
    
            fig.add_trace(
                go.Scatter(
                    x=wide["event_ts"],
                    y=y,
                    mode="lines+markers",
                    name=col,
                    customdata=wide["event_key"],
                    hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>",
                    line=dict(color=colr),
                    marker=dict(color=colr),
                ),
                row=1, col=1
            )
            # Plotly's make_subplots can override trace axes; force it AFTER adding
            fig.data[-1].update(xaxis="x", yaxis=yaxis_id)
    
        # ---------- group traces (mean) ----------
        if group_chosen:
            by_label = {
                g.get("label"): g
                for g in (st.session_state.get("sql_group_defs_for_plot", []) or [])
                if g.get("label")
            }
            base_idx = len(list(numeric_chosen or []))
            for j, glabel in enumerate(group_chosen):
                g = by_label.get(glabel)
                if not g:
                    continue
                params = [p for p in (g.get("params") or []) if p in wide.columns]
                if not params:
                    continue
    
                y_stack = pd.concat([pd.to_numeric(wide[p], errors="coerce") for p in params], axis=1)
                y_mean = y_stack.mean(axis=1, skipna=True)
                if y_mean.notna().sum() == 0:
                    continue
    
                idx = base_idx + j
                colr = palette[idx % len(palette)]
                ax_i = idx  # one axis per group parameter
                yaxis_id = _yaxis_id(ax_i)
    
                if ax_i not in axis_owner:
                    axis_owner[ax_i] = {"name": glabel, "color": colr}
    
                fig.add_trace(
                    go.Scatter(
                        x=wide["event_ts"],
                        y=y_mean,
                        mode="lines+markers",
                        name=glabel,
                        customdata=wide["event_key"],
                        hovertemplate=f"<b>{glabel}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>",
                        line=dict(color=colr, width=3),
                        marker=dict(color=colr),
                    ),
                    row=1, col=1
                )
                fig.data[-1].update(xaxis="x", yaxis=yaxis_id)
    
        # ---------- auto TEXT categorical axes ----------
        # If user selected any categorical params in the multiselect, render each as its own tick-labeled axis.
        text_params = list(cat_chosen or [])
        text_params = text_params[:3]  # safety limit
    
        text_axis_start = int(max(1, numeric_axis_count))  # after numeric axes
    
        for t_idx, text_param in enumerate(text_params):
            raw = wide[text_param].astype(str).fillna("").replace("nan", "").str.strip()
            mask = raw.ne("")
            if mask.sum() == 0:
                continue
    
            cats = _dedupe_keep_order(raw[mask].tolist())[:50]
            # Use real text values on Y (categorical axis)
            y_text = raw.where(mask, other=np.nan)
    
            ax_i = text_axis_start + t_idx
            text_yaxis_id = _yaxis_id(ax_i)
            text_yaxis_layout = _yaxis_layout_name(ax_i)
    
            colr = palette[(text_axis_start + t_idx) % len(palette)]
    
            fig.add_trace(
                go.Scatter(
                    x=wide["event_ts"],
                    y=y_text,
                    mode="markers",
                    name=f"{text_param} (text)",
                    customdata=wide["event_key"],
                    hovertemplate=f"<b>{text_param}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>",
                    marker=dict(
                        size=7,
                        opacity=0.95,
                        symbol="diamond",
                        color=colr,
                        line=dict(width=1, color=colr)
                    ),
                    showlegend=True,
                ),
                row=1, col=1
            )
            fig.data[-1].update(xaxis="x", yaxis=text_yaxis_id)
    
            fig.update_layout(
                **{
                    text_yaxis_layout: dict(
                        title=dict(text=f"{text_param}", font=dict(size=11, color=colr)),
                        overlaying="y",
                        side="left",
                        anchor="free",
                        position=0,
                        autoshift=True,
                        shift=-140 - (t_idx * 90),
                        showgrid=False,
                        zeroline=False,
                        showticklabels=True,
                        ticks="outside",
                        tickfont=dict(size=10, color=colr),
                        showline=True,
                        linecolor=colr,
                        type="category",
                        categoryorder="array",
                        categoryarray=cats,
                        automargin=True,
                    )
                }
            )
    
    # ---- Configure numeric MULTI Y axes (colored like traces) ----
    base_title = axis_owner.get(0, {}).get("name", "Value")
    base_color = axis_owner.get(0, {}).get("color", "#444444")
    
    fig.update_layout(
        yaxis=dict(
            title=dict(text=base_title, font=dict(color=base_color, size=11)),
            tickfont=dict(color=base_color),
            showticklabels=True,
            ticks="outside",
            showgrid=True,
            zeroline=False,
            side="left",
            showline=True,
            linecolor=base_color,
            automargin=True,
        ),
    )
    
    # numeric overlay axes for main plot: y3.. (y2 is reserved for row 2 events lane)
    # Make EVERY numeric/group parameter get its own visible Y axis on the right.
    for i in range(1, int(max(1, numeric_axis_count))):
        ax_name = _yaxis_layout_name(i)
        ttl = axis_owner.get(i, {}).get("name", f"Y{i + 1}")
        colr = axis_owner.get(i, {}).get("color", "#666666")
    
        fig.update_layout(
            **{
                ax_name: dict(
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=1.0,
                    # push each axis further right so each is visible
                    autoshift=True,
                    shift=(i - 1) * 65,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=True,
                    ticks="outside",
                    title=dict(text=ttl, font=dict(color=colr, size=11)),
                    tickfont=dict(color=colr),
                    showline=True,
                    linecolor=colr,
                    automargin=True,
                )
            }
        )
    
    # Force tick labels ON (sometimes overlay+free can hide them)
    fig.update_yaxes(showticklabels=True, ticks="outside", row=1, col=1)
    
    # =========================================================
    # Events lane (row 2): ONLY Maintenance + Faults
    # =========================================================
    y_fault = 0.30
    y_maint = 0.70
    
    show_maint_overlay = st.toggle(
        "Show Maintenance overlay",
        value=not maint_kv.empty,
        key="sql_show_maint_overlay",
        disabled=maint_kv.empty,
    )
    show_fault_overlay = st.toggle(
        "Show Faults overlay",
        value=not fault_kv.empty,
        key="sql_show_fault_overlay",
        disabled=fault_kv.empty,
    )
    
    if show_maint_overlay and not maint_kv.empty:
        mm = maint_kv.copy()
        mm["event_ts"] = pd.to_datetime(mm["event_ts"], errors="coerce")
        mm = mm[mm["event_ts"].notna()].copy()
        mm["event_key"] = "maintenance:" + mm["event_id"].astype(str)
        fig.add_trace(
            go.Scatter(
                x=mm["event_ts"],
                y=[y_maint] * len(mm),
                mode="markers",
                name="Maintenance",
                marker=dict(size=12, symbol="triangle-up", color="#BBBBBB"),
                customdata=mm["event_key"],
                text=("<b>" + mm["Parameter Name"].astype(str) + "</b><br>" + mm["Value"].astype(str)),
                hovertemplate="%{text}<br>%{x}<br>%{customdata}<extra></extra>",
            ),
            row=2, col=1
        )
    
    if show_fault_overlay and not fault_kv.empty:
        ff = fault_kv.copy()
        ff["event_ts"] = pd.to_datetime(ff["event_ts"], errors="coerce")
        ff = ff[ff["event_ts"].notna()].copy()
        ff["event_key"] = "fault:" + ff["event_id"].astype(str)
        fig.add_trace(
            go.Scatter(
                x=ff["event_ts"],
                y=[y_fault] * len(ff),
                mode="markers",
                name="Faults",
                marker=dict(size=13, symbol="x", color="#FF6666"),
                customdata=ff["event_key"],
                text=("<b>" + ff["Parameter Name"].astype(str) + "</b><br>" + ff["Value"].astype(str)),
                hovertemplate="%{text}<br>%{x}<br>%{customdata}<extra></extra>",
            ),
            row=2, col=1
        )
    
    fig.update_yaxes(
        row=2, col=1,
        range=[0, 1],
        tickmode="array",
        tickvals=[y_fault, y_maint],
        ticktext=["Faults", "Maintenance"],
        showgrid=False,
        zeroline=False,
        title_text="",
        showticklabels=True,
        ticks="outside",
    )
    
    # X padding so edges are not cropped
    all_ts = []
    if show_draw_traces and wide is not None and not wide.empty:
        all_ts.append(pd.to_datetime(wide["event_ts"], errors="coerce"))
    if show_maint_overlay and not maint_kv.empty:
        all_ts.append(pd.to_datetime(maint_kv["event_ts"], errors="coerce"))
    if show_fault_overlay and not fault_kv.empty:
        all_ts.append(pd.to_datetime(fault_kv["event_ts"], errors="coerce"))
    
    if all_ts:
        ts_cat = pd.concat(all_ts, ignore_index=True).dropna()
        if not ts_cat.empty:
            xmin, xmax = ts_cat.min(), ts_cat.max()
            if pd.notna(xmin) and pd.notna(xmax) and xmax > xmin:
                span = xmax - xmin
                pad = span * 0.07
                fig.update_xaxes(range=[xmin - pad, xmax + pad])
    
    # Left margin grows when text axes exist
    left_margin = 170 + (85 * min(3, len(list(cat_chosen or [])))) if show_draw_traces else 170
    
    # Right margin grows when many numeric axes are on the right
    right_margin = 140 + (70 * max(0, int(numeric_axis_count) - 1))
    right_margin = min(700, right_margin)
    
    fig.update_layout(
        height=780,
        margin=dict(l=left_margin, r=right_margin, t=70, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        title="Timeline: Draws + Events (filtered)",
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="sql_vis_plot")
    
    selected_key = None
    try:
        if isinstance(sel, dict):
            pts = sel.get("selection", {}).get("points", sel.get("points", []))
            if pts:
                selected_key = pts[0].get("customdata")
    except Exception:
        selected_key = None
    
    if selected_key:
        st.session_state["sql_selected_event_key"] = selected_key
    
    if st.session_state.get("sql_selected_event_key"):
        with st.expander("📌 Clicked event details", expanded=True):
            render_event_details(st.session_state["sql_selected_event_key"])
    
    # =========================================================
    # 🧮 MATH LAB
    # =========================================================
    st.subheader("🧮 Math Lab")
    
    if wide is None or wide.empty:
        st.info("No draw events available for Math Lab (run filter with Draws enabled).")
        st.stop()
    
    if not numeric_all:
        st.info("No numeric draw parameters available for Math Lab.")
        st.stop()
    
    st.caption("Expressions use **A**, **B**, **C** and **np**. Example: `A/B`, `np.log10(A)`")
    
    var_count = st.radio("How many parameters?", [1, 2, 3], horizontal=True, key="math_var_count")
    
    A_name = st.selectbox("A", numeric_all, key="math_A_name")
    B_name = None
    C_name = None
    if var_count >= 2:
        B_name = st.selectbox("B", [pp for pp in numeric_all if pp != A_name], key="math_B_name")
    if var_count >= 3:
        C_name = st.selectbox("C", [pp for pp in numeric_all if pp not in (A_name, B_name)], key="math_C_name")
    
    st.session_state.setdefault("math_expr", "A")
    expr = st.text_input("Expression", value=str(st.session_state["math_expr"]), key="math_expr_input").strip()
    st.session_state["math_expr"] = expr
    
    if not re.fullmatch(r"[0-9A-Za-z_\.\+\-\*\/\(\)\s,]+", expr or ""):
        st.error("Expression contains unsupported characters.")
        st.stop()
    
    
    def _series(name):
        if not name:
            return pd.Series([np.nan] * len(wide), index=wide.index)
        return pd.to_numeric(wide[name], errors="coerce").astype(float)
    
    
    A = _series(A_name)
    B = _series(B_name) if var_count >= 2 else pd.Series([np.nan] * len(wide), index=wide.index)
    C = _series(C_name) if var_count >= 3 else pd.Series([np.nan] * len(wide), index=wide.index)
    
    try:
        Y = eval(expr, {"__builtins__": {}}, {"np": np, "A": A, "B": B, "C": C})
        if isinstance(Y, (int, float, np.number)):
            Y = pd.Series([float(Y)] * len(wide), index=wide.index)
        elif isinstance(Y, np.ndarray):
            Y = pd.Series(Y, index=wide.index)
        elif not isinstance(Y, pd.Series):
            Y = pd.Series(Y, index=wide.index)
        Y = pd.to_numeric(Y, errors="coerce")
    except Exception as e:
        st.error(f"Expression error: {e}")
        st.stop()
    
    out = wide[["event_ts", "event_key"]].copy()
    out["math"] = Y
    out = out.dropna(subset=["math"]).sort_values("event_ts")
    
    if out.empty:
        st.warning("No values computed.")
        st.stop()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=out["event_ts"],
        y=out["math"],
        mode="lines+markers",
        name=expr,
        customdata=out["event_key"],
        hovertemplate=f"<b>{expr}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>"
    ))
    fig2.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=60, b=50),
        title="Math Lab result (click a point to inspect draw)",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="closest",
    )
    
    sel2 = st.plotly_chart(fig2, use_container_width=True, on_select="rerun", key="math_plot")
    
    selected_key2 = None
    try:
        if isinstance(sel2, dict):
            pts = sel2.get("selection", {}).get("points", sel2.get("points", []))
            if pts:
                selected_key2 = pts[0].get("customdata")
    except Exception:
        selected_key2 = None
    
    if selected_key2:
        st.session_state["math_selected_event_key"] = selected_key2
    
    if st.session_state.get("math_selected_event_key"):
        with st.expander("📌 Clicked event details (Math)", expanded=True):
            render_event_details(st.session_state["math_selected_event_key"])
    
    with st.expander("Math table", expanded=False):
        st.dataframe(out, use_container_width=True, height=420)
