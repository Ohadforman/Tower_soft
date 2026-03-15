def render_dashboard_tab(P):
        # ==========================================================
        # Imports (local)
        # ==========================================================
        import os
        import time
        from datetime import datetime
        import numpy as np
        import pandas as pd
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        
        from helpers.text_utils import safe_str
        from helpers.dataset_csv_io import append_rows_to_dataset_csv
        from helpers.dataset_param_parsers import (
            _parse_steps,
            zone_lengths_from_log_km,
            build_tm_instruction_rows_auto_from_good_zones,
        )
        t_run_start = time.perf_counter()
        perf_marks = [("start", t_run_start)]

        def _perf_mark(name: str):
            perf_marks.append((name, time.perf_counter()))
        
        # ==========================================================
        # WIDE (best set ONCE at top of app; safe-guard here)
        # ==========================================================
        if "_page_config_set" not in st.session_state:
            try:
                st.set_page_config(layout="wide")
            except Exception:
                pass
            st.session_state["_page_config_set"] = True
        
        st.markdown(
            """
            <style>
              .block-container { max-width: 98rem !important; padding-top: 2.4rem; }
              .dash-title{
                font-size: 1.62rem;
                font-weight: 900;
                margin: 0;
                padding-top: 4px;
                line-height: 1.2;
                color: rgba(236,248,255,0.98);
                text-shadow: 0 0 12px rgba(80,174,255,0.20);
              }
              .dash-sub{
                margin: 4px 0 10px 0;
                color: rgba(186,224,248,0.88);
                font-size: 0.92rem;
              }
              .dash-top-line{
                height: 1px;
                margin: 0 0 12px 0;
                background: linear-gradient(90deg, rgba(120,200,255,0.56), rgba(120,200,255,0.0));
              }
              .dash-soft-card{
                border: 1px solid rgba(128,206,255,0.22);
                border-radius: 12px;
                background: linear-gradient(180deg, rgba(14,32,56,0.30), rgba(8,16,28,0.22));
                padding: 8px 10px;
                margin: 4px 0 10px 0;
                backdrop-filter: blur(2px);
              }
              .dash-soft-card p{
                margin: 0 !important;
                color: rgba(197,229,249,0.90) !important;
                font-size: 0.84rem;
              }
              .dash-section{
                margin-top: 4px;
                margin-bottom: 6px;
                padding-left: 8px;
                border-left: 3px solid rgba(120,200,255,0.62);
                font-size: 1.00rem;
                font-weight: 800;
                color: rgba(224,243,255,0.98);
              }
              .dash-signal-chip-wrap{
                margin-top: -4px;
                margin-bottom: 10px;
                padding: 6px 8px 7px 8px;
                border: 1px solid rgba(128,206,255,0.18);
                border-radius: 10px;
                background: rgba(10,20,36,0.26);
              }
              div[data-testid="stMultiSelect"] div[data-baseweb="tag"]{
                background: linear-gradient(180deg, rgba(70,160,238,0.92), rgba(32,96,168,0.90)) !important;
                border: 1px solid rgba(170,232,255,0.78) !important;
                color: rgba(244,252,255,0.99) !important;
              }
              div[data-testid="stMultiSelect"] div[data-baseweb="tag"] svg{
                fill: rgba(238,250,255,0.98) !important;
              }
              div[data-testid="stButton"] > button:disabled{
                opacity: 0.78 !important;
                color: rgba(212,238,255,0.92) !important;
                border-color: rgba(128,206,255,0.30) !important;
                background: linear-gradient(180deg, rgba(24,62,102,0.50), rgba(12,34,64,0.46)) !important;
                box-shadow: 0 4px 10px rgba(8,30,58,0.20) !important;
              }
              div[data-testid="stButton"] > button:disabled:hover{
                transform: none !important;
                border-color: rgba(128,206,255,0.30) !important;
                box-shadow: 0 4px 10px rgba(8,30,58,0.20) !important;
              }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # ==========================================================
        # Constants / paths
        # ==========================================================
        DATASET_DIR = P.dataset_dir
        LOGS_DIR = getattr(P, "logs_dir", None) or getattr(P, "log_dir", None) or "logs"
        
        st.markdown('<div class="dash-title">📊 Draw Tower Logs Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="dash-sub">Analyze signals, mark good zones, and export T&M-ready zone summaries.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="dash-top-line"></div>', unsafe_allow_html=True)
        
        # ==========================================================
        # Helpers
        # ==========================================================
        def _mtime(path: str) -> float:
            try:
                return os.path.getmtime(path)
            except Exception:
                return 0.0

        @st.cache_data(show_spinner=False)
        def _cached_list_csv_names(dir_path: str, dir_mtime: float):
            if not os.path.exists(dir_path):
                return []
            return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".csv")])

        @st.cache_data(show_spinner=False)
        def _cached_sorted_csv_names_by_mtime(dir_path: str, dir_mtime: float):
            if not os.path.exists(dir_path):
                return []
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(".csv")]
            return sorted(
                files,
                key=lambda fn: os.path.getmtime(os.path.join(dir_path, fn)),
                reverse=True,
            )

        @st.cache_data(show_spinner=False)
        def _cached_latest_csv_name(dir_path: str, dir_mtime: float):
            if not os.path.exists(dir_path):
                return None
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".csv")]
            if not files:
                return None
            return os.path.basename(max(files, key=os.path.getmtime))

        @st.cache_data(show_spinner=False)
        def _cached_read_csv(path: str, keep_default_na: bool, file_mtime: float):
            return pd.read_csv(path, keep_default_na=keep_default_na)

        def list_dataset_csvs(dataset_dir):
            return _cached_list_csv_names(dataset_dir, _mtime(dataset_dir))
        
        def get_most_recent_dataset_csv(dataset_dir):
            return _cached_latest_csv_name(dataset_dir, _mtime(dataset_dir))
        
        def resolve_log_path(selected_name):
            if not selected_name:
                return ""
            if os.path.exists(selected_name):
                return selected_name
            cand = os.path.join(LOGS_DIR, selected_name)
            if os.path.exists(cand):
                return cand
            for d in ["logs", "log_csv", "draw_logs", "tower_logs", "data_logs"]:
                cand2 = os.path.join(d, selected_name)
                if os.path.exists(cand2):
                    return cand2
            return selected_name
        
        def _fmt_x(v):
            try:
                if isinstance(v, (pd.Timestamp, datetime)):
                    return pd.to_datetime(v).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            return str(v)
        
        def _sec(title: str):
            return {"Parameter Name": f"### {title}", "Value": "", "Units": ""}
        
        def _blank():
            return {"Parameter Name": "", "Value": "", "Units": ""}
        
        def _choose_length_col(df_):
            if df_ is None or df_.empty:
                return None
            cols = list(df_.columns)
            cmap = {str(c).strip().lower(): c for c in cols}
            for k in ["fibre length (km)", "fiber length (km)", "fibre length", "fiber length", "fibre_length", "fiber_length"]:
                if k in cmap:
                    return cmap[k]
            for c in cols:
                cl = str(c).lower()
                if "length" in cl and ("fiber" in cl or "fibre" in cl):
                    return c
            for c in cols:
                if "length" in str(c).lower():
                    return c
            return None
        
        def reorder_zones_by_spool_end(df_plot, x_axis_for_sort, zones, length_col):
            """
            Returns zones reordered so index 1 is closest to spool end.
            Sorting key = smallest km_from_end_start.
            NOTE: Uses df_plot (the plotted df with same selection/index space as zones).
            """
            if not zones or df_plot is None or df_plot.empty or not length_col:
                return zones
        
            dfw = df_plot.sort_values(by=x_axis_for_sort).copy()
            L_all = pd.to_numeric(dfw[length_col], errors="coerce").dropna()
            if L_all.empty:
                return zones
            L_end = float(L_all.iloc[-1])
        
            enriched = []
            for orig_i, (zs, ze) in enumerate(zones, start=1):
                try:
                    zdf = dfw[(dfw[x_axis_for_sort] >= zs) & (dfw[x_axis_for_sort] <= ze)]
                except Exception:
                    zdf = pd.DataFrame()
                if zdf.empty:
                    continue
                Lz = pd.to_numeric(zdf[length_col], errors="coerce").dropna()
                if Lz.empty:
                    continue
                L_max = float(Lz.max())
                km0 = max(0.0, L_end - L_max)
                enriched.append((km0, orig_i, (zs, ze)))
        
            if not enriched:
                return zones
        
            enriched.sort(key=lambda t: t[0])
            return [t[2] for t in enriched]
        
        # ==========================================================
        # ✅ FIXED SAVE: slice by selected ROWS (plot selection), not by datetime parsing
        # - df_all: full log (all columns, all rows)
        # - df_plot: plotted df (sorted, has __x_sel__ and __rownum__)
        # zones are in df_plot selection space (x_axis_slice)
        # ==========================================================
        def build_zone_save_rows(
            log_file_path,
            x_axis_display,   # real column name (e.g. "Date/Time")
            x_axis_slice,     # "__x_sel__" when datetime, else x_axis_display
            df_all,           # full log dataframe (with __rownum__)
            df_plot,          # plotted dataframe (with __rownum__, sorted, with __x_sel__ if dt)
            zones,
            exclude_cols=None,
        ):
            exclude_cols = set([str(c).strip() for c in (exclude_cols or [])])
        
            rows = []
            now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
            rows += [_sec("DASHBOARD ZONES"), _blank()]
            rows.append({"Parameter Name": "Zones Saved Timestamp", "Value": now_s, "Units": ""})
            rows.append({"Parameter Name": "Dashboard Log File", "Value": os.path.basename(log_file_path), "Units": ""})
            rows.append({"Parameter Name": "Good Zones Count", "Value": int(len(zones)), "Units": "count"})
            rows.append({"Parameter Name": "Good Zones X Column", "Value": str(x_axis_display), "Units": ""})
        
            if not zones:
                rows += [_blank()]
                return rows
        
            # summarize ALL columns except x + helper cols
            cols = []
            for c in df_all.columns.tolist():
                if c in exclude_cols:
                    continue
                if c in ["__rownum__", "__x_sel__"]:
                    continue
                if c == x_axis_display:
                    continue
                cols.append(c)
        
            def _to_num(s: pd.Series) -> pd.Series:
                return pd.to_numeric(s, errors="coerce")
        
            for i, (start, end) in enumerate(zones, start=1):
                # slice df_plot by selection axis
                try:
                    a = float(start); b = float(end)
                    if b < a:
                        a, b = b, a
                    zplot = df_plot[(df_plot[x_axis_slice] >= a) & (df_plot[x_axis_slice] <= b)].copy()
                except Exception:
                    zplot = pd.DataFrame()
        
                # start/end display values (REAL x values from df_plot)
                if not zplot.empty:
                    x0 = zplot[x_axis_display].iloc[0]
                    x1 = zplot[x_axis_display].iloc[-1]
                    rows.append({"Parameter Name": f"Zone {i} | Start", "Value": _fmt_x(x0), "Units": str(x_axis_display)})
                    rows.append({"Parameter Name": f"Zone {i} | End", "Value": _fmt_x(x1), "Units": str(x_axis_display)})
                else:
                    rows.append({"Parameter Name": f"Zone {i} | Start", "Value": str(start), "Units": str(x_axis_display)})
                    rows.append({"Parameter Name": f"Zone {i} | End", "Value": str(end), "Units": str(x_axis_display)})
                    rows.append({"Parameter Name": f"Zone {i} | DEBUG", "Value": "zplot EMPTY (selection window has no points)", "Units": ""})
                    continue
        
                # ✅ THIS IS THE KEY: slice the FULL LOG using the selected row numbers
                try:
                    rownums = zplot["__rownum__"].dropna().astype(int).tolist()
                    zdf = df_all[df_all["__rownum__"].isin(rownums)].copy()
                except Exception:
                    zdf = pd.DataFrame()
        
                if zdf.empty:
                    rows.append({"Parameter Name": f"Zone {i} | DEBUG", "Value": "zdf EMPTY (rownum slice failed)", "Units": ""})
                    continue
        
                for col in cols:
                    vals = _to_num(zdf[col]).dropna()
                    if vals.empty:
                        continue
                    rows.append({"Parameter Name": f"Zone {i} | {col} | Avg", "Value": float(vals.mean()), "Units": ""})
                    rows.append({"Parameter Name": f"Zone {i} | {col} | Min", "Value": float(vals.min()), "Units": ""})
                    rows.append({"Parameter Name": f"Zone {i} | {col} | Max", "Value": float(vals.max()), "Units": ""})
        
            rows += [_blank()]
            return rows
        
        def build_tm_rows_from_steps_allocate_only(dataset_csv_name: str, steps: list, zones_info: list, length_col_name: str = ""):
            rows = []
            rows += [_sec("T&M CUT/SAVE PLAN"), _blank()]
            rows.append({"Parameter Name": "Plan Source (dataset CSV)", "Value": str(os.path.basename(dataset_csv_name)), "Units": ""})
            rows.append({"Parameter Name": "Plan Mode", "Value": "STEP plan from dataset CSV (allocated on good zones)", "Units": ""})
            if length_col_name:
                rows.append({"Parameter Name": "Length Column (log)", "Value": str(length_col_name), "Units": ""})
        
            if not steps:
                rows.append({"Parameter Name": "T&M Instructions", "Value": "STEP plan empty.", "Units": ""})
                rows += [_blank()]
                return rows
        
            rows += [_blank(), _sec("STEP PLAN (from dataset)"), _blank()]
            for i, (a, L) in enumerate(steps, start=1):
                rows.append({"Parameter Name": f"STEP {i} | Action", "Value": str(a).upper(), "Units": ""})
                rows.append({"Parameter Name": f"STEP {i} | Length", "Value": float(L), "Units": "km"})
        
            rows += [_blank(), _sec("ALLOCATED ON GOOD ZONES"), _blank()]
        
            tm_i = 1
            step_idx = 0
            step_act, step_rem = steps[0][0], float(steps[0][1])
        
            saved = 0.0
            cut = 0.0
        
            for z in zones_info:
                zlen = z.get("len_km")
                zi = z.get("i", None)
                if zlen is None or float(zlen) <= 0:
                    continue
                zone_remaining = float(zlen)
        
                while zone_remaining > 1e-9 and step_idx < len(steps):
                    take = min(zone_remaining, step_rem)
        
                    rows.append({"Parameter Name": f"T&M {tm_i} | Action", "Value": str(step_act).upper(), "Units": ""})
                    rows.append({"Parameter Name": f"T&M {tm_i} | Length", "Value": float(take), "Units": "km"})
                    rows.append({"Parameter Name": f"T&M {tm_i} | From", "Value": f"Zone {zi}", "Units": ""})
        
                    if str(step_act).upper() == "SAVE":
                        saved += float(take)
                    else:
                        cut += float(take)
        
                    zone_remaining -= take
                    step_rem -= take
                    tm_i += 1
        
                    if step_rem <= 1e-9:
                        step_idx += 1
                        if step_idx < len(steps):
                            step_act, step_rem = steps[step_idx][0], float(steps[step_idx][1])
        
            rows += [_blank(), _sec("T&M TOTALS"), _blank()]
            rows.append({"Parameter Name": "Total Saved Length", "Value": float(saved), "Units": "km"})
            rows.append({"Parameter Name": "Total Cut Length", "Value": float(cut), "Units": "km"})
            rows += [_blank()]
            return rows
        
        def _extract_selected_drum_from_dataset_df(df_params: pd.DataFrame) -> str:
            if df_params is None or df_params.empty:
                return ""
            try:
                s = df_params["Parameter Name"].astype(str).str.strip()
                hit = df_params.loc[s.isin(["Process__Selected Drum", "Selected Drum"]), "Value"]
                if hit is None or hit.empty:
                    return ""
                return str(hit.iloc[-1]).strip()
            except Exception:
                return ""
        
        # ==========================================================
        # Load log CSV
        # ==========================================================
        csv_files = _cached_list_csv_names(LOGS_DIR, _mtime(LOGS_DIR))
        _perf_mark("logs_listed")
        if not csv_files:
            st.error("No CSV files found in the logs directory.")
            st.stop()

        csv_files_sorted = _cached_sorted_csv_names_by_mtime(LOGS_DIR, _mtime(LOGS_DIR))
        if not csv_files_sorted:
            csv_files_sorted = csv_files
        latest_file = csv_files_sorted[0]

        if not st.session_state.get("dataset_select") or st.session_state.get("dataset_select") not in csv_files_sorted:
            st.session_state["dataset_select"] = latest_file

        st.markdown("<div class='dash-section'>Dataset</div>", unsafe_allow_html=True)
        ds1, ds2 = st.columns([2.5, 1.2], gap="small")
        with ds1:
            selected_file = st.selectbox(
                "Select a dataset",
                options=csv_files_sorted,
                key="dataset_select",
            )
        with ds2:
            st.markdown(
                f"<div class='dash-soft-card'><p>Latest: <b>{latest_file}</b></p></div>",
                unsafe_allow_html=True,
            )

        log_path = resolve_log_path(selected_file)
        if not log_path or not os.path.exists(log_path):
            st.error(f"Failed to read log CSV: file not found.\n\nSelected: {selected_file}\nTried: {log_path}")
            st.stop()
        
        try:
            df = _cached_read_csv(log_path, keep_default_na=True, file_mtime=_mtime(log_path))
            _perf_mark("log_csv_read")
        except Exception as e:
            st.error(f"Failed to read log CSV: {e}")
            st.stop()
        
        if df is None or df.empty:
            st.warning("Log CSV loaded but is empty.")
            st.stop()
        
        # ✅ add stable row id immediately
        df = df.copy()
        df["__rownum__"] = np.arange(len(df), dtype=int)
        
        # ==========================================================
        # Dataset CSV context
        # ==========================================================
        recent_dataset_csvs = list_dataset_csvs(DATASET_DIR)
        latest_dataset_csv = get_most_recent_dataset_csv(DATASET_DIR)
        st.markdown(
            f"<div class='dash-soft-card'><p>Most recent dataset CSV: <b>{latest_dataset_csv if latest_dataset_csv else 'None'}</b></p></div>",
            unsafe_allow_html=True,
        )
        
        # ==========================================================
        # Session state
        # ==========================================================
        if "good_zones" not in st.session_state:
            st.session_state["good_zones"] = []
        if "dash_last_log_file" not in st.session_state:
            st.session_state["dash_last_log_file"] = ""
        if "dash_zone_msg" not in st.session_state:
            st.session_state["dash_zone_msg"] = ""
        
        if "dash_queued_zones" not in st.session_state:
            st.session_state["dash_queued_zones"] = []
        if "dash_preview_zone" not in st.session_state:
            st.session_state["dash_preview_zone"] = None
        if "dash_last_sel_sig" not in st.session_state:
            st.session_state["dash_last_sel_sig"] = None
        
        if st.session_state["dash_last_log_file"] != os.path.basename(log_path):
            st.session_state["good_zones"] = []
            st.session_state["dash_queued_zones"] = []
            st.session_state["dash_preview_zone"] = None
            st.session_state["dash_last_sel_sig"] = None
            st.session_state["dash_last_log_file"] = os.path.basename(log_path)
            st.session_state["dash_zone_msg"] = ""
        
        def _dedup_and_sort(zs):
            """De-duplicate zones while preserving the order they were added.
        
            NOTE: We intentionally DO NOT sort here, because sorting makes "Undo last" remove the
            last-by-x zone instead of the last-added zone.
            """
            out = []
            seen = set()
            for a, b in zs:
                try:
                    aa = float(a)
                    bb = float(b)
                except Exception:
                    # If for some reason it's not numeric, keep as-is but still try to dedup by string
                    key = (str(a), str(b))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((a, b))
                    continue
        
                # normalize ordering within a zone
                if bb < aa:
                    aa, bb = bb, aa
        
                # robust key (matches your _sig precision)
                key = (round(aa, 9), round(bb, 9))
                if key in seen:
                    continue
                seen.add(key)
                out.append((aa, bb))
            return out
        
        def _sig(a, b):
            return (round(float(a), 9), round(float(b), 9))
        
        # ==========================================================
        # Controls
        # ==========================================================
        column_options = [c for c in df.columns.tolist() if c != "__rownum__"]
        if not column_options:
            st.warning("No columns found in log CSV.")
            st.stop()
        
        st.markdown("<div class='dash-section'>Signals Setup</div>", unsafe_allow_html=True)
        s1, s2 = st.columns([1, 1], gap="medium")
        with s1:
            x_axis = st.selectbox("Select X-axis", column_options, key="x_axis_dash")
        with s2:
            y_axes = st.multiselect(
                "Select Y-axis column(s)",
                options=column_options,
                default=[],
                key="y_axes_dash_multi",
            )
        
        if not y_axes:
            st.info("Select one or more **Y-axis** columns to show the plot + zones.")
            st.stop()
        
        # ==========================================================
        # Nice UI labels (colored chips) for the selected Y signals
        # ==========================================================
        st.markdown("<div class='dash-section'>Selected signals</div>", unsafe_allow_html=True)
        chips = []
        _chip_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        for i, y_col in enumerate(y_axes):
            color = _chip_colors[i % len(_chip_colors)]
            chips.append(
                f"<span style='display:inline-block; padding:4px 10px; margin:4px 6px 0 0; border-radius:999px; "
                f"border:1px solid {color}; color:{color}; font-weight:600; font-size:0.95rem;'>" 
                f"{safe_str(y_col)}" 
                f"</span>"
            )
        st.markdown("""<div class='dash-signal-chip-wrap'>""" + "".join(chips) + "</div>", unsafe_allow_html=True)
        
        # ==========================================================
        # Stable x typing + build df_plot
        # ==========================================================
        df_work = df.copy()
        
        x_raw = df_work[x_axis]
        x_dt = pd.to_datetime(x_raw, errors="coerce", utc=False)
        dt_ok_ratio = float(x_dt.notna().mean()) if len(x_dt) else 0.0
        
        if dt_ok_ratio > 0.80:
            df_work[x_axis] = x_dt
        else:
            x_num = pd.to_numeric(x_raw, errors="coerce")
            num_ok_ratio = float(x_num.notna().mean()) if len(x_num) else 0.0
            if num_ok_ratio > 0.80:
                df_work[x_axis] = x_num
            else:
                df_work[x_axis] = x_raw.astype(str)
        
        # df_all = full log (keep ALL columns)
        df_all = df_work.copy()
        
        # df_plot = only rows that can be plotted (dropna on x+y), but KEEP __rownum__
        df_plot = df_work.dropna(subset=[x_axis] + y_axes).sort_values(by=x_axis).copy()
        if df_plot.empty:
            st.warning("No data to plot after filtering NA values for selected X/Y columns.")
            st.stop()
        _perf_mark("plot_dataframe_prepared")
        
        # ==========================================================
        # ✅ FORCE numeric selection axis for zones (works for ANY X type)
        # - Plotly selection returns numeric ranges when X is categorical
        # - We always use __x_sel__ to slice zones reliably
        # ==========================================================
        x_axis_display = x_axis          # real column name to display in UI/hover
        x_axis_slice   = "__x_sel__"     # numeric axis used for selection/slicing
        
        df_plot = df_plot.copy()
        df_plot["__x_sel__"] = np.arange(len(df_plot), dtype=float)
        df_plot["__x_sel__"] = pd.to_numeric(df_plot["__x_sel__"], errors="coerce")
        
        # ==========================================================
        # LIVE Zone Marker settings
        # ==========================================================
        if st.sidebar.checkbox("Show dashboard timings", value=False, key="dash_show_perf"):
            rows_perf = []
            for i in range(1, len(perf_marks)):
                step = perf_marks[i][0]
                ms = (perf_marks[i][1] - perf_marks[i - 1][1]) * 1000.0
                rows_perf.append({"step": step, "ms": round(ms, 1)})
            total_ms = (time.perf_counter() - t_run_start) * 1000.0
            rows_perf.append({"step": "total_so_far", "ms": round(total_ms, 1)})
            with st.expander("⏱ Dashboard Performance", expanded=False):
                st.dataframe(pd.DataFrame(rows_perf), use_container_width=True, hide_index=True)

        st.markdown("<div class='dash-section'>🟩 Zone Marker (LIVE on-plot selection)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='dash-soft-card'><p>Drag a horizontal window on the plot. Use <b>Auto-queue</b> for fast marking, then click <b>Add Queued Zones</b>.</p></div>",
            unsafe_allow_html=True,
        )
        
        opt1, opt2, opt3 = st.columns([1, 1, 2])
        with opt1:
            dash_auto_queue = st.checkbox("Auto-queue every selection", value=True, key="dash_auto_queue")
        with opt2:
            dash_keep_preview = st.checkbox("Keep preview after queue", value=False, key="dash_keep_preview")
        with opt3:
            st.info(
                f"Saved: **{len(st.session_state['good_zones'])}** | "
                f"Queued: **{len(st.session_state['dash_queued_zones'])}** | "
                f"Preview: **{'Yes' if st.session_state['dash_preview_zone'] else 'No'}**"
            )
        
        # ==========================================================
        # Plot interaction mode (Zone marking vs Pan/Zoom)
        # ==========================================================
        mcol1, mcol2, mcol3 = st.columns([1.1, 1.1, 3.0])
        with mcol1:
            zone_mode = st.toggle(
                "🟩 Zone Mode",
                value=True,
                key="dash_zone_mode_toggle",
                help="ON = mark zones (box select) | OFF = pan/zoom freely",
            )
        with mcol2:
            pan_or_zoom = st.selectbox(
                "When Zone Mode OFF",
                options=["Pan", "Zoom"],
                index=0,
                key="dash_pan_zoom_choice",
            )
        with mcol3:
            if zone_mode:
                st.info("Zone Mode is ON: drag on the plot to mark a zone.")
            else:
                st.caption("Zone Mode is OFF: use pan/zoom, then turn Zone Mode back ON to mark again.")
        
        # Zone Mode should actually switch to select drag.
        # View is preserved via persisted axis ranges + stable uirevision.
        drag_mode = "select" if zone_mode else ("pan" if pan_or_zoom == "Pan" else "zoom")
        # When leaving Zone Mode, also clear any active selection box in Plotly (prevents stuck selection)
        if not zone_mode:
            st.session_state["dash_last_sel_sig"] = None
        
        # ==========================================================
        # Plot (selection uses numeric axis when datetime)
        # ==========================================================
        st.markdown("<div class='dash-section'>📈 Plot</div>", unsafe_allow_html=True)
        
        # Use dtype only for formatting ticks/hover; slicing is ALWAYS via __x_sel__
        is_dt_x = pd.api.types.is_datetime64_any_dtype(df_plot[x_axis_display])
        
        # ----------------------------------------------------------
        # ✅ Keep zoom/pan when toggling Zone Mode (Streamlit reruns)
        # Plotly will preserve view as long as uirevision stays stable.
        # Change uirevision only when log/X/Y changes.
        # ----------------------------------------------------------
        _uirev_key = f"{os.path.basename(log_path)}|{x_axis_display}|{','.join([str(y) for y in y_axes])}"
        st.session_state["dash_uirev_key"] = _uirev_key
        
        # ----------------------------------------------------------
        # ✅ Persist current zoom/pan ranges across reruns/toggles
        # We store the last Plotly relayout ranges per "uirevision key".
        # This is more reliable than uirevision alone when dragmode changes.
        # ----------------------------------------------------------
        CHART_KEY = "dash_zone_plot_live"

        def _safe_pair(v0, v1):
            try:
                return [float(v0), float(v1)]
            except Exception:
                return None

        def _extract_axis_pair(_obj, axis_name):
            """
            Best-effort extraction of axis range from different Streamlit payload shapes:
            - relayoutData: {"xaxis.range[0]": ..., "xaxis.range[1]": ...}
            - nested layout: {"layout": {"xaxis": {"range": [a,b]}}}
            - flat keys at any nested level
            """
            if not isinstance(_obj, (dict, list)):
                return None

            queue = [_obj]
            seen = set()
            while queue:
                cur = queue.pop(0)
                if id(cur) in seen:
                    continue
                seen.add(id(cur))

                if isinstance(cur, dict):
                    # Flat relayout style
                    k0 = f"{axis_name}.range[0]"
                    k1 = f"{axis_name}.range[1]"
                    if k0 in cur and k1 in cur:
                        pair = _safe_pair(cur.get(k0), cur.get(k1))
                        if pair is not None:
                            return pair

                    # Nested style: {"xaxis": {"range": [..,..]}}
                    a_obj = cur.get(axis_name)
                    if isinstance(a_obj, dict):
                        rg = a_obj.get("range")
                        if isinstance(rg, (list, tuple)) and len(rg) >= 2:
                            pair = _safe_pair(rg[0], rg[1])
                            if pair is not None:
                                return pair

                    # Nested style under layout
                    lay = cur.get("layout")
                    if isinstance(lay, dict):
                        a_obj2 = lay.get(axis_name)
                        if isinstance(a_obj2, dict):
                            rg2 = a_obj2.get("range")
                            if isinstance(rg2, (list, tuple)) and len(rg2) >= 2:
                                pair = _safe_pair(rg2[0], rg2[1])
                                if pair is not None:
                                    return pair

                    for vv in cur.values():
                        if isinstance(vv, (dict, list)):
                            queue.append(vv)
                elif isinstance(cur, list):
                    for vv in cur:
                        if isinstance(vv, (dict, list)):
                            queue.append(vv)

            return None

        def _persist_relayout_payload(_payload):
            if not isinstance(_payload, dict):
                return
            _relayout = _payload.get("relayoutData") or _payload.get("relayout_data")
            if not isinstance(_relayout, dict):
                _relayout = {}

            st.session_state.setdefault("dash_view_state", {})
            st.session_state["dash_view_state"].setdefault(_uirev_key, {})
            _vs = st.session_state["dash_view_state"][_uirev_key]

            # x range (numeric selection axis)
            xpair = _extract_axis_pair({"relayout": _relayout, "payload": _payload}, "xaxis")
            if xpair is not None:
                _vs["xrange"] = xpair
            elif _relayout.get("xaxis.autorange") is True:
                _vs.pop("xrange", None)

            # left y range
            ypair = _extract_axis_pair({"relayout": _relayout, "payload": _payload}, "yaxis")
            if ypair is not None:
                _vs["yrange"] = ypair
            elif _relayout.get("yaxis.autorange") is True:
                _vs.pop("yrange", None)

            # additional y-axes ranges (yaxis2, yaxis3, ...)
            ym = _vs.setdefault("yrange_map", {})
            for k in list(_relayout.keys()):
                if not isinstance(k, str):
                    continue
                if k.startswith("yaxis") and ".range[" in k:
                    axis_name = k.split(".", 1)[0]  # yaxis2
                    r0 = _relayout.get(f"{axis_name}.range[0]")
                    r1 = _relayout.get(f"{axis_name}.range[1]")
                    if r0 is not None and r1 is not None:
                        pair = _safe_pair(r0, r1)
                        if pair is not None:
                            ym[axis_name] = pair
                elif k.startswith("yaxis") and k.endswith(".autorange") and _relayout.get(k) is True:
                    axis_name = k.split(".", 1)[0]
                    ym.pop(axis_name, None)

            # Fallback scan for yaxis2..yaxis12 in nested payloads
            for i in range(2, 13):
                axis_name = f"yaxis{i}"
                pair = _extract_axis_pair({"relayout": _relayout, "payload": _payload}, axis_name)
                if pair is not None:
                    ym[axis_name] = pair

        # Capture existing relayout before rendering a new figure.
        # This keeps current zoom/pan when toggling Zone Mode.
        _persist_relayout_payload(st.session_state.get(CHART_KEY, None))

        st.session_state.setdefault("dash_view_state", {})
        _view_state = st.session_state["dash_view_state"].get(_uirev_key, {})
        _xrange = _view_state.get("xrange", None)  # [xmin, xmax]
        _yrange = _view_state.get("yrange", None)  # [ymin, ymax] for left axis
        _yrange_map = _view_state.get("yrange_map", {})  # axis_name -> [min,max]
        
        fig = go.Figure()
        
        default_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        
        for i, y_col in enumerate(y_axes):
            axis_ref = "y" if i == 0 else f"y{i + 1}"
            color = default_colors[i % len(default_colors)]
            y_vals = pd.to_numeric(df_plot[y_col], errors="coerce")
        
            # Always plot using numeric selection axis; show real X via customdata
            if is_dt_x:
                custom = pd.to_datetime(df_plot[x_axis_display], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
                hover_x_label = "Time"
            else:
                custom = df_plot[x_axis_display].astype(str).to_numpy()
                hover_x_label = str(x_axis_display)
        
            fig.add_trace(go.Scatter(
                x=df_plot[x_axis_slice],
                y=y_vals,
                mode="lines",
                name=y_col,
                yaxis=axis_ref,
                line=dict(color=color),
                customdata=custom,
                hovertemplate=f"{hover_x_label}: %{{customdata}}<br>y: %{{y}}<extra></extra>",
                # IMPORTANT: stable uid so Plotly can preserve zoom/pan with uirevision across Streamlit reruns
                uid=f"dash_trace_{i}_{str(y_col)}",
            ))
        
        # Saved zones (green)
        for (start, end) in st.session_state["good_zones"]:
            fig.add_vrect(x0=start, x1=end, fillcolor="green", opacity=0.25, line_width=0)
        
        # Queued zones (orange)
        for (start, end) in st.session_state["dash_queued_zones"]:
            fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.18, line_width=0)
        
        # Preview (blue)
        if st.session_state["dash_preview_zone"] is not None:
            a, b = st.session_state["dash_preview_zone"]
            fig.add_vrect(x0=a, x1=b, fillcolor="blue", opacity=0.18, line_width=1, line_dash="dot")
        
        layout_updates = {}
        
        # Alternate axis sides: Left, Right, Left, Right...
        left_positions = [0.00, 0.03, 0.06, 0.09, 0.12, 0.15]
        right_positions = [1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79]
        
        # Y1 (index 0) -> LEFT
        layout_updates["yaxis"] = dict(
            title=dict(text=""),
            tickfont=dict(color=default_colors[0]),
            showgrid=True,
            side="left",
            position=float(left_positions[0]),
        )
        
        # restore last left y-range if available
        if isinstance(_yrange, (list, tuple)) and len(_yrange) == 2:
            try:
                layout_updates["yaxis"]["range"] = [float(_yrange[0]), float(_yrange[1])]
            except Exception:
                pass
        
        left_i = 1
        right_i = 0
        
        # Additional axes
        for i in range(1, len(y_axes)):
            axis_key = f"yaxis{i + 1}"
            color = default_colors[i % len(default_colors)]
        
            # i=1 -> right, i=2 -> left, i=3 -> right ...
            use_left = (i % 2 == 0)
            if use_left:
                pos = left_positions[left_i] if left_i < len(left_positions) else (0.15 + 0.03 * (left_i - len(left_positions) + 1))
                side = "left"
                left_i += 1
            else:
                pos = right_positions[right_i] if right_i < len(right_positions) else max(0.55, 1.0 - 0.03 * right_i)
                side = "right"
                right_i += 1
        
            axis_cfg = dict(
                title=dict(text=""),
                tickfont=dict(color=color),
                anchor="x",
                overlaying="y",
                side=side,
                position=float(pos),
                showgrid=False,
                zeroline=False,
            )
        
            # restore per-axis range if available
            try:
                r = _yrange_map.get(axis_key) if isinstance(_yrange_map, dict) else None
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    axis_cfg["range"] = [float(r[0]), float(r[1])]
            except Exception:
                pass
        
            layout_updates[axis_key] = axis_cfg
        
        xaxis_cfg = dict(automargin=True, nticks=8, tickangle=-90, showgrid=False)
        # restore last x-range (numeric __x_sel__) if available
        if isinstance(_xrange, (list, tuple)) and len(_xrange) == 2:
            try:
                xaxis_cfg["range"] = [float(_xrange[0]), float(_xrange[1])]
            except Exception:
                pass
        
        tick_count = 10
        idx = np.linspace(0, len(df_plot) - 1, tick_count).astype(int)
        tickvals = df_plot["__x_sel__"].iloc[idx].astype(float).tolist()
        
        if is_dt_x:
            ticktext = pd.to_datetime(df_plot[x_axis_display], errors="coerce").iloc[idx].dt.strftime("%d/%m/%Y %H:%M:%S").tolist()
        else:
            ticktext = df_plot[x_axis_display].astype(str).iloc[idx].tolist()
        
        xaxis_cfg.update(dict(tickmode="array", tickvals=tickvals, ticktext=ticktext))
        
        fig.update_layout(
            **layout_updates,
            xaxis=xaxis_cfg,
            title=f"{' , '.join([str(y) for y in y_axes])} vs {x_axis_display}",
            margin=dict(l=70, r=70, t=85, b=10),
            height=620,
            legend=dict(
                visible=True,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
            ),
            dragmode=drag_mode,
            selectdirection="h",
            uirevision=str(st.session_state.get("dash_uirev_key", "dash_keep_view")),
        )
        
        returned = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("box",),
            key=CHART_KEY,
        )
        
        payload = None
        if isinstance(returned, dict):
            payload = returned
        elif isinstance(st.session_state.get(CHART_KEY, None), dict):
            payload = st.session_state.get(CHART_KEY, None)
        
        # Keep ranges fresh after each render event.
        try:
            _persist_relayout_payload(payload)
        except Exception:
            pass
        
        sel_rng = None
        try:
            sel = (payload or {}).get("selection", {})
            box = sel.get("box", [])
            if isinstance(box, list) and box and isinstance(box[0], dict):
                xr = box[0].get("x", None)
                if isinstance(xr, (list, tuple)) and len(xr) >= 2:
                    a = float(xr[0]); b = float(xr[1])
                    if b < a:
                        a, b = b, a
                    sel_rng = (a, b)
        except Exception:
            sel_rng = None
        
        if zone_mode and sel_rng is not None:
            a, b = sel_rng
            sig = _sig(a, b)
            if sig != st.session_state["dash_last_sel_sig"]:
                st.session_state["dash_last_sel_sig"] = sig
                st.session_state["dash_preview_zone"] = (a, b)
        
                if dash_auto_queue:
                    st.session_state["dash_queued_zones"].append((a, b))
                    st.session_state["dash_queued_zones"] = _dedup_and_sort(st.session_state["dash_queued_zones"])
                    st.session_state["dash_zone_msg"] = f"🟧 Queued ({len(st.session_state['dash_queued_zones'])})"
                    if not dash_keep_preview:
                        st.session_state["dash_preview_zone"] = None
        
                st.rerun()
        
        # ==========================================================
        # Zone actions
        # ==========================================================
        st.markdown("<div class='dash-section'>🟩 Zone Actions</div>", unsafe_allow_html=True)
        
        b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 1])
        
        with b1:
            if st.button("➕ Add Preview", use_container_width=True, disabled=(st.session_state["dash_preview_zone"] is None)):
                st.session_state["good_zones"].append(st.session_state["dash_preview_zone"])
                st.session_state["good_zones"] = _dedup_and_sort(st.session_state["good_zones"])
                st.session_state["dash_preview_zone"] = None
                st.session_state["dash_zone_msg"] = f"✅ Added preview (saved: {len(st.session_state['good_zones'])})"
                st.rerun()
        
        with b2:
            if st.button("🟧 Queue Preview", use_container_width=True, disabled=(st.session_state["dash_preview_zone"] is None)):
                st.session_state["dash_queued_zones"].append(st.session_state["dash_preview_zone"])
                st.session_state["dash_queued_zones"] = _dedup_and_sort(st.session_state["dash_queued_zones"])
                if not dash_keep_preview:
                    st.session_state["dash_preview_zone"] = None
                st.session_state["dash_zone_msg"] = f"🟧 Queued (queued: {len(st.session_state['dash_queued_zones'])})"
                st.rerun()
        
        with b3:
            if st.button("✅ Add Queued Zones", use_container_width=True, disabled=(len(st.session_state["dash_queued_zones"]) == 0)):
                st.session_state["good_zones"].extend(st.session_state["dash_queued_zones"])
                st.session_state["good_zones"] = _dedup_and_sort(st.session_state["good_zones"])
                st.session_state["dash_queued_zones"] = []
                st.session_state["dash_zone_msg"] = f"✅ Added queued zones (saved: {len(st.session_state['good_zones'])})"
                st.rerun()
        
        with b4:
            if st.button("↩️ Undo Last Saved", use_container_width=True, disabled=(len(st.session_state["good_zones"]) == 0)):
                st.session_state["good_zones"].pop()
                st.session_state["dash_zone_msg"] = f"↩️ Undone last saved (saved: {len(st.session_state['good_zones'])})"
                st.rerun()
        
        with b5:
            if st.button("↩️ Undo Last Queued", use_container_width=True, disabled=(len(st.session_state["dash_queued_zones"]) == 0)):
                st.session_state["dash_queued_zones"].pop()
                st.session_state["dash_zone_msg"] = f"↩️ Undone last queued (queued: {len(st.session_state['dash_queued_zones'])})"
                st.rerun()
        
        c6, c7 = st.columns([1, 3])
        with c6:
            if st.button(
                "🧹 Clear All",
                use_container_width=True,
                disabled=(len(st.session_state["good_zones"]) == 0 and len(st.session_state["dash_queued_zones"]) == 0 and st.session_state["dash_preview_zone"] is None),
            ):
                st.session_state["good_zones"] = []
                st.session_state["dash_queued_zones"] = []
                st.session_state["dash_preview_zone"] = None
                st.session_state["dash_last_sel_sig"] = None
                st.session_state["dash_zone_msg"] = "🧽 Cleared everything"
                st.rerun()
        
        with c7:
            if st.session_state.get("dash_zone_msg"):
                st.success(st.session_state["dash_zone_msg"])
        
        # ==========================================================
        # SAVE
        # ==========================================================
        st.markdown("---")
        st.subheader("💾 Save Zones + T&M Cut/Save Instructions → Dataset CSV")
        
        if not recent_dataset_csvs:
            st.warning("No dataset CSV files found in data_set_csv/. Create one in Process Setup first.")
        else:
            left, right = st.columns([2, 1])
            with left:
                selected_dataset_csv = st.selectbox(
                    "Choose a dataset CSV (optional)",
                    options=[""] + recent_dataset_csvs,
                    index=0,
                    key="dash_save_target_csv_select",
                    help="If empty, use MOST RECENT dataset CSV.",
                )
            with right:
                st.caption(f"Most recent: **{latest_dataset_csv if latest_dataset_csv else 'None'}**")
        
            b1, b2 = st.columns(2)
            with b1:
                save_to_latest_clicked = st.button("⚡ Save to MOST RECENT dataset CSV", key="dash_save_to_latest_btn", use_container_width=True)
            with b2:
                save_to_selected_clicked = st.button("💾 Save to SELECTED dataset CSV", key="dash_save_to_selected_btn", use_container_width=True)
        
            target_csv = None
            if save_to_latest_clicked:
                target_csv = latest_dataset_csv
            elif save_to_selected_clicked:
                target_csv = selected_dataset_csv
        
            if target_csv:
                if not st.session_state["good_zones"]:
                    st.error("No zones to save. Add at least one zone first.")
                else:
                    dataset_path = os.path.join(DATASET_DIR, target_csv)
                    if not os.path.exists(dataset_path):
                        st.error(f"Dataset CSV not found: {dataset_path}")
                    else:
                        try:
                            df_params = _cached_read_csv(
                                dataset_path,
                                keep_default_na=False,
                                file_mtime=_mtime(dataset_path),
                            )
                        except Exception as e:
                            st.error(f"Failed reading dataset CSV: {e}")
                            df_params = None
        
                        steps = []
                        if df_params is not None:
                            try:
                                steps = _parse_steps(df_params)
                            except Exception:
                                steps = []
        
                        length_col = _choose_length_col(df_plot)
                        zones_for_save = st.session_state["good_zones"]
                        if length_col:
                            zones_for_save = reorder_zones_by_spool_end(df_plot, x_axis_slice, zones_for_save, length_col)
        
                        # ✅ Save zones + FULL log stats (fixed)
                        rows_to_save = build_zone_save_rows(
                            log_file_path=log_path,
                            x_axis_display=x_axis_display,
                            x_axis_slice=x_axis_slice,
                            df_all=df_all,
                            df_plot=df_plot,
                            zones=zones_for_save,
                            exclude_cols=[],
                        )
        
                        # zone lengths (still computed on df_plot)
                        zones_info, length_col_name = ([], "")
                        try:
                            zones_info, length_col_name = zone_lengths_from_log_km(df_plot, x_axis_slice, zones_for_save)
                        except Exception as e:
                            rows_to_save.append({"Parameter Name": "T&M Length Error", "Value": str(e), "Units": ""})
        
                        rows_to_save += [_blank(), _sec("WINDER & LENGTH"), _blank()]
        
                        selected_drum_val = ""
                        if df_params is not None:
                            selected_drum_val = _extract_selected_drum_from_dataset_df(df_params)
                        if selected_drum_val:
                            rows_to_save.append({"Parameter Name": "Drum | Selected", "Value": str(selected_drum_val), "Units": ""})
        
                        rows_to_save += [_blank()]
        
                        try:
                            if steps:
                                rows_to_save += build_tm_rows_from_steps_allocate_only(
                                    dataset_csv_name=target_csv,
                                    steps=steps,
                                    zones_info=zones_info,
                                    length_col_name=length_col_name or "",
                                )
                            else:
                                rows_to_save += build_tm_instruction_rows_auto_from_good_zones(
                                    filtered_df=df_plot,
                                    x_axis=x_axis_slice,
                                    good_zones=zones_for_save,
                                    length_col_name=length_col_name or None,
                                    dataset_csv_name=os.path.basename(target_csv),
                                )
                        except Exception as e:
                            rows_to_save.append({"Parameter Name": "T&M Instructions Error", "Value": str(e), "Units": ""})
        
                        with st.expander("Preview what will be written", expanded=False):
                            st.dataframe(pd.DataFrame(rows_to_save), use_container_width=True, hide_index=True)
        
                        try:
                            append_rows_to_dataset_csv(target_csv, rows_to_save)
                            st.success(f"✅ Saved zones + T&M instructions to: {target_csv}")
                        except Exception as e:
                            st.error(f"Failed saving to dataset CSV: {e}")
        
        # ==========================================================
        # Math Lab (kept minimal)
        # ==========================================================
        st.markdown("---")
        show_math_lab = st.toggle("Open Math Lab (advanced)", value=False, key="dash_open_math_lab")
        if show_math_lab:
            st.subheader("A) f(x,y) vs time")
            math_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(math_numeric_cols) < 1:
                st.info("No numeric columns found in this log.")
            else:
                m1, m2, m3 = st.columns([1, 1, 2])
                with m1:
                    math_x_col = st.selectbox("Math X column", math_numeric_cols, key="dash_math_x_col")
                with m2:
                    math_y_col = st.selectbox("Math Y column (optional)", ["None"] + math_numeric_cols, key="dash_math_y_col")
                with m3:
                    default_expr = "x ** y" if math_y_col != "None" else "x"
                    math_expr = st.text_input(
                        "Expression (use x, y and np)",
                        value=st.session_state.get("dash_math_expr", default_expr),
                        key="dash_math_expr",
                    )
                    st.caption("Examples: `x**y`, `x*y`, `np.log(x)`, `np.sqrt(x+y)`")
        
                math_df = df.copy()
                math_df[x_axis_display] = df_work[x_axis_display]
                x_arr = pd.to_numeric(math_df[math_x_col], errors="coerce").to_numpy(dtype=float)
                y_arr = None if (math_y_col == "None") else pd.to_numeric(math_df[math_y_col], errors="coerce").to_numpy(dtype=float)
        
                safe_env = {"x": x_arr, "y": y_arr, "np": np}
                try:
                    math_res = eval(math_expr, {"__builtins__": {}}, safe_env)
                    math_res = np.asarray(math_res, dtype=float)
                    if math_res.shape[0] != len(math_df):
                        st.error("Expression must return an array with the same length as the log.")
                    else:
                        math_df["__math_result__"] = math_res
                        math_plot_df = math_df.dropna(subset=[x_axis_display, "__math_result__"]).sort_values(by=x_axis_display)
                        fig_math = px.line(math_plot_df, x=x_axis_display, y="__math_result__", markers=False, title=f"Math Lab: f(x,y) vs {x_axis_display}")
                        st.plotly_chart(fig_math, use_container_width=True)
                except Exception as e:
                    st.error(f"Math Lab error: {e}")
        else:
            st.caption("Math Lab folded.")
        # ------------------ Order Finalize Tab ------------------
