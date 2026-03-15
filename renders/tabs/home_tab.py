def render_home_tab(
    P,
    image_base64,
    STATUS_COL,
    STATUS_UPDATED_COL,
    FAILED_REASON_COL,
    parse_dt_safe,
    now_str,
    safe_str,
    render_home_draw_orders_overview,
    render_done_home_section,
    render_schedule_home_minimal,
    render_parts_orders_home_all,
):
    import os
    import pandas as pd
    import streamlit as st
    import streamlit.components.v1 as components

    st.markdown('<div style="height: 42px;"></div>', unsafe_allow_html=True)
    st.title("️ Tower Management Software")

    # =========================================================
    # 🎨 CSS (yours + small dialog polish)
    # =========================================================
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{image_base64}") no-repeat 58% 34% fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
              radial-gradient(1200px 620px at 52% 38%, rgba(255,255,255,0.00), rgba(0,0,0,0.18) 72%, rgba(0,0,0,0.28) 100%),
              linear-gradient(to bottom, rgba(4,10,20,0.28) 0%, rgba(4,10,20,0.08) 22%, rgba(4,10,20,0.08) 72%, rgba(4,10,20,0.34) 100%);
        }}
        [data-testid="stAppViewContainer"] .main {{
            position: relative;
            z-index: 1;
        }}
        [data-testid="stMarkdownContainer"] h1 {{
            text-shadow:
                0 0 16px rgba(122, 210, 255, 0.55),
                0 0 34px rgba(62, 156, 255, 0.35),
                0 4px 18px rgba(0,0,0,0.45);
            letter-spacing: 0.2px;
            text-align: center;
        }}
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            text-shadow: 0 1px 8px rgba(0, 0, 0, 0.45);
        }}
        .css-1aumxhk {{ background-color: rgba(20, 20, 20, 0.90) !important; }}
        div[data-testid="stDialog"] {{
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # ❌ FAILED → AUTO BACK TO PENDING AFTER 4 DAYS
    # =========================================================
    ORDERS_FILE = P.orders_csv

    def _mtime(path: str) -> float:
        try:
            return float(os.path.getmtime(path))
        except Exception:
            return 0.0

    @st.cache_data(show_spinner=False)
    def _read_orders_cached(path: str, file_mtime: float):
        return pd.read_csv(path)

    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        if STATUS_COL not in df.columns:
            df[STATUS_COL] = "Pending"
        if STATUS_UPDATED_COL not in df.columns:
            df[STATUS_UPDATED_COL] = ""
        if FAILED_REASON_COL not in df.columns:
            df[FAILED_REASON_COL] = ""
        return df

    def auto_move_failed_to_pending(days: int = 4):
        if not os.path.exists(ORDERS_FILE):
            return

        try:
            df = _read_orders_cached(ORDERS_FILE, _mtime(ORDERS_FILE))
        except Exception:
            return

        df = _ensure_cols(df)
        if df.empty:
            return

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days)
        changed = False

        for i in range(len(df)):
            if str(df.at[i, STATUS_COL]).strip().lower() != "failed":
                continue

            t = parse_dt_safe(df.at[i, STATUS_UPDATED_COL])

            # stamp missing timestamps so the 4-day timer works
            if t is None:
                df.at[i, STATUS_UPDATED_COL] = now_str()
                changed = True
                continue

            if t < cutoff:
                df.at[i, STATUS_COL] = "Pending"
                df.at[i, STATUS_UPDATED_COL] = now_str()
                changed = True

        if changed:
            df.to_csv(ORDERS_FILE, index=False)


    # =========================================================
    # 🚨 CRITICAL OPEN FAULTS (Home indicator)
    # - Reads maintenance/faults_log.csv
    # - Counts severity == "critical" AND not closed
    # =========================================================
    FAULTS_CSV = os.path.join(P.maintenance_dir, "faults_log.csv")


    @st.cache_data(show_spinner=False, ttl=20)
    def compute_open_critical_faults(faults_csv: str) -> int:
        if not os.path.isfile(faults_csv):
            return 0
        try:
            df = pd.read_csv(faults_csv)
        except Exception:
            return 0
        if df.empty:
            return 0

        # normalize columns
        cols = {c.lower().strip(): c for c in df.columns}
        sev_col = cols.get("fault_severity", None)
        if not sev_col:
            return 0

        # Optional "Status" / "Closed" support (if you add it later)
        status_col = cols.get("fault_status", None)
        closed_col = cols.get("fault_closed", None)

        sev = df[sev_col].astype(str).str.strip().str.lower()

        # If no status info exists → treat everything as open
        is_open = pd.Series(True, index=df.index)

        if status_col:
            stt = df[status_col].astype(str).str.strip().str.lower()
            is_open = ~stt.isin(["closed", "done", "resolved", "fixed"])
        elif closed_col:
            # supports True/False or yes/no
            cl = df[closed_col].astype(str).str.strip().str.lower()
            is_open = ~cl.isin(["true", "1", "yes", "y", "closed"])

        return int((sev == "critical")[is_open].sum())

    @st.cache_data(show_spinner=False, ttl=10)
    def compute_maintenance_in_progress() -> int:
        # Primary signal: lifecycle state.
        state_file = os.path.join(P.maintenance_dir, "maintenance_task_state.csv")
        in_progress = 0
        try:
            if os.path.isfile(state_file):
                sdf = pd.read_csv(state_file, keep_default_na=False)
                if "state" in sdf.columns:
                    in_progress = int(sdf["state"].astype(str).str.upper().eq("IN_PROGRESS").sum())
        except Exception:
            in_progress = 0

        # Fallback signal: currently active maintenance indicator.
        try:
            import json
            if os.path.isfile(P.activity_indicator_json):
                with open(P.activity_indicator_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if bool(payload.get("active", False)) and str(payload.get("activity_type", "")).strip().lower() == "maintenance":
                    in_progress = max(in_progress, 1)
        except Exception:
            pass
        return int(max(0, in_progress))

    # =========================================================
    # ❌ FAILED (last 4 days) — compact list + POPUP reason
    # =========================================================
    def render_failed_home_popup(days_visible: int = 4, show_header: bool = True):
        if show_header:
            st.subheader("❌ Failed (last 4 days)")
        st.markdown(
            """
            <style>
            .fancy-section-title {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 12px;
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.14);
                background: rgba(7, 12, 20, 0.54);
                margin-bottom: 10px;
                transition: transform 260ms ease, box-shadow 260ms ease, border-color 260ms ease;
            }
            .fancy-section-title:hover {
                transform: translateY(-3px) scale(1.015);
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.34);
            }
            .fancy-orb {
                width: 18px;
                height: 18px;
                border-radius: 999px;
                position: relative;
                box-shadow: 0 0 0 2px rgba(255,255,255,0.15) inset;
            }
            .fancy-orb::before {
                content: "";
                position: absolute;
                inset: -6px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.22);
                transform: scale(0.9);
                transition: transform 260ms ease, opacity 260ms ease;
                opacity: 0.75;
            }
            .fancy-section-title:hover .fancy-orb::before {
                transform: scale(1.14);
                opacity: 1;
            }
            .fancy-title-text {
                font-size: 1.05rem;
                font-weight: 800;
                color: rgba(255,255,255,0.96);
                letter-spacing: 0.2px;
            }
            .failed-card {
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(35, 12, 12, 0.56);
                border-radius: 14px;
                padding: 12px 14px 10px 14px;
                margin-bottom: 10px;
                transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
                will-change: transform;
            }
            .failed-card:hover {
                transform: translateY(-4px) scale(1.01);
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.32);
                border-color: rgba(255, 110, 110, 0.38);
            }
            .failed-main {
                font-size: 1.04rem;
                font-weight: 800;
                color: rgba(255,255,255,0.97);
            }
            .failed-sub {
                color: rgba(255,255,255,0.75);
                font-size: 0.80rem;
                margin-top: 2px;
                line-height: 1.35;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if not os.path.exists(ORDERS_FILE):
            st.info("No orders file found.")
            return

        try:
            df = _read_orders_cached(ORDERS_FILE, _mtime(ORDERS_FILE))
        except Exception as e:
            st.error(f"Failed to read {ORDERS_FILE}: {e}")
            return

        df = _ensure_cols(df)

        if df.empty:
            st.info("No orders.")
            return

        failed = df[df[STATUS_COL].astype(str).str.strip().str.lower().eq("failed")].copy()
        if failed.empty:
            st.success("No Failed orders 👍")
            return

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days_visible)

        failed["_dt"] = failed[STATUS_UPDATED_COL].apply(parse_dt_safe)
        failed["_dt"] = failed["_dt"].fillna(now)
        failed = failed[failed["_dt"] >= cutoff].copy().sort_values("_dt", ascending=False)

        if failed.empty:
            st.info("No recent Failed orders.")
            return

        def _open_failed_dialog(title: str, reason: str, updated: str, extra_lines: list):
            @st.dialog(title)
            def _dlg():
                if reason:
                    st.error(reason)
                else:
                    st.info("No failed description recorded.")

                if updated:
                    st.caption(f"Updated: {updated}")

                if extra_lines:
                    st.markdown("**Info**")
                    for line in extra_lines:
                        if line:
                            st.write(f"• {line}")

            _dlg()

        for i, (_, row) in enumerate(failed.iterrows()):
            oid = safe_str(row.get("Order ID"))
            pf = safe_str(row.get("Preform Number"))
            ftype = safe_str(row.get("Fiber Type"))
            proj = safe_str(row.get("Fiber Project"))
            updated = safe_str(row.get(STATUS_UPDATED_COL))
            reason = safe_str(row.get(FAILED_REASON_COL))

            left = " | ".join([p for p in [f"#{oid}" if oid else "", ftype if ftype else ""] if p])

            extra = []
            if "Required Length (m) (for T&M+costumer)" in failed.columns:
                val = safe_str(row.get("Required Length (m) (for T&M+costumer)"))
                if val:
                    extra.append(f"Required Length: {val} m")
            elif "Required Length (m)" in failed.columns:
                val = safe_str(row.get("Required Length (m)"))
                if val:
                    extra.append(f"Required Length: {val} m")

            if "Priority" in failed.columns:
                val = safe_str(row.get("Priority"))
                if val:
                    extra.append(f"Priority: {val}")

            if "Notes" in failed.columns:
                val = safe_str(row.get("Notes"))
                if val:
                    extra.append(f"Notes: {val}")

            st.markdown("<div class='failed-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([3.2, 1.2], vertical_alignment="center")
            with c1:
                st.markdown(
                    f"<div class='failed-main'>Project: {proj or '—'} &nbsp;|&nbsp; Preform: {pf or '—'}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='failed-sub'>{left if left else 'Failed Order'}</div>",
                    unsafe_allow_html=True,
                )
                if updated:
                    st.markdown(f"<div class='failed-sub'>Updated: {updated}</div>", unsafe_allow_html=True)
            with c2:
                btn_key = f"failed_reason_btn_{i}_{oid}_{pf}"
                if st.button("View reason", key=btn_key, use_container_width=True):
                    dlg_title = left if left else "Failed Order"
                    _open_failed_dialog(
                        title=f"❌ Failed: {dlg_title}",
                        reason=reason,
                        updated=updated,
                        extra_lines=extra
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # 🔁 AUTO CLEANUP FIRST
    # =========================================================
    auto_move_failed_to_pending(days=4)
    
    def _section_gap():
        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

    def _fancy_section_title(label: str, orb_color: str):
        st.markdown(
            f"""
            <div class="fancy-section-title">
                <span class="fancy-orb" style="background:{orb_color}; box-shadow: 0 0 12px {orb_color};"></span>
                <span class="fancy-title-text">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_done_failed_compact(days_visible: int = 4):
        if not os.path.exists(ORDERS_FILE):
            st.info("No orders file found.")
            return

        try:
            df = _read_orders_cached(ORDERS_FILE, _mtime(ORDERS_FILE))
        except Exception as e:
            st.error(f"Failed to read {ORDERS_FILE}: {e}")
            return

        df = _ensure_cols(df)
        if df.empty:
            st.info("No orders.")
            return

        # compact styling
        st.markdown(
            """
            <style>
            .home-compact-card {
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 12px;
                padding: 10px 12px;
                background: rgba(7,12,20,0.58);
            }
            .home-compact-h {
                font-size: 1.02rem;
                font-weight: 800;
                margin-bottom: 8px;
                color: rgba(255,255,255,0.96);
            }
            .home-done-mini {
                border: 1px solid rgba(130, 255, 178, 0.28);
                border-radius: 10px;
                padding: 8px 10px;
                margin-bottom: 8px;
                background: rgba(20, 50, 34, 0.46);
            }
            .home-done-mini-main {
                font-size: 0.93rem;
                font-weight: 760;
                color: rgba(245,255,248,0.98);
                margin-bottom: 6px;
            }
            .home-done-tag {
                display: inline-block;
                margin-right: 6px;
                margin-bottom: 4px;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 0.76rem;
                border: 1px solid rgba(255,255,255,0.18);
                background: rgba(255,255,255,0.08);
                color: rgba(245,245,245,0.96);
            }
            .home-failed-mini {
                border: 1px solid rgba(255, 118, 118, 0.34);
                border-radius: 10px;
                padding: 8px 10px;
                margin-bottom: 8px;
                background: rgba(62, 24, 24, 0.48);
            }
            .home-failed-mini-main {
                font-size: 0.93rem;
                font-weight: 760;
                color: rgba(255,245,245,0.98);
                margin-bottom: 6px;
            }
            .home-failed-tag {
                display: inline-block;
                margin-right: 6px;
                margin-bottom: 4px;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 0.76rem;
                border: 1px solid rgba(255,200,200,0.22);
                background: rgba(255,120,120,0.10);
                color: rgba(255,232,232,0.98);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        done_df = df[df[STATUS_COL].astype(str).str.strip().str.lower().eq("done")].copy()
        failed_df = df[df[STATUS_COL].astype(str).str.strip().str.lower().eq("failed")].copy()

        # parse timestamps used for recency
        done_df["_ts"] = pd.to_datetime(done_df.get(STATUS_UPDATED_COL, ""), errors="coerce")
        failed_df["_ts"] = pd.to_datetime(failed_df.get(STATUS_UPDATED_COL, ""), errors="coerce")

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days_visible)

        done_recent = done_df[(done_df["_ts"].isna()) | (done_df["_ts"] >= cutoff)].copy()
        failed_recent = failed_df[(failed_df["_ts"].isna()) | (failed_df["_ts"] >= cutoff)].copy()

        done_recent = done_recent.sort_values("_ts", ascending=False).head(6)
        failed_recent = failed_recent.sort_values("_ts", ascending=False).head(6)

        c_done, c_failed = st.columns(2, gap="medium")
        with c_done:
            st.markdown("<div class='home-compact-card'>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='home-compact-h'>✅ DONE (last {days_visible} days): {len(done_recent)}</div>",
                unsafe_allow_html=True,
            )
            if done_recent.empty:
                st.caption("No recent done draws.")
            else:
                for _, r in done_recent.head(5).iterrows():
                    project = str(r.get("Fiber Project", "")).strip() or "—"
                    preform = str(r.get("Preform Number", "")).strip() or "—"
                    when = r.get("_ts")
                    when_s = when.strftime("%Y-%m-%d %H:%M") if pd.notna(when) else "—"
                    st.markdown(
                        f"""
                        <div class="home-done-mini">
                            <div class="home-done-mini-main">Project: {project}</div>
                            <span class="home-done-tag">Preform: {preform}</span>
                            <span class="home-done-tag">Done: {when_s}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        with c_failed:
            st.markdown("<div class='home-compact-card'>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='home-compact-h'>❌ FAILED (last {days_visible} days): {len(failed_recent)}</div>",
                unsafe_allow_html=True,
            )
            if failed_recent.empty:
                st.caption("No recent failed draws.")
            else:
                for _, r in failed_recent.head(5).iterrows():
                    project = str(r.get("Fiber Project", "")).strip() or "—"
                    preform = str(r.get("Preform Number", "")).strip() or "—"
                    reason = str(r.get(FAILED_REASON_COL, "")).strip() or "No reason"
                    reason = reason[:72]
                    st.markdown(
                        f"""
                        <div class="home-failed-mini">
                            <div class="home-failed-mini-main">Project: {project}</div>
                            <span class="home-failed-tag">Preform: {preform}</span>
                            <span class="home-failed-tag">Reason: {reason}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .home-manifest-nav {
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 82px;
            margin-top: 0;
            padding: 10px 8px 12px 8px;
            border-radius: 18px;
            background: transparent;
            box-shadow: none;
            position: relative;
        }
        .home-manifest-nav::after {
            content: "";
            position: absolute;
            left: -10px;
            top: 28px;
            bottom: 28px;
            width: 92px;
            border: 1px solid rgba(140, 220, 255, 0.42);
            border-right: none;
            border-radius: 999px 0 0 999px;
            pointer-events: none;
        }
        .home-manifest-nav > label {
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease, background 220ms ease, filter 220ms ease;
            border-radius: 999px;
            padding: 10px 14px 10px 30px !important;
            border: 1px solid rgba(255,255,255,0.20);
            background: linear-gradient(145deg, rgba(8, 16, 28, 0.62), rgba(22, 34, 52, 0.42));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 16px rgba(0,0,0,0.22);
            overflow: visible;
            transform-origin: left center;
            z-index: 2;
            backdrop-filter: blur(3px);
            width: 100%;
            max-width: 262px;
            margin-right: 10px;
        }
        /* Internal hidden option (last item = __none__). */
        .home-manifest-nav > label:last-of-type {
            display: none !important;
        }
        /* Hide Streamlit's native radio/check UI inside chips. */
        .home-manifest-nav > label [role="radio"],
        .home-manifest-nav > label input[type="radio"] {
            display: none !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .home-manifest-nav > label::before {
            content: "";
            position: absolute;
            left: 8px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, rgba(200, 243, 255, 0.95), rgba(96, 172, 240, 0.58));
            border: 1px solid rgba(168,232,255,0.38);
            box-shadow: 0 0 11px rgba(92,176,255,0.42);
            transition: transform 220ms ease, box-shadow 220ms ease, background 220ms ease;
        }
        .home-manifest-nav > label:nth-of-type(1) { margin-left: 74px; }
        .home-manifest-nav > label:nth-of-type(2) { margin-left: 48px; }
        .home-manifest-nav > label:nth-of-type(3) { margin-left: 14px; }
        .home-manifest-nav > label:nth-of-type(4) { margin-left: 14px; }
        .home-manifest-nav > label:nth-of-type(5) { margin-left: 14px; }
        .home-nav-wrap {
            min-height: auto;
            display: block;
            padding-top: clamp(36px, 9vh, 108px);
        }
        .home-hidden-option {
            position: absolute !important;
            left: -10000px !important;
            top: -10000px !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            overflow: hidden !important;
        }
        .home-panel-wrap {
            min-height: auto;
            display: block;
            padding-top: clamp(96px, 14vh, 170px);
        }
        .home-manifest-nav > label:hover {
            filter: saturate(1.12);
            box-shadow: 0 16px 32px rgba(0,0,0,0.36), 0 0 22px rgba(132,214,255,0.20);
            border-color: rgba(170,238,255,0.68);
            background: linear-gradient(145deg, rgba(16, 30, 52, 0.78), rgba(22, 40, 66, 0.58));
            z-index: 4;
        }
        .home-manifest-nav > label:nth-of-type(1):hover { transform: translateY(-4px) scale(1.10) rotate(-0.6deg); }
        .home-manifest-nav > label:nth-of-type(2):hover { transform: translateY(-4px) scale(1.10) rotate(-0.9deg); }
        .home-manifest-nav > label:nth-of-type(3):hover { transform: translateY(-4px) scale(1.10) rotate(0.8deg); }
        .home-manifest-nav > label:nth-of-type(4):hover { transform: translateY(-4px) scale(1.10) rotate(-0.7deg); }
        .home-manifest-nav > label:nth-of-type(5):hover { transform: translateY(-4px) scale(1.10) rotate(0.7deg); }
        .home-manifest-nav > label:hover::before {
            transform: translateY(-50%) scale(1.15);
            box-shadow: 0 0 18px rgba(110,200,255,0.42);
            background: radial-gradient(circle at 30% 30%, rgba(220, 249, 255, 0.98), rgba(102, 182, 255, 0.66));
        }
        .home-manifest-nav > label.home-chip-active {
            border-color: rgba(160, 231, 255, 0.88);
            background: linear-gradient(145deg, rgba(30, 66, 98, 0.82), rgba(28, 48, 78, 0.70));
            box-shadow: 0 16px 30px rgba(35, 138, 198, 0.34), inset 0 1px 0 rgba(255,255,255,0.24);
            filter: saturate(1.18);
        }
        .home-manifest-nav > label.home-chip-active::before {
            background: radial-gradient(circle at 30% 30%, rgba(232, 252, 255, 1), rgba(124, 201, 255, 0.76));
            border-color: rgba(194,246,255,0.72);
            box-shadow: 0 0 22px rgba(126,214,255,0.56);
        }
        .home-panel-shell {
            position: relative;
            margin-top: 0;
            padding: 18px 16px 16px 16px;
            border-radius: 18px;
            border: 1px solid rgba(154, 226, 255, 0.22);
            background:
                linear-gradient(160deg, rgba(7, 18, 33, 0.64), rgba(7, 14, 25, 0.50));
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(7px);
        }
        .home-panel-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(4, 10, 18, 0.22), rgba(4, 10, 18, 0.30));
            pointer-events: none;
            z-index: 0;
        }
        .home-panel-shell > * {
            position: relative;
            z-index: 1;
        }
        .home-panel-shell [data-testid="stMarkdownContainer"] h2,
        .home-panel-shell [data-testid="stMarkdownContainer"] h3,
        .home-panel-shell [data-testid="stMarkdownContainer"] p,
        .home-panel-shell [data-testid="stMarkdownContainer"] li {
            text-shadow: 0 1px 8px rgba(0, 0, 0, 0.44);
        }
        .home-panel-shell div[data-testid="stAlert"],
        .home-panel-shell div[data-testid="stMetric"],
        .home-panel-shell div[data-testid="stDataFrame"],
        .home-panel-shell div[data-testid="stPlotlyChart"] {
            border-radius: 12px;
            background: rgba(6, 14, 24, 0.30);
            backdrop-filter: blur(3px);
            -webkit-backdrop-filter: blur(3px);
        }
        .home-maint-shell {
            border: 1px solid rgba(146, 220, 255, 0.26);
            border-radius: 14px;
            padding: 10px 12px;
            background: linear-gradient(160deg, rgba(6, 14, 24, 0.68), rgba(6, 12, 22, 0.56));
            box-shadow: 0 10px 22px rgba(0,0,0,0.30);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        }
        .home-maint-shell [data-testid="stMarkdownContainer"] h3,
        .home-maint-shell [data-testid="stMarkdownContainer"] p,
        .home-maint-shell [data-testid="stCaptionContainer"] {
            color: rgba(236, 248, 255, 0.98) !important;
            text-shadow: 0 1px 10px rgba(0,0,0,0.55);
        }
        .home-maint-shell [data-testid="stMetric"] {
            background: rgba(6, 14, 24, 0.52) !important;
            border: 1px solid rgba(140, 220, 255, 0.24);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .home-maint-shell [data-testid="stMetricLabel"] p {
            color: rgba(220, 240, 255, 0.98) !important;
            text-shadow: 0 1px 8px rgba(0,0,0,0.55);
        }
        .home-maint-shell [data-testid="stMetricValue"] {
            color: rgba(245, 252, 255, 1) !important;
            text-shadow: 0 1px 10px rgba(0,0,0,0.60);
        }
        .home-maint-shell [data-testid="stAlert"] {
            background: rgba(10, 18, 30, 0.68) !important;
            border: 1px solid rgba(140, 220, 255, 0.20) !important;
        }
        .home-maint-shell [data-testid="stAlert"] p {
            color: rgba(236, 248, 255, 0.98) !important;
            text-shadow: 0 1px 8px rgba(0,0,0,0.55);
        }
        .home-faults-block {
            margin-top: 8px;
            border: 1px solid rgba(150, 220, 255, 0.24);
            border-radius: 12px;
            padding: 8px 10px 10px 10px;
            background: linear-gradient(160deg, rgba(7, 14, 24, 0.72), rgba(6, 12, 22, 0.62));
            box-shadow: 0 8px 18px rgba(0,0,0,0.30);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }
        .home-faults-title {
            font-size: 1.85rem;
            font-weight: 860;
            color: rgba(246, 252, 255, 0.99);
            text-shadow: 0 1px 10px rgba(0,0,0,0.55);
            margin: 2px 0 10px 0;
        }
        .home-faults-grid {
            display: grid;
            grid-template-columns: 1.0fr 1.0fr 1.0fr 1.6fr;
            gap: 10px;
        }
        .home-faults-card {
            border: 1px solid rgba(142, 214, 255, 0.24);
            border-radius: 10px;
            padding: 8px 10px;
            background: rgba(8, 16, 28, 0.66);
        }
        .home-faults-k {
            font-size: 0.82rem;
            color: rgba(196, 228, 250, 0.95);
            margin-bottom: 2px;
        }
        .home-faults-v {
            font-size: 2rem;
            font-weight: 900;
            color: rgba(246, 252, 255, 1);
            text-shadow: 0 1px 10px rgba(0,0,0,0.55);
            line-height: 1.1;
        }
        .home-faults-state-ok {
            color: rgba(136, 248, 178, 0.98);
            font-weight: 800;
        }
        .home-faults-state-warn {
            color: rgba(255, 210, 120, 0.98);
            font-weight: 800;
        }
        .home-faults-tip {
            color: rgba(225, 243, 255, 0.95);
            font-size: 0.90rem;
            line-height: 1.35;
        }
        @media (max-width: 1100px) {
            .home-faults-grid {
                grid-template-columns: 1fr;
            }
        }
        /* When viewport is tighter (e.g. sidebar open), keep nav chips further left. */
        @media (max-width: 1500px) {
            .home-manifest-nav > label {
                max-width: 236px;
            }
            .home-manifest-nav > label:nth-of-type(1) { margin-left: 32px; }
            .home-manifest-nav > label:nth-of-type(2) { margin-left: 10px; }
            .home-manifest-nav > label:nth-of-type(3) { margin-left: 0; }
            .home-manifest-nav > label:nth-of-type(4) { margin-left: 0; }
            .home-manifest-nav > label:nth-of-type(5) { margin-left: 0; }
            .home-manifest-nav > label:nth-of-type(1):hover { transform: translateY(-4px) scale(1.06) rotate(-0.5deg); }
            .home-manifest-nav > label:nth-of-type(2):hover { transform: translateY(-4px) scale(1.06) rotate(-0.7deg); }
            .home-manifest-nav > label:nth-of-type(3):hover { transform: translateY(-4px) scale(1.06) rotate(0.6deg); }
            .home-manifest-nav > label:nth-of-type(4):hover { transform: translateY(-4px) scale(1.06) rotate(-0.6deg); }
            .home-manifest-nav > label:nth-of-type(5):hover { transform: translateY(-4px) scale(1.06) rotate(0.6deg); }
        }
        @media (max-width: 900px) {
            .home-manifest-nav {
                width: 100%;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                justify-content: center;
                margin-bottom: 8px;
                padding: 6px 0 0 0;
                background: transparent;
                box-shadow: none;
            }
            .home-manifest-nav::after {
                display: none;
            }
            .home-manifest-nav > label {
                position: relative;
                left: unset !important;
                top: unset !important;
                transform: none !important;
                margin-left: 0 !important;
                width: auto;
                max-width: 100%;
            }
            .home-manifest-nav > label:hover {
                transform: translateY(-2px) scale(1.05) !important;
            }
            .home-nav-wrap,
            .home-panel-wrap {
                min-height: auto;
                display: block;
                padding-top: 0;
            }
            .home-panel-shell {
                margin-top: 10px;
            }
        }
        .home-panel-fade {
            animation: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    nav_col, panel_col = st.columns([1.22, 2.78], gap="large")
    if "home_focus_panel" not in st.session_state:
        st.session_state["home_focus_panel"] = "__none__"
    with nav_col:
        st.markdown('<div class="home-nav-wrap">', unsafe_allow_html=True)
        selected_panel = st.radio(
            "Home Sections",
            [
                "🚀 Draws Monitor",
                "✅ Done + ❌ Failed",
                "📅 Schedule",
                "🧰 Maintenance + 🚨 Faults",
                "🧩 Parts Orders",
                "__none__",
            ],
            horizontal=False,
            key="home_focus_panel",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    components.html(
        """
        <script>
        (function() {
            if (window.__homeHoverBootTimer) {
              clearInterval(window.__homeHoverBootTimer);
              window.__homeHoverBootTimer = null;
            }
            const expected = [
            "🚀 Draws Monitor",
            "✅ Done + ❌ Failed",
            "📅 Schedule",
            "🧰 Maintenance + 🚨 Faults",
            "🧩 Parts Orders"
          ];

          function bindHoverSwitch() {
            const root = window.parent?.document || document;
            const groups = root.querySelectorAll('div[role="radiogroup"]');
            const hoveredLabel = (group) =>
              Array.from(group.querySelectorAll("label")).find((l) => {
                return !l.classList.contains("home-hidden-option") && l.matches(":hover");
              }) || null;
            const syncHoverIndicator = (group) => {
              const labels = Array.from(group.querySelectorAll("label"));
              const hovered = hoveredLabel(group);
              labels.forEach((l) => {
                if (l.classList.contains("home-hidden-option")) return;
                if (hovered && l === hovered) l.classList.add("home-chip-active");
                else l.classList.remove("home-chip-active");
              });
              return hovered;
            };
            for (const group of groups) {
              const labels = Array.from(group.querySelectorAll("label"));
              if (!labels.length) continue;
              const labelTexts = labels.map(l => (l.textContent || "").trim());
              const matches = expected.filter(e => labelTexts.includes(e)).length;
              if (matches < 4) continue;
              const noneLabel = labels[labels.length - 1];
              if (noneLabel) noneLabel.classList.add("home-hidden-option");
              if (group.dataset.homeHoverBound === "1") {
                syncHoverIndicator(group);
                return true;
              }
              group.classList.add("home-manifest-nav");
              group.addEventListener("mouseleave", () => {
                const hovered = syncHoverIndicator(group);
                if (!hovered && noneLabel) {
                  const noneInput = noneLabel.querySelector('input[type="radio"]');
                  if (!noneInput || !noneInput.checked) {
                    try { noneLabel.click(); } catch (e) {}
                  }
                }
              });

              labels.forEach((label) => {
                if (label.classList.contains("home-hidden-option")) return;
                label.addEventListener("mouseenter", () => {
                  syncHoverIndicator(group);
                  try { label.click(); } catch (e) {}
                });
                label.addEventListener("mouseleave", () => {
                  const hovered = syncHoverIndicator(group);
                  if (!hovered && noneLabel) {
                    const noneInput = noneLabel.querySelector('input[type="radio"]');
                    if (!noneInput || !noneInput.checked) {
                      try { noneLabel.click(); } catch (e) {}
                    }
                  }
                });
              });
              group.dataset.homeHoverBound = "1";
              syncHoverIndicator(group);
              return true;
            }
            return false;
          }
          let attempts = 0;
          const MAX_ATTEMPTS = 40; // ~8s total
          function boot() {
            attempts += 1;
            const done = bindHoverSwitch();
            if (done || attempts >= MAX_ATTEMPTS) {
              if (window.__homeHoverBootTimer) {
                clearInterval(window.__homeHoverBootTimer);
                window.__homeHoverBootTimer = null;
              }
            }
          }
          window.__homeHoverBootTimer = setInterval(boot, 200);
          boot();
        })();
        </script>
        """,
        height=0,
        width=0,
    )
    _section_gap()

    # =========================================================
    # 5) MAINTENANCE OVERVIEW (unchanged below)
    # =========================================================
    @st.cache_data(show_spinner=False, ttl=20)
    def compute_maintenance_counts_for_home(
            maint_folder: str,
            dataset_dir: str,
            base_dir: str = None,
    ):
        # (your existing function unchanged)
        import os
        import json
        import datetime as dt
        import pandas as pd
        import numpy as np
        from helpers.orders_io import count_dataset_draws

        base_dir = base_dir or P.root_dir

        def get_draw_orders_count() -> int:
            return int(count_dataset_draws(P.dataset_dir))

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

        def mode_norm(x: str) -> str:
            s = "" if x is None or pd.isna(x) else str(x).strip().lower()
            if s in ("draw", "draws", "draws_count", "draw_count"):
                return "draws"
            return s

        def load_state(path: str) -> dict:
            try:
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
            return {}

        state_path = os.path.join(maint_folder, "_app_state.json")
        state = load_state(state_path)

        current_date = dt.date.today()
        furnace_hours = float(state.get("furnace_hours", 0.0) or 0.0)
        uv1_hours = float(state.get("uv1_hours", 0.0) or 0.0)
        uv2_hours = float(state.get("uv2_hours", 0.0) or 0.0)
        warn_days = int(state.get("warn_days", 14) or 14)
        warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

        current_draw_count = get_draw_orders_count()

        if not os.path.isdir(maint_folder):
            return 0, 0

        files = [f for f in os.listdir(maint_folder) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        if not files:
            return 0, 0

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

        frames = []
        for fname in sorted(files):
            fpath = os.path.join(maint_folder, fname)
            try:
                raw = read_file(fpath)
                if raw is None or raw.empty:
                    continue
                dfm = normalize_df(raw)
                dfm["Source_File"] = fname
                frames.append(dfm)
            except Exception:
                continue

        if not frames:
            return 0, 0

        dfm = pd.concat(frames, ignore_index=True)

        def pick_current_hours(hours_source: str) -> float:
            hs = norm_source(hours_source)
            if hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
                return float(uv2_hours)
            if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
                return float(uv1_hours)
            return float(furnace_hours)

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

            if nh is not None and not pd.isna(nh):
                nh = float(nh)
                cur_h = float(row.get("Current_Hours_For_Task", 0.0))
                if nh < cur_h:
                    overdue = True
                elif (nh - cur_h) <= float(warn_hours):
                    due_soon = True

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

        overdue = int((dfm["Status"] == "OVERDUE").sum())
        due_soon = int((dfm["Status"] == "DUE SOON").sum())
        return overdue, due_soon

    with panel_col:
        if selected_panel != "__none__":
            st.markdown('<div class="home-panel-wrap">', unsafe_allow_html=True)
            st.markdown(
                '<div class="home-panel-shell"><div class="home-panel-fade">',
                unsafe_allow_html=True,
            )
            if selected_panel == "🚀 Draws Monitor":
                render_home_draw_orders_overview()
            elif selected_panel == "✅ Done + ❌ Failed":
                _render_done_failed_compact(days_visible=4)
            elif selected_panel == "📅 Schedule":
                render_schedule_home_minimal()
            elif selected_panel == "🧰 Maintenance + 🚨 Faults":
                st.markdown('<div class="home-maint-shell">', unsafe_allow_html=True)
                MAINT_FOLDER = P.maintenance_dir
                DATASET_DIR = P.dataset_dir

                overdue, due_soon = compute_maintenance_counts_for_home(
                    maint_folder=MAINT_FOLDER,
                    dataset_dir=DATASET_DIR,
                )
                maint_in_progress = compute_maintenance_in_progress()

                st.session_state["maint_overdue"] = overdue
                st.session_state["maint_due_soon"] = due_soon

                st.markdown(
                    f"""
                    <div class="home-faults-block">
                      <div class="home-faults-title">🧰 Maintenance Overview</div>
                      <div class="home-faults-grid">
                        <div class="home-faults-card">
                          <div class="home-faults-k">Overdue</div>
                          <div class="home-faults-v">{overdue}</div>
                        </div>
                        <div class="home-faults-card">
                          <div class="home-faults-k">Due soon</div>
                          <div class="home-faults-v">{due_soon}</div>
                        </div>
                        <div class="home-faults-card">
                          <div class="home-faults-k">In progress</div>
                          <div class="home-faults-v">{maint_in_progress}</div>
                        </div>
                        <div class="home-faults-card">
                          <div class="home-faults-k">Focus</div>
                          <div class="home-faults-tip">Open Maintenance tab to review overdue tasks and schedule actions.</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                open_critical = compute_open_critical_faults(FAULTS_CSV)
                faults_state = "No critical faults ✅" if open_critical == 0 else "Check Maintenance → Faults"
                faults_state_cls = "home-faults-state-ok" if open_critical == 0 else "home-faults-state-warn"
                faults_tip = "Tip: open 🧰 Maintenance → Faults / Incidents to review." if open_critical > 0 else "System status is clear."

                st.markdown(
                    f"""
                    <div class="home-faults-block">
                      <div class="home-faults-title">🚨 Faults Overview</div>
                      <div class="home-faults-grid">
                        <div class="home-faults-card">
                          <div class="home-faults-k">Critical open faults</div>
                          <div class="home-faults-v">{open_critical}</div>
                        </div>
                        <div class="home-faults-card">
                          <div class="home-faults-k">State</div>
                          <div class="{faults_state_cls}">{faults_state}</div>
                        </div>
                        <div class="home-faults-card">
                          <div class="home-faults-k">Guide</div>
                          <div class="home-faults-tip">{faults_tip}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            elif selected_panel == "🧩 Parts Orders":
                render_parts_orders_home_all()
            st.markdown("</div></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
