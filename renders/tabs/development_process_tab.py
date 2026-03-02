def render_development_process_tab(P):
    import os
    import json
    import pandas as pd
    import streamlit as st
    from datetime import datetime

    # =========================================================
    # ✅ Development Process (FULL TAB)
    # ✅ Wide layout (guarded)
    # ✅ Attachments: Photos / PDFs (preview) / Notebooks (.ipynb open real)
    # ✅ Notes: Markdown + LaTeX
    # ✅ Per-experiment: Preview tab is DEFAULT, all inputs moved to Edit tab
    # ✅ Fixed: dev_selected_project session_state crash on delete
    # =========================================================

    # =========================
    # Page config (WIDE) - safe guard
    # =========================
    if "_page_config_set" not in st.session_state:
        try:
            st.set_page_config(layout="wide")
        except Exception:
            pass
        st.session_state["_page_config_set"] = True

    # =========================
    # CSS (HOME-LIKE POLISH)
    # =========================
    st.markdown("""
    <style>
    /* ---------- page spacing ---------- */
    .block-container { padding-top: 2.8rem; padding-bottom: 2.0rem; }

    /* ---------- header / hero card ---------- */
    .dp-hero{
      border-radius: 22px;
      padding: 16px 18px 14px 18px;
      margin: 6px 0 10px 0;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(980px 260px at 12% -10%, rgba(0,140,255,0.18), rgba(0,0,0,0) 60%),
        radial-gradient(680px 240px at 88% 10%, rgba(0,255,180,0.09), rgba(0,0,0,0) 55%),
        linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      box-shadow: 0 14px 34px rgba(0,0,0,0.34);
    }
    .dp-hero-title{
      font-size: 1.22rem;
      font-weight: 900;
      margin: 0;
      line-height: 1.15;
      letter-spacing: -0.2px;
    }
    .dp-hero-sub{
      margin-top: 6px;
      font-size: 0.93rem;
      color: rgba(255,255,255,0.72);
    }

    /* ---------- sticky toolbar ---------- */
    .dp-sticky{
      position: sticky;
      top: 0.25rem;
      z-index: 50;
      padding-top: 6px;
      padding-bottom: 6px;
      background: linear-gradient(180deg, rgba(10,10,10,0.75), rgba(10,10,10,0.0));
      backdrop-filter: blur(6px);
    }
    .dp-toolbar{
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      box-shadow: 0 10px 28px rgba(0,0,0,0.26);
      padding: 10px 12px;
    }
    .dp-pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      color: rgba(255,255,255,0.78);
      font-size: 0.88rem;
      white-space: nowrap;
    }

    /* ---------- inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div{
      border-radius: 14px !important;
    }
    textarea, input{
      border-radius: 14px !important;
    }

    /* ---------- expanders as cards ---------- */
    div[data-testid="stExpander"] details{
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.09);
      background: rgba(255,255,255,0.02);
      box-shadow: 0 8px 22px rgba(0,0,0,0.18);
      overflow: hidden;
    }
    div[data-testid="stExpander"] details > summary{
      padding: 12px 14px !important;
    }
    div[data-testid="stExpander"] details > div{
      padding: 6px 14px 14px 14px !important;
    }

    /* ---------- buttons (clean + consistent) ---------- */
    .stButton>button{
      border-radius: 14px !important;
      height: 44px !important;
      padding: 8px 14px !important;
      white-space: nowrap !important;
    }
    .stButton>button[kind="primary"]{
      border-radius: 14px !important;
      height: 44px !important;
      padding: 8px 16px !important;
    }

    /* ---------- dataframe looks like a card ---------- */
    div[data-testid="stDataFrame"]{
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      overflow: hidden;
    }

    /* ---------- nicer dividers ---------- */
    hr{ border-color: rgba(255,255,255,0.08) !important; }

    /* segmented label tighten */
    div[data-testid="stSegmentedControl"] label p{
      font-weight: 700;
      opacity: 0.85;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # Files / folders
    # =========================
    PROJECTS_FILE = os.path.join(P.data_dir, "development_projects.csv")
    EXPERIMENTS_FILE = os.path.join(P.data_dir, "development_experiments.csv")
    UPDATES_FILE = P.experiment_updates_csv
    DATASET_DIR = P.dataset_dir
    MEDIA_ROOT = os.path.join(P.data_dir, "development_media")
    os.makedirs(P.data_dir, exist_ok=True)

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

    # =========================
    # Ensure files exist
    # =========================
    def _ensure_files():
        if not os.path.exists(PROJECTS_FILE):
            pd.DataFrame(columns=[
                "Project Name", "Project Purpose", "Target", "Created At", "Archived"
            ]).to_csv(PROJECTS_FILE, index=False)

        if not os.path.exists(EXPERIMENTS_FILE):
            pd.DataFrame(columns=[
                "Project Name",
                "Experiment Title",
                "Date",
                "Researcher",
                "Methods",
                "Purpose",
                "Observations",
                "Results",
                "Is Drawing",
                "Drawing Details",
                "Draw CSV",
                "Attachments",
                "Attachment Captions",
                "Markdown Notes",
            ]).to_csv(EXPERIMENTS_FILE, index=False)

        if not os.path.exists(UPDATES_FILE):
            pd.DataFrame(columns=[
                "Project Name", "Experiment Title", "Update Date", "Researcher", "Update Notes"
            ]).to_csv(UPDATES_FILE, index=False)

    def _ensure_columns(path, required_cols):
        df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        changed = False
        for c in required_cols:
            if c not in df.columns:
                df[c] = False if c in ["Archived", "Is Drawing"] else ""
                changed = True
        if changed:
            df.to_csv(path, index=False)

    _ensure_files()
    _ensure_columns(PROJECTS_FILE, ["Project Name", "Project Purpose", "Target", "Created At", "Archived"])
    _ensure_columns(EXPERIMENTS_FILE, [
        "Project Name", "Experiment Title", "Date", "Researcher", "Methods", "Purpose",
        "Observations", "Results", "Is Drawing", "Drawing Details", "Draw CSV",
        "Attachments", "Attachment Captions", "Markdown Notes"
    ])
    _ensure_columns(UPDATES_FILE, ["Project Name", "Experiment Title", "Update Date", "Researcher", "Update Notes"])

    # =========================
    # Data helpers
    # =========================
    def load_projects():
        df = pd.read_csv(PROJECTS_FILE)
        df["Archived"] = df.get("Archived", False)
        df["Archived"] = df["Archived"].fillna(False).astype(bool)
        return df

    def save_projects(df):
        df.to_csv(PROJECTS_FILE, index=False)

    def load_experiments():
        df = pd.read_csv(EXPERIMENTS_FILE)
        if "Is Drawing" in df.columns:
            df["Is Drawing"] = df["Is Drawing"].fillna(False).astype(bool)
        df["Attachments"] = df.get("Attachments", "").fillna("")
        df["Attachment Captions"] = df.get("Attachment Captions", "").fillna("")
        df["Markdown Notes"] = df.get("Markdown Notes", "").fillna("")
        return df

    def save_experiments(df):
        df.to_csv(EXPERIMENTS_FILE, index=False)

    def load_updates():
        return pd.read_csv(UPDATES_FILE)

    def save_updates(df):
        df.to_csv(UPDATES_FILE, index=False)

    @st.cache_data(show_spinner=False)
    def load_draw_csv(csv_path: str):
        return pd.read_csv(csv_path)

    # =========================
    # Utility helpers
    # =========================
    def _safe(s: str) -> str:
        return str(s).replace("/", "_").replace("\\", "_").replace(":", "-").strip()

    def exp_media_dir(project_name: str, exp_title: str, exp_date: str) -> str:
        d = os.path.join(MEDIA_ROOT, _safe(project_name), f"{_safe(exp_title)}__{_safe(exp_date)}")
        os.makedirs(d, exist_ok=True)
        return d

    def parse_path_list(s):
        if not isinstance(s, str) or not s.strip():
            return []
        return [x for x in s.split(";") if x.strip()]

    def join_path_list(lst):
        return ";".join(lst)

    def parse_captions(s):
        if not isinstance(s, str) or not s.strip():
            return {}
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def dump_captions(d):
        try:
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            return ""

    def list_dataset_csvs_newest_first():
        if not os.path.isdir(DATASET_DIR):
            return []
        files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
        return sorted(files, key=lambda fn: os.path.getmtime(os.path.join(DATASET_DIR, fn)), reverse=True)

    def ext_of(name: str) -> str:
        return os.path.splitext(str(name).lower())[1]

    def is_image(name: str) -> bool:
        return ext_of(name) in IMG_EXTS

    def is_pdf(path: str) -> bool:
        return str(path).lower().endswith(".pdf")

    def is_notebook(path: str) -> bool:
        return str(path).lower().endswith(".ipynb")

    def open_notebook_real(path: str):
        """
        Best-effort: open .ipynb with the OS default application.
        Works when Streamlit runs locally (PyCharm). On a server it opens on the server machine.
        """
        import os as _os, sys as _sys, subprocess as _subprocess
        p = _os.path.abspath(path)
        if not _os.path.exists(p):
            raise FileNotFoundError(p)

        if _sys.platform.startswith("darwin"):
            _subprocess.Popen(["open", p])
        elif _sys.platform.startswith("win"):
            _os.startfile(p)  # type: ignore[attr-defined]
        else:
            _subprocess.Popen(["xdg-open", p])

    def _unique_path(path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 2
        while True:
            cand = f"{base}__{i}{ext}"
            if not os.path.exists(cand):
                return cand
            i += 1

    @st.cache_data(show_spinner=False)
    def read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def render_download_file(path: str, label: str, key: str):
        if not os.path.exists(path):
            st.warning(f"Missing file: {os.path.basename(path)}")
            return
        data = read_bytes(path)
        st.download_button(
            label=label,
            data=data,
            file_name=os.path.basename(path),
            mime=None,
            key=key,
            use_container_width=True
        )

    # =========================
    # PDF preview (RENDER ONLY)
    # =========================
    @st.cache_data(show_spinner=False)
    def pdf_render_pages(path: str, max_pages: int = 1, zoom: float = 1.6):
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

    def render_pdf_preview(path: str):
        if not os.path.exists(path):
            st.warning("PDF file not found.")
            return

        state_key = f"pdf_show_all__{path}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
        with c1:
            st.markdown("**PDF preview (rendered)**")
            st.caption("Default shows page 1. Click to render more pages.")
        with c2:
            zoom = st.selectbox("Quality", [1.3, 1.6, 2.0], index=1, key=f"pdf_zoom__{path}")
        with c3:
            max_pages = st.number_input("Pages (when expanded)", min_value=1, max_value=200, value=30, step=1, key=f"pdf_pages__{path}")

        b1, b2 = st.columns([1, 1])
        with b1:
            if not st.session_state[state_key]:
                if st.button("📄 Render more pages", use_container_width=True, key=f"pdf_more__{path}"):
                    st.session_state[state_key] = True
                    st.rerun()
            else:
                if st.button("⬅️ Back to page 1", use_container_width=True, key=f"pdf_less__{path}"):
                    st.session_state[state_key] = False
                    st.rerun()
        with b2:
            render_download_file(path, "⬇️ Download PDF", key=f"dl_pdf_viewer__{path}")

        try:
            if st.session_state[state_key]:
                imgs = pdf_render_pages(path, max_pages=int(max_pages), zoom=float(zoom))
                st.caption(f"Showing **{len(imgs)}** page(s).")
                for i, b in enumerate(imgs, start=1):
                    st.image(b, caption=f"Page {i}", use_container_width=True)
            else:
                imgs = pdf_render_pages(path, max_pages=1, zoom=float(zoom))
                if imgs:
                    st.image(imgs[0], caption="Page 1", use_container_width=True)
        except Exception as e:
            st.error(f"PDF render failed. Install PyMuPDF: `pip install pymupdf`  |  Error: {e}")

    # =========================
    # Attachments render (saved)
    # =========================
    def show_saved_attachments(paths, caps: dict, expander_key: str):
        if not paths:
            st.info("No attachments yet.")
            return

        st.caption("Images inline • PDFs preview • Notebooks open (local) • Everything downloadable.")

        imgs = [p for p in paths if is_image(os.path.basename(p))]
        others = [p for p in paths if p not in imgs]

        if imgs:
            st.markdown("**🖼️ Images**")
            captions_list = []
            for p in imgs:
                fn = os.path.basename(p)
                captions_list.append((caps.get(fn, "") or "").strip() or fn)
            st.image(imgs, caption=captions_list, use_container_width=True)

        if others:
            st.markdown("**📄 Files**")
            for i, p in enumerate(others):
                fn = os.path.basename(p)
                cap = (caps.get(fn, "") or "").strip()

                r1, r2, r3 = st.columns([3, 1.0, 1.2])
                with r1:
                    st.markdown(f"**{fn}**")
                    st.caption(cap if cap else "")

                with r2:
                    # PDF preview
                    if is_pdf(p) and os.path.exists(p):
                        if st.button("👁️ Preview", key=f"pdf_prev__{expander_key}__{i}__{fn}", use_container_width=True):
                            st.session_state[f"pdf_preview_path__{expander_key}"] = p

                    # Notebook open (real)
                    if is_notebook(p) and os.path.exists(p):
                        if st.button("📓 Open", key=f"nb_open__{expander_key}__{i}__{fn}", use_container_width=True):
                            try:
                                open_notebook_real(p)
                                st.success("Opened notebook locally (best-effort).")
                            except Exception as e:
                                st.warning(f"Could not open notebook: {e}")

                with r3:
                    if is_pdf(p):
                        render_download_file(p, "⬇️ Download PDF", key=f"dl_pdf__{expander_key}__{i}__{fn}")
                    elif is_notebook(p):
                        render_download_file(p, "⬇️ Download .ipynb", key=f"dl_nb__{expander_key}__{i}__{fn}")
                    else:
                        render_download_file(p, "⬇️ Download", key=f"dl_file__{expander_key}__{i}__{fn}")

            prev_path = st.session_state.get(f"pdf_preview_path__{expander_key}", "")
            if prev_path and os.path.exists(prev_path) and is_pdf(prev_path):
                st.markdown("---")
                st.markdown("### 📄 PDF Preview")
                st.caption(os.path.basename(prev_path))
                render_pdf_preview(prev_path)

                if st.button("✖ Close preview", key=f"pdf_close__{expander_key}", use_container_width=True):
                    st.session_state.pop(f"pdf_preview_path__{expander_key}", None)
                    st.rerun()

    # =========================
    # Session defaults
    # =========================
    st.session_state.setdefault("dev_view_mode_main", "Active")
    st.session_state.setdefault("dev_show_add_experiment", False)
    st.session_state.setdefault("dev_show_new_project", False)
    st.session_state.setdefault("dev_show_manage_project", False)

    # ✅ selection safe keys
    st.session_state.setdefault("dev_selected_project", "")
    st.session_state.setdefault("dev_project_select_ver", 0)

    # =========================
    # Header card
    # =========================
    st.markdown("""
    <div class="dp-hero">
      <div class="dp-hero-title">🧪 Development Process</div>
      <div class="dp-hero-sub">Plan experiments • Attach files • Track updates • Link draws • Notes with Markdown/LaTeX</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # Toolbar (STICKY + 2-ROW)
    # =========================
    projects_df = load_projects()
    view_mode = st.session_state.get("dev_view_mode_main", "Active")

    filtered = projects_df[projects_df["Archived"] == (view_mode == "Archived")]
    project_options = [""] + filtered["Project Name"].dropna().astype(str).unique().tolist()

    st.markdown('<div class="dp-sticky"><div class="dp-toolbar">', unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns([1.10, 1.90, 1.20], gap="medium")

    with r1a:
        vm = st.segmented_control(
            "View",
            options=["Active", "Archived"],
            default=view_mode,
            key="dev_view_mode_main_sc",
        )
        st.session_state["dev_view_mode_main"] = vm

    with r1b:
        view_mode = st.session_state["dev_view_mode_main"]
        projects_df = load_projects()
        filtered = projects_df[projects_df["Archived"] == (view_mode == "Archived")]
        project_options = [""] + filtered["Project Name"].dropna().astype(str).unique().tolist()

        cur_sel = st.session_state.get("dev_selected_project", "")
        if cur_sel and (cur_sel not in project_options):
            st.session_state["dev_selected_project"] = ""
            st.session_state["dev_project_select_ver"] += 1
            cur_sel = ""

        proj_widget_key = f"dev_selected_project_widget__v{st.session_state['dev_project_select_ver']}"
        default_idx = project_options.index(cur_sel) if cur_sel in project_options else 0

        picked = st.selectbox(
            "Project",
            options=project_options,
            index=default_idx,
            key=proj_widget_key,
        )
        st.session_state["dev_selected_project"] = picked

    with r1c:
        selected_project = st.session_state.get("dev_selected_project", "")
        if selected_project:
            label = "➕ Add Experiment" if not st.session_state["dev_show_add_experiment"] else "➖ Hide"
            if st.button(label, use_container_width=True, type="primary", key="dp_btn_add_exp_toggle"):
                st.session_state["dev_show_add_experiment"] = not st.session_state["dev_show_add_experiment"]
        else:
            st.button("➕ Add Experiment", use_container_width=True, disabled=True, key="dp_btn_add_exp_disabled")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    r2a, r2b, r2c = st.columns([1.05, 1.25, 2.70], gap="medium")

    with r2a:
        if st.button("➕ New Project", use_container_width=True, key="dp_btn_new_project"):
            st.session_state["dev_show_new_project"] = not st.session_state["dev_show_new_project"]
            if st.session_state["dev_show_new_project"]:
                st.session_state["dev_show_manage_project"] = False

    with r2b:
        selected_project = st.session_state.get("dev_selected_project", "")
        if st.button("📦 Manage Project", use_container_width=True, disabled=not bool(selected_project), key="dp_btn_manage"):
            st.session_state["dev_show_manage_project"] = not st.session_state["dev_show_manage_project"]
            if st.session_state["dev_show_manage_project"]:
                st.session_state["dev_show_new_project"] = False

    with r2c:
        selected_project = st.session_state.get("dev_selected_project", "")
        mode = st.session_state.get("dev_view_mode_main", "Active")
        if selected_project:
            st.markdown(
                f'<span class="dp-pill">🟢 <b>{selected_project}</b> &nbsp;•&nbsp; <b>{mode}</b> &nbsp;•&nbsp; Ready</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="dp-pill">⚪ No project selected &nbsp;•&nbsp; <b>{mode}</b></span>',
                unsafe_allow_html=True
            )

    st.markdown('</div></div>', unsafe_allow_html=True)

    # =========================
    # New Project panel
    # =========================
    if st.session_state.get("dev_show_new_project", False):
        with st.expander("➕ Create a new project", expanded=True):
            with st.form("dp_create_project_form", clear_on_submit=True):
                new_project_name = st.text_input("Project Name")
                new_project_purpose = st.text_area("Project Purpose", height=110)
                new_project_target = st.text_area("Target", height=90)
                create_project = st.form_submit_button("Create Project")

            if create_project:
                projects_df = load_projects()
                if not new_project_name.strip():
                    st.error("Project Name is required!")
                elif (projects_df["Project Name"].astype(str).str.strip() == new_project_name.strip()).any():
                    st.error("A project with this name already exists.")
                else:
                    new_row = pd.DataFrame([{
                        "Project Name": new_project_name.strip(),
                        "Project Purpose": new_project_purpose.strip(),
                        "Target": new_project_target.strip(),
                        "Created At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Archived": False
                    }])
                    projects_df = pd.concat([projects_df, new_row], ignore_index=True)
                    save_projects(projects_df)

                    st.success("Project created!")
                    st.session_state["dev_show_new_project"] = False
                    st.session_state["dev_selected_project"] = new_project_name.strip()
                    st.session_state["dev_project_select_ver"] += 1
                    st.rerun()

    # =========================
    # Manage Project panel
    # =========================
    selected_project = st.session_state.get("dev_selected_project", "")
    if st.session_state.get("dev_show_manage_project", False):
        with st.expander("📦 Manage selected project", expanded=True):
            if not selected_project:
                st.info("Select a project first.")
            else:
                projects_df = load_projects()
                row = projects_df[projects_df["Project Name"] == selected_project]
                if row.empty:
                    st.warning("Project not found.")
                else:
                    is_archived = bool(row.iloc[0].get("Archived", False))

                    cA, cB, cC = st.columns([1, 1, 1.2])
                    with cA:
                        if not is_archived:
                            if st.button("🗄️ Archive", use_container_width=True, key="dp_arch"):
                                projects_df.loc[projects_df["Project Name"] == selected_project, "Archived"] = True
                                save_projects(projects_df)
                                st.success("Archived.")
                                st.rerun()
                        else:
                            if st.button("♻️ Restore", use_container_width=True, key="dp_restore"):
                                projects_df.loc[projects_df["Project Name"] == selected_project, "Archived"] = False
                                save_projects(projects_df)
                                st.success("Restored.")
                                st.rerun()

                    with cB:
                        if st.button("🧾 Close panel", use_container_width=True, key="dp_close_manage"):
                            st.session_state["dev_show_manage_project"] = False
                            st.rerun()

                    with cC:
                        st.markdown("**Danger zone**")
                        if st.button("🗑️ Delete project (permanent)", use_container_width=True, key="dp_delete"):
                            exp_df_del = load_experiments()
                            upd_df_del = load_updates()

                            projects_df = projects_df[projects_df["Project Name"] != selected_project]
                            exp_df_del = exp_df_del[exp_df_del["Project Name"] != selected_project]
                            upd_df_del = upd_df_del[upd_df_del["Project Name"] != selected_project]

                            save_projects(projects_df)
                            save_experiments(exp_df_del)
                            save_updates(upd_df_del)

                            st.session_state["dev_selected_project"] = ""
                            st.session_state["dev_show_manage_project"] = False
                            st.session_state["dev_project_select_ver"] += 1
                            st.warning("Deleted permanently.")
                            st.rerun()

    st.divider()

    # =========================
    # Main content
    # =========================
    selected_project = st.session_state.get("dev_selected_project", "")
    if not selected_project:
        st.info("Use **Project** selector above to start.")
        st.stop()

    projects_df = load_projects()
    proj_row = projects_df[projects_df["Project Name"] == selected_project]
    if proj_row.empty:
        st.warning("Selected project not found.")
        st.stop()

    proj = proj_row.iloc[0]
    with st.expander("📌 Project Details", expanded=True):
        st.markdown(f"**Project Purpose:** {proj.get('Project Purpose', 'N/A')}")
        st.markdown(f"**Target:** {proj.get('Target', 'N/A')}")
        st.caption(f"Created at: {proj.get('Created At', '')} | Archived: {bool(proj.get('Archived', False))}")

    st.divider()

    # =========================
    # Add Experiment
    # =========================
    if st.session_state.get("dev_show_add_experiment", False):
        with st.expander("➕ Add Experiment", expanded=True):
            is_drawing_live = st.checkbox("Is this a Drawing?", key=f"newexp_is_drawing__{selected_project}")
            drawing_details_live = ""
            draw_csv_live = ""

            if is_drawing_live:
                drawing_details_live = st.text_area("Drawing Details", height=90, key=f"newexp_drawing_details__{selected_project}")

                dataset_files = list_dataset_csvs_newest_first()
                if dataset_files:
                    newest = dataset_files[0]
                    st.caption(f"Newest CSV: **{newest}**")
                    draw_csv_live = st.selectbox(
                        "Select Draw CSV (newest first)",
                        [""] + dataset_files,
                        index=1,
                        key=f"newexp_draw_csv__{selected_project}"
                    )
                else:
                    st.info("No CSV files found in data_set_csv/")

            st.divider()
            st.markdown("### 📎 Attach files (optional)")
            uploaded_new_files = st.file_uploader(
                "Drag & drop files (images / PDF / .ipynb / anything)",
                type=None,
                accept_multiple_files=True,
                key=f"newexp_attachments__{selected_project}"
            )

            caption_inputs = {}
            if uploaded_new_files:
                st.markdown("### 📝 Descriptions (one per file)")
                for f in uploaded_new_files:
                    caption_inputs[f.name] = st.text_area(
                        f"Description for {f.name}",
                        height=80,
                        key=f"newexp_caption__{selected_project}__{f.name}"
                    )

            st.divider()
            st.markdown("### 📓 Notes (Markdown + LaTeX)")
            notes_md = st.text_area("Write notes here", height=160, key=f"newexp_notes__{selected_project}")
            st.caption("Markdown + LaTeX: inline `$E=mc^2$` or block `$$\\Delta n(r)=n_0 e^{-r^2/w^2}$$`")

            st.divider()

            with st.form(f"add_experiment_form__{selected_project}", clear_on_submit=True):
                c1, c2 = st.columns([2, 1])
                experiment_title = c1.text_input("Experiment Title")
                date = c2.date_input("Date")

                researcher = st.text_input("Researcher Name")
                methods = st.text_area("Methods", height=90)
                purpose = st.text_area("Experiment Purpose", height=90)
                observations = st.text_area("Observations", height=90)
                results = st.text_area("Results", height=90)

                add_exp = st.form_submit_button("✅ Save Experiment")

            if add_exp:
                if not experiment_title.strip():
                    st.warning("Please provide an Experiment Title.")
                else:
                    exp_df = load_experiments()
                    exp_date_str = date.strftime("%Y-%m-%d")

                    dup = exp_df[
                        (exp_df["Project Name"] == selected_project) &
                        (exp_df["Experiment Title"].astype(str).str.strip() == experiment_title.strip()) &
                        (exp_df["Date"].astype(str).str.strip() == exp_date_str)
                    ]
                    if not dup.empty:
                        st.error("This experiment (same title + date) already exists in this project.")
                    else:
                        saved_paths = []
                        caps_map = {}

                        if uploaded_new_files:
                            media_dir = exp_media_dir(selected_project, experiment_title.strip(), exp_date_str)
                            for f in uploaded_new_files:
                                try:
                                    out_path = _unique_path(os.path.join(media_dir, f.name))
                                    with open(out_path, "wb") as w:
                                        w.write(f.getbuffer())
                                    saved_paths.append(out_path)
                                    caps_map[os.path.basename(out_path)] = (caption_inputs.get(f.name, "") or "").strip()
                                except Exception as e:
                                    st.error(f"Failed saving {f.name}: {e}")

                        new_exp = pd.DataFrame([{
                            "Project Name": selected_project,
                            "Experiment Title": experiment_title.strip(),
                            "Date": exp_date_str,
                            "Researcher": researcher.strip(),
                            "Methods": methods.strip(),
                            "Purpose": purpose.strip(),
                            "Observations": observations.strip(),
                            "Results": results.strip(),
                            "Is Drawing": bool(is_drawing_live),
                            "Drawing Details": drawing_details_live.strip() if is_drawing_live else "",
                            "Draw CSV": draw_csv_live.strip() if is_drawing_live else "",
                            "Attachments": join_path_list(saved_paths) if saved_paths else "",
                            "Attachment Captions": dump_captions(caps_map) if caps_map else "",
                            "Markdown Notes": (st.session_state.get(f"newexp_notes__{selected_project}", "") or "").strip(),
                        }])

                        exp_df = pd.concat([exp_df, new_exp], ignore_index=True)
                        save_experiments(exp_df)

                        st.success(f"Experiment saved. Attachments: {len(saved_paths)}")
                        st.session_state["dev_show_add_experiment"] = False
                        st.rerun()

        st.divider()

    # =========================
    # Experiments list
    # =========================
    exp_df = load_experiments()
    project_exps = exp_df[exp_df["Project Name"] == selected_project].copy()

    if project_exps.empty:
        st.info("No experiments yet.")
    else:
        st.subheader("🔬 Experiments Conducted")
        project_exps["Date_sort"] = pd.to_datetime(project_exps["Date"], errors="coerce")
        project_exps = project_exps.sort_values("Date_sort", ascending=False)

        for idx, exp in project_exps.iterrows():
            exp_title = str(exp.get("Experiment Title", "Untitled"))
            exp_date = str(exp.get("Date", ""))

            expander_key = f"exp_{selected_project}_{exp_title}_{exp_date}_{idx}"

            with st.expander(f"🧪 {exp_title} ({exp_date})", expanded=False):

                # ✅ DEFAULT TAB = Preview (put it first)
                tab_preview, tab_edit = st.tabs(["👁️ Preview", "✍️ Edit"])

                # -------------------------
                # PREVIEW (NO INPUTS)
                # -------------------------
                with tab_preview:
                    st.write(f"**Researcher:** {exp.get('Researcher', 'N/A')}")
                    st.write(f"**Methods:** {exp.get('Methods', 'N/A')}")
                    st.write(f"**Purpose:** {exp.get('Purpose', 'N/A')}")
                    st.write(f"**Observations:** {exp.get('Observations', 'N/A')}")
                    st.write(f"**Results:** {exp.get('Results', 'N/A')}")

                    # Notes preview
                    st.divider()
                    st.markdown("#### 📓 Notes (Markdown + LaTeX)")
                    notes_preview = str(exp.get("Markdown Notes", "") or "").strip()
                    if notes_preview:
                        st.markdown(notes_preview)
                    else:
                        st.caption("No notes yet.")

                    # Drawing preview (view-only button allowed)
                    if bool(exp.get("Is Drawing", False)):
                        st.divider()
                        st.markdown("#### 🧵 Drawing")
                        st.write(f"**Drawing Details:** {exp.get('Drawing Details', '')}")
                        draw_csv_name = str(exp.get("Draw CSV", "")).strip()
                        if draw_csv_name:
                            st.write(f"**Draw CSV:** `{draw_csv_name}`")
                            csv_path = os.path.join(DATASET_DIR, draw_csv_name)
                            if os.path.exists(csv_path):
                                if st.button("📄 Load & View Draw CSV", key=f"load_draw__{expander_key}"):
                                    df_draw = load_draw_csv(csv_path)
                                    st.dataframe(df_draw, use_container_width=True, height=320)

                    # Attachments preview (saved only)
                    st.divider()
                    st.markdown("#### 📎 Attachments")
                    saved_paths = parse_path_list(exp.get("Attachments", ""))
                    saved_caps = parse_captions(exp.get("Attachment Captions", ""))
                    show_saved_attachments(saved_paths, saved_caps, expander_key)

                    # Updates preview (list only)
                    st.divider()
                    st.markdown("#### 📜 Progress Updates")
                    upd_df = load_updates()
                    exp_updates = upd_df[
                        (upd_df["Project Name"] == selected_project) &
                        (upd_df["Experiment Title"] == exp_title)
                    ].copy()

                    if exp_updates.empty:
                        st.caption("No updates yet.")
                    else:
                        exp_updates["Update_sort"] = pd.to_datetime(exp_updates["Update Date"], errors="coerce")
                        exp_updates = exp_updates.sort_values("Update_sort", ascending=True)
                        for _, u in exp_updates.iterrows():
                            st.write(
                                f"📅 **{u.get('Update Date', '')}** — **{u.get('Researcher', '')}**: {u.get('Update Notes', '')}"
                            )

                # -------------------------
                # EDIT (ALL INPUTS HERE)
                # -------------------------
                with tab_edit:
                    # Notes editor
                    st.markdown("#### ✍️ Edit Notes (Markdown + LaTeX)")
                    edited_notes = st.text_area(
                        "Notes",
                        value=str(exp.get("Markdown Notes", "") or ""),
                        height=220,
                        key=f"md_notes__{expander_key}"
                    )
                    csave, chelp = st.columns([1, 2])
                    with csave:
                        if st.button("💾 Save Notes", use_container_width=True, key=f"save_notes__{expander_key}"):
                            exp_df2 = load_experiments()
                            mask = (
                                (exp_df2["Project Name"] == selected_project) &
                                (exp_df2["Experiment Title"].astype(str) == exp_title) &
                                (exp_df2["Date"].astype(str) == exp_date)
                            )
                            exp_df2.loc[mask, "Markdown Notes"] = (edited_notes or "").strip()
                            save_experiments(exp_df2)
                            st.success("Notes saved.")
                            st.rerun()
                    with chelp:
                        st.caption("Inline `$E=mc^2$` • block `$$\\Delta n(r)=n_0 e^{-r^2/w^2}$$` • tables, code blocks, etc.")

                    st.divider()

                    # Add attachments (inputs here)
                    st.markdown("#### ➕ Add attachments")
                    st.caption("Upload images / PDFs / notebooks (.ipynb) or any file.")
                    add_files = st.file_uploader(
                        "Drop files here",
                        type=None,
                        accept_multiple_files=True,
                        key=f"add_files__{expander_key}"
                    )

                    add_caps = {}
                    if add_files:
                        st.markdown("**Descriptions for new files**")
                        for f in add_files:
                            add_caps[f.name] = st.text_area(
                                f"Description for {f.name}",
                                height=80,
                                key=f"add_cap__{expander_key}__{f.name}"
                            )

                        if st.button("💾 Save attachments", use_container_width=True, key=f"save_added__{expander_key}"):
                            media_dir = exp_media_dir(selected_project, exp_title, exp_date)
                            exp_df2 = load_experiments()

                            # current saved state
                            current_paths = parse_path_list(exp.get("Attachments", ""))
                            current_caps = parse_captions(exp.get("Attachment Captions", ""))

                            new_paths = []
                            for f in add_files:
                                try:
                                    out_path = _unique_path(os.path.join(media_dir, f.name))
                                    with open(out_path, "wb") as w:
                                        w.write(f.getbuffer())
                                    new_paths.append(out_path)
                                    current_caps[os.path.basename(out_path)] = (add_caps.get(f.name, "") or "").strip()
                                except Exception as e:
                                    st.error(f"Failed saving {f.name}: {e}")

                            if new_paths:
                                mask = (
                                    (exp_df2["Project Name"] == selected_project) &
                                    (exp_df2["Experiment Title"].astype(str) == exp_title) &
                                    (exp_df2["Date"].astype(str) == exp_date)
                                )
                                merged = (current_paths or []) + new_paths
                                exp_df2.loc[mask, "Attachments"] = join_path_list(merged)
                                exp_df2.loc[mask, "Attachment Captions"] = dump_captions(current_caps)
                                save_experiments(exp_df2)

                                st.success(f"Saved {len(new_paths)} file(s).")
                                st.rerun()

                    st.divider()

                    # Add update (inputs here)
                    st.markdown("#### 🔄 Add Update")
                    with st.form(f"update_form__{expander_key}"):
                        update_researcher = st.text_input("Your name", key=f"upd_name__{expander_key}")
                        update_notes = st.text_area("Update notes", height=80, key=f"upd_notes__{expander_key}")
                        submit_update = st.form_submit_button("Add Update")

                    if submit_update:
                        if not update_notes.strip():
                            st.warning("Please write update notes.")
                        else:
                            upd_df2 = load_updates()
                            new_u = pd.DataFrame([{
                                "Project Name": selected_project,
                                "Experiment Title": exp_title,
                                "Update Date": datetime.now().strftime("%Y-%m-%d"),
                                "Researcher": update_researcher.strip(),
                                "Update Notes": update_notes.strip()
                            }])
                            upd_df2 = pd.concat([upd_df2, new_u], ignore_index=True)
                            save_updates(upd_df2)
                            st.success("Update added!")
                            st.rerun()

    st.divider()

    st.subheader("📢 Project Conclusion")
    conclusion_file = f"project_conclusion__{selected_project.replace(' ', '_')}.txt"

    existing = ""
    if os.path.exists(conclusion_file):
        try:
            existing = open(conclusion_file, "r", encoding="utf-8").read()
        except Exception:
            existing = ""

    conclusion = st.text_area("Conclusion / final summary", value=existing, height=170)

    if st.button("💾 Save Conclusion", key=f"save_conclusion__{selected_project}"):
        try:
            with open(conclusion_file, "w", encoding="utf-8") as f:
                f.write(conclusion)
            st.success("Conclusion saved.")
        except Exception as e:
            st.error(f"Failed to save conclusion: {e}")
# ------------------ Protocols Tab ------------------
