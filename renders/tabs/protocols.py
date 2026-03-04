import hashlib
import json
import os
import time

import streamlit as st
from app_io.paths import P


def render_protocols_tab(base_dir: str = None):
    if "_page_config_set" not in st.session_state:
        try:
            st.set_page_config(layout="wide")
        except Exception:
            pass
        st.session_state["_page_config_set"] = True

    st.markdown(
        """
    <style>
      .block-container { padding-top: 2.5rem; }

      .proto-topbar {
        display:flex; align-items:flex-end; justify-content:space-between;
        gap: 12px; margin-bottom: 8px; padding-top: 8px;
      }
      .proto-title h1 { margin: 0; padding: 4px 0 0 0; line-height: 1.2; }
      .proto-sub { opacity: 0.85; margin-top: 5px; font-size: 0.95rem; color: rgba(192,228,248,0.90); }
      .proto-line{
        height: 1px;
        margin: 0 0 10px 0;
        background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
      }

      .chips { display:flex; gap: 8px; flex-wrap: wrap; justify-content:flex-start; margin: 10px 0 14px; }
      .chip {
        border: 1px solid rgba(128,206,255,0.26);
        background: linear-gradient(180deg, rgba(25,62,102,0.36), rgba(10,28,50,0.28));
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        line-height: 1;
        white-space: nowrap;
        color: rgba(228,246,255,0.96);
      }

      .section-card {
        border: 1px solid rgba(128,206,255,0.22);
        background: linear-gradient(180deg, rgba(14,32,56,0.28), rgba(8,16,28,0.22));
        padding: 14px 14px;
        border-radius: 14px;
        margin-bottom: 14px;
      }

      .proto-card {
        border: 1px solid rgba(128,206,255,0.22);
        background: linear-gradient(180deg, rgba(14,32,56,0.28), rgba(8,16,28,0.22));
        padding: 12px 12px;
        border-radius: 14px;
        margin-bottom: 10px;
        box-shadow: 0 8px 18px rgba(8,30,58,0.20);
      }
      .proto-card .row1 {
        display:flex; align-items:center; justify-content:space-between; gap:10px;
      }
      .proto-name {
        font-weight: 750;
        font-size: 1.05rem;
        line-height: 1.2;
      }
      .proto-meta {
        display:flex; gap: 6px; flex-wrap:wrap; justify-content:flex-end;
        opacity: 0.9;
      }
      .pill {
        font-size: 0.78rem;
        border: 1px solid rgba(128,206,255,0.26);
        padding: 3px 8px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(26,70,116,0.34), rgba(10,28,52,0.28));
        color: rgba(226,244,255,0.96);
      }
      .hr-lite { height: 1px; background: rgba(120,200,255,0.28); margin: 10px 0; border-radius: 2px; }
      .muted { opacity: 0.75; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .small-note { font-size: 0.85rem; opacity: 0.75; }

      details[data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid rgba(128,206,255,0.24) !important;
        background: linear-gradient(180deg, rgba(14,32,56,0.28), rgba(8,16,28,0.22)) !important;
        padding: 6px 10px !important;
      }
      details[data-testid="stExpander"][open]{
        box-shadow: 0 12px 22px rgba(0,0,0,0.24), 0 0 14px rgba(82,182,255,0.20) !important;
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

      div[data-baseweb="tag"], span[data-baseweb="tag"]{
        background: linear-gradient(180deg, rgba(70,160,238,0.92), rgba(32,96,168,0.90)) !important;
        border: 1px solid rgba(170,232,255,0.78) !important;
        color: rgba(244,252,255,0.99) !important;
      }
      div[data-baseweb="tag"] *, span[data-baseweb="tag"] *{
        color: rgba(244,252,255,0.99) !important;
        fill: rgba(238,250,255,0.98) !important;
      }

      textarea {
        min-height: 420px !important;
        font-size: 0.98rem !important;
        line-height: 1.35 !important;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="proto-topbar">
      <div class="proto-title">
        <h1>📋 Protocols</h1>
        <div class="proto-sub">Browse, run checklists, attach photos, and manage tower protocols.</div>
      </div>
    </div>
    <div class="proto-line"></div>
    """,
        unsafe_allow_html=True,
    )

    protocols_file = P.protocols_json
    assets_dir = P.protocols_assets_dir
    os.makedirs(assets_dir, exist_ok=True)

    protocol_types = ["Drawings", "Maintenance", "Tower Regular Operations"]
    sub_types = ["Checklist", "Instructions"]

    def _safe_read_json(path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _normalize_protocols(raw):
        """
        Accept either:
        - list[dict] (current format)
        - {"protocols": list[dict]} (legacy/wrapped format)
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            val = raw.get("protocols", [])
            return val if isinstance(val, list) else []
        return []

    def _safe_write_json(path: str, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
        os.replace(tmp, path)

    # Keep session state synced with file if path/mtime changed,
    # and self-heal if session became empty while file has data.
    file_mtime = os.path.getmtime(protocols_file) if os.path.exists(protocols_file) else None
    current_source = st.session_state.get("_protocols_source_path")
    current_mtime = st.session_state.get("_protocols_source_mtime")
    loaded_from_file = _normalize_protocols(_safe_read_json(protocols_file, []))
    session_protocols = st.session_state.get("protocols")
    must_sync = (
        ("protocols" not in st.session_state)
        or (current_source != protocols_file)
        or (current_mtime != file_mtime)
        or (not isinstance(session_protocols, list))
        or (len(session_protocols or []) == 0 and len(loaded_from_file) > 0)
    )
    if must_sync:
        st.session_state["protocols"] = list(loaded_from_file)
        st.session_state["_protocols_source_path"] = protocols_file
        st.session_state["_protocols_source_mtime"] = file_mtime

    for proto in st.session_state["protocols"]:
        proto.setdefault("type", "Tower Regular Operations")
        proto.setdefault("sub_type", "Instructions")
        proto.setdefault("instructions", "")
        proto.setdefault("name", "Untitled")
        proto.setdefault("images", [])

    st.caption(f"Source: {protocols_file} | Loaded: {len(st.session_state['protocols'])}")

    # Utility action to re-sync from file on demand.
    if st.button("🔄 Reload Protocols From File", key="reload_protocols_btn"):
        loaded = _normalize_protocols(_safe_read_json(protocols_file, []))
        st.session_state["protocols"] = loaded
        st.session_state["_protocols_source_path"] = protocols_file
        st.session_state["_protocols_source_mtime"] = os.path.getmtime(protocols_file) if os.path.exists(protocols_file) else None
        st.success(f"Reloaded {len(loaded)} protocol(s) from: {protocols_file}")
        st.rerun()

    if "protocol_check_state" not in st.session_state:
        st.session_state["protocol_check_state"] = {}

    def _proto_id(proto: dict) -> str:
        s = f"{proto.get('name','')}|{proto.get('type','')}|{proto.get('sub_type','')}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

    def _normalize_lines(text: str):
        lines = [ln.strip() for ln in (text or "").splitlines()]
        return [ln for ln in lines if ln]

    def _matches(q: str, proto: dict) -> bool:
        if not q:
            return True
        blob = (
            f"{proto.get('name','')} {proto.get('type','')} "
            f"{proto.get('sub_type','')} {proto.get('instructions','')}"
        ).lower()
        return q.lower() in blob

    def _safe_ext(filename: str) -> str:
        fn = (filename or "").lower().strip()
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            if fn.endswith(ext):
                return ext
        return ".png"

    def _save_uploaded_image(uploaded_file, pid: str) -> str:
        raw = uploaded_file.getvalue()
        h = hashlib.md5(raw).hexdigest()[:10]
        ext = _safe_ext(uploaded_file.name)
        fname = f"{pid}_{int(time.time()*1000)}_{h}{ext}"
        path = os.path.join(assets_dir, fname)
        with open(path, "wb") as f:
            f.write(raw)
        return fname

    def _existing_image_paths(img_names: list) -> list:
        out = []
        for name in (img_names or []):
            path = os.path.join(assets_dir, str(name))
            if os.path.isfile(path):
                out.append(path)
        return out

    total_n = len(st.session_state["protocols"])
    by_type = {t: 0 for t in protocol_types}
    by_sub = {s: 0 for s in sub_types}
    for proto in st.session_state["protocols"]:
        if proto.get("type") in by_type:
            by_type[proto["type"]] += 1
        if proto.get("sub_type") in by_sub:
            by_sub[proto["sub_type"]] += 1

    st.markdown(
        f"""
        <div class="chips">
          <div class="chip">Total: <b>{total_n}</b></div>
          <div class="chip">Drawings: <b>{by_type["Drawings"]}</b></div>
          <div class="chip">Maintenance: <b>{by_type["Maintenance"]}</b></div>
          <div class="chip">Ops: <b>{by_type["Tower Regular Operations"]}</b></div>
          <div class="chip">Checklists: <b>{by_sub["Checklist"]}</b></div>
          <div class="chip">Instructions: <b>{by_sub["Instructions"]}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📚 Browse")

    f1, f2, f3, f4 = st.columns([1.7, 1.0, 1.0, 0.8])
    with f1:
        q = st.text_input("Search", placeholder="Search name / type / content…", key="proto_search_full")
    with f2:
        type_filter = st.selectbox("Type", ["All"] + protocol_types, key="proto_type_filter_full")
    with f3:
        sub_filter = st.selectbox("Sub-type", ["All"] + sub_types, key="proto_sub_filter_full")
    with f4:
        sort_by = st.selectbox("Sort", ["A→Z", "Z→A"], key="proto_sort_full")

    items = []
    for proto in st.session_state["protocols"]:
        if type_filter != "All" and proto.get("type") != type_filter:
            continue
        if sub_filter != "All" and proto.get("sub_type") != sub_filter:
            continue
        if not _matches(q, proto):
            continue
        items.append(proto)

    items.sort(key=lambda x: (x.get("name", "").lower()))
    if sort_by == "Z→A":
        items.reverse()

    st.markdown("</div>", unsafe_allow_html=True)

    if not items:
        st.info("No protocols match your filters.")
    else:
        for proto in items:
            pid = _proto_id(proto)
            name = proto.get("name", "Untitled")
            ptype = proto.get("type", "")
            psub = proto.get("sub_type", "Instructions")
            instructions = proto.get("instructions", "")
            images = proto.get("images", []) or []

            st.markdown(
                f"""
                <div class="proto-card">
                  <div class="row1">
                    <div class="proto-name">{name}</div>
                    <div class="proto-meta">
                      <span class="pill">{ptype}</span>
                      <span class="pill">{psub}</span>
                      <span class="pill">{len(images)} photos</span>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Open", expanded=False):
                img_paths = _existing_image_paths(images)
                if img_paths:
                    st.caption("📷 Photos")
                    cols = st.columns(3)
                    for i, path in enumerate(img_paths):
                        with cols[i % 3]:
                            st.image(path, use_container_width=True)
                    st.markdown('<div class="hr-lite"></div>', unsafe_allow_html=True)

                if psub == "Checklist":
                    lines = _normalize_lines(instructions)
                    if not lines:
                        st.warning("Checklist has no items (one item per line).")
                    else:
                        if pid not in st.session_state["protocol_check_state"]:
                            st.session_state["protocol_check_state"][pid] = {ln: False for ln in lines}

                        existing = st.session_state["protocol_check_state"][pid]
                        merged = {ln: bool(existing.get(ln, False)) for ln in lines}
                        st.session_state["protocol_check_state"][pid] = merged

                        done_count = sum(1 for v in merged.values() if v)
                        total_count = len(lines)

                        st.caption(f"Progress: **{done_count}/{total_count}**")
                        st.progress(done_count / total_count if total_count else 0)

                        for i, item in enumerate(lines):
                            key = f"proto_chk_{pid}_{i}"
                            val = st.checkbox(item, value=merged[item], key=key)
                            st.session_state["protocol_check_state"][pid][item] = val

                        if done_count == total_count and total_count > 0:
                            st.success("✅ Checklist completed!")

                        if st.button("Reset this checklist", key=f"reset_{pid}", use_container_width=True):
                            st.session_state["protocol_check_state"][pid] = {ln: False for ln in lines}
                            st.rerun()
                else:
                    pretty = (instructions or "").strip()
                    if not pretty:
                        st.info("No instructions found.")
                    else:
                        st.markdown(pretty.replace("\n", "  \n"))

                st.markdown('<div class="hr-lite"></div>', unsafe_allow_html=True)
                edit = st.toggle("✏️ Edit this protocol", key=f"proto_edit_{pid}", value=False)
                if edit:
                    st.subheader("Edit")

                    c1, c2, c3 = st.columns([1.7, 1.0, 1.0])
                    with c1:
                        new_name = st.text_input("Name", value=proto.get("name", ""), key=f"edit_name_{pid}")
                    with c2:
                        ptype_idx = (
                            protocol_types.index(proto.get("type"))
                            if proto.get("type") in protocol_types
                            else 0
                        )
                        new_type = st.selectbox("Type", protocol_types, index=ptype_idx, key=f"edit_type_{pid}")
                    with c3:
                        psub_idx = (
                            sub_types.index(proto.get("sub_type"))
                            if proto.get("sub_type") in sub_types
                            else 1
                        )
                        new_sub = st.selectbox("Sub-type", sub_types, index=psub_idx, key=f"edit_sub_{pid}")

                    new_text = st.text_area(
                        "Instructions / Checklist items",
                        value=proto.get("instructions", ""),
                        height=420,
                        key=f"edit_text_{pid}",
                    )

                    st.markdown("### 📷 Photos")
                    up = st.file_uploader(
                        "Upload photos (png/jpg/jpeg/webp)",
                        type=["png", "jpg", "jpeg", "webp"],
                        accept_multiple_files=True,
                        key=f"proto_uploader_{pid}",
                    )

                    existing_imgs = proto.get("images", []) or []
                    if existing_imgs:
                        del_sel = st.multiselect(
                            "Select photos to remove",
                            options=existing_imgs,
                            key=f"proto_del_sel_{pid}",
                        )
                    else:
                        del_sel = []

                    csave, cdel = st.columns([1, 1])
                    with csave:
                        if st.button("💾 Save changes", key=f"save_{pid}", use_container_width=True):
                            for j in range(len(st.session_state["protocols"])):
                                if _proto_id(st.session_state["protocols"][j]) == pid:
                                    imgs = list(existing_imgs)
                                    if up:
                                        for uf in up:
                                            try:
                                                fname = _save_uploaded_image(uf, pid)
                                                imgs.append(fname)
                                            except Exception as e:
                                                st.warning(f"Failed saving image: {e}")

                                    if del_sel:
                                        imgs2 = []
                                        for im in imgs:
                                            if im in del_sel:
                                                try:
                                                    fp = os.path.join(assets_dir, im)
                                                    if os.path.isfile(fp):
                                                        os.remove(fp)
                                                except Exception:
                                                    pass
                                            else:
                                                imgs2.append(im)
                                        imgs = imgs2

                                    st.session_state["protocols"][j] = {
                                        "name": (new_name or "Untitled").strip(),
                                        "type": new_type,
                                        "sub_type": new_sub,
                                        "instructions": (new_text or "").strip(),
                                        "images": imgs,
                                    }
                                    _safe_write_json(protocols_file, st.session_state["protocols"])
                                    st.success("Saved.")
                                    st.rerun()
                            st.warning("Could not find protocol to save (ID mismatch).")

                    with cdel:
                        if st.button("🗑️ Delete this protocol", key=f"del_proto_{pid}", use_container_width=True):
                            st.session_state["protocols"] = [
                                pp for pp in st.session_state["protocols"] if _proto_id(pp) != pid
                            ]
                            _safe_write_json(protocols_file, st.session_state["protocols"])
                            st.success("Deleted.")
                            st.rerun()

                st.markdown(
                    f"<div class='small-note muted'>ID: <span class='mono'>{pid}</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("➕ Create new protocol", expanded=False):
        st.markdown(
            "<div class='small-note muted'>Create in full width so it’s comfortable to write.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='hr-lite'></div>", unsafe_allow_html=True)

        with st.form("proto_create_form_big", clear_on_submit=True):
            r1c1, r1c2, r1c3 = st.columns([1.8, 1.0, 1.0])
            with r1c1:
                new_name = st.text_input("Protocol name", placeholder="e.g. Pre-Draw Checklist")
            with r1c2:
                new_type = st.selectbox("Type", protocol_types, index=0, key="proto_new_type_big")
            with r1c3:
                new_sub = st.selectbox("Sub-type", sub_types, index=0, key="proto_new_subtype_big")

            new_text = st.text_area(
                "Instructions",
                height=520,
                placeholder="Checklist: one item per line.\n\nInstructions: write steps freely.",
            )

            new_photos = st.file_uploader(
                "Optional: add photos now (png/jpg/jpeg/webp)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="proto_create_photos",
            )

            add = st.form_submit_button("Add protocol", use_container_width=True)
            if add:
                if not (new_name and new_text):
                    st.error("Please fill **name** and **instructions**.")
                else:
                    proto_obj = {
                        "name": new_name.strip(),
                        "type": new_type,
                        "sub_type": new_sub,
                        "instructions": new_text.strip(),
                        "images": [],
                    }
                    pid_new = _proto_id(proto_obj)

                    imgs = []
                    if new_photos:
                        for uf in new_photos:
                            try:
                                imgs.append(_save_uploaded_image(uf, pid_new))
                            except Exception as e:
                                st.warning(f"Failed saving image: {e}")

                    proto_obj["images"] = imgs

                    st.session_state["protocols"].append(proto_obj)
                    _safe_write_json(protocols_file, st.session_state["protocols"])
                    st.success(f"Added: {new_name}")
                    st.rerun()
