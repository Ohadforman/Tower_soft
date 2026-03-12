# renders/process_setup.py
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Union, Optional

import pandas as pd
import streamlit as st

from app_io.paths import P, dataset_csv_path
from helpers.text_utils import safe_str, now_str
from helpers.constants import STATUS_COL, STATUS_UPDATED_COL, MSG_SCHED, MSG_FAILED
from helpers.process_setup_state import apply_order_row_to_process_setup_state
from helpers.process_setup_state import apply_dataset_process_rows_to_state
from helpers.app_logger import log_event
from helpers.dataset_io import most_recent_csv
from helpers.activity_indicator import record_activity_start

# These are your existing “collect” UIs (already moved out of dash_try)
from renders.components.coating import render_coating_section
from renders.components.iris import render_iris_selection_section_collect
from renders.components.pid_tf import render_pid_tf_section_collect
from renders.components.drum import render_drum_selection_section_collect
from helpers.coating_config import load_coating_config
from helpers.coating_config import load_config_coating_json
from renders.components.save_all import render_save_all_block


@st.cache_data(show_spinner=False, ttl=20)
def _cached_distinct_col_values(path: str, col: str, file_mtime: float) -> list[str]:
    """Fast distinct-values loader for dropdown helpers."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return []
    if col not in df.columns:
        return []
    out = []
    seen = set()
    for v in df[col].astype(str).tolist():
        s = safe_str(v).strip()
        if not s:
            continue
        lk = s.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(s)
    return out


# ==========================================================
# Link helpers
# ==========================================================
def find_associated_dataset_csv(order_row: Union[pd.Series, dict]) -> str:
    """
    Returns the dataset CSV linked to an order row.
    Checks common columns we use across versions.
    """
    cols = [
        "Assigned Dataset CSV",
        "Active CSV",
        "Done CSV",
        "Failed CSV",
        "Fail Try Dataset CSV",
        "Last Try Dataset CSV",
    ]
    for col in cols:
        try:
            v = safe_str(order_row.get(col, "") if isinstance(order_row, dict) else order_row.get(col, ""))
        except Exception:
            v = ""
        if v:
            return v
    return ""


def process_setup_buttons(key: str = "process_setup_view") -> str:
    """Returns which section to show: 'all' | 'coating' | 'iris' | 'pid' | 'drum'."""
    if key not in st.session_state:
        st.session_state[key] = "all"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("🧴 Coating", use_container_width=True, key=f"{key}_btn_coating"):
            st.session_state[key] = "coating"
            st.rerun()
    with c2:
        if st.button("🔍 Iris", use_container_width=True, key=f"{key}_btn_iris"):
            st.session_state[key] = "iris"
            st.rerun()
    with c3:
        if st.button("🎛 PID & TF", use_container_width=True, key=f"{key}_btn_pid"):
            st.session_state[key] = "pid"
            st.rerun()
    with c4:
        if st.button("🧵 Drum", use_container_width=True, key=f"{key}_btn_drum"):
            st.session_state[key] = "drum"
            st.rerun()
    with c5:
        if st.button("✅ All", use_container_width=True, key=f"{key}_btn_all"):
            st.session_state[key] = "all"
            st.rerun()

    return st.session_state[key]


def next_draw_index_for_preform(preform_id: str, orders_df: pd.DataFrame) -> int:
    """
    Returns next draw index N for dataset name {PreformID}F_N.csv
    Scans all CSV references in the orders file.
    """
    if not preform_id or orders_df is None or orders_df.empty:
        return 1

    pat = re.compile(rf"^{re.escape(preform_id)}F_(\d+)\.csv$", re.IGNORECASE)
    max_n = 0

    for col in [
        "Assigned Dataset CSV",
        "Active CSV",
        "Done CSV",
        "Failed CSV",
        "Fail Try Dataset CSV",
        "Last Try Dataset CSV",
    ]:
        if col not in orders_df.columns:
            continue

        for v in orders_df[col].astype(str):
            m = pat.match(v.strip())
            if m:
                try:
                    max_n = max(max_n, int(m.group(1)))
                except Exception:
                    pass

    return max_n + 1


# ==========================================================
# Manual Quick Start (no order)
# ==========================================================
def render_manual_quick_start_no_order(
    orders_df: pd.DataFrame,
    key_prefix: str = "process_setup_manualqs",
):
    st.caption("Start a draw without an order row. Uses order-like fields and auto-creates a dataset CSV.")
    if orders_df is None:
        orders_df = pd.DataFrame()

    def _opts(col: str) -> list:
        if orders_df.empty or col not in orders_df.columns:
            return []
        vals = []
        seen = set()
        for v in orders_df[col].astype(str).tolist():
            s = safe_str(v).strip()
            if not s:
                continue
            lk = s.lower()
            if lk in seen:
                continue
            seen.add(lk)
            vals.append(s)
        return vals

    def _load_project_options() -> list:
        fp = P.projects_fiber_csv
        if not os.path.exists(fp):
            return _opts("Fiber Project")
        vals = _cached_distinct_col_values(fp, "Fiber Project", os.path.getmtime(fp))
        return vals or _opts("Fiber Project")

    def _load_preform_options() -> list:
        out = []
        seen = set()

        def _add(v: str):
            s = safe_str(v).strip()
            if not s:
                return
            lk = s.lower()
            if lk in seen:
                return
            seen.add(lk)
            out.append(s)

        # 1) from orders
        for v in _opts("Preform Number"):
            _add(v)

        # 2) from preform inventory
        try:
            if os.path.exists(P.preform_inventory_csv):
                for v in _cached_distinct_col_values(
                    P.preform_inventory_csv,
                    "Preform Number",
                    os.path.getmtime(P.preform_inventory_csv),
                ):
                    _add(v)
        except Exception:
            pass

        # 3) from existing dataset csv names: {preform}F_N.csv
        try:
            if os.path.exists(P.dataset_dir):
                pat = re.compile(r"^(.*)F_(\d+)\.csv$", re.IGNORECASE)
                for fn in os.listdir(P.dataset_dir):
                    m = pat.match(str(fn).strip())
                    if m:
                        _add(m.group(1))
        except Exception:
            pass
        return out

    geometry_options = [
        "",
        "PANDA - PM",
        "TIGER - PM",
        "Octagonal",
        "ROUND",
        "STEP INDEX",
        "Ring Core",
        "Hollow Core",
        "Photonic Crystal",
        "Custom (write in Notes)",
    ]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        project_opts = [""] + _load_project_options()
        project = safe_str(st.selectbox("Fiber Project", options=project_opts, index=0, key=f"{key_prefix}_project_pick")).strip()
    with c2:
        preform_hint = ""
        preform_opts = _load_preform_options()
        if preform_opts:
            preform_hint = f"Examples: {', '.join(preform_opts[:4])}"
        preform = safe_str(
            st.text_input(
                "Preform Number",
                value="",
                key=f"{key_prefix}_preform_input",
                placeholder=preform_hint or "e.g. P0888",
            )
        ).strip()
    with c3:
        opener = safe_str(
            st.text_input(
                "Order Opener",
                value="",
                key=f"{key_prefix}_opener_input",
                placeholder="e.g. Ohad",
            )
        ).strip()
    with c4:
        priority = st.selectbox("Priority", options=["Low", "Normal", "High"], index=1, key=f"{key_prefix}_priority")

    g1, g2, g3 = st.columns(3)
    with g1:
        geom = safe_str(st.selectbox("Fiber Geometry Type", options=geometry_options, index=0, key=f"{key_prefix}_geom_pick")).strip()
    with g2:
        req_len = st.text_input("Required Length (m)", value="", key=f"{key_prefix}_req_len")
    with g3:
        good_zones = st.text_input("Good Zones Count", value="", key=f"{key_prefix}_good_zones")
    notes = st.text_area("Order Notes", value="", height=70, key=f"{key_prefix}_notes")

    csv_name = ""
    if preform.strip():
        next_n = next_draw_index_for_preform(preform.strip(), orders_df)
        csv_name = f"{preform.strip()}F_{next_n}.csv"
        st.info(f"Auto dataset CSV name: `{csv_name}`")
    else:
        st.warning("Choose/fill `Preform Number` to enable auto dataset naming and start.")

    if st.button("✅ Start Draw (No Order)", key=f"{key_prefix}_start_btn", use_container_width=True, type="primary", disabled=not bool(preform.strip())):
        csv_path = dataset_csv_path(csv_name)
        if os.path.exists(csv_path):
            st.error(f"Dataset CSV already exists: {csv_name}")
            return

        def _section(title: str):
            return {"Parameter Name": f"=== {title} ===", "Value": "", "Units": ""}

        def _blank():
            return {"Parameter Name": "", "Value": "", "Units": ""}

        def _add_order(rows: list, name: str, val, unit: str = ""):
            rows.append({"Parameter Name": f"Order__{name}", "Value": val, "Units": unit})

        manual_row = {
            "Fiber Project": project,
            "Preform Number": preform,
            "Order Opener": opener,
            "Priority": priority,
            "Fiber Geometry Type": geom,
            "Required Length (m) (for T&M+costumer)": req_len,
            "Good Zones Count (required length zones)": good_zones,
            "Fiber Diameter (µm)": "",
            "Fiber Diameter Tol (± µm)": "",
            "Main Coating Diameter (µm)": "",
            "Main Coating Diameter Tol (± µm)": "",
            "Secondary Coating Diameter (µm)": "",
            "Secondary Coating Diameter Tol (± µm)": "",
            "Tension (g)": "",
            "Draw Speed (m/min)": "",
            "Main Coating": "",
            "Secondary Coating": "",
            "Tiger Cut (%)": "",
            "Octagonal F2F (mm)": "",
            "Notes": notes,
        }

        base_rows = []
        base_rows.append(_section("ORDER PARAMETERS"))
        _add_order(base_rows, "Draw Name", os.path.splitext(os.path.basename(csv_name))[0])
        _add_order(base_rows, "Draw Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        _add_order(base_rows, "Order Index", "")
        _add_order(base_rows, "Preform Number", manual_row["Preform Number"])
        _add_order(base_rows, "Fiber Project", manual_row["Fiber Project"])
        _add_order(base_rows, "Priority", manual_row["Priority"])
        _add_order(base_rows, "Order Opener", manual_row["Order Opener"])
        _add_order(base_rows, "Fiber Geometry Type", manual_row["Fiber Geometry Type"])
        _add_order(base_rows, "Tiger Cut (%)", manual_row["Tiger Cut (%)"], "%")
        _add_order(base_rows, "Octagonal F2F (mm)", manual_row["Octagonal F2F (mm)"], "mm")
        _add_order(base_rows, "Required Length (m) (for T&M+costumer)", manual_row["Required Length (m) (for T&M+costumer)"], "m")
        _add_order(base_rows, "Good Zones Count (required length zones)", manual_row["Good Zones Count (required length zones)"], "count")
        _add_order(base_rows, "Fiber Diameter (µm)", manual_row["Fiber Diameter (µm)"], "µm")
        _add_order(base_rows, "Fiber Diameter Tol (± µm)", manual_row["Fiber Diameter Tol (± µm)"], "µm")
        _add_order(base_rows, "Main Coating Diameter (µm)", manual_row["Main Coating Diameter (µm)"], "µm")
        _add_order(base_rows, "Main Coating Diameter Tol (± µm)", manual_row["Main Coating Diameter Tol (± µm)"], "µm")
        _add_order(base_rows, "Secondary Coating Diameter (µm)", manual_row["Secondary Coating Diameter (µm)"], "µm")
        _add_order(base_rows, "Secondary Coating Diameter Tol (± µm)", manual_row["Secondary Coating Diameter Tol (± µm)"], "µm")
        _add_order(base_rows, "Tension (g)", manual_row["Tension (g)"], "g")
        _add_order(base_rows, "Draw Speed (m/min)", manual_row["Draw Speed (m/min)"], "m/min")
        _add_order(base_rows, "Main Coating", manual_row["Main Coating"])
        _add_order(base_rows, "Secondary Coating", manual_row["Secondary Coating"])
        _add_order(base_rows, "Main Coating Temperature (°C)", "", "°C")
        _add_order(base_rows, "Secondary Coating Temperature (°C)", "", "°C")
        _add_order(base_rows, "Order Notes", manual_row["Notes"])
        base_rows.append(_blank())
        pd.DataFrame(base_rows).to_csv(csv_path, index=False)

        apply_order_row_to_process_setup_state(manual_row, overwrite=True)
        st.session_state["process_setup_last_dataset_csv"] = csv_name
        st.session_state["process_setup_last_order_idx"] = None
        st.session_state[MSG_SCHED] = f"✅ Manual Quick Start created {csv_name} (No Order mode)"
        log_event(
            "process_quick_start_manual_created",
            dataset_csv=csv_name,
            preform=preform,
            project=project,
            status="In Progress (Manual)",
        )
        try:
            record_activity_start(
                indicator_json_path=P.activity_indicator_json,
                events_csv_path=P.activity_events_csv,
                activity_type="draw",
                title=f"Draw Start | Manual | {safe_str(project)}",
                actor=safe_str(st.session_state.get("maint_actor", "operator")),
                source="process_setup_manual_quick_start",
                meta={
                    "dataset_csv": csv_name,
                    "preform": preform,
                    "project": project,
                    "mode": "manual_no_order",
                },
            )
        except Exception:
            pass
        st.rerun()


# ==========================================================
# Scheduled → Quick Start
# ==========================================================
def render_scheduled_quick_start(
    orders_df: pd.DataFrame,
    orders_file: str,
    data_set_dir: Optional[str] = None,
    key_prefix: str = "process_setup_schedqs",
):
    """
    Scheduled Orders → Quick Start:
      - Create dataset CSV in data_set_csv/
      - Link the CSV to the order (Active CSV)
      - Set Status -> In Progress
      - Auto-fill Process Setup session_state using apply_order_row_to_process_setup_state()
    """
    data_set_dir = data_set_dir or P.dataset_dir
    os.makedirs(data_set_dir, exist_ok=True)

    if orders_df is None or orders_df.empty:
        st.info("No orders file / no orders found.")
        return

    df = orders_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required columns exist (safe)
    required_defaults = {
        STATUS_COL: "",
        "Priority": "Normal",
        "Fiber Project": "",
        "Fiber Geometry Type": "",
        "Tiger Cut (%)": "",
        "Octagonal F2F (mm)": "",
        "Preform Number": "",
        "Order Opener": "",
        "Required Length (m) (for T&M+costumer)": "",
        "Good Zones Count (required length zones)": "",
        "Fiber Diameter (µm)": "",
        "Fiber Diameter Tol (± µm)": "",
        "Main Coating Diameter (µm)": "",
        "Main Coating Diameter Tol (± µm)": "",
        "Secondary Coating Diameter (µm)": "",
        "Secondary Coating Diameter Tol (± µm)": "",
        "Tension (g)": "",
        "Draw Speed (m/min)": "",
        "Main Coating": "",
        "Secondary Coating": "",
        "Main Coating Temperature (°C)": "",
        "Secondary Coating Temperature (°C)": "",
        "Notes": "",
        "Active CSV": "",
        "Timestamp": "",
        STATUS_UPDATED_COL: "",
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    scheduled = df[df[STATUS_COL].astype(str).str.strip().str.lower() == "scheduled"].copy()
    if scheduled.empty:
        st.info("No **Scheduled** orders right now.")
        return

    # Sort: High priority first, older first
    pr_map = {"high": 0, "normal": 1, "low": 2}
    scheduled["_prio_rank"] = (
        scheduled["Priority"].astype(str).str.strip().str.lower().map(lambda x: pr_map.get(x, 9))
    )
    scheduled["_ts"] = pd.to_datetime(scheduled.get("Timestamp", ""), errors="coerce")
    scheduled = scheduled.sort_values(by=["_prio_rank", "_ts"], ascending=[True, True])

    def show_order_summary(row: dict):
        colA, colB, colC = st.columns([1.2, 1.2, 1.2])
        with colA:
            st.markdown("**Core**")
            st.write(f"Project: {row.get('Fiber Project','')}")
            st.write(f"Geometry: {row.get('Fiber Geometry Type','')}")
            st.write(f"Preform: {row.get('Preform Number','')}")
            st.write(f"Priority: {row.get('Priority','')}")
        with colB:
            st.markdown("**Targets**")
            st.write(f"Fiber Ø: {row.get('Fiber Diameter (µm)','')} µm  (± {row.get('Fiber Diameter Tol (± µm)','')} µm)")
            st.write(f"Coat1 Ø: {row.get('Main Coating Diameter (µm)','')} µm  (± {row.get('Main Coating Diameter Tol (± µm)','')} µm)")
            st.write(f"Coat2 Ø: {row.get('Secondary Coating Diameter (µm)','')} µm  (± {row.get('Secondary Coating Diameter Tol (± µm)','')} µm)")
            st.write(f"Tension: {row.get('Tension (g)','')} g | Speed: {row.get('Draw Speed (m/min)','')} m/min")
        with colC:
            st.markdown("**Output**")
            st.write(f"Required length: {row.get('Required Length (m) (for T&M+costumer)','')} m")
            st.write(f"Good zones: {row.get('Good Zones Count (required length zones)','')}")
            st.write(f"Opener: {row.get('Order Opener','')}")

        st.markdown("**Materials**")
        m1, m2, m3, m4 = st.columns([1.2, 1.2, 1.0, 1.0])
        with m1:
            st.write(f"Main coating: {row.get('Main Coating','')}")
        with m2:
            st.write(f"Secondary coating: {row.get('Secondary Coating','')}")
        with m3:
            st.write(f"Main temp: {row.get('Main Coating Temperature (°C)','')} °C")
        with m4:
            st.write(f"Sec temp: {row.get('Secondary Coating Temperature (°C)','')} °C")

        geom = safe_str(row.get("Fiber Geometry Type", ""))
        if geom == "TIGER - PM":
            st.info(f"🐯 Tiger Cut (%): {row.get('Tiger Cut (%)','')}")
        if geom.lower() == "octagonal":
            st.info(f"🟪 Octagonal F2F (mm): {row.get('Octagonal F2F (mm)','')}")

        notes = safe_str(row.get("Notes", ""))
        if notes:
            st.caption(notes)

    # -----------------------------
    # CSV block helpers (ORDER block)
    # -----------------------------
    def _section(title: str):
        return {"Parameter Name": f"=== {title} ===", "Value": "", "Units": ""}

    def _blank():
        return {"Parameter Name": "", "Value": "", "Units": ""}

    def _add_order(base_rows: list, name: str, val, unit: str = ""):
        base_rows.append({"Parameter Name": f"Order__{name}", "Value": val, "Units": unit})

    for idx in scheduled.index:
        row = df.loc[idx].to_dict()
        header = (
            f"#{idx} | {row.get('Fiber Project','')} | {row.get('Fiber Geometry Type','')}"
            f" | Preform: {row.get('Preform Number','')} | Priority: {row.get('Priority','')}"
        )

        with st.expander(header, expanded=False):
            show_order_summary(row)

            st.markdown("---")
            st.markdown("#### ⚡ Create dataset CSV & auto-fill Process Setup")

            preform = safe_str(row.get("Preform Number", ""))

            if preform:
                next_n = next_draw_index_for_preform(preform, df)
                draw_name = f"{preform}F_{next_n}"
                default_csv_name = f"{draw_name}.csv"
            else:
                draw_name = "draw"
                default_csv_name = f"draw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            csv_name = st.text_input(
                "Dataset CSV filename",
                value=default_csv_name,
                key=f"{key_prefix}_csvname_{idx}",
            ).strip()

            # safety: never allow folders
            csv_name = os.path.basename(csv_name) if csv_name else os.path.basename(default_csv_name)
            if not csv_name.lower().endswith(".csv"):
                csv_name += ".csv"

            if st.button("✅ Quick Start (Create CSV)", key=f"{key_prefix}_btn_{idx}", disabled=not bool(csv_name)):
                csv_path = dataset_csv_path(csv_name)
                if os.path.exists(csv_path):
                    st.error(f"Dataset CSV already exists: {csv_name}")
                    st.stop()

                base_rows = []
                base_rows.append(_section("ORDER PARAMETERS"))

                # ✅ Draw Name consistent with filename scheme
                _add_order(base_rows, "Draw Name", os.path.splitext(os.path.basename(csv_name))[0])
                _add_order(base_rows, "Draw Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                _add_order(base_rows, "Order Index", int(idx))
                _add_order(base_rows, "Preform Number", preform)
                _add_order(base_rows, "Fiber Project", row.get("Fiber Project", ""))
                _add_order(base_rows, "Priority", row.get("Priority", ""))
                _add_order(base_rows, "Order Opener", row.get("Order Opener", ""))

                _add_order(base_rows, "Fiber Geometry Type", row.get("Fiber Geometry Type", ""))
                _add_order(base_rows, "Tiger Cut (%)", row.get("Tiger Cut (%)", ""), "%")
                _add_order(base_rows, "Octagonal F2F (mm)", row.get("Octagonal F2F (mm)", ""), "mm")

                _add_order(base_rows, "Required Length (m) (for T&M+costumer)", row.get("Required Length (m) (for T&M+costumer)", ""), "m")
                _add_order(base_rows, "Good Zones Count (required length zones)", row.get("Good Zones Count (required length zones)", ""), "count")

                _add_order(base_rows, "Fiber Diameter (µm)", row.get("Fiber Diameter (µm)", ""), "µm")
                _add_order(base_rows, "Fiber Diameter Tol (± µm)", row.get("Fiber Diameter Tol (± µm)", ""), "µm")
                _add_order(base_rows, "Main Coating Diameter (µm)", row.get("Main Coating Diameter (µm)", ""), "µm")
                _add_order(base_rows, "Main Coating Diameter Tol (± µm)", row.get("Main Coating Diameter Tol (± µm)", ""), "µm")
                _add_order(base_rows, "Secondary Coating Diameter (µm)", row.get("Secondary Coating Diameter (µm)", ""), "µm")
                _add_order(base_rows, "Secondary Coating Diameter Tol (± µm)", row.get("Secondary Coating Diameter Tol (± µm)", ""), "µm")
                _add_order(base_rows, "Tension (g)", row.get("Tension (g)", ""), "g")
                _add_order(base_rows, "Draw Speed (m/min)", row.get("Draw Speed (m/min)", ""), "m/min")

                _add_order(base_rows, "Main Coating", row.get("Main Coating", ""))
                _add_order(base_rows, "Secondary Coating", row.get("Secondary Coating", ""))
                _add_order(base_rows, "Main Coating Temperature (°C)", row.get("Main Coating Temperature (°C)", ""), "°C")
                _add_order(base_rows, "Secondary Coating Temperature (°C)", row.get("Secondary Coating Temperature (°C)", ""), "°C")
                _add_order(base_rows, "Order Notes", row.get("Notes", ""))

                base_rows.append(_blank())
                pd.DataFrame(base_rows).to_csv(csv_path, index=False)

                # ✅ Link order → Active CSV + move to In Progress
                df.loc[idx, "Active CSV"] = csv_name
                df.loc[idx, STATUS_COL] = "In Progress"
                df.loc[idx, STATUS_UPDATED_COL] = now_str()

                df.to_csv(orders_file, index=False)

                # ✅ Fill canonical keys so Process Setup shows values
                apply_order_row_to_process_setup_state(row, overwrite=True)

                st.session_state["process_setup_last_dataset_csv"] = csv_name
                st.session_state["process_setup_last_order_idx"] = int(idx)
                st.session_state[MSG_SCHED] = f"✅ Quick Start created {csv_name} and moved Order #{idx} → In Progress"
                log_event(
                    "process_quick_start_created",
                    order_index=int(idx),
                    dataset_csv=csv_name,
                    preform=preform,
                    status="In Progress",
                )
                try:
                    record_activity_start(
                        indicator_json_path=P.activity_indicator_json,
                        events_csv_path=P.activity_events_csv,
                        activity_type="draw",
                        title=f"Draw Start | Order #{int(idx)} | {safe_str(row.get('Fiber Project',''))}",
                        actor=safe_str(st.session_state.get("maint_actor", "operator")),
                        source="process_setup_quick_start",
                        meta={
                            "order_index": int(idx),
                            "dataset_csv": csv_name,
                            "preform": preform,
                            "status": "In Progress",
                        },
                    )
                except Exception:
                    pass

                st.rerun()


# ==========================================================
# Manual dataset CSV creation (optional)
# ==========================================================
def render_create_draw_dataset_csv(key_prefix: str = "ps_create"):
    st.subheader("🆕 Create New Draw Dataset CSV (manual)")

    name = st.text_input(
        "Enter unique CSV base name (without .csv)",
        "",
        key=f"{key_prefix}_name",
    ).strip()

    csv_name = f"{name}.csv" if name and not name.lower().endswith(".csv") else name
    csv_name = os.path.basename(csv_name) if csv_name else ""

    if st.button("Create CSV", key=f"{key_prefix}_btn", disabled=not bool(csv_name)):
        csv_path = dataset_csv_path(csv_name)

        if os.path.exists(csv_path):
            st.warning(f"CSV already exists: {csv_name}")
            return

        df_new = pd.DataFrame(
            [
                {"Parameter Name": "=== ORDER PARAMETERS ===", "Value": "", "Units": ""},
                {"Parameter Name": "Order__Draw Name", "Value": name, "Units": ""},
                {"Parameter Name": "Order__Draw Date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Units": ""},
                {"Parameter Name": "", "Value": "", "Units": ""},
            ],
            columns=["Parameter Name", "Value", "Units"],
        )
        df_new.to_csv(csv_path, index=False)
        st.success(f"Created: {csv_name}")


# ==========================================================
# Main tab renderer
# ==========================================================
def render_process_setup_tab(
    orders_df: pd.DataFrame,
    orders_file: str,
):
    st.markdown(
        """
        <style>
          .ps-top-spacer{ height: 8px; }
          .ps-title{
            font-size: 1.62rem;
            font-weight: 900;
            margin: 0;
            padding-top: 4px;
            line-height: 1.2;
            color: rgba(236,248,255,0.98);
            text-shadow: 0 0 14px rgba(86,178,255,0.22);
          }
          .ps-sub{
            margin: 4px 0 8px 0;
            font-size: 0.92rem;
            color: rgba(188,224,248,0.88);
          }
          .ps-line{
            height: 1px;
            margin: 0 0 12px 0;
            background: linear-gradient(90deg, rgba(120,200,255,0.58), rgba(120,200,255,0.0));
          }
          .ps-section{
            margin-top: 8px;
            margin-bottom: 8px;
            padding-left: 8px;
            border-left: 3px solid rgba(120,200,255,0.62);
            font-size: 1.04rem;
            font-weight: 820;
            color: rgba(230,246,255,0.98);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ps-top-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-title">⚙️ Process Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-sub">Quick Start -> Configure -> Save</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-line"></div>', unsafe_allow_html=True)

    # -------------------------------------------------
    # Auto-restore last saved Process setup (once per session)
    # -------------------------------------------------
    if not st.session_state.get("process_setup_restore_done", False):
        restore_file = ""
        preferred = str(st.session_state.get("process_setup_last_dataset_csv", "")).strip()
        if preferred:
            cand = dataset_csv_path(preferred)
            if os.path.exists(cand):
                restore_file = cand
        if not restore_file:
            latest = most_recent_csv(P.dataset_dir)
            if latest:
                restore_file = os.path.join(P.dataset_dir, latest)

        if restore_file and os.path.exists(restore_file):
            try:
                dfp = pd.read_csv(restore_file, keep_default_na=False)
                restored = apply_dataset_process_rows_to_state(dfp, overwrite=False)
                if restored > 0:
                    st.caption(f"Loaded last process setup from `{os.path.basename(restore_file)}`")
            except Exception:
                pass
        st.session_state["process_setup_restore_done"] = True

    # -------------------------------------------------
    # View selector
    # -------------------------------------------------
    view = process_setup_buttons()

    # -------------------------------------------------
    # Scheduled Quick Start (OWN expander)
    # -------------------------------------------------
    st.markdown('<div class="ps-section">🧭 Manual Quick Start (No Order)</div>', unsafe_allow_html=True)
    with st.expander("Open Manual Quick Start (No Order)", expanded=False):
        render_manual_quick_start_no_order(
            orders_df=orders_df if isinstance(orders_df, pd.DataFrame) else pd.DataFrame(),
            key_prefix="ps_manualqs",
        )

    st.divider()

    st.markdown('<div class="ps-section">⚡ Scheduled Orders → Quick Start</div>', unsafe_allow_html=True)
    render_scheduled_quick_start(
        orders_df=orders_df,
        orders_file=orders_file,
        data_set_dir=P.dataset_dir,
        key_prefix="ps_schedqs",
    )

    st.divider()

    # -------------------------------------------------
    # Collect data from sections
    # -------------------------------------------------
    iris_data = {}
    coating_data = {}
    pid_data = {}
    drum_data = {}

    if view in ("all", "iris"):
        iris_data = render_iris_selection_section_collect()

    if view in ("all", "coating"):
        coating_data = render_coating_section()

    if view in ("all", "pid"):
        pid_data = render_pid_tf_section_collect()

    if view in ("all", "drum"):
        drum_data = render_drum_selection_section_collect()

    # -------------------------------------------------
    # SAVE ALL — MUST BE LAST, MUST BE TOP LEVEL
    # -------------------------------------------------
    st.divider()
    render_save_all_block(
        iris_data=iris_data,
        coating_data=coating_data,
        pid_data=pid_data,
        drum_data=drum_data,
    )
