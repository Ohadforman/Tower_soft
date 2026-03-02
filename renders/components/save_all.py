# renders/save_all.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from app_io.paths import P, dataset_csv_path
from helpers.text_utils import safe_str
from helpers.dataset_io import (
    ensure_dataset_dir,
    list_dataset_csvs,
    most_recent_csv,
    append_rows_to_dataset_csv,
)
from helpers.json_io import load_json


def render_save_all_block(
    iris_data: Dict[str, Any],
    coating_data: Dict[str, Any],
    pid_data: Dict[str, Any],
    drum_data: Dict[str, Any],
):
    """
    Save Everything (one click) → appends rows to an existing dataset CSV.

    FIX (this version):
    - All appended rows are clearly marked as Process setup:  Process__<name>
    - Adds a section header before the process block
    - Adds a buffer + NEXT SECTION marker after Selected Drum
    """
    st.subheader("💾 Save Everything (one click)")

    # -----------------------------
    # Dataset files
    # -----------------------------
    ensure_dataset_dir(P.dataset_dir)
    csv_files = list_dataset_csvs(P.dataset_dir, full_paths=False)
    latest = most_recent_csv(P.dataset_dir)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _die_diameter_um_from_cfg(cfg: dict, die_name: str) -> str:
        try:
            dies = (cfg or {}).get("dies", {})
            dd = (dies or {}).get(str(die_name), {}) or {}
            val = dd.get("Die_Diameter", "")
            return "" if val is None else str(val)
        except Exception:
            return ""

    def _val(x) -> str:
        return "" if x is None else str(x)

    def _section(title: str) -> Dict[str, Any]:
        return {"Parameter Name": f"=== {title} ===", "Value": "", "Units": ""}

    def _blank() -> Dict[str, Any]:
        return {"Parameter Name": "", "Value": "", "Units": ""}

    def _proc(name: str) -> str:
        name = safe_str(name).strip()
        return f"Process__{name}" if name else ""

    # -----------------------------
    # Build rows (PROCESS BLOCK)
    # -----------------------------
    rows: List[Dict[str, Any]] = [
        _section("PROCESS SETUP"),
        {
            "Parameter Name": _proc("Process Setup Timestamp"),
            "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Units": ""
        }
    ]

    save_to_latest_clicked = st.button("⚡ Save ALL to MOST RECENT CSV", key="ps_saveall_to_latest")

    # =========================
    # IRIS rows (from iris_data)
    # =========================
    if iris_data:
        tiger_pct = iris_data.get("Tiger Cut (%)", 0)
        eff_d = iris_data.get("Effective Preform Diameter (mm)")
        sel_iris = iris_data.get("Selected Iris Diameter (mm)")
        gap_area = iris_data.get("Gap Area (mm^2)")

        is_oct = bool(iris_data.get("Is Octagonal", False))
        shape_lbl = safe_str(iris_data.get("Preform Shape", ""))
        oct_f2f = iris_data.get("Octagonal F2F (mm)")

        pm_sys = bool(iris_data.get("PM Iris System", False))
        iris_mode = safe_str(iris_data.get("Iris Mode", ""))

        base_area = iris_data.get("Base Area (mm^2)")
        adj_area = iris_data.get("Adjusted Area (mm^2)")
        tiger_flag = bool(iris_data.get("Tiger Preform", False))
        circ_d = iris_data.get("Circular Diameter (mm)")

        rows += [
            {"Parameter Name": _proc("Preform Diameter"), "Value": _val(circ_d), "Units": "mm"},
            {"Parameter Name": _proc("Preform Shape"), "Value": shape_lbl, "Units": ""},

            {"Parameter Name": _proc("Octagonal Preform"), "Value": 1 if is_oct else 0, "Units": "bool"},
            {"Parameter Name": _proc("Octagonal F2F"), "Value": _val(oct_f2f), "Units": "mm"},

            {"Parameter Name": _proc("Tiger Preform"), "Value": 1 if tiger_flag else 0, "Units": "bool"},
            {"Parameter Name": _proc("Tiger Cut"), "Value": _val(tiger_pct), "Units": "%"},

            {"Parameter Name": _proc("PM Iris System"), "Value": 1 if pm_sys else 0, "Units": "bool"},
            {"Parameter Name": _proc("Iris Mode"), "Value": iris_mode, "Units": ""},

            {"Parameter Name": _proc("Base Area"), "Value": _val(base_area), "Units": "mm^2"},
            {"Parameter Name": _proc("Adjusted Area"), "Value": _val(adj_area), "Units": "mm^2"},
            {"Parameter Name": _proc("Effective Preform Diameter"), "Value": _val(eff_d), "Units": "mm"},

            {"Parameter Name": _proc("Selected Iris Diameter"), "Value": _val(sel_iris), "Units": "mm"},
            {"Parameter Name": _proc("Iris Gap Area"), "Value": _val(gap_area), "Units": "mm^2"},
        ]

    # =========================
    # COATING rows (CANONICAL from session_state)
    # =========================
    coating_cfg = load_json(P.coating_config_json)

    entry_um = st.session_state.get("order_fiber_diam", "")
    tgt1_um = st.session_state.get("order_main_diam", "")
    tgt2_um = st.session_state.get("order_sec_diam", "")

    coat1 = safe_str(st.session_state.get("order_coating_main", ""))
    coat2 = safe_str(st.session_state.get("order_coating_secondary", ""))

    t1_c = st.session_state.get("order_main_coat_temp_c", "")
    t2_c = st.session_state.get("order_sec_coat_temp_c", "")

    die_mode = safe_str(st.session_state.get("coating_die_mode", "Auto"))
    p_die_name = safe_str(st.session_state.get("coating_primary_die", ""))
    s_die_name = safe_str(st.session_state.get("coating_secondary_die", ""))

    p_die_um = _die_diameter_um_from_cfg(coating_cfg, p_die_name)
    s_die_um = _die_diameter_um_from_cfg(coating_cfg, s_die_name)

    pred_fc_um = st.session_state.get("coating_pred_fc_um", "")
    pred_sc_um = st.session_state.get("coating_pred_sc_um", "")

    ideal_p = st.session_state.get("coating_ideal_primary_die_um", "")
    ideal_s = st.session_state.get("coating_ideal_secondary_die_um", "")
    ideal_fc = st.session_state.get("coating_fc_at_ideal_primary_um", "")
    ideal_sc = st.session_state.get("coating_sc_at_ideal_secondary_um", "")

    d_fc = st.session_state.get("coating_delta_fc_ideal_vs_target_um", "")
    d_sc = st.session_state.get("coating_delta_sc_ideal_vs_target_um", "")

    draw_speed_m_min = st.session_state.get("order_speed", "")

    has_any_coating = any([
        str(entry_um).strip(),
        str(tgt1_um).strip(),
        str(tgt2_um).strip(),
        coat1.strip(),
        coat2.strip(),
        str(t1_c).strip(),
        str(t2_c).strip(),
        p_die_name.strip(),
        s_die_name.strip(),
        str(pred_fc_um).strip(),
        str(pred_sc_um).strip(),
        str(ideal_p).strip(),
        str(ideal_s).strip(),
        str(ideal_fc).strip(),
        str(ideal_sc).strip(),
        str(d_fc).strip(),
        str(d_sc).strip(),
        str(draw_speed_m_min).strip(),
    ])

    if has_any_coating:
        rows += [
            {"Parameter Name": _proc("Entry Fiber Diameter"), "Value": _val(entry_um), "Units": "µm"},
            {"Parameter Name": _proc("Target First Coating Diameter"), "Value": _val(tgt1_um), "Units": "µm"},
            {"Parameter Name": _proc("Target Second Coating Diameter"), "Value": _val(tgt2_um), "Units": "µm"},

            {"Parameter Name": _proc("First Coating Diameter (Theoretical)"), "Value": _val(pred_fc_um), "Units": "µm"},
            {"Parameter Name": _proc("Second Coating Diameter (Theoretical)"), "Value": _val(pred_sc_um), "Units": "µm"},

            {"Parameter Name": _proc("Primary Coating"), "Value": coat1, "Units": ""},
            {"Parameter Name": _proc("Secondary Coating"), "Value": coat2, "Units": ""},

            {"Parameter Name": _proc("Primary Coating Temperature"), "Value": _val(t1_c), "Units": "°C"},
            {"Parameter Name": _proc("Secondary Coating Temperature"), "Value": _val(t2_c), "Units": "°C"},

            {"Parameter Name": _proc("Primary Die Diameter"), "Value": _val(p_die_um), "Units": "µm"},
            {"Parameter Name": _proc("Secondary Die Diameter"), "Value": _val(s_die_um), "Units": "µm"},

            {"Parameter Name": _proc("Primary Die Name"), "Value": p_die_name, "Units": ""},
            {"Parameter Name": _proc("Secondary Die Name"), "Value": s_die_name, "Units": ""},

            {"Parameter Name": _proc("Coating Die Selection Mode"), "Value": die_mode, "Units": ""},
        ]

        if str(draw_speed_m_min).strip():
            rows.append({"Parameter Name": _proc("Draw Speed"), "Value": _val(draw_speed_m_min), "Units": "m/min"})

        # ideal/continuous (optional)
        if str(ideal_p).strip():
            rows.append({"Parameter Name": _proc("Ideal Primary Die (µm)"), "Value": _val(ideal_p), "Units": "µm"})
        if str(ideal_s).strip():
            rows.append({"Parameter Name": _proc("Ideal Secondary Die (µm)"), "Value": _val(ideal_s), "Units": "µm"})
        if str(ideal_fc).strip():
            rows.append({"Parameter Name": _proc("Pred @ Ideal FC (µm)"), "Value": _val(ideal_fc), "Units": "µm"})
        if str(ideal_sc).strip():
            rows.append({"Parameter Name": _proc("Pred @ Ideal SC (µm)"), "Value": _val(ideal_sc), "Units": "µm"})
        if str(d_fc).strip():
            rows.append({"Parameter Name": _proc("Δ Ideal FC vs Target"), "Value": _val(d_fc), "Units": "µm"})
        if str(d_sc).strip():
            rows.append({"Parameter Name": _proc("Δ Ideal SC vs Target"), "Value": _val(d_sc), "Units": "µm"})

    # =========================
    # PID rows
    # =========================
    if pid_data:
        rows += [
            {"Parameter Name": _proc("P Gain (Diameter Control)"), "Value": _val(pid_data.get("p_gain")), "Units": ""},
            {"Parameter Name": _proc("I Gain (Diameter Control)"), "Value": _val(pid_data.get("i_gain")), "Units": ""},
            {"Parameter Name": _proc("TF Mode"), "Value": _val(pid_data.get("winder_mode")), "Units": ""},
            {"Parameter Name": _proc("Increment TF Value"), "Value": _val(pid_data.get("increment_value")), "Units": "mm"},
        ]

    # =========================
    # DRUM rows
    # =========================
    if drum_data:
        rows += [
            {"Parameter Name": _proc("Selected Drum"), "Value": _val(drum_data.get("Selected Drum")), "Units": ""}
        ]
        # ✅ buffer + next section marker after drum
        rows += [
            _blank(),
            _section("LOGS DATA SECTION"),
            _blank(),
        ]

    # Filter out Value=None only (keep blanks/sections)
    rows = [r for r in rows if r.get("Value") is not None]

    # -----------------------------
    # UI
    # -----------------------------
    colA, colB = st.columns([2, 1])
    with colA:
        selected_csv = st.selectbox(
            "Or choose a CSV from the list",
            options=[""] + csv_files,
            index=0,
            key="ps_saveall_selected_csv",
        )
    with colB:
        st.write("")
        st.caption(f"Most recent: **{latest if latest else 'None'}**")

    with st.expander("Preview what will be written", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # -----------------------------
    # Save actions
    # -----------------------------
    if st.button("💾 Save ALL to SELECTED CSV", key="ps_saveall_to_selected"):
        if not selected_csv:
            st.error("Pick a CSV first.")
        else:
            target_path = dataset_csv_path(selected_csv)
            ok, msg = append_rows_to_dataset_csv(target_path, rows)
            if ok:
                st.success(f"Saved ALL to: {selected_csv}")
            else:
                st.error(msg)

    if save_to_latest_clicked:
        if not latest:
            st.error("No CSV files found in data_set_csv/")
        else:
            latest_path = dataset_csv_path(latest)
            ok, msg = append_rows_to_dataset_csv(latest_path, rows)
            if ok:
                st.success(f"Saved ALL to most recent: {latest}")
            else:
                st.error(msg)