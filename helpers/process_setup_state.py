# helpers/process_setup_state.py
import streamlit as st

def apply_order_row_to_process_setup_state(order_row: dict, overwrite: bool = True):
    """
    Canonical mapping: Order row -> session_state keys used by BOTH Order tab and Process Setup.
    """
    def _safe_float(x, default=None):
        try:
            s = str(x).strip()
            if s == "":
                return default
            return float(s)
        except Exception:
            return default

    mapping = {
        "Main Coating": ("order_coating_main", str),
        "Secondary Coating": ("order_coating_secondary", str),
        "Main Coating Temperature (°C)": ("order_main_coat_temp_c", float),
        "Secondary Coating Temperature (°C)": ("order_sec_coat_temp_c", float),

        "Fiber Diameter (µm)": ("order_fiber_diam", float),
        "Main Coating Diameter (µm)": ("order_main_diam", float),
        "Secondary Coating Diameter (µm)": ("order_sec_diam", float),
        "Tension (g)": ("order_tension", float),
        "Draw Speed (m/min)": ("order_speed", float),

        "Fiber Geometry Type": ("order_fiber_geometry_required", str),
        "Tiger Cut (%)": ("order_tiger_cut_pct", float),
        "Octagonal F2F (mm)": ("order_oct_f2f_mm", float),

        "Preform Number": ("process_setup_prefill_preform_number", str),
        "Fiber Project": ("process_setup_prefill_fiber_project", str),
        "Notes": ("process_setup_prefill_notes", str),
        "Priority": ("process_setup_prefill_priority", str),
        "Order Opener": ("process_setup_prefill_opener", str),
        "Required Length (m) (for T&M+costumer)": ("process_setup_prefill_required_length_m", float),
        "Good Zones Count (required length zones)": ("process_setup_prefill_good_zones_count", float),
    }

    def is_empty(v):
        return str(v).strip().lower() in ("", "0", "0.0", "none", "nan")

    for order_col, (key, caster) in mapping.items():
        raw = order_row.get(order_col, "")
        if caster is float:
            val = _safe_float(raw, default="")
        else:
            val = str(raw).strip()

        if overwrite:
            st.session_state[key] = val
        else:
            cur = st.session_state.get(key, "")
            if is_empty(cur):
                st.session_state[key] = val