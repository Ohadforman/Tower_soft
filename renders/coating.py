# renders/coating.py
from __future__ import annotations

import hashlib
import streamlit as st

from helpers.text_utils import safe_str
from helpers.coating_calc import (
    auto_select_dies_from_coating_calc,
    coating_predict_fc_sc_um,
)
from app_io.paths import P
from helpers.coating_config import load_config_coating_json
from helpers.coating_calc import (
    coating_predict_fc_sc_um,
    auto_select_dies_from_coating_calc,
)
from helpers.text_utils import safe_str
coating_cfg = load_config_coating_json(P.coating_config_json)
def render_coating_section(_config: dict | None = None) -> dict:
    """
    FULL Coating render (Process Setup).
    Reads + writes CANONICAL state in st.session_state.

    Returns: coating_data dict (used by Save-All preview, etc.)
    """

    # =========================================================
    # helpers
    # =========================================================
    def _sf(v, default=0.0):
        try:
            s = str(v).strip()
            if s == "":
                return float(default)
            return float(s)
        except Exception:
            return float(default)

    def _ss(v, default=""):
        s = str(v if v is not None else "").strip()
        return s if s else default

    def _ensure_in_options(options, current_value):
        options = [str(x).strip() for x in (options or []) if str(x).strip()]
        cv = str(current_value or "").strip()
        if cv and cv not in options:
            return [cv] + options
        return options

    def _safe_index(options, current_value):
        if not options:
            return 0
        cv = str(current_value or "").strip()
        return options.index(cv) if cv in options else 0

    # =========================================================
    # load config_coating.json (REAL stock list)
    # =========================================================
    try:
        coating_cfg = load_config_coating_json(P.coating_config_json)
    except Exception as e:
        st.error(f"Failed to load coating config: {e}")
        return {}

    dies_dict = (coating_cfg or {}).get("dies", {}) or {}
    coats_dict = (coating_cfg or {}).get("coatings", {}) or {}

    DIE_NAMES = list(dies_dict.keys())
    COATING_NAMES = list(coats_dict.keys())

    if not DIE_NAMES:
        st.error("No dies found in config_coating.json → key 'dies'")
        return {}
    if not COATING_NAMES:
        st.error("No coatings found in config_coating.json → key 'coatings'")
        return {}

    # =========================================================
    # init state BEFORE widgets
    # =========================================================
    st.session_state.setdefault("order_fiber_diam", 40.0)
    st.session_state.setdefault("order_main_diam", 0.0)
    st.session_state.setdefault("order_sec_diam", 0.0)

    st.session_state.setdefault("order_coating_main", "")
    st.session_state.setdefault("order_coating_secondary", "")

    st.session_state.setdefault("order_main_coat_temp_c", 25.0)
    st.session_state.setdefault("order_sec_coat_temp_c", 25.0)

    st.session_state.setdefault("order_speed", 55.0)

    st.session_state.setdefault("coating_die_mode", "Auto")
    st.session_state.setdefault("coating_primary_die", DIE_NAMES[0])
    st.session_state.setdefault("coating_secondary_die", DIE_NAMES[0])

    st.session_state.setdefault("coating_pred_fc_um", "")
    st.session_state.setdefault("coating_pred_sc_um", "")

    st.session_state.setdefault("coating_ideal_primary_die_um", "")
    st.session_state.setdefault("coating_ideal_secondary_die_um", "")
    st.session_state.setdefault("coating_fc_at_ideal_primary_um", "")
    st.session_state.setdefault("coating_sc_at_ideal_secondary_um", "")

    st.session_state.setdefault("coating_auto_details", "")
    st.session_state.setdefault("coating_auto_last_sig", "")

    # =========================================================
    # UI
    # =========================================================
    st.header("🧴 Coating")

    st.subheader("Entry Fiber")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("40 µm", key="coating_quick_40", use_container_width=True):
            st.session_state["order_fiber_diam"] = 40.0
            st.rerun()
    with b2:
        if st.button("125 µm", key="coating_quick_125", use_container_width=True):
            st.session_state["order_fiber_diam"] = 125.0
            st.rerun()
    with b3:
        if st.button("400 µm", key="coating_quick_400", use_container_width=True):
            st.session_state["order_fiber_diam"] = 400.0
            st.rerun()

    st.number_input(
        "Entry (Bare) Fiber Diameter (µm)",
        min_value=0.0, step=0.5, format="%.1f",
        key="order_fiber_diam",
    )

    st.subheader("🎯 Targets (from Order / used by Auto calc)")

    t1, t2 = st.columns(2)
    with t1:
        st.number_input(
            "Target First Coating Diameter (µm)",
            min_value=0.0, step=0.5, format="%.1f",
            key="order_main_diam",
        )
    with t2:
        st.number_input(
            "Target Second Coating Diameter (µm)",
            min_value=0.0, step=0.5, format="%.1f",
            key="order_sec_diam",
        )

    st.subheader("🏃 Speed (from Order)")
    st.number_input(
        "Draw Speed (m/min)",
        min_value=0.0, step=0.5, format="%.2f",
        key="order_speed",
        help="Used by the viscosity model. Converted to m/s internally.",
    )

    st.subheader("Coatings & Temperatures")
    cL, cR = st.columns(2)
    coat_main_opts = [""] + COATING_NAMES
    coat_sec_opts = [""] + COATING_NAMES

    with cL:
        st.markdown("**Primary Coating**")
        st.selectbox(
            "Primary Coating",
            options=coat_main_opts,
            index=_safe_index(coat_main_opts, st.session_state.get("order_coating_main", "")),
            key="order_coating_main",
            label_visibility="collapsed",
        )
        st.number_input(
            "Primary Temp (°C)",
            min_value=-50.0, max_value=200.0, step=0.5, format="%.2f",
            key="order_main_coat_temp_c",
        )

    with cR:
        st.markdown("**Secondary Coating**")
        st.selectbox(
            "Secondary Coating",
            options=coat_sec_opts,
            index=_safe_index(coat_sec_opts, st.session_state.get("order_coating_secondary", "")),
            key="order_coating_secondary",
            label_visibility="collapsed",
        )
        st.number_input(
            "Secondary Temp (°C)",
            min_value=-50.0, max_value=200.0, step=0.5, format="%.2f",
            key="order_sec_coat_temp_c",
        )

    st.subheader("Die Selection")
    st.radio(
        "Mode",
        options=["Manual", "Auto"],
        horizontal=True,
        key="coating_die_mode",
    )

    # =========================================================
    # canonical values
    # =========================================================
    entry_um = _sf(st.session_state.get("order_fiber_diam", 0.0), 0.0)
    tgt1_um = _sf(st.session_state.get("order_main_diam", 0.0), 0.0)
    tgt2_um = _sf(st.session_state.get("order_sec_diam", 0.0), 0.0)

    t1_c = _sf(st.session_state.get("order_main_coat_temp_c", 25.0), 25.0)
    t2_c = _sf(st.session_state.get("order_sec_coat_temp_c", 25.0), 25.0)

    coat1 = _ss(st.session_state.get("order_coating_main", ""), "")
    coat2 = _ss(st.session_state.get("order_coating_secondary", ""), "")

    mode = _ss(st.session_state.get("coating_die_mode", "Auto"), "Auto")

    draw_speed_m_min = _sf(st.session_state.get("order_speed", 0.0), 0.0)
    V = (draw_speed_m_min / 60.0) if draw_speed_m_min > 0 else 0.917

    sig = f"{mode}|{entry_um}|{tgt1_um}|{tgt2_um}|{t1_c}|{t2_c}|{coat1}|{coat2}|{draw_speed_m_min}"
    sig_hash = hashlib.md5(sig.encode("utf-8")).hexdigest()

    # =========================================================
    # AUTO selection
    # =========================================================
    if mode == "Auto":
        if st.session_state.get("coating_auto_last_sig", "") != sig_hash:
            if coat1 and coat2:
                try:
                    pdie, sdie, best_fc, best_sc, details = auto_select_dies_from_coating_calc(
                        entry_um=entry_um,
                        tgt1_um=tgt1_um,
                        tgt2_um=tgt2_um,
                        coat1=coat1,
                        coat2=coat2,
                        t1_c=t1_c,
                        t2_c=t2_c,
                        config=coating_cfg,
                        draw_speed_m_min=draw_speed_m_min,
                        pulling_speed_m_s=None,
                        g_m_s2=9.80665,
                    )

                    st.session_state["coating_primary_die"] = pdie
                    st.session_state["coating_secondary_die"] = sdie
                    st.session_state["coating_pred_fc_um"] = float(best_fc)
                    st.session_state["coating_pred_sc_um"] = float(best_sc)
                    st.session_state["coating_auto_details"] = details
                    st.session_state["coating_auto_last_sig"] = sig_hash
                except Exception as e:
                    st.session_state["coating_auto_details"] = f"⚠️ Auto calc failed: {e}"
                    st.session_state["coating_auto_last_sig"] = sig_hash
            else:
                st.session_state["coating_auto_details"] = "ℹ️ Select both coatings to run Auto calc."
                st.session_state["coating_auto_last_sig"] = sig_hash

    # =========================================================
    # die dropdowns (always)
    # =========================================================
    st.markdown("#### 🎛️ Choose real-stock dies (always live)")

    die_options_prim = _ensure_in_options(DIE_NAMES, st.session_state.get("coating_primary_die", DIE_NAMES[0]))
    die_options_sec = _ensure_in_options(DIE_NAMES, st.session_state.get("coating_secondary_die", DIE_NAMES[0]))

    dL, dR = st.columns(2)
    with dL:
        st.selectbox(
            "Primary Die (real stock)",
            options=die_options_prim,
            index=_safe_index(die_options_prim, st.session_state.get("coating_primary_die", "")),
            key="coating_primary_die",
        )
    with dR:
        st.selectbox(
            "Secondary Die (real stock)",
            options=die_options_sec,
            index=_safe_index(die_options_sec, st.session_state.get("coating_secondary_die", "")),
            key="coating_secondary_die",
        )

    # =========================================================
    # LIVE calc from chosen dies
    # =========================================================
    fc_live = ""
    sc_live = ""

    if coat1 and coat2:
        try:
            p_name = str(st.session_state.get("coating_primary_die", "")).strip()
            s_name = str(st.session_state.get("coating_secondary_die", "")).strip()

            fc_tmp, sc_tmp, _aux = coating_predict_fc_sc_um(
                entry_um=entry_um,
                primary_die_name=p_name,
                secondary_die_name=s_name,
                primary_coating_name=coat1,
                secondary_coating_name=coat2,
                t1_c=t1_c,
                t2_c=t2_c,
                pulling_speed_m_s=V,
                g_m_s2=9.80665,
                coating_cfg=coating_cfg,
            )
            fc_live = float(fc_tmp)
            sc_live = float(sc_tmp)

            st.session_state["coating_pred_fc_um"] = fc_live
            st.session_state["coating_pred_sc_um"] = sc_live
        except Exception:
            fc_live = st.session_state.get("coating_pred_fc_um", "")
            sc_live = st.session_state.get("coating_pred_sc_um", "")

    details_txt = str(st.session_state.get("coating_auto_details", "")).strip()
    if details_txt:
        st.success(details_txt)

    st.subheader("📌 Results Summary")

    a, b, c, d = st.columns(4)

    # Entry fiber
    with a:
        st.metric("Entry (µm)", f"{entry_um:.2f}")

    # First coating
    with b:
        if fc_live == "" or tgt1_um <= 0:
            st.metric(
                "Chosen FC (µm)",
                "" if fc_live == "" else f"{float(fc_live):.2f}"
            )
        else:
            fc_val = float(fc_live)
            st.metric(
                "Chosen FC (µm)",
                f"{fc_val:.2f}",
                delta=f"{(fc_val - float(tgt1_um)):+.2f} µm",
                delta_color="normal",
            )

    # Second coating
    with c:
        if sc_live == "" or tgt2_um <= 0:
            st.metric(
                "Chosen SC (µm)",
                "" if sc_live == "" else f"{float(sc_live):.2f}"
            )
        else:
            sc_val = float(sc_live)
            st.metric(
                "Chosen SC (µm)",
                f"{sc_val:.2f}",
                delta=f"{(sc_val - float(tgt2_um)):+.2f} µm",
                delta_color="normal",
            )

    # Speed
    with d:
        st.metric("Speed used (m/s)", f"{float(V):.4f}")

    coating_data = {
        "Entry Fiber Diameter (µm)": entry_um,
        "Target First Coating Diameter (µm)": tgt1_um,
        "Target Second Coating Diameter (µm)": tgt2_um,
        "Primary Coating": coat1,
        "Secondary Coating": coat2,
        "Primary Coating Temperature (°C)": t1_c,
        "Secondary Coating Temperature (°C)": t2_c,
        "Coating Die Selection Mode": mode,
        "Primary Die Name": safe_str(st.session_state.get("coating_primary_die", "")),
        "Secondary Die Name": safe_str(st.session_state.get("coating_secondary_die", "")),
        "First Coating Diameter (Predicted) (µm)": st.session_state.get("coating_pred_fc_um", ""),
        "Second Coating Diameter (Predicted) (µm)": st.session_state.get("coating_pred_sc_um", ""),
        "Ideal Primary Die (µm)": st.session_state.get("coating_ideal_primary_die_um", ""),
        "Ideal Secondary Die (µm)": st.session_state.get("coating_ideal_secondary_die_um", ""),
        "Pred @ Ideal FC (µm)": st.session_state.get("coating_fc_at_ideal_primary_um", ""),
        "Pred @ Ideal SC (µm)": st.session_state.get("coating_sc_at_ideal_secondary_die_um", ""),
    }
    return coating_data