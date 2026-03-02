def render_iris_selection_section_collect():
    """
    Iris Selection Tool (Process Setup) with Order->Auto fill.

    Reads canonical Order keys:
      - order_fiber_geometry_required
      - order_tiger_cut_pct
      - order_oct_f2f_mm

    Rules:
      - PM iris system is ONLY for PANDA - PM (not TIGER, not Octagonal).
      - TIGER: auto preform diameter = 16 mm
      - PANDA: auto preform diameter = 28 mm + PM auto enabled (iris forced to 37 mm)
      - Octagonal: uses F2F -> area -> equivalent diameter for simple iris calc

    Iris auto-pick:
      - Pick iris diameter from a list (including half steps) that makes
        gap_area closest to target_gap_mm2 (default 200 mm^2).

    Outputs (keys stable for Save-All):
      Tiger Cut (%)
      Effective Preform Diameter (mm)
      Selected Iris Diameter (mm)
      Gap Area (mm^2)
      Is Octagonal
      Preform Shape
      Octagonal F2F (mm)
      PM Iris System
      Iris Mode
      Base Area (mm^2)
      Adjusted Area (mm^2)
      Tiger Preform
      Circular Diameter (mm)
    """
    import math
    import streamlit as st
    import hashlib

    # -------------------------
    # Safe helpers
    # -------------------------
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

    # -------------------------
    # Order inputs
    # -------------------------
    order_geom = _ss(st.session_state.get("order_fiber_geometry_required", ""), "")
    order_tiger_pct = _sf(st.session_state.get("order_tiger_cut_pct", 0.0), 0.0)
    order_oct_f2f = _sf(st.session_state.get("order_oct_f2f_mm", 0.0), 0.0)

    is_panda = (order_geom.strip() == "PANDA - PM")
    is_tiger = (order_geom.strip() == "TIGER - PM")
    is_oct   = (order_geom.strip() == "Octagonal")

    # -------------------------
    # Iris list (include half steps)
    # You can replace with your real stock list if you have it in config.
    # -------------------------
    IRIS_OPTIONS_MM = [
        18.0, 18.5, 19.0, 19.5, 20.0, 20.5,
        21.0, 21.5, 22.0, 22.5, 23.0, 23.5,
        24.0, 24.5, 25.0, 25.5, 26.0, 26.5,
        27.0, 27.5, 28.0, 28.5, 29.0, 29.5,
        30.0, 30.5, 31.0, 31.5, 32.0, 32.5,
        33.0, 33.5, 34.0, 34.5, 35.0, 35.5,
        36.0, 36.5, 37.0, 37.5, 38.0, 38.5,
        39.0, 39.5, 40.0
    ]

    # Default target gap area (you said 200mm^2)
    st.session_state.setdefault("iris_target_gap_mm2", 200.0)

    # -------------------------
    # Init session keys BEFORE widgets
    # -------------------------
    st.session_state.setdefault("iris_preform_value_mm", 0.0)     # diameter OR F2F
    st.session_state.setdefault("iris_is_oct", False)
    st.session_state.setdefault("iris_is_tiger", False)
    st.session_state.setdefault("iris_tiger_cut_pct", 0.0)
    st.session_state.setdefault("iris_pm_system", False)

    st.session_state.setdefault("iris_selected_diam_mm", 20.5)
    st.session_state.setdefault("iris_last_autofill_sig", "")

    # -------------------------
    # Auto-apply from order when order changes
    # -------------------------
    sig = f"{order_geom}|{order_tiger_pct:.3f}|{order_oct_f2f:.3f}"
    sig_hash = hashlib.md5(sig.encode("utf-8")).hexdigest()

    if st.session_state.get("iris_last_autofill_sig", "") != sig_hash:
        # Reset geometry flags first to avoid stale carry-over
        st.session_state["iris_is_oct"] = False
        st.session_state["iris_is_tiger"] = False
        st.session_state["iris_pm_system"] = False
        st.session_state["iris_tiger_cut_pct"] = 0.0

        # Octagonal
        if is_oct:
            st.session_state["iris_is_oct"] = True
            if order_oct_f2f > 0:
                st.session_state["iris_preform_value_mm"] = float(order_oct_f2f)

        # Tiger
        if is_tiger:
            st.session_state["iris_is_tiger"] = True
            st.session_state["iris_is_oct"] = False  # tiger is not oct
            st.session_state["iris_pm_system"] = False  # PM NOT for tiger
            st.session_state["iris_preform_value_mm"] = 16.0  # ✅ forced
            if order_tiger_pct > 0:
                st.session_state["iris_tiger_cut_pct"] = float(order_tiger_pct)

        # Panda
        if is_panda:
            st.session_state["iris_is_oct"] = False
            st.session_state["iris_is_tiger"] = False
            st.session_state["iris_preform_value_mm"] = 28.0  # ✅ forced default
            st.session_state["iris_pm_system"] = True          # ✅ auto-check

        st.session_state["iris_last_autofill_sig"] = sig_hash

    # -------------------------
    # UI
    # -------------------------
    st.header("🔎 Iris Selection Tool")
    st.subheader("Inputs")

    # If Octagonal: the value is F2F
    st.number_input(
        "Preform Diameter / F2F (mm)",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="iris_preform_value_mm",
        help="If Octagonal is ON, this value is interpreted as F2F (across flats). Otherwise it is circular diameter.",
    )

    st.subheader("⚙️ Options")

    c1, c2 = st.columns([1.2, 1.2])

    with c1:
        st.toggle(
            "Octagonal preform (interpret value as F2F)",
            key="iris_is_oct",
            disabled=is_oct,
            help="When enabled, Preform input is treated as F2F (mm) for a regular octagon.",
        )
        if is_oct:
            st.caption("🔒 Order geometry is **Octagonal** → this stays ON.")

    with c2:
        if is_panda:
            st.checkbox("PM iris system (auto iris = 37 mm)", key="iris_pm_system")
        else:
            st.session_state["iris_pm_system"] = False
            st.caption("PM iris system is available only for **PANDA - PM** orders.")

    # PM: iris forced to 37
    pm_on = bool(st.session_state.get("iris_pm_system", False))
    if pm_on:
        st.session_state["iris_selected_diam_mm"] = 37.0

    # -------------------------
    # Calculations
    # -------------------------
    preform_val = _sf(st.session_state.get("iris_preform_value_mm", 0.0), 0.0)

    is_oct_ui = bool(st.session_state.get("iris_is_oct", False))
    is_tiger_ui = bool(st.session_state.get("iris_is_tiger", False))
    tiger_pct = _sf(st.session_state.get("iris_tiger_cut_pct", 0.0), 0.0) if is_tiger_ui else 0.0

    # Force consistency:
    # Octagonal -> no tiger, no PM
    if is_oct_ui:
        is_tiger_ui = False
        tiger_pct = 0.0
        st.session_state["iris_is_tiger"] = False
        st.session_state["iris_tiger_cut_pct"] = 0.0
        st.session_state["iris_pm_system"] = False
        pm_on = False

    # Determine base area + equivalent diameter
    if preform_val <= 0:
        st.info("Enter a preform value to calculate.")
        return {
            "Tiger Cut (%)": tiger_pct,
            "Effective Preform Diameter (mm)": "",
            "Selected Iris Diameter (mm)": "",
            "Gap Area (mm^2)": "",
            "Is Octagonal": is_oct_ui,
            "Preform Shape": "Octagonal" if is_oct_ui else ("Tiger Cut" if is_tiger_ui else "Circular"),
            "Octagonal F2F (mm)": preform_val if is_oct_ui else "",
            "PM Iris System": pm_on,
            "Iris Mode": "PM Auto" if pm_on else "Manual",
            "Base Area (mm^2)": "",
            "Adjusted Area (mm^2)": "",
            "Tiger Preform": is_tiger_ui,
            "Circular Diameter (mm)": preform_val if not is_oct_ui else "",
        }

    base_area = 0.0
    circular_d = None
    oct_f2f = None

    # ✅ Octagonal: convert F2F -> area of regular octagon
    # Regular octagon:
    # across flats (F2F) = a*(1+sqrt(2)), where a = side length
    # Area = 2*(1+sqrt(2))*a^2
    if is_oct_ui:
        oct_f2f = float(preform_val)
        a = oct_f2f / (1.0 + math.sqrt(2.0))
        base_area = 2.0 * (1.0 + math.sqrt(2.0)) * (a ** 2)
    else:
        circular_d = float(preform_val)
        base_area = math.pi * (circular_d / 2.0) ** 2

    # Tiger adjustment: area reduction model
    adj_area = base_area
    if is_tiger_ui and tiger_pct > 0:
        frac = max(0.0, min(1.0, 1.0 - tiger_pct / 100.0))
        adj_area = base_area * frac

    # Effective diameter from adjusted area (area-equivalent)
    eff_d = 2.0 * math.sqrt(max(adj_area, 0.0) / math.pi)

    # -------------------------
    # Auto-pick iris diameter from list to match target gap area
    # -------------------------
    target_gap = _sf(st.session_state.get("iris_target_gap_mm2", 200.0), 200.0)

    def gap_for_iris(d_mm: float) -> float:
        iris_area = math.pi * (float(d_mm) / 2.0) ** 2
        return iris_area - adj_area

    # If PM is ON -> iris fixed at 37
    if pm_on:
        iris_d = 37.0
    else:
        best_d = None
        best_err = 1e99
        for d_mm in IRIS_OPTIONS_MM:
            g = gap_for_iris(d_mm)
            err = abs(g - target_gap)
            if err < best_err:
                best_err = err
                best_d = d_mm
        iris_d = float(best_d) if best_d is not None else float(st.session_state.get("iris_selected_diam_mm", 20.5))
        st.session_state["iris_selected_diam_mm"] = iris_d

    # Final computed gap/iris area
    iris_area = math.pi * (iris_d / 2.0) ** 2
    gap_area = iris_area - adj_area

    # -------------------------
    # UI summary
    # -------------------------
    shape_lbl = "Octagonal" if is_oct_ui else ("Tiger Cut" if is_tiger_ui else "Circular")

    st.markdown("### Results")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Preform Shape", shape_lbl)
    with r2:
        st.metric("Base Area (mm²)", f"{base_area:.2f}")
    with r3:
        st.metric("Adjusted Area (mm²)", f"{adj_area:.2f}")
    with r4:
        st.metric("Effective Diameter (mm)", f"{eff_d:.3f}")

    st.markdown("### Iris Gap")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("Target Gap (mm²)", f"{target_gap:.1f}")
    with g2:
        st.metric("Iris Diameter (mm)", f"{iris_d:.2f}")
    with g3:
        st.metric("Iris Area (mm²)", f"{iris_area:.2f}")
    with g4:
        st.metric("Gap Area (mm²)", f"{gap_area:.2f}", delta=f"{(gap_area - target_gap):+.2f} mm²")

    # -------------------------
    # Return dict (Save-All uses these)
    # -------------------------
    return {
        "Tiger Cut (%)": float(tiger_pct),
        "Effective Preform Diameter (mm)": float(eff_d),
        "Selected Iris Diameter (mm)": float(iris_d),
        "Gap Area (mm^2)": float(gap_area),

        "Is Octagonal": bool(is_oct_ui),
        "Preform Shape": shape_lbl,
        "Octagonal F2F (mm)": float(oct_f2f) if oct_f2f is not None else "",

        "PM Iris System": bool(pm_on),
        "Iris Mode": "PM Auto" if pm_on else "Manual",

        "Base Area (mm^2)": float(base_area),
        "Adjusted Area (mm^2)": float(adj_area),
        "Tiger Preform": bool(is_tiger_ui),
        "Circular Diameter (mm)": float(circular_d) if circular_d is not None else "",
    }