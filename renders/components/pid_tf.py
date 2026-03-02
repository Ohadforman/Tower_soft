def render_pid_tf_section_collect() -> dict:
    import streamlit as st
    import os
    from app_io.paths import P
    import json
    PID_CONFIG_PATH = P.pid_config_json
    st.subheader("🎛 PID & TF Configuration")

    # Load json defaults
    pid_config = {}
    if os.path.exists(PID_CONFIG_PATH):
        try:
            with open(PID_CONFIG_PATH, "r") as f:
                pid_config = json.load(f)
        except Exception:
            pid_config = {}

    p_gain = st.number_input(
        "P Gain (Diameter Control)",
        min_value=0.0, step=0.1,
        value=float(pid_config.get("p_gain", 1.0)),
        key="ps_pid_p_gain"
    )
    i_gain = st.number_input(
        "I Gain (Diameter Control)",
        min_value=0.0, step=0.1,
        value=float(pid_config.get("i_gain", 1.0)),
        key="ps_pid_i_gain"
    )
    winder_mode = st.selectbox(
        "TF Mode",
        ["Winder", "Straight Mode"],
        index=["Winder", "Straight Mode"].index(pid_config.get("winder_mode", "Winder")),
        key="ps_pid_tf_mode"
    )
    increment_value = st.number_input(
        "Increment Value [mm]",
        min_value=0.0, step=0.1,
        value=float(pid_config.get("increment_value", 0.5)),
        key="ps_pid_increment_value"
    )

    # Persist json immediately (optional but nice)
    if st.checkbox("Save PID defaults to pid_config.json", value=True, key="ps_pid_save_defaults"):
        new_pid = {
            "p_gain": float(p_gain),
            "i_gain": float(i_gain),
            "winder_mode": winder_mode,
            "increment_value": float(increment_value),
        }
        try:
            with open(PID_CONFIG_PATH, "w") as f:
                json.dump(new_pid, f, indent=4)
        except Exception:
            st.warning("Could not write pid_config.json (permission/path issue).")

    return {
        "p_gain": float(p_gain),
        "i_gain": float(i_gain),
        "winder_mode": winder_mode,
        "increment_value": float(increment_value),
    }