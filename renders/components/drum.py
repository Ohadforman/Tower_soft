
def render_drum_selection_section_collect():
    import streamlit as st

    st.subheader("🧵 Drum Selection")

    drum_options = [f"BN{i}" for i in range(1, 7)]

    selected_drum = st.selectbox(
        "Select Drum for this draw",
        options=drum_options,
        key="process_setup_selected_drum"
    )

    drum_data = {
        "Selected Drum": selected_drum
    }

    return drum_data