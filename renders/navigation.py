def render_navigation(NAV_GROUPS, TAB_TO_GROUP):
    import streamlit as st

    st.markdown("### 📌 Navigation")
    GROUPS = list(NAV_GROUPS.keys())

    def _on_group_change():
        g = st.session_state["nav_group_select"]
        next_tab = st.session_state["nav_last_tab_by_group"].get(
            g, NAV_GROUPS[g][0]
        )
        st.session_state["tab_select"] = next_tab
        st.session_state["last_tab"] = next_tab
        st.session_state["nav_last_tab_by_group"][g] = next_tab

    def _on_page_change():
        t = st.session_state["tab_select"]
        g = TAB_TO_GROUP[t]
        st.session_state["last_tab"] = t
        st.session_state["nav_last_tab_by_group"][g] = t

    group = st.selectbox(
        "📁 Group",
        GROUPS,
        index=GROUPS.index(st.session_state["nav_group_select"]),
        key="nav_group_select",
        on_change=_on_group_change,
    )

    current_tab = st.session_state["tab_select"]
    if current_tab not in NAV_GROUPS[group]:
        current_tab = NAV_GROUPS[group][0]
        st.session_state["tab_select"] = current_tab

    tab_selection = st.radio(
        "📄 Page",
        NAV_GROUPS[group],
        key="tab_select",
        on_change=_on_page_change,
    )

    return tab_selection