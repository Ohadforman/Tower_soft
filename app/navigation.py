from __future__ import annotations

import streamlit as st


# Sidebar groups are the single source of truth for navigation structure.
NAV_GROUPS = {
    "🏠 Home & Project Management": [
        "🏠 Home",
        "📅 Schedule",
        "📦 Order Draw",
        "🛠️ Tower Parts",
    ],
    "⚙️ Operations": [
        "🍃 Tower state - Consumables and dies",
        "⚙️ Process Setup",
        "🧰 Maintenance",
        "📊 Dashboard",
        "✅ Draw Finalize",
        "📈 Correlation & Outliers",
        "🛠️ Tower Parts",
        "📋 Protocols",
        "🩺 Data Diagnostics",
    ],
    "📚 Monitoring &  Research": [
        "🧪 SQL Lab",
        "🧪 Development Process",
    ],
}

SAFE_NAV_GROUPS = {
    "🛡 Safe Mode": [
        "🏠 Home",
        "🩺 Data Diagnostics",
        "🧪 SQL Lab",
    ]
}

TAB_TO_GROUPS = {}
for group_name, tabs in NAV_GROUPS.items():
    for tab_name in tabs:
        TAB_TO_GROUPS.setdefault(tab_name, []).append(group_name)
GROUPS = list(NAV_GROUPS.keys())


def init_session_state() -> None:
    st.session_state.setdefault("selected_tab", None)
    st.session_state.setdefault("tab_select", "🏠 Home")
    st.session_state.setdefault("last_tab", "🏠 Home")
    st.session_state.setdefault("nav_last_tab_by_group", {})
    st.session_state.setdefault("good_zones", [])


def resolve_group_for_tab(tab: str, tab_to_groups=None, fallback: str = "🏠 Home & Project Management") -> str:
    tab_to_groups = tab_to_groups or TAB_TO_GROUPS
    groups_for_tab = tab_to_groups.get(tab, [])
    current_group = st.session_state.get("nav_group_select")
    if current_group in groups_for_tab:
        return current_group
    if fallback in groups_for_tab:
        return fallback
    if groups_for_tab:
        return groups_for_tab[0]
    return fallback


def _build_groups(safe_mode: bool):
    groups = SAFE_NAV_GROUPS if safe_mode else NAV_GROUPS
    tab_to_groups = {}
    for group_name, tabs in groups.items():
        for tab_name in tabs:
            tab_to_groups.setdefault(tab_name, []).append(group_name)
    return groups, tab_to_groups, list(groups.keys())


def render_sidebar_navigation(safe_mode: bool = False) -> str:
    nav_groups, tab_to_groups, groups = _build_groups(safe_mode)

    with st.sidebar:
        st.markdown("### 📌 Navigation")

        desired_tab = (
            st.session_state.get("selected_tab")
            or st.session_state.get("tab_select")
            or st.session_state.get("last_tab")
            or "🏠 Home"
        )
        st.session_state["selected_tab"] = None

        if desired_tab not in tab_to_groups:
            desired_tab = list(nav_groups.values())[0][0]

        desired_group = resolve_group_for_tab(desired_tab, tab_to_groups=tab_to_groups)
        jump_tab = desired_tab
        jump_group = resolve_group_for_tab(jump_tab, tab_to_groups=tab_to_groups, fallback=desired_group)

        st.session_state["nav_group_select"] = jump_group
        st.session_state["tab_select"] = jump_tab
        st.session_state["last_tab"] = jump_tab
        st.session_state["nav_last_tab_by_group"][jump_group] = jump_tab

        def _on_group_change():
            group = st.session_state.get("nav_group_select")
            last_by_group = st.session_state.get("nav_last_tab_by_group", {})
            next_tab = last_by_group.get(group, nav_groups.get(group, [None])[0])
            if next_tab:
                st.session_state["tab_select"] = next_tab
                st.session_state["last_tab"] = next_tab
                st.session_state["nav_last_tab_by_group"][group] = next_tab

        def _on_page_change():
            tab = st.session_state.get("tab_select")
            group = st.session_state.get("nav_group_select", "🏠 Home & Project Management")
            if tab not in nav_groups.get(group, []):
                group = resolve_group_for_tab(tab, tab_to_groups=tab_to_groups, fallback=group)
                st.session_state["nav_group_select"] = group
            st.session_state["last_tab"] = tab
            st.session_state["nav_last_tab_by_group"][group] = tab

        group = st.selectbox(
            "📁 Group",
            groups,
            key="nav_group_select",
            on_change=_on_group_change,
        )

        current_tab = st.session_state.get("tab_select", desired_tab)
        if current_tab not in nav_groups[group]:
            current_tab = st.session_state.get("nav_last_tab_by_group", {}).get(group, nav_groups[group][0])
            st.session_state["tab_select"] = current_tab

        tab_selection = st.radio(
            "📄 Page",
            nav_groups[group],
            key="tab_select",
            on_change=_on_page_change,
        )

        st.session_state["last_tab"] = tab_selection
        st.session_state["nav_last_tab_by_group"][group] = tab_selection
        return tab_selection
