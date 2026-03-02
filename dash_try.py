import os
import pandas as pd
import json
import streamlit as st
from app_io.paths import P
from app_io.legacy_path_compat import install_legacy_path_compat

# Install path compatibility BEFORE app module logic imports can touch legacy root names.
install_legacy_path_compat(P)

from helpers.orders_status import parse_dt_safe
from helpers.text_utils import safe_str, now_str
from app_io.config import coating_options_from_cfg
from helpers.constants import (
    STATUS_COL,
    STATUS_UPDATED_COL,
    FAILED_DESC_COL,
)
from helpers.json_io import load_json
from renders.support.assets import get_base64_image
from helpers.duckdb_io import get_duckdb_conn
from renders.tabs.corr_outliers import render_corr_outliers_tab
from renders.tabs.dashboard_tab import render_dashboard_tab
from renders.tabs.home_tab import render_home_tab
from renders.tabs.draw_finalize_tab import render_draw_finalize_tab
from renders.tabs.development_process_tab import render_development_process_tab
from renders.tabs.maintenance_tab import render_maintenance_tab
from renders.tabs.order_draw_tab import render_order_draw_tab
from renders.tabs.process_setup_tab import render_process_setup_tab_main
from renders.tabs.protocols import render_protocols_tab
from renders.tabs.schedule_tab import render_schedule_tab
from renders.tabs.sql_lab import render_sql_lab_tab
from renders.tabs.tower_parts_tab import render_tower_parts_tab
from renders.tabs.consumables_tab import render_consumables_tab
from renders.components.home_sections import (
    render_home_draw_orders_overview,
    render_schedule_home_minimal,
    render_parts_orders_home_all,
    render_done_home_section,
)

FAILED_REASON_COL = FAILED_DESC_COL

CSV_SELECTION_FILE = P.selected_csv_json
SAP_INVENTORY_FILE = P.sap_rods_inventory_csv
DB_PATH = P.duckdb_path
PID_CONFIG_PATH = P.pid_config_json
ORDERS_FILE = P.orders_csv
PREFORMS_FILE = P.preform_inventory_csv
DATA_FOLDER = P.logs_dir
HISTORY_FILE = P.history_csv
PARTS_DIRECTORY = P.parts_dir
DEVELOPMENT_FILE = P.development_csv
DATASET_FOLDER = P.dataset_dir
con = get_duckdb_conn(P.duckdb_path)
DATASET_DIR = P.dataset_dir

image_base64 = get_base64_image(P.home_bg_image)

st.set_page_config(
    page_title="Tower",
    layout="wide",
    initial_sidebar_state="collapsed",  # <-- default collapsed
)
@st.cache_data(show_spinner=False)
def load_coating_config():
    return load_json(P.coating_config_json)


coating_cfg = load_coating_config()
COATING_OPTIONS = coating_options_from_cfg(coating_cfg)
if not COATING_OPTIONS:
    COATING_OPTIONS = [""]  # safe fallback so selectbox won't crash



# Load coatings and dies from the configuration file
try:
    config = load_json(P.coating_config_json)
except FileNotFoundError:
    st.error(f"Missing coating config file: {P.coating_config_json}")
    st.stop()

coatings = config.get("coatings", {})
dies = config.get("dies", {})
# Ensure coatings and dies are properly loaded
if not coatings or not dies:
    st.error("Coatings and/or Dies not configured in config_coating.json")
    st.stop()

# =========================================================
# Minimal global session state (NO logic here)
# =========================================================
if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = None

if "tab_select" not in st.session_state:
    st.session_state["tab_select"] = "🏠 Home"

if "last_tab" not in st.session_state:
    st.session_state["last_tab"] = "🏠 Home"

if "nav_last_tab_by_group" not in st.session_state:
    st.session_state["nav_last_tab_by_group"] = {}

if "good_zones" not in st.session_state:
    st.session_state["good_zones"] = []



# ---------------- Sidebar Navigation (Grouped, stable) ----------------
with st.sidebar:
    st.markdown("### 📌 Navigation")

    NAV_GROUPS = {
        "🏠 Home & Project Management": [
            "🏠 Home",
            "📅 Schedule",
            "📦 Order Draw",
            "🛠️ Tower Parts"
        ],
        "⚙️ Operations": [
            "🍃 Tower state - Consumables and dies",
            "⚙️ Process Setup",
            "🧰 Maintenance",
            "📊 Dashboard",
            "✅ Draw Finalize",
            "📈 Correlation & Outliers",
            "🛠️ Tower Parts",
            "📋 Protocols"
        ],
        "📚 Monitoring &  Research": [
            "🧪 SQL Lab",
            "🧪 Development Process",
        ],

    }

    TAB_TO_GROUP = {t: g for g, tabs in NAV_GROUPS.items() for t in tabs}
    TAB_TO_GROUPS = {}
    for g, tabs in NAV_GROUPS.items():
        for t in tabs:
            TAB_TO_GROUPS.setdefault(t, []).append(g)
    GROUPS = list(NAV_GROUPS.keys())

    def _resolve_group_for_tab(tab: str, fallback: str = "🏠 Home & Project Management") -> str:
        groups_for_tab = TAB_TO_GROUPS.get(tab, [])
        current_group = st.session_state.get("nav_group_select")
        if current_group in groups_for_tab:
            return current_group
        if fallback in groups_for_tab:
            return fallback
        if groups_for_tab:
            return groups_for_tab[0]
        return fallback

    # If a shortcut tab was set elsewhere, honor it once
    desired_tab = (
            st.session_state.get("selected_tab")
            or st.session_state.get("tab_select")
            or st.session_state.get("last_tab")
            or "🏠 Home"
    )
    st.session_state["selected_tab"] = None

    if desired_tab not in TAB_TO_GROUPS:
        desired_tab = "🏠 Home"

    desired_group = _resolve_group_for_tab(desired_tab)

    # remember last tab per group
    if "nav_last_tab_by_group" not in st.session_state:
        st.session_state["nav_last_tab_by_group"] = {}

    # force state consistent BEFORE widgets render
    # force state consistent BEFORE widgets render
    # ✅ If we got a "selected_tab" shortcut, FORCE jump (group + page)
    jump_tab = desired_tab  # desired_tab already includes selected_tab if provided

    jump_group = _resolve_group_for_tab(jump_tab, desired_group)

    st.session_state["nav_group_select"] = jump_group
    st.session_state["tab_select"] = jump_tab
    st.session_state["last_tab"] = jump_tab

    # (optional) keep last_tab_by_group correct
    st.session_state.setdefault("nav_last_tab_by_group", {})
    st.session_state["nav_last_tab_by_group"][jump_group] = jump_tab


    def _on_group_change():
        g = st.session_state.get("nav_group_select")
        last_by_group = st.session_state.get("nav_last_tab_by_group", {})
        next_tab = last_by_group.get(g, NAV_GROUPS.get(g, [None])[0])
        if next_tab:
            st.session_state["tab_select"] = next_tab
            st.session_state["last_tab"] = next_tab
            st.session_state.setdefault("nav_last_tab_by_group", {})
            st.session_state["nav_last_tab_by_group"][g] = next_tab


    def _on_page_change():
        t = st.session_state.get("tab_select")
        g = st.session_state.get("nav_group_select", "🏠 Home & Project Management")
        if t not in NAV_GROUPS.get(g, []):
            g = _resolve_group_for_tab(t, g)
            st.session_state["nav_group_select"] = g
        st.session_state["last_tab"] = t
        st.session_state.setdefault("nav_last_tab_by_group", {})
        st.session_state["nav_last_tab_by_group"][g] = t


    group = st.selectbox(
        "📁 Group",
        GROUPS,
        index=GROUPS.index(st.session_state.get("nav_group_select", desired_group)),
        key="nav_group_select",
        on_change=_on_group_change,
    )

    # ensure current page valid for current group
    current_tab = st.session_state.get("tab_select", desired_tab)
    if current_tab not in NAV_GROUPS[group]:
        current_tab = st.session_state.get("nav_last_tab_by_group", {}).get(group, NAV_GROUPS[group][0])
        st.session_state["tab_select"] = current_tab

    tab_selection = st.radio(
        "📄 Page",
        NAV_GROUPS[group],
        key="tab_select",
        on_change=_on_page_change,
    )

    # final sync
    st.session_state["last_tab"] = tab_selection
    st.session_state["nav_last_tab_by_group"][group] = tab_selection

df = pd.DataFrame()  # Initialize an empty DataFrame to avoid NameError


# ------------------ Home Tab ------------------
if tab_selection == "🏠 Home":
    render_home_tab(
        P=P,
        image_base64=image_base64,
        STATUS_COL=STATUS_COL,
        STATUS_UPDATED_COL=STATUS_UPDATED_COL,
        FAILED_REASON_COL=FAILED_REASON_COL,
        parse_dt_safe=parse_dt_safe,
        now_str=now_str,
        safe_str=safe_str,
        render_home_draw_orders_overview=render_home_draw_orders_overview,
        render_done_home_section=render_done_home_section,
        render_schedule_home_minimal=render_schedule_home_minimal,
        render_parts_orders_home_all=render_parts_orders_home_all,
    )
# ------------------ Process Tab ------------------
elif tab_selection == "⚙️ Process Setup":
    render_process_setup_tab_main(P, ORDERS_FILE)
elif tab_selection == "📊 Dashboard":
    render_dashboard_tab(P)
elif tab_selection == "✅ Draw Finalize":
    render_draw_finalize_tab(P)
# ------------------ Consumables Tab ------------------
elif tab_selection == "🍃 Tower state - Consumables and dies":
    render_consumables_tab(P)
# ------------------ Schedule Tab ------------------
elif tab_selection == "📅 Schedule":
    render_schedule_tab(P)
# ------------------ Order draw ------------------
elif tab_selection == "📦 Order Draw":
    render_order_draw_tab(P)
elif tab_selection == "🛠️ Tower Parts":
    render_tower_parts_tab(P)
elif tab_selection == "🧪 Development Process":
    render_development_process_tab(P)
elif tab_selection == "📋 Protocols":
    render_protocols_tab()
# ------------------ Maintenance Tab ------------------
elif tab_selection == "🧰 Maintenance":
    render_maintenance_tab(P)
elif tab_selection == "📈 Correlation & Outliers":
    st.title("📈 Correlation & Outliers")
    st.caption(
        "Builds a numeric snapshot per log file (time = log file mtime), then plots rolling correlation vs time "
        "for MANY column pairs."
    )
    render_corr_outliers_tab(draw_folder=P.logs_dir, maint_folder=P.maintenance_dir)
# ------------------ SQL Lab ------------------
elif tab_selection == "🧪 SQL Lab":
    render_sql_lab_tab(P)
