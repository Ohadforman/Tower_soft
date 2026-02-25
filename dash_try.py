import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import json
import math
from typing import Tuple, Optional, List
from helpers.vis_ds2015 import get_viscosityDS2015, get_viscosityDP1032
import streamlit as st
from hooks.after_done import run_after_done_hook
from orders.lifecycle import ensure_orders_cols
from app_io.paths import P, dataset_csv_path
from helpers.orders_status import (
    STATUS_COL, STATUS_UPDATED_COL, FAILED_REASON_COL,
    ensure_orders_cols, now_str,parse_dt_safe
)
from helpers.ui_state import safe_str_from_state
#from renders.process_setup import render_process_setup_tab
from renders.process_setup import render_scheduled_quick_start
from renders.process_setup import render_create_draw_dataset_csv,find_associated_dataset_csv
from renders.process_setup import process_setup_buttons
from renders.drum import render_drum_selection_section_collect
from renders.pid_tf import render_pid_tf_section_collect
from renders.iris import render_iris_selection_section_collect
from helpers.text_utils import safe_str, to_float, safe_int, now_str
from app_io.config import coating_options_from_cfg
from helpers.params_io import get_value, param_map, get_float_param
from helpers.format_utils import fmt_float, fmt_int
from helpers.constants import (
    STATUS_COL,
    STATUS_UPDATED_COL,
    FAILED_DESC_COL,
    TRY_COUNT_COL,
    LAST_TRY_TIME_COL,
    LAST_TRY_DATASET_COL,
    MSG_SCHED,
    MSG_FAILED,
)
from helpers.dataset_io import append_rows_to_dataset_csv
from helpers.json_io import load_json
from helpers.assets import get_base64_image
from helpers.style_utils import color_priority, color_status
from helpers.duckdb_io import get_duckdb_conn
from renders.coating import render_coating_section
# and
from renders.save_all import render_save_all_block
from renders.process_setup import render_process_setup_tab
from helpers.dataset_param_parsers import (
    param_map,
    _parse_steps,
    _parse_zones_from_end,
    zone_lengths_from_log_km,
    _parse_marked_zone_lengths,
    _find_zone_avg_values,
    build_tm_instruction_rows_auto_from_good_zones,
    build_tm_instruction_rows_auto_from_good_zones,   # ✅ ADD THIS
)
from helpers.dates import compute_next_planned_draw_date
from app_io.paths import P, ensure_logs_dir, ensure_gas_reports_dir, gas_report_path, _abs, ensure_dir
from helpers.process_setup_state import apply_order_row_to_process_setup_state
from renders.navigation import render_navigation

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

image_base64 = get_base64_image("IMG_1094.JPEG")

st.set_page_config(
    page_title="Tower",
    #layout="wide",
    initial_sidebar_state="collapsed",  # <-- default collapsed
)
@st.cache_data(show_spinner=False)
def load_coating_config():
    return load_json(P.coating_config_json)

coating_cfg = load_json(P.coating_config_json)
COATING_OPTIONS = coating_options_from_cfg(coating_cfg)
if not COATING_OPTIONS:
    COATING_OPTIONS = [""]  # safe fallback so selectbox won't crash

def render_failed_home_section(days_visible: int = 4, orders_file: str = ORDERS_FILE):
    st.subheader("❌ Failed (last 4 days)")

    if not os.path.exists(orders_file):
        st.info("No orders file yet.")
        return

    try:
        df = pd.read_csv(orders_file)
    except Exception as e:
        st.error(f"Failed to read {orders_file}: {e}")
        return

    df = ensure_orders_cols(df, STATUS_COL, STATUS_UPDATED_COL, FAILED_REASON_COL)
    if df.empty:
        st.info("No orders.")
        return

    # filter Failed
    failed = df[df[STATUS_COL].astype(str).str.lower().eq("failed")].copy()
    if failed.empty:
        st.success("No Failed orders right now ✅")
        return

    # keep only last <days_visible> days (based on Status Updated At)
    now = pd.Timestamp(dt.datetime.now())
    cutoff = now - pd.Timedelta(days=days_visible)

    failed["_dt"] = failed[STATUS_UPDATED_COL].apply(parse_dt_safe)
    # if missing timestamp -> treat as "now" so it shows up, and gets stamped by auto-move helper
    failed["_dt"] = failed["_dt"].fillna(now)

    recent = failed[failed["_dt"] >= cutoff].copy()
    older = failed[failed["_dt"] < cutoff].copy()

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("❌ Failed (recent)", int(len(recent)))
    c2.metric("⏳ Failed (older than 4d)", int(len(older)))
    if len(older) > 0:
        c3.warning("Some Failed orders are older than 4 days — they will be moved back to Pending automatically.")
    else:
        c3.info("Failed orders stay here for 4 days, then auto-return to Pending.")

    # Show recent failed in expanders
    recent = recent.sort_values("_dt", ascending=False)

    # Pick display columns if exist
    nice_cols = []
    for col in [
        "Order ID", "Fiber Type", "Preform Number", "Required Length (m) (for T&M+costumer)",
        "Required Length (m)", "Priority", "Notes", "Dataset CSV", "Assigned Dataset CSV"
    ]:
        if col in recent.columns and col not in nice_cols:
            nice_cols.append(col)

    # Always show these if exist
    if FAILED_REASON_COL in recent.columns and FAILED_REASON_COL not in nice_cols:
        nice_cols.append(FAILED_REASON_COL)
    if STATUS_UPDATED_COL in recent.columns and STATUS_UPDATED_COL not in nice_cols:
        nice_cols.append(STATUS_UPDATED_COL)

    # Fallback: show everything if we found nothing
    if not nice_cols:
        nice_cols = list(recent.columns)

    for idx, row in recent.iterrows():
        title_bits = []
        if "Preform Number" in recent.columns:
            title_bits.append(f"PF: {row.get('Preform Number', '')}")
        if "Fiber Type" in recent.columns:
            title_bits.append(str(row.get("Fiber Type", "")))
        title_bits.append(f"Updated: {row.get(STATUS_UPDATED_COL, '')}")

        with st.expander(" | ".join([b for b in title_bits if b.strip()]), expanded=False):
            st.dataframe(pd.DataFrame([row[nice_cols]]), use_container_width=True)

            # Optional: If you have a dataset csv reference, show the same “done summary” widget you already have.
            # Try these common columns; adjust to your real one:
            dataset_csv = None
            for k in ["Dataset CSV", "Assigned Dataset CSV", "dataset_csv", "CSV File"]:
                if k in recent.columns:
                    v = str(row.get(k, "")).strip()
                    if v and v.lower() != "nan":
                        dataset_csv = v
                        break

            if dataset_csv:
                st.markdown("**📄 Dataset summary (same idea as Done):**")
                # If you already have a function for this, call it here:
                # render_done_summary_from_dataset_csv(dataset_csv)
                # Otherwise keep a simple link:
                st.caption(f"Dataset CSV: `{dataset_csv}`")

            if str(row.get(FAILED_REASON_COL, "")).strip():
                st.error(f"Reason: {row.get(FAILED_REASON_COL)}")
            else:
                st.caption("No failed reason recorded.")

    if not older.empty:
        with st.expander("🗂️ Older Failed (will auto-return to Pending)", expanded=False):
            show_cols = [c for c in ["Order ID", "Fiber Type", "Preform Number", FAILED_REASON_COL, STATUS_UPDATED_COL] if c in older.columns]
            if not show_cols:
                show_cols = list(older.columns)
            st.dataframe(older[show_cols].sort_values("_dt", ascending=False), use_container_width=True)

def maintenance_quick_counts(
        base_dir: str,
        current_date,
        furnace_hours: float = 0.0,
        uv1_hours: float = 0.0,
        uv2_hours: float = 0.0,
        warn_days: int = 14,
        warn_hours: float = 50.0,
):
    """
    Returns (overdue_count, due_soon_count). Loads ALL xlsx/xls/csv from /maintenance.
    Uses the same rules as the maintenance tab.
    """

    maint_folder = P.maintenance_dir
    if not os.path.isdir(maint_folder):
        return 0, 0

    files = [f for f in os.listdir(maint_folder) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
    if not files:
        return 0, 0

    normalize_map = {
        "equipment": "Component",
        "task name": "Task",
        "task id": "Task_ID",
        "interval value": "Interval_Value",
        "interval unit": "Interval_Unit",
        "tracking mode": "Tracking_Mode",
        "hours source": "Hours_Source",
        "due threshold (days)": "Due_Threshold_Days",
        "last done date": "Last_Done_Date",
        "last done hours": "Last_Done_Hours",
    }

    def read_file(path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)

    def norm_source(s) -> str:
        s = "" if s is None or pd.isna(s) else str(s)
        return s.strip().lower()

    def pick_current_hours(hours_source: str) -> float:
        hs = norm_source(hours_source)
        if hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
            return float(uv2_hours)
        if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
            return float(uv1_hours)
        # default furnace
        return float(furnace_hours)

    def parse_date(x):
        if pd.isna(x) or x == "":
            return None
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date()

    def parse_float(x):
        if pd.isna(x) or x == "":
            return None
        try:
            return float(x)
        except Exception:
            return None

    frames = []
    for fname in files:
        p = os.path.join(maint_folder, fname)
        try:
            df = read_file(p)
            if df is None or df.empty:
                continue
            df = df.rename(columns={c: normalize_map.get(str(c).strip().lower(), c) for c in df.columns})
            # ensure columns exist
            for col in ["Tracking_Mode", "Interval_Value", "Interval_Unit", "Hours_Source", "Due_Threshold_Days",
                        "Last_Done_Date", "Last_Done_Hours"]:
                if col not in df.columns:
                    df[col] = np.nan
            frames.append(df)
        except Exception:
            # quick counts should never crash the app
            continue

    if not frames:
        return 0, 0

    dfm = pd.concat(frames, ignore_index=True)

    dfm["Last_Done_Date_parsed"] = dfm["Last_Done_Date"].apply(parse_date)
    dfm["Last_Done_Hours_parsed"] = dfm["Last_Done_Hours"].apply(parse_float)
    dfm["Current_Hours_For_Task"] = dfm["Hours_Source"].apply(pick_current_hours)

    def next_due_date(row):
        mode = str(row.get("Tracking_Mode", "")).strip().lower()
        if mode != "calendar":
            return None
        last = row.get("Last_Done_Date_parsed", None)
        if last is None or pd.isna(last):
            return None
        try:
            v = int(float(row.get("Interval_Value", np.nan)))
        except Exception:
            return None
        unit = str(row.get("Interval_Unit", "")).strip().lower()
        base = pd.Timestamp(last)
        if pd.isna(base) or base is pd.NaT:
            return None
        if "day" in unit:
            out = base + pd.DateOffset(days=v)
        elif "week" in unit:
            out = base + pd.DateOffset(weeks=v)
        elif "month" in unit:
            out = base + pd.DateOffset(months=v)
        elif "year" in unit:
            out = base + pd.DateOffset(years=v)
        else:
            out = base + pd.DateOffset(days=v)
        if pd.isna(out) or out is pd.NaT:
            return None
        return out.date()

    def next_due_hours(row):
        mode = str(row.get("Tracking_Mode", "")).strip().lower()
        if mode != "hours":
            return None
        last_h = row.get("Last_Done_Hours_parsed", None)
        if last_h is None or pd.isna(last_h):
            return None
        try:
            v = float(row.get("Interval_Value", np.nan))
        except Exception:
            return None
        if pd.isna(v):
            return None
        return float(last_h) + float(v)

    dfm["Next_Due_Date"] = dfm.apply(next_due_date, axis=1)
    dfm["Next_Due_Hours"] = dfm.apply(next_due_hours, axis=1)

    def status_row(row):
        mode = str(row.get("Tracking_Mode", "")).strip().lower()
        if mode == "event":
            return "OK"  # routine doesn't count toward overdue/due soon in quick widget

        nd = row.get("Next_Due_Date", None)
        nh = row.get("Next_Due_Hours", None)

        overdue = False
        due_soon = False

        if nd is not None and not pd.isna(nd):
            if isinstance(nd, pd.Timestamp):
                nd = nd.date()
            if nd < current_date:
                overdue = True
            else:
                thresh = row.get("Due_Threshold_Days", np.nan)
                try:
                    thresh = int(float(thresh)) if not pd.isna(thresh) else int(warn_days)
                except Exception:
                    thresh = int(warn_days)
                if (nd - current_date).days <= thresh:
                    due_soon = True

        if nh is not None and not pd.isna(nh):
            nh = float(nh)
            cur_h = float(row.get("Current_Hours_For_Task", 0.0))
            if nh < cur_h:
                overdue = True
            elif (nh - cur_h) <= float(warn_hours):
                due_soon = True

        if overdue:
            return "OVERDUE"
        if due_soon:
            return "DUE SOON"
        return "OK"

    statuses = dfm.apply(status_row, axis=1)
    overdue_count = int((statuses == "OVERDUE").sum())
    due_soon_count = int((statuses == "DUE SOON").sum())

    return overdue_count, due_soon_count

# Load coatings and dies from the configuration file
with open(P.coating_config_json, "r") as config_file:
    config = json.load(config_file)
coatings = config.get("coatings", {})
dies = config.get("dies", {})
with open(P.coating_config_json, "r") as config_file:
    config = json.load(config_file)
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

def calculate_coating_thickness(entry_fiber_diameter, die_diameter, mu, rho, L, V, g):
    """Calculates coating thickness and coated fiber diameter."""
    import math

    R = (die_diameter / 2) * 1e-6   # Die Radius (m)
    r = (entry_fiber_diameter / 2) * 1e-6  # Fiber Radius (m)

    # ---- Guards ----
    if R <= 0 or r <= 0:
        return float("nan")

    # Fiber must fit inside die (strict!)
    if r >= R:
        return float("nan")

    k = r / R
    # k in (0,1) so ln(k) is negative and not close to 0
    ln_k = math.log(k)

    # Protect against numeric edge cases (k≈1 -> ln_k≈0)
    if not math.isfinite(ln_k) or abs(ln_k) < 1e-12:
        return float("nan")

    # Pressure drop calculation
    delta_P = L * rho * g

    # Φ calculation
    if mu == 0 or L == 0 or V == 0:
        return float("nan")
    Phi = (delta_P * R ** 2) / (8 * mu * L * V)

    term1 = Phi * (1 - k ** 4 + ((1 - k ** 2) ** 2) / ln_k)
    term2 = - (k ** 2 + (1 - k ** 2) / (2 * ln_k))

    inside = (term1 + term2 + k ** 2)
    if inside <= 0 or not math.isfinite(inside):
        return float("nan")

    t = R * (math.sqrt(inside) - k)

    coated_fiber_diameter = entry_fiber_diameter + (t * 2 * 1e6)  # thickness -> microns
    return coated_fiber_diameter

def evaluate_viscosity(T, function_str):
    """Computes viscosity by evaluating the stored function string from config."""
    try:
        return eval(function_str, {"T": T, "math": math})
    except Exception as e:
        st.error(f"Error evaluating viscosity function: {e}")
        return None

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
    GROUPS = list(NAV_GROUPS.keys())

    # If a shortcut tab was set elsewhere, honor it once
    desired_tab = (
            st.session_state.get("selected_tab")
            or st.session_state.get("tab_select")
            or st.session_state.get("last_tab")
            or "🏠 Home"
    )
    st.session_state["selected_tab"] = None

    if desired_tab not in TAB_TO_GROUP:
        desired_tab = "🏠 Home"

    desired_group = TAB_TO_GROUP.get(desired_tab, "🏠 Core")

    # remember last tab per group
    if "nav_last_tab_by_group" not in st.session_state:
        st.session_state["nav_last_tab_by_group"] = {}

    # force state consistent BEFORE widgets render
    # force state consistent BEFORE widgets render
    # ✅ If we got a "selected_tab" shortcut, FORCE jump (group + page)
    jump_tab = desired_tab  # desired_tab already includes selected_tab if provided

    jump_group = TAB_TO_GROUP.get(jump_tab, desired_group)

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
        g = TAB_TO_GROUP.get(t, st.session_state.get("nav_group_select", "🏠 Core"))
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

if tab_selection == "📊 Dashboard":
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".csv")]

    if not csv_files:
        st.error("No CSV files found in the directory.")
        st.stop()

    csv_files_sorted = sorted(
        csv_files,
        key=lambda fn: os.path.getmtime(os.path.join(DATA_FOLDER, fn)),
        reverse=True,
    )
    latest_file = csv_files_sorted[0]

    # ✅ Default to latest ONLY if nothing is selected yet
    if not st.session_state.get("dataset_select"):
        st.session_state["dataset_select"] = latest_file

    # ✅ If selection doesn't exist in THIS folder -> fallback
    if st.session_state.get("dataset_select") not in csv_files_sorted:
        st.session_state["dataset_select"] = latest_file

    selected_file = st.sidebar.selectbox(
        "Select a dataset",
        options=csv_files_sorted,
        key="dataset_select",
    )

    st.sidebar.caption(f"Latest: **{latest_file}**")

    df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))


    # ==========================================================
    # ✅ Auto-load GOOD ZONES when coming from SQL Lab
    # - Reads zone boundaries from the DATASET CSV (data_set_csv/<selected_file>)
    # - Writes into st.session_state["good_zones"] so your existing plot vrects work
    # ==========================================================
    def _read_dataset_params_csv_for_log(selected_log_file: str) -> pd.DataFrame:
        # dataset CSV has same filename as log selection in your system
        pth = os.path.join(DATASET_DIR, os.path.basename(selected_log_file))
        if not os.path.exists(pth):
            return pd.DataFrame()
        try:
            return pd.read_csv(pth, keep_default_na=False)
        except Exception:
            return pd.DataFrame()





    def _parse_zones_from_params(df_params: pd.DataFrame):
        """
        Returns zones as list of (start_m, end_m) in meters from START (absolute).
        Supports:
          - Zone i Start / Zone i End  (saved by your dashboard)
        """
        if df_params is None or df_params.empty:
            return []

        if "Parameter Name" not in df_params.columns or "Value" not in df_params.columns:
            return []

        p = df_params.copy()
        p["Parameter Name"] = p["Parameter Name"].astype(str).str.strip()
        p["Value"] = p["Value"].astype(str).str.strip()

        start_pat = re.compile(r"^Zone\s*(\d+)\s*Start$", re.IGNORECASE)
        end_pat = re.compile(r"^Zone\s*(\d+)\s*End$", re.IGNORECASE)

        starts = {}
        ends = {}

        for _, row in p.iterrows():
            nm = row["Parameter Name"]
            val = row["Value"]
            try:
                fv = float(val)
            except Exception:
                continue

            ms = start_pat.match(nm)
            me = end_pat.match(nm)
            if ms:
                starts[int(ms.group(1))] = fv
            if me:
                ends[int(me.group(1))] = fv

        zones = []
        for i in sorted(set(starts) & set(ends)):
            a = float(starts[i])
            b = float(ends[i])
            zones.append((min(a, b), max(a, b)))

        # remove invalid
        zones = [(a, b) for a, b in zones if b > a]
        return zones


    # one-shot trigger from SQL Lab
    autoload = bool(st.session_state.get("dash_autoload_zones", False))
    autoload_for = (st.session_state.get("dash_autoload_zones_for") or "").replace("\\", "/").split("/")[-1]

    if autoload and (autoload_for == os.path.basename(selected_file)):
        df_params_for_log = _read_dataset_params_csv_for_log(selected_file)
        zones_m = _parse_zones_from_params(df_params_for_log)

        if zones_m:
            # store zones in the exact format your plot already uses
            st.session_state["good_zones"] = zones_m
            st.success(f"✅ Loaded {len(zones_m)} good zone(s) from dataset CSV for {selected_file}")
        else:
            st.info("ℹ️ No saved Zone i Start/End found in dataset CSV (nothing to auto-mark).")

        # consume the flag (do once)
        st.session_state["dash_autoload_zones"] = False
if not df.empty and "Date/Time" in df.columns:
    def try_parse_datetime(dt_str):
        try:
            return pd.to_datetime(dt_str)
        except Exception:
            try:
                if isinstance(dt_str, str) and len(dt_str.split(":")[-1]) > 2:
                    parts = dt_str.rsplit(":", 1)
                    fixed_time = parts[0] + ":" + parts[1][:2] + "." + parts[1][2:]
                    return pd.to_datetime(fixed_time)
            except:
                return pd.NaT
        return pd.NaT


    df["Date/Time"] = df["Date/Time"].apply(try_parse_datetime)

column_options = df.columns.tolist() if not df.empty else []

def render_home_draw_orders_overview(
        orders_file: str = P.orders_csv,
        title: str = "📦 Draw Orders",
        height: int = 360,
):
    import os
    import pandas as pd
    import streamlit as st

    # ---------- Title ----------
    st.markdown(
        f"""
        <div style="
            font-size: 1.5rem;
            font-weight: 700;
            color: rgba(255,255,255,0.95);
            margin-bottom: 0.6em;
            text-align: left;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Small card helper ----------
    def _card(title_txt, value, border_color, emoji=""):
        st.markdown(
            f"""
            <div style="
                width: 100%;
                min-height: 140px;
                background: rgba(0,0,0,0.35);
                border: 2px solid {border_color};
                border-radius: 18px;
                padding: 14px;
                text-align: center;
                box-shadow: 0 6px 18px rgba(0,0,0,0.25);
                backdrop-filter: blur(6px);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                gap: 6px;
            ">
                <div style="font-size:18px;font-weight:800;color:white;">
                    {emoji} {title_txt}
                </div>
                <div style="font-size:44px;font-weight:900;color:white;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- NOW DRAWING BOX ----------
    def _now_box(text_left, text_right="", border_color="#b77bff", emoji="🟣"):
        st.markdown(
            f"""
            <div style="
                width:100%;
                padding:18px 22px;
                border-radius:20px;
                border:2px solid {border_color};
                background:rgba(40,20,60,0.55);
                box-shadow:0 10px 30px rgba(0,0,0,0.35);
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:18px;
            ">
                <div style="font-size:22px;font-weight:900;color:white;">
                    {emoji} {text_left}
                </div>
                <div style="font-size:14px;color:#e6d9ff;">
                    {text_right}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not os.path.exists(orders_file):
        st.info("No orders submitted yet.")
        return

    # ---------- Load orders (safe) ----------
    try:
        df = pd.read_csv(orders_file, keep_default_na=False)
    except Exception:
        st.info("No orders submitted yet.")
        return

    if df.empty and len(df.columns) == 0:
        st.info("No orders submitted yet.")
        return

    # Ensure columns
    for col, default in {
        "Status": "Pending",
        "Priority": "Normal",
        "Fiber Project": "",
        "Preform Number": "",
        "Timestamp": "",
        "Desired Date": "",
        "Length (m)": "",
        "Spools": "",
        "Notes": "",
        "Done CSV": "",
        "Done Description": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }.items():
        if col not in df.columns:
            df[col] = default

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Desired Date"] = pd.to_datetime(df["Desired Date"], errors="coerce").dt.date

    # Normalize text
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""}).fillna("").str.strip()

    # Normalize T&M flag
    df["T&M Moved"] = df["T&M Moved"].apply(
        lambda x: str(x).strip().lower() in ("true", "1", "yes", "y", "moved")
    )

    df_visible = df[~df["T&M Moved"]].copy()

    # ==========================================================
    # 🟣 NOW DRAWING (Status == In progress)
    # ✅ FIX: render box ONLY if something is in progress
    # ==========================================================
    def _is_in_progress(x):
        return str(x).strip().lower() in (
            "in progress",
            "in-progress",
            "progress",
            "in prograss",
            "drawing",
        )

    df_prog = df_visible[df_visible["Status"].apply(_is_in_progress)].copy()

    if not df_prog.empty:
        df_prog = df_prog.sort_values("Timestamp", ascending=False)
        row = df_prog.iloc[0]

        preform = row.get("Preform Number", "")
        fiber = row.get("Fiber Project", "")

        if preform:
            now_text = f"Now drawing: F{preform}"
        elif fiber:
            now_text = f"Now drawing: {fiber}"
        else:
            now_text = "Now drawing: In progress"

        n_ip = len(df_prog)
        ts = row.get("Timestamp")

        if pd.notna(ts):
            now_right = f"In progress: {n_ip} | {ts.strftime('%Y-%m-%d %H:%M')}"
        else:
            now_right = f"In progress: {n_ip}"

        # ✅ Only here
        _now_box(now_text, now_right)

    # ==========================================================
    # KPI COUNTS (visible only)
    # ==========================================================
    def _count(s):
        return int((df_visible["Status"] == s).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card("Pending", _count("Pending"), "orange", "🟠")
    with c2:
        _card("Scheduled", _count("Scheduled"), "dodgerblue", "🗓️")
    with c3:
        _card("Done", _count("Done"), "limegreen", "✅")
    with c4:
        _card("Failed", _count("Failed"), "crimson", "❌")

    # ==========================================================
    # TABLE
    # ==========================================================


def render_open_in_dashboard_from_filter(df_all: pd.DataFrame):
    st.subheader("🚀 Open matched draw in Dashboard")

    if df_all is None or df_all.empty:
        st.info("Run a filter first.")
        return

    # Keep only dataset rows (draws)
    ds = df_all[df_all["source_kind"].astype(str) == "dataset"].copy()
    if ds.empty:
        st.info("Filter matched no dataset draws (only maintenance / none).")
        return

    # Prefer draw id (event_id) + filename when available
    # event_id is your draw id, filename points to csv file
    cols = ds.columns.tolist()
    if "filename" in cols:
        # unique draws with filename
        draws = (
            ds[["event_id", "filename", "event_ts"]]
            .dropna(subset=["event_id"])
            .drop_duplicates(subset=["event_id"])
            .sort_values("event_ts", ascending=False, na_position="last")
        )
    else:
        draws = (
            ds[["event_id", "event_ts"]]
            .dropna(subset=["event_id"])
            .drop_duplicates(subset=["event_id"])
            .sort_values("event_ts", ascending=False, na_position="last")
        )
        draws["filename"] = None

    if draws.empty:
        st.info("No unique draws found.")
        return

    # nice labels
    def _label_row(r):
        ts = r.get("event_ts")
        ts_s = ""
        try:
            if pd.notna(ts):
                ts_s = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        draw_id = str(r.get("event_id", "")).strip()
        file_s = str(r.get("filename", "")).strip()
        file_short = file_s.split("/")[-1] if file_s else ""
        if ts_s and file_short:
            return f"{draw_id}  •  {ts_s}  •  {file_short}"
        if ts_s:
            return f"{draw_id}  •  {ts_s}"
        return draw_id

    draw_options = draws.to_dict("records")
    labels = [_label_row(r) for r in draw_options]

    chosen_idx = st.selectbox(
        "Choose one matched draw",
        list(range(len(draw_options))),
        format_func=lambda i: labels[i],
        key="sql_open_dash_choice",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        open_btn = st.button("📊 Open in Dashboard", use_container_width=True, key="sql_open_in_dashboard_btn")
    with c2:
        st.caption("Loads the selected draw CSV + zones in the Dashboard tab.")

    if open_btn:
        r = draw_options[int(chosen_idx)]

        target_file = (str(r.get("filename", "")).strip() or "")
        target_file = target_file.replace("\\", "/").split("/")[-1] if target_file else None

        if not target_file:
            target_file = f"{str(r.get('event_id', '')).strip()}.csv"

        # ✅ IMPORTANT: store only basename that exists in DATA_FOLDER
        st.session_state["dataset_select"] = target_file
        st.session_state["dash_autoload_zones"] = True
        st.session_state["dash_autoload_zones_for"] = target_file  # remember which file
        st.session_state["selected_tab"] = "📊 Dashboard"
        st.rerun()

def _append_dict_rows(rows, data: dict, units_map: dict = None):
    if not data:
        return
    units_map = units_map or {}
    for k, v in data.items():
        if v is None:
            continue
        # avoid writing empty strings
        if isinstance(v, str) and v.strip() == "":
            continue
        rows.append({
            "Parameter Name": str(k),
            "Value": v,
            "Units": units_map.get(k, "")
        })

def calculate_coating_thickness_diameter_um(
    entry_fiber_diameter_um: float,
    die_diameter_um: float,
    mu_kg_m_s: float,
    rho_kg_m3: float,
    neck_length_m: float,
    pulling_speed_m_s: float,
    g_m_s2: float = 9.80665,
) -> float:
    """
    Your exact model, returns COATED DIAMETER in µm.
    """
    entry_fiber_diameter_um = to_float(entry_fiber_diameter_um, 0.0)
    die_diameter_um = to_float(die_diameter_um, 0.0)
    mu = to_float(mu_kg_m_s, 1.0)
    rho = to_float(rho_kg_m3, 1000.0)
    L = to_float(neck_length_m, 0.01)
    V = to_float(pulling_speed_m_s, 0.917)
    g = to_float(g_m_s2, 9.80665)

    if entry_fiber_diameter_um <= 0 or die_diameter_um <= 0 or mu <= 0 or rho <= 0 or L <= 0 or V <= 0:
        return float(entry_fiber_diameter_um)

    R = (die_diameter_um / 2.0) * 1e-6  # die radius (m)
    r = (entry_fiber_diameter_um / 2.0) * 1e-6  # fiber radius (m)

    if r <= 0 or R <= 0 or r >= R:
        return float(entry_fiber_diameter_um)

    k = r / R
    ln_k = math.log(k)  # k<1 => ln(k)<0 ok

    delta_P = L * rho * g
    Phi = (delta_P * (R ** 2)) / (8.0 * mu * L * V)

    term1 = Phi * (1.0 - k**4 + ((1.0 - k**2)**2) / ln_k)
    term2 = -(k**2 + (1.0 - k**2) / (2.0 * ln_k))

    inside = term1 + term2 + k**2
    if inside <= 0:
        return float(entry_fiber_diameter_um)

    t = R * (math.sqrt(inside) - k)  # meters
    coated_um = entry_fiber_diameter_um + (t * 2.0 * 1e6)
    return float(coated_um)

def _match_key_case_insensitive(name: str, keys: List[str]) -> Optional[str]:
    """Find best match of name in keys (case-insensitive, trims)."""
    n = str(name or "").strip()
    if not n:
        return None
    low = n.lower()

    for k in keys:
        if str(k).strip().lower() == low:
            return k

    for k in keys:
        kl = str(k).strip().lower()
        if low in kl or kl in low:
            return k

    return None

def _get_viscosity_for_coating(coating_name: str, temp_c: float) -> float:
    """
    Uses your known functions.
    Extend here if you add more coatings.
    """
    name = str(coating_name or "").strip().lower()

    if "dp1032" in name or "dp-1032" in name:
        return float(get_viscosityDP1032(float(temp_c)))

    if name.startswith("ds") or "ds2015" in name or "ds2032" in name or "ds-2015" in name or "ds-2032" in name:
        return float(get_viscosityDS2015(float(temp_c)))

    return float(get_viscosityDS2015(float(temp_c)))

def calculate_coated_diameter_um(entry_fiber_diameter_um, die_diameter_um, mu, rho, L, V, g=9.80665):
    """
    Wrapper around your model, returns predicted coated diameter [um] or NaN if invalid.
    """
    # Your function already returns diameter in um, but protect invalid geometries
    if die_diameter_um is None or entry_fiber_diameter_um is None:
        return float("nan")

    if die_diameter_um <= 0 or entry_fiber_diameter_um <= 0:
        return float("nan")

    # Fiber must fit in die
    if entry_fiber_diameter_um >= die_diameter_um:
        return float("nan")

    try:
        return calculate_coating_thickness(
            entry_fiber_diameter_um,
            die_diameter_um,
            mu, rho, L, V, g
        )
    except Exception:
        return float("nan")

def render_schedule_home_minimal():
    import plotly.express as px
    st.subheader("📅 Schedule")

    SCHEDULE_FILE = P.schedule_csv
    required_columns = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    # Ensure file exists (works even if empty / no events)
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=required_columns).to_csv(SCHEDULE_FILE, index=False)

    schedule_df = pd.read_csv(SCHEDULE_FILE)
    schedule_df.columns = schedule_df.columns.str.strip()

    # Ensure required columns exist (auto-fix)
    for col in required_columns:
        if col not in schedule_df.columns:
            schedule_df[col] = pd.Series(dtype="object")
    schedule_df = schedule_df[required_columns]

    # Parse datetimes safely
    schedule_df["Start DateTime"] = pd.to_datetime(schedule_df["Start DateTime"], errors="coerce")
    schedule_df["End DateTime"] = pd.to_datetime(schedule_df["End DateTime"], errors="coerce")

    # Clean Description/Recurrence strings
    schedule_df["Description"] = (
        schedule_df["Description"]
        .fillna("")
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Description=", "", regex=False)
        .str.strip()
    )

    schedule_df["Recurrence"] = (
        schedule_df["Recurrence"]
        .fillna("None")
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Recurrence=", "", regex=False)
        .str.strip()
    )
    schedule_df.loc[schedule_df["Recurrence"].eq(""), "Recurrence"] = "None"

    # -------------------------
    # UI: range buttons (Week / Month / 3 Months) + Show all
    # -------------------------
    today = pd.Timestamp.today().normalize()

    if "home_schedule_range" not in st.session_state:
        st.session_state.home_schedule_range = "week"  # default

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        if st.button("📅 Week", use_container_width=True, key="home_sched_btn_week"):
            st.session_state.home_schedule_range = "week"
            st.rerun()

    with c2:
        if st.button("🗓️ Month", use_container_width=True, key="home_sched_btn_month"):
            st.session_state.home_schedule_range = "month"
            st.rerun()

    with c3:
        if st.button("📆 3 Months", use_container_width=True, key="home_sched_btn_3months"):
            st.session_state.home_schedule_range = "3months"
            st.rerun()

    with c4:
        show_all = st.checkbox("Show all", value=False, key="home_schedule_show_all_min")

    # Compute filters from selection
    range_sel = st.session_state.home_schedule_range
    start_filter = today

    if range_sel == "week":
        end_filter = today + pd.DateOffset(weeks=1)
    elif range_sel == "month":
        end_filter = today + pd.DateOffset(months=1)
    else:  # "3months"
        end_filter = today + pd.DateOffset(months=3)

    # Filter to valid rows first
    valid = schedule_df.dropna(subset=["Start DateTime", "End DateTime"]).copy()

    # -------------------------
    # Expand recurrences so Home shows ALL occurrences in the selected range
    # -------------------------
    def _expand_recurrences(df_in: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        if df_in.empty:
            return df_in

        out_rows = []
        max_instances = 2000  # safety cap

        for _, r in df_in.iterrows():
            st0 = r.get("Start DateTime")
            en0 = r.get("End DateTime")
            if pd.isna(st0) or pd.isna(en0):
                continue

            rec = str(r.get("Recurrence", "None")).strip()
            rec_low = rec.lower()

            duration = en0 - st0
            if pd.isna(duration) or duration <= pd.Timedelta(seconds=0):
                duration = pd.Timedelta(minutes=1)

            # Non-recurring
            if rec_low in ("", "none", "nan"):
                d0 = r.to_dict()
                d0["Recurrence"] = "None"
                out_rows.append(d0)
                continue

            def _add_step(ts: pd.Timestamp) -> pd.Timestamp:
                if "week" in rec_low:
                    return ts + pd.DateOffset(weeks=1)
                if "month" in rec_low:
                    return ts + pd.DateOffset(months=1)
                if "year" in rec_low:
                    return ts + pd.DateOffset(years=1)
                return pd.NaT  # unknown

            # Unknown recurrence -> treat as non-recurring
            if pd.isna(_add_step(pd.Timestamp(st0))):
                d0 = r.to_dict()
                d0["Recurrence"] = "None"
                out_rows.append(d0)
                continue

            cur_start = pd.Timestamp(st0)
            cur_end = cur_start + duration

            # Fast-forward to near the window
            guard = 0
            while cur_end < window_start and guard < max_instances:
                nxt = _add_step(cur_start)
                if pd.isna(nxt):
                    break
                cur_start = nxt
                cur_end = cur_start + duration
                guard += 1

            # Generate occurrences that intersect the window
            gen = 0
            while cur_start <= window_end and gen < max_instances:
                cur_end = cur_start + duration
                if (cur_end >= window_start) and (cur_start <= window_end):
                    d = r.to_dict()
                    d["Start DateTime"] = cur_start
                    d["End DateTime"] = cur_end
                    d["Recurrence"] = rec if rec else "None"
                    out_rows.append(d)

                nxt = _add_step(cur_start)
                if pd.isna(nxt):
                    break
                cur_start = nxt
                gen += 1

        if not out_rows:
            return pd.DataFrame(columns=df_in.columns)

        out = pd.DataFrame(out_rows)
        for c in df_in.columns:
            if c not in out.columns:
                out[c] = ""
        out = out[df_in.columns]
        out["Start DateTime"] = pd.to_datetime(out["Start DateTime"], errors="coerce")
        out["End DateTime"] = pd.to_datetime(out["End DateTime"], errors="coerce")

        # Ensure Recurrence always has a nice value
        out["Recurrence"] = out["Recurrence"].fillna("None").astype(str).str.strip()
        out.loc[out["Recurrence"].isin(["", "none", "None", "nan", "NaN"]), "Recurrence"] = "None"
        return out

    win_start = pd.to_datetime(start_filter)
    win_end = pd.to_datetime(end_filter)

    # If show_all: show a bigger horizon so recurring events really look “recurring”
    if show_all:
        win_start = today
        win_end = today + pd.DateOffset(months=12)

    expanded = _expand_recurrences(valid, win_start, win_end)

    # Now filter (overlap logic)
    filtered = expanded[
        (expanded["End DateTime"] >= win_start) &
        (expanded["Start DateTime"] <= win_end)
    ].copy()

    st.write("### Schedule Timeline")

    event_colors = {
        "Maintenance": "blue",
        "Drawing": "green",
        "Stop": "red",
        "Management Event": "purple",
    }

    if filtered.empty:
        st.info("No events in the selected range (or schedule is empty).")
        return

    # =========================================================
    # ✅ HOVER FIX (ONLY CHANGE YOU ASKED FOR)
    # px.timeline hover can't reliably format x_end, so we precompute strings
    # and force a clean hovertemplate (like the Schedule tab)
    # =========================================================
    filtered["StartStr"] = pd.to_datetime(filtered["Start DateTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    filtered["EndStr"] = pd.to_datetime(filtered["End DateTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    filtered["RecurrenceDisp"] = filtered["Recurrence"].fillna("None").astype(str).str.strip()
    filtered.loc[filtered["RecurrenceDisp"].isin(["", "none", "None", "nan", "NaN"]), "RecurrenceDisp"] = "None"

    fig = px.timeline(
        filtered,
        x_start="Start DateTime",
        x_end="End DateTime",
        y="Event Type",
        color="Event Type",
        color_discrete_map=event_colors,
        title="Tower Schedule",
        custom_data=["StartStr", "EndStr", "RecurrenceDisp", "Description"],
    )

    # Force clean hover (NO weird %{customdata[0]} text leaks, NO broken x_end formatting)
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Start: %{customdata[0]}<br>"
            "End: %{customdata[1]}<br>"
            "Recurrence: %{customdata[2]}<br>"
            "Description: %{customdata[3]}"
            "<extra></extra>"
        )
    )

    # Keep your layout exactly as you had it
    fig.update_layout(
        paper_bgcolor="rgba(15,15,20,0.92)",
        plot_bgcolor="rgba(15,15,20,0.70)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Event Type",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    st.plotly_chart(fig, use_container_width=True)

def render_parts_orders_home_all():
    st.subheader("🧩 Parts Orders")

    ORDER_FILE = P.parts_orders_csv

    # ✅ NEW canonical statuses (Needed -> Opened)
    status_order = ["Opened", "Approved", "Ordered", "Shipped", "Received", "Installed"]

    # ✅ display columns (remove Purpose, keep single Details)
    column_order = [
        "Status",
        "Part Name",
        "Serial Number",
        "Project Name",
        "Details",
        "Approved",
        "Approved By",
        "Approval Date",
        "Ordered By",
        "Date Ordered",
        "Company",
    ]

    # ---------------- Load / Safety ----------------
    if not os.path.exists(ORDER_FILE):
        st.info("No orders file yet (part_orders.csv).")
        return

    orders_df = pd.read_csv(ORDER_FILE)
    orders_df.columns = orders_df.columns.str.strip()

    # Ensure required columns exist
    for col in column_order:
        if col not in orders_df.columns:
            orders_df[col] = ""

    orders_df = orders_df[column_order].copy().fillna("")
    orders_df["Status"] = orders_df["Status"].astype(str).str.strip()

    # ✅ Backward-compat: map old "Needed" to "Opened"
    orders_df["Status"] = orders_df["Status"].replace({"Needed": "Opened", "needed": "Opened"})

    if orders_df.empty:
        st.warning("No orders have been placed yet.")
        return

    # ---------------- Counts ----------------
    status_lower = orders_df["Status"].astype(str).str.lower()
    opened_count = int((status_lower == "opened").sum())
    approved_count = int((status_lower == "approved").sum())
    ordered_count = int((status_lower == "ordered").sum())
    received_count = int((status_lower == "received").sum())

    # ---------------- KPI Cards + GLASS TABLE CSS ----------------
    st.markdown(
        """
        <style>
        /* ================= KPI CARDS ================= */
        .kpi-card{
            background: rgba(0,0,0,0.35);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px;
            padding: 14px 16px;
            text-align: center;
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-title{
            font-size: 16px;
            font-weight: 800;
            color: rgba(255,255,255,0.92);
        }
        .kpi-value{
            font-size: 34px;
            font-weight: 900;
            color: rgba(255,255,255,0.98);
            margin-top: 6px;
            line-height: 1;
        }
        .kpi-opened{
            border: 2px solid rgba(255, 80, 80, 0.95) !important;
            box-shadow: 0 0 18px rgba(255, 80, 80, 0.85);
            background: rgba(255, 80, 80, 0.22);
        }
        .kpi-received{
            border: 2px solid rgba(80, 255, 120, 0.95) !important;
            box-shadow: 0 0 18px rgba(80, 255, 120, 0.85);
            background: rgba(80, 255, 120, 0.22);
        }

        /* ================= FULL GLASS TABLE (AG-GRID) ================= */

        div[data-testid="stDataFrame"]{
            background: transparent !important;
        }

        div[data-testid="stDataFrame"] > div{
            background: rgba(0,0,0,0.28) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 18px !important;
            padding: 10px !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25) !important;
        }

        .ag-root-wrapper, .ag-root, .ag-body-viewport, .ag-center-cols-viewport,
        .ag-center-cols-container, .ag-floating-top, .ag-floating-bottom,
        .ag-pinned-left-cols-container, .ag-pinned-right-cols-container,
        .ag-row, .ag-row-odd, .ag-row-even{
            background: transparent !important;
        }

        .ag-header{
            background: rgba(0,0,0,0.30) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border-bottom: 1px solid rgba(255,255,255,0.10) !important;
        }
        .ag-header-cell, .ag-header-group-cell{
            background: transparent !important;
            color: rgba(255,255,255,0.85) !important;
            font-weight: 800 !important;
            border-right: 1px solid rgba(255,255,255,0.07) !important;
        }

        .ag-cell{
            background: rgba(0,0,0,0.14) !important;
            color: rgba(255,255,255,0.92) !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
            border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        }

        .ag-row-hover .ag-cell{
            background: rgba(255,255,255,0.06) !important;
        }

        .ag-body-viewport{
            background: rgba(0,0,0,0.14) !important;
        }

        ::-webkit-scrollbar { height: 10px; width: 10px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- KPI Cards (symmetric) ----------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    opened_class = "kpi-card kpi-opened" if opened_count > 0 else "kpi-card"
    with c1:
        st.markdown(
            f"""
            <div class="{opened_class}">
                <div class="kpi-title">🔴 Opened</div>
                <div class="kpi-value">{opened_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🟢 Approved</div>
                <div class="kpi-value">{approved_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🟡 Ordered</div>
                <div class="kpi-value">{ordered_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    received_class = "kpi-card kpi-received" if received_count > 0 else "kpi-card"
    with c4:
        st.markdown(
            f"""
            <div class="{received_class}">
                <div class="kpi-title">✅ Received</div>
                <div class="kpi-value">{received_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- Sorting ----------------
    # ensure all statuses are valid; unknown -> Opened
    orders_df["Status"] = orders_df["Status"].apply(lambda s: s if s in status_order else "Opened")

    orders_df["__status_sort__"] = pd.Categorical(
        orders_df["Status"],
        categories=status_order,
        ordered=True
    )
    orders_df = (
        orders_df
        .sort_values(["__status_sort__", "Part Name"], na_position="last")
        .drop(columns="__status_sort__")
    )

    # ---------------- Table Coloring (Status cell only) ----------------
    # keep same "colors logic" but with Opened replacing Needed
    def highlight_status(row):
        color_map = {
            "Opened": "background-color: lightcoral; color: black; font-weight: 900;",
            "Approved": "background-color: lightgreen; color: black; font-weight: 900;",
            "Ordered": "background-color: lightyellow; color: black; font-weight: 900;",
            "Shipped": "background-color: lightblue; color: black; font-weight: 900;",
            "Received": "background-color: green; color: black; font-weight: 900;",
            "Installed": "background-color: lightgray; color: black; font-weight: 900;",
        }
        s = str(row.get("Status", "")).strip()
        return [color_map.get(s, "")] + [""] * (len(row) - 1)

    st.dataframe(
        orders_df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        height=360
    )

def render_corr_outliers_tab(DRAW_FOLDER: str, MAINT_FOLDER: str):
    st.subheader("📈 Correlation & Outliers (auto, incremental)")
    st.caption(
        "Computes correlation vs time for ALL unique numeric column pairs across draw logs. "
        "Only new/changed CSVs are processed and appended to cache."
    )

    # ---------------------------------------------------------
    # Cache paths (inside maintenance/)
    # ---------------------------------------------------------
    CACHE_DIR = os.path.join(MAINT_FOLDER, "_corr_outliers_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    PROCESSED_JSON = os.path.join(CACHE_DIR, "processed_logs.json")
    CACHE_PARQUET = os.path.join(CACHE_DIR, "corr_cache.parquet")
    CACHE_CSV = os.path.join(CACHE_DIR, "corr_cache.csv")  # fallback

    # ---------------------------------------------------------
    # Settings (safe defaults)
    # ---------------------------------------------------------
    TIME_COL_CANDIDATES = [
        "Date/Time", "DateTime", "Datetime", "Timestamp", "Time", "time", "datetime", "date_time", "date"
    ]

    CORR_METHOD = "spearman"  # robust
    TIME_WINDOW_SECONDS = 60  # corr point every 60s (if timestamp exists)
    ROW_WINDOW = 1500  # if no timestamp -> window by rows
    MIN_POINTS_PER_WINDOW = 80
    MAX_NUMERIC_COLS = 28  # safety: pairs explode fast (28 => 378 pairs)
    DROP_CONSTANT_COLS = True

    # Outlier settings (MAD on correlation time-series per pair)
    OUTLIER_K = 6.0
    MIN_HISTORY_FOR_OUTLIERS = 14

    # Plot UI
    DEFAULT_PLOTS_PER_PAGE = 24

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _safe_read_csv(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, engine="python")

    def _file_signature(path: str) -> dict:
        stt = os.stat(path)
        return {"mtime": stt.st_mtime, "size": stt.st_size}

    def _load_processed() -> dict:
        if not os.path.exists(PROCESSED_JSON):
            return {}
        try:
            with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_processed(d: dict) -> None:
        with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    def _can_use_parquet() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except Exception:
            try:
                import fastparquet  # noqa: F401
                return True
            except Exception:
                return False

    def _load_cache() -> pd.DataFrame:
        if _can_use_parquet() and os.path.exists(CACHE_PARQUET):
            try:
                df = pd.read_parquet(CACHE_PARQUET)
                return df
            except Exception:
                pass
        if os.path.exists(CACHE_CSV):
            try:
                return pd.read_csv(CACHE_CSV)
            except Exception:
                pass
        return pd.DataFrame()

    def _save_cache(df: pd.DataFrame) -> None:
        if _can_use_parquet():
            df.to_parquet(CACHE_PARQUET, index=False)
        else:
            df.to_csv(CACHE_CSV, index=False)

    def _append_cache(new_rows: pd.DataFrame) -> None:
        if new_rows is None or new_rows.empty:
            return
        new_rows = new_rows.copy()
        new_rows["window_time"] = pd.to_datetime(new_rows["window_time"], errors="coerce")

        old = _load_cache()
        if old.empty:
            merged = new_rows
        else:
            merged = pd.concat([old, new_rows], ignore_index=True)

        # de-dup (log_id, pair_key, window_time) so repeated scans don’t inflate
        merged.drop_duplicates(subset=["log_id", "pair_key", "window_time"], keep="last", inplace=True)

        _save_cache(merged)

    from typing import Optional

    def _find_time_col(df: pd.DataFrame) -> Optional[str]:
        cols_map = {str(c).strip().lower(): c for c in df.columns}
        for cand in TIME_COL_CANDIDATES:
            key = str(cand).strip().lower()
            if key in cols_map:
                return cols_map[key]
        return None

    def _select_numeric_cols(df: pd.DataFrame) -> list[str]:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove common index-like columns
        num = [c for c in num if str(c).strip().lower() not in {"index", "idx"}]

        if DROP_CONSTANT_COLS:
            keep = []
            for c in num:
                s = df[c]
                if s.dropna().nunique() <= 1:
                    continue
                keep.append(c)
            num = keep

        num = sorted(num)[:MAX_NUMERIC_COLS]
        return num

    def _canonical_pair(a: str, b: str) -> tuple[str, str, str]:
        a2, b2 = sorted([a, b])
        return a2, b2, f"{a2}__VS__{b2}"

    def _corr_to_rows(corr: pd.DataFrame, log_id: str, window_time: pd.Timestamp, n_points: int) -> list[dict]:
        cols = list(corr.columns)
        rows = []
        # UNIQUE pairs only (upper triangle) => no A,B vs B,A duplicates
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                a2, b2, pair_key = _canonical_pair(a, b)
                v = corr.loc[a, b]
                rows.append({
                    "log_id": log_id,
                    "window_time": window_time,
                    "col_a": a2,
                    "col_b": b2,
                    "pair_key": pair_key,
                    "corr": float(v) if pd.notna(v) else np.nan,
                    "method": CORR_METHOD,
                    "n_points": int(n_points),
                })
        return rows

    def _compute_rows_for_log(csv_path: str) -> pd.DataFrame:
        df = _safe_read_csv(csv_path)
        if df is None or df.empty:
            return pd.DataFrame()

        numeric_cols = _select_numeric_cols(df)
        if len(numeric_cols) < 2:
            return pd.DataFrame()

        log_id = os.path.basename(csv_path)
        time_col = _find_time_col(df)

        # Timestamp path
        if time_col is not None:
            t = pd.to_datetime(df[time_col], errors="coerce")
            ok = t.notna()
            df = df.loc[ok, numeric_cols].copy()
            t = t.loc[ok]

            if len(df) < MIN_POINTS_PER_WINDOW:
                return pd.DataFrame()

            df["__t__"] = t.values
            df.sort_values("__t__", inplace=True)
            df.set_index("__t__", inplace=True)

            freq = f"{int(TIME_WINDOW_SECONDS)}S"
            out_rows = []
            for win_end, chunk in df[numeric_cols].resample(freq):
                chunk = chunk.dropna()
                if len(chunk) < MIN_POINTS_PER_WINDOW:
                    continue
                corr = chunk.corr(method=CORR_METHOD)
                out_rows.extend(_corr_to_rows(corr, log_id, win_end, len(chunk)))

            return pd.DataFrame(out_rows)

        # No timestamp => row windows
        arr = df[numeric_cols].copy()
        n = len(arr)
        if n < MIN_POINTS_PER_WINDOW:
            return pd.DataFrame()

        out_rows = []
        mtime = os.path.getmtime(csv_path)
        base_time = pd.to_datetime(mtime, unit="s")

        step = ROW_WINDOW  # non-overlap (fast)
        for start in range(0, n, step):
            chunk = arr.iloc[start:start + ROW_WINDOW].dropna()
            if len(chunk) < MIN_POINTS_PER_WINDOW:
                continue
            corr = chunk.corr(method=CORR_METHOD)
            win_time = base_time + pd.to_timedelta(start, unit="s")
            out_rows.extend(_corr_to_rows(corr, log_id, win_time, len(chunk)))

        return pd.DataFrame(out_rows)

    def _scan_logs(folder: str) -> list[str]:
        if not os.path.isdir(folder):
            return []
        return sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".csv") and not f.startswith("~$")
        ])

    def _update_incremental() -> tuple[int, int]:
        processed = _load_processed()
        files = _scan_logs(DRAW_FOLDER)
        if not files:
            return (0, 0)

        new_or_changed = []
        for path in files:
            sig = _file_signature(path)
            key = os.path.basename(path)
            prev = processed.get(key)
            if prev is None or prev.get("mtime") != sig["mtime"] or prev.get("size") != sig["size"]:
                new_or_changed.append(path)

        if not new_or_changed:
            return (0, 0)

        added_rows = 0
        for path in new_or_changed:
            df_new = _compute_rows_for_log(path)
            if df_new is not None and not df_new.empty:
                _append_cache(df_new)
                added_rows += len(df_new)

            # mark processed even if empty (bad file) to prevent endless loop
            key = os.path.basename(path)
            processed[key] = _file_signature(path)

        _save_processed(processed)
        return (len(new_or_changed), added_rows)

    def _mad(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        med = np.median(x)
        return float(np.median(np.abs(x - med)))

    def _add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        d["window_time"] = pd.to_datetime(d["window_time"], errors="coerce")
        d = d.dropna(subset=["window_time"])
        d.sort_values(["pair_key", "window_time"], inplace=True)

        flags = []
        for pair_key, g in d.groupby("pair_key", sort=False):
            vals = g["corr"].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() < MIN_HISTORY_FOR_OUTLIERS:
                flags.extend([False] * len(g))
                continue
            base = np.median(vals[finite])
            mad = _mad(vals)
            if not np.isfinite(mad) or mad == 0:
                flags.extend([False] * len(g))
                continue
            z = np.abs(vals - base) / (mad + 1e-12)
            out = (z > OUTLIER_K) & np.isfinite(z)
            flags.extend(out.tolist())

        d["is_outlier"] = flags
        return d

    def _pair_volatility(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("pair_key")["corr"].agg(["count", "std", "mean", "min", "max"]).reset_index()
        g.rename(columns={"count": "n_points"}, inplace=True)
        g.sort_values(["std", "n_points"], ascending=[False, False], inplace=True)
        return g

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.write(f"**Logs folder:** `{DRAW_FOLDER}`")
    with c2:
        auto_scan = st.toggle("Auto-scan", value=True, key="corr_auto_scan")
    with c3:
        if st.button("🔄 Scan now", use_container_width=True, key="corr_scan_now"):
            n_files, n_rows = _update_incremental()
            st.success(f"Processed {n_files} new/changed logs • added {n_rows} rows")

    if auto_scan:
        n_files, n_rows = _update_incremental()
        if n_files > 0:
            st.info(f"Auto update: processed {n_files} logs • +{n_rows} rows")

    # ---------------------------------------------------------
    # Load cache
    # ---------------------------------------------------------
    cache = _load_cache()
    if cache.empty:
        st.warning("No cached correlation data yet. Add draw logs to data_set_csv and scan.")
        return

    cache = _add_outlier_flags(cache)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Cached rows", f"{len(cache):,}")
    s2.metric("Pairs", f"{cache['pair_key'].nunique():,}")
    s3.metric("Logs", f"{cache['log_id'].nunique():,}")
    s4.metric("Outlier points", f"{int(cache['is_outlier'].sum()):,}")

    # Latest-outlier table (latest point per pair)
    st.markdown("### 🚨 Latest outliers (per pair)")
    latest = (
        cache.sort_values("window_time")
        .groupby("pair_key", as_index=False)
        .tail(1)
        .sort_values(["is_outlier", "window_time"], ascending=[False, False])
    )
    latest_out = latest[latest["is_outlier"] == True].copy()

    if latest_out.empty:
        st.success("No latest-point outliers detected.")
    else:
        st.dataframe(
            latest_out[["window_time", "pair_key", "corr", "n_points", "log_id", "method"]].head(80),
            use_container_width=True,
            hide_index=True
        )

    # Volatility ranking
    st.markdown("### 📈 Top changing pairs (auto)")
    vol = _pair_volatility(cache.dropna(subset=["corr"]))
    st.dataframe(vol.head(40), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # Many auto plots
    # ---------------------------------------------------------
    st.markdown("### 🧪 Correlation vs time (auto plots)")

    plot_mode = st.radio(
        "Plot mode",
        ["Top changing pairs", "Only pairs with latest outlier", "All pairs (paginated)"],
        index=0,
        horizontal=True,
        key="corr_plot_mode"
    )

    if plot_mode == "Only pairs with latest outlier":
        pair_list = latest_out["pair_key"].dropna().unique().tolist()
        if not pair_list:
            st.info("No outlier pairs to plot.")
            return
    elif plot_mode == "All pairs (paginated)":
        pair_list = sorted(cache["pair_key"].dropna().unique().tolist())
    else:
        pair_list = vol["pair_key"].head(120).tolist()

    plots_per_page = st.number_input(
        "Plots per page",
        min_value=6, max_value=60,
        value=DEFAULT_PLOTS_PER_PAGE, step=6,
        key="corr_plots_per_page"
    )

    total_pages = max(1, math.ceil(len(pair_list) / plots_per_page))
    page = st.number_input(
        "Page",
        min_value=1, max_value=total_pages,
        value=1, step=1,
        key="corr_page"
    )

    a = (page - 1) * plots_per_page
    b = min(len(pair_list), a + plots_per_page)
    st.caption(f"Showing pairs {a + 1}–{b} of {len(pair_list)}")

    show_pairs = pair_list[a:b]

    for pair_key in show_pairs:
        g = cache[cache["pair_key"] == pair_key].copy()
        if g.empty:
            continue
        g["window_time"] = pd.to_datetime(g["window_time"], errors="coerce")
        g = g.dropna(subset=["window_time"]).sort_values("window_time")

        title = f"{pair_key} • points={len(g)}"
        if bool(g["is_outlier"].tail(1).iloc[0]):
            title += "  🚨"

        with st.expander(title, expanded=False):
            st.write(
                f"corr min/mean/max: "
                f"`{g['corr'].min():.3f}` / `{g['corr'].mean():.3f}` / `{g['corr'].max():.3f}`"
            )

            fig = px.line(g, x="window_time", y="corr", markers=True)
            out = g[g["is_outlier"] == True]
            if not out.empty:
                fig2 = px.scatter(out, x="window_time", y="corr")
                for tr in fig2.data:
                    fig.add_trace(tr)

            fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                g.tail(12)[["window_time", "corr", "n_points", "log_id", "is_outlier"]],
                use_container_width=True,
                hide_index=True
            )

    with st.expander("📚 Cache details"):
        st.write("Cache folder:", CACHE_DIR)
        st.write("Cache file:", CACHE_PARQUET if os.path.exists(CACHE_PARQUET) else CACHE_CSV)
        st.dataframe(cache.head(200), use_container_width=True, hide_index=True)

def render_manuals_browser(base_dir: str, folder_name: str = "manuals"):
    import os, sys, pathlib, subprocess
    import streamlit as st

    manuals_folder = os.path.join(base_dir, folder_name)
    os.makedirs(manuals_folder, exist_ok=True)

    st.subheader("📚 Manuals")
    st.caption(f"Put manuals directly in: {manuals_folder} (PDF, DOCX, XLSX, images…)")

    exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".png", ".jpg", ".jpeg"}

    def open_file(path: str):
        path = os.path.abspath(path)
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore
            else:
                subprocess.Popen(["xdg-open", path])
            return True, ""
        except Exception as e:
            return False, str(e)

    p = pathlib.Path(manuals_folder)
    subfolders = sorted([x.name for x in p.iterdir() if x.is_dir()])
    folder_options = ["(root)"] + subfolders

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_folder = st.selectbox("Location", folder_options, key="manuals_location")

    active_path = manuals_folder if selected_folder == "(root)" else os.path.join(manuals_folder, selected_folder)
    files_here = sorted(
        [x for x in pathlib.Path(active_path).iterdir()
         if x.is_file() and x.suffix.lower() in exts and not x.name.startswith("~$")],
        key=lambda x: x.name.lower()
    )

    with c2:
        if not files_here:
            st.info("No manuals found here.")
            return

        selected_file = st.selectbox("Select manual", [f.name for f in files_here], key="manuals_file")
        full_path = os.path.join(active_path, selected_file)

        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("📄 Open", key="manuals_open", use_container_width=True):
                ok, err = open_file(full_path)
                st.success("Opened") if ok else st.error(err)
        with colB:
            st.code(full_path, language="text")

def render_maintenance_history(con, limit: int = 200, height: int = 320):
    import pandas as pd
    import streamlit as st

    with st.expander("🗃️ Maintenance history (DuckDB)"):
        try:
            recent = con.execute(f"""
                SELECT action_ts, component, task, tracking_mode, hours_source,
                       done_date, done_hours, done_draw, actor, source_file
                FROM maintenance_actions
                ORDER BY action_ts DESC
                LIMIT {int(limit)}
            """).fetchdf()

            if not recent.empty:
                recent["done_date"] = pd.to_datetime(recent["done_date"], errors="coerce").dt.date
                recent["action_ts"] = pd.to_datetime(recent["action_ts"], errors="coerce")

            st.dataframe(recent, use_container_width=True, height=int(height))
        except Exception as e:
            st.warning(f"DB read failed: {e}")

def render_maintenance_load_report(files, load_errors):
    import pandas as pd
    import streamlit as st

    with st.expander("Load report"):
        try:
            st.write("Loaded files:", sorted(list(files or [])))
        except Exception:
            st.write("Loaded files:", files)

        if load_errors:
            st.warning("Some files failed to load:")
            st.dataframe(
                pd.DataFrame(load_errors, columns=["File", "Error"]),
                use_container_width=True
            )

def render_new_draw_checklist(
        dfm,
        current_draw_count: int,
        state: dict,
        state_path: str,
        save_state_fn,
):
    import streamlit as st

    last_draw_count = int(state.get("last_draw_count", current_draw_count))
    new_draws = current_draw_count - last_draw_count

    st.caption(
        f"📦 Draw CSVs in data_set_csv: **{current_draw_count}**"
        + (f"  |  🆕 new since last run: **{new_draws}**" if new_draws > 0 else "")
    )

    if new_draws <= 0:
        state["last_draw_count"] = current_draw_count
        save_state_fn(state_path, state)
        return

    st.warning(f"🆕 New draw detected! {new_draws} new draw CSV file(s) were added.")

    routine = dfm[dfm["Tracking_Mode"].str.lower().eq("event")].copy()
    text = (
            routine["Task"].fillna("")
            + " "
            + routine["Notes"].fillna("")
            + " "
            + routine["Procedure_Summary"].fillna("")
    ).str.lower()

    pre = routine[text.str.contains(r"\bpre\b|\bbefore\b|\bstartup\b")].copy()
    post = routine[text.str.contains(r"\bpost\b|\bafter\b|\bshutdown\b|\bend\b")].copy()

    with st.container(border=True):
        st.markdown("### ✅ Routine checklist (Pre / Post)")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Pre-Draw")
            if pre.empty:
                st.info("No pre-draw routine tasks found.")
            for i, r in pre.iterrows():
                st.checkbox(
                    f"{r.get('Component', '')} — {r.get('Task', '')}",
                    key=f"pre_{i}"
                )

        with colB:
            st.markdown("#### Post-Draw")
            if post.empty:
                st.info("No post-draw routine tasks found.")
            for i, r in post.iterrows():
                st.checkbox(
                    f"{r.get('Component', '')} — {r.get('Task', '')}",
                    key=f"post_{i}"
                )

        if st.button("✅ Acknowledge checklist (hide until next draw)", type="primary"):
            state["last_draw_count"] = current_draw_count
            save_state_fn(state_path, state)
            st.success("Checklist acknowledged.")
            st.rerun()

def render_maintenance_dashboard_metrics(dfm):
    import streamlit as st

    st.subheader("Dashboard")

    overdue = int((dfm["Status"] == "OVERDUE").sum())
    due_soon = int((dfm["Status"] == "DUE SOON").sum())
    routine = int((dfm["Status"] == "ROUTINE").sum())
    ok = int((dfm["Status"] == "OK").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OVERDUE", overdue)
    c2.metric("DUE SOON", due_soon)
    c3.metric("ROUTINE", routine)
    c4.metric("OK", ok)

def render_maintenance_horizon_selector(current_draw_count: int):
    import streamlit as st

    st.subheader("📅 Future schedule view")

    st.markdown(
        """
        <style>
        div.stButton > button {
            width: 100%;
            height: 44px;
            border-radius: 12px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.session_state.setdefault("maint_horizon_hours", 10)
    st.session_state.setdefault("maint_horizon_days", 7)
    st.session_state.setdefault("maint_horizon_draws", 5)

    def button_group(title, options, value, key):
        st.caption(title)
        cols = st.columns(len(options))
        for col, (label, v) in zip(cols, options):
            if col.button(label, key=f"{key}_{v}", type="primary" if v == value else "secondary"):
                return v
        return value

    c1, c2, c3 = st.columns(3)

    with c1:
        st.session_state["maint_horizon_hours"] = button_group(
            "Hours horizon",
            [("10", 10), ("50", 50), ("100", 100)],
            st.session_state["maint_horizon_hours"],
            "mh"
        )

    with c2:
        st.session_state["maint_horizon_days"] = button_group(
            "Calendar horizon",
            [("Week", 7), ("Month", 30), ("3 Months", 90)],
            st.session_state["maint_horizon_days"],
            "md"
        )

    with c3:
        st.session_state["maint_horizon_draws"] = button_group(
            "Draw horizon",
            [("5", 5), ("10", 10), ("50", 50)],
            st.session_state["maint_horizon_draws"],
            "mD"
        )

    st.caption(
        f"📦 Now: **{current_draw_count}** → "
        f"Horizon: **{st.session_state['maint_horizon_draws']}** → "
        f"Up to draw **#{current_draw_count + st.session_state['maint_horizon_draws']}**"
    )

    return (
        st.session_state["maint_horizon_hours"],
        st.session_state["maint_horizon_days"],
        st.session_state["maint_horizon_draws"],
    )

def render_maintenance_roadmaps(
        dfm: pd.DataFrame,
        current_date,
        current_draw_count: int,
        furnace_hours: float,
        uv1_hours: float,
        uv2_hours: float,
        horizon_hours: int,
        horizon_days: int,
        horizon_draws: int,
):
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    import pandas as pd
    import streamlit as st

    def status_color(s):
        s = str(s).upper()
        if s == "OVERDUE":
            return "#ff4d4d"
        if s == "DUE SOON":
            return "#ffcc00"
        return "#66ff99"

    def roadmap(x0, x1, title, xlabel, df, xcol, hover):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[x0, x1], y=[0, 0], mode="lines",
                                 line=dict(width=6, color="rgba(180,180,180,0.2)"),
                                 hoverinfo="skip"))
        fig.add_vline(x=x0, line_dash="dash")

        if df is not None and not df.empty:
            fig.add_trace(go.Scatter(
                x=df[xcol],
                y=[0] * len(df),
                mode="markers",
                marker=dict(
                    size=13,
                    color=[status_color(s) for s in df["Status"]],
                    line=dict(width=1, color="rgba(255,255,255,0.5)")
                ),
                text=df[hover],
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            mid = x0 + (x1 - x0) / 2
            fig.add_annotation(x=mid, y=0, text="No tasks in horizon", showarrow=False)

        fig.update_layout(
            title=title,
            height=220,
            yaxis=dict(visible=False),
            xaxis=dict(title=xlabel),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    def norm_group(src):
        s = str(src).lower()
        if "uv1" in s:
            return "UV1"
        if "uv2" in s:
            return "UV2"
        return "FURNACE"

    # -------- Build datasets --------
    hours_df = dfm[dfm["Tracking_Mode_norm"] == "hours"].copy()
    hours_df["Due"] = pd.to_numeric(hours_df["Next_Due_Hours"], errors="coerce")
    hours_df = hours_df.dropna(subset=["Due"])
    hours_df["Group"] = hours_df["Hours_Source"].apply(norm_group)
    hours_df["Hover"] = hours_df["Component"] + " — " + hours_df["Task"] + "<br>Status: " + hours_df["Status"]

    cal_df = dfm[dfm["Tracking_Mode_norm"] == "calendar"].copy()
    cal_df["Due"] = pd.to_datetime(cal_df["Next_Due_Date"], errors="coerce")
    cal_df = cal_df.dropna(subset=["Due"])
    cal_df["Hover"] = cal_df["Component"] + " — " + cal_df["Task"] + "<br>Status: " + cal_df["Status"]

    draw_df = dfm[dfm["Tracking_Mode_norm"] == "draws"].copy()
    draw_df["Due"] = pd.to_numeric(draw_df["Next_Due_Draw"], errors="coerce")
    draw_df = draw_df.dropna(subset=["Due"])
    draw_df["Hover"] = draw_df["Component"] + " — " + draw_df["Task"] + "<br>Status: " + draw_df["Status"]

    # -------- Render --------
    st.markdown("### 🔥 Furnace / 💡 UV timelines")
    c1, c2, c3 = st.columns(3)

    with c1:
        x0, x1 = furnace_hours, furnace_hours + horizon_hours
        st.plotly_chart(
            roadmap(x0, x1, "FURNACE", "Hours",
                    hours_df[(hours_df["Group"] == "FURNACE") & hours_df["Due"].between(x0, x1)],
                    "Due", "Hover"),
            use_container_width=True
        )

    with c2:
        x0, x1 = uv1_hours, uv1_hours + horizon_hours
        st.plotly_chart(
            roadmap(x0, x1, "UV1", "Hours",
                    hours_df[(hours_df["Group"] == "UV1") & hours_df["Due"].between(x0, x1)],
                    "Due", "Hover"),
            use_container_width=True
        )

    with c3:
        x0, x1 = uv2_hours, uv2_hours + horizon_hours
        st.plotly_chart(
            roadmap(x0, x1, "UV2", "Hours",
                    hours_df[(hours_df["Group"] == "UV2") & hours_df["Due"].between(x0, x1)],
                    "Due", "Hover"),
            use_container_width=True
        )

    st.markdown("### 🧵 Draw timeline")
    d0, d1 = current_draw_count, current_draw_count + horizon_draws
    st.plotly_chart(
        roadmap(d0, d1, "Draw-based tasks", "Draw #",
                draw_df[draw_df["Due"].between(d0, d1)],
                "Due", "Hover"),
        use_container_width=True
    )

    st.markdown("### 🗓️ Calendar timeline")
    t0 = pd.Timestamp(current_date)
    t1 = t0 + pd.Timedelta(days=horizon_days)
    st.plotly_chart(
        roadmap(t0, t1, "Calendar tasks", "Date",
                cal_df[(cal_df["Due"] >= t0) & (cal_df["Due"] <= t1)],
                "Due", "Hover"),
        use_container_width=True
    )

def render_maintenance_done_editor(dfm):
    import streamlit as st

    st.subheader("Mark tasks as done")

    focus_default = ["OVERDUE", "DUE SOON", "ROUTINE"]
    focus_status = st.multiselect(
        "Work on these statuses",
        ["OVERDUE", "DUE SOON", "ROUTINE", "OK"],
        default=focus_default,
        key="maint_focus_status"
    )

    work = (
        dfm[dfm["Status"].isin(focus_status)]
        .copy()
        .sort_values(["Status", "Component", "Task"])
    )
    work["Done_Now"] = False

    cols = [
        "Done_Now",
        "Status", "Component", "Task", "Task_ID",
        "Tracking_Mode", "Hours_Source", "Current_Hours_For_Task",
        "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
        "Next_Due_Date", "Next_Due_Hours", "Next_Due_Draw",
        "Manual_Name", "Page", "Document",
        "Owner", "Source_File"
    ]
    cols = [c for c in cols if c in work.columns]

    edited = st.data_editor(
        work[cols],
        use_container_width=True,
        height=420,
        column_config={
            "Done_Now": st.column_config.CheckboxColumn(
                "Done now", help="Tick tasks you completed"
            )
        },
        disabled=[c for c in cols if c != "Done_Now"],
        key="maint_editor"
    )

    return edited

def render_maintenance_apply_done(
        edited,
        *,
        dfm,
        current_date,
        current_draw_count,
        actor,
        MAINT_FOLDER,
        STATE_PATH,
        con,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
        pick_current_hours,
        mode_norm,
        MAINT_ACTIONS_CSV,                  # ✅ NEW
        append_maintenance_actions_csv,      # ✅ NEW
):
    import os
    import streamlit as st
    import pandas as pd
    import datetime as dt
    import time

    if not st.button("✅ Apply 'Done Now' updates", type="primary"):
        return

    done_rows = edited[edited["Done_Now"] == True].copy()
    if done_rows.empty:
        st.info("No tasks selected.")
        return

    updated = 0
    problems = []

    # ----------------------------
    # 1) Update each source file
    # ----------------------------
    for src, grp in done_rows.groupby("Source_File"):
        path = os.path.join(MAINT_FOLDER, str(src))
        try:
            raw = read_file(path)
            df_src = normalize_df(raw)
            df_src["Tracking_Mode_norm"] = df_src["Tracking_Mode"].apply(mode_norm)

            for _, r in grp.iterrows():
                mode = mode_norm(r.get("Tracking_Mode", ""))

                mask = (
                    df_src["Component"].astype(str).eq(str(r.get("Component", ""))) &
                    df_src["Task"].astype(str).eq(str(r.get("Task", "")))
                )

                if not mask.any():
                    continue

                # always set done date
                df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()

                if mode == "hours":
                    df_src.loc[mask, "Last_Done_Hours"] = float(pick_current_hours(r.get("Hours_Source", "")))
                elif mode == "draws":
                    df_src.loc[mask, "Last_Done_Draw"] = int(current_draw_count)

                updated += int(mask.sum())

            out = templateize_df(df_src, list(raw.columns))
            write_file(path, out)

        except Exception as e:
            problems.append((src, str(e)))

    st.success(f"Updated {updated} task(s).")

    # ----------------------------
    # 2) Build DONE actions rows (one per selected task)
    # ----------------------------
    # Use python datetime (not numpy) to avoid dtype issues
    now_dt = dt.datetime.now()
    action_ts = dt.datetime.combine(current_date, now_dt.time())

    base_ms = int(time.time() * 1000)
    actions = []

    for i, (_, r) in enumerate(done_rows.reset_index(drop=True).iterrows()):
        mode = mode_norm(r.get("Tracking_Mode", ""))

        done_hours = None
        done_draw = None
        if mode == "hours":
            done_hours = float(pick_current_hours(r.get("Hours_Source", "")))
        elif mode == "draws":
            done_draw = int(current_draw_count)

        actions.append({
            "action_id": int(base_ms + i),  # ✅ unique per row
            "action_ts": action_ts,
            "component": str(r.get("Component", "")),
            "task": str(r.get("Task", "")),
            "task_id": str(r.get("Task_ID", "")),
            "tracking_mode": str(r.get("Tracking_Mode", "")),
            "hours_source": str(r.get("Hours_Source", "")),
            "done_date": current_date,
            "done_hours": done_hours,
            "done_draw": done_draw,
            "source_file": str(r.get("Source_File", "")),
            "actor": str(actor),
            "note": "",
        })

    # ----------------------------
    # 3) Append to CSV log (one-line rows)
    # ----------------------------
    try:
        append_maintenance_actions_csv(MAINT_ACTIONS_CSV, actions)
    except Exception as e:
        st.warning(f"Failed writing maintenance_actions_log.csv: {e}")

    # ----------------------------
    # 4) Insert into DuckDB safely (no con.register)
    # ----------------------------
    try:
        insert_sql = """
            INSERT INTO maintenance_actions (
                action_id, action_ts, component, task, task_id, tracking_mode, hours_source,
                done_date, done_hours, done_draw, source_file, actor, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for a in actions:
            con.execute(insert_sql, [
                int(a["action_id"]),
                a["action_ts"],  # python datetime
                a["component"],
                a["task"],
                a["task_id"],
                a["tracking_mode"],
                a["hours_source"],
                a["done_date"],  # python date
                a["done_hours"],
                a["done_draw"],
                a["source_file"],
                a["actor"],
                a["note"],
            ])
    except Exception as e:
        st.warning(f"DB insert failed (CSV log still written): {e}")

    if problems:
        st.warning("Some files had issues:")
        st.dataframe(pd.DataFrame(problems, columns=["File", "Error"]))

    st.rerun()

def render_maintenance_tasks_snapshot(dfm, con):
    import hashlib
    import datetime as dt
    import numpy as np
    import streamlit as st

    def task_key(r):
        s = f"{r.get('Source_File', '')}|{r.get('Task_ID', '')}|{r.get('Component', '')}|{r.get('Task', '')}"
        return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

    df = dfm.copy()
    df["task_key"] = df.apply(task_key, axis=1)
    df["loaded_at"] = dt.datetime.now()

    cols = [
        "task_key", "Task_ID", "Component", "Task",
        "Tracking_Mode", "Hours_Source",
        "Interval_Value", "Interval_Unit",
        "Due_Threshold_Days",
        "Manual_Name", "Page", "Document",
        "Procedure_Summary", "Notes", "Owner",
        "Source_File", "loaded_at"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].rename(columns={
        "Task_ID": "task_id",
        "Component": "component",
        "Task": "task",
        "Tracking_Mode": "tracking_mode",
        "Hours_Source": "hours_source",
        "Interval_Value": "interval_value",
        "Interval_Unit": "interval_unit",
        "Due_Threshold_Days": "due_threshold_days",
        "Manual_Name": "manual_name",
        "Page": "page",
        "Document": "document",
        "Procedure_Summary": "procedure_summary",
        "Notes": "notes",
        "Owner": "owner",
        "Source_File": "source_file",
    })

    try:
        con.execute("DELETE FROM maintenance_tasks;")
        con.register("tmp_maint_tasks", df)
        con.execute("INSERT INTO maintenance_tasks SELECT * FROM tmp_maint_tasks;")
        con.unregister("tmp_maint_tasks")
    except Exception as e:
        st.warning(f"DuckDB sync failed: {e}")

def load_maintenance_files(MAINT_FOLDER=None):
    import os
    import numpy as np
    import pandas as pd
    import streamlit as st
    if not MAINT_FOLDER:
        MAINT_FOLDER = os.path.join(os.getcwd(), "maintenance")

    os.makedirs(MAINT_FOLDER, exist_ok=True)
    normalize_map = {
        "equipment": "Component",
        "task name": "Task",
        "task id": "Task_ID",
        "interval type": "Interval_Type",
        "interval value": "Interval_Value",
        "interval unit": "Interval_Unit",
        "tracking mode": "Tracking_Mode",
        "hours source": "Hours_Source",
        "calendar rule": "Calendar_Rule",
        "due threshold (days)": "Due_Threshold_Days",
        "document name": "Manual_Name",
        "document file/link": "Document",
        "manual page": "Page",
        "procedure summary": "Procedure_Summary",
        "safety/notes": "Notes",
        "owner": "Owner",
        "last done date": "Last_Done_Date",
        "last done hours": "Last_Done_Hours",
        "last done draw": "Last_Done_Draw",
    }

    REQUIRED = ["Component", "Task", "Tracking_Mode"]
    OPTIONAL = [
        "Task_ID",
        "Interval_Type", "Interval_Value", "Interval_Unit",
        "Due_Threshold_Days",
        "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
        "Manual_Name", "Page", "Document",
        "Procedure_Summary", "Notes", "Owner",
        "Hours_Source", "Calendar_Rule",
    ]

    def read_file(path):
        return pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)

    def normalize_df(df):
        df = df.copy()
        df.rename(columns={c: normalize_map.get(str(c).strip().lower(), c) for c in df.columns}, inplace=True)
        for c in REQUIRED:
            if c not in df.columns:
                df[c] = np.nan
        for c in OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan
        return df

    files = [
        f for f in os.listdir(MAINT_FOLDER)
        if f.lower().endswith((".xlsx", ".xls", ".csv"))
    ]

    if not files:
        st.warning("No maintenance files found in /maintenance folder.")
        st.stop()

    frames = []
    load_errors = []

    for fname in sorted(files):
        path = os.path.join(MAINT_FOLDER, fname)
        try:
            raw = read_file(path)
            if raw is None or raw.empty:
                continue
            df = normalize_df(raw)
            df["Source_File"] = fname
            frames.append(df)
        except Exception as e:
            load_errors.append((fname, str(e)))

    if not frames:
        st.error("No valid maintenance data could be loaded.")
        if load_errors:
            st.dataframe(pd.DataFrame(load_errors, columns=["File", "Error"]))
        st.stop()

    dfm = pd.concat(frames, ignore_index=True)
    return dfm, files, load_errors

def render_maintenance_tasks_editor(
        *,
        MAINT_FOLDER: str,
        files: list,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
):
    import os
    import pandas as pd
    import streamlit as st

    st.subheader("🛠️ Maintenance tasks editor (Update / Create)")
    st.caption("Edit maintenance tasks directly in the source Excel/CSV. Supports creating a new file template.")

    with st.expander("✏️ Edit an existing maintenance file", expanded=False):
        if not files:
            st.info("No maintenance files found.")
            return

        selected = st.selectbox("Select file", options=sorted(files), key="maint_task_editor_file")
        path = os.path.join(MAINT_FOLDER, selected)

        try:
            raw = read_file(path)
            if raw is None or raw.empty:
                st.warning("File is empty. You can still add rows and save.")
                raw = pd.DataFrame()

            original_cols = list(raw.columns)
            df_src = normalize_df(raw)

            # Keep only "real" columns (not computed ones)
            keep_order = [
                "Component", "Task", "Task_ID",
                "Tracking_Mode", "Hours_Source",
                "Interval_Type", "Interval_Value", "Interval_Unit",
                "Due_Threshold_Days",
                "Manual_Name", "Page", "Document",
                "Procedure_Summary", "Notes", "Owner",
                "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
            ]
            for c in keep_order:
                if c not in df_src.columns:
                    df_src[c] = pd.NA
            df_src = df_src[keep_order]

            edited = st.data_editor(
                df_src,
                use_container_width=True,
                height=420,
                num_rows="dynamic",
                key="maint_task_editor_grid",
            )

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("💾 Save file", type="primary", key="maint_task_editor_save"):
                    out = templateize_df(edited, original_cols if original_cols else list(edited.columns))
                    write_file(path, out)
                    st.success("Saved.")
                    st.rerun()

            with c2:
                st.code(path, language="text")

        except Exception as e:
            st.error(f"Failed to load/edit file: {e}")

    with st.expander("➕ Create a new maintenance file", expanded=False):
        new_name = st.text_input("New file name (example: maintenance_tasks.csv)", key="maint_new_file_name").strip()
        if st.button("Create template file", key="maint_new_file_create"):
            if not new_name:
                st.warning("Enter a file name.")
                return
            if not (new_name.lower().endswith(".csv") or new_name.lower().endswith(".xlsx")):
                st.warning("Use .csv or .xlsx")
                return

            new_path = os.path.join(MAINT_FOLDER, new_name)
            if os.path.exists(new_path):
                st.warning("File already exists.")
                return

            template_cols = [
                "Component", "Task", "Task_ID",
                "Tracking_Mode", "Hours_Source",
                "Interval_Type", "Interval_Value", "Interval_Unit",
                "Due_Threshold_Days",
                "Manual_Name", "Page", "Document",
                "Procedure_Summary", "Notes", "Owner",
                "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
            ]
            df_new = pd.DataFrame(columns=template_cols)
            write_file(new_path, df_new)
            st.success(f"Created: {new_path}")
            st.rerun()

def render_faults_section(*, con, MAINT_FOLDER: str, actor: str):
    import os
    import time
    import datetime as dt
    import pandas as pd
    import streamlit as st

    st.subheader("🧯 Faults")

    # =========================
    # Config
    # =========================
    FAULTS_FILE = os.path.join(MAINT_FOLDER, "faults.xlsx")  # or faults.csv
    REQUIRED_COLS = [
        "Fault ID",
        "DateTime",
        "Component",
        "Subsystem/Area",
        "Severity",
        "Title",
        "Description",
        "Immediate Action",
        "Root Cause",
        "Corrective Action",
        "Preventive Action",
        "Status",
        "Owner",
        "Attachments/Links",
        "Related Draw",
        "Logged By",
        "Closed Date",
        "Notes",
    ]
    STATUS_OPTIONS = ["Open", "Monitoring", "Fixed", "Closed"]
    SEVERITY_OPTIONS = ["Low", "Medium", "High", "Critical"]

    # =========================
    # I/O helpers
    # =========================
    def _read_faults(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame(columns=REQUIRED_COLS)
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)

    def _write_faults(path: str, df: pd.DataFrame):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)

    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in REQUIRED_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[REQUIRED_COLS]

    # =========================
    # Pull components list from maintenance_tasks snapshot
    # =========================
    components = []
    try:
        rows = con.execute("""
            SELECT DISTINCT component
            FROM maintenance_tasks
            WHERE component IS NOT NULL AND TRIM(component) <> ''
            ORDER BY component
        """).fetchall()
        components = [r[0] for r in rows]
    except Exception:
        components = []

    # =========================
    # Load data
    # =========================
    faults_df = _ensure_cols(_read_faults(FAULTS_FILE))

    # Parse datetime for sorting/logic
    faults_df["_dt"] = pd.to_datetime(faults_df["DateTime"], errors="coerce")
    faults_df["_sev"] = faults_df["Severity"].fillna("").astype(str).str.strip()
    faults_df["_status"] = faults_df["Status"].fillna("").astype(str).str.strip()

    # =========================
    # LIVE INDICATOR (High/Critical OPEN)
    # =========================
    open_mask = faults_df["_status"].str.lower().isin(["open", "monitoring"])
    hc_mask = faults_df["_sev"].str.lower().isin(["high", "critical"])
    urgent_open = faults_df[open_mask & hc_mask].copy()

    if not urgent_open.empty:
        urgent_open = urgent_open.sort_values(["_dt"], ascending=False)
        n_crit = int((urgent_open["_sev"].str.lower() == "critical").sum())
        n_high = int((urgent_open["_sev"].str.lower() == "high").sum())
        st.error(f"🚨 ACTIVE SEVERE FAULTS: {len(urgent_open)}  (Critical: {n_crit} | High: {n_high})")

        with st.container(border=True):
            st.markdown("**Open High/Critical faults (latest first):**")
            for _, r in urgent_open.head(6).iterrows():
                fid = str(r.get("Fault ID", "") or "")
                sev = str(r.get("Severity", "") or "")
                comp = str(r.get("Component", "") or "")
                title = str(r.get("Title", "") or "")
                when = r.get("_dt")
                when_s = when.strftime("%Y-%m-%d %H:%M") if pd.notna(when) else str(r.get("DateTime", "") or "")
                st.write(f"- **{sev}** | **{fid}** | {comp} — {title}  _(since {when_s})_")
    else:
        st.success("✅ No High/Critical open faults right now.")

    # =========================
    # Small top actions
    # =========================
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        open_report = st.button("➕ Report fault", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with c3:
        st.caption(f"Source: `{os.path.basename(FAULTS_FILE)}` in `/maintenance`")

    # =========================
    # Dialog: Report new fault
    # =========================
    @st.dialog("Report a new fault", width="large")
    def _dlg_report_fault():
        nonlocal faults_df

        now = dt.datetime.now()

        colA, colB, colC = st.columns([1.2, 1, 1])
        with colA:
            component = st.selectbox(
                "Component / Part",
                options=(components if components else ["(type manually)"]),
                key="dlg_fault_component"
            )
            if component == "(type manually)":
                component = st.text_input("Component name", key="dlg_fault_component_manual").strip()
        with colB:
            severity = st.selectbox("Severity", SEVERITY_OPTIONS, index=2, key="dlg_fault_severity")
        with colC:
            status = st.selectbox("Status", STATUS_OPTIONS, index=0, key="dlg_fault_status")

        area = st.text_input("Subsystem / Area (optional)", key="dlg_fault_area")
        title = st.text_input("Title", key="dlg_fault_title")
        desc = st.text_area("Description (what happened?)", key="dlg_fault_desc", height=140)

        c4, c5 = st.columns(2)
        with c4:
            immediate = st.text_area("Immediate action", key="dlg_fault_immediate", height=90)
            owner = st.text_input("Owner", key="dlg_fault_owner")
        with c5:
            root = st.text_area("Root cause (if known)", key="dlg_fault_root", height=90)
            related_draw = st.text_input("Related draw (optional)", key="dlg_fault_draw")

        corr = st.text_area("Corrective action (fix)", key="dlg_fault_corrective", height=80)
        prev = st.text_area("Preventive action (avoid repeat)", key="dlg_fault_preventive", height=80)
        links = st.text_input("Attachments/Links (paths, URLs, etc.)", key="dlg_fault_links")
        notes = st.text_area("Notes", key="dlg_fault_notes", height=80)

        colX, colY = st.columns([1, 1])
        with colX:
            do_log = st.button("🧯 Log fault", type="primary", use_container_width=True)
        with colY:
            st.button("Cancel", use_container_width=True)

        if do_log:
            if not title.strip():
                st.error("Title is required.")
                return
            if not str(component).strip():
                st.error("Component is required.")
                return

            fault_id = f"F-{int(time.time())}"
            new_row = {
                "Fault ID": fault_id,
                "DateTime": now.isoformat(sep=" ", timespec="seconds"),
                "Component": component,
                "Subsystem/Area": area,
                "Severity": severity,
                "Title": title,
                "Description": desc,
                "Immediate Action": immediate,
                "Root Cause": root,
                "Corrective Action": corr,
                "Preventive Action": prev,
                "Status": status,
                "Owner": owner,
                "Attachments/Links": links,
                "Related Draw": related_draw,
                "Logged By": actor or "operator",
                "Closed Date": "" if status != "Closed" else now.date().isoformat(),
                "Notes": notes,
            }

            # update file
            base = _ensure_cols(_read_faults(FAULTS_FILE))
            base = pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)
            _write_faults(FAULTS_FILE, base)

            # ALSO log into DuckDB as "maintenance event"
            try:
                con.execute("""
                    INSERT INTO maintenance_actions (
                        action_id, action_ts, component, task, task_id, tracking_mode,
                        hours_source, done_date, done_hours, done_draw,
                        source_file, actor, note
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    int(time.time() * 1000),
                    now,
                    str(component),
                    f"FAULT: {title}",
                    fault_id,
                    "fault",
                    "",
                    now.date(),
                    None,
                    None,
                    os.path.basename(FAULTS_FILE),
                    actor or "operator",
                    (desc or "")[:5000],
                ])
            except Exception as e:
                st.warning(f"Saved to file, but DB log failed: {e}")

            st.success(f"Logged fault: {fault_id}")
            st.rerun()

    if open_report:
        _dlg_report_fault()

    # =========================
    # Compact table + per-row popup view/edit
    # =========================
    st.markdown("### Faults list")

    # Build compact view (no long text)
    view = faults_df.copy()
    view = view.drop(columns=["_sev", "_status"], errors="ignore")
    view = view.sort_values("_dt", ascending=False).drop(columns=["_dt"], errors="ignore")

    compact_cols = ["DateTime", "Fault ID", "Severity", "Status", "Component", "Title", "Owner", "Related Draw"]
    for c in compact_cols:
        if c not in view.columns:
            view[c] = ""

    # Filter controls
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        filt_status = st.multiselect("Status", STATUS_OPTIONS, default=["Open", "Monitoring"], key="faults_filt_status")
    with f2:
        filt_sev = st.multiselect("Severity", SEVERITY_OPTIONS, default=SEVERITY_OPTIONS, key="faults_filt_sev")
    with f3:
        text_q = st.text_input("Search (id/title/component)", key="faults_search").strip().lower()

    filtered = view.copy()
    if filt_status:
        filtered = filtered[filtered["Status"].fillna("").astype(str).isin(filt_status)]
    if filt_sev:
        filtered = filtered[filtered["Severity"].fillna("").astype(str).isin(filt_sev)]
    if text_q:
        hay = (
                filtered["Fault ID"].fillna("").astype(str)
                + " " + filtered["Title"].fillna("").astype(str)
                + " " + filtered["Component"].fillna("").astype(str)
        ).str.lower()
        filtered = filtered[hay.str.contains(text_q, na=False)]

    st.dataframe(filtered[compact_cols], use_container_width=True, height=260)

    # Pick a fault to open in a popup
    ids = filtered["Fault ID"].fillna("").astype(str).tolist()
    ids = [x for x in ids if x.strip()]

    sel = st.selectbox("Open fault (view/edit)", options=([""] + ids), index=0, key="faults_open_select")

    @st.dialog("Fault details", width="large")
    def _dlg_edit_fault(fault_id: str):
        nonlocal faults_df

        df = _ensure_cols(_read_faults(FAULTS_FILE))
        idx = df.index[df["Fault ID"].astype(str) == str(fault_id)]
        if len(idx) == 0:
            st.error("Fault not found in file.")
            return
        i = int(idx[0])
        row = df.loc[i].to_dict()

        st.caption(f"Fault ID: **{fault_id}**")

        colA, colB, colC = st.columns([1.2, 1, 1])
        with colA:
            component = st.selectbox(
                "Component / Part",
                options=(components if components else [row.get("Component", "")]),
                index=(
                    components.index(row.get("Component")) if components and row.get("Component") in components else 0),
                key=f"edit_comp_{fault_id}"
            ) if components else st.text_input("Component / Part", value=str(row.get("Component", "")),
                                               key=f"edit_comp_txt_{fault_id}")
        with colB:
            severity = st.selectbox(
                "Severity",
                SEVERITY_OPTIONS,
                index=(SEVERITY_OPTIONS.index(row.get("Severity")) if row.get("Severity") in SEVERITY_OPTIONS else 1),
                key=f"edit_sev_{fault_id}"
            )
        with colC:
            status = st.selectbox(
                "Status",
                STATUS_OPTIONS,
                index=(STATUS_OPTIONS.index(row.get("Status")) if row.get("Status") in STATUS_OPTIONS else 0),
                key=f"edit_status_{fault_id}"
            )

        area = st.text_input("Subsystem/Area", value=str(row.get("Subsystem/Area", "") or ""),
                             key=f"edit_area_{fault_id}")
        title = st.text_input("Title", value=str(row.get("Title", "") or ""), key=f"edit_title_{fault_id}")
        desc = st.text_area("Description", value=str(row.get("Description", "") or ""), key=f"edit_desc_{fault_id}",
                            height=140)

        c4, c5 = st.columns(2)
        with c4:
            immediate = st.text_area("Immediate Action", value=str(row.get("Immediate Action", "") or ""),
                                     key=f"edit_im_{fault_id}", height=90)
            owner = st.text_input("Owner", value=str(row.get("Owner", "") or ""), key=f"edit_owner_{fault_id}")
        with c5:
            root = st.text_area("Root Cause", value=str(row.get("Root Cause", "") or ""), key=f"edit_root_{fault_id}",
                                height=90)
            related_draw = st.text_input("Related Draw", value=str(row.get("Related Draw", "") or ""),
                                         key=f"edit_draw_{fault_id}")

        corr = st.text_area("Corrective Action", value=str(row.get("Corrective Action", "") or ""),
                            key=f"edit_corr_{fault_id}", height=80)
        prev = st.text_area("Preventive Action", value=str(row.get("Preventive Action", "") or ""),
                            key=f"edit_prev_{fault_id}", height=80)
        links = st.text_input("Attachments/Links", value=str(row.get("Attachments/Links", "") or ""),
                              key=f"edit_links_{fault_id}")
        notes = st.text_area("Notes", value=str(row.get("Notes", "") or ""), key=f"edit_notes_{fault_id}", height=80)

        closed_date = str(row.get("Closed Date", "") or "")
        if status == "Closed" and not closed_date.strip():
            closed_date = dt.date.today().isoformat()
        if status != "Closed":
            closed_date = ""

        colX, colY = st.columns([1, 1])
        with colX:
            save_btn = st.button("💾 Save changes", type="primary", use_container_width=True,
                                 key=f"save_fault_{fault_id}")
        with colY:
            st.button("Close", use_container_width=True, key=f"close_fault_{fault_id}")

        if save_btn:
            df.loc[i, "Component"] = component
            df.loc[i, "Subsystem/Area"] = area
            df.loc[i, "Severity"] = severity
            df.loc[i, "Status"] = status
            df.loc[i, "Title"] = title
            df.loc[i, "Description"] = desc
            df.loc[i, "Immediate Action"] = immediate
            df.loc[i, "Root Cause"] = root
            df.loc[i, "Corrective Action"] = corr
            df.loc[i, "Preventive Action"] = prev
            df.loc[i, "Owner"] = owner
            df.loc[i, "Attachments/Links"] = links
            df.loc[i, "Related Draw"] = related_draw
            df.loc[i, "Closed Date"] = closed_date
            df.loc[i, "Notes"] = notes

            _write_faults(FAULTS_FILE, df)
            st.success("Saved.")
            st.rerun()

    if sel:
        _dlg_edit_fault(sel)

def append_preform_length(preform_name: str, length_cm: float, source_draw: str):
    import pandas as pd
    from datetime import datetime
    import os

    row = {
        "Preform Name": str(preform_name).strip(),
        "Length": float(length_cm),
        "Unit": "cm",
        "Updated Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source Draw": str(source_draw).strip(),
    }

    if os.path.exists(PREFORMS_FILE):
        df = pd.read_csv(PREFORMS_FILE)
    else:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(PREFORMS_FILE, index=False)
    st.success(f"🧱 Preform **{preform_name}** updated → {preform_len_after_cm:.1f} cm remaining.")

def get_most_recent_dataset_csv(dataset_dir=P.dataset_dir):
    if not os.path.exists(dataset_dir):
        return None
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    if not files:
        return None
    return os.path.basename(max(files, key=os.path.getmtime))

def render_tm_drum_fiber_visual_from_csv(df_params: pd.DataFrame, dataset_name: str):
    """
    Draws drum + fiber + zones/segments.
    Uses ONLY what exists in the dataset CSV (no log needed).
    Priority:
      1) Zone i Start/End (from end)  -> shows ALL zones accurately
      2) Marked Zone i Length         -> shows ALL zones sequentially
      3) STEP i Action/Length         -> fallback merged plan
    """
    total_km = get_float_param(df_params, "Fiber Total Length (Log End)", 0.0)
    total_save = get_float_param(df_params, "Total Saved Length", 0.0)
    total_cut = get_float_param(df_params, "Total Cut Length", 0.0)

    # Try to get explicit zone positions
    zones = _parse_zones_from_end(df_params)

    # If no positions, try to get marked zone lengths and place sequentially
    if not zones:
        marked_lengths = _parse_marked_zone_lengths(df_params)
        if marked_lengths:
            a = 0.0
            zones = []
            for i, L in enumerate(marked_lengths, start=1):
                zones.append({"i": i, "a": a, "b": a + L, "len": L})
                a += L

    # If still none, fallback to steps (SAVE segments)
    if not zones:
        steps = _parse_steps(df_params)
        if steps:
            a = 0.0
            zones = []
            zi = 1
            for action, L in steps:
                if action == "SAVE":
                    zones.append({"i": zi, "a": a, "b": a + L, "len": L})
                    zi += 1
                a += L

    if not zones:
        st.info(
            "No zone information found in dataset CSV (no Zone-from-end, no Marked Zone Lengths, no STEP SAVE segments).")
        return

    # If total length missing, infer from max(b)
    if total_km <= 0:
        total_km = float(max(z["b"] for z in zones))

    # ---- draw
    fig = go.Figure()

    # Drum
    fig.add_shape(
        type="circle",
        xref="paper", yref="paper",
        x0=0.02, y0=0.35, x1=0.18, y1=0.65,
        line=dict(width=3),
        fillcolor="rgba(255,255,255,0.06)",
    )

    # Fiber baseline
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.18, y0=0.50, x1=0.98, y1=0.50,
        line=dict(width=6),
    )

    # Zones (green blocks)
    for z in sorted(zones, key=lambda r: r["a"]):
        x0 = max(0.0, min(1.0, z["a"] / total_km))
        x1 = max(0.0, min(1.0, z["b"] / total_km))
        x0p = 0.18 + 0.80 * x0
        x1p = 0.18 + 0.80 * x1

        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x0p, x1=x1p,
            y0=0.46, y1=0.54,
            fillcolor="rgba(0,180,0,0.40)",
            line=dict(width=1),
        )

        # label if visible enough
        if (x1 - x0) > 0.05:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5 * (x0p + x1p),
                y=0.58,
                text=f"Zone {z['i']}  {z['len']:.3f} km",
                showarrow=False
            )

        fig.add_trace(go.Scatter(
            x=[0.5 * (x0p + x1p)],
            y=[0.50],
            mode="markers",
            marker=dict(size=18, opacity=0),
            hovertemplate=(
                f"<b>Zone {z['i']}</b><br>"
                f"From end: {z['a']:.6f} → {z['b']:.6f} km<br>"
                f"Length: {z['len']:.6f} km"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    # Header + metrics inside popup
    st.markdown("### 🧵 Drum & Fiber – Good Zones Map")
    st.caption(f"Dataset: **{dataset_name}**  |  0 km = fiber end")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total length (km)", f"{total_km:.6f}")
    c2.metric("Total SAVE (km)", f"{total_save:.6f}" if total_save else "—")
    c3.metric("Total CUT (km)", f"{total_cut:.6f}" if total_cut else "—")

    st.plotly_chart(fig, use_container_width=True)

def render_tm_home_section():
    import os
    import json
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    from datetime import datetime

    st.subheader("📦 T&M – Pending Transfer")
    st.caption("Draws completed but not yet transferred to T&M")

    ORDERS_FILE = P.orders_csv
    DATASET_DIR = P.dataset_dir

    if not os.path.exists(ORDERS_FILE):
        st.info("No draw_orders.csv found.")
        return

    df = pd.read_csv(ORDERS_FILE, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    for col in ["Status", "Active CSV", "Done CSV", "Done Description", "T&M Moved", "T&M Moved Timestamp"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize boolean column
    df["T&M Moved"] = df["T&M Moved"].astype(str).str.lower().isin(["true", "1", "yes"])

    # Done but not moved
    pending_tm = df[(df["Status"].astype(str).str.strip() == "Done") & (~df["T&M Moved"])]

    if pending_tm.empty:
        st.success("✅ No pending T&M transfers.")
        return

    st.markdown(
        """
        <style>
        .tm-card {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(10,10,10,0.45);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            margin-bottom: 12px;
        }
        .tm-meta {
            color: rgba(255,255,255,0.80);
            font-size: 0.92rem;
            margin-top: -6px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # Persistent checklist store (per dataset CSV)
    # =========================================================
    def _tm_checklist_path():
        MAINT_FOLDER = P.maintenance_dir
        os.makedirs(MAINT_FOLDER, exist_ok=True)
        return os.path.join(MAINT_FOLDER, "tm_step_checklists.json")

    def _load_tm_checklists() -> dict:
        p = _tm_checklist_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _save_tm_checklists(d: dict):
        p = _tm_checklist_path()
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # =========================================================
    # Dataset CSV readers
    # =========================================================
    def _read_dataset_kv(csv_path: str) -> dict:
        """Reads dataset CSV (Parameter Name, Value, Units) -> {param_lower: value_str}"""
        try:
            dfx = pd.read_csv(csv_path, keep_default_na=False)
        except Exception:
            return {}

        if dfx is None or dfx.empty:
            return {}

        dfx.columns = [str(c).strip() for c in dfx.columns]
        pn_col = "Parameter Name" if "Parameter Name" in dfx.columns else None
        v_col = "Value" if "Value" in dfx.columns else None
        if not pn_col or not v_col:
            return {}

        dfx[pn_col] = dfx[pn_col].astype(str).str.strip()
        dfx[v_col] = dfx[v_col].astype(str).str.strip()

        out = {}
        for _, r in dfx.iterrows():
            k = str(r.get(pn_col, "")).strip().lower()
            v = str(r.get(v_col, "")).strip()
            if k and k not in out and v != "nan":
                out[k] = v
        return out

    def _read_dataset_df(csv_path: str):
        """Return full df for plotting."""
        try:
            dfx = pd.read_csv(csv_path, keep_default_na=False)
        except Exception:
            return None
        if dfx is None or dfx.empty:
            return None
        dfx.columns = [str(c).strip() for c in dfx.columns]
        for c in ["Parameter Name", "Value", "Units"]:
            if c not in dfx.columns:
                dfx[c] = ""
        return dfx

    def _pick(kv: dict, aliases: list) -> str:
        for a in aliases:
            a2 = str(a).strip().lower()
            if a2 in kv and str(kv[a2]).strip() and str(kv[a2]).strip().lower() != "nan":
                return str(kv[a2]).strip()
        return ""

    def _parse_zones_from_end(df_params: pd.DataFrame):
        d =  param_map(df_params)
        zones = []
        i = 1
        while True:
            ks = f"Zone {i} Start (from end)"
            ke = f"Zone {i} End (from end)"
            if ks not in d or ke not in d:
                break
            try:
                a = float(d[ks])
                b = float(d[ke])
            except Exception:
                a, b = None, None
            if a is not None and b is not None:
                if b < a:
                    a, b = b, a
                zones.append({"i": i, "a": a, "b": b, "len": (b - a)})
            i += 1
        zones.sort(key=lambda z: z["a"])
        return zones

    def _parse_marked_zone_lengths(df_params: pd.DataFrame):
        d =  param_map(df_params)
        out = []
        i = 1
        while True:
            k = f"Zone {i} Length"
            if k not in d:
                break
            try:
                L = float(d[k])
            except Exception:
                L = None
            if L is not None and L > 0:
                out.append(L)
            i += 1
        return out

    # =========================================================
    # Drum + fiber visual + checklist
    # =========================================================
    def render_tm_drum_fiber_visual_from_csv(df_params: pd.DataFrame, csv_name: str):
        """
        Drum + fiber bar:
          - 0 km = fiber end at RIGHT side
          - drum at LEFT side (near total length)
          - CUT + SAVE segments shown across full fiber
          - NO hover
          - Each segment is labeled with the step number (#) that matches the table
          - Table is a checklist (checkboxes) persisted per dataset CSV

        Uses ONLY dataset CSV.

        Priority for defining SAVE zones:
          1) Zone i Start/End (from end)
          2) Marked Zone i Length (sequential from 0)
          3) STEP SAVE segments (fallback)
        """
        d =  param_map(df_params)

        total_km = get_float_param(d, "Fiber Length Max (log)", 0.0)
        total_save = get_float_param(d, "Total Saved Length", 0.0)
        total_cut = get_float_param(d, "Total Cut Length", 0.0)

        # ---------- Get SAVE intervals (a,b,zone_index)
        zones = _parse_zones_from_end(df_params)
        save_intervals = [(z["a"], z["b"], z["i"]) for z in zones]

        if not save_intervals:
            marked = _parse_marked_zone_lengths(df_params)
            if marked:
                a = 0.0
                save_intervals = []
                for i, L in enumerate(marked, start=1):
                    L = float(L)
                    save_intervals.append((a, a + L, i))
                    a += L

        if not save_intervals:
            steps = _parse_steps(df_params)
            if steps:
                a = 0.0
                zi = 1
                save_intervals = []
                for action, L in steps:
                    L = float(L)
                    if action == "SAVE":
                        save_intervals.append((a, a + L, zi))
                        zi += 1
                    a += L

        if not save_intervals:
            st.info("No zone information found in dataset CSV for visualization.")
            return

        if total_km <= 0:
            total_km = float(max(b for a, b, _ in save_intervals))

        # Clip + sort
        clipped = []
        for a, b, zi in save_intervals:
            a = max(0.0, min(total_km, float(a)))
            b = max(0.0, min(total_km, float(b)))
            if b > a:
                clipped.append((a, b, zi))
        clipped.sort(key=lambda t: t[0])
        if not clipped:
            st.info("Zones exist but all are empty after clipping.")
            return

        # ---------- Full segmentation bounds
        bounds = [0.0, total_km]
        for a, b, _ in clipped:
            bounds.extend([a, b])
        bounds = sorted(set(bounds))

        def zone_at(mid):
            for a, b, zi in clipped:
                if a <= mid <= b:
                    return zi
            return None

        segments = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            zi = zone_at(mid)
            action = "SAVE" if zi is not None else "CUT"
            segments.append((action, a, b, zi))

        # ---------- Optional stats reader (for the table only)
        def find_zone_avg(zone_i: int, contains_name: str):
            p = df_params["Parameter Name"].astype(str)
            mask = p.str.contains(fr"^Good Zone {zone_i} Avg - ", regex=True, na=False) & p.str.contains(
                contains_name, na=False
            )
            v = pd.to_numeric(df_params.loc[mask, "Value"], errors="coerce").dropna()
            return float(v.iloc[0]) if not v.empty else None

        # =========================
        # Build table rows
        # =========================
        table_rows = []
        for step_num, (action, a, b, zi) in enumerate(segments, start=1):
            length = float(b - a)
            row = {
                "#": step_num,
                "Action": "SAVE" if action == "SAVE" else "CUT",
                "Start (km from end)": float(a),
                "End (km from end)": float(b),
                "Length (km)": float(length),
            }
            if action == "SAVE" and zi is not None:
                row["Zone"] = int(zi)
                row["Furnace Temp avg"] = find_zone_avg(zi, "Furnace Deg")
                row["Tension avg"] = find_zone_avg(zi, "Tension")
                row["Fiber Ø avg"] = find_zone_avg(zi, "Bare")
                row["Fiber inner coat Ø avg"] = find_zone_avg(zi, "Coated Inner")
                row["Fiber Outer coat Ø avg"] = find_zone_avg(zi, "Coated Outer")
            table_rows.append(row)

        df_plan = pd.DataFrame(table_rows)
        for c in ["Start (km from end)", "End (km from end)", "Length (km)"]:
            if c in df_plan.columns:
                df_plan[c] = pd.to_numeric(df_plan[c], errors="coerce").round(6)

        # =========================
        # Plot (0 km at RIGHT)
        # =========================
        fig = go.Figure()

        x_left, x_right = 0.18, 0.98
        track_y0, track_y1 = 0.46, 0.54

        # Drum
        fig.add_shape(
            type="circle",
            xref="paper", yref="paper",
            x0=0.02, y0=0.33, x1=0.16, y1=0.67,
            line=dict(width=3, color="rgba(255,255,255,0.25)"),
            fillcolor="rgba(255,255,255,0.05)",
        )
        fig.add_shape(
            type="circle",
            xref="paper", yref="paper",
            x0=0.07, y0=0.45, x1=0.11, y1=0.55,
            line=dict(width=2, color="rgba(255,255,255,0.18)"),
            fillcolor="rgba(255,255,255,0.03)",
        )

        # Track background
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x_left, x1=x_right,
            y0=track_y0, y1=track_y1,
            fillcolor="rgba(255,255,255,0.06)",
            line=dict(width=1, color="rgba(255,255,255,0.10)"),
        )

        def x_from_end(p_km: float) -> float:
            t = 1.0 - (p_km / total_km)  # 0 -> right, total -> left
            return x_left + (x_right - x_left) * t

        # Scale labels only
        fig.add_annotation(xref="paper", yref="paper", x=x_left, y=0.40, text=f"{total_km:.3f} km", showarrow=False)
        fig.add_annotation(xref="paper", yref="paper", x=x_right, y=0.40, text="0 km (fiber end)", showarrow=False)

        for t in np.linspace(0, 1, 6):
            xt = x_left + (x_right - x_left) * t
            fig.add_shape(
                type="line", xref="paper", yref="paper",
                x0=xt, x1=xt, y0=0.43, y1=0.46,
                line=dict(width=1, color="rgba(255,255,255,0.14)")
            )

        cut_fill = "rgba(230,90,90,0.35)"
        save_fill = "rgba(40,190,110,0.55)"
        edge_col = "rgba(0,0,0,0.35)"

        # Label threshold: only write # if segment is wide enough
        label_min_frac = 0.035  # 3.5% of the bar width

        for step_num, (action, a, b, zi) in enumerate(segments, start=1):
            xa = x_from_end(a)
            xb = x_from_end(b)
            x0p, x1p = (xb, xa) if xa > xb else (xa, xb)

            fill = save_fill if action == "SAVE" else cut_fill
            seg_len = float(b - a)
            frac = seg_len / total_km if total_km > 0 else 0.0

            # segment shape
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=x0p, x1=x1p,
                y0=track_y0, y1=track_y1,
                fillcolor=fill,
                line=dict(width=1, color=edge_col),
            )

            # write step number on the segment (if large enough)
            if frac >= label_min_frac:
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5 * (x0p + x1p),
                    y=0.50,
                    text=f"<b>{step_num}</b>",
                    showarrow=False,
                    font=dict(size=14, color="rgba(255,255,255,0.92)"),
                )
            else:
                # tiny segments: place number above the bar
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5 * (x0p + x1p),
                    y=0.60,
                    text=f"<b>{step_num}</b>",
                    showarrow=False,
                    font=dict(size=12, color="rgba(255,255,255,0.85)"),
                )

        fig.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=0, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # =========================
        # UI: plot + checklist table
        # =========================
        st.markdown("**🧵 Fiber Map + Cut Plan **")
        #st.caption(f"Dataset: `{csv_name}`  |  0 km = fiber end (right)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total (km)", f"{total_km:.4f}")
        m2.metric("SAVE (km)", f"{total_save:.4f}" if total_save else "—")
        m3.metric("CUT (km)", f"{total_cut:.4f}" if total_cut else "—")

        left, right = st.columns([2.2, 1.3], vertical_alignment="top")

        with left:
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### ✅ Cut / Save Checklist")

            # ----------------------------
            # Persistent checklist per CSV
            # ----------------------------
            all_lists = _load_tm_checklists()
            csv_key = str(csv_name)

            if csv_key not in all_lists:
                all_lists[csv_key] = {}

            # Ensure every step exists in store
            for n in df_plan["#"].astype(int).tolist():
                sn = str(int(n))
                if sn not in all_lists[csv_key]:
                    all_lists[csv_key][sn] = False

            # Build editable view
            df_chk = df_plan.copy()
            df_chk["Done"] = df_chk["#"].astype(int).astype(str).map(all_lists[csv_key]).fillna(False)

            # Put Done first
            front_cols = ["Done", "#", "Action", "Length (km)"]
            rest_cols = [c for c in df_chk.columns if c not in front_cols]
            show_cols = [c for c in front_cols if c in df_chk.columns] + rest_cols
            df_chk = df_chk[show_cols].copy()

            # Progress
            # Placeholders so progress can reflect the edited table in the same run
            prog_ph = st.empty()
            cap_ph = st.empty()

            edited = st.data_editor(
                df_chk,
                use_container_width=True,
                hide_index=True,
                height=260,
                key=f"tm_steps_editor__{csv_key}",
                column_config={
                    "Done": st.column_config.CheckboxColumn(
                        "Done",
                        help="Mark each Cut/Save step as completed",
                        default=False,
                        width="small",
                    ),
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "Length (km)": st.column_config.NumberColumn("Length (km)", format="%.6f"),
                    "Start (km from end)": st.column_config.NumberColumn("Start", format="%.6f"),
                    "End (km from end)": st.column_config.NumberColumn("End", format="%.6f"),
                },
                disabled=[c for c in df_chk.columns if c != "Done"],  # only checkbox editable
            )

            # Compute progress from the edited values (THIS is the key)
            try:
                done_count = int(pd.Series(edited["Done"]).astype(bool).sum())
                total_count = int(len(edited))
            except Exception:
                done_count = int(df_chk["Done"].astype(bool).sum())
                total_count = int(len(df_chk))

            prog_ph.progress(done_count / total_count if total_count else 0.0)
            cap_ph.caption(f"Progress: **{done_count} / {total_count}** steps done")

            # Persist changes
            try:
                for _, r in edited.iterrows():
                    step_num = str(int(r["#"]))
                    all_lists[csv_key][step_num] = bool(r["Done"])
                _save_tm_checklists(all_lists)
            except Exception:
                pass

            a1, a2 = st.columns(2)
            with a1:
                if st.button("✅ Mark all done", key=f"tm_steps_all_done__{csv_key}", use_container_width=True):
                    for k in list(all_lists[csv_key].keys()):
                        all_lists[csv_key][k] = True
                    _save_tm_checklists(all_lists)
                    st.rerun()

            with a2:
                if st.button("↩️ Reset", key=f"tm_steps_reset__{csv_key}", use_container_width=True):
                    for k in list(all_lists[csv_key].keys()):
                        all_lists[csv_key][k] = False
                    _save_tm_checklists(all_lists)
                    st.rerun()

            st.caption("Checklist is saved per dataset CSV (persists after refresh).")

    # =========================================================
    # Cards render
    # =========================================================
    for idx, row in pending_tm.iterrows():
        draw_id = row.get("Preform Name") or row.get("Preform Number") or f"Row {idx}"
        done_csv = str(row.get("Done CSV") or "").strip()
        active_csv = str(row.get("Active CSV") or "").strip()

        csv_name = done_csv if done_csv else active_csv
        csv_path = dataset_csv_path(csv_name) if csv_name else ""

        done_desc = str(row.get("Done Description") or "").strip()

        kv = _read_dataset_kv(csv_path) if (csv_path and os.path.exists(csv_path)) else {}

        project = _pick(
            kv,
            ["Order__Fiber Project ", "Project Name", "Fiber Project", "Fiber name and number", "Fiber Name and Number"]
        ) or str(row.get("Project Name") or "").strip()

        preform = _pick(
            kv,
            ["Order__Preform Number", "Preform Name", "Preform", "Draw Name"]
        ) or str(row.get("Preform Name") or row.get("Preform Number") or "").strip()

        fiber = _pick(kv, ["Order__Fiber Geometry Type"]) or str(row.get("Fiber Type") or row.get("Fiber Project") or "").strip()
        drum = _pick(kv, ["Drum", "Selected Drum"])

        project_disp = project if project else "—"
        preform_disp = preform if preform else "—"
        fiber_disp = fiber if fiber else "—"
        drum_disp = drum if drum else "—"
        csv_disp = csv_name if csv_name else "—"

        st.markdown("<div class='tm-card'>", unsafe_allow_html=True)
        #st.markdown(f"**🧾 Draw:** {draw_id}", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='tm-meta'>
                Project: <b>{project_disp}</b>
                &nbsp; | &nbsp; Preform: <b>{preform_disp}</b>
                &nbsp; | &nbsp; Fiber type: <b>{fiber_disp}</b>
                &nbsp; | &nbsp; Drum: <b>{drum_disp}</b>
                <br/>
                CSV: <code>{csv_disp}</code>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Mini checklist status in header (if exists)
        if csv_name:
            all_lists = _load_tm_checklists()
            ck = all_lists.get(str(csv_name), {})
            if ck:
                done_n = sum(1 for v in ck.values() if v)
                tot_n = len(ck)
                st.caption(f"Checklist: **{done_n}/{tot_n}** steps done")

        if done_desc:
            st.caption(f"Done notes: {done_desc}")

        # Visual + checklist (inside each card)
        if csv_name and os.path.exists(csv_path):
            df_params = _read_dataset_df(csv_path)
            if df_params is not None:
                render_tm_drum_fiber_visual_from_csv(df_params, csv_name)

        b1, b2, b3 = st.columns([1.2, 1.2, 2.2])

        with b1:
            if csv_name and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "📄 CSV",
                        data=f,
                        file_name=csv_name,
                        mime="text/csv",
                        key=f"tm_dl_{idx}",
                        use_container_width=True
                    )
            else:
                st.button("📄 CSV", key=f"tm_csv_missing_{idx}", disabled=True, use_container_width=True)

        with b2:
            if st.button("📤 Mark Moved", key=f"tm_move_{idx}", use_container_width=True):
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.loc[idx, "T&M Moved"] = True
                df.loc[idx, "T&M Moved Timestamp"] = now_str
                df.to_csv(ORDERS_FILE, index=False)
                st.success(f"T&M marked as moved for {csv_name if csv_name else draw_id}")
                st.rerun()

        with b3:
            if csv_name and csv_path:
                st.caption(f"Path: `{csv_path}`")

        st.markdown("</div>", unsafe_allow_html=True)

def render_tm_cut_plan_visual(df_params: pd.DataFrame, dataset_name: str):
    """
    Drum + fiber + step segments.
    Uses STEP rows you already save to dataset CSV.
    """
    steps = _parse_steps(df_params)
    if not steps:
        st.info("No CUT/SAVE STEP data found yet in the dataset CSV. Mark zones in Dashboard and save.")
        return

    # totals (already in km in your CSV)
    try:
        total_len = float( get_value(df_params, "Fiber Length Max (log)", 0.0) or 0.0)
    except Exception:
        total_len = 0.0
    try:
        total_save = float( get_value(df_params, "Total Saved Length", 0.0) or 0.0)
    except Exception:
        total_save = 0.0
    try:
        total_cut = float( get_value(df_params, "Total Cut Length", 0.0) or 0.0)
    except Exception:
        total_cut = 0.0

    # Key averages (edit this list to match your log column names)
    wanted = [
        "Furnace Temperature",
        "Furnace Temp",
        "Diameter",
        "Fiber Diameter",
        "Primary Coating Diameter",
        "Secondary Coating Diameter",
        "Tension",
        "Speed",
    ]
    avg_map = _find_zone_avg_values(df_params, wanted)

    # Build segment coordinates along fiber (0 = end)
    total_steps = sum(L for _, L in steps)
    if total_steps <= 0:
        st.warning("STEP lengths sum to 0. Cannot draw cut plan.")
        return

    segs = []
    cum = 0.0
    for i, (action, L) in enumerate(steps, start=1):
        x0 = cum / total_steps
        x1 = (cum + L) / total_steps
        segs.append((i, action, x0, x1, L, cum, cum + L))  # include start/end (from end) for hover only
        cum += L

    # Figure
    fig = go.Figure()

    # Drum circle (left)
    fig.add_shape(
        type="circle",
        xref="paper", yref="paper",
        x0=0.02, y0=0.35, x1=0.18, y1=0.65,
        line=dict(width=3),
        fillcolor="rgba(255,255,255,0.06)",
    )

    # Fiber baseline
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.18, y0=0.50, x1=0.98, y1=0.50,
        line=dict(width=4),
    )

    # Segments blocks on the line
    for (idx, action, x0, x1, L, a_km, b_km) in segs:
        fill = "rgba(0,180,0,0.35)" if action == "SAVE" else "rgba(220,0,0,0.25)"
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=0.18 + 0.80 * x0,
            x1=0.18 + 0.80 * x1,
            y0=0.47, y1=0.53,
            fillcolor=fill,
            line=dict(width=1),
        )

        # Label only if segment is long enough visually
        if (x1 - x0) > 0.06:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.18 + 0.80 * (0.5 * (x0 + x1)),
                y=0.57,
                text=f"{action} {L:.3f} km",
                showarrow=False
            )

        # Hover point
        hover_lines = [
            f"<b>STEP {idx}</b>",
            f"Action: {action}",
            f"Length: {L:.6f} km",
            f"From end: {a_km:.6f} → {b_km:.6f} km",
        ]
        # Add key averages into hover (global averages across zones)
        for k, v in avg_map.items():
            hover_lines.append(f"{k} (avg): {v:.4g}")

        fig.add_trace(go.Scatter(
            x=[0.18 + 0.80 * (0.5 * (x0 + x1))],
            y=[0.50],
            mode="markers",
            marker=dict(size=18, opacity=0),
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>"
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    # UI block
    st.markdown("### 🧵 Fiber Cut/Save Plan (0 km = fiber end)")
    st.caption(f"Dataset: **{dataset_name}**  |  Hover on the line for step details.")

    left, right = st.columns([2.3, 1])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.metric("Total length (km)", f"{total_len:.6f}" if total_len else "—")
        st.metric("Total SAVE (km)", f"{total_save:.6f}" if total_save else "—")
        st.metric("Total CUT (km)", f"{total_cut:.6f}" if total_cut else "—")

        # Show key averages as small list
        if avg_map:
            st.markdown("**Key averages (good zones)**")
            for k, v in avg_map.items():
                st.write(f"- {k}: `{v:.4g}`")

def _extract_good_zone_summary(df_params: pd.DataFrame) -> pd.DataFrame:
    """
    Per-zone summary using your real CSV keys.

    Includes:
      - PF Start/End:   Good Zone i Min/Max - Pf Process Position
      - Zone Length:   (Good Zone i Max - Fibre Length) - (Good Zone i Min - Fibre Length)
      - Furnace:       Good Zone i Avg - Furnace DegC Actual
      - Tension:       Good Zone i Avg - Tension N
      - Bare Ø:        Good Zone i Avg - Bare Fibre Diameter
      - Coat 1 Ø:      Good Zone i Avg - Coated Inner Diameter
      - Coat 2 Ø:      Good Zone i Avg - Coated Outer Diameter

    Reads ONLY: Parameter Name, Value, Units
    """
    import pandas as pd
    import re

    if df_params is None or df_params.empty:
        return pd.DataFrame()

    dfp = df_params.copy()
    for c in ["Parameter Name", "Value", "Units"]:
        if c not in dfp.columns:
            dfp[c] = ""

    dfp["Parameter Name"] = dfp["Parameter Name"].astype(str).str.strip()
    dfp["Value"] = dfp["Value"].astype(str).str.strip()
    dfp["Units"] = dfp["Units"].astype(str).str.strip()

    # name -> (value, units)
    kv = {}
    for _, r in dfp.iterrows():
        k = str(r["Parameter Name"]).strip()
        if k and k not in kv:
            kv[k] = (str(r["Value"]).strip(), str(r["Units"]).strip())

    def _num(x):
        try:
            x = str(x).strip()
            if not x or x.lower() == "nan":
                return None
            return float(x)
        except Exception:
            return None

    def _get(key: str):
        v, u = kv.get(key, ("", ""))
        v = "" if str(v).lower() == "nan" else str(v).strip()
        u = "" if str(u).lower() == "nan" else str(u).strip()
        return v, u

    # detect zone indices by scanning Good Zone i Avg - ...
    zones = set()
    rz = re.compile(r"^Good Zone\s+(\d+)\s+Avg\s+-\s+", re.IGNORECASE)
    for k in kv.keys():
        m = rz.match(k)
        if m:
            try:
                zones.add(int(m.group(1)))
            except Exception:
                pass

    if not zones:
        return pd.DataFrame()

    rows = []
    for zi in sorted(zones):
        # ---- PF min/max (what you asked for)
        pf_min_v, pf_u1 = _get(f"Good Zone {zi} Min - Pf Process Position")
        pf_max_v, pf_u2 = _get(f"Good Zone {zi} Max - Pf Process Position")
        pf_u = pf_u2 or pf_u1

        # ---- zone length from fibre length min/max (km in your files)
        fmin_v, f_u1 = _get(f"Good Zone {zi} Min - Fibre Length")
        fmax_v, f_u2 = _get(f"Good Zone {zi} Max - Fibre Length")
        fmin = _num(fmin_v)
        fmax = _num(fmax_v)
        zlen = (fmax - fmin) if (fmin is not None and fmax is not None and fmax >= fmin) else None
        zlen_u = f_u2 or f_u1

        # ---- avg process values
        furnace_v, furnace_u = _get(f"Good Zone {zi} Avg - Furnace DegC Actual")
        tension_v, tension_u = _get(f"Good Zone {zi} Avg - Tension N")
        bare_v, bare_u       = _get(f"Good Zone {zi} Avg - Bare Fibre Diameter")
        c1_v, c1_u           = _get(f"Good Zone {zi} Avg - Coated Inner Diameter")
        c2_v, c2_u           = _get(f"Good Zone {zi} Avg - Coated Outer Diameter")

        rows.append({
            "Zone": zi,
            "PF Start": _num(pf_min_v),
            "PF End": _num(pf_max_v),
            "PF Unit": pf_u,

            "Zone Length": zlen,
            "Zone Length Unit": zlen_u,

            "Furnace": _num(furnace_v),
            "Furnace Unit": furnace_u,

            "Tension": _num(tension_v),
            "Tension Unit": tension_u,

            "Bare Ø": _num(bare_v),
            "Bare Unit": bare_u,

            "Coat 1 Ø": _num(c1_v),
            "Coat 1 Unit": c1_u,

            "Coat 2 Ø": _num(c2_v),
            "Coat 2 Unit": c2_u,
        })

    dfz = pd.DataFrame(rows)

    # put units into headers if available (and drop unit columns)
    def _first_unit(col):
        vals = [x for x in dfz[col].astype(str).tolist() if x and x.lower() != "nan"]
        return vals[0] if vals else ""

    pf_u  = _first_unit("PF Unit")
    len_u = _first_unit("Zone Length Unit")
    fur_u = _first_unit("Furnace Unit")
    ten_u = _first_unit("Tension Unit")
    bare_u = _first_unit("Bare Unit")
    c1_u = _first_unit("Coat 1 Unit")
    c2_u = _first_unit("Coat 2 Unit")

    ren = {}
    if pf_u:
        ren["PF Start"] = f"PF Start ({pf_u})"
        ren["PF End"]   = f"PF End ({pf_u})"
    if len_u:
        ren["Zone Length"] = f"Zone Length ({len_u})"
    if fur_u:
        ren["Furnace"] = f"Furnace ({fur_u})"
    if ten_u:
        ren["Tension"] = f"Tension ({ten_u})"
    if bare_u:
        ren["Bare Ø"] = f"Bare Ø ({bare_u})"
    if c1_u:
        ren["Coat 1 Ø"] = f"Coat 1 Ø ({c1_u})"
    if c2_u:
        ren["Coat 2 Ø"] = f"Coat 2 Ø ({c2_u})"

    dfz = dfz.rename(columns=ren)

    drop_cols = [
        "PF Unit", "Zone Length Unit", "Furnace Unit", "Tension Unit",
        "Bare Unit", "Coat 1 Unit", "Coat 2 Unit"
    ]
    dfz = dfz.drop(columns=[c for c in drop_cols if c in dfz.columns], errors="ignore")

    # rounding
    for c in dfz.columns:
        if c.startswith("Zone Length"):
            dfz[c] = pd.to_numeric(dfz[c], errors="coerce").round(6)
        elif c.startswith("PF "):
            dfz[c] = pd.to_numeric(dfz[c], errors="coerce").round(3)
        elif "Ø" in c:
            dfz[c] = pd.to_numeric(dfz[c], errors="coerce").round(3)
        elif c.startswith("Furnace") or c.startswith("Tension"):
            dfz[c] = pd.to_numeric(dfz[c], errors="coerce").round(2)

    return dfz

def render_done_home_section():
    import os
    import pandas as pd
    import streamlit as st
    import datetime as dt

    st.subheader("✅ DONE – Recent Draws (last 4 days)")
    st.caption("Summarizes finished draws from the dataset CSV. After 4 days, they auto-move to T&M.")

    ORDERS_FILE = P.orders_csv
    DATASET_DIR = P.dataset_dir
    AUTO_MOVE_DAYS = 4

    if not os.path.exists(ORDERS_FILE):
        st.info("No draw_orders.csv found.")
        return

    df = pd.read_csv(ORDERS_FILE, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Ensure columns exist
    needed_cols = [
        "Status", "Active CSV", "Done CSV", "Done Description",
        "T&M Moved", "T&M Moved Timestamp",
        "Done Timestamp"
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = ""

    # Normalize boolean
    df["T&M Moved"] = df["T&M Moved"].astype(str).str.lower().isin(["true", "1", "yes"])

    # -----------------------------
    # Helpers
    # -----------------------------
    def _read_dataset_df(csv_path: str):
        try:
            dfx = pd.read_csv(csv_path, keep_default_na=False)
        except Exception:
            return None
        if dfx is None or dfx.empty:
            return None
        dfx.columns = [str(c).strip() for c in dfx.columns]
        for c in ["Parameter Name", "Value", "Units"]:
            if c not in dfx.columns:
                dfx[c] = ""
        return dfx

    def _read_dataset_kv(csv_path: str) -> dict:
        """
        Reads dataset CSV and returns a dict:
          key = normalized Parameter Name (lower, stripped)
          val = Value (string)
        NOTE: keeps first occurrence only (older files may have duplicates).
        """
        dfx = _read_dataset_df(csv_path)
        if dfx is None:
            return {}
        dfx["Parameter Name"] = dfx["Parameter Name"].astype(str).str.strip()
        dfx["Value"] = dfx["Value"].astype(str).str.strip()

        out = {}
        for _, r in dfx.iterrows():
            k = str(r.get("Parameter Name", "")).strip().lower()
            v = str(r.get("Value", "")).strip()
            if k and k not in out and v.lower() != "nan":
                out[k] = v
        return out

    def _pick(kv: dict, aliases: list) -> str:
        """
        Tries aliases in order. Aliases must be exact strings as stored in CSV.
        We normalize to lowercase to match kv keys.
        """
        for a in aliases:
            k = str(a).strip().lower()
            if k in kv and str(kv[k]).strip() and str(kv[k]).strip().lower() != "nan":
                return str(kv[k]).strip()
        return ""

    def _to_dt(s: str):
        s = str(s or "").strip()
        if not s:
            return None
        try:
            x = pd.to_datetime(s, errors="coerce")
            if pd.isna(x):
                return None
            return x.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None

    def _infer_done_dt(row, kv: dict):
        """
        Priority:
          1) draw_orders.csv "Done Timestamp" (if exists)
          2) dataset CSV "Order__Draw Date" (new order scheme)
          3) dataset CSV "Draw Date" (legacy)
          4) dataset CSV "Process__Process Setup Timestamp" / "Process Setup Timestamp"
          5) None
        """
        dt1 = _to_dt(row.get("Done Timestamp", ""))
        if dt1:
            return dt1

        dt2 = _to_dt(_pick(kv, ["Order__Draw Date"]))
        if dt2:
            return dt2

        dt3 = _to_dt(_pick(kv, ["Draw Date"]))
        if dt3:
            return dt3

        dt4 = _to_dt(_pick(kv, ["Process__Process Setup Timestamp", "Process Setup Timestamp"]))
        if dt4:
            return dt4

        return None

    def fmt_float(x, nd=2):
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return "—"
            return f"{float(s):.{nd}f}"
        except Exception:
            return "—"

    def fmt_int(x):
        try:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return "—"
            return str(int(float(s)))
        except Exception:
            return "—"

    # -----------------------------
    # Filter "Done and not moved"
    # -----------------------------
    done_not_moved = df[
        (df["Status"].astype(str).str.strip().str.lower() == "done")
        & (~df["T&M Moved"])
    ].copy()

    if done_not_moved.empty:
        st.success("✅ No recent DONE draws waiting here (everything is already moved to T&M).")
        return

    # -----------------------------
    # Auto-move after 4 days
    # -----------------------------
    now = dt.datetime.now()
    changed = False
    recent_rows = []

    for idx, row in done_not_moved.iterrows():
        done_csv = str(row.get("Done CSV") or "").strip()
        active_csv = str(row.get("Active CSV") or "").strip()
        csv_name = done_csv if done_csv else active_csv

        csv_path = dataset_csv_path(csv_name) if csv_name else ""
        kv = _read_dataset_kv(csv_path) if (csv_name and os.path.exists(csv_path)) else {}

        done_dt = _infer_done_dt(row, kv)

        if done_dt and not str(row.get("Done Timestamp", "")).strip():
            df.loc[idx, "Done Timestamp"] = done_dt.strftime("%Y-%m-%d %H:%M:%S")
            changed = True

        if done_dt:
            age_days = (now - done_dt).total_seconds() / 86400.0
        else:
            age_days = 0.0

        if age_days >= AUTO_MOVE_DAYS:
            df.loc[idx, "T&M Moved"] = True
            if not str(df.loc[idx, "T&M Moved Timestamp"]).strip():
                df.loc[idx, "T&M Moved Timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
            changed = True
        else:
            recent_rows.append((idx, row, kv, csv_name, csv_path, done_dt, age_days))

    if changed:
        df.to_csv(ORDERS_FILE, index=False)

    if not recent_rows:
        st.success("✅ No recent DONE draws (older than 4 days were auto-moved to T&M).")
        return

    # -----------------------------
    # UI styling (cards)
    # -----------------------------
    st.markdown(
        """
        <style>
        .done-card {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(10,10,10,0.45);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            margin-bottom: 12px;
        }
        .done-meta {
            color: rgba(255,255,255,0.82);
            font-size: 0.92rem;
            margin-top: -6px;
            margin-bottom: 10px;
        }
        .done-pill {
            display:inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.06);
            margin-left: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Cards
    # -----------------------------
    for (idx, row, kv, csv_name, csv_path, done_dt, age_days) in recent_rows:
        done_desc = str(row.get("Done Description") or "").strip()

        # ✅ NEW: Prefer Order__/Process__ names first, then legacy aliases
        project = _pick(kv, [
            "Order__Fiber Project",
            "Fiber Project",
            "Project",
            "Project Name",
            "Fiber name and number",
            "Fiber Name and Number",
        ]) or str(row.get("Project Name") or "").strip()

        preform = _pick(kv, [
            "Order__Preform Number",
            "Preform Number",
            "Preform Name",
            "Preform",
        ]) or str(row.get("Preform Name") or row.get("Preform Number") or "").strip()

        fiber = _pick(kv, [
            "Order__Fiber Geometry Type",
            "Fiber Geometry Type",
            "Fiber Type",
        ]) or str(row.get("Fiber Type") or row.get("Fiber Project") or "").strip()

        # Drum: prefer dashboard group, then process setup, then legacy
        drum = _pick(kv, [
            "Drum | Selected",
            "Process__Selected Drum",
            "Selected Drum",
            "Drum",
        ])

        # Lengths: new dashboard name first
        total_km = _pick(kv, [
            "Fiber Length | End (log end)",
            "Fiber Length End (log end)",
            "Fibre Length End (log end)",
        ])

        save_km = _pick(kv, [
            "Total Saved Length",
            "Total Saved Length (km)",
        ])

        cut_km = _pick(kv, [
            "Total Cut Length",
            "Total Cut Length (km)",
        ])

        zones_n = _pick(kv, [
            "Good Zones Count",
            "Order__Good Zones Count (required length zones)",
            "Good Zones Count (required length zones)",
        ])

        done_str = done_dt.strftime("%Y-%m-%d %H:%M:%S") if done_dt else "—"
        age_str = f"{age_days:.1f} days" if done_dt else "—"

        st.markdown("<div class='done-card'>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='done-meta'>
                Project: <b>{project or "—"}</b>
                &nbsp; | &nbsp; Preform: <b>{preform or "—"}</b>
                &nbsp; | &nbsp; Fiber type: <b>{fiber or "—"}</b>
                &nbsp; | &nbsp; Drum: <b>{drum or "—"}</b>
                <span class="done-pill">Done: {done_str}</span>
                <span class="done-pill">Age: {age_str}</span>
                <br/>
                CSV: <code>{csv_name or "—"}</code>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total (km)", fmt_float(total_km, 2))
        c2.metric("SAVE (km)", fmt_float(save_km, 2))
        c3.metric("CUT (km)", fmt_float(cut_km, 2))
        c4.metric("Zones", fmt_int(zones_n))

        if done_desc:
            st.caption(f"Done notes: {done_desc}")

        b1, b2 = st.columns([1.1, 2.9])

        with b1:
            if csv_name and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "📄 Download CSV",
                        data=f,
                        file_name=csv_name,
                        mime="text/csv",
                        key=f"done_dl_{idx}",
                        use_container_width=True
                    )
            else:
                st.button("📄 Download CSV", disabled=True, use_container_width=True, key=f"done_missing_{idx}")

        with b2:
            if done_dt:
                st.caption(
                    f"Auto-move to T&M after **{AUTO_MOVE_DAYS} days** "
                    f"(this one moves in ~{max(0.0, AUTO_MOVE_DAYS - age_days):.1f} days)."
                )
            else:
                st.caption(f"Auto-move to T&M after **{AUTO_MOVE_DAYS} days** (done time unknown).")

        st.markdown("</div>", unsafe_allow_html=True)

def is_pm_draw_from_dataset_csv(df_params: pd.DataFrame) -> bool:
    """
    PM detection is based ONLY on the explicit boolean flag:
      Parameter Name = 'PM Iris System'
      Value = 1 / True / 'true'

    This is the authoritative source.
    """

    try:
        row = df_params.loc[
            df_params["Parameter Name"].astype(str).str.strip().str.lower()
            == "pm iris system".lower()
        ]

        if row.empty:
            return False

        val = row["Value"].iloc[0]

        # Accept numeric or string truthy values
        if isinstance(val, (int, float)):
            return int(val) == 1

        val_str = str(val).strip().lower()
        return val_str in {"1", "true", "yes", "y"}

    except Exception:
        return False

def show_sap_status_banner(is_pm: bool, status: str, details: str = ""):
    """
    status: "updated" | "not_pm" | "skipped" | "failed"
    """
    if status == "updated":
        st.success(f"🧪 SAP rods inventory updated (PM detected).")
        if details:
            st.caption(details)

    elif status == "not_pm":
        st.info("ℹ️ SAP rods inventory not updated (not a PM draw).")
        if details:
            st.caption(details)

    elif status == "skipped":
        st.warning("⚠️ SAP rods inventory update skipped.")
        if details:
            st.caption(details)

    else:  # "failed"
        st.warning("⚠️ SAP rods inventory update failed.")
        if details:
            st.caption(details)

def ensure_sap_inventory_file():
    if os.path.exists(SAP_INVENTORY_FILE):
        return

    df = pd.DataFrame([{
        "Item": "SAP Rods Set",
        "Count": 0,
        "Units": "sets",
        "Last Updated": "",
        "Notes": ""
    }])
    df.to_csv(SAP_INVENTORY_FILE, index=False)

def decrement_sap_rods_set_by_one(source_draw: str, when_str: str = None):
    """
    Returns: (ok: bool, msg: str)
    """
    ensure_sap_inventory_file()
    when_str = when_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        inv = pd.read_csv(SAP_INVENTORY_FILE)
    except Exception as e:
        return False, f"Failed reading {SAP_INVENTORY_FILE}: {e}"

    if inv.empty or "Item" not in inv.columns:
        return False, f"{SAP_INVENTORY_FILE} format is invalid."

    # Find (or create) the row for SAP Rods Set
    m = inv["Item"].astype(str).str.strip().str.lower() == "sap rods set"
    if not m.any():
        inv = pd.concat([inv, pd.DataFrame([{
            "Item": "SAP Rods Set",
            "Count": 0,
            "Units": "sets",
            "Last Updated": "",
            "Notes": ""
        }])], ignore_index=True)
        m = inv["Item"].astype(str).str.strip().str.lower() == "sap rods set"

    idx = inv.index[m][0]

    # Parse count safely
    try:
        current = int(float(inv.loc[idx, "Count"]))
    except Exception:
        current = 0

    if current <= 0:
        # Don’t go negative; still log the attempt in Notes
        inv.loc[idx, "Last Updated"] = when_str
        prev_notes = safe_str(inv.loc[idx, "Notes"])
        add = f"[{when_str}] Tried decrement (PM draw {source_draw}) but Count was {current}."
        inv.loc[idx, "Notes"] = (prev_notes + "\n" + add).strip() if prev_notes else add
        inv.to_csv(SAP_INVENTORY_FILE, index=False)
        return False, f"SAP inventory NOT decremented (Count={current}). Please refill/update inventory."

    # Decrement
    inv.loc[idx, "Count"] = current - 1
    inv.loc[idx, "Last Updated"] = when_str

    prev_notes = safe_str(inv.loc[idx, "Notes"])
    add = f"[{when_str}] -1 set (PM draw {source_draw}). New Count={current-1}."
    inv.loc[idx, "Notes"] = (prev_notes + "\n" + add).strip() if prev_notes else add

    inv.to_csv(SAP_INVENTORY_FILE, index=False)
    return True, f"SAP Rods Set inventory updated: {current} → {current-1}"

def mark_draw_order_failed_by_dataset_csv(dataset_csv_filename: str, failed_desc: str, preform_len_after_cm: float):
    """
    Sets Status=Failed, writes failure description + preform left, timestamp.
    Matches rows by Active CSV / Done CSV (same logic as Done).
    """
    if not os.path.exists(ORDERS_FILE):
        return False, f"{ORDERS_FILE} not found (couldn't mark order failed)."

    try:
        orders = pd.read_csv(ORDERS_FILE, keep_default_na=False)
    except Exception as e:
        return False, f"Failed reading {ORDERS_FILE}: {e}"

    orders.columns = [str(c).replace("\ufeff", "").strip() for c in orders.columns]

    # Ensure columns exist
    for col, default in {
        "Status": "Pending",
        "Active CSV": "",
        "Done CSV": "",
        "Done Description": "",
        "Done Timestamp": "",
        "Failed CSV": "",
        "Failed Description": "",
        "Failed Timestamp": "",
        "Preform Length After Draw (cm)": "",
        "Next Planned Draw Date": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }.items():
        if col not in orders.columns:
            orders[col] = default

    def norm_col(series):
        return (
            series.astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
        )

    target = _norm_str(dataset_csv_filename)
    target_alts = _alt_names(target)

    active_norm = norm_col(orders["Active CSV"])
    done_norm = norm_col(orders["Done CSV"])

    match = pd.Series([False] * len(orders))
    for t in target_alts:
        match = match | (active_norm == t) | (done_norm == t)

    if not match.any():
        for t in target_alts:
            match = match | active_norm.str.endswith(t, na=False) | done_norm.str.endswith(t, na=False)

    if not match.any():
        for t in target_alts:
            contains = active_norm.str.contains(re.escape(t), na=False) | done_norm.str.contains(re.escape(t), na=False)
            if contains.sum() == 1:
                match = contains
                break

    if not match.any():
        sample_active = active_norm.dropna().unique()[:12].tolist()
        return False, (
            f"No matching row found in draw_orders.csv for '{dataset_csv_filename}' "
            f"(matched against Active CSV / Done CSV).\n"
            f"Sample Active CSV values: {sample_active}"
        )

    if match.sum() > 1:
        return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    orders.loc[match, "Status"] = "Failed"
    orders.loc[match, "Failed CSV"] = os.path.basename(dataset_csv_filename)
    orders.loc[match, "Failed Description"] = str(failed_desc).strip()
    orders.loc[match, "Failed Timestamp"] = now_str
    orders.loc[match, "Preform Length After Draw (cm)"] = float(preform_len_after_cm)

    orders.to_csv(ORDERS_FILE, index=False)
    return True, "Order marked as FAILED."

def reset_failed_order_to_beginning_and_schedule(dataset_csv_filename: str):
    """
    'Start from beginning of the draw':
    - Status -> Pending
    - Clear Active/Done pointers and Done fields
    - Clear failure fields (optional, but cleaner for a new attempt)
    - Write Next Planned Draw Date (tomorrow or Sunday after Thu)
    """
    if not os.path.exists(ORDERS_FILE):
        return False, f"{ORDERS_FILE} not found."

    try:
        orders = pd.read_csv(ORDERS_FILE, keep_default_na=False)
    except Exception as e:
        return False, f"Failed reading {ORDERS_FILE}: {e}"

    orders.columns = [str(c).replace("\ufeff", "").strip() for c in orders.columns]

    for col, default in {
        "Status": "Pending",
        "Active CSV": "",
        "Done CSV": "",
        "Done Description": "",
        "Done Timestamp": "",
        "Failed CSV": "",
        "Failed Description": "",
        "Failed Timestamp": "",
        "Next Planned Draw Date": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }.items():
        if col not in orders.columns:
            orders[col] = default

    def norm_col(series):
        return (
            series.astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
        )

    target = _norm_str(dataset_csv_filename)
    target_alts = _alt_names(target)

    active_norm = norm_col(orders["Active CSV"])
    done_norm   = norm_col(orders["Done CSV"])
    fail_norm   = norm_col(orders["Failed CSV"]) if "Failed CSV" in orders.columns else pd.Series([""] * len(orders))

    match = pd.Series([False] * len(orders))
    for t in target_alts:
        match = match | (active_norm == t) | (done_norm == t) | (fail_norm == t)

    if not match.any():
        for t in target_alts:
            match = match | active_norm.str.endswith(t, na=False) | done_norm.str.endswith(t, na=False) | fail_norm.str.endswith(t, na=False)

    if not match.any():
        return False, f"No matching row found for '{dataset_csv_filename}'."

    if match.sum() > 1:
        return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

    next_date = compute_next_planned_draw_date(datetime.now())

    # ✅ Reset to "start from beginning"
    if str(schedule_date).strip():
        orders.loc[match, "Status"] = "Scheduled"
    else:
        orders.loc[match, "Status"] = "Pending"
    orders.loc[match, "Next Planned Draw Date"] = next_date

    orders.loc[match, "Active CSV"] = ""
    orders.loc[match, "Done CSV"] = ""
    orders.loc[match, "Done Description"] = ""
    orders.loc[match, "Done Timestamp"] = ""

    orders.loc[match, "Failed CSV"] = ""
    orders.loc[match, "Failed Description"] = ""
    orders.loc[match, "Failed Timestamp"] = ""

    # Optional: reset T&M moved flags for a fresh attempt
    orders.loc[match, "T&M Moved"] = False
    orders.loc[match, "T&M Moved Timestamp"] = ""

    orders.to_csv(ORDERS_FILE, index=False)
    return True, f"Reset to Pending and scheduled Next Planned Draw Date = {next_date}."

# ------------------ Home Tab ------------------
# ------------------ Home Tab ------------------
if tab_selection == "🏠 Home":
    import os
    import datetime as dt
    import pandas as pd
    import streamlit as st

    st.title("️ Tower Management Software")

    # =========================================================
    # 🎨 CSS (yours + small dialog polish)
    # =========================================================
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .css-1aumxhk {{ background-color: rgba(20, 20, 20, 0.90) !important; }}
        div[data-testid="stDialog"] {{
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # ❌ FAILED → AUTO BACK TO PENDING AFTER 4 DAYS
    # =========================================================
    ORDERS_FILE = P.orders_csv

    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        if STATUS_COL not in df.columns:
            df[STATUS_COL] = "Pending"
        if STATUS_UPDATED_COL not in df.columns:
            df[STATUS_UPDATED_COL] = ""
        if FAILED_REASON_COL not in df.columns:
            df[FAILED_REASON_COL] = ""
        return df

    def auto_move_failed_to_pending(days: int = 4):
        if not os.path.exists(ORDERS_FILE):
            return

        try:
            df = pd.read_csv(ORDERS_FILE)
        except Exception:
            return

        df = _ensure_cols(df)
        if df.empty:
            return

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days)
        changed = False

        for i in range(len(df)):
            if str(df.at[i, STATUS_COL]).strip().lower() != "failed":
                continue

            t = parse_dt_safe(df.at[i, STATUS_UPDATED_COL])

            # stamp missing timestamps so the 4-day timer works
            if t is None:
                df.at[i, STATUS_UPDATED_COL] = now_str()
                changed = True
                continue

            if t < cutoff:
                df.at[i, STATUS_COL] = "Pending"
                df.at[i, STATUS_UPDATED_COL] = now_str()
                changed = True

        if changed:
            df.to_csv(ORDERS_FILE, index=False)


    # =========================================================
    # 🚨 CRITICAL OPEN FAULTS (Home indicator)
    # - Reads maintenance/faults_log.csv
    # - Counts severity == "critical" AND not closed
    # =========================================================
    FAULTS_CSV = os.path.join(P.maintenance_dir, "faults_log.csv")


    def compute_open_critical_faults(faults_csv: str) -> int:
        if not os.path.isfile(faults_csv):
            return 0
        try:
            df = pd.read_csv(faults_csv)
        except Exception:
            return 0
        if df.empty:
            return 0

        # normalize columns
        cols = {c.lower().strip(): c for c in df.columns}
        sev_col = cols.get("fault_severity", None)
        if not sev_col:
            return 0

        # Optional "Status" / "Closed" support (if you add it later)
        status_col = cols.get("fault_status", None)
        closed_col = cols.get("fault_closed", None)

        sev = df[sev_col].astype(str).str.strip().str.lower()

        # If no status info exists → treat everything as open
        is_open = pd.Series(True, index=df.index)

        if status_col:
            stt = df[status_col].astype(str).str.strip().str.lower()
            is_open = ~stt.isin(["closed", "done", "resolved", "fixed"])
        elif closed_col:
            # supports True/False or yes/no
            cl = df[closed_col].astype(str).str.strip().str.lower()
            is_open = ~cl.isin(["true", "1", "yes", "y", "closed"])

        return int((sev == "critical")[is_open].sum())

    # =========================================================
    # ❌ FAILED (last 4 days) — compact list + POPUP reason
    # =========================================================
    def render_failed_home_popup(days_visible: int = 4):
        st.subheader("❌ Failed (last 4 days)")

        if not os.path.exists(ORDERS_FILE):
            st.info("No orders file found.")
            return

        try:
            df = pd.read_csv(ORDERS_FILE)
        except Exception as e:
            st.error(f"Failed to read {ORDERS_FILE}: {e}")
            return

        df = _ensure_cols(df)

        if df.empty:
            st.info("No orders.")
            return

        failed = df[df[STATUS_COL].astype(str).str.strip().str.lower().eq("failed")].copy()
        if failed.empty:
            st.success("No Failed orders 👍")
            return

        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days_visible)

        failed["_dt"] = failed[STATUS_UPDATED_COL].apply(parse_dt_safe)
        failed["_dt"] = failed["_dt"].fillna(now)
        failed = failed[failed["_dt"] >= cutoff].copy().sort_values("_dt", ascending=False)

        if failed.empty:
            st.info("No recent Failed orders.")
            return

        def _open_failed_dialog(title: str, reason: str, updated: str, extra_lines: list):
            @st.dialog(title)
            def _dlg():
                if reason:
                    st.error(reason)
                else:
                    st.info("No failed description recorded.")

                if updated:
                    st.caption(f"Updated: {updated}")

                if extra_lines:
                    st.markdown("**Info**")
                    for line in extra_lines:
                        if line:
                            st.write(f"• {line}")

            _dlg()

        for i, (_, row) in enumerate(failed.iterrows()):
            oid = safe_str(row.get("Order ID"))
            pf = safe_str(row.get("Preform Number"))
            ftype = safe_str(row.get("Fiber Type"))
            proj = safe_str(row.get("Fiber Project"))
            updated = safe_str(row.get(STATUS_UPDATED_COL))
            reason = safe_str(row.get(FAILED_REASON_COL))

            left = " | ".join([p for p in [
                f"#{oid}" if oid else "",
                f"PF {pf}" if pf else "",
                ftype if ftype else "",
                proj if proj else ""
            ] if p])

            extra = []
            if "Required Length (m) (for T&M+costumer)" in failed.columns:
                val = safe_str(row.get("Required Length (m) (for T&M+costumer)"))
                if val:
                    extra.append(f"Required Length: {val} m")
            elif "Required Length (m)" in failed.columns:
                val = safe_str(row.get("Required Length (m)"))
                if val:
                    extra.append(f"Required Length: {val} m")

            if "Priority" in failed.columns:
                val = safe_str(row.get("Priority"))
                if val:
                    extra.append(f"Priority: {val}")

            if "Notes" in failed.columns:
                val = safe_str(row.get("Notes"))
                if val:
                    extra.append(f"Notes: {val}")

            c1, c2 = st.columns([3.2, 1.2])
            with c1:
                st.markdown(f"**{left if left else 'Failed Order'}**")
                if updated:
                    st.caption(f"Updated: {updated}")
            with c2:
                btn_key = f"failed_reason_btn_{i}_{oid}_{pf}"
                if st.button("View reason", key=btn_key, use_container_width=True):
                    dlg_title = left if left else "Failed Order"
                    _open_failed_dialog(
                        title=f"❌ Failed: {dlg_title}",
                        reason=reason,
                        updated=updated,
                        extra_lines=extra
                    )

            st.markdown("---")

    # =========================================================
    # 🔁 AUTO CLEANUP FIRST
    # =========================================================
    auto_move_failed_to_pending(days=4)

    # =========================================================
    # ✅ 1) DRAW ORDERS (keep as-is)
    # =========================================================
    render_home_draw_orders_overview()
    st.markdown("---")

    # =========================================================
    # ✅ 2) DONE
    # =========================================================
    render_done_home_section()
    st.markdown("---")

    # =========================================================
    # ✅ 3) FAILED
    # =========================================================
    render_failed_home_popup(days_visible=4)
    st.markdown("---")

    # =========================================================
    # ✅ 4) CALENDAR / SCHEDULE (MOVED HERE ✅)
    # =========================================================
    render_schedule_home_minimal()
    st.markdown("---")

    # =========================================================
    # 5) MAINTENANCE OVERVIEW (unchanged below)
    # =========================================================
    def compute_maintenance_counts_for_home(
            maint_folder: str,
            dataset_dir: str,
            base_dir: str = None,
    ):
        # (your existing function unchanged)
        import os
        import json
        import datetime as dt
        import pandas as pd
        import numpy as np

        base_dir = base_dir or os.getcwd()

        def get_draw_csv_count(folder: str) -> int:
            if not os.path.isdir(folder):
                return 0
            return sum(1 for f in os.listdir(folder) if f.lower().endswith(".csv") and not f.startswith("~$"))

        def parse_date(x):
            if pd.isna(x) or x == "":
                return None
            d = pd.to_datetime(x, errors="coerce")
            if pd.isna(d):
                return None
            return d.date()

        def parse_float(x):
            if pd.isna(x) or x == "":
                return None
            try:
                return float(x)
            except Exception:
                return None

        def parse_int(x):
            if pd.isna(x) or x == "":
                return None
            try:
                return int(float(x))
            except Exception:
                return None

        def norm_source(s) -> str:
            s = "" if s is None or pd.isna(s) else str(s)
            return s.strip().lower()

        def mode_norm(x: str) -> str:
            s = "" if x is None or pd.isna(x) else str(x).strip().lower()
            if s in ("draw", "draws", "draws_count", "draw_count"):
                return "draws"
            return s

        def load_state(path: str) -> dict:
            try:
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
            return {}

        state_path = os.path.join(maint_folder, "_app_state.json")
        state = load_state(state_path)

        current_date = dt.date.today()
        furnace_hours = float(state.get("furnace_hours", 0.0) or 0.0)
        uv1_hours = float(state.get("uv1_hours", 0.0) or 0.0)
        uv2_hours = float(state.get("uv2_hours", 0.0) or 0.0)
        warn_days = int(state.get("warn_days", 14) or 14)
        warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

        current_draw_count = get_draw_csv_count(dataset_dir)

        if not os.path.isdir(maint_folder):
            return 0, 0

        files = [f for f in os.listdir(maint_folder) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        if not files:
            return 0, 0

        normalize_map = {
            "equipment": "Component",
            "task name": "Task",
            "task id": "Task_ID",
            "interval type": "Interval_Type",
            "interval value": "Interval_Value",
            "interval unit": "Interval_Unit",
            "tracking mode": "Tracking_Mode",
            "hours source": "Hours_Source",
            "calendar rule": "Calendar_Rule",
            "due threshold (days)": "Due_Threshold_Days",
            "document name": "Manual_Name",
            "document file/link": "Document",
            "manual page": "Page",
            "procedure summary": "Procedure_Summary",
            "safety/notes": "Notes",
            "owner": "Owner",
            "last done date": "Last_Done_Date",
            "last done hours": "Last_Done_Hours",
            "last done draw": "Last_Done_Draw",
        }

        REQUIRED = ["Component", "Task", "Tracking_Mode"]
        OPTIONAL = [
            "Task_ID",
            "Interval_Type", "Interval_Value", "Interval_Unit",
            "Due_Threshold_Days",
            "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
            "Manual_Name", "Page", "Document",
            "Procedure_Summary", "Notes", "Owner",
            "Hours_Source", "Calendar_Rule",
        ]

        def read_file(path: str) -> pd.DataFrame:
            if path.lower().endswith(".csv"):
                return pd.read_csv(path)
            return pd.read_excel(path)

        def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.rename(columns={c: normalize_map.get(str(c).strip().lower(), c) for c in df.columns}, inplace=True)
            for r in REQUIRED:
                if r not in df.columns:
                    df[r] = np.nan
            for c in OPTIONAL:
                if c not in df.columns:
                    df[c] = np.nan
            return df

        frames = []
        for fname in sorted(files):
            fpath = os.path.join(maint_folder, fname)
            try:
                raw = read_file(fpath)
                if raw is None or raw.empty:
                    continue
                dfm = normalize_df(raw)
                dfm["Source_File"] = fname
                frames.append(dfm)
            except Exception:
                continue

        if not frames:
            return 0, 0

        dfm = pd.concat(frames, ignore_index=True)

        def pick_current_hours(hours_source: str) -> float:
            hs = norm_source(hours_source)
            if hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
                return float(uv2_hours)
            if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
                return float(uv1_hours)
            return float(furnace_hours)

        dfm["Last_Done_Date_parsed"] = dfm["Last_Done_Date"].apply(parse_date)
        dfm["Last_Done_Hours_parsed"] = dfm["Last_Done_Hours"].apply(parse_float)
        dfm["Last_Done_Draw_parsed"] = dfm["Last_Done_Draw"].apply(parse_int)
        dfm["Current_Hours_For_Task"] = dfm["Hours_Source"].apply(pick_current_hours)
        dfm["Tracking_Mode_norm"] = dfm["Tracking_Mode"].apply(mode_norm)

        def next_due_date(row):
            if row.get("Tracking_Mode_norm") != "calendar":
                return None
            last = row.get("Last_Done_Date_parsed", None)
            if last is None:
                return None
            try:
                v = int(float(row.get("Interval_Value", np.nan)))
            except Exception:
                return None
            unit = str(row.get("Interval_Unit", "")).strip().lower()
            base = pd.Timestamp(last)
            if pd.isna(base) or base is pd.NaT:
                return None
            if "day" in unit:
                out = base + pd.DateOffset(days=v)
            elif "week" in unit:
                out = base + pd.DateOffset(weeks=v)
            elif "month" in unit:
                out = base + pd.DateOffset(months=v)
            elif "year" in unit:
                out = base + pd.DateOffset(years=v)
            else:
                out = base + pd.DateOffset(days=v)
            if pd.isna(out) or out is pd.NaT:
                return None
            return out.date()

        def next_due_hours(row):
            if row.get("Tracking_Mode_norm") != "hours":
                return None
            last_h = row.get("Last_Done_Hours_parsed", None)
            if last_h is None:
                return None
            try:
                v = float(row.get("Interval_Value", np.nan))
            except Exception:
                return None
            if pd.isna(v):
                return None
            return float(last_h) + float(v)

        def next_due_draw(row):
            if row.get("Tracking_Mode_norm") != "draws":
                return None
            last_d = row.get("Last_Done_Draw_parsed", None)
            if last_d is None:
                return None
            try:
                v = int(float(row.get("Interval_Value", np.nan)))
            except Exception:
                return None
            return int(last_d) + int(v)

        dfm["Next_Due_Date"] = dfm.apply(next_due_date, axis=1)
        dfm["Next_Due_Hours"] = dfm.apply(next_due_hours, axis=1)
        dfm["Next_Due_Draw"] = dfm.apply(next_due_draw, axis=1)

        def status_row(row):
            mode = row.get("Tracking_Mode_norm", "")
            if mode == "event":
                return "ROUTINE"

            overdue = False
            due_soon = False

            nd = row.get("Next_Due_Date", None)
            nh = row.get("Next_Due_Hours", None)
            ndr = row.get("Next_Due_Draw", None)

            if nd is not None and not pd.isna(nd):
                if nd < current_date:
                    overdue = True
                else:
                    thresh = row.get("Due_Threshold_Days", np.nan)
                    try:
                        thresh = int(float(thresh)) if not pd.isna(thresh) else int(warn_days)
                    except Exception:
                        thresh = int(warn_days)
                    if (nd - current_date).days <= thresh:
                        due_soon = True

            if nh is not None and not pd.isna(nh):
                nh = float(nh)
                cur_h = float(row.get("Current_Hours_For_Task", 0.0))
                if nh < cur_h:
                    overdue = True
                elif (nh - cur_h) <= float(warn_hours):
                    due_soon = True

            if ndr is not None and not pd.isna(ndr):
                ndr = int(ndr)
                if ndr < int(current_draw_count):
                    overdue = True
                elif (ndr - int(current_draw_count)) <= 5:
                    due_soon = True

            if overdue:
                return "OVERDUE"
            if due_soon:
                return "DUE SOON"
            return "OK"

        dfm["Status"] = dfm.apply(status_row, axis=1)

        overdue = int((dfm["Status"] == "OVERDUE").sum())
        due_soon = int((dfm["Status"] == "DUE SOON").sum())
        return overdue, due_soon

    st.subheader("🧰 Maintenance Overview")

    MAINT_FOLDER = P.maintenance_dir
    DATASET_DIR = os.path.join(os.getcwd(), P.dataset_dir)

    overdue, due_soon = compute_maintenance_counts_for_home(
        maint_folder=MAINT_FOLDER,
        dataset_dir=DATASET_DIR,
    )

    st.session_state["maint_overdue"] = overdue
    st.session_state["maint_due_soon"] = due_soon

    c1, c2 = st.columns(2)
    c1.metric("🔴 Overdue", overdue)
    c2.metric("🟠 Due soon", due_soon)

    st.subheader("🚨 Faults Overview")

    open_critical = compute_open_critical_faults(FAULTS_CSV)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("🟥 Critical open faults", open_critical)

    with c2:
        if open_critical == 0:
            st.success("No critical faults ✅")
        else:
            st.warning("Check Maintenance → Faults")

    with c3:
        if open_critical > 0:
            st.caption("Tip: open 🧰 Maintenance → Faults / Incidents to review.")
    st.markdown("---")
    # =========================================================
    # 6) PARTS NEEDED
    # =========================================================

    render_parts_orders_home_all()
# ------------------ Process Tab ------------------
elif tab_selection == "⚙️ Process Setup":
    df_orders = pd.read_csv(P.orders_csv, keep_default_na=False) if os.path.exists(P.orders_csv) else pd.DataFrame()
    try:
        df_orders = pd.read_csv(ORDERS_FILE, keep_default_na=False) if os.path.exists(ORDERS_FILE) else pd.DataFrame()
    except Exception:
        df_orders = pd.DataFrame()

    render_process_setup_tab(
        orders_df=df_orders,
        orders_file=ORDERS_FILE,
    )
# ------------------ Dashboard Tab ------------------
elif tab_selection == "📊 Dashboard":
    # ==========================================================
    # Imports (local)
    # ==========================================================
    import os
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    from helpers.text_utils import safe_str
    from helpers.dataset_csv_io import append_rows_to_dataset_csv
    from helpers.dataset_param_parsers import (
        _parse_steps,
        zone_lengths_from_log_km,
        build_tm_instruction_rows_auto_from_good_zones,
    )

    # ==========================================================
    # Constants / paths
    # ==========================================================
    DATASET_DIR = P.dataset_dir
    LOGS_DIR = getattr(P, "logs_dir", None) or getattr(P, "log_dir", None) or "logs"

    st.title(f"📊 Draw Tower Logs Dashboard - {selected_file}")

    # ==========================================================
    # Helpers
    # ==========================================================
    def list_dataset_csvs(dataset_dir):
        if not os.path.exists(dataset_dir):
            return []
        return sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")])

    def get_most_recent_dataset_csv(dataset_dir):
        if not os.path.exists(dataset_dir):
            return None
        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
        if not files:
            return None
        return os.path.basename(max(files, key=os.path.getmtime))

    def resolve_log_path(selected_name):
        if not selected_name:
            return ""
        if os.path.exists(selected_name):
            return selected_name
        cand = os.path.join(LOGS_DIR, selected_name)
        if os.path.exists(cand):
            return cand
        for d in ["logs", "log_csv", "draw_logs", "tower_logs", "data_logs"]:
            cand2 = os.path.join(d, selected_name)
            if os.path.exists(cand2):
                return cand2
        return selected_name

    def _fmt_x(v):
        try:
            if isinstance(v, (pd.Timestamp, datetime)):
                return pd.to_datetime(v).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        return str(v)

    # ---------- NEW: clean uniform section titles ----------
    def _sec(title: str):
        return {"Parameter Name": f"### {title}", "Value": "", "Units": ""}

    def _blank():
        return {"Parameter Name": "", "Value": "", "Units": ""}

    def _choose_length_col(df_):
        if df_ is None or df_.empty:
            return None
        cols = list(df_.columns)
        cmap = {str(c).strip().lower(): c for c in cols}
        for k in ["fibre length (km)", "fiber length (km)", "fibre length", "fiber length", "fibre_length", "fiber_length"]:
            if k in cmap:
                return cmap[k]
        for c in cols:
            cl = str(c).lower()
            if "length" in cl and ("fiber" in cl or "fibre" in cl):
                return c
        for c in cols:
            if "length" in str(c).lower():
                return c
        return None

    def reorder_zones_by_spool_end(filtered_df, x_axis, zones, length_col):
        """
        Returns zones reordered so index 1 is closest to spool end.
        Sorting key = smallest km_from_end_start.
        """
        if not zones or filtered_df is None or filtered_df.empty or not length_col:
            return zones

        dfw = filtered_df.sort_values(by=x_axis).copy()
        L_all = pd.to_numeric(dfw[length_col], errors="coerce").dropna()
        if L_all.empty:
            return zones
        L_end = float(L_all.iloc[-1])

        enriched = []
        for orig_i, (zs, ze) in enumerate(zones, start=1):
            try:
                zdf = dfw[(dfw[x_axis] >= zs) & (dfw[x_axis] <= ze)]
            except Exception:
                zdf = pd.DataFrame()
            if zdf.empty:
                continue
            Lz = pd.to_numeric(zdf[length_col], errors="coerce").dropna()
            if Lz.empty:
                continue
            L_max = float(Lz.max())
            km0 = max(0.0, L_end - L_max)  # zone near-end edge
            enriched.append((km0, orig_i, (zs, ze)))

        if not enriched:
            return zones

        enriched.sort(key=lambda t: t[0])  # closest to end first
        return [t[2] for t in enriched]

    def build_zone_save_rows(
        log_file_path,
        x_axis,
        y_axes_selected,
        filtered_df,
        zones,
        include_all_numeric_cols=True,
        always_include_cols=None,
        exclude_cols=None,
    ):
        always_include_cols = always_include_cols or []
        exclude_cols = set([str(c).strip() for c in (exclude_cols or [])])

        rows = []
        now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---------- NEW: nicer, uniform header ----------
        rows += [_sec("DASHBOARD ZONES"), _blank()]

        rows.append({"Parameter Name": "Zones Saved Timestamp", "Value": now_s, "Units": ""})
        rows.append({"Parameter Name": "Dashboard Log File", "Value": os.path.basename(log_file_path), "Units": ""})
        rows.append({"Parameter Name": "Good Zones Count", "Value": int(len(zones)), "Units": "count"})
        rows.append({"Parameter Name": "Good Zones X Column", "Value": str(x_axis), "Units": ""})

        if not zones:
            return rows

        start_end_units = "index/label"
        if pd.api.types.is_datetime64_any_dtype(filtered_df[x_axis]):
            start_end_units = "datetime"
        elif pd.api.types.is_numeric_dtype(filtered_df[x_axis]):
            start_end_units = str(x_axis)

        if include_all_numeric_cols:
            cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in cols if c != x_axis and c not in exclude_cols]
        else:
            cols = [c for c in (y_axes_selected or []) if c not in exclude_cols]

        for c in always_include_cols:
            if c in filtered_df.columns and c not in cols and c not in exclude_cols:
                cols.append(c)

        for i, (start, end) in enumerate(zones, start=1):
            # ---------- NEW: cleaner zone markers ----------
            rows.append({"Parameter Name": f"Zone {i} | Start", "Value": _fmt_x(start), "Units": start_end_units})
            rows.append({"Parameter Name": f"Zone {i} | End", "Value": _fmt_x(end), "Units": start_end_units})

            try:
                zdf = filtered_df[(filtered_df[x_axis] >= start) & (filtered_df[x_axis] <= end)]
            except Exception:
                zdf = pd.DataFrame()

            if zdf.empty:
                continue

            for col in cols:
                vals = pd.to_numeric(zdf[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append({"Parameter Name": f"Zone {i} | {col} | Avg", "Value": float(vals.mean()), "Units": ""})
                rows.append({"Parameter Name": f"Zone {i} | {col} | Min", "Value": float(vals.min()), "Units": ""})
                rows.append({"Parameter Name": f"Zone {i} | {col} | Max", "Value": float(vals.max()), "Units": ""})

        rows += [_blank()]
        return rows

    def build_tm_rows_from_steps_allocate_only(dataset_csv_name: str, steps: list, zones_info: list, length_col_name: str = ""):
        """
        SIMPLE STEP allocation across zones (no spool-end geometry here).
        This is stable and avoids mixing STEP with AUTO.
        """
        rows = []
        # ---------- NEW: nicer header, no ugly dashed rows ----------
        rows += [_sec("T&M CUT/SAVE PLAN"), _blank()]
        rows.append({"Parameter Name": "Plan Source (dataset CSV)", "Value": str(os.path.basename(dataset_csv_name)), "Units": ""})
        rows.append({"Parameter Name": "Plan Mode", "Value": "STEP plan from dataset CSV (allocated on good zones)", "Units": ""})
        if length_col_name:
            rows.append({"Parameter Name": "Length Column (log)", "Value": str(length_col_name), "Units": ""})

        if not steps:
            rows.append({"Parameter Name": "T&M Instructions", "Value": "STEP plan empty.", "Units": ""})
            rows += [_blank()]
            return rows

        rows += [_blank(), _sec("STEP PLAN (from dataset)"), _blank()]
        for i, (a, L) in enumerate(steps, start=1):
            rows.append({"Parameter Name": f"STEP {i} | Action", "Value": str(a).upper(), "Units": ""})
            rows.append({"Parameter Name": f"STEP {i} | Length", "Value": float(L), "Units": "km"})

        rows += [_blank(), _sec("ALLOCATED ON GOOD ZONES"), _blank()]

        tm_i = 1
        step_idx = 0
        step_act, step_rem = steps[0][0], float(steps[0][1])

        saved = 0.0
        cut = 0.0

        for z in zones_info:
            zlen = z.get("len_km")
            zi = z.get("i", None)
            if zlen is None or float(zlen) <= 0:
                continue
            zone_remaining = float(zlen)

            while zone_remaining > 1e-9 and step_idx < len(steps):
                take = min(zone_remaining, step_rem)

                rows.append({"Parameter Name": f"T&M {tm_i} | Action", "Value": str(step_act).upper(), "Units": ""})
                rows.append({"Parameter Name": f"T&M {tm_i} | Length", "Value": float(take), "Units": "km"})
                rows.append({"Parameter Name": f"T&M {tm_i} | From", "Value": f"Zone {zi}", "Units": ""})

                if str(step_act).upper() == "SAVE":
                    saved += float(take)
                else:
                    cut += float(take)

                zone_remaining -= take
                step_rem -= take
                tm_i += 1

                if step_rem <= 1e-9:
                    step_idx += 1
                    if step_idx < len(steps):
                        step_act, step_rem = steps[step_idx][0], float(steps[step_idx][1])

        rows += [_blank(), _sec("T&M TOTALS"), _blank()]
        rows.append({"Parameter Name": "Total Saved Length", "Value": float(saved), "Units": "km"})
        rows.append({"Parameter Name": "Total Cut Length", "Value": float(cut), "Units": "km"})
        rows += [_blank()]
        return rows

    def _extract_selected_drum_from_dataset_df(df_params: pd.DataFrame) -> str:
        """
        Try both old and new parameter names.
        - New Process Setup: Process__Selected Drum
        - Old: Selected Drum
        """
        if df_params is None or df_params.empty:
            return ""
        try:
            s = df_params["Parameter Name"].astype(str).str.strip()
            hit = df_params.loc[s.isin(["Process__Selected Drum", "Selected Drum"]), "Value"]
            if hit is None or hit.empty:
                return ""
            return str(hit.iloc[-1]).strip()
        except Exception:
            return ""

    # ==========================================================
    # Load log CSV
    # ==========================================================
    log_path = resolve_log_path(selected_file)
    if not log_path or not os.path.exists(log_path):
        st.error(f"Failed to read log CSV: file not found.\n\nSelected: {selected_file}\nTried: {log_path}")
        st.stop()

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        st.error(f"Failed to read log CSV: {e}")
        st.stop()

    if df is None or df.empty:
        st.warning("Log CSV loaded but is empty.")
        st.stop()

    # ==========================================================
    # Dataset CSV context
    # ==========================================================
    recent_dataset_csvs = list_dataset_csvs(DATASET_DIR)
    latest_dataset_csv = get_most_recent_dataset_csv(DATASET_DIR)
    st.caption(f"Most recent dataset CSV: **{latest_dataset_csv if latest_dataset_csv else 'None'}**")

    # ==========================================================
    # Session state
    # ==========================================================
    if "good_zones" not in st.session_state:
        st.session_state["good_zones"] = []
    if "dash_last_log_file" not in st.session_state:
        st.session_state["dash_last_log_file"] = ""
    if "dash_zone_msg" not in st.session_state:
        st.session_state["dash_zone_msg"] = ""

    if st.session_state["dash_last_log_file"] != os.path.basename(log_path):
        st.session_state["good_zones"] = []
        st.session_state["dash_last_log_file"] = os.path.basename(log_path)
        st.session_state["dash_zone_msg"] = ""

    # ==========================================================
    # Controls
    # ==========================================================
    column_options = df.columns.tolist()
    if not column_options:
        st.warning("No columns found in log CSV.")
        st.stop()

    x_axis = st.selectbox("Select X-axis", column_options, key="x_axis_dash")

    y_axes = st.multiselect(
        "Select Y-axis column(s)",
        options=column_options,
        default=[],
        key="y_axes_dash_multi",
    )

    if not y_axes:
        st.info("Select one or more **Y-axis** columns to show the plot + zones.")
        st.stop()

    # ==========================================================
    # Stable x typing + filtered df
    # ==========================================================
    df_work = df.copy()

    x_raw = df_work[x_axis]
    x_dt = pd.to_datetime(x_raw, errors="coerce", utc=False)
    dt_ok_ratio = float(x_dt.notna().mean()) if len(x_dt) else 0.0

    if dt_ok_ratio > 0.80:
        df_work[x_axis] = x_dt
    else:
        x_num = pd.to_numeric(x_raw, errors="coerce")
        num_ok_ratio = float(x_num.notna().mean()) if len(x_num) else 0.0
        if num_ok_ratio > 0.80:
            df_work[x_axis] = x_num
        else:
            df_work[x_axis] = x_raw.astype(str)

    filtered_df = df_work.dropna(subset=[x_axis] + y_axes).sort_values(by=x_axis)
    if filtered_df.empty:
        st.warning("No data to plot after filtering NA values for selected X/Y columns.")
        st.stop()

    # ==========================================================
    # Zone Marker UI
    # ==========================================================
    st.subheader("🟩 Zone Marker")
    st.caption("Use the slider to pick a range, then click **Add Selected Zone**.")

    base_key = f"dash_zone_slider__{safe_str(x_axis)}"
    time_range = None

    if pd.api.types.is_datetime64_any_dtype(filtered_df[x_axis]):
        tmin = filtered_df[x_axis].min().to_pydatetime()
        tmax = filtered_df[x_axis].max().to_pydatetime()
        time_range = st.slider(
            "Select Time Range for Good Zone",
            min_value=tmin,
            max_value=tmax,
            value=(tmin, tmax),
            step=pd.Timedelta(seconds=1).to_pytimedelta(),
            format="HH:mm:ss",
            key=f"{base_key}_dt",
        )
    elif pd.api.types.is_numeric_dtype(filtered_df[x_axis]):
        xmin = float(filtered_df[x_axis].min())
        xmax = float(filtered_df[x_axis].max())
        step = (xmax - xmin) / 1000.0 if xmax > xmin else 1.0
        if step <= 0:
            step = 1.0
        time_range = st.slider(
            f"Select Range for Good Zone ({x_axis})",
            min_value=xmin,
            max_value=xmax,
            value=(xmin, xmax),
            step=step,
            key=f"{base_key}_num",
        )
    else:
        i0, i1 = st.slider(
            "Select Index Range for Good Zone",
            min_value=0,
            max_value=max(0, len(filtered_df) - 1),
            value=(0, max(0, len(filtered_df) - 1)),
            step=1,
            key=f"{base_key}_idx",
        )
        xs = filtered_df[x_axis].iloc[int(i0)]
        xe = filtered_df[x_axis].iloc[int(i1)]
        time_range = (xs, xe)

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("➕ Add Selected Zone", key=f"dash_add_zone__{safe_str(x_axis)}", use_container_width=True, disabled=not bool(time_range)):
            st.session_state["good_zones"].append(time_range)
            st.session_state["dash_zone_msg"] = f"✅ Zone added ({len(st.session_state['good_zones'])} total)"
            st.rerun()

    with cB:
        if st.button("🧹 Clear Zones", key=f"dash_clear_zones__{safe_str(x_axis)}", use_container_width=True, disabled=not bool(st.session_state["good_zones"])):
            st.session_state["good_zones"] = []
            st.session_state["dash_zone_msg"] = "🧽 Zones cleared"
            st.rerun()

    with cC:
        st.info(f"Zones currently: **{len(st.session_state['good_zones'])}**")

    if st.session_state.get("dash_zone_msg"):
        st.success(st.session_state["dash_zone_msg"])

    # ==========================================================
    # Plot
    # ==========================================================
    st.subheader("📈 Plot")

    fig = go.Figure()

    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    for i, y_col in enumerate(y_axes):
        axis_ref = "y" if i == 0 else f"y{i + 1}"
        color = default_colors[i % len(default_colors)]

        fig.add_trace(go.Scatter(
            x=filtered_df[x_axis],
            y=pd.to_numeric(filtered_df[y_col], errors="coerce"),
            mode="lines",
            name=y_col,
            yaxis=axis_ref,
            line=dict(color=color),
        ))

    for (start, end) in st.session_state["good_zones"]:
        fig.add_vrect(x0=start, x1=end, fillcolor="green", opacity=0.25, line_width=0)

    if time_range:
        fig.add_vrect(
            x0=time_range[0], x1=time_range[1],
            fillcolor="blue", opacity=0.15,
            line_width=1, line_dash="dot"
        )

    layout_updates = {}
    layout_updates["yaxis"] = dict(
        title=dict(text=""),
        tickfont=dict(color=default_colors[0]),
        showgrid=True,
    )

    right_positions = [1.00, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79]
    for i in range(1, len(y_axes)):
        axis_key = f"yaxis{i + 1}"
        color = default_colors[i % len(default_colors)]
        pos_idx = i - 1
        pos = right_positions[pos_idx] if pos_idx < len(right_positions) else max(0.55, 1.0 - 0.03 * pos_idx)

        layout_updates[axis_key] = dict(
            title=dict(text=""),
            tickfont=dict(color=color),
            anchor="x",
            overlaying="y",
            side="right",
            position=float(pos),
            showgrid=False,
            zeroline=False,
        )

    xaxis_cfg = dict(
        automargin=True,
        nticks=8,
        tickangle=-90,
        showgrid=False,
    )

    if pd.api.types.is_datetime64_any_dtype(filtered_df[x_axis]):
        xaxis_cfg.update(dict(
            tickformat="%d/%m/%Y %H:%M:%S",
            ticklabelmode="instant",
        ))

    annotations = []
    y0 = 1.08
    dy = 0.08
    for i, col in enumerate(y_axes):
        color = default_colors[i % len(default_colors)]
        annotations.append(dict(
            x=0.01,
            y=y0 - i * dy,
            xref="paper",
            yref="paper",
            text=f"<b>{col}</b>",
            showarrow=False,
            align="left",
            font=dict(color=color, size=12),
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            borderpad=6,
        ))

    fig.update_layout(
        **layout_updates,
        xaxis=xaxis_cfg,
        annotations=annotations,
        title=f"{' , '.join([str(y) for y in y_axes])} vs {x_axis}",
        margin=dict(l=10, r=10, t=85, b=10),
        height=620,
        legend=dict(visible=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # Zones summary (kept)
    # ==========================================================
    if st.session_state["good_zones"]:
        st.subheader("✅ Good Zones Summary")
        summary_rows = []
        combined = {y: [] for y in y_axes}

        for i, (start, end) in enumerate(st.session_state["good_zones"], start=1):
            try:
                zone_data = filtered_df[(filtered_df[x_axis] >= start) & (filtered_df[x_axis] <= end)]
            except Exception:
                zone_data = pd.DataFrame()
            if zone_data.empty:
                continue

            for y_col in y_axes:
                vals = pd.to_numeric(zone_data[y_col], errors="coerce").dropna()
                if vals.empty:
                    continue
                summary_rows.append({
                    "Zone": f"Zone {i}",
                    "Y": y_col,
                    "Start": _fmt_x(start),
                    "End": _fmt_x(end),
                    "Avg": float(vals.mean()),
                    "Min": float(vals.min()),
                    "Max": float(vals.max()),
                })
                combined[y_col].extend(vals.values.tolist())

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ==========================================================
    # SAVE
    # ==========================================================
    st.markdown("---")
    st.subheader("💾 Save Zones + T&M Cut/Save Instructions → Dataset CSV")

    if not recent_dataset_csvs:
        st.warning("No dataset CSV files found in data_set_csv/. Create one in Process Setup first.")
    else:
        left, right = st.columns([2, 1])
        with left:
            selected_dataset_csv = st.selectbox(
                "Choose a dataset CSV (optional)",
                options=[""] + recent_dataset_csvs,
                index=0,
                key="dash_save_target_csv_select",
                help="If empty, use MOST RECENT dataset CSV.",
            )
        with right:
            st.caption(f"Most recent: **{latest_dataset_csv if latest_dataset_csv else 'None'}**")

        b1, b2 = st.columns(2)
        with b1:
            save_to_latest_clicked = st.button("⚡ Save to MOST RECENT dataset CSV", key="dash_save_to_latest_btn", use_container_width=True)
        with b2:
            save_to_selected_clicked = st.button("💾 Save to SELECTED dataset CSV", key="dash_save_to_selected_btn", use_container_width=True)

        target_csv = None
        if save_to_latest_clicked:
            target_csv = latest_dataset_csv
        elif save_to_selected_clicked:
            target_csv = selected_dataset_csv

        if target_csv:
            if not st.session_state["good_zones"]:
                st.error("No zones to save. Add at least one zone first.")
            else:
                dataset_path = os.path.join(DATASET_DIR, target_csv)
                if not os.path.exists(dataset_path):
                    st.error(f"Dataset CSV not found: {dataset_path}")
                else:
                    # read dataset
                    try:
                        df_params = pd.read_csv(dataset_path, keep_default_na=False)
                    except Exception as e:
                        st.error(f"Failed reading dataset CSV: {e}")
                        df_params = None

                    # Parse STEP plan (optional)
                    steps = []
                    if df_params is not None:
                        try:
                            steps = _parse_steps(df_params)
                        except Exception:
                            steps = []

                    # Determine length column and reorder zones for saving + AUTO
                    length_col = _choose_length_col(filtered_df)
                    zones_for_save = st.session_state["good_zones"]
                    if length_col:
                        zones_for_save = reorder_zones_by_spool_end(filtered_df, x_axis, zones_for_save, length_col)

                    # 1) Save zone stats (in spool-end order)
                    rows_to_save = build_zone_save_rows(
                        log_file_path=log_path,
                        x_axis=x_axis,
                        y_axes_selected=y_axes,
                        filtered_df=filtered_df,
                        zones=zones_for_save,
                        include_all_numeric_cols=True,
                    )

                    # 2) zone lengths (same order)
                    zones_info, length_col_name = ([], "")
                    try:
                        zones_info, length_col_name = zone_lengths_from_log_km(filtered_df, x_axis, zones_for_save)
                    except Exception as e:
                        rows_to_save.append({"Parameter Name": "T&M Length Error", "Value": str(e), "Units": ""})

                    # ---------- NEW: WINDER & LENGTH group (Drum + Length End + Zone Length Min/Max) ----------
                    rows_to_save += [_blank(), _sec("WINDER & LENGTH"), _blank()]

                    # Drum selected (from dataset CSV)
                    selected_drum_val = ""
                    if df_params is not None:
                        selected_drum_val = _extract_selected_drum_from_dataset_df(df_params)
                    if selected_drum_val:
                        rows_to_save.append({"Parameter Name": "Drum | Selected", "Value": str(selected_drum_val), "Units": ""})

                    # Fiber length end (log end)
                    if length_col:
                        try:
                            L_all = pd.to_numeric(filtered_df[length_col], errors="coerce").dropna()
                            if not L_all.empty:
                                rows_to_save.append({"Parameter Name": "Fiber Length | End (log end)", "Value": float(L_all.iloc[-1]), "Units": "km"})
                        except Exception:
                            pass

                        # Zone fiber length min/max (grouped with drum)
                        for i, (zs, ze) in enumerate(zones_for_save, start=1):
                            try:
                                zdf = filtered_df[(filtered_df[x_axis] >= zs) & (filtered_df[x_axis] <= ze)]
                                Lz = pd.to_numeric(zdf[length_col], errors="coerce").dropna()
                                if Lz.empty:
                                    continue
                                rows_to_save.append({"Parameter Name": f"Zone {i} | Fiber Length | Min", "Value": float(Lz.min()), "Units": "km"})
                                rows_to_save.append({"Parameter Name": f"Zone {i} | Fiber Length | Max", "Value": float(Lz.max()), "Units": "km"})
                            except Exception:
                                continue

                    rows_to_save += [_blank()]

                    # 3) T&M instructions:
                    try:
                        if steps:
                            rows_to_save += build_tm_rows_from_steps_allocate_only(
                                dataset_csv_name=target_csv,
                                steps=steps,
                                zones_info=zones_info,
                                length_col_name=length_col_name or "",
                            )
                        else:
                            rows_to_save += build_tm_instruction_rows_auto_from_good_zones(
                                filtered_df=filtered_df,
                                x_axis=x_axis,
                                good_zones=zones_for_save,
                                length_col_name=length_col_name or None,
                                dataset_csv_name=os.path.basename(target_csv),
                            )
                    except Exception as e:
                        rows_to_save.append({"Parameter Name": "T&M Instructions Error", "Value": str(e), "Units": ""})

                    with st.expander("Preview what will be written", expanded=False):
                        st.dataframe(pd.DataFrame(rows_to_save), use_container_width=True, hide_index=True)

                    try:
                        append_rows_to_dataset_csv(target_csv, rows_to_save)
                        st.success(f"✅ Saved zones + T&M instructions to: {target_csv}")
                    except Exception as e:
                        st.error(f"Failed saving to dataset CSV: {e}")

    # ==========================================================
    # Math Lab (kept minimal)
    # ==========================================================
    st.markdown("---")
    with st.expander("🧮 Math Lab (advanced)", expanded=False):
        st.subheader("A) f(x,y) vs time")
        math_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(math_numeric_cols) < 1:
            st.info("No numeric columns found in this log.")
        else:
            m1, m2, m3 = st.columns([1, 1, 2])
            with m1:
                math_x_col = st.selectbox("Math X column", math_numeric_cols, key="dash_math_x_col")
            with m2:
                math_y_col = st.selectbox("Math Y column (optional)", ["None"] + math_numeric_cols, key="dash_math_y_col")
            with m3:
                default_expr = "x ** y" if math_y_col != "None" else "x"
                math_expr = st.text_input("Expression (use x, y and np)", value=st.session_state.get("dash_math_expr", default_expr), key="dash_math_expr")
                st.caption("Examples: `x**y`, `x*y`, `np.log(x)`, `np.sqrt(x+y)`")

            math_df = df.copy()
            math_df[x_axis] = df_work[x_axis]
            x_arr = pd.to_numeric(math_df[math_x_col], errors="coerce").to_numpy(dtype=float)
            y_arr = None if (math_y_col == "None") else pd.to_numeric(math_df[math_y_col], errors="coerce").to_numpy(dtype=float)

            safe_env = {"x": x_arr, "y": y_arr, "np": np}
            try:
                math_res = eval(math_expr, {"__builtins__": {}}, safe_env)
                math_res = np.asarray(math_res, dtype=float)
                if math_res.shape[0] != len(math_df):
                    st.error("Expression must return an array with the same length as the log.")
                else:
                    math_df["__math_result__"] = math_res
                    math_plot_df = math_df.dropna(subset=[x_axis, "__math_result__"]).sort_values(by=x_axis)
                    fig_math = px.line(math_plot_df, x=x_axis, y="__math_result__", markers=False, title=f"Math Lab: f(x,y) vs {x_axis}")
                    st.plotly_chart(fig_math, use_container_width=True)
            except Exception as e:
                st.error(f"Math Lab error: {e}")
# ------------------ Order Finalize Tab ------------------
elif tab_selection == "✅ Draw Finalize":
    # ==========================================================
    # Imports (local)
    # ==========================================================
    import os, re, time
    import datetime as dt
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd
    import streamlit as st
    import duckdb

    from helpers.text_utils import safe_str
    from helpers.dataset_io import (
        append_rows_to_dataset_csv,
        resolve_dataset_csv_path,
        ensure_dataset_dir,
    )

    st.title("✅ Draw Finalize")
    st.caption("Mark orders as ✅ Done / ❌ Failed from a selected dataset CSV. (Moved out of Dashboard)")

    # ==========================================================
    # Constants
    # ==========================================================
    ORDERS_FILE = P.orders_csv
    DATASET_DIR = P.dataset_dir
    SAP_INVENTORY_FILE = "sap_rods_inventory.csv"

    BASE_DIR = os.getcwd()
    MAINT_FOLDER = P.maintenance_dir
    DB_PATH = os.path.join(BASE_DIR, P.duckdb_path)

    # ✅ Fault logs (same as Maintenance tab)
    os.makedirs(MAINT_FOLDER, exist_ok=True)
    FAULTS_CSV = os.path.join(MAINT_FOLDER, "faults_log.csv")
    FAULTS_ACTIONS_CSV = os.path.join(MAINT_FOLDER, "faults_actions_log.csv")

    FAULTS_COLS = [
        "fault_id",
        "fault_ts",
        "fault_component",
        "fault_title",
        "fault_description",
        "fault_severity",
        "fault_actor",
        "fault_source_file",
        "fault_related_draw",
    ]

    FAULTS_ACTIONS_COLS = [
        "fault_action_id",
        "fault_id",
        "action_ts",
        "action_type",     # close / reopen / note
        "actor",
        "note",
        "fix_summary",
    ]

    ensure_dataset_dir(DATASET_DIR)

    # ==========================================================
    # DuckDB connection (shared with SQL Lab + Maintenance)
    # ==========================================================
    if "sql_duck_con" not in st.session_state:
        st.session_state["sql_duck_con"] = duckdb.connect(DB_PATH)
    con = st.session_state["sql_duck_con"]
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass

    # Create tables if missing (same schema as Maintenance tab)
    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_events (
        fault_id        BIGINT,
        fault_ts        TIMESTAMP,
        component       VARCHAR,
        title           VARCHAR,
        description     VARCHAR,
        severity        VARCHAR,
        actor           VARCHAR,
        source_file     VARCHAR,
        related_draw    VARCHAR
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_actions (
        fault_action_id  BIGINT,
        fault_id         BIGINT,
        action_ts        TIMESTAMP,
        action_type      VARCHAR,
        actor            VARCHAR,
        note             VARCHAR,
        fix_summary      VARCHAR
    );
    """)

    # ==========================================================
    # CSV helpers (append-only)
    # ==========================================================
    def _ensure_csv(path: str, cols: list):
        if not os.path.isfile(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)

    def _append_csv(path: str, cols: list, df_rows: pd.DataFrame):
        _ensure_csv(path, cols)
        df = df_rows.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

        # stringify time fields to avoid dtype crash
        for tcol in [c for c in cols if c.endswith("_ts")]:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        df.to_csv(path, mode="a", header=False, index=False)

    def _read_csv_safe(path: str, cols: list) -> pd.DataFrame:
        if not os.path.isfile(path):
            return pd.DataFrame(columns=cols)
        try:
            df = pd.read_csv(path)
            if df is None:
                return pd.DataFrame(columns=cols)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols].copy()
        except Exception:
            return pd.DataFrame(columns=cols)

    def _latest_fault_state(actions_df: pd.DataFrame) -> dict:
        """
        fault_id -> last action (close/reopen/note) ; closed if last action is 'close'
        """
        out = {}
        if actions_df is None or actions_df.empty:
            return out

        a = actions_df.copy()
        a["action_ts"] = pd.to_datetime(a["action_ts"], errors="coerce")
        a["fault_id"] = pd.to_numeric(a["fault_id"], errors="coerce")
        a = a.dropna(subset=["fault_id"]).copy()
        a["fault_id"] = a["fault_id"].astype(int)

        a = a.sort_values(["fault_id", "action_ts"], ascending=[True, True])
        last = a.groupby("fault_id").tail(1)

        for _, r in last.iterrows():
            fid = int(r["fault_id"])
            typ = safe_str(r.get("action_type", "")).strip().lower()
            out[fid] = {
                "is_closed": (typ == "close"),
                "last_ts": r.get("action_ts", None),
                "last_note": safe_str(r.get("note", "")),
                "last_fix": safe_str(r.get("fix_summary", "")),
                "last_type": typ,
                "last_actor": safe_str(r.get("actor", "")),
            }
        return out

    def _write_fault_action(con, *, fault_id: int, action_type: str, actor: str, note: str = "", fix_summary: str = ""):
        now_dt = dt.datetime.now()
        aid = int(time.time() * 1000)

        try:
            con.execute("""
                INSERT INTO faults_actions
                (fault_action_id, fault_id, action_ts, action_type, actor, note, fix_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [aid, int(fault_id), now_dt, str(action_type), str(actor), str(note), str(fix_summary)])
        except Exception as e:
            st.warning(f"Fault action DB insert failed (still saving CSV): {e}")

        row = pd.DataFrame([{
            "fault_action_id": aid,
            "fault_id": int(fault_id),
            "action_ts": now_dt,
            "action_type": str(action_type),
            "actor": str(actor),
            "note": str(note),
            "fix_summary": str(fix_summary),
        }])
        _append_csv(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS, row)

    # ==========================================================
    # Actor (same key as Maintenance so you type once)
    # ==========================================================
    st.session_state.setdefault("maint_actor", "operator")
    st.text_input("Actor / operator name (for history)", key="maint_actor")
    actor = st.session_state.get("maint_actor", "operator")

    # ==========================================================
    # Component list helper (try to reuse Maintenance task components)
    # ==========================================================
    normalize_map = {
        "equipment": "Component",
        "component": "Component",
        "task name": "Task",
        "task": "Task",
    }

    def _norm_colname(c: str) -> str:
        return str(c).strip().lower()

    def _load_components_from_maintenance_folder(folder: str) -> list:
        comps = set()

        # 1) from existing faults log
        fdf = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
        if not fdf.empty and "fault_component" in fdf.columns:
            for x in fdf["fault_component"].astype(str).fillna("").tolist():
                x = str(x).strip()
                if x:
                    comps.add(x)

        # 2) from maintenance task files (best)
        if not os.path.isdir(folder):
            return sorted(comps)

        files = [f for f in os.listdir(folder) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        for fname in files:
            p = os.path.join(folder, fname)
            try:
                if p.lower().endswith(".csv"):
                    df = pd.read_csv(p)
                else:
                    df = pd.read_excel(p)
                if df is None or df.empty:
                    continue
                # normalize minimal
                df = df.rename(columns={c: normalize_map.get(_norm_colname(c), c) for c in df.columns})
                if "Component" not in df.columns:
                    continue
                for x in df["Component"].astype(str).fillna("").tolist():
                    x = str(x).strip()
                    if x:
                        comps.add(x)
            except Exception:
                continue

        return sorted(comps)

    @st.cache_data(show_spinner=False)
    def _cached_components(folder: str) -> list:
        return _load_components_from_maintenance_folder(folder)

    # ==========================================================
    # Short-lived message window (under Done / Failed)
    # ==========================================================
    FLASH_SECONDS = 6  # window visible for N seconds

    def _set_flash(level: str, title: str, details: str = ""):
        st.session_state["_finalize_flash"] = {
            "ts": time.time(),
            "level": level,      # "success" | "warning" | "info" | "error"
            "title": title,
            "details": details or "",
            "just_set": True,
        }

    def _render_flash_window(where: str):
        flash = st.session_state.get("_finalize_flash")
        if not flash:
            return

        now = time.time()
        if flash.get("just_set"):
            flash["just_set"] = False
            st.session_state["_finalize_flash"] = flash
        else:
            age = now - float(flash.get("ts", 0))
            if age > FLASH_SECONDS:
                st.session_state.pop("_finalize_flash", None)
                return

        try:
            st.autorefresh(interval=1000, limit=FLASH_SECONDS + 2, key=f"finalize_flash_refresh_{where}")
        except Exception:
            pass

        with st.container(border=True):
            lvl = flash.get("level", "info")
            title = flash.get("title", "")
            details = flash.get("details", "")

            if lvl == "success":
                st.success(title)
            elif lvl == "warning":
                st.warning(title)
            elif lvl == "error":
                st.error(title)
            else:
                st.info(title)

            if details:
                st.caption(details)

    # ==========================================================
    # Dataset CSV context
    # ==========================================================
    recent_csv_files = (
        sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")])
        if os.path.exists(DATASET_DIR) else []
    )

    def get_most_recent_dataset_csv(dataset_dir=DATASET_DIR):
        if not os.path.exists(dataset_dir):
            return None
        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
        if not files:
            return None
        return os.path.basename(max(files, key=os.path.getmtime))

    latest_csv = get_most_recent_dataset_csv(DATASET_DIR)
    st.caption(f"Most recent dataset CSV: **{latest_csv if latest_csv else 'None'}**")

    # ==========================================================
    # Helpers (shared)
    # ==========================================================
    def dataset_csv_path(name_or_path: str, dataset_dir: str) -> str:
        return resolve_dataset_csv_path(name_or_path, dataset_dir=dataset_dir)

    def _norm_str(s: str) -> str:
        return (
            pd.Series([s]).astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
            .iloc[0]
        )

    def _alt_names(fn: str) -> list:
        base = _norm_str(os.path.basename(fn))
        alts = {base}
        if base.startswith("fp"):
            alts.add("f" + base[2:])
        if base.startswith("f") and not base.startswith("fp"):
            alts.add("fp" + base[1:])
        return list(alts)

    def _norm_col(series: pd.Series) -> pd.Series:
        return (
            series.astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
        )

    def _ensure_orders_schema(orders: pd.DataFrame) -> pd.DataFrame:
        orders.columns = [str(c).replace("\ufeff", "").strip() for c in orders.columns]
        for col, default in {
            "Status": "Pending",
            "Active CSV": "",
            "Done CSV": "",
            "Done Description": "",
            "Done Timestamp": "",
            "Failed CSV": "",
            "Failed Description": "",
            "Failed Timestamp": "",
            "Preform Length After Draw (cm)": "",
            "Next Planned Draw Date": "",
            "T&M Moved": False,
            "T&M Moved Timestamp": "",
            "Status Updated At": "",
            "Assigned Dataset CSV": "",
        }.items():
            if col not in orders.columns:
                orders[col] = default
        return orders

    def _match_order_row(orders: pd.DataFrame, dataset_csv_filename: str) -> pd.Series:
        target = _norm_str(dataset_csv_filename)
        target_alts = _alt_names(target)

        cols_to_check = []
        for c in ["Assigned Dataset CSV", "Active CSV", "Done CSV", "Failed CSV"]:
            if c in orders.columns:
                cols_to_check.append(c)

        if not cols_to_check:
            return pd.Series([False] * len(orders))

        m = pd.Series([False] * len(orders))
        for c in cols_to_check:
            normed = _norm_col(orders[c])
            for t in target_alts:
                m = m | (normed == t) | normed.str.endswith(t, na=False) | normed.str.contains(re.escape(t), na=False)

        return m

    def _read_dataset_params(target_csv: str):
        p = dataset_csv_path(target_csv, DATASET_DIR)
        if not p or not os.path.exists(p):
            return None, f"Dataset CSV not found: {p}"
        try:
            dfp = pd.read_csv(p, keep_default_na=False)
            return dfp, ""
        except Exception as e:
            return None, f"Failed reading dataset CSV: {e}"

    # ==========================================================
    # PM detection + SAP (optional)
    # ==========================================================
    def is_pm_draw_from_dataset_csv(df_params: pd.DataFrame) -> bool:
        def norm(s):
            return (
                pd.Series([s]).astype(str).fillna("")
                .str.replace("\ufeff", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.replace("'", "", regex=False)
                .str.strip()
                .str.lower()
                .iloc[0]
            )

        try:
            if df_params is None or df_params.empty:
                return False
            if "Parameter Name" not in df_params.columns or "Value" not in df_params.columns:
                return False

            pn = df_params["Parameter Name"].astype(str).apply(norm)
            m = pn == "pm iris system"
            if not m.any():
                return False

            last_row = df_params.loc[m].iloc[-1]
            val = last_row["Value"]

            if isinstance(val, bool):
                return bool(val)

            num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.notna(num):
                return float(num) == 1.0

            s = norm(val)
            return s in {"1", "true", "yes", "y", "on", "t"}
        except Exception:
            return False

    def ensure_sap_inventory_file():
        if os.path.exists(SAP_INVENTORY_FILE):
            return
        inv = pd.DataFrame([{
            "Item": "SAP Rods Set",
            "Count": 0,
            "Units": "sets",
            "Last Updated": "",
            "Notes": ""
        }])
        inv.to_csv(SAP_INVENTORY_FILE, index=False)

    def decrement_sap_rods_set_by_one(source_draw: str, when_str: str = None):
        ensure_sap_inventory_file()
        when_str = when_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            inv = pd.read_csv(SAP_INVENTORY_FILE)
        except Exception as e:
            return False, f"Failed reading {SAP_INVENTORY_FILE}: {e}"

        if "Item" not in inv.columns:
            return False, f"{SAP_INVENTORY_FILE} format invalid (missing 'Item')."

        m = inv["Item"].astype(str).str.strip().str.lower() == "sap rods set"
        if not m.any():
            inv = pd.concat([inv, pd.DataFrame([{
                "Item": "SAP Rods Set",
                "Count": 0,
                "Units": "sets",
                "Last Updated": "",
                "Notes": ""
            }])], ignore_index=True)
            m = inv["Item"].astype(str).str.strip().str.lower() == "sap rods set"

        idx = inv.index[m][0]

        try:
            current = int(float(inv.loc[idx, "Count"]))
        except Exception:
            current = 0

        if current <= 0:
            inv.loc[idx, "Last Updated"] = when_str
            prev = safe_str(inv.loc[idx, "Notes"])
            add = f"[{when_str}] Tried -1 set (PM draw {source_draw}) but Count was {current}."
            inv.loc[idx, "Notes"] = (prev + "\n" + add).strip() if prev else add
            inv.to_csv(SAP_INVENTORY_FILE, index=False)
            return False, f"SAP NOT decremented (Count={current}). Please refill/update inventory."

        inv.loc[idx, "Count"] = current - 1
        inv.loc[idx, "Last Updated"] = when_str
        prev = safe_str(inv.loc[idx, "Notes"])
        add = f"[{when_str}] -1 set (PM draw {source_draw}). New Count={current - 1}."
        inv.loc[idx, "Notes"] = (prev + "\n" + add).strip() if prev else add

        inv.to_csv(SAP_INVENTORY_FILE, index=False)
        return True, f"SAP Rods Set inventory: {current} → {current - 1}"

    # ==========================================================
    # Orders CSV: mark DONE / FAILED
    # ==========================================================
    def mark_draw_order_done_by_dataset_csv(dataset_csv_filename: str, done_desc: str, preform_len_after_cm: float):
        if not os.path.exists(ORDERS_FILE):
            return False, f"{ORDERS_FILE} not found (couldn't mark order done)."

        try:
            orders = pd.read_csv(ORDERS_FILE, keep_default_na=False)
        except Exception as e:
            return False, f"Failed reading {ORDERS_FILE}: {e}"

        orders = _ensure_orders_schema(orders)
        match = _match_order_row(orders, dataset_csv_filename)

        if not match.any():
            sample_active = _norm_col(orders.get("Active CSV", pd.Series([], dtype=str))).dropna().unique()[:12].tolist()
            return False, (
                f"No matching row found in draw_orders.csv for '{dataset_csv_filename}'.\n"
                f"Sample Done/Active CSV values: {sample_active}"
            )

        if match.sum() > 1:
            return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        orders.loc[match, "Status"] = "Done"
        orders.loc[match, "Done CSV"] = os.path.basename(dataset_csv_filename)
        orders.loc[match, "Done Description"] = str(done_desc).strip()
        orders.loc[match, "Preform Length After Draw (cm)"] = float(preform_len_after_cm)
        orders.loc[match, "Done Timestamp"] = now_str
        orders.loc[match, "Status Updated At"] = now_str

        orders.loc[match, "Failed CSV"] = ""
        orders.loc[match, "Failed Description"] = ""
        orders.loc[match, "Failed Timestamp"] = ""
        orders.loc[match, "Next Planned Draw Date"] = ""

        if "Assigned Dataset CSV" in orders.columns:
            cur = orders.loc[match, "Assigned Dataset CSV"].astype(str).iloc[0].strip()
            if cur == "" or cur.lower() == "nan":
                orders.loc[match, "Assigned Dataset CSV"] = os.path.basename(dataset_csv_filename)

        orders.to_csv(ORDERS_FILE, index=False)
        return True, "Order marked as Done."

    def mark_draw_order_failed_by_dataset_csv(dataset_csv_filename: str, failed_desc: str, preform_left_cm: float):
        if not os.path.exists(ORDERS_FILE):
            return False, f"{ORDERS_FILE} not found (couldn't mark order failed)."

        try:
            orders = pd.read_csv(ORDERS_FILE, keep_default_na=False)
        except Exception as e:
            return False, f"Failed reading {ORDERS_FILE}: {e}"

        orders = _ensure_orders_schema(orders)
        match = _match_order_row(orders, dataset_csv_filename)

        if not match.any():
            sample_active = _norm_col(orders.get("Active CSV", pd.Series([], dtype=str))).dropna().unique()[:12].tolist()
            return False, (
                f"No matching row found in draw_orders.csv for '{dataset_csv_filename}'.\n"
                f"Sample Done/Active CSV values: {sample_active}"
            )

        if match.sum() > 1:
            return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        orders.loc[match, "Status"] = "Failed"
        orders.loc[match, "Failed CSV"] = os.path.basename(dataset_csv_filename)
        orders.loc[match, "Failed Description"] = str(failed_desc).strip()
        orders.loc[match, "Failed Timestamp"] = now_str
        orders.loc[match, "Preform Length After Draw (cm)"] = float(preform_left_cm)
        orders.loc[match, "Status Updated At"] = now_str

        orders.loc[match, "Done CSV"] = ""
        orders.loc[match, "Done Description"] = ""
        orders.loc[match, "Done Timestamp"] = ""

        if "Assigned Dataset CSV" in orders.columns:
            cur = orders.loc[match, "Assigned Dataset CSV"].astype(str).iloc[0].strip()
            if cur == "" or cur.lower() == "nan":
                orders.loc[match, "Assigned Dataset CSV"] = os.path.basename(dataset_csv_filename)

        orders.to_csv(ORDERS_FILE, index=False)
        return True, "Order marked as FAILED."

    def reset_failed_order_to_beginning_and_schedule(
        dataset_csv_filename: str,
        schedule_date: str = None,
        scheduled_status: str = "Scheduled",
    ):
        if not os.path.exists(ORDERS_FILE):
            return False, f"{ORDERS_FILE} not found."

        try:
            orders = pd.read_csv(ORDERS_FILE, keep_default_na=False)
        except Exception as e:
            return False, f"Failed reading {ORDERS_FILE}: {e}"

        orders = _ensure_orders_schema(orders)
        match = _match_order_row(orders, dataset_csv_filename)
        if not match.any() or match.sum() != 1:
            return False, f"Could not uniquely match order row for '{dataset_csv_filename}'."

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if schedule_date is None:
            schedule_date = compute_next_planned_draw_date(datetime.now())
        schedule_date = "" if schedule_date is None else str(schedule_date).strip()

        if schedule_date:
            orders.loc[match, "Status"] = scheduled_status
            orders.loc[match, "Next Planned Draw Date"] = schedule_date
        else:
            orders.loc[match, "Status"] = "Pending"
            orders.loc[match, "Next Planned Draw Date"] = ""

        orders.loc[match, "Active CSV"] = ""
        orders.loc[match, "Done CSV"] = ""
        orders.loc[match, "Done Description"] = ""
        orders.loc[match, "Done Timestamp"] = ""
        orders.loc[match, "Failed CSV"] = ""
        orders.loc[match, "Failed Description"] = ""
        orders.loc[match, "Failed Timestamp"] = ""

        orders.loc[match, "T&M Moved"] = False
        orders.loc[match, "T&M Moved Timestamp"] = ""
        orders.loc[match, "Status Updated At"] = now_str

        if "Last Reset Timestamp" not in orders.columns:
            orders["Last Reset Timestamp"] = ""
        orders.loc[match, "Last Reset Timestamp"] = now_str

        orders.to_csv(ORDERS_FILE, index=False)

        if schedule_date:
            return True, f"Reset to **Scheduled**. Next Planned Draw Date = {schedule_date}."
        return True, "Reset to **Pending** (no schedule)."

    def append_failed_metadata_to_dataset_csv(dataset_csv_filename: str, failed_desc: str, preform_left_cm: float):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = [
            {"Parameter Name": "Failed Description", "Value": str(failed_desc).strip(), "Units": ""},
            {"Parameter Name": "Preform Length After Failed Draw", "Value": float(preform_left_cm), "Units": "cm"},
            {"Parameter Name": "Failed Timestamp", "Value": now_str, "Units": ""},
        ]
        return append_rows_to_dataset_csv(dataset_csv_filename, rows, dataset_dir=DATASET_DIR)

    # ==========================================================
    # Fault logging helper (used in Failed tab)
    # ==========================================================
    def log_fault_event_for_draw(
        *,
        con,
        actor: str,
        fault_component: str,
        severity: str,
        title: str,
        description: str,
        source_file: str,
        related_draw: str,
    ):
        now_dt = dt.datetime.now()
        fid = int(time.time() * 1000)

        # DuckDB
        try:
            con.execute("""
                INSERT INTO faults_events
                (fault_id, fault_ts, component, title, description, severity, actor, source_file, related_draw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                fid, now_dt,
                str(fault_component), str(title), str(description),
                str(severity), str(actor), str(source_file), str(related_draw)
            ])
        except Exception as e:
            st.warning(f"DuckDB fault insert failed (still saving CSV): {e}")

        # CSV
        row = pd.DataFrame([{
            "fault_id": fid,
            "fault_ts": now_dt,
            "fault_component": str(fault_component),
            "fault_title": str(title),
            "fault_description": str(description),
            "fault_severity": str(severity),
            "fault_actor": str(actor),
            "fault_source_file": str(source_file),
            "fault_related_draw": str(related_draw),
        }])
        _append_csv(FAULTS_CSV, FAULTS_COLS, row)
        return fid

    # ==========================================================
    # Dataset picker (shared UI)
    # ==========================================================
    st.markdown("---")
    st.subheader("🎯 Select Target Dataset CSV")

    pick_mode = st.radio(
        "Target selection",
        options=["Most recent", "Choose from list"],
        horizontal=True,
        key="finalize_pick_mode",
    )

    if pick_mode == "Most recent":
        target_csv = latest_csv
    else:
        target_csv = st.selectbox(
            "Choose a dataset CSV",
            options=[""] + recent_csv_files,
            index=0,
            key="finalize_choose_csv",
        ) or None

    if not target_csv:
        st.info("Select a dataset CSV to enable Done/Failed actions.")
        st.stop()

    st.success(f"Target: **{target_csv}**")

    related_draw_default = os.path.splitext(os.path.basename(target_csv))[0]

    # Show matched order preview
    if os.path.exists(ORDERS_FILE):
        try:
            orders_preview = pd.read_csv(ORDERS_FILE, keep_default_na=False)
            orders_preview = _ensure_orders_schema(orders_preview)
            match = _match_order_row(orders_preview, target_csv)
            if match.any() and match.sum() == 1:
                r = orders_preview.loc[match].iloc[0]
                st.caption(
                    f"Matched order: **PF {r.get('Preform Number','')}** | "
                    f"Project: **{r.get('Fiber Project','')}** | "
                    f"Status: **{r.get('Status','')}**"
                )
            elif match.sum() > 1:
                st.warning("⚠️ Multiple order rows match this dataset CSV (duplicates).")
            else:
                st.warning("⚠️ No order row matches this dataset CSV in draw_orders.csv.")
        except Exception as e:
            st.warning(f"Order preview error: {e}")

    # ==========================================================
    # Inner tabs: Done / Failed
    # ==========================================================
    tab_done, tab_failed = st.tabs(["✅ Done", "❌ Failed"])

    with tab_done:
        st.subheader("✅ Mark Done")

        done_desc = st.text_area(
            "Done description (what happened / notes)",
            value=st.session_state.get("final_done_desc", ""),
            key="final_done_desc",
            height=100
        )

        preform_len_after_cm = st.number_input(
            "Preform length after draw (cm) — can be 0",
            min_value=0.0,
            value=float(st.session_state.get("final_preform_len_after_cm", 0.0)),
            step=0.5,
            format="%.1f",
            key="final_preform_len_after_cm",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            do_mark_done = st.button(
                "✅ Mark DONE",
                use_container_width=True,
                disabled=(not str(done_desc).strip()),
                key="final_mark_done_btn",
            )
        with c2:
            st.caption("Also appends Done info into dataset CSV.")

        _render_flash_window(where="done")

        if do_mark_done:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            summary_lines = []
            final_level = "success"

            ok_csv, msg_csv = append_rows_to_dataset_csv(
                target_csv,
                [
                    {"Parameter Name": "Preform Length After Draw", "Value": float(preform_len_after_cm), "Units": "cm"},
                    {"Parameter Name": "Done Description", "Value": str(done_desc).strip(), "Units": ""},
                    {"Parameter Name": "Done Timestamp", "Value": now_str, "Units": ""},
                ],
                dataset_dir=DATASET_DIR
            )
            if ok_csv:
                st.toast("✅ Dataset CSV updated", icon="✅")
                summary_lines.append("✅ Dataset CSV updated")
            else:
                st.toast("⚠️ Dataset CSV update failed", icon="⚠️")
                summary_lines.append("⚠️ Dataset CSV update failed")
                final_level = "warning"

            ok, msg = mark_draw_order_done_by_dataset_csv(target_csv, done_desc, float(preform_len_after_cm))
            if not ok:
                st.toast("❌ Failed to mark DONE", icon="❌")
                _set_flash("error", "Finalize FAILED", msg)
                st.rerun()

            st.toast("✅ Order marked DONE", icon="✅")
            summary_lines.append("✅ Order marked DONE")

            try:
                hook_ok, hook_msg = run_after_done_hook(
                    target_csv=target_csv,
                    done_desc=done_desc,
                    preform_len_after_cm=float(preform_len_after_cm),
                    hook_dir=P.hooks_dir,
                    timeout_sec=120,
                )
                if hook_ok:
                    st.toast("✅ After-done hook executed", icon="✅")
                    summary_lines.append("✅ After-done hook executed")
                else:
                    st.toast("⚠️ After-done hook failed", icon="⚠️")
                    summary_lines.append("⚠️ After-done hook failed")
                    final_level = "warning"
            except Exception:
                st.toast("ℹ️ After-done hook skipped", icon="ℹ️")
                summary_lines.append("ℹ️ After-done hook skipped")

            try:
                df_params, err = _read_dataset_params(target_csv)
                if df_params is None:
                    st.toast("⚠️ SAP check failed", icon="⚠️")
                    summary_lines.append("⚠️ SAP check failed")
                    final_level = "warning"
                else:
                    pm_detected = is_pm_draw_from_dataset_csv(df_params)
                    if not pm_detected:
                        st.toast("ℹ️ SAP not updated (not PM)", icon="ℹ️")
                        summary_lines.append("ℹ️ SAP not updated (not PM)")
                    else:
                        inv_ok, inv_msg = decrement_sap_rods_set_by_one(
                            source_draw=related_draw_default,
                            when_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        if inv_ok:
                            st.toast("🧪 SAP inventory updated", icon="✅")
                            summary_lines.append("✅ SAP inventory updated")
                        else:
                            st.toast("⚠️ SAP NOT decremented", icon="⚠️")
                            summary_lines.append("⚠️ SAP NOT decremented")
                            final_level = "warning"
            except Exception:
                st.toast("ℹ️ SAP update skipped", icon="ℹ️")
                summary_lines.append("ℹ️ SAP update skipped")

            try:
                df_params, err = _read_dataset_params(target_csv)
                if df_params is not None and "Parameter Name" in df_params.columns:
                    m = df_params["Parameter Name"].astype(str).str.strip() == "Preform Number"
                    if m.any():
                        pf_name = df_params.loc[m, "Value"].iloc[0]
                        append_preform_length(
                            preform_name=str(pf_name),
                            length_cm=float(preform_len_after_cm),
                            source_draw=related_draw_default,
                        )
                        st.toast("📏 Preform length saved", icon="✅")
                        summary_lines.append(f"✅ Preform saved ({pf_name})")
            except Exception:
                st.toast("ℹ️ Preform registry skipped", icon="ℹ️")
                summary_lines.append("ℹ️ Preform registry skipped")

            _set_flash(final_level, "Finalize DONE", "\n".join(summary_lines))
            st.rerun()

    with tab_failed:
        st.subheader("❌ Mark Failed")

        failed_desc = st.text_area(
            "Failed description (why it failed)",
            value=st.session_state.get("final_failed_desc", ""),
            key="final_failed_desc",
            height=100
        )

        preform_left_cm = st.number_input(
            "Preform length after failed draw (cm) — can be 0",
            min_value=0.0,
            value=float(st.session_state.get("final_preform_left_cm", 0.0)),
            step=0.5,
            format="%.1f",
            key="final_preform_left_cm",
        )

        # ======================================================
        # ✅ Fault insert (same idea as Maintenance tab)
        # ======================================================
        st.markdown("### 🚨 Failure → optionally log a Fault / Incident")

        log_as_fault = st.toggle(
            "This failure is a fault / incident (log to Faults)",
            value=bool(st.session_state.get("final_failed_is_fault", False)),
            key="final_failed_is_fault",
        )

        # Defaults/inputs (only when toggle is ON)
        fault_payload = None
        if log_as_fault:
            comps = _cached_components(MAINT_FOLDER)
            comp_options = (comps if comps else []) + ["Other (custom)"]

            c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
            with c1:
                selected_comp = st.selectbox(
                    "Fault component",
                    options=comp_options,
                    key="final_fault_component_select",
                )
                if selected_comp == "Other (custom)":
                    fault_component = st.text_input("Custom component name", key="final_fault_component_custom")
                else:
                    fault_component = selected_comp
            with c2:
                fault_severity = st.selectbox(
                    "Severity",
                    ["low", "medium", "high", "critical"],
                    index=1,
                    key="final_fault_sev_in",
                )
            with c3:
                st.text_input(
                    "Related draw",
                    value=related_draw_default,
                    disabled=True,
                    key="final_fault_related_draw_view",
                )

            fault_title = st.text_input(
                "Fault title",
                placeholder="Short title",
                value=st.session_state.get("final_fault_title_in", ""),
                key="final_fault_title_in",
            )

            fault_desc_extra = st.text_area(
                "Fault description (details / what to check next time)",
                placeholder="You can paste the failed description + more details…",
                value=st.session_state.get("final_fault_desc_in", ""),
                height=120,
                key="final_fault_desc_in",
            )

            fault_source_file = st.text_input(
                "Source file (optional)",
                placeholder="e.g. photo.jpg / email.pdf / log screenshot",
                value=st.session_state.get("final_fault_src_in", ""),
                key="final_fault_src_in",
            )

            # Build payload (we’ll validate right before saving)
            fault_payload = {
                "fault_component": safe_str(fault_component).strip(),
                "fault_severity": safe_str(fault_severity).strip().lower(),
                "fault_title": safe_str(fault_title).strip(),
                "fault_description": safe_str(fault_desc_extra).strip(),
                "fault_source_file": safe_str(fault_source_file).strip(),
                "fault_related_draw": related_draw_default,
            }

            st.caption("Will be saved to **DuckDB faults_events** + **maintenance/faults_log.csv**")

        st.markdown("---")

        do_mark_failed = st.button(
            "❌ Mark FAILED",
            use_container_width=True,
            disabled=(not str(failed_desc).strip()),
            key="final_mark_failed_btn",
        )

        _render_flash_window(where="failed")

        if do_mark_failed:
            summary_lines = []
            final_level = "success"

            # 1) Mark failed in orders
            okf, msgf = mark_draw_order_failed_by_dataset_csv(target_csv, failed_desc, float(preform_left_cm))
            if not okf:
                st.toast("⚠️ Failed to mark order FAILED", icon="⚠️")
                _set_flash("warning", "Failed to mark order FAILED", msgf)
                st.rerun()

            st.toast("❌ Order marked FAILED", icon="✅")
            summary_lines.append("✅ Order marked FAILED")

            # 2) Append failed metadata into dataset CSV
            ok_csv, msg_csv = append_failed_metadata_to_dataset_csv(target_csv, failed_desc, float(preform_left_cm))
            if ok_csv:
                st.toast("✅ Failed metadata saved to dataset CSV", icon="✅")
                summary_lines.append("✅ Failed metadata saved to dataset CSV")
            else:
                st.toast("⚠️ Failed metadata NOT saved to dataset CSV", icon="⚠️")
                summary_lines.append("⚠️ Failed metadata NOT saved to dataset CSV")
                final_level = "warning"

            # 3) Optional: log fault (same style as Maintenance)
            if log_as_fault:
                try:
                    if not fault_payload:
                        raise RuntimeError("Fault payload missing.")

                    comp = fault_payload["fault_component"]
                    title = fault_payload["fault_title"]
                    desc = fault_payload["fault_description"]

                    # If user didn’t type extra desc, reuse failed_desc
                    if not desc:
                        desc = str(failed_desc).strip()

                    # If user didn’t give title, auto-create from failed_desc
                    if not title:
                        title = (str(failed_desc).strip()[:80] + "…") if len(str(failed_desc).strip()) > 80 else str(failed_desc).strip()

                    if not comp:
                        st.toast("⚠️ Fault component is required", icon="⚠️")
                        summary_lines.append("⚠️ Fault NOT logged (missing component)")
                        final_level = "warning"
                    elif not title and not desc:
                        st.toast("⚠️ Give fault title or description", icon="⚠️")
                        summary_lines.append("⚠️ Fault NOT logged (missing title/desc)")
                        final_level = "warning"
                    else:
                        fid = log_fault_event_for_draw(
                            con=con,
                            actor=actor,
                            fault_component=comp,
                            severity=fault_payload["fault_severity"] or "medium",
                            title=title,
                            description=desc,
                            source_file=fault_payload["fault_source_file"],
                            related_draw=related_draw_default,
                        )
                        st.toast("🚨 Fault logged", icon="✅")
                        summary_lines.append(f"✅ Fault logged (ID {fid})")
                except Exception as e:
                    st.toast("⚠️ Fault logging failed", icon="⚠️")
                    summary_lines.append(f"⚠️ Fault logging failed: {e}")
                    final_level = "warning"

            # ✅ One flash
            _set_flash(final_level, "Finalize FAILED", "\n".join(summary_lines))

            # Next step buttons
            st.info("Next step:")
            c1, c2 = st.columns(2)

            with c1:
                if st.button("📅 Draw next day (reset + schedule)", key="final_failed_schedule_nextday", use_container_width=True):
                    schedule_date = compute_next_planned_draw_date(datetime.now())
                    oks, msgs = reset_failed_order_to_beginning_and_schedule(
                        target_csv,
                        schedule_date=schedule_date,
                        scheduled_status="Scheduled",
                    )
                    if oks:
                        st.toast("✅ Reset + scheduled", icon="✅")
                        _set_flash("success", "Reset + scheduled", msgs)
                    else:
                        st.toast("⚠️ Reset failed", icon="⚠️")
                        _set_flash("warning", "Reset failed", msgs)
                    st.rerun()

            with c2:
                if st.button("↩ Return to Pending (no schedule)", key="final_failed_return_pending", use_container_width=True):
                    oks, msgs = reset_failed_order_to_beginning_and_schedule(
                        target_csv,
                        schedule_date="",
                        scheduled_status="Scheduled",
                    )
                    if oks:
                        st.toast("✅ Reset to Pending", icon="✅")
                        _set_flash("success", "Reset to Pending", msgs)
                    else:
                        st.toast("⚠️ Reset failed", icon="⚠️")
                        _set_flash("warning", "Reset failed", msgs)
                    st.rerun()

            # If user doesn’t click next-step, still rerun to show flash cleanly
            st.rerun()
# ------------------ Consumables Tab ------------------
elif tab_selection == "🍃 Tower state - Consumables and dies":
    # ==========================================================
    # Imports (local to tab)
    # ==========================================================
    import os, json, math
    from datetime import datetime, timedelta

    import pandas as pd
    import streamlit as st

    # ✅ Your project imports
    from app_io.paths import (
        P, ensure_logs_dir, ensure_gas_reports_dir, gas_report_path, _abs, ensure_dir
    )
    from renders.navigation import render_navigation

    ensure_logs_dir()
    ensure_gas_reports_dir()

    # ==========================================================
    # UI polish
    # ==========================================================
    st.markdown("""
    <style>
      .block-container { padding-top: 1.55rem; }

      .section-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        padding: 14px 14px;
        border-radius: 14px;
        margin-bottom: 14px;
      }

      .vessel {
        height: 120px; width: 34px; border: 1px solid rgba(255,255,255,0.22);
        margin: auto; position: relative; border-radius: 10px;
        background: rgba(255,255,255,0.04);
        overflow: hidden;
      }
      .vessel-fill { position: absolute; bottom: 0; width: 100%; background: rgba(76, 175, 80, 0.85); }

      .muted { opacity: 0.75; }
      .low-card { border: 1px solid rgba(255, 77, 77, 0.60) !important; background: rgba(255, 77, 77, 0.05) !important; }
      .low-num { color: rgba(255, 170, 170, 1.0); font-weight: 800; }
      code { font-size: 0.86rem; }
    </style>
    """, unsafe_allow_html=True)

    try:
        render_navigation()
    except Exception:
        pass

    st.title("🍃 Tower state — Consumables & Dies")

    # ==========================================================
    # Constants / Files
    # ==========================================================
    container_labels = ["A", "B", "C", "D"]

    LOW_STOCK_KG = 1.0
    WAREHOUSE_STOCK_FILE = _abs("coating_type_stock.json")

    TOWER_TEMPS_CSV = getattr(P, "tower_temps_csv", _abs("tower_temps.csv"))
    TOWER_CONTAINERS_CSV = getattr(P, "tower_containers_csv", _abs("tower_containers.csv"))

    CONTAINER_SNAPSHOT_FILE = _abs("container_levels_prev.json")
    CONTAINER_CFG_PATH = P.container_config_json

    # ==========================================================
    # Helpers
    # ==========================================================
    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return float(default)
            if isinstance(x, str) and x.strip() == "":
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _read_json(path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default

    def _write_json(path, obj):
        ensure_dir(os.path.dirname(path) or ".")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=4)
        os.replace(tmp, path)

    def _read_one_row_csv(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_csv(path)
            if df.empty:
                return {}
            return df.iloc[-1].to_dict()
        except Exception:
            return {}

    def _write_one_row_csv(path: str, cols: list, data: dict):
        ensure_dir(os.path.dirname(path) or ".")
        row = {c: "" for c in cols}
        row.update(data or {})
        row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([row], columns=cols).to_csv(path, index=False)

    def _list_files_recursive(root: str, exts=(".csv",)):
        out = []
        try:
            for base, _, files in os.walk(root):
                for fn in files:
                    if fn.lower().endswith(exts):
                        out.append(os.path.join(base, fn))
        except Exception:
            pass
        out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return out

    # ==========================================================
    # Load coating types
    # ==========================================================
    with open(P.coating_config_json, "r") as config_file:
        config = json.load(config_file)
    coatings = config.get("coatings", {})
    coating_types = list(coatings.keys())

    # ==========================================================
    # 1) TEMPS CSV (wide)
    # ==========================================================
    TEMP_COLS = [
        "updated_at",
        "die_holder_primary_c",
        "die_holder_secondary_c",
        "A_container_c", "A_pipe_c",
        "B_container_c", "B_pipe_c",
        "C_container_c", "C_pipe_c",
        "D_container_c", "D_pipe_c",
    ]
    TEMP_STATE_KEYS = {
        "die_holder_primary_c": "die_holder_primary_temp_state",
        "die_holder_secondary_c": "die_holder_secondary_temp_state",
        "A_container_c": "temp_A_container_state",
        "A_pipe_c": "temp_A_pipe_state",
        "B_container_c": "temp_B_container_state",
        "B_pipe_c": "temp_B_pipe_state",
        "C_container_c": "temp_C_container_state",
        "C_pipe_c": "temp_C_pipe_state",
        "D_container_c": "temp_D_container_state",
        "D_pipe_c": "temp_D_pipe_state",
    }

    wide_temps = _read_one_row_csv(TOWER_TEMPS_CSV)
    for col, skey in TEMP_STATE_KEYS.items():
        if skey not in st.session_state:
            if col in wide_temps and str(wide_temps.get(col, "")).strip() != "":
                st.session_state[skey] = float(_safe_float(wide_temps[col], 25.0))
            else:
                st.session_state[skey] = 25.0

    # ==========================================================
    # 2) CONTAINERS CSV (wide)
    # ==========================================================
    CONTAINER_COLS = [
        "updated_at",
        "A_level_kg", "A_type",
        "B_level_kg", "B_type",
        "C_level_kg", "C_type",
        "D_level_kg", "D_type",
    ]
    def _lvl_key(lab): return f"cont_level_{lab}"
    def _type_key(lab): return f"cont_type_{lab}"

    wide_cont = _read_one_row_csv(TOWER_CONTAINERS_CSV)

    legacy_cfg = _read_json(CONTAINER_CFG_PATH, {})
    if not isinstance(legacy_cfg, dict):
        legacy_cfg = {}

    for lab in container_labels:
        default_level = 0.0
        default_type = coating_types[0] if coating_types else ""

        lvl_col = f"{lab}_level_kg"
        typ_col = f"{lab}_type"

        if lvl_col in wide_cont and str(wide_cont.get(lvl_col, "")).strip() != "":
            default_level = _safe_float(wide_cont.get(lvl_col), default_level)
        else:
            if isinstance(legacy_cfg.get(lab, {}), dict):
                default_level = _safe_float(legacy_cfg.get(lab, {}).get("level", default_level), default_level)

        if typ_col in wide_cont and str(wide_cont.get(typ_col, "")).strip() != "":
            default_type = str(wide_cont.get(typ_col))
        else:
            if isinstance(legacy_cfg.get(lab, {}), dict):
                default_type = str(legacy_cfg.get(lab, {}).get("type", default_type))

        if coating_types and default_type not in coating_types:
            default_type = coating_types[0]

        st.session_state.setdefault(_lvl_key(lab), float(default_level))
        st.session_state.setdefault(_type_key(lab), default_type)

    # ==========================================================
    # Top toolbar: Refresh buttons
    # ==========================================================
    tL, tM, tR = st.columns([1.3, 1.3, 1])
    with tL:
        st.caption(f"Temps CSV: `{TOWER_TEMPS_CSV}`")
        refresh_temps = st.button("🔄 Refresh temps", use_container_width=True, key="refresh_temps_btn")
    with tM:
        st.caption(f"Containers CSV: `{TOWER_CONTAINERS_CSV}`")
        refresh_containers = st.button("🔄 Refresh containers", use_container_width=True, key="refresh_containers_btn")
    with tR:
        st.caption(" ")
        st.caption(" ")

    if refresh_temps:
        wide_temps = _read_one_row_csv(TOWER_TEMPS_CSV)
        for col, skey in TEMP_STATE_KEYS.items():
            if col in wide_temps and str(wide_temps.get(col, "")).strip() != "":
                st.session_state[skey] = float(_safe_float(wide_temps[col], st.session_state.get(skey, 25.0)))
        st.success("Temps reloaded from CSV.")
        st.rerun()

    if refresh_containers:
        wide_cont = _read_one_row_csv(TOWER_CONTAINERS_CSV)
        for lab in container_labels:
            lvl_col = f"{lab}_level_kg"
            typ_col = f"{lab}_type"
            if lvl_col in wide_cont and str(wide_cont.get(lvl_col, "")).strip() != "":
                st.session_state[_lvl_key(lab)] = float(_safe_float(wide_cont[lvl_col], st.session_state[_lvl_key(lab)]))
            if typ_col in wide_cont and str(wide_cont.get(typ_col, "")).strip() != "":
                t = str(wide_cont[typ_col])
                if coating_types and t not in coating_types:
                    t = coating_types[0]
                st.session_state[_type_key(lab)] = t
        st.success("Containers reloaded from CSV.")
        st.rerun()

    # ==========================================================
    # Warehouse stock (kg) - JSON
    # ==========================================================
    warehouse_stock = _read_json(WAREHOUSE_STOCK_FILE, {})
    if not isinstance(warehouse_stock, dict):
        warehouse_stock = {}
    for t in coating_types:
        warehouse_stock[t] = _safe_float(warehouse_stock.get(t, 0.0), 0.0)

    prev_snapshot = _read_json(CONTAINER_SNAPSHOT_FILE, {})
    if not isinstance(prev_snapshot, dict):
        prev_snapshot = {}
    for lab in container_labels:
        prev_snapshot.setdefault(
            lab,
            {"level": float(st.session_state[_lvl_key(lab)]), "type": str(st.session_state[_type_key(lab)])}
        )

    # ==========================================================
    # UI: Containers
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🧪 Coating Containers (A–D)")
    st.caption("Rule: level ↑ = refill (auto subtract from warehouse). level ↓ = consumption.")

    cols = st.columns(4)
    current_container_state = {}

    for col, lab in zip(cols, container_labels):
        with col:
            st.markdown(f"**Container {lab}**")

            lvl = st.slider(
                f"Fill Level {lab} (kg)",
                0.0, 4.0,
                float(st.session_state[_lvl_key(lab)]),
                0.1,
                key=_lvl_key(lab)
            )

            if coating_types:
                cur_t = st.session_state[_type_key(lab)]
                if cur_t not in coating_types:
                    cur_t = coating_types[0]
                idx_t = coating_types.index(cur_t)
                ctype = st.selectbox(
                    f"Coating Type {lab}",
                    options=coating_types,
                    index=idx_t,
                    key=_type_key(lab)
                )
            else:
                ctype = ""
                st.info("No coating types configured.")

            fill_height = int((float(lvl) / 4.0) * 100.0)
            st.markdown(
                f"""
                <div class="vessel"><div class="vessel-fill" style="height:{fill_height}%;"></div></div>
                <div style="text-align:center; margin-top:6px;"><b>{float(lvl):.2f} kg</b></div>
                """,
                unsafe_allow_html=True
            )

            current_container_state[lab] = {"level": float(lvl), "type": str(ctype)}

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # Apply refill delta to warehouse (snapshot-based)
    # ==========================================================
    refill_events = []
    for lab in container_labels:
        prev_level = _safe_float(prev_snapshot.get(lab, {}).get("level", 0.0), 0.0)
        cur_level = _safe_float(current_container_state[lab]["level"], 0.0)
        cur_type = str(current_container_state[lab]["type"])
        delta = cur_level - prev_level

        if delta > 1e-9 and cur_type in warehouse_stock:
            before = _safe_float(warehouse_stock.get(cur_type, 0.0), 0.0)
            after = max(0.0, before - float(delta))
            warehouse_stock[cur_type] = after
            refill_events.append((lab, cur_type, float(delta), before, after))

        prev_snapshot[lab] = {"level": float(cur_level), "type": cur_type}

    try:
        _write_json(WAREHOUSE_STOCK_FILE, warehouse_stock)
        _write_json(CONTAINER_SNAPSHOT_FILE, prev_snapshot)
    except Exception as e:
        st.error(f"Auto-save failed: {e}")

    if refill_events:
        with st.expander("🧾 Detected refills (auto)", expanded=False):
            for lab, ctype, delta, before, after in refill_events:
                st.write(
                    f"Container **{lab}** refilled **+{delta:.2f} kg** of **{ctype}** → "
                    f"Warehouse: {before:.2f} → {after:.2f} kg"
                )

    # ==========================================================
    # Auto-save containers to CSV
    # ==========================================================
    last_cont_saved = st.session_state.get("containers_last_saved_snapshot", {})
    cur_cont_snapshot = {lab: (current_container_state[lab]["level"], current_container_state[lab]["type"]) for lab in container_labels}

    if not last_cont_saved:
        st.session_state["containers_last_saved_snapshot"] = cur_cont_snapshot
        last_cont_saved = cur_cont_snapshot

    if cur_cont_snapshot != last_cont_saved:
        out = {}
        for lab in container_labels:
            out[f"{lab}_level_kg"] = float(current_container_state[lab]["level"])
            out[f"{lab}_type"] = str(current_container_state[lab]["type"])

        try:
            _write_one_row_csv(TOWER_CONTAINERS_CSV, CONTAINER_COLS, out)
            st.session_state["containers_last_saved_snapshot"] = cur_cont_snapshot

            legacy_out = {lab: {"level": float(current_container_state[lab]["level"]), "type": str(current_container_state[lab]["type"])} for lab in container_labels}
            _write_json(CONTAINER_CFG_PATH, legacy_out)
        except Exception as e:
            st.error(f"Failed to write containers CSV: {e}")

    # ==========================================================
    # Stock by type (warehouse + containers)
    # ==========================================================
    def _sum_containers_by_type(container_state: dict):
        sums = {t: 0.0 for t in coating_types}
        for lab in container_labels:
            t = container_state.get(lab, {}).get("type", "")
            lvl = _safe_float(container_state.get(lab, {}).get("level", 0.0), 0.0)
            if t in sums:
                sums[t] += float(lvl)
        return sums

    container_sums = _sum_containers_by_type(current_container_state)
    total_by_type = {
        t: _safe_float(warehouse_stock.get(t, 0.0), 0.0) + _safe_float(container_sums.get(t, 0.0), 0.0)
        for t in coating_types
    }

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🏷️ Coating Stock by Type (Auto)")
    st.caption("Computed from warehouse + container contents. Red when total < 1 kg.")

    if coating_types:
        max_total = max(total_by_type.values()) if total_by_type else 0.0
        display_max = max(40.0, math.ceil(max_total / 5.0) * 5.0) if max_total > 0 else 40.0

        rows = [coating_types[i:i + 4] for i in range(0, len(coating_types), 4)]
        for row in rows:
            cols = st.columns(len(row))
            for col, ctype in zip(cols, row):
                with col:
                    total_kg = float(total_by_type.get(ctype, 0.0))
                    ware_kg = float(warehouse_stock.get(ctype, 0.0))
                    cont_kg = float(container_sums.get(ctype, 0.0))
                    fill_height = int(min(100.0, (total_kg / display_max) * 100.0)) if display_max > 0 else 0
                    is_low = total_kg < LOW_STOCK_KG

                    st.markdown(f"**{ctype}**")
                    st.markdown(
                        f"""
                        <div class="{'vessel low-card' if is_low else 'vessel'}">
                          <div class="vessel-fill" style="height:{fill_height}%; {'background: rgba(255, 77, 77, 0.85);' if is_low else ''}"></div>
                        </div>
                        <div style="text-align:center; margin-top:6px;">
                          <div class="{ 'low-num' if is_low else '' }"><b>{total_kg:.2f} kg</b></div>
                          <div class="muted" style="font-size:0.85rem;">Warehouse {ware_kg:.2f} + Containers {cont_kg:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info("No coating types found in coating config.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # Warehouse editor
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📦 Warehouse Stock (Edit when new material arrives)")
    st.caption("Bulk stock not inside containers A–D.")

    if coating_types:
        edited = False
        rows = [coating_types[i:i + 3] for i in range(0, len(coating_types), 3)]
        for row in rows:
            cols = st.columns(len(row))
            for col, ctype in zip(cols, row):
                with col:
                    k = f"wh_{ctype}"
                    st.session_state.setdefault(k, float(warehouse_stock.get(ctype, 0.0)))
                    val = st.number_input(
                        f"{ctype} (kg)",
                        min_value=0.0,
                        step=0.1,
                        value=float(st.session_state[k]),
                        key=k
                    )
                    if abs(val - float(warehouse_stock.get(ctype, 0.0))) > 1e-9:
                        warehouse_stock[ctype] = float(val)
                        edited = True

        if edited:
            try:
                _write_json(WAREHOUSE_STOCK_FILE, warehouse_stock)
                st.success("Warehouse stock updated.")
            except Exception as e:
                st.error(f"Failed to save warehouse stock: {e}")
    else:
        st.info("No coating types configured.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # Temps UI
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🌡️ Temperatures (CSV-based)")

    for lab in container_labels:
        st.markdown(f"**Container {lab} temps**")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                f"Container temp {lab} (°C)",
                min_value=0.0,
                step=0.1,
                value=float(st.session_state[TEMP_STATE_KEYS[f"{lab}_container_c"]]),
                key=TEMP_STATE_KEYS[f"{lab}_container_c"]
            )
        with c2:
            st.number_input(
                f"Pipe temp {lab} (°C)",
                min_value=0.0,
                step=0.1,
                value=float(st.session_state[TEMP_STATE_KEYS[f"{lab}_pipe_c"]]),
                key=TEMP_STATE_KEYS[f"{lab}_pipe_c"]
            )

    st.markdown("---")
    st.subheader("🔥 Die Holder Heater (Global)")
    cH1, cH2 = st.columns(2)
    with cH1:
        st.number_input(
            "Primary die holder heater temp (°C)",
            min_value=0.0,
            step=0.1,
            value=float(st.session_state[TEMP_STATE_KEYS["die_holder_primary_c"]]),
            key=TEMP_STATE_KEYS["die_holder_primary_c"]
        )
    with cH2:
        st.number_input(
            "Secondary die holder heater temp (°C)",
            min_value=0.0,
            step=0.1,
            value=float(st.session_state[TEMP_STATE_KEYS["die_holder_secondary_c"]]),
            key=TEMP_STATE_KEYS["die_holder_secondary_c"]
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-save temps to CSV if changed
    last_temp_saved = st.session_state.get("temps_last_saved_snapshot", {})
    cur_temp_snapshot = {skey: float(st.session_state.get(skey, 0.0)) for skey in TEMP_STATE_KEYS.values()}

    if not last_temp_saved:
        st.session_state["temps_last_saved_snapshot"] = cur_temp_snapshot
        last_temp_saved = cur_temp_snapshot

    if cur_temp_snapshot != last_temp_saved:
        out = {
            "die_holder_primary_c": float(st.session_state[TEMP_STATE_KEYS["die_holder_primary_c"]]),
            "die_holder_secondary_c": float(st.session_state[TEMP_STATE_KEYS["die_holder_secondary_c"]]),
        }
        for lab in container_labels:
            out[f"{lab}_container_c"] = float(st.session_state[TEMP_STATE_KEYS[f"{lab}_container_c"]])
            out[f"{lab}_pipe_c"] = float(st.session_state[TEMP_STATE_KEYS[f"{lab}_pipe_c"]])

        try:
            _write_one_row_csv(TOWER_TEMPS_CSV, TEMP_COLS, out)
            st.session_state["temps_last_saved_snapshot"] = cur_temp_snapshot
        except Exception as e:
            st.error(f"Failed to write temps CSV: {e}")

    # ==========================================================
    # Dies system
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🔩 Dies System")

    DIES_CONFIG_PATH = _abs("dies_6station.json")

    default_cfg = {
        f"Station {i}": {
            "entry_die_um": 0.0,
            "primary_die_um": 0.0,
            "primary_on_tower": False,
            "secondary_on_tower": False
        } for i in range(1, 7)
    }

    if os.path.exists(DIES_CONFIG_PATH):
        with open(DIES_CONFIG_PATH, "r") as f:
            try:
                dies_cfg = json.load(f)
                if not isinstance(dies_cfg, dict) or len(dies_cfg) == 0:
                    dies_cfg = default_cfg
            except Exception:
                dies_cfg = default_cfg
    else:
        dies_cfg = default_cfg
        with open(DIES_CONFIG_PATH, "w") as f:
            json.dump(dies_cfg, f, indent=4)

    station_names = list(dies_cfg.keys())

    for name in station_names:
        safe_key = name.replace(" ", "_").replace("/", "_")
        st.session_state.setdefault(f"dies_entry_{safe_key}", float(dies_cfg.get(name, {}).get("entry_die_um", 0.0)))
        st.session_state.setdefault(f"dies_primary_{safe_key}", float(dies_cfg.get(name, {}).get("primary_die_um", 0.0)))
        st.session_state.setdefault(f"dies_primary_on_{safe_key}", bool(dies_cfg.get(name, {}).get("primary_on_tower", False)))
        st.session_state.setdefault(f"dies_secondary_on_{safe_key}", bool(dies_cfg.get(name, {}).get("secondary_on_tower", False)))

    rows = [station_names[i:i + 3] for i in range(0, len(station_names), 3)]
    updated_dies_cfg = {}

    for row in rows:
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            safe_key = name.replace(" ", "_").replace("/", "_")
            with col:
                st.markdown(f"### {name}")

                entry_um = st.number_input("Entry die (µm)", min_value=0.0, step=1.0, format="%.1f", key=f"dies_entry_{safe_key}")
                primary_um = st.number_input("Primary die (µm)", min_value=0.0, step=1.0, format="%.1f", key=f"dies_primary_{safe_key}")
                primary_on = st.checkbox("Primary on tower", key=f"dies_primary_on_{safe_key}")
                secondary_on = st.checkbox("Secondary on tower", key=f"dies_secondary_on_{safe_key}")

                updated_dies_cfg[name] = {
                    "entry_die_um": float(entry_um),
                    "primary_die_um": float(primary_um),
                    "primary_on_tower": bool(primary_on),
                    "secondary_on_tower": bool(secondary_on),
                }

                st.caption(f"Entry: **{entry_um:.1f} µm** | Primary: **{primary_um:.1f} µm**")

    try:
        with open(DIES_CONFIG_PATH, "w") as f:
            json.dump(updated_dies_cfg, f, indent=4)
        st.caption(f"Auto-saved to `{DIES_CONFIG_PATH}`")
    except Exception as e:
        st.error(f"Failed to save dies config: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # 🧯 GAS REPORTS (AUTO MONTHLY CSV) ✅ ALL MFCs = ARGON
    #   - NO JSON
    #   - NO daily/weekly
    #   - outputs one CSV: argon_monthly_report.csv
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🧯 Argon Report — Monthly (AUTO from logs)")
    st.caption("Auto-builds a single monthly CSV report from logs. All Furnace MFC1–4 Actual are summed as Argon.")

    GAS_DIR = getattr(P, "gas_reports_dir", None) or _abs("gas_reports")
    LOGS_DIR = getattr(P, "logs_dir", None) or _abs("logs")
    ensure_dir(GAS_DIR)
    ensure_dir(LOGS_DIR)

    REPORT_CSV = os.path.join(GAS_DIR, "argon_monthly_report.csv")
    STATE_JSON = os.path.join(GAS_DIR, "_argon_monthly_state.json")

    TIME_COL = "Date/Time"
    MFC_ACTUAL_COLS = [
        "Furnace MFC1 Actual",
        "Furnace MFC2 Actual",
        "Furnace MFC3 Actual",
        "Furnace MFC4 Actual",
    ]


    def _read_json(path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default


    def _write_json(path, obj):
        ensure_dir(os.path.dirname(path) or ".")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=4)
        os.replace(tmp, path)


    def _parse_dt_series_date_time(col: pd.Series) -> pd.Series:
        # Handles your format: '19/11/2024 12:44:33772'
        s = col.astype(str)

        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt.notna().sum() >= max(10, int(0.6 * len(dt))):
            return dt

        out = []
        for v in s.tolist():
            try:
                parts = v.split()
                if len(parts) < 2:
                    out.append(pd.NaT);
                    continue
                dpart = parts[0]
                tpart = parts[1]
                tt = tpart.split(":")
                if len(tt) != 3:
                    out.append(pd.NaT);
                    continue
                hh = int(tt[0]);
                mm = int(tt[1])
                secms = tt[2].strip()  # e.g. "33772"
                ss = int(secms[:2]) if len(secms) >= 2 else 0
                ms_str = secms[2:] if len(secms) > 2 else ""
                ms = int(ms_str) if (ms_str.isdigit() and ms_str != "") else 0
                if ms >= 1000:
                    ms = int(ms_str[:3]) if len(ms_str) >= 3 else ms % 1000

                dd, mon, yy = dpart.split("/")
                out.append(datetime(int(yy), int(mon), int(dd), hh, mm, ss, int(ms) * 1000))
            except Exception:
                out.append(pd.NaT)

        return pd.to_datetime(pd.Series(out), errors="coerce")


    def _list_logs():
        out = []
        try:
            for base, _, files in os.walk(LOGS_DIR):
                for fn in files:
                    if fn.lower().endswith(".csv"):
                        out.append(os.path.join(base, fn))
        except Exception:
            pass
        return out


    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)


    # UI controls
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        dt_cap_s = st.number_input("dt cap (sec)", min_value=0.1, step=0.1, value=2.0, key="argon_monthly_dt_cap")
    with g2:
        last_days = st.number_input("Scan last N days (0=all)", min_value=0, step=1, value=365,
                                    key="argon_monthly_scan_days")
    with g3:
        force = st.button("♻️ Force rebuild", use_container_width=True, key="argon_monthly_force")

    state = _read_json(STATE_JSON, {"last_scan_mtime": 0.0, "last_run": ""})


    def _build_monthly_csv(force_rebuild: bool = False):
        logs = _list_logs()
        if not logs:
            return {"info": f"No logs found in `{LOGS_DIR}`"}

        if int(last_days) > 0:
            cutoff = datetime.now() - timedelta(days=int(last_days))
            keep = []
            for p in logs:
                try:
                    if datetime.fromtimestamp(os.path.getmtime(p)) >= cutoff:
                        keep.append(p)
                except Exception:
                    pass
            logs = keep

        newest_mtime = 0.0
        for p in logs:
            try:
                newest_mtime = max(newest_mtime, os.path.getmtime(p))
            except Exception:
                pass

        if (not force_rebuild) and newest_mtime <= float(state.get("last_scan_mtime", 0.0)):
            return {"info": "Up-to-date (no new logs detected)."}

        # accum by YYYY-MM
        accum = {}  # month -> dict

        for lp in logs:
            try:
                df = pd.read_csv(lp)
            except Exception:
                continue

            if TIME_COL not in df.columns:
                continue

            if not any(c in df.columns for c in MFC_ACTUAL_COLS):
                continue

            df["_t"] = _parse_dt_series_date_time(df[TIME_COL])
            df = df.dropna(subset=["_t"]).sort_values("_t")
            if len(df) < 3:
                continue

            # Argon flow = sum of MFCs (SLPM)
            flow = None
            for c in MFC_ACTUAL_COLS:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    flow = s if flow is None else (flow + s)
            if flow is None:
                continue

            dt_s = df["_t"].diff().dt.total_seconds().fillna(0.0)
            dt_s = dt_s.clip(lower=0.0, upper=float(dt_cap_s))
            dt_min = dt_s / 60.0

            month_key = df["_t"].dt.strftime("%Y-%m")
            work = pd.DataFrame({"month": month_key, "flow": flow, "dt_min": dt_min})
            work["SL"] = work["flow"] * work["dt_min"]

            for m, g in work.groupby("month", dropna=True):
                if m not in accum:
                    accum[m] = {
                        "total_SL": 0.0,
                        "total_minutes": 0.0,
                        "min_slpm": None,
                        "max_slpm": None,
                        "sum_flow_weighted": 0.0,  # flow * minutes
                        "logs": set(),
                        "rows": 0,
                    }

                total_sl = float(g["SL"].sum())
                total_min = float(g["dt_min"].sum())

                accum[m]["total_SL"] += total_sl
                accum[m]["total_minutes"] += total_min
                accum[m]["sum_flow_weighted"] += float((g["flow"] * g["dt_min"]).sum())
                mn = float(g["flow"].min())
                mx = float(g["flow"].max())
                accum[m]["min_slpm"] = mn if accum[m]["min_slpm"] is None else min(accum[m]["min_slpm"], mn)
                accum[m]["max_slpm"] = mx if accum[m]["max_slpm"] is None else max(accum[m]["max_slpm"], mx)
                accum[m]["logs"].add(lp)
                accum[m]["rows"] += int(len(g))

        if not accum:
            return {"info": "No valid data found in scanned logs."}

        # Build CSV
        rows = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for m, a in accum.items():
            avg_slpm = (a["sum_flow_weighted"] / a["total_minutes"]) if a["total_minutes"] > 0 else 0.0
            rows.append({
                "month": m,
                "gas": "Argon",
                "total_standard_liters": float(a["total_SL"]),
                "total_minutes": float(a["total_minutes"]),
                "avg_slpm": float(avg_slpm),
                "min_slpm": float(a["min_slpm"] if a["min_slpm"] is not None else 0.0),
                "max_slpm": float(a["max_slpm"] if a["max_slpm"] is not None else 0.0),
                "logs_count": int(len(a["logs"])),
                "rows_used": int(a["rows"]),
                "updated_at": now_str
            })

        out_df = pd.DataFrame(rows).sort_values("month", ascending=False)
        out_df.to_csv(REPORT_CSV, index=False)

        state["last_scan_mtime"] = float(newest_mtime)
        state["last_run"] = now_str
        _write_json(STATE_JSON, state)

        return {"updated": True, "rows": len(out_df), "logs_scanned": len(logs)}


    # ✅ AUTO build on load
    info = _build_monthly_csv(force_rebuild=bool(force))

    if "info" in info:
        st.caption(info["info"])
    else:
        st.caption(f"AUTO updated ✅ | logs scanned: {info.get('logs_scanned', 0)} | months: {info.get('rows', 0)}")

    # View report
    if os.path.exists(REPORT_CSV):
        try:
            rep = pd.read_csv(REPORT_CSV)
            st.dataframe(rep, use_container_width=True, hide_index=True)
            with open(REPORT_CSV, "r") as f:
                st.download_button(
                    "⬇️ Download monthly CSV",
                    data=f.read(),
                    file_name=os.path.basename(REPORT_CSV),
                    mime="text/csv",
                    use_container_width=True,
                    key="argon_monthly_download"
                )
        except Exception as e:
            st.error(f"Failed to load report CSV: {e}")
    else:
        st.info("Monthly report CSV not created yet.")

    st.markdown("</div>", unsafe_allow_html=True)
# ------------------ Schedule Tab ------------------
elif tab_selection == "📅 Schedule":
    import os
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    st.title("📅 Tower Schedule")

    SCHEDULE_FILE = P.schedule_csv
    required_columns = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    # =========================================================
    # Ensure schedule file exists + required columns
    # =========================================================
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=required_columns).to_csv(SCHEDULE_FILE, index=False)
        st.warning("Schedule file was missing. New file with required columns created.")

    schedule_df = pd.read_csv(SCHEDULE_FILE)

    # Ensure required columns exist
    missing_columns = [c for c in required_columns if c not in schedule_df.columns]
    for c in missing_columns:
        schedule_df[c] = ""

    # Enforce column order and persist (keeps file stable)
    schedule_df = schedule_df[required_columns]
    schedule_df.to_csv(SCHEDULE_FILE, index=False)

    # Parse datetimes safely
    schedule_df["Start DateTime"] = pd.to_datetime(schedule_df["Start DateTime"], errors="coerce")
    schedule_df["End DateTime"] = pd.to_datetime(schedule_df["End DateTime"], errors="coerce")

    # Clean leaked Plotly template text from Description
    schedule_df["Description"] = (
        schedule_df["Description"]
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Description=", "", regex=False)
        .str.strip()
    )

    # Normalize recurrence display in MASTER (so empty shows "None")
    def _norm_recur(v) -> str:
        r = str(v).strip()
        return "None" if r in ["", "None", "none", "NONE", "nan", "NaN"] else r

    schedule_df["Recurrence"] = schedule_df["Recurrence"].apply(_norm_recur)

    # =========================================================
    # Quick range presets (SAFE: apply before widgets instantiate)
    # =========================================================
    if "schedule_apply_preset" not in st.session_state:
        st.session_state["schedule_apply_preset"] = None  # None / "w1" / "m1" / "m3"

    preset = st.session_state.get("schedule_apply_preset")
    if preset:
        today = pd.Timestamp.now().date()
        if preset == "w1":
            st.session_state["schedule_start_date"] = today
            st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(weeks=1)).date()
        elif preset == "m1":
            st.session_state["schedule_start_date"] = today
            st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(months=1)).date()
        elif preset == "m3":
            st.session_state["schedule_start_date"] = today
            st.session_state["schedule_end_date"] = (pd.Timestamp.now() + pd.DateOffset(months=3)).date()

        st.session_state["schedule_apply_preset"] = None

    # =========================================================
    # Main-page Filters
    # =========================================================
    st.subheader("🗓️ View Range")

    f1, f2, f3 = st.columns([1.1, 1.1, 1.4])

    with f1:
        start_filter = st.date_input(
            "Start Date",
            value=st.session_state.get("schedule_start_date", pd.Timestamp.now().date()),
            key="schedule_start_date",
        )

    with f2:
        end_filter = st.date_input(
            "End Date",
            value=st.session_state.get("schedule_end_date", (pd.Timestamp.now() + pd.DateOffset(weeks=1)).date()),
            key="schedule_end_date",
        )

    with f3:
        st.markdown("#### Quick ranges")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("1 Week", use_container_width=True, key="sched_preset_w1"):
                st.session_state["schedule_apply_preset"] = "w1"
                st.rerun()
        with b2:
            if st.button("1 Month", use_container_width=True, key="sched_preset_m1"):
                st.session_state["schedule_apply_preset"] = "m1"
                st.rerun()
        with b3:
            if st.button("3 Months", use_container_width=True, key="sched_preset_m3"):
                st.session_state["schedule_apply_preset"] = "m3"
                st.rerun()

    range_start = pd.to_datetime(start_filter)
    range_end = pd.to_datetime(end_filter) + pd.to_timedelta(1, unit="day")  # include end day

    base = schedule_df.dropna(subset=["Start DateTime", "End DateTime"]).copy()

    # =========================================================
    # Expand recurring events so they "show all"
    # =========================================================
    def _next_dt(dt: pd.Timestamp, recurrence: str) -> pd.Timestamp:
        r = str(recurrence).strip().lower()
        if r == "weekly":
            return dt + pd.DateOffset(weeks=1)
        if r == "monthly":
            return dt + pd.DateOffset(months=1)
        if r == "yearly":
            return dt + pd.DateOffset(years=1)
        return dt

    expanded_rows = []
    for _, row in base.iterrows():
        rec = _norm_recur(row.get("Recurrence", "None"))
        start_dt = row["Start DateTime"]
        end_dt = row["End DateTime"]

        if pd.isna(start_dt) or pd.isna(end_dt):
            continue

        # If no recurrence -> keep single
        if rec == "None":
            rdict = row.to_dict()
            rdict["Recurrence"] = "None"
            expanded_rows.append(rdict)
            continue

        duration = end_dt - start_dt
        occ_start = start_dt
        occ_end = occ_start + duration

        safety = 0
        while occ_end < range_start and safety < 5000:
            occ_start = _next_dt(occ_start, rec)
            occ_end = occ_start + duration
            safety += 1

        safety = 0
        while occ_start <= range_end and safety < 5000:
            new_row = row.to_dict()
            new_row["Start DateTime"] = occ_start
            new_row["End DateTime"] = occ_end
            new_row["Recurrence"] = rec  # keep as display-ready
            expanded_rows.append(new_row)

            occ_start = _next_dt(occ_start, rec)
            occ_end = occ_start + duration
            safety += 1

    expanded_df = pd.DataFrame(expanded_rows)
    if not expanded_df.empty:
        expanded_df["Start DateTime"] = pd.to_datetime(expanded_df["Start DateTime"], errors="coerce")
        expanded_df["End DateTime"] = pd.to_datetime(expanded_df["End DateTime"], errors="coerce")
        expanded_df = expanded_df.dropna(subset=["Start DateTime", "End DateTime"])

    # =========================================================
    # Filter by overlap (on expanded)
    # =========================================================
    if expanded_df.empty:
        filtered_schedule = expanded_df
    else:
        filtered_schedule = expanded_df[
            (expanded_df["End DateTime"] >= range_start) &
            (expanded_df["Start DateTime"] <= range_end)
        ].copy()

    # =========================================================
    # Timeline (FIXED hover formatting)
    #   px.timeline does NOT support %{x_end|...} in hovertemplate.
    #   So we precompute formatted strings and show them via custom_data.
    # =========================================================
    st.subheader("📈 Timeline")

    event_colors = {
        "Maintenance": "blue",
        "Drawing": "green",
        "Stop": "red",
        "Management Event": "purple",
    }

    if not filtered_schedule.empty:
        # Precompute clean display strings
        filtered_schedule["StartStr"] = filtered_schedule["Start DateTime"].dt.strftime("%Y-%m-%d %H:%M")
        filtered_schedule["EndStr"] = filtered_schedule["End DateTime"].dt.strftime("%Y-%m-%d %H:%M")
        filtered_schedule["RecurrenceDisp"] = filtered_schedule["Recurrence"].apply(_norm_recur)

        # Also ensure description is clean for hover
        filtered_schedule["Description"] = (
            filtered_schedule["Description"]
            .astype(str)
            .str.replace(r"%\{.*?\}", "", regex=True)
            .str.replace("Description=", "", regex=False)
            .str.strip()
        )

        fig = px.timeline(
            filtered_schedule,
            x_start="Start DateTime",
            x_end="End DateTime",
            y="Event Type",
            color="Event Type",
            color_discrete_map=event_colors,
            custom_data=["StartStr", "EndStr", "RecurrenceDisp", "Description"],
            title="Tower Schedule",
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Start: %{customdata[0]}<br>"
                "End: %{customdata[1]}<br>"
                "Recurrence: %{customdata[2]}<br>"
                "Description: %{customdata[3]}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events in the selected date range.")

    st.divider()

    # =========================================================
    # Management area (Main page, no sidebar)
    # =========================================================
    st.subheader("🧩 Manage Schedule")

    left, right = st.columns([1.05, 0.95], gap="large")

    # -----------------------------
    # LEFT: Master table editor
    # -----------------------------
    with left:
        st.write("### Current Schedule (Master)")
        st.caption("Master stores recurrence once; timeline shows expanded occurrences.")
        st.data_editor(schedule_df, height=320, use_container_width=True, key="sched_master_editor")

        csave, creload = st.columns([1, 1])
        with csave:
            if st.button("💾 Save Master Table", use_container_width=True, key="sched_save_master"):
                edited = st.session_state.get("sched_master_editor", schedule_df)

                for c in required_columns:
                    if c not in edited.columns:
                        edited[c] = ""
                edited = edited[required_columns].copy()

                edited["Start DateTime"] = pd.to_datetime(edited["Start DateTime"], errors="coerce")
                edited["End DateTime"] = pd.to_datetime(edited["End DateTime"], errors="coerce")

                edited["Description"] = (
                    edited["Description"]
                    .astype(str)
                    .str.replace(r"%\{.*?\}", "", regex=True)
                    .str.replace("Description=", "", regex=False)
                    .str.strip()
                )

                edited["Recurrence"] = edited["Recurrence"].apply(_norm_recur)

                edited.to_csv(SCHEDULE_FILE, index=False)
                st.success("Saved master schedule.")
                st.rerun()

        with creload:
            if st.button("🔄 Reload From File", use_container_width=True, key="sched_reload"):
                st.rerun()

    # -----------------------------
    # RIGHT: Add / Delete
    # -----------------------------
    with right:
        with st.expander("➕ Add New Event", expanded=True):
            event_type = st.selectbox(
                "Event Type",
                ["Maintenance", "Drawing", "Stop", "Management Event"],
                key="sched_type",
            )
            event_description = st.text_area("Description", key="sched_desc", height=90)

            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input("Start Date", pd.Timestamp.now().date(), key="sched_start_date2")
                start_time = st.time_input("Start Time", key="sched_start_time")
            with d2:
                end_date = st.date_input("End Date", pd.Timestamp.now().date(), key="sched_end_date2")
                end_time = st.time_input("End Time", key="sched_end_time")

            recurrence = st.selectbox("Recurrence", ["None", "Weekly", "Monthly", "Yearly"], key="sched_recur")

            start_datetime = pd.to_datetime(f"{start_date} {start_time}")
            end_datetime = pd.to_datetime(f"{end_date} {end_time}")

            if end_datetime < start_datetime:
                st.warning("End DateTime is before Start DateTime.")

            if st.button("Add Event", use_container_width=True, key="sched_add_btn"):
                new_event = pd.DataFrame([{
                    "Event Type": event_type,
                    "Start DateTime": start_datetime,
                    "End DateTime": end_datetime,
                    "Description": str(event_description).strip(),
                    "Recurrence": _norm_recur(recurrence),
                }])

                full_schedule_df = pd.read_csv(SCHEDULE_FILE)
                for c in required_columns:
                    if c not in full_schedule_df.columns:
                        full_schedule_df[c] = ""
                full_schedule_df = full_schedule_df[required_columns]

                full_schedule_df = pd.concat([full_schedule_df, new_event], ignore_index=True)
                full_schedule_df.to_csv(SCHEDULE_FILE, index=False)
                st.success("Event added to schedule!")
                st.rerun()

        with st.expander("🗑️ Delete Event", expanded=False):
            if schedule_df.empty:
                st.info("No events available for deletion.")
            else:
                # show Recurrence too (helps users pick the right one)
                delete_options = [
                    f"{i}: {schedule_df.loc[i, 'Event Type']} | "
                    f"{schedule_df.loc[i, 'Start DateTime']} | "
                    f"{_norm_recur(schedule_df.loc[i, 'Recurrence'])} | "
                    f"{str(schedule_df.loc[i, 'Description'])[:60]}"
                    for i in schedule_df.index
                ]
                to_delete = st.selectbox("Select event to delete", delete_options, key="sched_del_select")
                del_idx = int(to_delete.split(":")[0])

                if st.button("Delete Selected Event", use_container_width=True, key="sched_del_btn"):
                    schedule_df2 = schedule_df.drop(index=del_idx).reset_index(drop=True)
                    schedule_df2.to_csv(SCHEDULE_FILE, index=False)
                    st.success("Event deleted successfully!")
                    st.rerun()
# ------------------ Order draw ------------------
elif tab_selection == "📦 Order Draw":
    st.title("📦 Order Draw")

    import os
    import datetime as dt
    import pandas as pd
    import streamlit as st
    import json

    orders_file = P.orders_csv
    SCHEDULE_FILE = P.schedule_csv
    schedule_required_cols = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    SCHEDULE_PASSWORD = "DORON"

    GOOD_ZONES_COL = "Good Zones Count (required length zones)"
    FIBER_GEOMETRY_COL = "Fiber Geometry Type"

    SAP_INVENTORY_FILE = "sap_rods_inventory.csv"

    PROJECTS_FILE = P.projects_fiber_csv
    PROJECTS_COL = "Fiber Project"
    PROJECT_TEMPLATES_FILE = P.projects_fiber_templates_csv

    # ✅ coating temperature columns
    MAIN_COAT_TEMP_COL = "Main Coating Temperature (°C)"
    SEC_COAT_TEMP_COL = "Secondary Coating Temperature (°C)"

    # ✅ geometry-specific columns
    TIGER_CUT_COL = "Tiger Cut (%)"
    OCT_F2F_COL = "Octagonal F2F (mm)"

    # ✅ config_coating.json path (coatings list must match this!)
    COATING_CFG_PATH = P.coating_config_json

    # ✅ tolerance columns
    FIBER_D_TOL_COL = "Fiber Diameter Tol (± µm)"
    MAIN_D_TOL_COL = "Main Coating Diameter Tol (± µm)"
    SEC_D_TOL_COL = "Secondary Coating Diameter Tol (± µm)"

    FIBER_GEOMETRY_OPTIONS = [
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

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------


    def __from_state(key: str, default=0.0) -> float:
        return to_float(st.session_state.get(key, default), default)


    def _fmt_pm(val: float, tol: float, unit: str = "µm") -> str:
        try:
            val = float(val)
            tol = float(tol)
        except Exception:
            return ""
        if val <= 0:
            return ""
        if tol > 0:
            return f"{val:g} ± {tol:g} {unit}"
        return f"{val:g} {unit}"

    # ---------------------------------------------------------
    # Load coating options from config_coating.json
    # ---------------------------------------------------------
    def load_config_coating_json(path: str = COATING_CFG_PATH) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def coating_options_from_cfg(cfg: dict) -> list:
        coats = (cfg or {}).get("coatings", {})
        if isinstance(coats, dict):
            return [str(k).strip() for k in coats.keys() if str(k).strip() != ""]
        return []


    coating_cfg = load_coating_config()
    COATING_OPTIONS = coating_options_from_cfg(coating_cfg) or [""]
    if not COATING_OPTIONS:
        st.warning("⚠️ No coatings found in config_coating.json → using empty list.")
        COATING_OPTIONS = []

    # ---------------------------------------------------------
    # Ensure schedule file exists
    # ---------------------------------------------------------
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=schedule_required_cols).to_csv(SCHEDULE_FILE, index=False)

    # ---------------------------------------------------------
    # Projects list helpers
    # ---------------------------------------------------------
    def ensure_projects_file():
        if not os.path.exists(PROJECTS_FILE):
            pd.DataFrame(columns=[PROJECTS_COL]).to_csv(PROJECTS_FILE, index=False)

    def load_projects() -> list:
        ensure_projects_file()
        try:
            d = pd.read_csv(PROJECTS_FILE, keep_default_na=False)
        except Exception:
            return []
        if PROJECTS_COL not in d.columns:
            return []
        items = (
            d[PROJECTS_COL].astype(str)
            .replace({"nan": "", "None": ""})
            .fillna("")
            .map(lambda x: x.strip())
        )
        items = [x for x in items.tolist() if x]
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def add_project(new_name: str):
        new_name = str(new_name or "").strip()
        if not new_name:
            return False, "Project name is empty."
        ensure_projects_file()
        existing = load_projects()
        if new_name in existing:
            return False, "Project already exists."
        dfp = pd.read_csv(PROJECTS_FILE, keep_default_na=False) if os.path.exists(PROJECTS_FILE) else pd.DataFrame()
        if PROJECTS_COL not in dfp.columns:
            dfp[PROJECTS_COL] = ""
        dfp = pd.concat([dfp, pd.DataFrame([{PROJECTS_COL: new_name}])], ignore_index=True)
        dfp.to_csv(PROJECTS_FILE, index=False)
        return True, f"Added project: {new_name}"

    # ---------------------------------------------------------
    # Project template helpers (includes tolerances)
    # ---------------------------------------------------------
    TEMPLATE_FIELDS = [
        PROJECTS_COL,
        FIBER_GEOMETRY_COL,
        TIGER_CUT_COL,
        OCT_F2F_COL,
        "Fiber Diameter (µm)",
        FIBER_D_TOL_COL,
        "Main Coating Diameter (µm)",
        MAIN_D_TOL_COL,
        "Secondary Coating Diameter (µm)",
        SEC_D_TOL_COL,
        "Tension (g)",
        "Draw Speed (m/min)",
        "Main Coating",
        "Secondary Coating",
        MAIN_COAT_TEMP_COL,
        SEC_COAT_TEMP_COL,
        "Notes Default",
    ]

    TEMPLATE_TO_WIDGET_KEY = {
        FIBER_GEOMETRY_COL: "order_fiber_geometry_required",
        TIGER_CUT_COL: "order_tiger_cut_pct",
        OCT_F2F_COL: "order_oct_f2f_mm",
        "Fiber Diameter (µm)": "order_fiber_diam",
        FIBER_D_TOL_COL: "order_fiber_diam_tol",
        "Main Coating Diameter (µm)": "order_main_diam",
        MAIN_D_TOL_COL: "order_main_diam_tol",
        "Secondary Coating Diameter (µm)": "order_sec_diam",
        SEC_D_TOL_COL: "order_sec_diam_tol",
        "Tension (g)": "order_tension",
        "Draw Speed (m/min)": "order_speed",
        "Main Coating": "order_coating_main",
        "Secondary Coating": "order_coating_secondary",
        MAIN_COAT_TEMP_COL: "order_main_coat_temp_c",
        SEC_COAT_TEMP_COL: "order_sec_coat_temp_c",
        "Notes Default": "order_notes",
    }

    NUMERIC_WIDGET_KEYS = {
        "order_fiber_diam",
        "order_fiber_diam_tol",
        "order_main_diam",
        "order_main_diam_tol",
        "order_sec_diam",
        "order_sec_diam_tol",
        "order_tension",
        "order_speed",
        "order_main_coat_temp_c",
        "order_sec_coat_temp_c",
        "order_tiger_cut_pct",
        "order_oct_f2f_mm",
    }

    def ensure_templates_file():
        if not os.path.exists(PROJECT_TEMPLATES_FILE):
            pd.DataFrame(columns=TEMPLATE_FIELDS).to_csv(PROJECT_TEMPLATES_FILE, index=False)

    def load_templates_df() -> pd.DataFrame:
        ensure_templates_file()
        try:
            d = pd.read_csv(PROJECT_TEMPLATES_FILE, keep_default_na=False)
        except Exception:
            d = pd.DataFrame(columns=TEMPLATE_FIELDS)
        for c in TEMPLATE_FIELDS:
            if c not in d.columns:
                d[c] = ""
        return d[TEMPLATE_FIELDS].copy()

    def get_template_for_project(project_name: str) -> dict:
        project_name = str(project_name or "").strip()
        if not project_name:
            return {}
        d = load_templates_df()
        m = d[PROJECTS_COL].astype(str).str.strip() == project_name
        if not m.any():
            return {}
        return d.loc[m].iloc[-1].to_dict()

    def save_or_update_template(project_name: str, template_payload: dict):
        project_name = str(project_name or "").strip()
        if not project_name:
            return False, "No project selected."

        d = load_templates_df()
        m = d[PROJECTS_COL].astype(str).str.strip() == project_name

        row = {k: "" for k in TEMPLATE_FIELDS}
        row[PROJECTS_COL] = project_name
        for k, v in (template_payload or {}).items():
            if k in row:
                row[k] = v

        if m.any():
            d.loc[m, :] = pd.DataFrame([row]).iloc[0].values
        else:
            d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)

        d.to_csv(PROJECT_TEMPLATES_FILE, index=False)
        return True, f"✅ Template saved for project: {project_name}"

    def apply_template_to_form(project_name: str):
        tpl = get_template_for_project(project_name)
        if not tpl:
            return False

        for col, widget_key in TEMPLATE_TO_WIDGET_KEY.items():
            val = tpl.get(col, "")
            if widget_key in NUMERIC_WIDGET_KEYS:
                num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                st.session_state[widget_key] = float(num) if pd.notna(num) else 0.0
            else:
                st.session_state[widget_key] = str(val)
        return True

    # ---------------------------------------------------------
    # Auto-apply state
    # ---------------------------------------------------------
    if "order_last_project_applied" not in st.session_state:
        st.session_state["order_last_project_applied"] = ""

    # ---------------------------------------------------------
    # SAP inventory helper (read-only)
    # ---------------------------------------------------------
    def get_sap_rods_set_count() -> float:
        if not os.path.exists(SAP_INVENTORY_FILE):
            return 0.0
        try:
            inv = pd.read_csv(SAP_INVENTORY_FILE, keep_default_na=False)
        except Exception:
            return 0.0
        if inv.empty or "Item" not in inv.columns or "Count" not in inv.columns:
            return 0.0
        m = inv["Item"].astype(str).str.strip().str.lower() == "sap rods set"
        if not m.any():
            return 0.0
        val = inv.loc[m, "Count"].iloc[-1]
        num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        return float(num) if pd.notna(num) else 0.0

    def render_sap_inventory_banner():
        sap_cnt = get_sap_rods_set_count()
        if sap_cnt < 1:
            st.warning(f"⚠️ SAP Rods Set inventory is LOW: **{sap_cnt:g}** sets (under 1).")
        else:
            st.success(f"🧪 SAP Rods Set inventory available: **{sap_cnt:g}** sets.")

    # =========================================================
    # 1) TABLE FIRST (with colors)
    # =========================================================
    st.subheader("📋 Existing Draw Orders")

    if not os.path.exists(orders_file):
        st.info("No orders submitted yet.")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(orders_file, keep_default_na=False)

    if not df.empty:
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        for col, default in {
            "Status": "Pending",
            "Priority": "Normal",
            PROJECTS_COL: "",
            "Order Opener": "",
            "Preform Number": "",
            FIBER_GEOMETRY_COL: "",
            TIGER_CUT_COL: "",
            OCT_F2F_COL: "",
            "Done CSV": "",
            "Done Description": "",
            "Active CSV": "",
            "T&M Moved": False,
            "T&M Moved Timestamp": "",
            "Required Length (m) (for T&M+costumer)": "",
            GOOD_ZONES_COL: "",
            "Notes": "",
            "Main Coating": "",
            "Secondary Coating": "",
            MAIN_COAT_TEMP_COL: "",
            SEC_COAT_TEMP_COL: "",
            "Fiber Diameter (µm)": "",
            FIBER_D_TOL_COL: "",
            "Main Coating Diameter (µm)": "",
            MAIN_D_TOL_COL: "",
            "Secondary Coating Diameter (µm)": "",
            SEC_D_TOL_COL: "",
            "Tension (g)": "",
            "Draw Speed (m/min)": "",
        }.items():
            if col not in df.columns:
                df[col] = default

        if "T&M Moved" in df.columns:
            df["T&M Moved"] = df["T&M Moved"].apply(
                lambda x: str(x).strip().lower() in ("true", "1", "yes", "y", "moved")
            )
        df_visible = df[~df["T&M Moved"]].copy() if "T&M Moved" in df.columns else df.copy()


        styled_df = (
            df_visible.style
            .applymap(color_status, subset=["Status"] if "Status" in df_visible.columns else None)
            .applymap(color_priority, subset=["Priority"] if "Priority" in df_visible.columns else None)
        )

        st.dataframe(styled_df, use_container_width=True)
    else:
        df_visible = pd.DataFrame()

    # =========================================================
    # ✅ Pending → Schedule (quick)
    # =========================================================
    st.markdown("---")
    st.subheader("🕒 Pending → Schedule (quick)")

    if df_visible is None or df_visible.empty:
        st.info("No orders to schedule.")
    else:
        df_pending = df_visible[df_visible["Status"].astype(str).str.strip() == "Pending"].copy()

        if df_pending.empty:
            st.info("No Pending orders.")
        else:
            pending_indices = df_pending.index.tolist()

            def _fmt_pending(i: int) -> str:
                try:
                    prj = str(df_pending.loc[i, PROJECTS_COL]).strip()
                    pref = str(df_pending.loc[i, "Preform Number"]).strip()
                    pri = str(df_pending.loc[i, "Priority"]).strip()
                    ts = df_pending.loc[i, "Timestamp"] if "Timestamp" in df_pending.columns else ""
                    return f"#{i} | {prj} | Preform: {pref} | Priority: {pri} | {ts}"
                except Exception:
                    return f"#{i}"

            selected_idx = st.selectbox(
                "Select Pending order",
                options=pending_indices,
                format_func=_fmt_pending,
                key="pending_to_schedule_selectbox",
            )

            sel_row = df.loc[selected_idx]  # ORIGINAL df row

            with st.expander("📅 Schedule selected Pending order", expanded=True):
                preform_now = str(sel_row.get("Preform Number", "")).strip()
                need_preform = (preform_now == "" or preform_now == "0" or preform_now.lower() == "none")

                preform_real = ""
                if need_preform:
                    preform_real = st.text_input(
                        "Preform Number (required for scheduling — cannot be 0)",
                        placeholder="e.g., P0888",
                        key="pending_sched_real_preform_input",
                    )

                pwd2 = st.text_input("Scheduling password", type="password", key="pending_sched_pwd2")
                sched_ok2 = (pwd2 == SCHEDULE_PASSWORD)
                if pwd2.strip():
                    (st.success if sched_ok2 else st.error)("Password OK ✅" if sched_ok2 else "Wrong password ❌")

                default_date2 = pd.Timestamp.today().date()
                preset2 = st.radio(
                    "Preset",
                    ["All day (08:00–16:00)", "Before lunch (08:00–12:00)", "After lunch (12:00–16:00)"],
                    horizontal=True,
                    key="pending_sched_preset2",
                    label_visibility="collapsed",
                )

                if preset2.startswith("All day"):
                    preset_start2 = dt.time(8, 0)
                    preset_duration2 = 8 * 60
                elif preset2.startswith("Before lunch"):
                    preset_start2 = dt.time(8, 0)
                    preset_duration2 = 4 * 60
                else:
                    preset_start2 = dt.time(12, 0)
                    preset_duration2 = 4 * 60

                cA2, cB2, cC2 = st.columns([1, 1, 1], vertical_alignment="bottom")
                with cA2:
                    sched_date2 = st.date_input("Schedule Date", value=default_date2, key="pending_sched_date2")
                with cB2:
                    sched_start2 = st.time_input("Start Time", value=preset_start2, key="pending_sched_start2")
                with cC2:
                    sched_dur2 = st.number_input(
                        "Duration (min)",
                        min_value=1,
                        step=5,
                        value=int(preset_duration2),
                        key="pending_sched_dur2",
                    )

                start_dt2 = pd.to_datetime(f"{sched_date2} {sched_start2}")
                end_dt2 = start_dt2 + pd.to_timedelta(int(sched_dur2), unit="m")

                if st.button("✅ Schedule this Pending Order", key="pending_schedule_confirm_btn"):
                    if not sched_ok2:
                        st.error("Not scheduled: password missing/wrong.")
                        st.stop()

                    if need_preform and not str(preform_real).strip():
                        st.error("Please enter a real Preform Number (cannot schedule with 0).")
                        st.stop()

                    existing2 = pd.read_csv(SCHEDULE_FILE) if os.path.exists(SCHEDULE_FILE) else pd.DataFrame()
                    for c in schedule_required_cols:
                        if c not in existing2.columns:
                            existing2[c] = ""
                    existing2 = existing2[schedule_required_cols]

                    geom2 = str(sel_row.get(FIBER_GEOMETRY_COL, "")).strip()
                    prj2 = str(sel_row.get(PROJECTS_COL, "")).strip()
                    pri2 = str(sel_row.get("Priority", "")).strip()
                    pref2 = str(preform_real).strip() if need_preform else preform_now

                    length2 = sel_row.get("Required Length (m) (for T&M+costumer)", "")
                    zones2 = sel_row.get(GOOD_ZONES_COL, "")

                    tiger2 = to_float(sel_row.get(TIGER_CUT_COL, 0.0), 0.0)
                    oct2 = to_float(sel_row.get(OCT_F2F_COL, 0.0), 0.0)

                    mtemp2 = to_float(sel_row.get(MAIN_COAT_TEMP_COL, 0.0), 0.0)
                    stemp2 = to_float(sel_row.get(SEC_COAT_TEMP_COL, 0.0), 0.0)

                    notes2 = str(sel_row.get("Notes", "")).strip()

                    fd2 = to_float(sel_row.get("Fiber Diameter (µm)", 0.0), 0.0)
                    md2 = to_float(sel_row.get("Main Coating Diameter (µm)", 0.0), 0.0)
                    sd2 = to_float(sel_row.get("Secondary Coating Diameter (µm)", 0.0), 0.0)
                    fdt2 = to_float(sel_row.get(FIBER_D_TOL_COL, 0.0), 0.0)
                    mdt2 = to_float(sel_row.get(MAIN_D_TOL_COL, 0.0), 0.0)
                    sdt2 = to_float(sel_row.get(SEC_D_TOL_COL, 0.0), 0.0)

                    diam_bits2 = []
                    s_fd2 = _fmt_pm(fd2, fdt2)
                    s_md2 = _fmt_pm(md2, mdt2)
                    s_sd2 = _fmt_pm(sd2, sdt2)
                    if s_fd2:
                        diam_bits2.append(f"Fiber {s_fd2}")
                    if s_md2:
                        diam_bits2.append(f"Coat1 {s_md2}")
                    if s_sd2:
                        diam_bits2.append(f"Coat2 {s_sd2}")

                    desc_lines2 = [
                        f"ORDER #{selected_idx} | Priority: {pri2}",
                        f"Fiber: {prj2} | Geometry: {geom2} | Preform: {pref2}",
                        f"Required Length: {length2} m | Good Zones Count: {zones2}",
                    ]
                    if diam_bits2:
                        desc_lines2.append("Diameters: " + " | ".join(diam_bits2))

                    if geom2 == "TIGER - PM" and tiger2 > 0:
                        desc_lines2.append(f"Tiger Cut: {tiger2:.1f}%")
                    if geom2 == "Octagonal" and oct2 > 0:
                        desc_lines2.append(f"Oct F2F: {oct2:.2f} mm")

                    if mtemp2 > 0:
                        desc_lines2.append(f"Main Coat Temp: {mtemp2:.0f}°C")
                    if stemp2 > 0:
                        desc_lines2.append(f"Sec Coat Temp: {stemp2:.0f}°C")
                    if notes2:
                        desc_lines2.append(f"Notes: {notes2}")

                    event_description2 = " | ".join([x for x in desc_lines2 if str(x).strip()])

                    new_event2 = pd.DataFrame([{
                        "Event Type": "Drawing",
                        "Start DateTime": start_dt2,
                        "End DateTime": end_dt2,
                        "Description": event_description2,
                        "Recurrence": "None",
                    }])

                    pd.concat([existing2, new_event2], ignore_index=True).to_csv(SCHEDULE_FILE, index=False)

                    if need_preform:
                        df.at[selected_idx, "Preform Number"] = pref2
                    df.at[selected_idx, "Status"] = "Scheduled"
                    df.to_csv(orders_file, index=False)

                    st.success("✅ Scheduled + moved Status to Scheduled.")
                    st.rerun()

    # =========================================================
    # 2) CREATE NEW ORDER (UI + tolerances + notes recommended)
    #   IMPORTANT FIX: Schedule UI is OUTSIDE the form
    # =========================================================
    st.markdown("---")
    st.markdown("### ➕ Create New Order")

    if "show_new_order_form" not in st.session_state:
        st.session_state["show_new_order_form"] = False

    show_form = st.checkbox(
        "Create new order",
        value=bool(st.session_state["show_new_order_form"]),
        key="order_create_new_cb",
    )
    st.session_state["show_new_order_form"] = bool(show_form)

    if st.session_state["show_new_order_form"]:
        st.markdown(
            """
            <style>
            div[data-testid="stForm"] { padding-top: 0.25rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            projects = load_projects()

            # Project row (outside form to support template auto-apply)
            selA, selB = st.columns([2.4, 1.0], vertical_alignment="bottom")
            with selA:
                selected_project = st.selectbox(
                    "Project * (auto-fills if template exists)",
                    options=[""] + projects,
                    index=0,
                    key="order_project_select",
                    placeholder="Select project...",
                )
            with selB:
                with st.popover("➕ Add project", use_container_width=True):
                    new_proj = st.text_input("New project name", key="order_new_project_name")
                    if st.button("Add", key="order_add_project_btn", use_container_width=True):
                        okp, msgp = add_project(new_proj)
                        (st.success if okp else st.warning)(msgp)
                        if okp:
                            st.rerun()

            # Auto-apply template when project changes
            if (
                str(selected_project).strip()
                and st.session_state.get("order_last_project_applied", "") != str(selected_project).strip()
            ):
                applied = apply_template_to_form(selected_project)
                st.session_state["order_last_project_applied"] = str(selected_project).strip()
                if applied:
                    st.toast("Template auto-applied ✅", icon="✅")
                    st.rerun()

            if str(selected_project).strip():
                tpl_exists = bool(get_template_for_project(selected_project))
                st.caption("✅ Template exists for this project." if tpl_exists else "ℹ️ No template yet for this project.")

            # -----------------------------
            # FORM (single submit)
            # -----------------------------
            save_tpl = False
            submit = False
            cancel = False

            with st.form("order_create_form", clear_on_submit=False):
                tab_req, tab_targets, tab_materials, tab_template = st.tabs(
                    ["✅ Required", "🧪 Targets", "🧴 Materials", "💾 Template"]
                )

                # ✅ REQUIRED TAB
                with tab_req:
                    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.0, 1.4], vertical_alignment="bottom")
                    with c1:
                        st.text_input(
                            "Preform Number *",
                            key="order_preform_name",
                            placeholder="0 (if not exist yet) or P0888",
                            help="Use 0 if preform does not exist yet.",
                        )
                    with c2:
                        st.text_input(
                            "Fiber Project *",
                            value=str(selected_project),
                            disabled=True,
                            key="order_fiber_project_disabled",
                        )
                    with c3:
                        st.selectbox("Priority *", ["Low", "Normal", "High"], index=1, key="order_priority")
                    with c4:
                        st.selectbox(
                            f"{FIBER_GEOMETRY_COL} *",
                            options=FIBER_GEOMETRY_OPTIONS,
                            index=0,
                            key="order_fiber_geometry_required",
                        )

                    if "order_tiger_cut_pct" not in st.session_state:
                        st.session_state["order_tiger_cut_pct"] = 0.0
                    if "order_oct_f2f_mm" not in st.session_state:
                        st.session_state["order_oct_f2f_mm"] = 0.0

                    geom = str(st.session_state.get("order_fiber_geometry_required", "")).strip()
                    g1, g2, g3 = st.columns([1.2, 1.2, 1.6], vertical_alignment="bottom")

                    with g1:
                        if geom == "TIGER - PM":
                            st.number_input(
                                "Tiger Cut (%) *",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.5,
                                value=__from_state("order_tiger_cut_pct", 0.0),
                                key="order_tiger_cut_pct",
                            )
                        else:
                            st.caption("Tiger Cut (%) — only for TIGER")

                    with g2:
                        if geom == "Octagonal":
                            st.number_input(
                                "Octagonal F2F (mm) *",
                                min_value=0.0,
                                step=0.01,
                                value=__from_state("order_oct_f2f_mm", 0.0),
                                format="%.2f",
                                key="order_oct_f2f_mm",
                            )
                        else:
                            st.caption("Octagonal F2F — only for Octagonal")

                    with g3:
                        if geom == "PANDA - PM":
                            st.markdown("**🧪 SAP Inventory**")
                            render_sap_inventory_banner()
                        else:
                            st.caption("SAP inventory — only for PANDA - PM")

                    r5, r6, r7 = st.columns([1.3, 1.1, 1.6], vertical_alignment="bottom")
                    with r5:
                        st.number_input(
                            "Required Length (m) *",
                            min_value=0.0,
                            key="order_length_required_required",
                            help="Required Length (m) (for T&M+costumer)",
                        )
                    with r6:
                        st.number_input(
                            "Good Zones Count *",
                            min_value=1,
                            step=1,
                            value=int(st.session_state.get("order_good_zones_required", 1) or 1),
                            key="order_good_zones_required",
                            help=GOOD_ZONES_COL,
                        )
                    with r7:
                        st.text_input("Order Opened By *", key="order_opener", placeholder="Name / initials")

                    # Notes shown in required, recommended but NOT blocking submit
                    st.markdown("##### Notes (recommended)")
                    st.text_area(
                        "Additional Notes / Instructions",
                        key="order_notes",
                        height=120,
                        placeholder="Optional but recommended (special instructions, customer notes, risks, etc.)",
                    )

                # 🧪 TARGETS TAB
                with tab_targets:
                    st.caption("Optional targets. Leave 0 if unknown.")
                    d1, d2, d3 = st.columns(3, vertical_alignment="bottom")
                    with d1:
                        st.number_input("Fiber Diameter (µm)", min_value=0.0, key="order_fiber_diam")
                        st.number_input(FIBER_D_TOL_COL, min_value=0.0, step=0.1, format="%.2f", key="order_fiber_diam_tol")
                    with d2:
                        st.number_input("Main Coating Diameter (µm)", min_value=0.0, key="order_main_diam")
                        st.number_input(MAIN_D_TOL_COL, min_value=0.0, step=0.1, format="%.2f", key="order_main_diam_tol")
                    with d3:
                        st.number_input("Secondary Coating Diameter (µm)", min_value=0.0, key="order_sec_diam")
                        st.number_input(SEC_D_TOL_COL, min_value=0.0, step=0.1, format="%.2f", key="order_sec_diam_tol")

                    st.markdown("---")
                    t1, t2 = st.columns(2, vertical_alignment="bottom")
                    with t1:
                        st.number_input("Tension (g)", min_value=0.0, key="order_tension")
                    with t2:
                        st.number_input("Draw Speed (m/min)", min_value=0.0, key="order_speed")

                # 🧴 MATERIALS TAB
                with tab_materials:
                    st.caption("Coating names are loaded from config_coating.json.")
                    m1, m2 = st.columns(2, vertical_alignment="bottom")
                    with m1:
                        st.selectbox("Main Coating", options=[""] + COATING_OPTIONS, index=0, key="order_coating_main")
                    with m2:
                        st.selectbox("Secondary Coating", options=[""] + COATING_OPTIONS, index=0, key="order_coating_secondary")

                    tt1, tt2 = st.columns(2, vertical_alignment="bottom")
                    with tt1:
                        st.number_input(
                            "Main Coating Temperature (°C)",
                            value=__from_state("order_main_coat_temp_c", 25.0),
                            step=0.5,
                            format="%.1f",
                            key="order_main_coat_temp_c",
                        )
                    with tt2:
                        st.number_input(
                            "Secondary Coating Temperature (°C)",
                            value=__from_state("order_sec_coat_temp_c", 25.0),
                            step=0.5,
                            format="%.1f",
                            key="order_sec_coat_temp_c",
                        )

                # 💾 TEMPLATE TAB (only template save button)
                with tab_template:
                    st.markdown("#### 💾 Project Template (auto-fill defaults)")
                    tA, tB = st.columns([1.2, 2.8], vertical_alignment="center")
                    with tA:
                        save_tpl = st.form_submit_button(
                            "💾 Save / Update Template",
                            disabled=(not str(selected_project).strip()),
                            use_container_width=True,
                        )
                    with tB:
                        st.caption("Saves geometry + tiger/f2f + diameters+tolerances + tension + speed + coatings + temps + notes.")

                st.markdown("---")
                a1, a2 = st.columns([1, 1], vertical_alignment="center")
                with a1:
                    submit = st.form_submit_button("📤 Submit Draw Order", use_container_width=True)
                with a2:
                    cancel = st.form_submit_button("❌ Cancel", use_container_width=True)

            # =========================================================
            # ✅ Scheduling UI OUTSIDE the form (so checkbox works instantly)
            # =========================================================
            st.markdown("---")
            st.markdown("#### 📅 Optional: schedule immediately (password protected)")

            schedule_now = st.checkbox("Schedule now", value=False, key="order_schedule_now_cb")

            sched_ok = False
            start_dt_new = None
            end_dt_new = None

            if schedule_now:
                pwd = st.text_input("Scheduling password", type="password", key="order_sched_pwd")
                if pwd == SCHEDULE_PASSWORD:
                    sched_ok = True
                    st.success("Password OK ✅")
                elif pwd.strip():
                    st.error("Wrong password ❌")

                default_date = pd.Timestamp.today().date()
                preset = st.radio(
                    "Preset",
                    ["All day (08:00–16:00)", "Before lunch (08:00–12:00)", "After lunch (12:00–16:00)"],
                    horizontal=True,
                    key="order_create_sched_preset",
                    label_visibility="collapsed",
                )

                if preset.startswith("All day"):
                    preset_start = dt.time(8, 0)
                    preset_duration = 8 * 60
                elif preset.startswith("Before lunch"):
                    preset_start = dt.time(8, 0)
                    preset_duration = 4 * 60
                else:
                    preset_start = dt.time(12, 0)
                    preset_duration = 4 * 60

                sA, sB, sC = st.columns([1.2, 1.0, 1.0], vertical_alignment="bottom")
                with sA:
                    sched_date_new = st.date_input("Schedule Date", value=default_date, key="order_create_sched_date")
                with sB:
                    sched_start_new = st.time_input("Start Time", value=preset_start, key="order_create_sched_start")
                with sC:
                    sched_dur_new = st.number_input(
                        "Duration (min)",
                        min_value=1,
                        step=5,
                        value=int(preset_duration),
                        key="order_create_sched_dur",
                    )

                start_dt_new = pd.to_datetime(f"{sched_date_new} {sched_start_new}")
                end_dt_new = start_dt_new + pd.to_timedelta(int(sched_dur_new), unit="m")

            # =========================================================
            # Handle Template Save
            # =========================================================
            if save_tpl:
                payload = {
                    FIBER_GEOMETRY_COL: safe_str_from_state("order_fiber_geometry_required", ""),
                    TIGER_CUT_COL: __from_state("order_tiger_cut_pct", 0.0),
                    OCT_F2F_COL: __from_state("order_oct_f2f_mm", 0.0),

                    "Fiber Diameter (µm)": __from_state("order_fiber_diam", 0.0),
                    FIBER_D_TOL_COL: __from_state("order_fiber_diam_tol", 0.0),

                    "Main Coating Diameter (µm)": __from_state("order_main_diam", 0.0),
                    MAIN_D_TOL_COL: __from_state("order_main_diam_tol", 0.0),

                    "Secondary Coating Diameter (µm)": __from_state("order_sec_diam", 0.0),
                    SEC_D_TOL_COL: __from_state("order_sec_diam_tol", 0.0),

                    "Tension (g)": __from_state("order_tension", 0.0),
                    "Draw Speed (m/min)": __from_state("order_speed", 0.0),
                    "Main Coating": safe_str_from_state("order_coating_main", ""),
                    "Secondary Coating": safe_str_from_state("order_coating_secondary", ""),
                    MAIN_COAT_TEMP_COL: __from_state("order_main_coat_temp_c", 25.0),
                    SEC_COAT_TEMP_COL: __from_state("order_sec_coat_temp_c", 25.0),
                    "Notes Default": safe_str_from_state("order_notes", ""),
                }
                ok_s, msg_s = save_or_update_template(selected_project, payload)
                (st.success if ok_s else st.warning)(msg_s)

            # Cancel
            if cancel and not submit:
                st.session_state["show_new_order_form"] = False
                st.rerun()

            # =========================================================
            # Submit order
            # =========================================================
            if submit:
                missing = []
                geom = str(st.session_state.get("order_fiber_geometry_required", "")).strip()

                if not str(st.session_state.get("order_preform_name", "")).strip():
                    missing.append("Preform Number")
                if not str(selected_project).strip():
                    missing.append("Fiber Project")
                if not str(st.session_state.get("order_opener", "")).strip():
                    missing.append("Order Opened By")

                length_required_val = to_float(st.session_state.get("order_length_required_required", 0.0), 0.0)
                if length_required_val <= 0:
                    missing.append("Required Length (m)")

                good_zones_val = int(st.session_state.get("order_good_zones_required", 1) or 1)
                if good_zones_val <= 0:
                    missing.append("Good Zones Count")

                if not geom:
                    missing.append(FIBER_GEOMETRY_COL)

                if geom == "TIGER - PM" and __from_state("order_tiger_cut_pct", 0.0) <= 0:
                    missing.append("Tiger Cut (%)")
                if geom == "Octagonal" and __from_state("order_oct_f2f_mm", 0.0) <= 0:
                    missing.append("Octagonal F2F (mm)")

                if missing:
                    st.error("Please fill required fields: " + ", ".join(missing))
                    st.stop()

                # sanitize geometry fields
                if geom != "TIGER - PM":
                    st.session_state["order_tiger_cut_pct"] = 0.0
                if geom != "Octagonal":
                    st.session_state["order_oct_f2f_mm"] = 0.0

                tiger_cut_val = __from_state("order_tiger_cut_pct", 0.0) if geom == "TIGER - PM" else 0.0
                oct_f2f_val = __from_state("order_oct_f2f_mm", 0.0) if geom == "Octagonal" else 0.0

                # tolerances
                fiber_diam_tol = __from_state("order_fiber_diam_tol", 0.0)
                main_diam_tol = __from_state("order_main_diam_tol", 0.0)
                sec_diam_tol = __from_state("order_sec_diam_tol", 0.0)

                order_data = {
                    "Status": "Pending",
                    "Priority": str(st.session_state.get("order_priority", "Normal")).strip(),
                    "Order Opener": str(st.session_state.get("order_opener", "")).strip(),
                    "Preform Number": str(st.session_state.get("order_preform_name", "")).strip(),
                    PROJECTS_COL: str(selected_project).strip(),
                    FIBER_GEOMETRY_COL: geom,
                    TIGER_CUT_COL: tiger_cut_val,
                    OCT_F2F_COL: oct_f2f_val,
                    "Timestamp": pd.Timestamp.now(),

                    "Fiber Diameter (µm)": __from_state("order_fiber_diam", 0.0),
                    FIBER_D_TOL_COL: float(fiber_diam_tol),

                    "Main Coating Diameter (µm)": __from_state("order_main_diam", 0.0),
                    MAIN_D_TOL_COL: float(main_diam_tol),

                    "Secondary Coating Diameter (µm)": __from_state("order_sec_diam", 0.0),
                    SEC_D_TOL_COL: float(sec_diam_tol),

                    "Tension (g)": __from_state("order_tension", 0.0),
                    "Draw Speed (m/min)": __from_state("order_speed", 0.0),

                    "Required Length (m) (for T&M+costumer)": float(length_required_val),
                    GOOD_ZONES_COL: int(good_zones_val),

                    "Main Coating": safe_str_from_state("order_coating_main", ""),
                    "Secondary Coating": safe_str_from_state("order_coating_secondary", ""),
                    MAIN_COAT_TEMP_COL: __from_state("order_main_coat_temp_c", 25.0),
                    SEC_COAT_TEMP_COL: __from_state("order_sec_coat_temp_c", 25.0),

                    "Notes": safe_str_from_state("order_notes", ""),

                    "Active CSV": "",
                    "Done CSV": "",
                    "Done Description": "",
                    "T&M Moved": False,
                    "T&M Moved Timestamp": "",
                }

                old = pd.read_csv(orders_file, keep_default_na=False) if os.path.exists(orders_file) else pd.DataFrame()
                new_df = pd.concat([old, pd.DataFrame([order_data])], ignore_index=True)
                new_df.to_csv(orders_file, index=False)
                new_idx = int(len(new_df) - 1)

                # Optional schedule now
                if st.session_state.get("order_schedule_now_cb", False):
                    if not sched_ok or start_dt_new is None or end_dt_new is None:
                        st.error("Order saved, but NOT scheduled (password missing/wrong or schedule details missing).")
                    else:
                        existing = pd.read_csv(SCHEDULE_FILE) if os.path.exists(SCHEDULE_FILE) else pd.DataFrame()
                        for c in schedule_required_cols:
                            if c not in existing.columns:
                                existing[c] = ""
                        existing = existing[schedule_required_cols]

                        priority = str(st.session_state.get("order_priority", "Normal")).strip()
                        preform_name = str(st.session_state.get("order_preform_name", "")).strip()

                        fd = __from_state("order_fiber_diam", 0.0)
                        md = __from_state("order_main_diam", 0.0)
                        sd = __from_state("order_sec_diam", 0.0)

                        diam_bits = []
                        s_fd = _fmt_pm(fd, fiber_diam_tol)
                        s_md = _fmt_pm(md, main_diam_tol)
                        s_sd = _fmt_pm(sd, sec_diam_tol)
                        if s_fd:
                            diam_bits.append(f"Fiber {s_fd}")
                        if s_md:
                            diam_bits.append(f"Coat1 {s_md}")
                        if s_sd:
                            diam_bits.append(f"Coat2 {s_sd}")

                        desc_lines = [
                            f"ORDER #{new_idx} | Priority: {priority}",
                            f"Fiber: {selected_project} | Geometry: {geom} | Preform: {preform_name}",
                            f"Required Length: {length_required_val} m | Good Zones Count: {int(good_zones_val)}",
                        ]
                        if diam_bits:
                            desc_lines.append("Diameters: " + " | ".join(diam_bits))

                        if geom == "TIGER - PM":
                            desc_lines.append(f"Tiger Cut: {tiger_cut_val:.1f}%")
                        if geom == "Octagonal":
                            desc_lines.append(f"Oct F2F: {oct_f2f_val:.2f} mm")

                        mtemp = __from_state("order_main_coat_temp_c", 0.0)
                        stemp = __from_state("order_sec_coat_temp_c", 0.0)
                        if mtemp > 0:
                            desc_lines.append(f"Main Coat Temp: {mtemp:.0f}°C")
                        if stemp > 0:
                            desc_lines.append(f"Sec Coat Temp: {stemp:.0f}°C")

                        notes = safe_str_from_state("order_notes", "")
                        if notes:
                            desc_lines.append(f"Notes: {notes}")

                        event_description = " | ".join([x for x in desc_lines if str(x).strip()])

                        new_event = pd.DataFrame([{
                            "Event Type": "Drawing",
                            "Start DateTime": start_dt_new,
                            "End DateTime": end_dt_new,
                            "Description": event_description,
                            "Recurrence": "None",
                        }])

                        pd.concat([existing, new_event], ignore_index=True).to_csv(SCHEDULE_FILE, index=False)

                        new_df.at[new_idx, "Status"] = "Scheduled"
                        new_df.to_csv(orders_file, index=False)
                        st.success("✅ Order saved + scheduled (status set to Scheduled).")

                st.session_state["show_new_order_form"] = False
                st.success("✅ Draw order submitted!")
                st.rerun()
# ------------------ Tower Parts Tab ------------------
elif tab_selection == "🛠️ Tower Parts":
    import os
    import base64
    import pandas as pd
    import streamlit as st

    st.title("🛠️ Tower Parts Management")

    ORDER_FILE = P.parts_orders_csv
    archive_file = P.parts_archived_csv

    # ✅ Status rename (Needed -> Opened)
    STATUS_ORDER = ["Opened", "Approved", "Ordered", "Shipped", "Received", "Installed"]

    # ✅ Single description field (remove Purpose completely)
    BASE_COLUMNS = [
        "Status", "Part Name", "Serial Number",
        "Project Name", "Details",
        "Opened By",
        "Approved", "Approved By", "Approval Date",
        "Ordered By", "Date Ordered", "Company"
    ]

    # ---------------- Load / init ----------------
    if os.path.exists(ORDER_FILE):
        orders_df = pd.read_csv(ORDER_FILE, keep_default_na=False)
    else:
        orders_df = pd.DataFrame(columns=BASE_COLUMNS)

    orders_df.columns = orders_df.columns.str.strip()

    # Backward compat: ensure columns exist + map old "Needed" to "Opened"
    for col in BASE_COLUMNS:
        if col not in orders_df.columns:
            orders_df[col] = ""

    # Drop old Purpose if exists
    orders_df = orders_df.drop(columns=["Purpose"], errors="ignore")

    orders_df["Status"] = orders_df["Status"].fillna("").astype(str).str.strip()
    orders_df["Status"] = orders_df["Status"].replace({"Needed": "Opened", "needed": "Opened"})

    # Unknown / empty -> Opened
    orders_df["Status"] = orders_df["Status"].apply(lambda s: s if s in STATUS_ORDER else "Opened")

    # ---------------- Projects list (match 📦 Order Draw) ----------------
    PROJECTS_FILE = P.projects_fiber_csv
    PROJECTS_COL = "Fiber Project"

    project_options = ["None"]
    try:
        if os.path.exists(PROJECTS_FILE):
            projects_df = pd.read_csv(PROJECTS_FILE, keep_default_na=False)
            projects_df.columns = [str(c).strip() for c in projects_df.columns]
            if PROJECTS_COL in projects_df.columns:
                vals = (
                    projects_df[PROJECTS_COL]
                    .astype(str)
                    .fillna("")
                    .map(lambda x: x.strip())
                )
                vals = [v for v in vals.tolist() if v and v.lower() != "nan"]
                project_options += sorted(list(pd.Series(vals).unique()))
    except Exception:
        pass

    # =========================
    # TABLE (FIRST)
    # =========================
    st.write("### 📋 Orders Table")

    column_order = [
        "Status",
        "Part Name",
        "Serial Number",
        "Project Name",
        "Details",
        "Opened By",
        "Approved",
        "Approved By",
        "Approval Date",
        "Ordered By",
        "Date Ordered",
        "Company",
    ]
    for col in column_order:
        if col not in orders_df.columns:
            orders_df[col] = ""

    # Sort by status
    tmp = orders_df.copy()
    tmp["__status_sort__"] = pd.Categorical(tmp["Status"], categories=STATUS_ORDER, ordered=True)
    tmp = tmp.sort_values(["__status_sort__", "Part Name"], na_position="last").drop(columns="__status_sort__")

    # Color status cell only
    def highlight_status(row):
        color_map = {
            "Opened": "background-color: lightcoral; color: black; font-weight: 900;",
            "Approved": "background-color: lightgreen; color: black; font-weight: 900;",
            "Ordered": "background-color: lightyellow; color: black; font-weight: 900;",
            "Shipped": "background-color: lightblue; color: black; font-weight: 900;",
            "Received": "background-color: green; color: black; font-weight: 900;",
            "Installed": "background-color: lightgray; color: black; font-weight: 900;",
        }
        s = str(row.get("Status", "")).strip()
        return [color_map.get(s, "")] + [""] * (len(row) - 1)

    if not tmp.empty:
        st.dataframe(
            tmp[column_order].fillna("").style.apply(highlight_status, axis=1),
            height=420,
            use_container_width=True,
        )
    else:
        st.info("No orders have been placed yet.")

    st.divider()

    # =========================
    # CLEAN POP AREA (AFTER TABLE)
    # =========================
    st.write("### ✍️ Manage Orders")

    action = st.radio(
        "Choose action",
        ["Add New Order", "Update Existing Order"],
        horizontal=True,
        key="order_action_main",
    )

    # ---------- Add New ----------
    if action == "Add New Order":
        with st.expander("➕ Add New Order", expanded=True):
            with st.form("add_new_order_form", clear_on_submit=True):
                c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

                with c1:
                    part_name = st.text_input("Part Name")
                    serial_number = st.text_input("Serial Number")
                    status = st.selectbox("Status", STATUS_ORDER, index=0)

                with c2:
                    opened_by = st.text_input("Opened By")
                    selected_project = st.selectbox("Fiber Project", project_options)
                    company = st.text_input("Company (optional)")

                with c3:
                    approved = st.selectbox("Approved", ["No", "Yes"], index=0)
                    approved_by = st.text_input("Approved By (optional)")
                    approval_date = st.date_input("Approval Date", value=pd.Timestamp.today())

                details = st.text_area("Details", height=120)

                save = st.form_submit_button("💾 Save Order", use_container_width=True)

                if save:
                    if not part_name.strip():
                        st.error("Part Name is required.")
                    else:
                        new_row = {
                            "Status": status,
                            "Part Name": part_name.strip(),
                            "Serial Number": serial_number.strip(),
                            "Project Name": "" if selected_project == "None" else str(selected_project),
                            "Details": details.strip(),
                            "Opened By": opened_by.strip(),
                            "Company": company.strip(),
                            "Approved": approved,
                            "Approved By": approved_by.strip(),
                            "Approval Date": approval_date.strftime("%Y-%m-%d") if approved == "Yes" else "",
                            "Ordered By": "",
                            "Date Ordered": "",
                        }
                        orders_df = pd.concat([orders_df, pd.DataFrame([new_row])], ignore_index=True)
                        orders_df.to_csv(ORDER_FILE, index=False)
                        st.success("✅ Order saved.")
                        st.rerun()

    # ---------- Update Existing ----------
    else:
        with st.expander("🛠️ Update Existing Order", expanded=True):
            if orders_df.empty:
                st.warning("No orders to update.")
            else:
                labels = (orders_df["Part Name"].astype(str).fillna("") + "  |  " +
                          orders_df["Serial Number"].astype(str).fillna(""))
                label_to_idx = {labels.iloc[i]: i for i in range(len(labels))}
                selected_label = st.selectbox("Select an order", list(label_to_idx.keys()), key="order_update_select")
                order_index = label_to_idx[selected_label]
                cur = orders_df.loc[order_index].to_dict()

                with st.form("update_order_form"):
                    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

                    with c1:
                        updated_part_name = st.text_input("Part Name", value=str(cur.get("Part Name", "")))
                        updated_serial_number = st.text_input("Serial Number", value=str(cur.get("Serial Number", "")))
                        cur_status = str(cur.get("Status", "Opened")).strip()
                        new_status = st.selectbox(
                            "Status",
                            STATUS_ORDER,
                            index=STATUS_ORDER.index(cur_status) if cur_status in STATUS_ORDER else 0,
                        )

                    with c2:
                        cur_proj = str(cur.get("Project Name", ""))
                        updated_project = st.selectbox(
                            "Fiber Project",
                            project_options,
                            index=project_options.index(cur_proj) if cur_proj in project_options else 0,
                        )
                        updated_opened_by = st.text_input("Opened By", value=str(cur.get("Opened By", "")))
                        company = st.text_input("Company", value=str(cur.get("Company", "")))

                    with c3:
                        approved_value = str(cur.get("Approved", "No"))
                        approved = st.selectbox("Approved", ["No", "Yes"], index=0 if approved_value != "Yes" else 1)
                        approved_by = st.text_input("Approved By", value=str(cur.get("Approved By", "")))

                        date_ordered_raw = str(cur.get("Date Ordered", ""))
                        date_ordered_dt = pd.to_datetime(date_ordered_raw, errors="coerce")
                        if pd.isna(date_ordered_dt):
                            date_ordered_dt = pd.Timestamp.today()
                        date_ordered = st.date_input("Date Ordered", value=date_ordered_dt)

                        approval_raw = str(cur.get("Approval Date", ""))
                        approval_dt = pd.to_datetime(approval_raw, errors="coerce")
                        if pd.isna(approval_dt):
                            approval_dt = pd.Timestamp.today()
                        approval_date = st.date_input("Approval Date", value=approval_dt)

                    details = st.text_area("Details", value=str(cur.get("Details", "")), height=120)
                    ordered_by = st.text_input("Ordered By", value=str(cur.get("Ordered By", "")))

                    do_update = st.form_submit_button("✅ Update Order", use_container_width=True)

                    if do_update:
                        orders_df.at[order_index, "Part Name"] = updated_part_name.strip()
                        orders_df.at[order_index, "Serial Number"] = updated_serial_number.strip()
                        orders_df.at[order_index, "Status"] = new_status

                        orders_df.at[order_index, "Project Name"] = "" if updated_project == "None" else str(updated_project)
                        orders_df.at[order_index, "Opened By"] = updated_opened_by.strip()

                        orders_df.at[order_index, "Details"] = details.strip()
                        orders_df.at[order_index, "Company"] = company.strip()
                        orders_df.at[order_index, "Ordered By"] = ordered_by.strip()

                        orders_df.at[order_index, "Approved"] = approved
                        orders_df.at[order_index, "Approved By"] = approved_by.strip()
                        orders_df.at[order_index, "Approval Date"] = approval_date.strftime("%Y-%m-%d") if approved == "Yes" else ""

                        orders_df.at[order_index, "Date Ordered"] = date_ordered.strftime("%Y-%m-%d")

                        orders_df.to_csv(ORDER_FILE, index=False)
                        st.success("✅ Order updated.")
                        st.rerun()

    st.divider()

    # =========================
    # ARCHIVE / VIEW ARCHIVE
    # =========================
    st.write("### 🗃️ Archive")
    cA, cB = st.columns([1, 1])

    with cA:
        if st.button("📦 Archive Installed Orders", use_container_width=True):
            installed_df = orders_df[orders_df["Status"].astype(str).str.strip().str.lower() == "installed"]
            remaining_df = orders_df[orders_df["Status"].astype(str).str.strip().str.lower() != "installed"]

            if installed_df.empty:
                st.info("No installed parts to archive.")
            else:
                if os.path.exists(archive_file):
                    archived_df = pd.read_csv(archive_file, keep_default_na=False)
                    archived_df.columns = archived_df.columns.str.strip()
                    for col in BASE_COLUMNS:
                        if col not in archived_df.columns:
                            archived_df[col] = ""
                    archived_df = pd.concat([archived_df, installed_df], ignore_index=True)
                else:
                    archived_df = installed_df.copy()

                archived_df.to_csv(archive_file, index=False)
                remaining_df.to_csv(ORDER_FILE, index=False)
                st.success(f"✅ {len(installed_df)} installed order(s) archived.")
                st.rerun()

    with cB:
        show_archive = st.button("📂 View Archived Orders", use_container_width=True)

    if show_archive:
        if os.path.exists(archive_file):
            archived_df = pd.read_csv(archive_file, keep_default_na=False)
            archived_df.columns = archived_df.columns.str.strip()
            for col in BASE_COLUMNS:
                if col not in archived_df.columns:
                    archived_df[col] = ""
            if archived_df.empty:
                st.info("The archive is currently empty.")
            else:
                st.write("#### Archived Orders")
                show_cols = [c for c in column_order if c in archived_df.columns]
                st.dataframe(archived_df[show_cols], height=320, use_container_width=True)
        else:
            st.info("Archive file does not exist yet.")

    st.divider()

    # =========================
    # DELETE
    # =========================
    st.write("### 🗑️ Delete")
    if orders_df.empty:
        st.info("Nothing to delete.")
    else:
        del_labels = (orders_df["Part Name"].astype(str).fillna("") + "  |  " +
                      orders_df["Serial Number"].astype(str).fillna(""))
        del_map = {del_labels.iloc[i]: i for i in range(len(del_labels))}
        del_choice = st.selectbox("Select an order to delete", list(del_map.keys()), key="delete_part_main")

        if st.button("Delete Selected Order", use_container_width=True):
            idx = del_map[del_choice]
            orders_df = orders_df.drop(index=idx).reset_index(drop=True)
            orders_df.to_csv(ORDER_FILE, index=False)
            st.success("✅ Deleted.")
            st.rerun()

    st.divider()

    # =========================
    # Parts Datasheet (OLD FLOW) + NICE VIEWER
    # =========================
    st.write("### 📚 Parts Datasheet (Hierarchical View)")

    # NOTE: PARTS_DIRECTORY must exist in your app globals/config.
    # Example: PARTS_DIRECTORY = "tower_parts_docs"

    def render_pdf_embed(path, height=760):
        """Nice in-app PDF viewer (like other tabs)."""
        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            st.markdown(
                f"""
                <iframe
                    src="data:application/pdf;base64,{b64}"
                    width="100%"
                    height="{height}"
                    style="border:none; border-radius: 12px; background: rgba(0,0,0,0.04);"
                ></iframe>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Failed to render PDF: {e}")

    def display_directory(current_path, level=0):
        try:
            items = sorted(os.listdir(current_path))
        except Exception as e:
            st.error(f"Error accessing {current_path}: {e}")
            return None

        folder_options = []
        files = []
        for item in items:
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                folder_options.append(item)
            else:
                files.append(full_path)

        selected_folder = st.selectbox(
            f"📂 Select folder in {os.path.basename(current_path)}:",
            [""] + folder_options,
            key=f"parts_folder_{level}",
        )

        selected_file = None
        if selected_folder:
            selected_file = display_directory(os.path.join(current_path, selected_folder), level + 1)

        # old style file buttons -> now we just set selected_file for preview
        for file_path in files:
            file_name = os.path.basename(file_path)
            if st.button(f"📄 Select {file_name}", key=f"select_{file_path}"):
                selected_file = file_path

        return selected_file

    if "PARTS_DIRECTORY" in globals() and os.path.exists(PARTS_DIRECTORY) and os.listdir(PARTS_DIRECTORY):
        st.write("Pick folder(s), then select a file to preview:")

        selected_file = display_directory(PARTS_DIRECTORY)

        st.divider()
        st.write("### 👁️ Preview")

        if not selected_file:
            st.info("Select a file above to preview it here.")
        else:
            ext = os.path.splitext(selected_file)[1].lower()

            # Always allow download
            try:
                with open(selected_file, "rb") as f:
                    data = f.read()
                st.download_button(
                    "⬇️ Download file",
                    data=data,
                    file_name=os.path.basename(selected_file),
                    use_container_width=True,
                    key=f"parts_dl_{selected_file}"
                )
            except Exception:
                pass

            if ext == ".pdf":
                render_pdf_embed(selected_file, height=780)
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
                st.image(selected_file, use_container_width=True)
            elif ext in [".txt", ".log", ".csv", ".json", ".md"]:
                try:
                    txt = open(selected_file, "r", encoding="utf-8", errors="ignore").read()
                    st.code(txt if len(txt) < 80_000 else (txt[:80_000] + "\n\n... (truncated)"), language="text")
                except Exception as e:
                    st.error(f"Failed to preview text: {e}")
            else:
                st.info("Preview not supported for this file type. Use Download and open locally.")
                st.write(f"**Path:** `{selected_file}`")
    else:
        st.info("No parts documents found in PARTS_DIRECTORY (or PARTS_DIRECTORY not set).")
# ------------------ Development Tab ------------------
elif tab_selection == "🧪 Development Process":
    import os
    import json
    import pandas as pd
    import streamlit as st
    from datetime import datetime

    # =========================================================
    # ✅ Development Process (FULL TAB)
    # ✅ Wide layout (guarded)
    # ✅ Attachments: Photos / PDFs (preview) / Notebooks (.ipynb open real)
    # ✅ Notes: Markdown + LaTeX
    # ✅ Per-experiment: Preview tab is DEFAULT, all inputs moved to Edit tab
    # ✅ Fixed: dev_selected_project session_state crash on delete
    # =========================================================

    # =========================
    # Page config (WIDE) - safe guard
    # =========================
    if "_page_config_set" not in st.session_state:
        try:
            st.set_page_config(layout="wide")
        except Exception:
            pass
        st.session_state["_page_config_set"] = True

    # =========================
    # CSS (HOME-LIKE POLISH)
    # =========================
    st.markdown("""
    <style>
    /* ---------- page spacing ---------- */
    .block-container { padding-top: 2.8rem; padding-bottom: 2.0rem; }

    /* ---------- header / hero card ---------- */
    .dp-hero{
      border-radius: 22px;
      padding: 16px 18px 14px 18px;
      margin: 6px 0 10px 0;
      border: 1px solid rgba(255,255,255,0.10);
      background:
        radial-gradient(980px 260px at 12% -10%, rgba(0,140,255,0.18), rgba(0,0,0,0) 60%),
        radial-gradient(680px 240px at 88% 10%, rgba(0,255,180,0.09), rgba(0,0,0,0) 55%),
        linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      box-shadow: 0 14px 34px rgba(0,0,0,0.34);
    }
    .dp-hero-title{
      font-size: 1.22rem;
      font-weight: 900;
      margin: 0;
      line-height: 1.15;
      letter-spacing: -0.2px;
    }
    .dp-hero-sub{
      margin-top: 6px;
      font-size: 0.93rem;
      color: rgba(255,255,255,0.72);
    }

    /* ---------- sticky toolbar ---------- */
    .dp-sticky{
      position: sticky;
      top: 0.25rem;
      z-index: 50;
      padding-top: 6px;
      padding-bottom: 6px;
      background: linear-gradient(180deg, rgba(10,10,10,0.75), rgba(10,10,10,0.0));
      backdrop-filter: blur(6px);
    }
    .dp-toolbar{
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      box-shadow: 0 10px 28px rgba(0,0,0,0.26);
      padding: 10px 12px;
    }
    .dp-pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      color: rgba(255,255,255,0.78);
      font-size: 0.88rem;
      white-space: nowrap;
    }

    /* ---------- inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div{
      border-radius: 14px !important;
    }
    textarea, input{
      border-radius: 14px !important;
    }

    /* ---------- expanders as cards ---------- */
    div[data-testid="stExpander"] details{
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.09);
      background: rgba(255,255,255,0.02);
      box-shadow: 0 8px 22px rgba(0,0,0,0.18);
      overflow: hidden;
    }
    div[data-testid="stExpander"] details > summary{
      padding: 12px 14px !important;
    }
    div[data-testid="stExpander"] details > div{
      padding: 6px 14px 14px 14px !important;
    }

    /* ---------- buttons (clean + consistent) ---------- */
    .stButton>button{
      border-radius: 14px !important;
      height: 44px !important;
      padding: 8px 14px !important;
      white-space: nowrap !important;
    }
    .stButton>button[kind="primary"]{
      border-radius: 14px !important;
      height: 44px !important;
      padding: 8px 16px !important;
    }

    /* ---------- dataframe looks like a card ---------- */
    div[data-testid="stDataFrame"]{
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      overflow: hidden;
    }

    /* ---------- nicer dividers ---------- */
    hr{ border-color: rgba(255,255,255,0.08) !important; }

    /* segmented label tighten */
    div[data-testid="stSegmentedControl"] label p{
      font-weight: 700;
      opacity: 0.85;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # Files / folders
    # =========================
    PROJECTS_FILE = "development_projects.csv"
    EXPERIMENTS_FILE = "development_experiments.csv"
    UPDATES_FILE = "experiment_updates.csv"
    DATASET_DIR = P.dataset_dir
    MEDIA_ROOT = "development_media"

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

    # =========================
    # Ensure files exist
    # =========================
    def _ensure_files():
        if not os.path.exists(PROJECTS_FILE):
            pd.DataFrame(columns=[
                "Project Name", "Project Purpose", "Target", "Created At", "Archived"
            ]).to_csv(PROJECTS_FILE, index=False)

        if not os.path.exists(EXPERIMENTS_FILE):
            pd.DataFrame(columns=[
                "Project Name",
                "Experiment Title",
                "Date",
                "Researcher",
                "Methods",
                "Purpose",
                "Observations",
                "Results",
                "Is Drawing",
                "Drawing Details",
                "Draw CSV",
                "Attachments",
                "Attachment Captions",
                "Markdown Notes",
            ]).to_csv(EXPERIMENTS_FILE, index=False)

        if not os.path.exists(UPDATES_FILE):
            pd.DataFrame(columns=[
                "Project Name", "Experiment Title", "Update Date", "Researcher", "Update Notes"
            ]).to_csv(UPDATES_FILE, index=False)

    def _ensure_columns(path, required_cols):
        df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        changed = False
        for c in required_cols:
            if c not in df.columns:
                df[c] = False if c in ["Archived", "Is Drawing"] else ""
                changed = True
        if changed:
            df.to_csv(path, index=False)

    _ensure_files()
    _ensure_columns(PROJECTS_FILE, ["Project Name", "Project Purpose", "Target", "Created At", "Archived"])
    _ensure_columns(EXPERIMENTS_FILE, [
        "Project Name", "Experiment Title", "Date", "Researcher", "Methods", "Purpose",
        "Observations", "Results", "Is Drawing", "Drawing Details", "Draw CSV",
        "Attachments", "Attachment Captions", "Markdown Notes"
    ])
    _ensure_columns(UPDATES_FILE, ["Project Name", "Experiment Title", "Update Date", "Researcher", "Update Notes"])

    # =========================
    # Data helpers
    # =========================
    def load_projects():
        df = pd.read_csv(PROJECTS_FILE)
        df["Archived"] = df.get("Archived", False)
        df["Archived"] = df["Archived"].fillna(False).astype(bool)
        return df

    def save_projects(df):
        df.to_csv(PROJECTS_FILE, index=False)

    def load_experiments():
        df = pd.read_csv(EXPERIMENTS_FILE)
        if "Is Drawing" in df.columns:
            df["Is Drawing"] = df["Is Drawing"].fillna(False).astype(bool)
        df["Attachments"] = df.get("Attachments", "").fillna("")
        df["Attachment Captions"] = df.get("Attachment Captions", "").fillna("")
        df["Markdown Notes"] = df.get("Markdown Notes", "").fillna("")
        return df

    def save_experiments(df):
        df.to_csv(EXPERIMENTS_FILE, index=False)

    def load_updates():
        return pd.read_csv(UPDATES_FILE)

    def save_updates(df):
        df.to_csv(UPDATES_FILE, index=False)

    @st.cache_data(show_spinner=False)
    def load_draw_csv(csv_path: str):
        return pd.read_csv(csv_path)

    # =========================
    # Utility helpers
    # =========================
    def _safe(s: str) -> str:
        return str(s).replace("/", "_").replace("\\", "_").replace(":", "-").strip()

    def exp_media_dir(project_name: str, exp_title: str, exp_date: str) -> str:
        d = os.path.join(MEDIA_ROOT, _safe(project_name), f"{_safe(exp_title)}__{_safe(exp_date)}")
        os.makedirs(d, exist_ok=True)
        return d

    def parse_path_list(s):
        if not isinstance(s, str) or not s.strip():
            return []
        return [x for x in s.split(";") if x.strip()]

    def join_path_list(lst):
        return ";".join(lst)

    def parse_captions(s):
        if not isinstance(s, str) or not s.strip():
            return {}
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def dump_captions(d):
        try:
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            return ""

    def list_dataset_csvs_newest_first():
        if not os.path.isdir(DATASET_DIR):
            return []
        files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
        return sorted(files, key=lambda fn: os.path.getmtime(os.path.join(DATASET_DIR, fn)), reverse=True)

    def ext_of(name: str) -> str:
        return os.path.splitext(str(name).lower())[1]

    def is_image(name: str) -> bool:
        return ext_of(name) in IMG_EXTS

    def is_pdf(path: str) -> bool:
        return str(path).lower().endswith(".pdf")

    def is_notebook(path: str) -> bool:
        return str(path).lower().endswith(".ipynb")

    def open_notebook_real(path: str):
        """
        Best-effort: open .ipynb with the OS default application.
        Works when Streamlit runs locally (PyCharm). On a server it opens on the server machine.
        """
        import os as _os, sys as _sys, subprocess as _subprocess
        p = _os.path.abspath(path)
        if not _os.path.exists(p):
            raise FileNotFoundError(p)

        if _sys.platform.startswith("darwin"):
            _subprocess.Popen(["open", p])
        elif _sys.platform.startswith("win"):
            _os.startfile(p)  # type: ignore[attr-defined]
        else:
            _subprocess.Popen(["xdg-open", p])

    def _unique_path(path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 2
        while True:
            cand = f"{base}__{i}{ext}"
            if not os.path.exists(cand):
                return cand
            i += 1

    @st.cache_data(show_spinner=False)
    def read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def render_download_file(path: str, label: str, key: str):
        if not os.path.exists(path):
            st.warning(f"Missing file: {os.path.basename(path)}")
            return
        data = read_bytes(path)
        st.download_button(
            label=label,
            data=data,
            file_name=os.path.basename(path),
            mime=None,
            key=key,
            use_container_width=True
        )

    # =========================
    # PDF preview (RENDER ONLY)
    # =========================
    @st.cache_data(show_spinner=False)
    def pdf_render_pages(path: str, max_pages: int = 1, zoom: float = 1.6):
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        n = min(len(doc), int(max_pages))
        out = []
        mat = fitz.Matrix(float(zoom), float(zoom))
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out.append(pix.tobytes("png"))
        doc.close()
        return out

    def render_pdf_preview(path: str):
        if not os.path.exists(path):
            st.warning("PDF file not found.")
            return

        state_key = f"pdf_show_all__{path}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
        with c1:
            st.markdown("**PDF preview (rendered)**")
            st.caption("Default shows page 1. Click to render more pages.")
        with c2:
            zoom = st.selectbox("Quality", [1.3, 1.6, 2.0], index=1, key=f"pdf_zoom__{path}")
        with c3:
            max_pages = st.number_input("Pages (when expanded)", min_value=1, max_value=200, value=30, step=1, key=f"pdf_pages__{path}")

        b1, b2 = st.columns([1, 1])
        with b1:
            if not st.session_state[state_key]:
                if st.button("📄 Render more pages", use_container_width=True, key=f"pdf_more__{path}"):
                    st.session_state[state_key] = True
                    st.rerun()
            else:
                if st.button("⬅️ Back to page 1", use_container_width=True, key=f"pdf_less__{path}"):
                    st.session_state[state_key] = False
                    st.rerun()
        with b2:
            render_download_file(path, "⬇️ Download PDF", key=f"dl_pdf_viewer__{path}")

        try:
            if st.session_state[state_key]:
                imgs = pdf_render_pages(path, max_pages=int(max_pages), zoom=float(zoom))
                st.caption(f"Showing **{len(imgs)}** page(s).")
                for i, b in enumerate(imgs, start=1):
                    st.image(b, caption=f"Page {i}", use_container_width=True)
            else:
                imgs = pdf_render_pages(path, max_pages=1, zoom=float(zoom))
                if imgs:
                    st.image(imgs[0], caption="Page 1", use_container_width=True)
        except Exception as e:
            st.error(f"PDF render failed. Install PyMuPDF: `pip install pymupdf`  |  Error: {e}")

    # =========================
    # Attachments render (saved)
    # =========================
    def show_saved_attachments(paths, caps: dict, expander_key: str):
        if not paths:
            st.info("No attachments yet.")
            return

        st.caption("Images inline • PDFs preview • Notebooks open (local) • Everything downloadable.")

        imgs = [p for p in paths if is_image(os.path.basename(p))]
        others = [p for p in paths if p not in imgs]

        if imgs:
            st.markdown("**🖼️ Images**")
            captions_list = []
            for p in imgs:
                fn = os.path.basename(p)
                captions_list.append((caps.get(fn, "") or "").strip() or fn)
            st.image(imgs, caption=captions_list, use_container_width=True)

        if others:
            st.markdown("**📄 Files**")
            for i, p in enumerate(others):
                fn = os.path.basename(p)
                cap = (caps.get(fn, "") or "").strip()

                r1, r2, r3 = st.columns([3, 1.0, 1.2])
                with r1:
                    st.markdown(f"**{fn}**")
                    st.caption(cap if cap else "")

                with r2:
                    # PDF preview
                    if is_pdf(p) and os.path.exists(p):
                        if st.button("👁️ Preview", key=f"pdf_prev__{expander_key}__{i}__{fn}", use_container_width=True):
                            st.session_state[f"pdf_preview_path__{expander_key}"] = p

                    # Notebook open (real)
                    if is_notebook(p) and os.path.exists(p):
                        if st.button("📓 Open", key=f"nb_open__{expander_key}__{i}__{fn}", use_container_width=True):
                            try:
                                open_notebook_real(p)
                                st.success("Opened notebook locally (best-effort).")
                            except Exception as e:
                                st.warning(f"Could not open notebook: {e}")

                with r3:
                    if is_pdf(p):
                        render_download_file(p, "⬇️ Download PDF", key=f"dl_pdf__{expander_key}__{i}__{fn}")
                    elif is_notebook(p):
                        render_download_file(p, "⬇️ Download .ipynb", key=f"dl_nb__{expander_key}__{i}__{fn}")
                    else:
                        render_download_file(p, "⬇️ Download", key=f"dl_file__{expander_key}__{i}__{fn}")

            prev_path = st.session_state.get(f"pdf_preview_path__{expander_key}", "")
            if prev_path and os.path.exists(prev_path) and is_pdf(prev_path):
                st.markdown("---")
                st.markdown("### 📄 PDF Preview")
                st.caption(os.path.basename(prev_path))
                render_pdf_preview(prev_path)

                if st.button("✖ Close preview", key=f"pdf_close__{expander_key}", use_container_width=True):
                    st.session_state.pop(f"pdf_preview_path__{expander_key}", None)
                    st.rerun()

    # =========================
    # Session defaults
    # =========================
    st.session_state.setdefault("dev_view_mode_main", "Active")
    st.session_state.setdefault("dev_show_add_experiment", False)
    st.session_state.setdefault("dev_show_new_project", False)
    st.session_state.setdefault("dev_show_manage_project", False)

    # ✅ selection safe keys
    st.session_state.setdefault("dev_selected_project", "")
    st.session_state.setdefault("dev_project_select_ver", 0)

    # =========================
    # Header card
    # =========================
    st.markdown("""
    <div class="dp-hero">
      <div class="dp-hero-title">🧪 Development Process</div>
      <div class="dp-hero-sub">Plan experiments • Attach files • Track updates • Link draws • Notes with Markdown/LaTeX</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # Toolbar (STICKY + 2-ROW)
    # =========================
    projects_df = load_projects()
    view_mode = st.session_state.get("dev_view_mode_main", "Active")

    filtered = projects_df[projects_df["Archived"] == (view_mode == "Archived")]
    project_options = [""] + filtered["Project Name"].dropna().astype(str).unique().tolist()

    st.markdown('<div class="dp-sticky"><div class="dp-toolbar">', unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns([1.10, 1.90, 1.20], gap="medium")

    with r1a:
        vm = st.segmented_control(
            "View",
            options=["Active", "Archived"],
            default=view_mode,
            key="dev_view_mode_main_sc",
        )
        st.session_state["dev_view_mode_main"] = vm

    with r1b:
        view_mode = st.session_state["dev_view_mode_main"]
        projects_df = load_projects()
        filtered = projects_df[projects_df["Archived"] == (view_mode == "Archived")]
        project_options = [""] + filtered["Project Name"].dropna().astype(str).unique().tolist()

        cur_sel = st.session_state.get("dev_selected_project", "")
        if cur_sel and (cur_sel not in project_options):
            st.session_state["dev_selected_project"] = ""
            st.session_state["dev_project_select_ver"] += 1
            cur_sel = ""

        proj_widget_key = f"dev_selected_project_widget__v{st.session_state['dev_project_select_ver']}"
        default_idx = project_options.index(cur_sel) if cur_sel in project_options else 0

        picked = st.selectbox(
            "Project",
            options=project_options,
            index=default_idx,
            key=proj_widget_key,
        )
        st.session_state["dev_selected_project"] = picked

    with r1c:
        selected_project = st.session_state.get("dev_selected_project", "")
        if selected_project:
            label = "➕ Add Experiment" if not st.session_state["dev_show_add_experiment"] else "➖ Hide"
            if st.button(label, use_container_width=True, type="primary", key="dp_btn_add_exp_toggle"):
                st.session_state["dev_show_add_experiment"] = not st.session_state["dev_show_add_experiment"]
        else:
            st.button("➕ Add Experiment", use_container_width=True, disabled=True, key="dp_btn_add_exp_disabled")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    r2a, r2b, r2c = st.columns([1.05, 1.25, 2.70], gap="medium")

    with r2a:
        if st.button("➕ New Project", use_container_width=True, key="dp_btn_new_project"):
            st.session_state["dev_show_new_project"] = not st.session_state["dev_show_new_project"]
            if st.session_state["dev_show_new_project"]:
                st.session_state["dev_show_manage_project"] = False

    with r2b:
        selected_project = st.session_state.get("dev_selected_project", "")
        if st.button("📦 Manage Project", use_container_width=True, disabled=not bool(selected_project), key="dp_btn_manage"):
            st.session_state["dev_show_manage_project"] = not st.session_state["dev_show_manage_project"]
            if st.session_state["dev_show_manage_project"]:
                st.session_state["dev_show_new_project"] = False

    with r2c:
        selected_project = st.session_state.get("dev_selected_project", "")
        mode = st.session_state.get("dev_view_mode_main", "Active")
        if selected_project:
            st.markdown(
                f'<span class="dp-pill">🟢 <b>{selected_project}</b> &nbsp;•&nbsp; <b>{mode}</b> &nbsp;•&nbsp; Ready</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="dp-pill">⚪ No project selected &nbsp;•&nbsp; <b>{mode}</b></span>',
                unsafe_allow_html=True
            )

    st.markdown('</div></div>', unsafe_allow_html=True)

    # =========================
    # New Project panel
    # =========================
    if st.session_state.get("dev_show_new_project", False):
        with st.expander("➕ Create a new project", expanded=True):
            with st.form("dp_create_project_form", clear_on_submit=True):
                new_project_name = st.text_input("Project Name")
                new_project_purpose = st.text_area("Project Purpose", height=110)
                new_project_target = st.text_area("Target", height=90)
                create_project = st.form_submit_button("Create Project")

            if create_project:
                projects_df = load_projects()
                if not new_project_name.strip():
                    st.error("Project Name is required!")
                elif (projects_df["Project Name"].astype(str).str.strip() == new_project_name.strip()).any():
                    st.error("A project with this name already exists.")
                else:
                    new_row = pd.DataFrame([{
                        "Project Name": new_project_name.strip(),
                        "Project Purpose": new_project_purpose.strip(),
                        "Target": new_project_target.strip(),
                        "Created At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Archived": False
                    }])
                    projects_df = pd.concat([projects_df, new_row], ignore_index=True)
                    save_projects(projects_df)

                    st.success("Project created!")
                    st.session_state["dev_show_new_project"] = False
                    st.session_state["dev_selected_project"] = new_project_name.strip()
                    st.session_state["dev_project_select_ver"] += 1
                    st.rerun()

    # =========================
    # Manage Project panel
    # =========================
    selected_project = st.session_state.get("dev_selected_project", "")
    if st.session_state.get("dev_show_manage_project", False):
        with st.expander("📦 Manage selected project", expanded=True):
            if not selected_project:
                st.info("Select a project first.")
            else:
                projects_df = load_projects()
                row = projects_df[projects_df["Project Name"] == selected_project]
                if row.empty:
                    st.warning("Project not found.")
                else:
                    is_archived = bool(row.iloc[0].get("Archived", False))

                    cA, cB, cC = st.columns([1, 1, 1.2])
                    with cA:
                        if not is_archived:
                            if st.button("🗄️ Archive", use_container_width=True, key="dp_arch"):
                                projects_df.loc[projects_df["Project Name"] == selected_project, "Archived"] = True
                                save_projects(projects_df)
                                st.success("Archived.")
                                st.rerun()
                        else:
                            if st.button("♻️ Restore", use_container_width=True, key="dp_restore"):
                                projects_df.loc[projects_df["Project Name"] == selected_project, "Archived"] = False
                                save_projects(projects_df)
                                st.success("Restored.")
                                st.rerun()

                    with cB:
                        if st.button("🧾 Close panel", use_container_width=True, key="dp_close_manage"):
                            st.session_state["dev_show_manage_project"] = False
                            st.rerun()

                    with cC:
                        st.markdown("**Danger zone**")
                        if st.button("🗑️ Delete project (permanent)", use_container_width=True, key="dp_delete"):
                            exp_df_del = load_experiments()
                            upd_df_del = load_updates()

                            projects_df = projects_df[projects_df["Project Name"] != selected_project]
                            exp_df_del = exp_df_del[exp_df_del["Project Name"] != selected_project]
                            upd_df_del = upd_df_del[upd_df_del["Project Name"] != selected_project]

                            save_projects(projects_df)
                            save_experiments(exp_df_del)
                            save_updates(upd_df_del)

                            st.session_state["dev_selected_project"] = ""
                            st.session_state["dev_show_manage_project"] = False
                            st.session_state["dev_project_select_ver"] += 1
                            st.warning("Deleted permanently.")
                            st.rerun()

    st.divider()

    # =========================
    # Main content
    # =========================
    selected_project = st.session_state.get("dev_selected_project", "")
    if not selected_project:
        st.info("Use **Project** selector above to start.")
        st.stop()

    projects_df = load_projects()
    proj_row = projects_df[projects_df["Project Name"] == selected_project]
    if proj_row.empty:
        st.warning("Selected project not found.")
        st.stop()

    proj = proj_row.iloc[0]
    with st.expander("📌 Project Details", expanded=True):
        st.markdown(f"**Project Purpose:** {proj.get('Project Purpose', 'N/A')}")
        st.markdown(f"**Target:** {proj.get('Target', 'N/A')}")
        st.caption(f"Created at: {proj.get('Created At', '')} | Archived: {bool(proj.get('Archived', False))}")

    st.divider()

    # =========================
    # Add Experiment
    # =========================
    if st.session_state.get("dev_show_add_experiment", False):
        with st.expander("➕ Add Experiment", expanded=True):
            is_drawing_live = st.checkbox("Is this a Drawing?", key=f"newexp_is_drawing__{selected_project}")
            drawing_details_live = ""
            draw_csv_live = ""

            if is_drawing_live:
                drawing_details_live = st.text_area("Drawing Details", height=90, key=f"newexp_drawing_details__{selected_project}")

                dataset_files = list_dataset_csvs_newest_first()
                if dataset_files:
                    newest = dataset_files[0]
                    st.caption(f"Newest CSV: **{newest}**")
                    draw_csv_live = st.selectbox(
                        "Select Draw CSV (newest first)",
                        [""] + dataset_files,
                        index=1,
                        key=f"newexp_draw_csv__{selected_project}"
                    )
                else:
                    st.info("No CSV files found in data_set_csv/")

            st.divider()
            st.markdown("### 📎 Attach files (optional)")
            uploaded_new_files = st.file_uploader(
                "Drag & drop files (images / PDF / .ipynb / anything)",
                type=None,
                accept_multiple_files=True,
                key=f"newexp_attachments__{selected_project}"
            )

            caption_inputs = {}
            if uploaded_new_files:
                st.markdown("### 📝 Descriptions (one per file)")
                for f in uploaded_new_files:
                    caption_inputs[f.name] = st.text_area(
                        f"Description for {f.name}",
                        height=80,
                        key=f"newexp_caption__{selected_project}__{f.name}"
                    )

            st.divider()
            st.markdown("### 📓 Notes (Markdown + LaTeX)")
            notes_md = st.text_area("Write notes here", height=160, key=f"newexp_notes__{selected_project}")
            st.caption("Markdown + LaTeX: inline `$E=mc^2$` or block `$$\\Delta n(r)=n_0 e^{-r^2/w^2}$$`")

            st.divider()

            with st.form(f"add_experiment_form__{selected_project}", clear_on_submit=True):
                c1, c2 = st.columns([2, 1])
                experiment_title = c1.text_input("Experiment Title")
                date = c2.date_input("Date")

                researcher = st.text_input("Researcher Name")
                methods = st.text_area("Methods", height=90)
                purpose = st.text_area("Experiment Purpose", height=90)
                observations = st.text_area("Observations", height=90)
                results = st.text_area("Results", height=90)

                add_exp = st.form_submit_button("✅ Save Experiment")

            if add_exp:
                if not experiment_title.strip():
                    st.warning("Please provide an Experiment Title.")
                else:
                    exp_df = load_experiments()
                    exp_date_str = date.strftime("%Y-%m-%d")

                    dup = exp_df[
                        (exp_df["Project Name"] == selected_project) &
                        (exp_df["Experiment Title"].astype(str).str.strip() == experiment_title.strip()) &
                        (exp_df["Date"].astype(str).str.strip() == exp_date_str)
                    ]
                    if not dup.empty:
                        st.error("This experiment (same title + date) already exists in this project.")
                    else:
                        saved_paths = []
                        caps_map = {}

                        if uploaded_new_files:
                            media_dir = exp_media_dir(selected_project, experiment_title.strip(), exp_date_str)
                            for f in uploaded_new_files:
                                try:
                                    out_path = _unique_path(os.path.join(media_dir, f.name))
                                    with open(out_path, "wb") as w:
                                        w.write(f.getbuffer())
                                    saved_paths.append(out_path)
                                    caps_map[os.path.basename(out_path)] = (caption_inputs.get(f.name, "") or "").strip()
                                except Exception as e:
                                    st.error(f"Failed saving {f.name}: {e}")

                        new_exp = pd.DataFrame([{
                            "Project Name": selected_project,
                            "Experiment Title": experiment_title.strip(),
                            "Date": exp_date_str,
                            "Researcher": researcher.strip(),
                            "Methods": methods.strip(),
                            "Purpose": purpose.strip(),
                            "Observations": observations.strip(),
                            "Results": results.strip(),
                            "Is Drawing": bool(is_drawing_live),
                            "Drawing Details": drawing_details_live.strip() if is_drawing_live else "",
                            "Draw CSV": draw_csv_live.strip() if is_drawing_live else "",
                            "Attachments": join_path_list(saved_paths) if saved_paths else "",
                            "Attachment Captions": dump_captions(caps_map) if caps_map else "",
                            "Markdown Notes": (st.session_state.get(f"newexp_notes__{selected_project}", "") or "").strip(),
                        }])

                        exp_df = pd.concat([exp_df, new_exp], ignore_index=True)
                        save_experiments(exp_df)

                        st.success(f"Experiment saved. Attachments: {len(saved_paths)}")
                        st.session_state["dev_show_add_experiment"] = False
                        st.rerun()

        st.divider()

    # =========================
    # Experiments list
    # =========================
    exp_df = load_experiments()
    project_exps = exp_df[exp_df["Project Name"] == selected_project].copy()

    if project_exps.empty:
        st.info("No experiments yet.")
    else:
        st.subheader("🔬 Experiments Conducted")
        project_exps["Date_sort"] = pd.to_datetime(project_exps["Date"], errors="coerce")
        project_exps = project_exps.sort_values("Date_sort", ascending=False)

        for idx, exp in project_exps.iterrows():
            exp_title = str(exp.get("Experiment Title", "Untitled"))
            exp_date = str(exp.get("Date", ""))

            expander_key = f"exp_{selected_project}_{exp_title}_{exp_date}_{idx}"

            with st.expander(f"🧪 {exp_title} ({exp_date})", expanded=False):

                # ✅ DEFAULT TAB = Preview (put it first)
                tab_preview, tab_edit = st.tabs(["👁️ Preview", "✍️ Edit"])

                # -------------------------
                # PREVIEW (NO INPUTS)
                # -------------------------
                with tab_preview:
                    st.write(f"**Researcher:** {exp.get('Researcher', 'N/A')}")
                    st.write(f"**Methods:** {exp.get('Methods', 'N/A')}")
                    st.write(f"**Purpose:** {exp.get('Purpose', 'N/A')}")
                    st.write(f"**Observations:** {exp.get('Observations', 'N/A')}")
                    st.write(f"**Results:** {exp.get('Results', 'N/A')}")

                    # Notes preview
                    st.divider()
                    st.markdown("#### 📓 Notes (Markdown + LaTeX)")
                    notes_preview = str(exp.get("Markdown Notes", "") or "").strip()
                    if notes_preview:
                        st.markdown(notes_preview)
                    else:
                        st.caption("No notes yet.")

                    # Drawing preview (view-only button allowed)
                    if bool(exp.get("Is Drawing", False)):
                        st.divider()
                        st.markdown("#### 🧵 Drawing")
                        st.write(f"**Drawing Details:** {exp.get('Drawing Details', '')}")
                        draw_csv_name = str(exp.get("Draw CSV", "")).strip()
                        if draw_csv_name:
                            st.write(f"**Draw CSV:** `{draw_csv_name}`")
                            csv_path = os.path.join(DATASET_DIR, draw_csv_name)
                            if os.path.exists(csv_path):
                                if st.button("📄 Load & View Draw CSV", key=f"load_draw__{expander_key}"):
                                    df_draw = load_draw_csv(csv_path)
                                    st.dataframe(df_draw, use_container_width=True, height=320)

                    # Attachments preview (saved only)
                    st.divider()
                    st.markdown("#### 📎 Attachments")
                    saved_paths = parse_path_list(exp.get("Attachments", ""))
                    saved_caps = parse_captions(exp.get("Attachment Captions", ""))
                    show_saved_attachments(saved_paths, saved_caps, expander_key)

                    # Updates preview (list only)
                    st.divider()
                    st.markdown("#### 📜 Progress Updates")
                    upd_df = load_updates()
                    exp_updates = upd_df[
                        (upd_df["Project Name"] == selected_project) &
                        (upd_df["Experiment Title"] == exp_title)
                    ].copy()

                    if exp_updates.empty:
                        st.caption("No updates yet.")
                    else:
                        exp_updates["Update_sort"] = pd.to_datetime(exp_updates["Update Date"], errors="coerce")
                        exp_updates = exp_updates.sort_values("Update_sort", ascending=True)
                        for _, u in exp_updates.iterrows():
                            st.write(
                                f"📅 **{u.get('Update Date', '')}** — **{u.get('Researcher', '')}**: {u.get('Update Notes', '')}"
                            )

                # -------------------------
                # EDIT (ALL INPUTS HERE)
                # -------------------------
                with tab_edit:
                    # Notes editor
                    st.markdown("#### ✍️ Edit Notes (Markdown + LaTeX)")
                    edited_notes = st.text_area(
                        "Notes",
                        value=str(exp.get("Markdown Notes", "") or ""),
                        height=220,
                        key=f"md_notes__{expander_key}"
                    )
                    csave, chelp = st.columns([1, 2])
                    with csave:
                        if st.button("💾 Save Notes", use_container_width=True, key=f"save_notes__{expander_key}"):
                            exp_df2 = load_experiments()
                            mask = (
                                (exp_df2["Project Name"] == selected_project) &
                                (exp_df2["Experiment Title"].astype(str) == exp_title) &
                                (exp_df2["Date"].astype(str) == exp_date)
                            )
                            exp_df2.loc[mask, "Markdown Notes"] = (edited_notes or "").strip()
                            save_experiments(exp_df2)
                            st.success("Notes saved.")
                            st.rerun()
                    with chelp:
                        st.caption("Inline `$E=mc^2$` • block `$$\\Delta n(r)=n_0 e^{-r^2/w^2}$$` • tables, code blocks, etc.")

                    st.divider()

                    # Add attachments (inputs here)
                    st.markdown("#### ➕ Add attachments")
                    st.caption("Upload images / PDFs / notebooks (.ipynb) or any file.")
                    add_files = st.file_uploader(
                        "Drop files here",
                        type=None,
                        accept_multiple_files=True,
                        key=f"add_files__{expander_key}"
                    )

                    add_caps = {}
                    if add_files:
                        st.markdown("**Descriptions for new files**")
                        for f in add_files:
                            add_caps[f.name] = st.text_area(
                                f"Description for {f.name}",
                                height=80,
                                key=f"add_cap__{expander_key}__{f.name}"
                            )

                        if st.button("💾 Save attachments", use_container_width=True, key=f"save_added__{expander_key}"):
                            media_dir = exp_media_dir(selected_project, exp_title, exp_date)
                            exp_df2 = load_experiments()

                            # current saved state
                            current_paths = parse_path_list(exp.get("Attachments", ""))
                            current_caps = parse_captions(exp.get("Attachment Captions", ""))

                            new_paths = []
                            for f in add_files:
                                try:
                                    out_path = _unique_path(os.path.join(media_dir, f.name))
                                    with open(out_path, "wb") as w:
                                        w.write(f.getbuffer())
                                    new_paths.append(out_path)
                                    current_caps[os.path.basename(out_path)] = (add_caps.get(f.name, "") or "").strip()
                                except Exception as e:
                                    st.error(f"Failed saving {f.name}: {e}")

                            if new_paths:
                                mask = (
                                    (exp_df2["Project Name"] == selected_project) &
                                    (exp_df2["Experiment Title"].astype(str) == exp_title) &
                                    (exp_df2["Date"].astype(str) == exp_date)
                                )
                                merged = (current_paths or []) + new_paths
                                exp_df2.loc[mask, "Attachments"] = join_path_list(merged)
                                exp_df2.loc[mask, "Attachment Captions"] = dump_captions(current_caps)
                                save_experiments(exp_df2)

                                st.success(f"Saved {len(new_paths)} file(s).")
                                st.rerun()

                    st.divider()

                    # Add update (inputs here)
                    st.markdown("#### 🔄 Add Update")
                    with st.form(f"update_form__{expander_key}"):
                        update_researcher = st.text_input("Your name", key=f"upd_name__{expander_key}")
                        update_notes = st.text_area("Update notes", height=80, key=f"upd_notes__{expander_key}")
                        submit_update = st.form_submit_button("Add Update")

                    if submit_update:
                        if not update_notes.strip():
                            st.warning("Please write update notes.")
                        else:
                            upd_df2 = load_updates()
                            new_u = pd.DataFrame([{
                                "Project Name": selected_project,
                                "Experiment Title": exp_title,
                                "Update Date": datetime.now().strftime("%Y-%m-%d"),
                                "Researcher": update_researcher.strip(),
                                "Update Notes": update_notes.strip()
                            }])
                            upd_df2 = pd.concat([upd_df2, new_u], ignore_index=True)
                            save_updates(upd_df2)
                            st.success("Update added!")
                            st.rerun()

    st.divider()

    st.subheader("📢 Project Conclusion")
    conclusion_file = f"project_conclusion__{selected_project.replace(' ', '_')}.txt"

    existing = ""
    if os.path.exists(conclusion_file):
        try:
            existing = open(conclusion_file, "r", encoding="utf-8").read()
        except Exception:
            existing = ""

    conclusion = st.text_area("Conclusion / final summary", value=existing, height=170)

    if st.button("💾 Save Conclusion", key=f"save_conclusion__{selected_project}"):
        try:
            with open(conclusion_file, "w", encoding="utf-8") as f:
                f.write(conclusion)
            st.success("Conclusion saved.")
        except Exception as e:
            st.error(f"Failed to save conclusion: {e}")
# ------------------ Protocols Tab ------------------
elif tab_selection == "📋 Protocols":
    import os, json, hashlib, time
    import streamlit as st

    # ==========================================================
    # Page config (wide)
    # ==========================================================
    if "_page_config_set" not in st.session_state:
        try:
            st.set_page_config(layout="wide")
        except Exception:
            pass
        st.session_state["_page_config_set"] = True

    # ==========================================================
    # CSS (title lower + wide create form + clean cards)
    # ==========================================================
    st.markdown("""
    <style>
      .block-container { padding-top: 2.2rem; }

      .proto-topbar {
        display:flex; align-items:flex-end; justify-content:space-between;
        gap: 12px; margin-bottom: 10px;
      }
      .proto-title h1 { margin: 0; padding: 0; line-height: 1.1; }
      .proto-sub { opacity: 0.75; margin-top: 4px; font-size: 0.95rem; }

      .chips { display:flex; gap: 8px; flex-wrap: wrap; justify-content:flex-start; margin: 10px 0 14px; }
      .chip {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        line-height: 1;
        white-space: nowrap;
      }

      .section-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        padding: 14px 14px;
        border-radius: 14px;
        margin-bottom: 14px;
      }

      .proto-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        padding: 12px 12px;
        border-radius: 14px;
        margin-bottom: 10px;
      }
      .proto-card .row1 {
        display:flex; align-items:center; justify-content:space-between; gap:10px;
      }
      .proto-name {
        font-weight: 750;
        font-size: 1.05rem;
        line-height: 1.2;
      }
      .proto-meta {
        display:flex; gap: 6px; flex-wrap:wrap; justify-content:flex-end;
        opacity: 0.9;
      }
      .pill {
        font-size: 0.78rem;
        border: 1px solid rgba(255,255,255,0.12);
        padding: 3px 8px;
        border-radius: 999px;
        background: rgba(255,255,255,0.03);
      }
      .hr-lite { height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0; border-radius: 2px; }
      .muted { opacity: 0.75; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .small-note { font-size: 0.85rem; opacity: 0.75; }

      details[data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: rgba(255,255,255,0.03) !important;
        padding: 6px 10px !important;
      }

      textarea {
        min-height: 420px !important;
        font-size: 0.98rem !important;
        line-height: 1.35 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================================
    # Header
    # ==========================================================
    st.markdown("""
    <div class="proto-topbar">
      <div class="proto-title">
        <h1>📋 Protocols</h1>
        <div class="proto-sub">Browse, run checklists, attach photos, and manage tower protocols.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================
    # Storage
    # ==========================================================
    PROTOCOLS_FILE = "protocols.json"
    ASSETS_DIR = "protocols_assets"
    os.makedirs(ASSETS_DIR, exist_ok=True)

    protocol_types = ["Drawings", "Maintenance", "Tower Regular Operations"]
    sub_types = ["Checklist", "Instructions"]

    def _safe_read_json(path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _safe_write_json(path: str, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
        os.replace(tmp, path)

    if "protocols" not in st.session_state:
        st.session_state["protocols"] = _safe_read_json(PROTOCOLS_FILE, [])

    # Ensure schema (backward compatible)
    for p in st.session_state["protocols"]:
        p.setdefault("type", "Tower Regular Operations")
        p.setdefault("sub_type", "Instructions")
        p.setdefault("instructions", "")
        p.setdefault("name", "Untitled")
        p.setdefault("images", [])  # ✅ new

    # Checklist progress state (session)
    if "protocol_check_state" not in st.session_state:
        st.session_state["protocol_check_state"] = {}  # {protocol_id: {item: bool}}

    def _proto_id(p: dict) -> str:
        s = f"{p.get('name','')}|{p.get('type','')}|{p.get('sub_type','')}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

    def _normalize_lines(text: str):
        lines = [ln.strip() for ln in (text or "").splitlines()]
        return [ln for ln in lines if ln]

    def _matches(q: str, p: dict) -> bool:
        if not q:
            return True
        blob = f"{p.get('name','')} {p.get('type','')} {p.get('sub_type','')} {p.get('instructions','')}".lower()
        return q.lower() in blob

    # ==========================================================
    # Image helpers
    # ==========================================================
    def _safe_ext(filename: str) -> str:
        fn = (filename or "").lower().strip()
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            if fn.endswith(ext):
                return ext
        return ".png"

    def _save_uploaded_image(uploaded_file, pid: str) -> str:
        # Unique + stable-ish name: pid + time + hash
        raw = uploaded_file.getvalue()
        h = hashlib.md5(raw).hexdigest()[:10]
        ext = _safe_ext(uploaded_file.name)
        fname = f"{pid}_{int(time.time()*1000)}_{h}{ext}"
        path = os.path.join(ASSETS_DIR, fname)
        with open(path, "wb") as f:
            f.write(raw)
        return fname

    def _existing_image_paths(img_names: list) -> list:
        out = []
        for n in (img_names or []):
            p = os.path.join(ASSETS_DIR, str(n))
            if os.path.isfile(p):
                out.append(p)
        return out

    # ==========================================================
    # Stats
    # ==========================================================
    total_n = len(st.session_state["protocols"])
    by_type = {t: 0 for t in protocol_types}
    by_sub = {s: 0 for s in sub_types}
    for p in st.session_state["protocols"]:
        if p.get("type") in by_type:
            by_type[p["type"]] += 1
        if p.get("sub_type") in by_sub:
            by_sub[p["sub_type"]] += 1

    st.markdown(
        f"""
        <div class="chips">
          <div class="chip">Total: <b>{total_n}</b></div>
          <div class="chip">Drawings: <b>{by_type["Drawings"]}</b></div>
          <div class="chip">Maintenance: <b>{by_type["Maintenance"]}</b></div>
          <div class="chip">Ops: <b>{by_type["Tower Regular Operations"]}</b></div>
          <div class="chip">Checklists: <b>{by_sub["Checklist"]}</b></div>
          <div class="chip">Instructions: <b>{by_sub["Instructions"]}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================
    # 1) BROWSE FIRST (FULL WIDTH)
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📚 Browse")

    f1, f2, f3, f4 = st.columns([1.7, 1.0, 1.0, 0.8])
    with f1:
        q = st.text_input("Search", placeholder="Search name / type / content…", key="proto_search_full")
    with f2:
        type_filter = st.selectbox("Type", ["All"] + protocol_types, key="proto_type_filter_full")
    with f3:
        sub_filter = st.selectbox("Sub-type", ["All"] + sub_types, key="proto_sub_filter_full")
    with f4:
        sort_by = st.selectbox("Sort", ["A→Z", "Z→A"], key="proto_sort_full")

    items = []
    for p in st.session_state["protocols"]:
        if type_filter != "All" and p.get("type") != type_filter:
            continue
        if sub_filter != "All" and p.get("sub_type") != sub_filter:
            continue
        if not _matches(q, p):
            continue
        items.append(p)

    items.sort(key=lambda x: (x.get("name", "").lower()))
    if sort_by == "Z→A":
        items.reverse()

    st.markdown("</div>", unsafe_allow_html=True)

    if not items:
        st.info("No protocols match your filters.")
    else:
        for idx, p in enumerate(items):
            pid = _proto_id(p)
            name = p.get("name", "Untitled")
            ptype = p.get("type", "")
            psub = p.get("sub_type", "Instructions")
            instructions = p.get("instructions", "")
            images = p.get("images", []) or []

            st.markdown(
                f"""
                <div class="proto-card">
                  <div class="row1">
                    <div class="proto-name">{name}</div>
                    <div class="proto-meta">
                      <span class="pill">{ptype}</span>
                      <span class="pill">{psub}</span>
                      <span class="pill">{len(images)} photos</span>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Open", expanded=False):

                # ------------------------------
                # Preview: Photos (always)
                # ------------------------------
                img_paths = _existing_image_paths(images)
                if img_paths:
                    st.caption("📷 Photos")
                    # Show up to 3 per row
                    cols = st.columns(3)
                    for i, path in enumerate(img_paths):
                        with cols[i % 3]:
                            st.image(path, use_container_width=True)

                    st.markdown('<div class="hr-lite"></div>', unsafe_allow_html=True)

                # ------------------------------
                # Preview: Checklist / Instructions
                # ------------------------------
                if psub == "Checklist":
                    lines = _normalize_lines(instructions)
                    if not lines:
                        st.warning("Checklist has no items (one item per line).")
                    else:
                        if pid not in st.session_state["protocol_check_state"]:
                            st.session_state["protocol_check_state"][pid] = {ln: False for ln in lines}

                        existing = st.session_state["protocol_check_state"][pid]
                        merged = {ln: bool(existing.get(ln, False)) for ln in lines}
                        st.session_state["protocol_check_state"][pid] = merged

                        done_count = sum(1 for v in merged.values() if v)
                        total_count = len(lines)

                        st.caption(f"Progress: **{done_count}/{total_count}**")
                        st.progress(done_count / total_count if total_count else 0)

                        for i, item in enumerate(lines):
                            k = f"proto_chk_{pid}_{i}"
                            val = st.checkbox(item, value=merged[item], key=k)
                            st.session_state["protocol_check_state"][pid][item] = val

                        if done_count == total_count and total_count > 0:
                            st.success("✅ Checklist completed!")

                        if st.button("Reset this checklist", key=f"reset_{pid}", use_container_width=True):
                            st.session_state["protocol_check_state"][pid] = {ln: False for ln in lines}
                            st.rerun()
                else:
                    pretty = (instructions or "").strip()
                    if not pretty:
                        st.info("No instructions found.")
                    else:
                        st.markdown(pretty.replace("\n", "  \n"))

                st.markdown('<div class="hr-lite"></div>', unsafe_allow_html=True)

                # ------------------------------
                # Edit mode (only shows inputs)
                # ------------------------------
                edit = st.toggle("✏️ Edit this protocol", key=f"proto_edit_{pid}", value=False)
                if edit:
                    st.subheader("Edit")

                    c1, c2, c3 = st.columns([1.7, 1.0, 1.0])
                    with c1:
                        new_name = st.text_input("Name", value=p.get("name", ""), key=f"edit_name_{pid}")
                    with c2:
                        new_type = st.selectbox("Type", protocol_types, index=protocol_types.index(p.get("type","Tower Regular Operations")) if p.get("type") in protocol_types else 0, key=f"edit_type_{pid}")
                    with c3:
                        new_sub = st.selectbox("Sub-type", sub_types, index=sub_types.index(p.get("sub_type","Instructions")) if p.get("sub_type") in sub_types else 1, key=f"edit_sub_{pid}")

                    new_text = st.text_area(
                        "Instructions / Checklist items",
                        value=p.get("instructions", ""),
                        height=420,
                        key=f"edit_text_{pid}",
                    )

                    st.markdown("### 📷 Photos")
                    up = st.file_uploader(
                        "Upload photos (png/jpg/jpeg/webp)",
                        type=["png", "jpg", "jpeg", "webp"],
                        accept_multiple_files=True,
                        key=f"proto_uploader_{pid}",
                    )

                    # show list + delete options
                    existing_imgs = p.get("images", []) or []
                    if existing_imgs:
                        del_sel = st.multiselect(
                            "Select photos to remove",
                            options=existing_imgs,
                            key=f"proto_del_sel_{pid}",
                        )
                    else:
                        del_sel = []

                    csave, cdel = st.columns([1, 1])
                    with csave:
                        if st.button("💾 Save changes", key=f"save_{pid}", use_container_width=True):
                            # find original object in session list and update it
                            for j in range(len(st.session_state["protocols"])):
                                if _proto_id(st.session_state["protocols"][j]) == pid:
                                    # save uploads
                                    imgs = list(existing_imgs)
                                    if up:
                                        for uf in up:
                                            try:
                                                fname = _save_uploaded_image(uf, pid)
                                                imgs.append(fname)
                                            except Exception as e:
                                                st.warning(f"Failed saving image: {e}")

                                    # delete selected photos (and files)
                                    if del_sel:
                                        imgs2 = []
                                        for im in imgs:
                                            if im in del_sel:
                                                try:
                                                    fp = os.path.join(ASSETS_DIR, im)
                                                    if os.path.isfile(fp):
                                                        os.remove(fp)
                                                except Exception:
                                                    pass
                                            else:
                                                imgs2.append(im)
                                        imgs = imgs2

                                    st.session_state["protocols"][j] = {
                                        "name": (new_name or "Untitled").strip(),
                                        "type": new_type,
                                        "sub_type": new_sub,
                                        "instructions": (new_text or "").strip(),
                                        "images": imgs,
                                    }
                                    _safe_write_json(PROTOCOLS_FILE, st.session_state["protocols"])
                                    st.success("Saved.")
                                    st.rerun()
                            st.warning("Could not find protocol to save (ID mismatch).")

                    with cdel:
                        if st.button("🗑️ Delete this protocol", key=f"del_proto_{pid}", use_container_width=True):
                            st.session_state["protocols"] = [
                                pp for pp in st.session_state["protocols"]
                                if _proto_id(pp) != pid
                            ]
                            _safe_write_json(PROTOCOLS_FILE, st.session_state["protocols"])
                            st.success("Deleted.")
                            st.rerun()

                st.markdown(
                    f"<div class='small-note muted'>ID: <span class='mono'>{pid}</span></div>",
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # 2) CREATE / MANAGE AFTER (FULL WIDTH + HUGE WRITING AREA)
    # ==========================================================
    with st.expander("➕ Create new protocol", expanded=False):
        st.markdown("<div class='small-note muted'>Create in full width so it’s comfortable to write.</div>", unsafe_allow_html=True)
        st.markdown("<div class='hr-lite'></div>", unsafe_allow_html=True)

        with st.form("proto_create_form_big", clear_on_submit=True):
            r1c1, r1c2, r1c3 = st.columns([1.8, 1.0, 1.0])
            with r1c1:
                new_name = st.text_input("Protocol name", placeholder="e.g. Pre-Draw Checklist")
            with r1c2:
                new_type = st.selectbox("Type", protocol_types, index=0, key="proto_new_type_big")
            with r1c3:
                new_sub = st.selectbox("Sub-type", sub_types, index=0, key="proto_new_subtype_big")

            new_text = st.text_area(
                "Instructions",
                height=520,
                placeholder="Checklist: one item per line.\n\nInstructions: write steps freely.",
            )

            new_photos = st.file_uploader(
                "Optional: add photos now (png/jpg/jpeg/webp)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="proto_create_photos",
            )

            add = st.form_submit_button("Add protocol", use_container_width=True)
            if add:
                if not (new_name and new_text):
                    st.error("Please fill **name** and **instructions**.")
                else:
                    # create first (so we have a pid for filenames)
                    proto_obj = {
                        "name": new_name.strip(),
                        "type": new_type,
                        "sub_type": new_sub,
                        "instructions": new_text.strip(),
                        "images": [],
                    }
                    pid_new = _proto_id(proto_obj)

                    imgs = []
                    if new_photos:
                        for uf in new_photos:
                            try:
                                imgs.append(_save_uploaded_image(uf, pid_new))
                            except Exception as e:
                                st.warning(f"Failed saving image: {e}")

                    proto_obj["images"] = imgs

                    st.session_state["protocols"].append(proto_obj)
                    _safe_write_json(PROTOCOLS_FILE, st.session_state["protocols"])
                    st.success(f"Added: {new_name}")
                    st.rerun()
# ------------------ Maintenance Tab ------------------
elif tab_selection == "🧰 Maintenance":
    import os, json, glob, time
    import datetime as dt

    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    import duckdb

    st.title("🧰 Maintenance")
    st.caption(
        "Auto-loads ALL maintenance files from /maintenance. "
        "Furnace + UV1 + UV2 hours are persisted to maintenance/_app_state.json. "
        "New draw detection shows an inline Pre/Post checklist."
    )

    # =========================================================
    # Small utils
    # =========================================================
    def safe_str(x) -> str:
        try:
            if x is None:
                return ""
            if isinstance(x, float) and np.isnan(x):
                return ""
            return str(x)
        except Exception:
            return ""

    # =========================================================
    # Paths
    # =========================================================
    BASE_DIR = os.getcwd()
    MAINT_FOLDER = P.maintenance_dir
    DRAW_FOLDER = os.path.join(BASE_DIR, P.dataset_dir)   # dataset CSVs (summary)
    LOGS_FOLDER = os.path.join(BASE_DIR, P.logs_dir)      # ✅ LOG CSVs (MFC actual)
    STATE_PATH = os.path.join(MAINT_FOLDER, "_app_state.json")
    os.makedirs(MAINT_FOLDER, exist_ok=True)

    # ✅ Append-only CSV logs (for SQL Lab line-search)
    MAINT_ACTIONS_CSV = os.path.join(MAINT_FOLDER, "maintenance_actions_log.csv")
    FAULTS_CSV = os.path.join(MAINT_FOLDER, "faults_log.csv")
    FAULTS_ACTIONS_CSV = os.path.join(MAINT_FOLDER, "faults_actions_log.csv")

    MAINT_ACTIONS_COLS = [
        "maintenance_id",
        "maintenance_ts",
        "maintenance_component",
        "maintenance_task",
        "maintenance_task_id",
        "maintenance_mode",
        "maintenance_hours_source",
        "maintenance_done_date",
        "maintenance_done_hours",
        "maintenance_done_draw",
        "maintenance_source_file",
        "maintenance_actor",
        "maintenance_note",
    ]

    FAULTS_COLS = [
        "fault_id",
        "fault_ts",
        "fault_component",
        "fault_title",
        "fault_description",
        "fault_severity",
        "fault_actor",
        "fault_source_file",
        "fault_related_draw",
    ]

    # ✅ Fault actions (close/reopen/notes) — append-only
    FAULTS_ACTIONS_COLS = [
        "fault_action_id",
        "fault_id",
        "action_ts",
        "action_type",     # close / reopen / note
        "actor",
        "note",
        "fix_summary",
    ]

    # =========================================================
    # DuckDB connection (shared with SQL Lab)
    # =========================================================
    if "sql_duck_con" not in st.session_state:
        st.session_state["sql_duck_con"] = duckdb.connect(os.path.join(BASE_DIR, P.duckdb_path))
    con = st.session_state["sql_duck_con"]
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass

    # =========================================================
    # Create DB tables
    # =========================================================
    con.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_tasks (
        task_key            VARCHAR,
        task_id             VARCHAR,
        component           VARCHAR,
        task                VARCHAR,
        tracking_mode       VARCHAR,
        hours_source        VARCHAR,
        interval_value      VARCHAR,
        interval_unit       VARCHAR,
        due_threshold_days  VARCHAR,
        manual_name         VARCHAR,
        page                VARCHAR,
        document            VARCHAR,
        procedure_summary   VARCHAR,
        notes               VARCHAR,
        owner               VARCHAR,
        source_file         VARCHAR,
        loaded_at           TIMESTAMP
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_actions (
        action_id       BIGINT,
        action_ts       TIMESTAMP,
        component       VARCHAR,
        task            VARCHAR,
        task_id         VARCHAR,
        tracking_mode   VARCHAR,
        hours_source    VARCHAR,
        done_date       DATE,
        done_hours      DOUBLE,
        done_draw       INTEGER,
        source_file     VARCHAR,
        actor           VARCHAR,
        note            VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_events (
        fault_id        BIGINT,
        fault_ts        TIMESTAMP,
        component       VARCHAR,
        title           VARCHAR,
        description     VARCHAR,
        severity        VARCHAR,
        actor           VARCHAR,
        source_file     VARCHAR,
        related_draw    VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_actions (
        fault_action_id  BIGINT,
        fault_id         BIGINT,
        action_ts        TIMESTAMP,
        action_type      VARCHAR,
        actor            VARCHAR,
        note             VARCHAR,
        fix_summary      VARCHAR
    );
    """)

    # =========================================================
    # Persistent state helpers
    # =========================================================
    def load_state(path: str) -> dict:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save_state(path: str, state: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def _sanitize(o):
            if isinstance(o, (dt.date, dt.datetime)):
                return o.isoformat()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        clean = {k: _sanitize(v) for k, v in state.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)

    state = load_state(STATE_PATH)

    # =========================================================
    # CSV helpers (append-only)
    # =========================================================
    def _ensure_csv(path: str, cols: list):
        if not os.path.isfile(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)

    def _append_csv(path: str, cols: list, df_rows: pd.DataFrame):
        _ensure_csv(path, cols)
        df = df_rows.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

        # stringify time fields to avoid dtype crash
        for tcol in [c for c in cols if c.endswith("_ts")]:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        for dcol in [c for c in cols if c.endswith("_date")]:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.strftime("%Y-%m-%d")

        df.to_csv(path, mode="a", header=False, index=False)

    def _read_csv_safe(path: str, cols: list) -> pd.DataFrame:
        if not os.path.isfile(path):
            return pd.DataFrame(columns=cols)
        try:
            df = pd.read_csv(path)
            if df is None:
                return pd.DataFrame(columns=cols)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols].copy()
        except Exception:
            return pd.DataFrame(columns=cols)

    def _latest_fault_state(actions_df: pd.DataFrame) -> dict:
        """
        fault_id -> last action (close/reopen/note) ; closed if last action is 'close'
        """
        out = {}
        if actions_df is None or actions_df.empty:
            return out

        a = actions_df.copy()
        a["action_ts"] = pd.to_datetime(a["action_ts"], errors="coerce")
        a["fault_id"] = pd.to_numeric(a["fault_id"], errors="coerce")
        a = a.dropna(subset=["fault_id"]).copy()
        a["fault_id"] = a["fault_id"].astype(int)

        a = a.sort_values(["fault_id", "action_ts"], ascending=[True, True])
        last = a.groupby("fault_id").tail(1)

        for _, r in last.iterrows():
            fid = int(r["fault_id"])
            typ = safe_str(r.get("action_type", "")).strip().lower()
            out[fid] = {
                "is_closed": (typ == "close"),
                "last_ts": r.get("action_ts", None),
                "last_note": safe_str(r.get("note", "")),
                "last_fix": safe_str(r.get("fix_summary", "")),
                "last_type": typ,
                "last_actor": safe_str(r.get("actor", "")),
            }
        return out

    def _write_fault_action(con, *, fault_id: int, action_type: str, actor: str, note: str = "", fix_summary: str = ""):
        now_dt = dt.datetime.now()
        aid = int(time.time() * 1000)

        # DuckDB
        try:
            con.execute("""
                INSERT INTO faults_actions
                (fault_action_id, fault_id, action_ts, action_type, actor, note, fix_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [aid, int(fault_id), now_dt, str(action_type), str(actor), str(note), str(fix_summary)])
        except Exception as e:
            st.warning(f"Fault action DB insert failed (still saving CSV): {e}")

        # CSV
        row = pd.DataFrame([{
            "fault_action_id": aid,
            "fault_id": int(fault_id),
            "action_ts": now_dt,
            "action_type": str(action_type),
            "actor": str(actor),
            "note": str(note),
            "fix_summary": str(fix_summary),
        }])
        _append_csv(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS, row)

    # =========================================================
    # Draw count helper
    # =========================================================
    def get_draw_csv_count(folder: str) -> int:
        if not os.path.isdir(folder):
            return 0
        return sum(
            1 for f in os.listdir(folder)
            if f.lower().endswith(".csv") and not f.startswith("~$")
        )

    current_draw_count = get_draw_csv_count(DRAW_FOLDER)

    # =========================================================
    # Maintenance file loading
    # =========================================================
    files = [f for f in os.listdir(MAINT_FOLDER) if f.lower().endswith((".xlsx", ".xls", ".csv"))]
    if not files:
        st.warning("No maintenance files found in /maintenance folder.")
        st.stop()

    normalize_map = {
        "equipment": "Component",
        "task name": "Task",
        "task id": "Task_ID",
        "interval type": "Interval_Type",
        "interval value": "Interval_Value",
        "interval unit": "Interval_Unit",
        "tracking mode": "Tracking_Mode",
        "hours source": "Hours_Source",
        "calendar rule": "Calendar_Rule",
        "due threshold (days)": "Due_Threshold_Days",
        "document name": "Manual_Name",
        "document file/link": "Document",
        "manual page": "Page",
        "procedure summary": "Procedure_Summary",
        "safety/notes": "Notes",
        "owner": "Owner",
        "last done date": "Last_Done_Date",
        "last done hours": "Last_Done_Hours",
        "last done draw": "Last_Done_Draw",
    }
    inverse_map = {v: k for k, v in normalize_map.items()}

    REQUIRED = ["Component", "Task", "Tracking_Mode"]
    OPTIONAL = [
        "Task_ID",
        "Interval_Type", "Interval_Value", "Interval_Unit",
        "Due_Threshold_Days",
        "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
        "Manual_Name", "Page", "Document",
        "Procedure_Summary", "Notes", "Owner",
        "Hours_Source", "Calendar_Rule",
    ]

    def read_file(path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)

    def write_file(path: str, df: pd.DataFrame):
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)

    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.rename(columns={c: normalize_map.get(str(c).strip().lower(), c) for c in df.columns}, inplace=True)
        for r in REQUIRED:
            if r not in df.columns:
                df[r] = np.nan
        for c in OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan
        return df

    def templateize_df(df_internal: pd.DataFrame, original_cols: list) -> pd.DataFrame:
        df = df_internal.copy()
        rename_back = {}
        for internal_col, template_key_lower in inverse_map.items():
            match = None
            for oc in original_cols:
                if str(oc).strip().lower() == template_key_lower:
                    match = oc
                    break
            if match is not None and internal_col in df.columns:
                rename_back[internal_col] = match
        df.rename(columns=rename_back, inplace=True)
        return df

    frames = []
    load_errors = []
    for fname in sorted(files):
        fpath = os.path.join(MAINT_FOLDER, fname)
        try:
            raw = read_file(fpath)
            if raw is None or raw.empty:
                continue
            df = normalize_df(raw)
            df["Source_File"] = fname
            frames.append(df)
        except ImportError as e:
            st.error("Excel engine missing. Install openpyxl in your .venv:")
            st.code("pip install openpyxl", language="bash")
            st.exception(e)
            st.stop()
        except Exception as e:
            load_errors.append((fname, str(e)))

    if not frames:
        st.error("No valid maintenance data could be loaded.")
        if load_errors:
            st.dataframe(pd.DataFrame(load_errors, columns=["File", "Error"]), use_container_width=True)
        st.stop()

    dfm = pd.concat(frames, ignore_index=True)

    # =========================================================
    # Persisted inputs (hours + settings)
    # =========================================================
    def _persist_inputs():
        state["current_date"] = dt.date.today().isoformat()
        state["furnace_hours"] = float(st.session_state.get("maint_furnace_hours", 0.0))
        state["uv1_hours"] = float(st.session_state.get("maint_uv1_hours", 0.0))
        state["uv2_hours"] = float(st.session_state.get("maint_uv2_hours", 0.0))
        state["warn_days"] = int(st.session_state.get("maint_warn_days", 14))
        state["warn_hours"] = float(st.session_state.get("maint_warn_hours", 50.0))
        save_state(STATE_PATH, state)
        st.session_state["furnace_hours"] = state["furnace_hours"]
        st.session_state["uv1_hours"] = state["uv1_hours"]
        st.session_state["uv2_hours"] = state["uv2_hours"]

    default_furnace = float(state.get("furnace_hours", 0.0) or 0.0)
    default_uv1 = float(state.get("uv1_hours", 0.0) or 0.0)
    default_uv2 = float(state.get("uv2_hours", 0.0) or 0.0)
    default_warn_days = int(state.get("warn_days", 14) or 14)
    default_warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

    st.subheader("Current status inputs (saved)")
    current_date = dt.date.today()
    st.date_input("Today", value=current_date, disabled=True)

    c2, c3, c4, c5 = st.columns([1, 1, 1, 1])
    with c2:
        furnace_hours = st.number_input(
            "Furnace hours", min_value=0.0, value=default_furnace, step=1.0,
            key="maint_furnace_hours", on_change=_persist_inputs
        )
    with c3:
        uv1_hours = st.number_input(
            "UV System 1 hours", min_value=0.0, value=default_uv1, step=1.0,
            key="maint_uv1_hours", on_change=_persist_inputs
        )
    with c4:
        uv2_hours = st.number_input(
            "UV System 2 hours", min_value=0.0, value=default_uv2, step=1.0,
            key="maint_uv2_hours", on_change=_persist_inputs
        )
    with c5:
        warn_days = st.number_input(
            "Warn if due within (days)", min_value=0, value=default_warn_days, step=1,
            key="maint_warn_days", on_change=_persist_inputs
        )

    warn_hours = st.number_input(
        "Warn if due within (hours)", min_value=0.0, value=default_warn_hours, step=1.0,
        key="maint_warn_hours", on_change=_persist_inputs
    )

    _persist_inputs()
    st.caption("Hours-based tasks use **Hours Source**: FURNACE / UV1 / UV2. If empty → defaults to FURNACE.")

    # =========================================================
    # Actor
    # =========================================================
    st.session_state.setdefault("maint_actor", "operator")
    st.text_input("Actor / operator name (for history)", key="maint_actor")
    actor = st.session_state.get("maint_actor", "operator")

    # =========================================================
    # Helpers
    # =========================================================
    def parse_date(x):
        if pd.isna(x) or x == "":
            return None
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date()

    def parse_float(x):
        if pd.isna(x) or x == "":
            return None
        try:
            return float(x)
        except Exception:
            return None

    def parse_int(x):
        if pd.isna(x) or x == "":
            return None
        try:
            return int(float(x))
        except Exception:
            return None

    def norm_source(s) -> str:
        s = "" if s is None or pd.isna(s) else str(s)
        return s.strip().lower()

    def pick_current_hours(hours_source: str) -> float:
        hs = norm_source(hours_source)
        if hs in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
            return float(uv2_hours)
        if hs in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
            return float(uv1_hours)
        return float(furnace_hours)

    def mode_norm(x: str) -> str:
        s = "" if x is None or pd.isna(x) else str(x).strip().lower()
        if s in ("draw", "draws", "draws_count", "draw_count"):
            return "draws"
        return s

    # =========================================================
    # Compute Next Due + Status
    # =========================================================
    dfm["Last_Done_Date_parsed"] = dfm["Last_Done_Date"].apply(parse_date)
    dfm["Last_Done_Hours_parsed"] = dfm["Last_Done_Hours"].apply(parse_float)
    dfm["Last_Done_Draw_parsed"] = dfm["Last_Done_Draw"].apply(parse_int)
    dfm["Current_Hours_For_Task"] = dfm["Hours_Source"].apply(pick_current_hours)
    dfm["Tracking_Mode_norm"] = dfm["Tracking_Mode"].apply(mode_norm)

    def next_due_date(row):
        if row.get("Tracking_Mode_norm") != "calendar":
            return None
        last = row.get("Last_Done_Date_parsed", None)
        if last is None:
            return None
        try:
            v = int(float(row.get("Interval_Value", np.nan)))
        except Exception:
            return None

        unit = str(row.get("Interval_Unit", "")).strip().lower()
        base = pd.Timestamp(last)
        if pd.isna(base) or base is pd.NaT:
            return None

        if "day" in unit:
            out = base + pd.DateOffset(days=v)
        elif "week" in unit:
            out = base + pd.DateOffset(weeks=v)
        elif "month" in unit:
            out = base + pd.DateOffset(months=v)
        elif "year" in unit:
            out = base + pd.DateOffset(years=v)
        else:
            out = base + pd.DateOffset(days=v)

        if pd.isna(out) or out is pd.NaT:
            return None
        return out.date()

    def next_due_hours(row):
        if row.get("Tracking_Mode_norm") != "hours":
            return None
        last_h = row.get("Last_Done_Hours_parsed", None)
        if last_h is None:
            return None
        try:
            v = float(row.get("Interval_Value", np.nan))
        except Exception:
            return None
        if pd.isna(v):
            return None
        return float(last_h) + float(v)

    def next_due_draw(row):
        if row.get("Tracking_Mode_norm") != "draws":
            return None
        last_d = row.get("Last_Done_Draw_parsed", None)
        if last_d is None:
            return None
        try:
            v = int(float(row.get("Interval_Value", np.nan)))
        except Exception:
            return None
        return int(last_d) + int(v)

    dfm["Next_Due_Date"] = dfm.apply(next_due_date, axis=1)
    dfm["Next_Due_Hours"] = dfm.apply(next_due_hours, axis=1)
    dfm["Next_Due_Draw"] = dfm.apply(next_due_draw, axis=1)

    def status_row(row):
        mode = row.get("Tracking_Mode_norm", "")
        if mode == "event":
            return "ROUTINE"

        overdue = False
        due_soon = False

        nd = row.get("Next_Due_Date", None)
        nh = row.get("Next_Due_Hours", None)
        ndr = row.get("Next_Due_Draw", None)

        # calendar
        if nd is not None and not pd.isna(nd):
            if nd < current_date:
                overdue = True
            else:
                thresh = row.get("Due_Threshold_Days", np.nan)
                try:
                    thresh = int(float(thresh)) if not pd.isna(thresh) else int(warn_days)
                except Exception:
                    thresh = int(warn_days)
                if (nd - current_date).days <= thresh:
                    due_soon = True

        # hours
        if nh is not None and not pd.isna(nh):
            nh = float(nh)
            cur_h = float(row.get("Current_Hours_For_Task", 0.0))
            if nh < cur_h:
                overdue = True
            elif (nh - cur_h) <= float(warn_hours):
                due_soon = True

        # draws
        if ndr is not None and not pd.isna(ndr):
            ndr = int(ndr)
            if ndr < int(current_draw_count):
                overdue = True
            elif (ndr - int(current_draw_count)) <= 5:
                due_soon = True

        if overdue:
            return "OVERDUE"
        if due_soon:
            return "DUE SOON"
        return "OK"

    dfm["Status"] = dfm.apply(status_row, axis=1)

    st.session_state["maint_overdue"] = int((dfm["Status"] == "OVERDUE").sum())
    st.session_state["maint_due_soon"] = int((dfm["Status"] == "DUE SOON").sum())

    # =========================================================
    # Dashboard metrics + Open Critical Faults
    # =========================================================
    def get_open_faults_counts():
        faults_csv = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
        actions_csv = _read_csv_safe(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS)
        smap = _latest_fault_state(actions_csv)

        if faults_csv.empty:
            return 0, 0

        faults_csv["fault_id"] = pd.to_numeric(faults_csv["fault_id"], errors="coerce")
        faults_csv = faults_csv.dropna(subset=["fault_id"]).copy()
        faults_csv["fault_id"] = faults_csv["fault_id"].astype(int)

        faults_csv["_is_closed"] = faults_csv["fault_id"].apply(lambda fid: bool(smap.get(int(fid), {}).get("is_closed", False)))
        open_df = faults_csv[~faults_csv["_is_closed"]].copy()

        crit_open = int((open_df["fault_severity"].astype(str).str.lower() == "critical").sum()) if not open_df.empty else 0
        open_total = int(len(open_df))
        return open_total, crit_open

    def render_maintenance_dashboard_metrics(dfm):
        st.subheader("Dashboard")
        overdue = int((dfm["Status"] == "OVERDUE").sum())
        due_soon = int((dfm["Status"] == "DUE SOON").sum())
        routine = int((dfm["Status"] == "ROUTINE").sum())
        ok = int((dfm["Status"] == "OK").sum())
        open_faults, crit_faults = get_open_faults_counts()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("OVERDUE", overdue)
        c2.metric("DUE SOON", due_soon)
        c3.metric("ROUTINE", routine)
        c4.metric("OK", ok)
        c5.metric("🚨 Open faults", open_faults)
        c6.metric("🟥 Critical open", crit_faults)

    # =========================================================
    # Horizon selector + roadmaps
    # =========================================================
    def render_maintenance_horizon_selector(current_draw_count: int):
        st.subheader("📅 Future schedule view")

        st.markdown(
            """
            <style>
            div.stButton > button {
                width: 100%;
                height: 44px;
                border-radius: 12px;
                font-weight: 600;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.session_state.setdefault("maint_horizon_hours", 10)
        st.session_state.setdefault("maint_horizon_days", 7)
        st.session_state.setdefault("maint_horizon_draws", 5)

        def button_group(title, options, value, key):
            st.caption(title)
            cols = st.columns(len(options))
            for col, (label, v) in zip(cols, options):
                if col.button(label, key=f"{key}_{v}", type="primary" if v == value else "secondary"):
                    return v
            return value

        c1, c2, c3 = st.columns(3)

        with c1:
            st.session_state["maint_horizon_hours"] = button_group(
                "Hours horizon",
                [("10", 10), ("50", 50), ("100", 100)],
                st.session_state["maint_horizon_hours"],
                "mh"
            )

        with c2:
            st.session_state["maint_horizon_days"] = button_group(
                "Calendar horizon",
                [("Week", 7), ("Month", 30), ("3 Months", 90)],
                st.session_state["maint_horizon_days"],
                "md"
            )

        with c3:
            st.session_state["maint_horizon_draws"] = button_group(
                "Draw horizon",
                [("5", 5), ("10", 10), ("50", 50)],
                st.session_state["maint_horizon_draws"],
                "mD"
            )

        st.caption(
            f"📦 Now: **{current_draw_count}** → "
            f"Horizon: **{st.session_state['maint_horizon_draws']}** → "
            f"Up to draw **#{current_draw_count + st.session_state['maint_horizon_draws']}**"
        )

        return (
            st.session_state["maint_horizon_hours"],
            st.session_state["maint_horizon_days"],
            st.session_state["maint_horizon_draws"],
        )

    def render_maintenance_roadmaps(
        dfm: pd.DataFrame,
        current_date,
        current_draw_count: int,
        furnace_hours: float,
        uv1_hours: float,
        uv2_hours: float,
        horizon_hours: int,
        horizon_days: int,
        horizon_draws: int,
    ):
        def status_color(s):
            s = str(s).upper()
            if s == "OVERDUE":
                return "#ff4d4d"
            if s == "DUE SOON":
                return "#ffcc00"
            return "#66ff99"

        def roadmap(x0, x1, title, xlabel, df, xcol, hover):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[0, 0], mode="lines",
                line=dict(width=6, color="rgba(180,180,180,0.2)"),
                hoverinfo="skip"
            ))
            fig.add_vline(x=x0, line_dash="dash")

            if df is not None and not df.empty:
                fig.add_trace(go.Scatter(
                    x=df[xcol],
                    y=[0] * len(df),
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=[status_color(s) for s in df["Status"]],
                        line=dict(width=1, color="rgba(255,255,255,0.5)")
                    ),
                    text=df[hover],
                    hovertemplate="%{text}<extra></extra>",
                ))
            else:
                mid = x0 + (x1 - x0) / 2
                fig.add_annotation(x=mid, y=0, text="No tasks in horizon", showarrow=False)

            fig.update_layout(
                title=title,
                height=220,
                yaxis=dict(visible=False),
                xaxis=dict(title=xlabel),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return fig

        def norm_group(src):
            s = str(src).lower()
            if "uv1" in s:
                return "UV1"
            if "uv2" in s:
                return "UV2"
            return "FURNACE"

        hours_df = dfm[dfm["Tracking_Mode_norm"] == "hours"].copy()
        hours_df["Due"] = pd.to_numeric(hours_df["Next_Due_Hours"], errors="coerce")
        hours_df = hours_df.dropna(subset=["Due"])
        hours_df["Group"] = hours_df["Hours_Source"].apply(norm_group)
        hours_df["Hover"] = hours_df["Component"] + " — " + hours_df["Task"] + "<br>Status: " + hours_df["Status"]

        cal_df = dfm[dfm["Tracking_Mode_norm"] == "calendar"].copy()
        cal_df["Due"] = pd.to_datetime(cal_df["Next_Due_Date"], errors="coerce")
        cal_df = cal_df.dropna(subset=["Due"])
        cal_df["Hover"] = cal_df["Component"] + " — " + cal_df["Task"] + "<br>Status: " + cal_df["Status"]

        draw_df = dfm[dfm["Tracking_Mode_norm"] == "draws"].copy()
        draw_df["Due"] = pd.to_numeric(draw_df["Next_Due_Draw"], errors="coerce")
        draw_df = draw_df.dropna(subset=["Due"])
        draw_df["Hover"] = draw_df["Component"] + " — " + draw_df["Task"] + "<br>Status: " + draw_df["Status"]

        st.markdown("### 🔥 Furnace / 💡 UV timelines")
        c1, c2, c3 = st.columns(3)

        with c1:
            x0, x1 = furnace_hours, furnace_hours + horizon_hours
            st.plotly_chart(
                roadmap(x0, x1, "FURNACE", "Hours",
                        hours_df[(hours_df["Group"] == "FURNACE") & hours_df["Due"].between(x0, x1)],
                        "Due", "Hover"),
                use_container_width=True
            )

        with c2:
            x0, x1 = uv1_hours, uv1_hours + horizon_hours
            st.plotly_chart(
                roadmap(x0, x1, "UV1", "Hours",
                        hours_df[(hours_df["Group"] == "UV1") & hours_df["Due"].between(x0, x1)],
                        "Due", "Hover"),
                use_container_width=True
            )

        with c3:
            x0, x1 = uv2_hours, uv2_hours + horizon_hours
            st.plotly_chart(
                roadmap(x0, x1, "UV2", "Hours",
                        hours_df[(hours_df["Group"] == "UV2") & hours_df["Due"].between(x0, x1)],
                        "Due", "Hover"),
                use_container_width=True
            )

        st.markdown("### 🧵 Draw timeline")
        d0, d1 = current_draw_count, current_draw_count + horizon_draws
        st.plotly_chart(
            roadmap(d0, d1, "Draw-based tasks", "Draw #",
                    draw_df[draw_df["Due"].between(d0, d1)],
                    "Due", "Hover"),
            use_container_width=True
        )

        st.markdown("### 🗓️ Calendar timeline")
        t0 = pd.Timestamp(current_date)
        t1 = t0 + pd.Timedelta(days=horizon_days)
        st.plotly_chart(
            roadmap(t0, t1, "Calendar tasks", "Date",
                    cal_df[(cal_df["Due"] >= t0) & (cal_df["Due"] <= t1)],
                    "Due", "Hover"),
            use_container_width=True
        )

    # =========================================================
    # Done editor + apply done (updates + logs DB + CSV)
    # =========================================================
    def render_maintenance_done_editor(dfm):
        st.subheader("Mark tasks as done")

        focus_default = ["OVERDUE", "DUE SOON", "ROUTINE"]
        focus_status = st.multiselect(
            "Work on these statuses",
            ["OVERDUE", "DUE SOON", "ROUTINE", "OK"],
            default=focus_default,
            key="maint_focus_status"
        )

        work = (
            dfm[dfm["Status"].isin(focus_status)]
            .copy()
            .sort_values(["Status", "Component", "Task"])
        )
        work["Done_Now"] = False

        cols = [
            "Done_Now",
            "Status", "Component", "Task", "Task_ID",
            "Tracking_Mode", "Hours_Source", "Current_Hours_For_Task",
            "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
            "Next_Due_Date", "Next_Due_Hours", "Next_Due_Draw",
            "Manual_Name", "Page", "Document",
            "Owner", "Source_File"
        ]
        cols = [c for c in cols if c in work.columns]

        edited = st.data_editor(
            work[cols],
            use_container_width=True,
            height=420,
            column_config={
                "Done_Now": st.column_config.CheckboxColumn("Done now", help="Tick tasks you completed")
            },
            disabled=[c for c in cols if c != "Done_Now"],
            key="maint_editor"
        )
        return edited

    def render_maintenance_apply_done(
        edited,
        *,
        dfm,
        current_date,
        current_draw_count,
        actor,
        MAINT_FOLDER,
        con,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
        pick_current_hours,
        mode_norm,
    ):
        if not st.button("✅ Apply 'Done Now' updates", type="primary"):
            return

        done_rows = edited[edited["Done_Now"] == True].copy()
        if done_rows.empty:
            st.info("No tasks selected.")
            return

        updated = 0
        problems = []

        # ---- Update source files ----
        for src, grp in done_rows.groupby("Source_File"):
            path = os.path.join(MAINT_FOLDER, src)
            try:
                raw = read_file(path)
                df_src = normalize_df(raw)

                for _, r in grp.iterrows():
                    mode = mode_norm(r.get("Tracking_Mode", ""))

                    mask = (
                        df_src["Component"].astype(str).eq(str(r.get("Component", ""))) &
                        df_src["Task"].astype(str).eq(str(r.get("Task", "")))
                    )
                    if not mask.any():
                        continue

                    df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()

                    if mode == "hours":
                        df_src.loc[mask, "Last_Done_Hours"] = float(pick_current_hours(r.get("Hours_Source", "")))
                    elif mode == "draws":
                        df_src.loc[mask, "Last_Done_Draw"] = int(current_draw_count)

                    updated += int(mask.sum())

                out = templateize_df(df_src, list(raw.columns))
                write_file(path, out)

            except Exception as e:
                problems.append((src, str(e)))

        st.success(f"Updated {updated} task(s).")

        # ---- Log to DuckDB + CSV line log ----
        now_dt = dt.datetime.now()
        csv_rows = []

        for _, r in done_rows.iterrows():
            action_id = int(time.time() * 1000)
            mode = mode_norm(r.get("Tracking_Mode", ""))

            hs_raw = r.get("Hours_Source", "")
            hs_str = "" if hs_raw is None or (isinstance(hs_raw, float) and np.isnan(hs_raw)) else str(hs_raw).strip()
            if hs_str == "":
                hs_str = "FURNACE"

            # ALWAYS snapshot hours (for filtering/search)
            hours_snapshot = float(pick_current_hours(hs_str))

            done_hours_db = None
            done_draw = None
            if mode == "hours":
                done_hours_db = hours_snapshot
            elif mode == "draws":
                done_draw = int(current_draw_count)

            try:
                con.execute("""
                    INSERT INTO maintenance_actions
                    (action_id, action_ts, component, task, task_id, tracking_mode, hours_source,
                     done_date, done_hours, done_draw, source_file, actor, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    action_id,
                    now_dt,
                    str(r.get("Component", "")),
                    str(r.get("Task", "")),
                    str(r.get("Task_ID", "")),
                    str(r.get("Tracking_Mode", "")),
                    hs_str,
                    current_date,
                    done_hours_db,
                    done_draw,
                    str(r.get("Source_File", "")),
                    str(actor),
                    "",
                ])
            except Exception as e:
                st.warning(f"DuckDB insert failed (still saving CSV log): {e}")

            csv_rows.append({
                "maintenance_id": action_id,
                "maintenance_ts": now_dt,
                "maintenance_component": str(r.get("Component", "")),
                "maintenance_task": str(r.get("Task", "")),
                "maintenance_task_id": str(r.get("Task_ID", "")),
                "maintenance_mode": str(r.get("Tracking_Mode", "")),
                "maintenance_hours_source": hs_str,
                "maintenance_done_date": current_date,
                "maintenance_done_hours": hours_snapshot,  # ✅ always filled
                "maintenance_done_draw": done_draw if done_draw is not None else "",
                "maintenance_source_file": str(r.get("Source_File", "")),
                "maintenance_actor": str(actor),
                "maintenance_note": "",
            })

        if csv_rows:
            try:
                _append_csv(MAINT_ACTIONS_CSV, MAINT_ACTIONS_COLS, pd.DataFrame(csv_rows))
                st.caption("✅ Logged maintenance lines to maintenance_actions_log.csv")
            except Exception as e:
                st.error(f"Failed writing maintenance_actions_log.csv: {e}")

        if problems:
            st.warning("Some files had issues:")
            st.dataframe(pd.DataFrame(problems, columns=["File", "Error"]), use_container_width=True)

        st.rerun()

    # =========================================================
    # History viewer (DuckDB + CSV)
    # =========================================================
    def render_maintenance_history(con, limit: int = 200, height: int = 320):
        with st.expander("🗃️ Maintenance history (DuckDB)", expanded=False):
            try:
                recent = con.execute(f"""
                    SELECT action_ts, component, task, tracking_mode, hours_source,
                           done_date, done_hours, done_draw, actor, source_file
                    FROM maintenance_actions
                    ORDER BY action_ts DESC
                    LIMIT {int(limit)}
                """).fetchdf()

                if not recent.empty:
                    recent["done_date"] = pd.to_datetime(recent["done_date"], errors="coerce").dt.date
                    recent["action_ts"] = pd.to_datetime(recent["action_ts"], errors="coerce")

                st.dataframe(recent, use_container_width=True, height=int(height))
            except Exception as e:
                st.warning(f"DB read failed: {e}")

        with st.expander("🧾 Maintenance lines (CSV log)", expanded=False):
            if not os.path.isfile(MAINT_ACTIONS_CSV):
                st.info("No maintenance_actions_log.csv yet (mark something done first).")
            else:
                try:
                    df = pd.read_csv(MAINT_ACTIONS_CSV)
                    st.dataframe(df.tail(250), use_container_width=True, height=360)
                except Exception as e:
                    st.warning(f"CSV read failed: {e}")

    def render_gas_report(LOGS_FOLDER: str):
        """
        Gas usage report (MFC ACTUAL)
        Assumptions:
        - MFC columns contain BOTH 'MFC' and 'Actual'
        - Units are SLM (Standard Liters per Minute)
        - Integration: SL = Σ(SLM × dt_minutes)
        """

        st.markdown("---")
        st.subheader("🧪 Gas usage report (MFC actual, SLM)")

        show = st.toggle("Show gas report", value=False, key="gasrep_show")
        if not show:
            st.caption("(Hidden by default to keep UI light)")
            return

        if not os.path.isdir(LOGS_FOLDER):
            st.warning(f"Logs folder not found: {LOGS_FOLDER}")
            return

        # --------------------------------------------------
        # Collect log files
        # --------------------------------------------------
        csv_files = sorted(
            [os.path.join(LOGS_FOLDER, f)
             for f in os.listdir(LOGS_FOLDER)
             if f.lower().endswith(".csv") and not f.startswith("~$")],
            key=lambda p: os.path.getmtime(p),
        )

        if not csv_files:
            st.info("No log CSV files found.")
            return

        st.caption(f"Found {len(csv_files)} log files.")

        # --------------------------------------------------
        # Reports folder (auto-save)
        # --------------------------------------------------
        BASE_DIR_LOCAL = os.getcwd()
        REPORT_DIR = os.path.join(BASE_DIR_LOCAL, "reports", "gas")
        os.makedirs(REPORT_DIR, exist_ok=True)
        st.caption(f"Reports folder: {REPORT_DIR}")

        # --------------------------------------------------
        # Time window selector
        # --------------------------------------------------
        st.markdown("#### Time window")
        c1, c2, c3, c4 = st.columns([1,1,1,2])

        st.session_state.setdefault("gasrep_window_days", 30)

        with c1:
            if st.button("Last 7 days", key="gasrep_btn_7", use_container_width=True):
                st.session_state["gasrep_window_days"] = 7
        with c2:
            if st.button("Last 30 days", key="gasrep_btn_30", use_container_width=True):
                st.session_state["gasrep_window_days"] = 30
        with c3:
            if st.button("Last 90 days", key="gasrep_btn_90", use_container_width=True):
                st.session_state["gasrep_window_days"] = 90
        with c4:
            st.caption(f"Selected: {st.session_state['gasrep_window_days']} days")

        window_days = int(st.session_state.get("gasrep_window_days", 30))

        # --------------------------------------------------
        # Helpers
        # --------------------------------------------------
        def _norm(s):
            return str(s).strip().lower()

        def _find_time_col(cols):
            for c in cols:
                if _norm(c) in {"date/time","datetime","timestamp","date time"}:
                    return c
            for c in cols:
                if "date" in _norm(c) and "time" in _norm(c):
                    return c
            return None

        def _is_mfc_actual(c):
            s = _norm(c)
            return ("mfc" in s) and ("actual" in s)

        # --------------------------------------------------
        # Scan logs and integrate usage
        # --------------------------------------------------
        rows = []

        for p in csv_files:
            try:
                df = pd.read_csv(p)
                if df is None or df.empty:
                    continue

                time_col = _find_time_col(df.columns)
                if not time_col:
                    continue

                t = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
                if t.isna().all():
                    continue

                df["__t"] = t
                df = df.dropna(subset=["__t"]).sort_values("__t").reset_index(drop=True)
                if len(df) < 2:
                    continue

                # dt in minutes
                dt_min = df["__t"].diff().dt.total_seconds() / 60.0
                dt_min = dt_min.fillna(0.0).clip(lower=0.0)

                mfc_cols = [c for c in df.columns if _is_mfc_actual(c)]
                if not mfc_cols:
                    continue

                total_sl = 0.0
                for c in mfc_cols:
                    flow_slm = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    total_sl += float((flow_slm * dt_min).sum())

                rows.append({
                    "log_file": os.path.basename(p),
                    "start_time": df["__t"].iloc[0],
                    "end_time": df["__t"].iloc[-1],
                    "duration_min": float(dt_min.sum()),
                    "Total SL": total_sl,
                })
            except Exception:
                continue

        if not rows:
            st.info("No usable MFC ACTUAL data detected in logs.")
            return

        usage = pd.DataFrame(rows)
        usage["start_time"] = pd.to_datetime(usage["start_time"], errors="coerce")
        usage = usage.sort_values("start_time").reset_index(drop=True)

        # --------------------------------------------------
        # Apply time window
        # --------------------------------------------------
        latest = usage["start_time"].max()
        if pd.isna(latest):
            latest = pd.Timestamp.now()

        t0 = latest - pd.Timedelta(days=window_days)
        usage = usage[usage["start_time"] >= t0]

        if usage.empty:
            st.warning("No logs in selected window.")
            return

        # --------------------------------------------------
        # Summary metrics
        # --------------------------------------------------
        total_sl = float(usage["Total SL"].sum())
        total_hours = float(usage["duration_min"].sum()) / 60.0
        avg_slm = (total_sl / usage["duration_min"].sum()) if usage["duration_min"].sum() > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Gas Used (SL)", f"{total_sl:,.2f}")
        m2.metric("Total Duration (hours)", f"{total_hours:,.2f}")
        m3.metric("Average Flow (SLM)", f"{avg_slm:,.3f}")

        # --------------------------------------------------
        # Period reports
        # --------------------------------------------------
        tmp = usage.copy()
        tmp["Week"] = tmp["start_time"].dt.to_period("W").astype(str)
        tmp["Month"] = tmp["start_time"].dt.to_period("M").astype(str)
        tmp["Quarter"] = tmp["start_time"].dt.to_period("Q").astype(str)

        week_rep = tmp.groupby("Week", as_index=False)["Total SL"].sum().sort_values("Week")
        month_rep = tmp.groupby("Month", as_index=False)["Total SL"].sum().sort_values("Month")
        quarter_rep = tmp.groupby("Quarter", as_index=False)["Total SL"].sum().sort_values("Quarter")

        t1, t2, t3 = st.tabs(["Weekly", "Monthly", "3 Months"])
        with t1:
            st.dataframe(week_rep, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(month_rep, use_container_width=True, hide_index=True)
        with t3:
            st.dataframe(quarter_rep, use_container_width=True, hide_index=True)

        # --------------------------------------------------
        # Per log breakdown
        # --------------------------------------------------
        st.markdown("#### Per log file breakdown")
        st.dataframe(usage.tail(250), use_container_width=True, height=350)

        # --------------------------------------------------
        # Auto-save reports (FULL history from folder, not only selected window)
        # --------------------------------------------------
        try:
            full_usage = pd.DataFrame(rows)
            full_usage["start_time"] = pd.to_datetime(full_usage["start_time"], errors="coerce")
            full_usage["end_time"] = pd.to_datetime(full_usage["end_time"], errors="coerce")
            full_usage = full_usage.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)

            if not full_usage.empty:
                full_usage["Week"] = full_usage["start_time"].dt.to_period("W").astype(str)
                full_usage["Month"] = full_usage["start_time"].dt.to_period("M").astype(str)
                full_usage["Quarter"] = full_usage["start_time"].dt.to_period("Q").astype(str)

                # 1) Per-log summary
                out_all_logs = full_usage[[
                    "log_file", "start_time", "end_time", "duration_min", "Total SL", "Week", "Month", "Quarter"
                ]].copy()
                p1 = os.path.join(REPORT_DIR, "gas_summary_all_logs.csv")
                out_all_logs.to_csv(p1, index=False)

                # 2) Weekly totals
                week_agg = full_usage.groupby("Week", as_index=False).agg(total_sl=("Total SL", "sum"))
                week_agg = week_agg.sort_values("Week").reset_index(drop=True)
                p2 = os.path.join(REPORT_DIR, "gas_weekly_totals.csv")
                week_agg.to_csv(p2, index=False)

                # 3) Monthly totals + avg SLM for month
                month_agg = full_usage.groupby("Month", as_index=False).agg(
                    total_sl=("Total SL", "sum"),
                    total_minutes=("duration_min", "sum"),
                    n_logs=("log_file", "count"),
                    first_start=("start_time", "min"),
                    last_end=("end_time", "max"),
                )
                month_agg["avg_slm"] = month_agg.apply(
                    lambda r: (float(r["total_sl"]) / float(r["total_minutes"])) if float(r["total_minutes"]) > 0 else 0.0,
                    axis=1,
                )
                month_agg = month_agg.sort_values("Month").reset_index(drop=True)
                p3 = os.path.join(REPORT_DIR, "gas_monthly_totals.csv")
                month_agg.to_csv(p3, index=False)

                # 4) Quarterly totals
                q_agg = full_usage.groupby("Quarter", as_index=False).agg(total_sl=("Total SL", "sum"))
                q_agg = q_agg.sort_values("Quarter").reset_index(drop=True)
                p4 = os.path.join(REPORT_DIR, "gas_quarterly_totals.csv")
                q_agg.to_csv(p4, index=False)

                # Missing months detection (between first and last month)
                first_m = pd.Period(full_usage["start_time"].min(), freq="M")
                last_m = pd.Period(full_usage["start_time"].max(), freq="M")
                expected = [str(p) for p in pd.period_range(first_m, last_m, freq="M")]
                present = set(month_agg["Month"].astype(str).tolist())
                missing = [m for m in expected if m not in present]

                st.success("✅ Gas reports saved automatically")
                st.code("\n".join([p1, p2, p3, p4]))

                if missing:
                    st.warning("Missing months (no logs found): " + ", ".join(missing))
                else:
                    st.caption("No missing months detected between first and last log month.")
            else:
                st.info("No full-history rows available to save reports.")
        except Exception as e:
            st.warning(f"Auto-save failed: {e}")

        st.caption("Units: MFC Actual assumed SLM. Integrated to SL via SLM × dt(minutes).")

    # =========================================================
    # ✅ Faults section
    def render_faults_section(con, MAINT_FOLDER, actor):
        st.subheader("🚨 Faults / Incidents")

        faults_csv = _read_csv_safe(FAULTS_CSV, FAULTS_COLS)
        actions_csv = _read_csv_safe(FAULTS_ACTIONS_CSV, FAULTS_ACTIONS_COLS)
        state_map = _latest_fault_state(actions_csv)

        if not faults_csv.empty:
            faults_csv["fault_id"] = pd.to_numeric(faults_csv["fault_id"], errors="coerce")
            faults_csv = faults_csv.dropna(subset=["fault_id"]).copy()
            faults_csv["fault_id"] = faults_csv["fault_id"].astype(int)
            faults_csv["fault_ts"] = pd.to_datetime(faults_csv["fault_ts"], errors="coerce")

            faults_csv["_is_closed"] = faults_csv["fault_id"].apply(
                lambda fid: bool(state_map.get(int(fid), {}).get("is_closed", False))
            )
            faults_csv["_last_action_ts"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_ts", None)
            )
            faults_csv["_last_action_type"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_type", "")
            )
            faults_csv["_last_action_actor"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_actor", "")
            )
            faults_csv["_last_fix"] = faults_csv["fault_id"].apply(
                lambda fid: state_map.get(int(fid), {}).get("last_fix", "")
            )
        else:
            faults_csv = pd.DataFrame(columns=FAULTS_COLS + ["_is_closed", "_last_action_ts", "_last_action_type", "_last_action_actor", "_last_fix"])

        # ---- Log a new fault ----
        with st.expander("➕ Log a new fault", expanded=False):
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                comp_list = (
                    dfm["Component"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
                comp_list = sorted([c for c in comp_list if c])
                comp_options = comp_list + ["Other (custom)"]

                selected_comp = st.selectbox(
                    "Fault component",
                    options=comp_options,
                    key="fault_component_select"
                )

                if selected_comp == "Other (custom)":
                    fault_component = st.text_input(
                        "Custom component name",
                        key="fault_component_custom"
                    )
                else:
                    fault_component = selected_comp
            with c2:
                severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], index=1, key="fault_sev_in")
            with c3:
                related_draw = st.text_input("Related draw (optional)", placeholder="e.g. FP0888_1", key="fault_draw_in")

            title = st.text_input("Fault title", placeholder="Short title", key="fault_title_in")
            desc = st.text_area("Fault description", placeholder="What happened? what did you do? what to check next time?", height=120, key="fault_desc_in")

            cA, cB = st.columns([1, 1])
            with cA:
                src_file = st.text_input("Source file (optional)", placeholder="e.g. faults.xlsx / email.pdf / photo.jpg", key="fault_src_in")
            with cB:
                st.caption("Saved as BOTH DuckDB + faults_log.csv")

            if st.button("➕ Log fault", type="primary", use_container_width=True, key="fault_add_btn"):
                if not str(fault_component).strip():
                    st.warning("Fault component is required.")
                    st.stop()
                if not str(title).strip() and not str(desc).strip():
                    st.warning("Give at least a title or description.")
                    st.stop()

                now_dt = dt.datetime.now()
                fid = int(time.time() * 1000)

                try:
                    con.execute("""
                        INSERT INTO faults_events
                        (fault_id, fault_ts, component, title, description, severity, actor, source_file, related_draw)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        fid, now_dt,
                        str(fault_component), str(title), str(desc),
                        str(severity), str(actor), str(src_file), str(related_draw)
                    ])
                except Exception as e:
                    st.warning(f"DuckDB insert failed (still saving CSV log): {e}")

                row = pd.DataFrame([{
                    "fault_id": fid,
                    "fault_ts": now_dt,
                    "fault_component": str(fault_component),
                    "fault_title": str(title),
                    "fault_description": str(desc),
                    "fault_severity": str(severity),
                    "fault_actor": str(actor),
                    "fault_source_file": str(src_file),
                    "fault_related_draw": str(related_draw),
                }])
                try:
                    _append_csv(FAULTS_CSV, FAULTS_COLS, row)
                    st.success("Fault logged.")
                except Exception as e:
                    st.error(f"Failed writing faults_log.csv: {e}")

                st.rerun()

        # ---- Open faults list ----
        st.markdown("#### 🔓 Open faults")
        open_df = faults_csv[faults_csv["_is_closed"] == False].copy()
        open_df = open_df.sort_values("fault_ts", ascending=False)

        if open_df.empty:
            st.success("No open faults 👍")
        else:
            for _, r in open_df.iterrows():
                fid = int(r["fault_id"])
                comp = safe_str(r.get("fault_component", ""))
                sev = safe_str(r.get("fault_severity", ""))
                title = safe_str(r.get("fault_title", "")) or "Fault"
                ts = safe_str(r.get("fault_ts", ""))

                c1, c2, c3 = st.columns([3.4, 1.1, 1.1])
                with c1:
                    st.markdown(f"**[{sev.upper()}] {comp} — {title}**")
                    st.caption(f"ID: `{fid}`  |  Time: {ts}")

                with c2:
                    @st.dialog(f"Close fault: {comp} — {title} (#{fid})")
                    def _dlg_close():
                        fix = st.text_input("Fix summary (short)", key=f"fix_sum__{fid}")
                        note = st.text_area("Closure notes", height=120, key=f"fix_note__{fid}")
                        if st.button("✅ Close fault", type="primary", use_container_width=True, key=f"close_do__{fid}"):
                            _write_fault_action(con, fault_id=fid, action_type="close", actor=actor, note=note, fix_summary=fix)
                            st.success("Closed.")
                            st.rerun()

                    if st.button("✅ Close", use_container_width=True, key=f"btn_close__{fid}"):
                        _dlg_close()

                with c3:
                    @st.dialog(f"Add note: #{fid}")
                    def _dlg_note():
                        note = st.text_area("Note", height=120, key=f"note_txt__{fid}")
                        if st.button("➕ Save note", type="primary", use_container_width=True, key=f"note_do__{fid}"):
                            _write_fault_action(con, fault_id=fid, action_type="note", actor=actor, note=note, fix_summary="")
                            st.success("Saved note.")
                            st.rerun()

                    if st.button("📝 Note", use_container_width=True, key=f"btn_note__{fid}"):
                        _dlg_note()

                with st.expander("Details", expanded=False):
                    st.write(safe_str(r.get("fault_description", "")) or "—")
                    st.caption(f"Source file: {safe_str(r.get('fault_source_file',''))} | Related draw: {safe_str(r.get('fault_related_draw',''))}")

                st.divider()

        # ---- All faults table + reopen ----
        with st.expander("📜 All faults (table)", expanded=False):
            df_all = faults_csv.copy()
            if df_all.empty:
                st.info("No faults yet.")
            else:
                df_all["Status"] = np.where(df_all["_is_closed"], "Closed", "Open")
                df_all["Last Action"] = df_all["_last_action_type"]
                df_all["Last Action By"] = df_all["_last_action_actor"]
                df_all["Last Fix Summary"] = df_all["_last_fix"]
                show = df_all[[
                    "fault_ts", "Status", "fault_id", "fault_component", "fault_severity",
                    "fault_title", "fault_actor", "fault_related_draw",
                    "Last Action", "Last Action By", "Last Fix Summary"
                ]].copy()
                st.dataframe(show, use_container_width=True, height=360, hide_index=True)

                closed_ids = df_all[df_all["_is_closed"] == True]["fault_id"].astype(int).tolist()
                if closed_ids:
                    st.markdown("##### Reopen a fault")
                    pick = st.selectbox("Closed fault ID", options=[""] + [str(x) for x in closed_ids], key="reopen_pick")
                    if pick and st.button("♻️ Reopen", use_container_width=True, key="reopen_btn"):
                        _write_fault_action(con, fault_id=int(pick), action_type="reopen", actor=actor, note="Reopened", fix_summary="")
                        st.success("Reopened.")
                        st.rerun()

        with st.expander("🧾 Fault actions (CSV log)", expanded=False):
            if not os.path.isfile(FAULTS_ACTIONS_CSV):
                st.info("No faults_actions_log.csv yet (close/reopen/note first).")
            else:
                try:
                    df = pd.read_csv(FAULTS_ACTIONS_CSV)
                    st.dataframe(df.tail(300), use_container_width=True, height=360)
                except Exception as e:
                    st.warning(f"Fault actions CSV read failed: {e}")

    # =========================================================
    # Load report + tasks editor
    # =========================================================
    def render_maintenance_load_report(files, load_errors):
        with st.expander("Load report", expanded=False):
            try:
                st.write("Loaded files:", sorted(list(files or [])))
            except Exception:
                st.write("Loaded files:", files)

            if load_errors:
                st.warning("Some files failed to load:")
                st.dataframe(pd.DataFrame(load_errors, columns=["File", "Error"]), use_container_width=True)

    def render_maintenance_tasks_editor(
        MAINT_FOLDER,
        files,
        read_file,
        write_file,
        normalize_df,
        templateize_df,
    ):
        with st.expander("📝 Maintenance tasks editor (source files)", expanded=False):
            st.caption("Edits the selected maintenance file (Excel/CSV) and saves back.")
            pick = st.selectbox("Select maintenance file", options=sorted(files), key="maint_edit_file_pick")
            if not pick:
                return
            path = os.path.join(MAINT_FOLDER, pick)
            try:
                raw = read_file(path)
                if raw is None or raw.empty:
                    st.info("File is empty.")
                    return
                df = normalize_df(raw)

                show_cols = [c for c in df.columns if c != "Source_File"]
                edited = st.data_editor(df[show_cols], use_container_width=True, height=420, key="maint_tasks_editor_grid")

                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("💾 Save file", type="primary", use_container_width=True, key="maint_save_file_btn"):
                        out = templateize_df(edited, list(raw.columns))
                        write_file(path, out)
                        st.success("Saved.")
                        st.rerun()
                with c2:
                    st.caption("Saved back in the original template columns.")
            except Exception as e:
                st.warning(f"Tasks editor failed: {e}")

    # =========================================================
    # Manuals / Documents browser (same preview style)
    # =========================================================
    def render_manuals_browser(BASE_DIR):
        MANUALS_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

        def _ext(p: str) -> str:
            return os.path.splitext(str(p).lower())[1]

        def _is_pdf(p: str) -> bool:
            return str(p).lower().endswith(".pdf")

        def _is_img(p: str) -> bool:
            return _ext(p) in MANUALS_IMG_EXTS

        def _short_name(fn: str, max_len: int = 42) -> str:
            fn = str(fn)
            if len(fn) <= max_len:
                return fn
            keep_tail = 16
            head = max_len - keep_tail - 3
            return fn[:head] + "..." + fn[-keep_tail:]

        @st.cache_data(show_spinner=False)
        def _read_bytes(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        def _download_btn(path: str, label: str, key: str):
            if not os.path.exists(path):
                st.warning(f"Missing file: {os.path.basename(path)}")
                return
            data = _read_bytes(path)
            st.download_button(
                label=label,
                data=data,
                file_name=os.path.basename(path),
                mime=None,
                key=key,
                use_container_width=True,
            )

        @st.cache_data(show_spinner=False)
        def _pdf_render_pages(path: str, max_pages: int = 1, zoom: float = 1.6):
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            n = min(len(doc), int(max_pages))
            out = []
            mat = fitz.Matrix(float(zoom), float(zoom))
            for i in range(n):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out.append(pix.tobytes("png"))
            doc.close()
            return out

        def _render_pdf_preview(path: str, *, key_prefix: str):
            if not os.path.exists(path):
                st.warning("PDF file not found.")
                return

            state_key = f"{key_prefix}__show_all"
            st.session_state.setdefault(state_key, False)

            c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
            with c1:
                st.markdown("**PDF preview (rendered)**")
                st.caption("Default shows page 1. Click to render more pages.")
            with c2:
                zoom = st.selectbox("Quality", [1.3, 1.6, 2.0], index=1, key=f"{key_prefix}__zoom")
            with c3:
                max_pages = st.number_input(
                    "Pages (when expanded)", min_value=1, max_value=200, value=30, step=1, key=f"{key_prefix}__pages"
                )

            b1, b2 = st.columns([1, 1])
            with b1:
                if not st.session_state[state_key]:
                    if st.button("📄 Render more pages", use_container_width=True, key=f"{key_prefix}__more"):
                        st.session_state[state_key] = True
                        st.rerun()
                else:
                    if st.button("⬅️ Back to page 1", use_container_width=True, key=f"{key_prefix}__less"):
                        st.session_state[state_key] = False
                        st.rerun()
            with b2:
                _download_btn(path, "⬇️ Download PDF", key=f"{key_prefix}__dl")

            try:
                if st.session_state[state_key]:
                    imgs = _pdf_render_pages(path, max_pages=int(max_pages), zoom=float(zoom))
                    st.caption(f"Showing **{len(imgs)}** page(s).")
                    for i, b in enumerate(imgs, start=1):
                        st.image(b, caption=f"Page {i}", use_container_width=True)
                else:
                    imgs = _pdf_render_pages(path, max_pages=1, zoom=float(zoom))
                    if imgs:
                        st.image(imgs[0], caption="Page 1", use_container_width=True)
            except Exception as e:
                st.error(f"PDF render failed. Install PyMuPDF: `pip install pymupdf`  |  Error: {e}")

        with st.expander("📚 Manuals / Documents browser", expanded=False):
            st.caption("Tight checklist view: select manuals, then preview one.")

            candidate_dirs = [
                os.path.join(BASE_DIR, "manuals"),
                os.path.join(BASE_DIR, "docs"),
                os.path.join(BASE_DIR, "maintenance", "manuals"),
                os.path.join(BASE_DIR, "maintenance", "docs"),
            ]
            existing = [d for d in candidate_dirs if os.path.isdir(d)]
            if not existing:
                st.info("No manuals/docs folder found. (Create /manuals or /docs).")
                return

            root = st.selectbox("Folder", existing, key="maint_manuals_root_pick")

            paths = sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True))
            paths = [p for p in paths if os.path.isfile(p)]
            if not paths:
                st.info("No files found.")
                return

            c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
            with c1:
                q = st.text_input("Search", placeholder="type filename…", key="maint_manuals_search")
            with c2:
                kind = st.selectbox("Type", ["All", "PDF", "Images", "Other"], key="maint_manuals_type")
            with c3:
                limit = st.number_input("Show (max)", 10, 500, 120, 10, key="maint_manuals_limit")

            def _match(p):
                fn = os.path.basename(p).lower()
                if q and q.lower().strip() not in fn:
                    return False
                if kind == "PDF" and not _is_pdf(p):
                    return False
                if kind == "Images" and not _is_img(p):
                    return False
                if kind == "Other" and (_is_pdf(p) or _is_img(p)):
                    return False
                return True

            shown = [p for p in paths if _match(p)]
            st.caption(f"Files found: **{len(shown)}** (total in folder: {len(paths)})")
            shown = shown[: int(limit)]

            st.session_state.setdefault("maint_manuals_checked", [])
            st.session_state.setdefault("maint_manuals_active", "")

            st.markdown("#### ✅ Select manuals")
            checked = set(st.session_state.get("maint_manuals_checked", []))

            for i, p in enumerate(shown):
                fn = os.path.basename(p)
                col0, col1, col2 = st.columns([0.35, 5.0, 1.0], gap="small")
                with col0:
                    is_on = st.checkbox("", value=(p in checked), key=f"maint_manuals_chk__{i}")
                with col1:
                    st.markdown(f"**{_short_name(fn)}**")
                with col2:
                    _download_btn(p, "⬇️", key=f"maint_manuals_dl__{i}__{fn}")

                if is_on:
                    checked.add(p)
                else:
                    checked.discard(p)

            st.session_state["maint_manuals_checked"] = sorted(list(checked))

            st.divider()

            picked_list = st.session_state["maint_manuals_checked"]
            if not picked_list:
                st.info("Select at least one manual to preview.")
                return

            if st.session_state["maint_manuals_active"] not in picked_list:
                st.session_state["maint_manuals_active"] = picked_list[0]

            labels = {p: os.path.basename(p) for p in picked_list}
            active = st.selectbox(
                "👁️ Preview selected manual",
                options=picked_list,
                format_func=lambda p: labels.get(p, p),
                key="maint_manuals_active",
            )

            st.markdown("### Preview")
            st.caption(os.path.basename(active))

            if _is_pdf(active):
                _render_pdf_preview(active, key_prefix=f"maint_manuals_pdf__{os.path.basename(active)}")
            elif _is_img(active):
                st.image(active, use_container_width=True)
            else:
                st.info("No preview for this file type (use Download).")

            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("🧹 Clear selection", use_container_width=True, key="maint_manuals_clear"):
                    st.session_state["maint_manuals_checked"] = []
                    st.session_state["maint_manuals_active"] = ""
                    st.rerun()
            with cB:
                _download_btn(active, "⬇️ Download active", key="maint_manuals_dl_active")

    # =========================================================
    # UI flow
    # =========================================================
    render_maintenance_dashboard_metrics(dfm)

    horizon_hours, horizon_days, horizon_draws = render_maintenance_horizon_selector(current_draw_count)

    render_maintenance_roadmaps(
        dfm,
        current_date,
        current_draw_count,
        furnace_hours,
        uv1_hours,
        uv2_hours,
        horizon_hours,
        horizon_days,
        horizon_draws,
    )

    edited = render_maintenance_done_editor(dfm)

    render_maintenance_apply_done(
        edited,
        dfm=dfm,
        current_date=current_date,
        current_draw_count=current_draw_count,
        actor=actor,
        MAINT_FOLDER=MAINT_FOLDER,
        con=con,
        read_file=read_file,
        write_file=write_file,
        normalize_df=normalize_df,
        templateize_df=templateize_df,
        pick_current_hours=pick_current_hours,
        mode_norm=mode_norm,
    )

    render_maintenance_history(con)

    # ✅ Gas report from LOGS (MFC actual)
    render_gas_report(LOGS_FOLDER)

    render_faults_section(
        con=con,
        MAINT_FOLDER=MAINT_FOLDER,
        actor=actor,
    )

    render_maintenance_load_report(files, load_errors)

    render_maintenance_tasks_editor(
        MAINT_FOLDER=MAINT_FOLDER,
        files=files,
        read_file=read_file,
        write_file=write_file,
        normalize_df=normalize_df,
        templateize_df=templateize_df,
    )

    render_manuals_browser(BASE_DIR)
# ------------------ Correlation & Outliers ------------------
elif tab_selection == "📈 Correlation & Outliers":
    st.title("📈 Correlation & Outliers")
    st.caption(
        "Builds a numeric snapshot per log file (time = log file mtime), then plots rolling correlation vs time "
        "for MANY column pairs."
    )

    # ==========================================================
    # Imports (local)
    # ==========================================================
    import os, re, json, itertools
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    # ==========================================================
    # Paths
    # ==========================================================
    BASE_DIR = os.getcwd()
    LOGS_FOLDER = os.path.join(BASE_DIR, P.logs_dir)
    MAINT_FOLDER = P.maintenance_dir
    os.makedirs(MAINT_FOLDER, exist_ok=True)

    # ==========================================================
    # Main renderer
    # ==========================================================
    def render_corr_outliers_tab(DRAW_FOLDER: str, MAINT_FOLDER: str):
        os.makedirs(MAINT_FOLDER, exist_ok=True)

        # ---------------------------
        # Helpers
        # ---------------------------
        def _safe_key(s: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))[:90]

        def _to_float(v):
            try:
                if v is None:
                    return np.nan
                if isinstance(v, (int, float, np.integer, np.floating)):
                    return float(v)
                s = str(v).strip()
                if s == "" or s.lower() in {"nan", "none"}:
                    return np.nan
                return float(s)
            except Exception:
                return np.nan

        # rolling corr series for a pair
        def _rolling_corr(df: pd.DataFrame, col_a: str, col_b: str, win: int):
            s = df[["window_time", "file", col_a, col_b]].copy()
            s = s.dropna(subset=[col_a, col_b])
            if len(s) < win:
                return None

            corrs, times, files = [], [], []
            for i in range(win - 1, len(s)):
                w = s.iloc[i - win + 1 : i + 1]
                c = w[col_a].corr(w[col_b])
                if pd.notna(c):
                    corrs.append(float(c))
                    times.append(s.iloc[i]["window_time"])
                    files.append(s.iloc[i]["file"])

            if not corrs:
                return None

            g = pd.DataFrame({"window_time": times, "file": files, "corr": corrs})
            return g

        # ---------------------------
        # Scan logs and build base table
        # ---------------------------
        st.markdown("---")
        st.subheader("1) Scan logs → one numeric snapshot per file (time = file mtime)")

        if not os.path.exists(DRAW_FOLDER):
            st.warning(f"Logs folder not found: {DRAW_FOLDER}")
            return

        log_files = sorted(
            [os.path.join(DRAW_FOLDER, f) for f in os.listdir(DRAW_FOLDER) if f.lower().endswith(".csv")],
            key=lambda p: os.path.getmtime(p)
        )

        if not log_files:
            st.info("No log CSVs found.")
            return

        st.caption(f"Found {len(log_files)} log CSV files.")

        # robust slider bounds for few files
        n_files = len(log_files)
        max_cap = min(2000, n_files)
        if max_cap <= 1:
            st.info(f"Only {n_files} log file(s) found — need at least 2.")
            return

        max_files = st.slider(
            "Max files to process",
            min_value=2,
            max_value=max_cap,
            value=min(300, max_cap),
            step=1 if max_cap < 25 else 10,
            key="corr_many_max_files"
        )
        log_files = log_files[-max_files:]

        rows = []
        fail = 0

        # We intentionally use file mtime as the time axis (your request)
        for p in log_files:
            try:
                df = pd.read_csv(p)
                if df is None or df.empty:
                    continue

                # use file mtime
                t = datetime.fromtimestamp(os.path.getmtime(p))

                last = df.iloc[-1].copy()
                rec = {"window_time": pd.to_datetime(t), "file": os.path.basename(p)}

                # numeric cols: last row values
                for c in df.columns:
                    vv = last[c]
                    num = _to_float(vv)
                    if np.isfinite(num):
                        rec[c] = num

                rows.append(rec)
            except Exception:
                fail += 1

        if not rows:
            st.warning("No usable numeric data found in logs.")
            return

        base = pd.DataFrame(rows)
        base["window_time"] = pd.to_datetime(base["window_time"], errors="coerce")
        base = base.sort_values("window_time").reset_index(drop=True)

        st.caption(f"Usable rows: {len(base)} | Failed files: {fail}")

        with st.expander("Preview numeric table", expanded=False):
            st.dataframe(base.tail(50), use_container_width=True)

        numeric_cols = [c for c in base.columns if c not in {"window_time", "file"}]
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns across logs to compute correlations.")
            return

        # ==========================================================
        # Pair settings
        # ==========================================================
        st.markdown("---")
        st.subheader("2) Pair settings")

        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.0])

        with c1:
            win_max = max(5, min(200, len(base)))
            win = st.slider(
                "Rolling window (points)",
                min_value=3,
                max_value=win_max,
                value=min(20, win_max),
                step=1 if win_max < 25 else 5,
                key="corr_many_win"
            )

        with c2:
            min_points = st.number_input(
                "Min points for a pair (after NaN drop)",
                min_value=3,
                value=max(10, int(win)),
                step=1,
                key="corr_many_min_points"
            )

        with c3:
            max_pairs = st.number_input(
                "Max pairs to plot",
                min_value=1,
                value=20,
                step=1,
                key="corr_many_max_pairs"
            )

        with c4:
            pair_sort = st.selectbox(
                "Sort pairs by",
                ["|median corr| (strongest)", "corr variability (std)", "alphabetical"],
                index=0,
                key="corr_many_sort"
            )

        # Optional: column include filter
        col_filter = st.text_input(
            "Column filter (optional): only include columns containing this text (case-insensitive)",
            value="",
            key="corr_many_filter"
        ).strip().lower()

        cols_use = numeric_cols
        if col_filter:
            cols_use = [c for c in numeric_cols if col_filter in str(c).lower()]

        if len(cols_use) < 2:
            st.warning("Filter left fewer than 2 numeric columns.")
            return

        # ==========================================================
        # Generate all pairs + score them
        # ==========================================================
        st.markdown("---")
        st.subheader("3) Compute + plot correlations for MANY pairs")

        # Build pair list
        pairs = list(itertools.combinations(cols_use, 2))
        if not pairs:
            st.info("No pairs available.")
            return

        st.caption(f"Candidate pairs: {len(pairs)} (from {len(cols_use)} numeric columns)")

        # Score pairs quickly (without building full plot for each)
        scored = []
        for a, b in pairs:
            s = base[[a, b]].dropna()
            n = len(s)
            if n < int(min_points):
                continue

            # quick score: overall corr + variability
            c = s[a].corr(s[b])
            if pd.isna(c):
                continue

            # also estimate variability on rolling corr if enough points
            # (lightweight: only if n is big enough)
            var_est = np.nan
            if n >= max(int(win), 8):
                # compute corr on a few windows to estimate std cheaply
                # take up to 30 windows evenly spaced
                idxs = np.linspace(max(int(win) - 1, 0), n - 1, num=min(30, n - int(win) + 1), dtype=int)
                cc = []
                for ii in idxs:
                    w = s.iloc[ii - int(win) + 1: ii + 1]
                    if len(w) == int(win):
                        vv = w[a].corr(w[b])
                        if pd.notna(vv):
                            cc.append(float(vv))
                if cc:
                    var_est = float(np.nanstd(cc))

            scored.append({
                "a": a, "b": b,
                "n": int(n),
                "corr": float(c),
                "abs_corr": float(abs(c)),
                "var_est": var_est,
            })

        if not scored:
            st.warning("No pairs passed the Min points requirement.")
            return

        df_pairs = pd.DataFrame(scored)

        if pair_sort == "|median corr| (strongest)":
            df_pairs = df_pairs.sort_values(["abs_corr", "n"], ascending=[False, False])
        elif pair_sort == "corr variability (std)":
            df_pairs = df_pairs.sort_values(["var_est", "n"], ascending=[False, False])
        else:
            df_pairs = df_pairs.sort_values(["a", "b"], ascending=[True, True])

        df_pairs = df_pairs.head(int(max_pairs)).reset_index(drop=True)

        with st.expander("Selected pairs (preview)", expanded=False):
            st.dataframe(df_pairs, use_container_width=True)

        # ==========================================================
        # Plot each pair (many plots)
        # ==========================================================
        for i, row in df_pairs.iterrows():
            a = row["a"]
            b = row["b"]

            g = _rolling_corr(base, a, b, int(win))
            if g is None or g.empty:
                continue

            title = f"{a}  vs  {b}  | rolling corr (win={int(win)})"
            fig = px.line(g, x="window_time", y="corr", markers=True, title=title)

            # ✅ Unique key prevents StreamlitDuplicateElementId
            k = f"corr_pair_{i}_{_safe_key(a)}__{_safe_key(b)}__w{int(win)}"
            with st.expander(f"📌 Pair {i+1}: {a} ↔ {b}", expanded=(i == 0)):
                st.caption(f"Points used (after NaN drop): {len(base[[a,b]].dropna())} | Rolling points: {len(g)}")
                st.plotly_chart(fig, use_container_width=True, key=k)

    # ==========================================================
    # Run it
    # ==========================================================
    render_corr_outliers_tab(DRAW_FOLDER=LOGS_FOLDER, MAINT_FOLDER=MAINT_FOLDER)
# ------------------ SQL Lab ------------------
elif tab_selection == "🧪 SQL Lab":
    import os, glob, re
    import pandas as pd
    import numpy as np
    import streamlit as st
    import duckdb
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from plotly.subplots import make_subplots

    # NOTE:
    # Best practice is to call st.set_page_config(layout="wide") ONCE at the top of the app.
    # We guard it here so re-runs / other tabs won't break the app.
    if "_page_config_set" not in st.session_state:
        try:
            st.set_page_config(layout="wide")
        except Exception:
            pass
        st.session_state["_page_config_set"] = True

    # Force wide-looking layout even if page_config is set elsewhere
    st.markdown(
        """
        <style>
        .block-container { max-width: 98% !important; padding-left: 1.8rem; padding-right: 1.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧪 SQL Lab")
    st.caption(
        "Filter **draw CSVs** with AND/OR/NOT, then overlay **Maintenance** and **Faults** "
        "in a separate events lane. Click any point to inspect."
    )

    BASE_DIR = os.getcwd()
    DATASET_DIR = os.path.join(BASE_DIR, P.dataset_dir)
    DB_PATH = os.path.join(BASE_DIR, P.duckdb_path)

    # =========================================================
    # Persistent DuckDB connection
    # =========================================================
    if "sql_duck_con" not in st.session_state:
        st.session_state["sql_duck_con"] = duckdb.connect(DB_PATH)
    con = st.session_state["sql_duck_con"]

    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass

    # =========================================================
    # Ensure required DB tables exist
    # =========================================================
    con.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_actions (
        action_id       BIGINT,
        action_ts       TIMESTAMP,
        component       VARCHAR,
        task            VARCHAR,
        task_id         VARCHAR,
        tracking_mode   VARCHAR,
        hours_source    VARCHAR,
        done_date       DATE,
        done_hours      DOUBLE,
        done_draw       INTEGER,
        source_file     VARCHAR,
        actor           VARCHAR,
        note            VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS faults_events (
        fault_id        BIGINT,
        fault_ts        TIMESTAMP,
        component       VARCHAR,
        title           VARCHAR,
        description     VARCHAR,
        severity        VARCHAR,
        actor           VARCHAR,
        source_file     VARCHAR,
        related_draw    VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_tasks (
        task_key            VARCHAR,
        task_id             VARCHAR,
        component           VARCHAR,
        task                VARCHAR,
        tracking_mode       VARCHAR,
        hours_source        VARCHAR,
        interval_value      VARCHAR,
        interval_unit       VARCHAR,
        due_threshold_days  VARCHAR,
        manual_name         VARCHAR,
        page                VARCHAR,
        document            VARCHAR,
        procedure_summary   VARCHAR,
        notes               VARCHAR,
        owner               VARCHAR,
        source_file         VARCHAR,
        loaded_at           TIMESTAMP
    );
    """)

    # =========================================================
    # Helpers
    # =========================================================
    def _esc(s: str) -> str:
        return (s or "").replace("'", "''")

    def _lit(x) -> str:
        return "'" + _esc(str(x)) + "'"

    def _is_num(x) -> bool:
        try:
            float(str(x))
            return True
        except Exception:
            return False

    def _dedupe_keep_order(seq):
        out, seen = [], set()
        for x in seq or []:
            s = str(x).strip() if x is not None else ""
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _mtime_ts(path: str):
        try:
            return pd.to_datetime(os.path.getmtime(path), unit="s")
        except Exception:
            return pd.NaT

    def _tokenize_search(q: str):
        q = (q or "").strip().lower()
        if not q:
            return []
        return [t.strip() for t in re.split(r"[,\s]+", q) if t.strip()]

    def _match_params_by_tokens(params, tokens):
        if not tokens:
            return list(params)
        out = []
        for p in params:
            pl = str(p).lower()
            if all(t in pl for t in tokens):
                out.append(p)
        return out

    def _extract_zone_num(pname: str):
        s = str(pname or "")
        m = re.search(r"(?i)\bzone\D*([0-9]{1,3})\b", s)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _filters_summary_for_draws(used_params_list, human_lines_list) -> str:
        used_params_list = _dedupe_keep_order(used_params_list or [])
        parts = []
        if used_params_list:
            parts.append("Params: " + ", ".join(used_params_list))
        if human_lines_list:
            hl = [str(x).strip() for x in (human_lines_list or []) if str(x).strip()]
            if hl:
                parts.append(" | ".join(hl[:6]) + (" …" if len(hl) > 6 else ""))
        return " || ".join(parts).strip()

    # =========================================================
    # Build DuckDB view for dataset CSVs (KV)
    # =========================================================
    def build_datasets_kv_view_from_disk() -> int:
        files = glob.glob(os.path.join(DATASET_DIR, "**", "*.csv"), recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            con.execute("""
                CREATE OR REPLACE VIEW datasets_kv AS
                SELECT
                    'dataset'::VARCHAR AS source_kind,
                    NULL::TIMESTAMP AS event_ts,
                    NULL::VARCHAR AS event_id,
                    NULL::VARCHAR AS _draw,
                    NULL::VARCHAR AS _file,
                    NULL::VARCHAR AS filename,
                    NULL::VARCHAR AS "Parameter Name",
                    NULL::VARCHAR AS "Value",
                    NULL::VARCHAR AS "Units"
                WHERE FALSE;
            """)
            return 0

        files = [f.replace("\\", "/") for f in files]
        files_sql = "[" + ",".join(_lit(f) for f in files) + "]"

        con.execute(f"""
            CREATE OR REPLACE VIEW datasets_kv AS
            WITH raw AS (
                SELECT
                    *,
                    filename,
                    regexp_extract(filename, '[^/]+$', 0) AS _file,
                    regexp_extract(filename, '([^/]+)\\.csv$', 1) AS _draw
                FROM read_csv_auto(
                    {files_sql},
                    filename=true,
                    union_by_name=true,
                    ALL_VARCHAR=TRUE,
                    ignore_errors=true
                )
            ),
            ts AS (
                SELECT
                    _draw,
                    MAX(TRY_CAST("Value" AS TIMESTAMP)) FILTER (
                        WHERE lower(trim("Parameter Name")) IN ('draw date','draw datetime')
                           OR lower(trim("Parameter Name")) LIKE '%draw date%'
                           OR lower(trim("Parameter Name")) LIKE '%draw time%'
                           OR lower(trim("Parameter Name")) LIKE '%datetime%'
                    ) AS draw_ts
                FROM raw
                GROUP BY _draw
            )
            SELECT
                'dataset'::VARCHAR AS source_kind,
                ts.draw_ts AS event_ts,
                raw._draw::VARCHAR AS event_id,
                raw._draw,
                raw._file,
                raw.filename,
                raw."Parameter Name",
                raw."Value",
                raw."Units"
            FROM raw
            LEFT JOIN ts USING (_draw);
        """)
        return len(files)

    def ensure_views():
        try:
            con.execute("SELECT COUNT(*) FROM datasets_kv").fetchone()
        except Exception:
            n = build_datasets_kv_view_from_disk()
            st.caption(f"Indexed dataset CSVs: {n}")

    # =========================================================
    # Indexing controls
    # =========================================================
    with st.expander("📁 Indexing", expanded=False):
        c1, c2 = st.columns([1, 1])
        if c1.button("🔄 Rebuild dataset index", use_container_width=True, key="sql_rebuild_kv"):
            for k in [
                "sql_df_all", "sql_matched_draws", "sql_selected_event_key", "math_selected_event_key",
                "ds_conditions", "ds_conditions_human", "ds_conditions_struct",
                "sql_filter_params_seq", "sql_group_selected_params",
                "sql_values_found_long", "sql_values_found_wide", "sql_last_filters_summary",
                "sql_matched_params_only", "sql_plot_params",
                "sql_group_defs_for_plot",
            ]:
                st.session_state.pop(k, None)
            n = build_datasets_kv_view_from_disk()
            st.success(f"Rebuilt dataset index. Files: {n}")

        if c2.button("🧹 Reset SQL state", use_container_width=True, key="sql_reset_state"):
            for k in list(st.session_state.keys()):
                if k.startswith(("sql_", "math_", "ds_")):
                    st.session_state.pop(k, None)
            st.success("Reset done.")
            st.stop()

    ensure_views()

    # =========================================================
    # Filter builder state
    # =========================================================
    st.session_state.setdefault("ds_conditions", [])
    st.session_state.setdefault("ds_conditions_human", [])
    st.session_state.setdefault("ds_conditions_struct", [])
    st.session_state.setdefault("sql_filter_params_seq", [])
    st.session_state.setdefault("sql_last_human_lines", [])
    st.session_state.setdefault("sql_last_filters_summary", "")
    st.session_state.setdefault("sql_group_selected_params", [])
    st.session_state.setdefault("sql_matched_params_only", [])
    st.session_state.setdefault("sql_group_defs_for_plot", [])
    st.session_state.setdefault("sql_plot_params", [])

    # =========================================================
    # Available params list
    # =========================================================
    params_df = con.execute("""
        SELECT DISTINCT "Parameter Name"
        FROM datasets_kv
        WHERE "Parameter Name" IS NOT NULL AND trim("Parameter Name") <> ''
        ORDER BY 1
    """).fetchdf()
    all_params = params_df["Parameter Name"].astype(str).tolist() if not params_df.empty else []

    if not all_params:
        st.warning("No dataset parameters were found (datasets_kv empty).")
        st.stop()

    # =========================================================
    # FILTER UI
    # =========================================================
    st.subheader("🧱 Filters")

    with st.expander("1) 🔎 Pick parameter / group", expanded=True):
        st.markdown("#### 🔎 Parameter search (single or group)")
        search_q = st.text_input(
            "Search parameters",
            placeholder="Examples: zone avg diameter | tension | furnace",
            key="sql_param_search2",
            help="Type multiple words — results must contain ALL words.",
        )
        tokens = _tokenize_search(search_q)
        matches_all = _match_params_by_tokens(all_params, tokens)
        matches = matches_all[:500]

        zone_map = {}
        for nm in matches:
            zi = _extract_zone_num(nm)
            if zi is not None:
                zone_map.setdefault(zi, []).append(nm)
        zone_nums = sorted(zone_map.keys())
        has_zone_matches = len(zone_nums) >= 2

        cA, cB, cC = st.columns([1.2, 1.2, 1])
        with cA:
            select_mode = st.radio(
                "Pick mode",
                ["Single parameter", "Group from search results"],
                horizontal=True,
                key="sql_pick_mode_builder",
            )
        with cB:
            group_mode = st.radio(
                "Group logic",
                ["ALL (AND)", "ANY (OR)"],
                horizontal=True,
                key="sql_group_logic",
                help="ALL = every selected parameter must match. ANY = at least one matches.",
            )
        with cC:
            st.caption(f"Matches: **{len(matches_all):,}**" + (" (showing first 500)" if len(matches_all) > 500 else ""))

        param_search = (search_q or "").strip().lower()
        shown_params = [pp for pp in all_params if param_search in pp.lower()] if param_search else all_params
        p = st.selectbox(
            "Parameter Name (single)",
            shown_params,
            key="sql_param_name",
            disabled=(select_mode != "Single parameter"),
        )

        st.markdown("#### ✅ Group selection (from search results)")
        if select_mode != "Group from search results":
            st.info("Switch **Pick mode** to **Group from search results** to select many parameters at once.")
            selected_group = []
        else:
            if not matches:
                st.warning("No matches. Type a search above (e.g. `zone avg diameter`).")
                selected_group = []
            else:
                zone_filtered_matches = list(matches)

                if has_zone_matches:
                    zmin, zmax = min(zone_nums), max(zone_nums)
                    z1, z2 = st.slider(
                        "Zone range helper (only affects matches that contain “Zone <n>”)",
                        min_value=zmin,
                        max_value=zmax,
                        value=(zmin, zmax),
                        step=1,
                        key="sql_zone_range",
                    )
                    zone_keep = set([z for z in zone_nums if z1 <= z <= z2])
                    tmp = []
                    for nm in zone_filtered_matches:
                        zi = _extract_zone_num(nm)
                        if zi is None or zi in zone_keep:
                            tmp.append(nm)
                    zone_filtered_matches = tmp

                cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
                with cc1:
                    quick_avg = st.checkbox("Only Avg", value=False, key="sql_quick_avg")
                with cc2:
                    quick_min = st.checkbox("Only Min", value=False, key="sql_quick_min")
                with cc3:
                    quick_max = st.checkbox("Only Max", value=False, key="sql_quick_max")
                with cc4:
                    quick_diam = st.checkbox("Only ‘diameter’", value=False, key="sql_quick_diam")

                def _metric_filter_list(lst):
                    out = list(lst)
                    metric_flags = []
                    if quick_avg: metric_flags.append("avg")
                    if quick_min: metric_flags.append("min")
                    if quick_max: metric_flags.append("max")
                    if metric_flags:
                        out = [nm for nm in out if any(f in nm.lower() for f in metric_flags)]
                    if quick_diam:
                        out = [nm for nm in out if "diameter" in nm.lower()]
                    return out

                zone_filtered_matches = _metric_filter_list(zone_filtered_matches)

                csel1, csel2 = st.columns([1, 1])
                with csel1:
                    if st.button("✅ Select all (shown)", use_container_width=True, key="sql_sel_all_shown"):
                        st.session_state["sql_group_selected_params"] = list(zone_filtered_matches)
                        st.rerun()
                with csel2:
                    if st.button("🧼 Clear selection", use_container_width=True, key="sql_clear_group_sel"):
                        st.session_state["sql_group_selected_params"] = []
                        st.rerun()

                selected_group = st.multiselect(
                    "Selected parameters",
                    options=zone_filtered_matches,
                    default=[x for x in st.session_state.get("sql_group_selected_params", []) if x in zone_filtered_matches],
                    key="sql_group_selected_params",
                    help="Tip: search → Select all → add one condition for all zones.",
                )

    # ---- detect numeric vs categorical based on param_for_type ----
    is_param_numeric = False
    param_values = []
    param_for_type = p
    if select_mode == "Group from search results":
        gg = st.session_state.get("sql_group_selected_params", []) or []
        if gg:
            param_for_type = gg[0]

    try:
        df_param_sample = con.execute(f"""
            SELECT "Value"
            FROM datasets_kv
            WHERE "Parameter Name" = {_lit(param_for_type)}
            LIMIT 200
        """).fetchdf()
        if not df_param_sample.empty:
            sample_series = pd.to_numeric(df_param_sample["Value"], errors="coerce")
            is_param_numeric = sample_series.notna().sum() > 0
            if not is_param_numeric:
                df_opts = con.execute(f"""
                    SELECT DISTINCT trim("Value") AS val
                    FROM datasets_kv
                    WHERE "Parameter Name" = {_lit(param_for_type)}
                      AND trim(COALESCE("Value",'')) <> ''
                    ORDER BY val
                    LIMIT 200
                """).fetchdf()
                param_values = df_opts["val"].astype(str).tolist() if not df_opts.empty else []
    except Exception:
        pass

    with st.expander("2) ⚙️ Condition", expanded=True):
        c_op, c_v1, c_v2 = st.columns([1.2, 2, 2])

        with c_op:
            op = st.selectbox(
                "Operator",
                ["any", "=", "!=", ">", ">=", "<", "<=", "between", "contains"],
                key="sql_op",
            )

        with c_v1:
            if (not is_param_numeric) and op not in ["any", "contains"] and param_values:
                v1 = st.selectbox("Value", options=[""] + param_values, key="sql_v1")
            else:
                v1 = st.text_input("Value", key="sql_v1")

        with c_v2:
            if op == "between":
                if (not is_param_numeric) and param_values:
                    v2 = st.selectbox("Second value (between)", options=[""] + param_values, key="sql_v2")
                else:
                    v2 = st.text_input("Second value (between)", key="sql_v2")
            else:
                v2 = st.text_input("Second value (between)", key="sql_v2")

        c_join, c_not = st.columns([1, 1])
        with c_join:
            joiner = st.radio("Join", ["AND", "OR"], horizontal=True, key="sql_joiner")
        with c_not:
            negate = st.checkbox("NOT", value=False, key="sql_negate")

        st.markdown("#### 🗓️ Time filter (optional)")
        time_on = st.checkbox("Enable time filter", value=False, key="sql_time_on")
        t1, t2 = st.columns(2)
        with t1:
            d_from = st.date_input("From", value=None, key="sql_time_from")
        with t2:
            d_to = st.date_input("To", value=None, key="sql_time_to")

        st.markdown("#### 🧩 Include")
        inc1, inc2, inc3 = st.columns(3)
        with inc1:
            include_draws = st.checkbox("Draws", value=True, key="sql_inc_draws")
        with inc2:
            include_maint = st.checkbox("Maintenance", value=False, key="sql_inc_maint")
        with inc3:
            include_faults = st.checkbox("Faults", value=False, key="sql_inc_faults")

        if not (include_draws or include_maint or include_faults):
            st.warning("Pick at least one: Draws / Maintenance / Faults.")
            st.stop()

    # =========================================================
    # Maintenance + Fault filters (collapsed by default)
    # =========================================================
    with st.expander("🛠 Maintenance & Fault Filters (optional)", expanded=False):
        if include_maint or include_faults:
            st.markdown("#### ⏱️ Event scope")
            event_scope = st.radio(
                "How to constrain Maintenance/Faults relative to your draw filter?",
                [
                    "All events (respect only Maintenance/Fault filters)",
                    "Only within time filter window",
                    "Only within matched draws window",
                ],
                index=2,
                key="sql_event_scope",
                help="Matched draws window = min/max timestamp of the draws you matched (after timestamp fallback).",
            )
            st.markdown("---")
        else:
            event_scope = st.session_state.get("sql_event_scope", "Only within matched draws window")

        st.markdown("##### 🛠 Maintenance")
        maint_on = st.checkbox("Enable maintenance filter", value=False, key="sql_maint_on")
        maint_text = ""
        maint_component = ""
        if maint_on:
            m1, m2 = st.columns(2)
            with m1:
                maint_text = st.text_input(
                    "Maintenance text contains",
                    key="sql_maint_text",
                    placeholder="task / note / source_file",
                )
            with m2:
                comps = []
                try:
                    comps = (
                        con.execute("""
                            SELECT DISTINCT component
                            FROM maintenance_tasks
                            WHERE component IS NOT NULL AND TRIM(component) <> ''
                            ORDER BY component
                        """)
                        .fetchdf()["component"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    comps = []

                if not comps:
                    try:
                        comps = (
                            con.execute("""
                                SELECT DISTINCT component
                                FROM maintenance_actions
                                WHERE component IS NOT NULL AND TRIM(component) <> ''
                                ORDER BY component
                            """)
                            .fetchdf()["component"]
                            .astype(str)
                            .tolist()
                        )
                    except Exception:
                        comps = []

                pick = st.selectbox(
                    "Maintenance component",
                    options=["All"] + comps + ["Custom contains…"],
                    key="sql_maint_comp_pick",
                )
                if pick == "All":
                    maint_component = ""
                elif pick == "Custom contains…":
                    maint_component = st.text_input(
                        "Maintenance component contains",
                        key="sql_maint_comp",
                        placeholder="type part of component name…",
                    )
                else:
                    maint_component = pick

        st.markdown("---")

        st.markdown("##### 🚨 Faults")
        fault_on = st.checkbox("Enable faults filter", value=False, key="sql_fault_on")
        fault_text = ""
        fault_component = ""
        fault_sev = ""
        if fault_on:
            f1, f2, f3 = st.columns([1.2, 1.2, 1])
            with f1:
                fault_text = st.text_input(
                    "Fault text contains",
                    key="sql_fault_text",
                    placeholder="title / description / source_file",
                )
            with f2:
                fault_comps = []
                try:
                    fault_comps = (
                        con.execute("""
                            SELECT DISTINCT component
                            FROM faults_events
                            WHERE component IS NOT NULL AND TRIM(component) <> ''
                            ORDER BY component
                        """)
                        .fetchdf()["component"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    fault_comps = []

                maint_comps = []
                try:
                    maint_comps = (
                        con.execute("""
                            SELECT DISTINCT component
                            FROM maintenance_tasks
                            WHERE component IS NOT NULL AND TRIM(component) <> ''
                            ORDER BY component
                        """)
                        .fetchdf()["component"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    maint_comps = []

                comp_pool = sorted(set([c for c in (fault_comps + maint_comps) if str(c).strip()]))

                pick = st.selectbox(
                    "Fault component",
                    options=["All"] + comp_pool + ["Custom contains…"],
                    key="sql_fault_comp_pick",
                )
                if pick == "All":
                    fault_component = ""
                elif pick == "Custom contains…":
                    fault_component = st.text_input(
                        "Fault component contains",
                        key="sql_fault_comp",
                        placeholder="type part of component name…",
                    )
                else:
                    fault_component = pick
            with f3:
                fault_sev = st.selectbox(
                    "Severity",
                    ["", "low", "medium", "high", "critical"],
                    index=0,
                    key="sql_fault_sev",
                )

    # =========================================================
    # Condition SQL builder (against draws)
    # =========================================================
    def build_cond_sql(p, op, v1, v2):
        p = (p or "").strip()
        v1 = (v1 or "").strip()
        v2 = (v2 or "").strip()
        if not p:
            return None

        base = f'kv."Parameter Name" = {_lit(p)} AND kv._draw = d._draw'

        if op == "any":
            return f"EXISTS (SELECT 1 FROM datasets_kv kv WHERE {base})"

        if op == "contains":
            if not v1:
                return None
            return (
                "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                f"{base} AND CAST(kv.\"Value\" AS VARCHAR) ILIKE '%{_esc(v1)}%')"
            )

        if op == "between":
            if not v1 or not v2:
                return None
            if _is_num(v1) and _is_num(v2):
                return (
                    "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                    f"{base} AND TRY_CAST(kv.\"Value\" AS DOUBLE) BETWEEN {v1} AND {v2})"
                )
            return (
                "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                f"{base} AND kv.\"Value\" BETWEEN {_lit(v1)} AND {_lit(v2)})"
            )

        if not v1:
            return None

        if _is_num(v1):
            return (
                "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
                f"{base} AND TRY_CAST(kv.\"Value\" AS DOUBLE) {op} {v1})"
            )

        return (
            "EXISTS (SELECT 1 FROM datasets_kv kv WHERE "
            f"{base} AND kv.\"Value\" {op} {_lit(v1)})"
        )

    def build_cond_human(p, op, v1, v2):
        v1s = (v1 or "").strip()
        v2s = (v2 or "").strip()
        if op == "any":
            return f"{p}: is present"
        if op == "contains":
            if not v1s:
                return None
            return f"{p}: contains “{v1s}”"
        if op == "between":
            if not v1s or not v2s:
                return None
            return f"{p}: between {v1s} and {v2s}"
        if not v1s:
            return None
        op_map = {"=": "=", "!=": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤"}
        return f"{p}: {op_map.get(op, op)} {v1s}"

    def wrap_not(sql, human, negate):
        if not sql or not human:
            return None, None
        if negate:
            return f"(NOT ({sql}))", f"NOT {human}"
        return f"({sql})", human

    def build_group_cond_sql(params, op, v1, v2, group_logic):
        params = _dedupe_keep_order(params or [])
        if not params:
            return None
        parts = []
        for pp in params:
            s = build_cond_sql(pp, op, v1, v2)
            if s:
                parts.append(f"({s})")
        if not parts:
            return None
        glue = " AND " if str(group_logic).startswith("ALL") else " OR "
        return "(" + glue.join(parts) + ")"

    def build_group_cond_human(params, op, v1, v2, group_logic):
        params = _dedupe_keep_order(params or [])
        if not params:
            return None
        preview = ", ".join(params[:5]) + (" …" if len(params) > 5 else "")
        v1s = (v1 or "").strip()
        v2s = (v2 or "").strip()

        if op == "any":
            cond = "is present"
        elif op == "contains":
            if not v1s:
                return None
            cond = f"contains “{v1s}”"
        elif op == "between":
            if not v1s or not v2s:
                return None
            cond = f"between {v1s} and {v2s}"
        else:
            if not v1s:
                return None
            op_map = {"=": "=", "!=": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤"}
            cond = f"{op_map.get(op, op)} {v1s}"

        return f"{group_logic}: {len(params)} params [{preview}] → {cond}"

    def _kv_predicate_sql(op, v1, v2):
        v1 = (v1 or "").strip()
        v2 = (v2 or "").strip()

        if op == "any":
            return "TRUE"

        if op == "contains":
            if not v1:
                return None
            return f"CAST(kv.\"Value\" AS VARCHAR) ILIKE '%{_esc(v1)}%'"

        if op == "between":
            if not v1 or not v2:
                return None
            if _is_num(v1) and _is_num(v2):
                return f"TRY_CAST(kv.\"Value\" AS DOUBLE) BETWEEN {v1} AND {v2}"
            return f"kv.\"Value\" BETWEEN {_lit(v1)} AND {_lit(v2)}"

        if not v1:
            return None

        if _is_num(v1):
            return f"TRY_CAST(kv.\"Value\" AS DOUBLE) {op} {v1}"

        return f"kv.\"Value\" {op} {_lit(v1)}"

    # =========================================================
    # Add/remove conditions (UI)
    # =========================================================
    with st.expander("3) ➕ Build filter", expanded=True):
        b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

        if b1.button("➕ Add condition", use_container_width=True, key="sql_add_cond"):
            sql_raw = build_cond_sql(p, op, v1, v2)
            human_raw = build_cond_human(p, op, v1, v2)
            sql_cond, human_cond = wrap_not(sql_raw, human_raw, negate)

            if not sql_cond or not human_cond:
                st.warning("Condition not complete.")
            else:
                if st.session_state.ds_conditions:
                    st.session_state.ds_conditions.append(f"{joiner} {sql_cond}")
                else:
                    st.session_state.ds_conditions.append(sql_cond)

                if st.session_state.ds_conditions_human:
                    st.session_state.ds_conditions_human.append(f"{joiner} {human_cond}")
                else:
                    st.session_state.ds_conditions_human.append(human_cond)

                st.session_state.sql_filter_params_seq.append(p)

                st.session_state.ds_conditions_struct.append({
                    "params": [p],
                    "op": op,
                    "v1": v1,
                    "v2": v2,
                    "negate": bool(negate),
                })

        if b2.button("🧩 Add group (from search)", use_container_width=True, key="sql_add_group"):
            params_group = st.session_state.get("sql_group_selected_params", []) if select_mode == "Group from search results" else []
            sql_raw = build_group_cond_sql(params_group, op, v1, v2, st.session_state.get("sql_group_logic", "ALL (AND)"))
            human_raw = build_group_cond_human(params_group, op, v1, v2, st.session_state.get("sql_group_logic", "ALL (AND)"))
            sql_cond, human_cond = wrap_not(sql_raw, human_raw, negate)

            if not sql_cond or not human_cond:
                st.warning("Group condition not complete (select params + set values).")
            else:
                if st.session_state.ds_conditions:
                    st.session_state.ds_conditions.append(f"{joiner} {sql_cond}")
                else:
                    st.session_state.ds_conditions.append(sql_cond)

                if st.session_state.ds_conditions_human:
                    st.session_state.ds_conditions_human.append(f"{joiner} {human_cond}")
                else:
                    st.session_state.ds_conditions_human.append(human_cond)

                for pp in (_dedupe_keep_order(params_group) or []):
                    st.session_state.sql_filter_params_seq.append(pp)

                st.session_state.ds_conditions_struct.append({
                    "params": _dedupe_keep_order(params_group),
                    "op": op,
                    "v1": v1,
                    "v2": v2,
                    "negate": bool(negate),
                    "group_logic": st.session_state.get("sql_group_logic", "ALL (AND)"),
                })

        if b3.button("↩ Remove last", use_container_width=True, key="sql_pop_cond"):
            if st.session_state.ds_conditions:
                st.session_state.ds_conditions.pop()
            if st.session_state.ds_conditions_human:
                st.session_state.ds_conditions_human.pop()
            if st.session_state.sql_filter_params_seq:
                st.session_state.sql_filter_params_seq.pop()
            if st.session_state.get("ds_conditions_struct"):
                st.session_state.ds_conditions_struct.pop()

        if b4.button("🧹 Clear", use_container_width=True, key="sql_clear_cond"):
            st.session_state.ds_conditions = []
            st.session_state.ds_conditions_human = []
            st.session_state.ds_conditions_struct = []
            st.session_state.sql_filter_params_seq = []
            st.session_state.sql_group_defs_for_plot = []
            st.session_state.sql_matched_params_only = []
            st.session_state.sql_values_found_long = pd.DataFrame()
            st.session_state.sql_values_found_wide = pd.DataFrame()
            st.session_state.sql_plot_params = []

        human_lines = list(st.session_state.ds_conditions_human)
        if time_on and d_from and d_to:
            human_lines.append(f"Time: {d_from} → {d_to}")
        if maint_on and (maint_text.strip() or maint_component.strip()):
            human_lines.append(f"Maintenance: {maint_component.strip()} {maint_text.strip()}".strip())
        if fault_on and (fault_text.strip() or fault_component.strip() or fault_sev.strip()):
            human_lines.append(f"Faults: {fault_component.strip()} {fault_text.strip()} {fault_sev.strip()}".strip())

        if human_lines:
            st.success("**Active filter:**\n" + "\n".join([f"- {x}" for x in human_lines]))
        else:
            st.info("No filter set (will include all selected event types).")

        st.session_state["sql_last_human_lines"] = list(human_lines)

    # =========================================================
    # Build draw WHERE (conditions only; time applied after fallback)
    # =========================================================
    where_sql_draws = ""
    if st.session_state.ds_conditions:
        conds = list(st.session_state.ds_conditions)
        if conds:
            first = str(conds[0]).lstrip()
            if first.upper().startswith("OR "):
                conds[0] = first[3:].lstrip()
            elif first.upper().startswith("AND "):
                conds[0] = first[4:].lstrip()
        where_sql_draws = "WHERE " + "\n  ".join(conds)

    # =========================================================
    # Build maint/fault WHERE (text/component/sev only; scope applied later)
    # =========================================================
    maint_where_base = ""
    if maint_on and (maint_text.strip() or maint_component.strip()):
        conds = []
        if maint_text.strip():
            s = maint_text.strip()
            conds.append(
                f"(COALESCE(task,'') ILIKE '%{_esc(s)}%' OR COALESCE(note,'') ILIKE '%{_esc(s)}%' OR COALESCE(source_file,'') ILIKE '%{_esc(s)}%')"
            )
        if maint_component.strip():
            s2 = maint_component.strip()
            if st.session_state.get("sql_maint_comp_pick", "") not in ("All", "Custom contains…"):
                conds.append(f"(COALESCE(component,'') = {_lit(s2)})")
            else:
                conds.append(f"(COALESCE(component,'') ILIKE '%{_esc(s2)}%')")
        maint_where_base = "WHERE " + " AND ".join(conds)

    fault_where_base = ""
    if fault_on and (fault_text.strip() or fault_component.strip() or fault_sev.strip()):
        conds = []
        if fault_text.strip():
            s = fault_text.strip()
            conds.append(
                f"(COALESCE(title,'') ILIKE '%{_esc(s)}%' OR COALESCE(description,'') ILIKE '%{_esc(s)}%' OR COALESCE(source_file,'') ILIKE '%{_esc(s)}%')"
            )
        if fault_component.strip():
            s2 = fault_component.strip()
            if st.session_state.get("sql_fault_comp_pick", "") not in ("All", "Custom contains…"):
                conds.append(f"(COALESCE(component,'') = {_lit(s2)})")
            else:
                conds.append(f"(COALESCE(component,'') ILIKE '%{_esc(s2)}%')")
        if fault_sev.strip():
            conds.append(f"(COALESCE(severity,'') = {_lit(fault_sev.strip())})")
        fault_where_base = "WHERE " + " AND ".join(conds)

    # =========================================================
    # RUN FILTER
    # =========================================================
    st.subheader("▶ Run")

    sql_draws = f"""
    WITH draws AS (
        SELECT _draw, MAX(event_ts) AS event_ts
        FROM datasets_kv
        GROUP BY _draw
    )
    SELECT d._draw, d.event_ts
    FROM draws d
    {where_sql_draws}
    ORDER BY COALESCE(d.event_ts, TIMESTAMP '1900-01-01') ASC, d._draw;
    """

    if st.button("▶ Run filter", type="primary", use_container_width=True, key="sql_run"):
        used_params_run = _dedupe_keep_order(st.session_state.get("sql_filter_params_seq", []))
        human_lines_run = list(st.session_state.get("sql_last_human_lines", []) or [])
        filters_summary_run = _filters_summary_for_draws(used_params_run, human_lines_run)
        st.session_state["sql_last_filters_summary"] = filters_summary_run

        # ---- draws ----
        df_draws = pd.DataFrame(columns=["_draw", "event_ts"])
        if include_draws:
            df_draws = con.execute(sql_draws).fetchdf()

            # Fill missing event_ts using file mtime
            if not df_draws.empty:
                draw_list = df_draws["_draw"].dropna().astype(str).unique().tolist()
                draws_sql = "(" + ",".join(_lit(d) for d in draw_list) + ")"
                df_files = con.execute(f"""
                    SELECT _draw, ANY_VALUE(filename) AS filename
                    FROM datasets_kv
                    WHERE CAST(_draw AS VARCHAR) IN {draws_sql}
                    GROUP BY _draw
                """).fetchdf()
                if not df_files.empty:
                    df_draws = df_draws.merge(df_files, on="_draw", how="left")
                    df_draws["event_ts"] = pd.to_datetime(df_draws["event_ts"], errors="coerce")
                    df_draws["event_ts"] = df_draws["event_ts"].fillna(df_draws["filename"].astype(str).apply(_mtime_ts))
                    df_draws = df_draws.drop(columns=["filename"], errors="ignore")

            # Apply time filter AFTER fallback
            if time_on and d_from and d_to and not df_draws.empty:
                start_ts = pd.Timestamp(d_from)
                end_ts = pd.Timestamp(d_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df_draws["event_ts"] = pd.to_datetime(df_draws["event_ts"], errors="coerce")
                df_draws = df_draws[df_draws["event_ts"].between(start_ts, end_ts)]

            if not df_draws.empty:
                df_draws = df_draws.sort_values(["event_ts", "_draw"], na_position="last")

        # ---- supporting “values found” ----
        values_long = pd.DataFrame()
        values_wide = pd.DataFrame()
        matched_params_only = []
        group_defs = []

        if df_draws is not None and not df_draws.empty:
            df_draws = df_draws.copy()
            df_draws["matched_by"] = filters_summary_run if filters_summary_run else (
                ", ".join(used_params_run) if used_params_run else "(no draw filter)"
            )

            draw_list = df_draws["_draw"].dropna().astype(str).unique().tolist()
            draws_sql = "(" + ",".join(_lit(d) for d in draw_list) + ")"

            match_terms = []
            for cond in (st.session_state.get("ds_conditions_struct") or []):
                pred = _kv_predicate_sql(cond.get("op"), cond.get("v1"), cond.get("v2"))
                if not pred:
                    continue
                if cond.get("negate"):
                    pred = f"NOT ({pred})"
                for pp in (cond.get("params") or []):
                    match_terms.append(f'(kv."Parameter Name" = {_lit(pp)} AND ({pred}))')

            if match_terms:
                match_where = " OR ".join(match_terms)
                values_long = con.execute(f"""
                    SELECT
                        CAST(kv._draw AS VARCHAR) AS _draw,
                        CAST(kv."Parameter Name" AS VARCHAR) AS "Parameter Name",
                        CAST(kv."Value" AS VARCHAR) AS "Value",
                        CAST(COALESCE(kv."Units",'') AS VARCHAR) AS "Units"
                    FROM datasets_kv kv
                    WHERE CAST(kv._draw AS VARCHAR) IN {draws_sql}
                      AND TRIM(COALESCE(CAST(kv."Value" AS VARCHAR),'')) <> ''
                      AND ({match_where})
                """).fetchdf()

            if values_long is not None and not values_long.empty:
                matched_params_only = _dedupe_keep_order(values_long["Parameter Name"].astype(str).tolist())

                values_long = (
                    values_long.drop_duplicates(subset=["_draw", "Parameter Name"], keep="first")
                    .sort_values(["_draw", "Parameter Name"])
                    .reset_index(drop=True)
                )

                values_wide = (
                    values_long.pivot_table(
                        index="_draw",
                        columns="Parameter Name",
                        values="Value",
                        aggfunc="first",
                    )
                    .reset_index()
                )

                join_cols = [c for c in values_wide.columns if c != "_draw"][:12]
                if join_cols:
                    df_draws = df_draws.merge(values_wide[["_draw"] + join_cols], on="_draw", how="left")

            # group defs for plotting
            for i, cond in enumerate(st.session_state.get("ds_conditions_struct") or []):
                params = _dedupe_keep_order(cond.get("params") or [])
                if len(params) < 2:
                    continue

                op_i = str(cond.get("op", "any"))
                v1_i = str(cond.get("v1", "")).strip()
                v2_i = str(cond.get("v2", "")).strip()
                if op_i == "any":
                    rhs = "present"
                elif op_i == "contains":
                    rhs = f"contains '{v1_i}'" if v1_i else "contains"
                elif op_i == "between":
                    rhs = f"between {v1_i}..{v2_i}" if (v1_i and v2_i) else "between"
                else:
                    rhs = f"{op_i} {v1_i}" if v1_i else op_i

                gl = str(cond.get("group_logic", st.session_state.get("sql_group_logic", "ALL (AND)")))
                neg = " NOT" if bool(cond.get("negate")) else ""
                label = f"🧩 Group {i+1}: {gl}{neg} ({len(params)} params) → {rhs}"

                group_defs.append({"label": label, "params": params})

        # ---- event scope window (for maint/fault overlay) ----
        scope_start = None
        scope_end = None

        if (include_maint or include_faults) and event_scope == "Only within time filter window" and time_on and d_from and d_to:
            scope_start = pd.Timestamp(d_from)
            scope_end = pd.Timestamp(d_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        if (include_maint or include_faults) and event_scope == "Only within matched draws window" and include_draws and df_draws is not None and not df_draws.empty:
            tmin = pd.to_datetime(df_draws["event_ts"], errors="coerce").min()
            tmax = pd.to_datetime(df_draws["event_ts"], errors="coerce").max()
            if pd.notna(tmin) and pd.notna(tmax):
                scope_start = tmin
                scope_end = tmax

        def _add_scope(where_base: str, ts_col: str):
            if scope_start is None or scope_end is None:
                return where_base
            add = f"({ts_col} BETWEEN {_lit(scope_start)} AND {_lit(scope_end)})"
            return (where_base + " AND " if where_base else "WHERE ") + add

        maint_where = _add_scope(maint_where_base, "action_ts")
        fault_where = _add_scope(fault_where_base, "fault_ts")

        # ---- Load draw KV rows for matched draws ----
        df_kv = pd.DataFrame()
        if include_draws and df_draws is not None and not df_draws.empty:
            draw_list = df_draws["_draw"].dropna().astype(str).unique().tolist()
            draws_sql = "(" + ",".join(_lit(d) for d in draw_list) + ")"
            df_kv = con.execute(f"""
                SELECT
                    'dataset'::VARCHAR AS source_kind,
                    NULL::VARCHAR AS source_file,
                    event_ts,
                    event_id,
                    _draw,
                    filename,
                    "Parameter Name",
                    "Value",
                    "Units"
                FROM datasets_kv
                WHERE CAST(_draw AS VARCHAR) IN {draws_sql}
            """).fetchdf()

        # ---- Maintenance events ----
        df_m = pd.DataFrame()
        if include_maint:
            df_m = con.execute(f"""
                SELECT
                    'maintenance'::VARCHAR AS source_kind,
                    COALESCE(source_file,'')::VARCHAR AS source_file,
                    action_ts AS event_ts,
                    CAST(action_id AS VARCHAR) AS event_id,
                    NULL::VARCHAR AS _draw,
                    NULL::VARCHAR AS filename,
                    COALESCE(component,'')::VARCHAR AS "Parameter Name",
                    (trim(COALESCE(task,'')) || CASE WHEN COALESCE(note,'') <> '' THEN (' | ' || note) ELSE '' END)::VARCHAR AS "Value",
                    ''::VARCHAR AS "Units"
                FROM maintenance_actions
                {maint_where}
            """).fetchdf()

        # ---- Fault events ----
        df_f = pd.DataFrame()
        if include_faults:
            df_f = con.execute(f"""
                SELECT
                    'fault'::VARCHAR AS source_kind,
                    COALESCE(source_file,'')::VARCHAR AS source_file,
                    fault_ts AS event_ts,
                    CAST(fault_id AS VARCHAR) AS event_id,
                    NULL::VARCHAR AS _draw,
                    NULL::VARCHAR AS filename,
                    COALESCE(component,'')::VARCHAR AS "Parameter Name",
                    (trim(COALESCE(title,'')) || CASE WHEN COALESCE(severity,'') <> '' THEN (' | severity=' || severity) ELSE '' END ||
                     CASE WHEN COALESCE(description,'') <> '' THEN (' | ' || description) ELSE '' END)::VARCHAR AS "Value",
                    ''::VARCHAR AS "Units"
                FROM faults_events
                {fault_where}
            """).fetchdf()

        df_all = pd.concat([df_kv, df_m, df_f], ignore_index=True)

        st.session_state["sql_df_all"] = df_all
        st.session_state["sql_matched_draws"] = df_draws
        st.session_state["sql_values_found_long"] = values_long if values_long is not None else pd.DataFrame()
        st.session_state["sql_values_found_wide"] = values_wide if values_wide is not None else pd.DataFrame()
        st.session_state["sql_matched_params_only"] = matched_params_only
        st.session_state["sql_group_defs_for_plot"] = group_defs

        st.session_state.pop("sql_selected_event_key", None)
        st.session_state.pop("math_selected_event_key", None)

        with st.expander("✅ Results summary", expanded=True):
            if include_draws:
                md = st.session_state.get("sql_matched_draws", pd.DataFrame())
                st.success(f"Matched draws: {0 if md is None else len(md):,}")

                fs = st.session_state.get("sql_last_filters_summary", "")
                if fs:
                    show_filters = st.checkbox(
                        "🔎 Show filters used to match these draws",
                        value=False,
                        key="sql_show_filters_summary",
                    )
                    if show_filters:
                        st.write(fs)

                show_values = st.checkbox(
                    "📌 Show values that actually matched your filter",
                    value=True,
                    key="sql_show_matched_values",
                )
                if show_values:
                    values_long2 = st.session_state.get("sql_values_found_long", pd.DataFrame())
                    values_wide2 = st.session_state.get("sql_values_found_wide", pd.DataFrame())
                    if values_long2 is None or values_long2.empty:
                        st.info("No supporting matched values found (maybe NOT-only logic or 'any' conditions).")
                    else:
                        st.caption("Long view (only rows that satisfied your conditions):")
                        st.dataframe(values_long2, use_container_width=True, height=260)
                        if values_wide2 is not None and not values_wide2.empty:
                            st.caption("Wide view (one row per draw):")
                            st.dataframe(values_wide2, use_container_width=True, height=260)

                st.caption("Matched draws table:")
                st.dataframe(md, use_container_width=True, height=260)
            else:
                st.info("Draws excluded (timeline can show only Maintenance/Faults).")

    # =========================================================
    # Guards
    # =========================================================
    if "sql_df_all" not in st.session_state:
        st.stop()

    df_all = st.session_state["sql_df_all"]
    if df_all is None or df_all.empty:
        st.warning("No results loaded. Run filter.")
        st.stop()

    # =========================================================
    # Event details
    # =========================================================
    def render_event_details(event_key: str):
        if not event_key or ":" not in event_key:
            return
        kind, eid = event_key.split(":", 1)
        kind = (kind or "").strip()

        st.markdown("### 🔎 Event details")
        st.caption(f"Selected: **{event_key}**")

        if kind == "dataset":
            df_kv2 = con.execute(f"""
                SELECT
                    event_ts,
                    event_id,
                    _draw,
                    filename,
                    "Parameter Name",
                    "Value",
                    "Units"
                FROM datasets_kv
                WHERE CAST(_draw AS VARCHAR) = {_lit(eid)}
                   OR CAST(event_id AS VARCHAR) = {_lit(eid)}
                ORDER BY "Parameter Name"
            """).fetchdf()
            if df_kv2.empty:
                st.warning("No KV rows found for this draw.")
                return
            top = df_kv2.head(1)
            st.markdown(f"**Draw:** `{top['event_id'].iloc[0]}`")
            if "event_ts" in top.columns:
                st.caption(f"Time: {top['event_ts'].iloc[0]}")
            st.dataframe(df_kv2[["Parameter Name", "Value", "Units"]], use_container_width=True, height=460)

        elif kind == "maintenance":
            df_act = con.execute(f"""
                SELECT
                    action_ts,
                    component,
                    task,
                    task_id,
                    tracking_mode,
                    hours_source,
                    done_date,
                    done_hours,
                    done_draw,
                    actor,
                    note,
                    source_file
                FROM maintenance_actions
                WHERE CAST(action_id AS VARCHAR) = {_lit(eid)}
                LIMIT 1
            """).fetchdf()
            if df_act.empty:
                st.warning("Maintenance action not found.")
                return
            st.dataframe(df_act, use_container_width=True, height=180)

        elif kind == "fault":
            df_fault = con.execute(f"""
                SELECT
                    fault_ts,
                    component,
                    severity,
                    title,
                    description,
                    actor,
                    source_file,
                    related_draw
                FROM faults_events
                WHERE CAST(fault_id AS VARCHAR) = {_lit(eid)}
                LIMIT 1
            """).fetchdf()
            if df_fault.empty:
                st.warning("Fault not found.")
                return
            st.dataframe(df_fault, use_container_width=True, height=180)
        else:
            st.info("Unknown event type.")

    # =========================================================
    # VISUAL LAB  (FIXED: main plot always shows numeric/group;
    # text is a categorical Y axis on main plot; maint/fault only below)
    # =========================================================
    # =========================================================
    # VISUAL LAB  (Main plot: numeric/group + auto text categorical axes;
    # Events lane: maintenance+fault only)
    # =========================================================
    # =========================================================
    # VISUAL LAB  (Main plot: numeric/group + auto text categorical axes;
    # Events lane: maintenance+fault only)
    # =========================================================
    st.subheader("📈 Visual Lab")

    df = df_all.copy()
    df["event_ts"] = pd.to_datetime(df.get("event_ts"), errors="coerce")

    ds_kv = df[df["source_kind"].astype(str).eq("dataset")].copy() if "source_kind" in df.columns else pd.DataFrame()
    maint_kv = df[
        df["source_kind"].astype(str).eq("maintenance")].copy() if "source_kind" in df.columns else pd.DataFrame()
    fault_kv = df[df["source_kind"].astype(str).eq("fault")].copy() if "source_kind" in df.columns else pd.DataFrame()

    show_draw_traces = st.toggle(
        "Show Draw traces",
        value=True,
        key="sql_show_draw_traces",
        help="Turn OFF if you want only Maintenance/Faults.",
    )

    wide = pd.DataFrame()
    numeric_all = []
    if not ds_kv.empty:
        if "filename" in ds_kv.columns:
            ds_kv["event_ts"] = ds_kv["event_ts"].fillna(ds_kv["filename"].astype(str).apply(_mtime_ts))

        ds_kv = ds_kv[ds_kv["event_ts"].notna()].copy()
        ds_kv["event_key"] = "dataset:" + ds_kv["event_id"].astype(str)

        wide = (
            ds_kv.pivot_table(
                index=["event_ts", "event_key"],
                columns="Parameter Name",
                values="Value",
                aggfunc="first",
            )
            .reset_index()
            .sort_values("event_ts")
        )

        META = {"event_ts", "event_key"}
        all_plot_params = [c for c in wide.columns if c not in META]
        numeric_all = [c for c in all_plot_params if pd.to_numeric(wide[c], errors="coerce").notna().sum() > 0]

    # ------------------- Plot picker -------------------
    numeric_chosen, cat_chosen, group_chosen, chosen_all = [], [], [], []

    if wide is not None and not wide.empty:
        matched_params_only = st.session_state.get("sql_matched_params_only", []) or []

        st.markdown("#### 🎛 Plot helper")

        pick_mode = st.radio(
            "Parameter picker",
            [
                "Only parameters used in the filter",
                "Only parameters that actually matched (supporting rows)",
                "Any parameter from matched draws",
            ],
            horizontal=True,
            key="sql_pick_mode",
        )

        used_params2 = _dedupe_keep_order(st.session_state.get("sql_filter_params_seq", []))
        if pick_mode.startswith("Only parameters used"):
            pool = [pp for pp in used_params2 if pp in wide.columns]
        elif pick_mode.startswith("Only parameters that actually matched"):
            pool = [pp for pp in matched_params_only if pp in wide.columns]
        else:
            pool = [c for c in wide.columns if c not in ("event_ts", "event_key")]

        group_defs = st.session_state.get("sql_group_defs_for_plot", []) or []
        group_labels = [g.get("label") for g in group_defs if str(g.get("label", "")).strip()]
        pool = list(group_labels) + [pp for pp in pool if pp not in group_labels]

        prev = [x for x in (st.session_state.get("sql_plot_params") or []) if x in pool]
        if prev != (st.session_state.get("sql_plot_params") or []):
            st.session_state["sql_plot_params"] = prev

        cbtn1, cbtn2 = st.columns([1, 1])
        with cbtn1:
            if st.button("🎯 Set plot = all matched params", use_container_width=True, key="sql_apply_plot_matched"):
                st.session_state["sql_plot_params"] = [pp for pp in matched_params_only if pp in pool]
                st.rerun()
        with cbtn2:
            if st.button("🧹 Clear plot selection", use_container_width=True, key="sql_clear_plot_sel"):
                st.session_state["sql_plot_params"] = []
                st.rerun()

        chosen_all = st.multiselect(
            "Draw parameters to plot",
            pool,
            key="sql_plot_params",
        )

        group_label_set = set(group_labels)
        group_chosen = [x for x in (chosen_all or []) if x in group_label_set]
        normal_chosen = [x for x in (chosen_all or []) if x not in group_label_set]

        for c in (normal_chosen or []):
            if c in ("event_ts", "event_key"):
                continue
            s_num = pd.to_numeric(wide[c], errors="coerce")
            if s_num.notna().sum() > 0:
                numeric_chosen.append(c)
            else:
                # Treat as text/categorical if it has any non-empty string values
                if wide[c].astype(str).replace("nan", "").str.strip().ne("").any():
                    cat_chosen.append(c)
    else:
        st.info("No draw timestamps available. Maintenance/Faults can still show.")

    # =========================================================
    # Plot: row1 main draw plot (MULTI-Y + auto TEXT categorical axes),
    # row2 events lane (maintenance+fault only)
    # =========================================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.06,
    )

    palette = list(getattr(qualitative, "Plotly", [])) or [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    st.markdown("#### ⚙️ Main plot axes")

    # One axis per requested numeric/group parameter (no sharing)
    total_requested = len(numeric_chosen) + len(group_chosen)
    numeric_axis_count = max(1, int(total_requested))

    st.caption(
        f"Numeric/group Y-axes: **{numeric_axis_count}** (one per selected numeric/group parameter). "
        "Text parameters automatically get their own categorical Y-axes."
    )


    # IMPORTANT:
    # make_subplots(rows=2, cols=1) uses:
    #   row 1 -> y  (layout.yaxis)
    #   row 2 -> y2 (layout.yaxis2)
    # We must NOT use y2 for the main plot. So main-plot overlays start at y3.

    def _yaxis_id(i: int) -> str:
        # i is 0-based axis index for MAIN PLOT axes.
        # make_subplots uses y for row1 and y2 for row2, so we skip y2.
        return "y" if i == 0 else f"y{i + 2}"  # y, y3, y4... (skip y2)


    def _yaxis_layout_name(i: int) -> str:
        return "yaxis" if i == 0 else f"yaxis{i + 2}"  # yaxis, yaxis3, yaxis4... (skip yaxis2)


    axis_owner = {}  # numeric axis index -> {name,color}

    # ---- MAIN PLOT traces ----
    if show_draw_traces and wide is not None and not wide.empty:

        # ---------- numeric traces ----------
        for i, col in enumerate(list(numeric_chosen or [])):
            y = pd.to_numeric(wide[col], errors="coerce")
            if y.notna().sum() == 0:
                continue

            colr = palette[i % len(palette)]
            ax_i = i  # one axis per numeric parameter
            yaxis_id = _yaxis_id(ax_i)

            if ax_i not in axis_owner:
                axis_owner[ax_i] = {"name": col, "color": colr}

            fig.add_trace(
                go.Scatter(
                    x=wide["event_ts"],
                    y=y,
                    mode="lines+markers",
                    name=col,
                    customdata=wide["event_key"],
                    hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>",
                    line=dict(color=colr),
                    marker=dict(color=colr),
                ),
                row=1, col=1
            )
            # Plotly's make_subplots can override trace axes; force it AFTER adding
            fig.data[-1].update(xaxis="x", yaxis=yaxis_id)

        # ---------- group traces (mean) ----------
        if group_chosen:
            by_label = {
                g.get("label"): g
                for g in (st.session_state.get("sql_group_defs_for_plot", []) or [])
                if g.get("label")
            }
            base_idx = len(list(numeric_chosen or []))
            for j, glabel in enumerate(group_chosen):
                g = by_label.get(glabel)
                if not g:
                    continue
                params = [p for p in (g.get("params") or []) if p in wide.columns]
                if not params:
                    continue

                y_stack = pd.concat([pd.to_numeric(wide[p], errors="coerce") for p in params], axis=1)
                y_mean = y_stack.mean(axis=1, skipna=True)
                if y_mean.notna().sum() == 0:
                    continue

                idx = base_idx + j
                colr = palette[idx % len(palette)]
                ax_i = idx  # one axis per group parameter
                yaxis_id = _yaxis_id(ax_i)

                if ax_i not in axis_owner:
                    axis_owner[ax_i] = {"name": glabel, "color": colr}

                fig.add_trace(
                    go.Scatter(
                        x=wide["event_ts"],
                        y=y_mean,
                        mode="lines+markers",
                        name=glabel,
                        customdata=wide["event_key"],
                        hovertemplate=f"<b>{glabel}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>",
                        line=dict(color=colr, width=3),
                        marker=dict(color=colr),
                    ),
                    row=1, col=1
                )
                fig.data[-1].update(xaxis="x", yaxis=yaxis_id)

        # ---------- auto TEXT categorical axes ----------
        # If user selected any categorical params in the multiselect, render each as its own tick-labeled axis.
        text_params = list(cat_chosen or [])
        text_params = text_params[:3]  # safety limit

        text_axis_start = int(max(1, numeric_axis_count))  # after numeric axes

        for t_idx, text_param in enumerate(text_params):
            raw = wide[text_param].astype(str).fillna("").replace("nan", "").str.strip()
            mask = raw.ne("")
            if mask.sum() == 0:
                continue

            cats = _dedupe_keep_order(raw[mask].tolist())[:50]
            # Use real text values on Y (categorical axis)
            y_text = raw.where(mask, other=np.nan)

            ax_i = text_axis_start + t_idx
            text_yaxis_id = _yaxis_id(ax_i)
            text_yaxis_layout = _yaxis_layout_name(ax_i)

            colr = palette[(text_axis_start + t_idx) % len(palette)]

            fig.add_trace(
                go.Scatter(
                    x=wide["event_ts"],
                    y=y_text,
                    mode="markers",
                    name=f"{text_param} (text)",
                    customdata=wide["event_key"],
                    hovertemplate=f"<b>{text_param}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>",
                    marker=dict(
                        size=7,
                        opacity=0.95,
                        symbol="diamond",
                        color=colr,
                        line=dict(width=1, color=colr)
                    ),
                    showlegend=True,
                ),
                row=1, col=1
            )
            fig.data[-1].update(xaxis="x", yaxis=text_yaxis_id)

            fig.update_layout(
                **{
                    text_yaxis_layout: dict(
                        title=dict(text=f"{text_param}", font=dict(size=11, color=colr)),
                        overlaying="y",
                        side="left",
                        anchor="free",
                        position=0,
                        autoshift=True,
                        shift=-140 - (t_idx * 90),
                        showgrid=False,
                        zeroline=False,
                        showticklabels=True,
                        ticks="outside",
                        tickfont=dict(size=10, color=colr),
                        showline=True,
                        linecolor=colr,
                        type="category",
                        categoryorder="array",
                        categoryarray=cats,
                        automargin=True,
                    )
                }
            )

    # ---- Configure numeric MULTI Y axes (colored like traces) ----
    base_title = axis_owner.get(0, {}).get("name", "Value")
    base_color = axis_owner.get(0, {}).get("color", "#444444")

    fig.update_layout(
        yaxis=dict(
            title=dict(text=base_title, font=dict(color=base_color, size=11)),
            tickfont=dict(color=base_color),
            showticklabels=True,
            ticks="outside",
            showgrid=True,
            zeroline=False,
            side="left",
            showline=True,
            linecolor=base_color,
            automargin=True,
        ),
    )

    # numeric overlay axes for main plot: y3.. (y2 is reserved for row 2 events lane)
    # Make EVERY numeric/group parameter get its own visible Y axis on the right.
    for i in range(1, int(max(1, numeric_axis_count))):
        ax_name = _yaxis_layout_name(i)
        ttl = axis_owner.get(i, {}).get("name", f"Y{i + 1}")
        colr = axis_owner.get(i, {}).get("color", "#666666")

        fig.update_layout(
            **{
                ax_name: dict(
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=1.0,
                    # push each axis further right so each is visible
                    autoshift=True,
                    shift=(i - 1) * 65,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=True,
                    ticks="outside",
                    title=dict(text=ttl, font=dict(color=colr, size=11)),
                    tickfont=dict(color=colr),
                    showline=True,
                    linecolor=colr,
                    automargin=True,
                )
            }
        )

    # Force tick labels ON (sometimes overlay+free can hide them)
    fig.update_yaxes(showticklabels=True, ticks="outside", row=1, col=1)

    # =========================================================
    # Events lane (row 2): ONLY Maintenance + Faults
    # =========================================================
    y_fault = 0.30
    y_maint = 0.70

    show_maint_overlay = st.toggle(
        "Show Maintenance overlay",
        value=not maint_kv.empty,
        key="sql_show_maint_overlay",
        disabled=maint_kv.empty,
    )
    show_fault_overlay = st.toggle(
        "Show Faults overlay",
        value=not fault_kv.empty,
        key="sql_show_fault_overlay",
        disabled=fault_kv.empty,
    )

    if show_maint_overlay and not maint_kv.empty:
        mm = maint_kv.copy()
        mm["event_ts"] = pd.to_datetime(mm["event_ts"], errors="coerce")
        mm = mm[mm["event_ts"].notna()].copy()
        mm["event_key"] = "maintenance:" + mm["event_id"].astype(str)
        fig.add_trace(
            go.Scatter(
                x=mm["event_ts"],
                y=[y_maint] * len(mm),
                mode="markers",
                name="Maintenance",
                marker=dict(size=12, symbol="triangle-up", color="#BBBBBB"),
                customdata=mm["event_key"],
                text=("<b>" + mm["Parameter Name"].astype(str) + "</b><br>" + mm["Value"].astype(str)),
                hovertemplate="%{text}<br>%{x}<br>%{customdata}<extra></extra>",
            ),
            row=2, col=1
        )

    if show_fault_overlay and not fault_kv.empty:
        ff = fault_kv.copy()
        ff["event_ts"] = pd.to_datetime(ff["event_ts"], errors="coerce")
        ff = ff[ff["event_ts"].notna()].copy()
        ff["event_key"] = "fault:" + ff["event_id"].astype(str)
        fig.add_trace(
            go.Scatter(
                x=ff["event_ts"],
                y=[y_fault] * len(ff),
                mode="markers",
                name="Faults",
                marker=dict(size=13, symbol="x", color="#FF6666"),
                customdata=ff["event_key"],
                text=("<b>" + ff["Parameter Name"].astype(str) + "</b><br>" + ff["Value"].astype(str)),
                hovertemplate="%{text}<br>%{x}<br>%{customdata}<extra></extra>",
            ),
            row=2, col=1
        )

    fig.update_yaxes(
        row=2, col=1,
        range=[0, 1],
        tickmode="array",
        tickvals=[y_fault, y_maint],
        ticktext=["Faults", "Maintenance"],
        showgrid=False,
        zeroline=False,
        title_text="",
        showticklabels=True,
        ticks="outside",
    )

    # X padding so edges are not cropped
    all_ts = []
    if show_draw_traces and wide is not None and not wide.empty:
        all_ts.append(pd.to_datetime(wide["event_ts"], errors="coerce"))
    if show_maint_overlay and not maint_kv.empty:
        all_ts.append(pd.to_datetime(maint_kv["event_ts"], errors="coerce"))
    if show_fault_overlay and not fault_kv.empty:
        all_ts.append(pd.to_datetime(fault_kv["event_ts"], errors="coerce"))

    if all_ts:
        ts_cat = pd.concat(all_ts, ignore_index=True).dropna()
        if not ts_cat.empty:
            xmin, xmax = ts_cat.min(), ts_cat.max()
            if pd.notna(xmin) and pd.notna(xmax) and xmax > xmin:
                span = xmax - xmin
                pad = span * 0.07
                fig.update_xaxes(range=[xmin - pad, xmax + pad])

    # Left margin grows when text axes exist
    left_margin = 170 + (85 * min(3, len(list(cat_chosen or [])))) if show_draw_traces else 170

    # Right margin grows when many numeric axes are on the right
    right_margin = 140 + (70 * max(0, int(numeric_axis_count) - 1))
    right_margin = min(700, right_margin)

    fig.update_layout(
        height=780,
        margin=dict(l=left_margin, r=right_margin, t=70, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        title="Timeline: Draws + Events (filtered)",
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)

    sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="sql_vis_plot")

    selected_key = None
    try:
        if isinstance(sel, dict):
            pts = sel.get("selection", {}).get("points", sel.get("points", []))
            if pts:
                selected_key = pts[0].get("customdata")
    except Exception:
        selected_key = None

    if selected_key:
        st.session_state["sql_selected_event_key"] = selected_key

    if st.session_state.get("sql_selected_event_key"):
        with st.expander("📌 Clicked event details", expanded=True):
            render_event_details(st.session_state["sql_selected_event_key"])

    # =========================================================
    # 🧮 MATH LAB
    # =========================================================
    st.subheader("🧮 Math Lab")

    if wide is None or wide.empty:
        st.info("No draw events available for Math Lab (run filter with Draws enabled).")
        st.stop()

    if not numeric_all:
        st.info("No numeric draw parameters available for Math Lab.")
        st.stop()

    st.caption("Expressions use **A**, **B**, **C** and **np**. Example: `A/B`, `np.log10(A)`")

    var_count = st.radio("How many parameters?", [1, 2, 3], horizontal=True, key="math_var_count")

    A_name = st.selectbox("A", numeric_all, key="math_A_name")
    B_name = None
    C_name = None
    if var_count >= 2:
        B_name = st.selectbox("B", [pp for pp in numeric_all if pp != A_name], key="math_B_name")
    if var_count >= 3:
        C_name = st.selectbox("C", [pp for pp in numeric_all if pp not in (A_name, B_name)], key="math_C_name")

    st.session_state.setdefault("math_expr", "A")
    expr = st.text_input("Expression", value=str(st.session_state["math_expr"]), key="math_expr_input").strip()
    st.session_state["math_expr"] = expr

    if not re.fullmatch(r"[0-9A-Za-z_\.\+\-\*\/\(\)\s,]+", expr or ""):
        st.error("Expression contains unsupported characters.")
        st.stop()


    def _series(name):
        if not name:
            return pd.Series([np.nan] * len(wide), index=wide.index)
        return pd.to_numeric(wide[name], errors="coerce").astype(float)


    A = _series(A_name)
    B = _series(B_name) if var_count >= 2 else pd.Series([np.nan] * len(wide), index=wide.index)
    C = _series(C_name) if var_count >= 3 else pd.Series([np.nan] * len(wide), index=wide.index)

    try:
        Y = eval(expr, {"__builtins__": {}}, {"np": np, "A": A, "B": B, "C": C})
        if isinstance(Y, (int, float, np.number)):
            Y = pd.Series([float(Y)] * len(wide), index=wide.index)
        elif isinstance(Y, np.ndarray):
            Y = pd.Series(Y, index=wide.index)
        elif not isinstance(Y, pd.Series):
            Y = pd.Series(Y, index=wide.index)
        Y = pd.to_numeric(Y, errors="coerce")
    except Exception as e:
        st.error(f"Expression error: {e}")
        st.stop()

    out = wide[["event_ts", "event_key"]].copy()
    out["math"] = Y
    out = out.dropna(subset=["math"]).sort_values("event_ts")

    if out.empty:
        st.warning("No values computed.")
        st.stop()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=out["event_ts"],
        y=out["math"],
        mode="lines+markers",
        name=expr,
        customdata=out["event_key"],
        hovertemplate=f"<b>{expr}</b><br>%{{x}}<br>%{{y}}<br>%{{customdata}}<extra></extra>"
    ))
    fig2.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=60, b=50),
        title="Math Lab result (click a point to inspect draw)",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="closest",
    )

    sel2 = st.plotly_chart(fig2, use_container_width=True, on_select="rerun", key="math_plot")

    selected_key2 = None
    try:
        if isinstance(sel2, dict):
            pts = sel2.get("selection", {}).get("points", sel2.get("points", []))
            if pts:
                selected_key2 = pts[0].get("customdata")
    except Exception:
        selected_key2 = None

    if selected_key2:
        st.session_state["math_selected_event_key"] = selected_key2

    if st.session_state.get("math_selected_event_key"):
        with st.expander("📌 Clicked event details (Math)", expanded=True):
            render_event_details(st.session_state["math_selected_event_key"])

    with st.expander("Math table", expanded=False):
        st.dataframe(out, use_container_width=True, height=420)