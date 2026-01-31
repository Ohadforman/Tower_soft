import streamlit as st
import base64
import pandas as pd
import json
CSV_SELECTION_FILE = "selected_csv.json"
import plotly.express as px
import math
import duckdb, os
import os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
DB_PATH = os.path.join(os.getcwd(), "tower.duckdb")

if "tower_con" not in st.session_state:
    st.session_state.tower_con = duckdb.connect(DB_PATH)

con = st.session_state.tower_con
def save_selected_csv(selected_csv):
    """Save the selected CSV file path in a JSON file"""
    with open(CSV_SELECTION_FILE, 'w') as file:
        json.dump({"selected_csv": selected_csv}, file)
def load_selected_csv():
    """Load the selected CSV file path from the JSON file"""
    if os.path.exists(CSV_SELECTION_FILE):
        with open(CSV_SELECTION_FILE, 'r') as file:
            data = json.load(file)
            return data.get("selected_csv")
    return None
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

    maint_folder = os.path.join(base_dir, "maintenance")
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
            for col in ["Tracking_Mode", "Interval_Value", "Interval_Unit", "Hours_Source", "Due_Threshold_Days", "Last_Done_Date", "Last_Done_Hours"]:
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
with open("config_coating.json", "r") as config_file:
    config = json.load(config_file)
coatings = config.get("coatings", {})
dies = config.get("dies", {})
with open("config_coating.json", "r") as config_file:
    config = json.load(config_file)
# Ensure coatings and dies are properly loaded
if not coatings or not dies:
    st.error("Coatings and/or Dies not configured in config_coating.json")
    st.stop()

tab_labels = [
    "🏠 Home",
    "📅 Schedule",
    "🍃 Tower state - Consumables and dies",
    "⚙️ Process Setup",
    "🧰 Maintenance",
    "📦 Order Draw",
    "🛠️ Tower Parts",
    "📋 Protocols",
    "📊 Dashboard",
    "🧪 Development Process",
    "📝 History Log",
    "✅ Closed Processes"
]

if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = None
if "tab_select" not in st.session_state:
    st.session_state["tab_select"] = "🏠 Home"
if "last_tab" not in st.session_state:
    st.session_state["last_tab"] = "🏠 Home"
    if "tab_labels" not in st.session_state:
        st.session_state["tab_labels"] = tab_labels
if st.session_state.get("selected_tab"):
    st.session_state["tab_select"] = st.session_state["selected_tab"]
    st.session_state["last_tab"] = st.session_state["selected_tab"]
st.session_state["selected_tab"] = None
if "good_zones" not in st.session_state:
    st.session_state["good_zones"] = []
def get_base64_image(image_path):
    """Encodes an image to base64 format for inline CSS."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# Ensure the image file exists in the project folder
image_base64 = get_base64_image("Martin.jpeg")
def calculate_coating_thickness(entry_fiber_diameter, die_diameter, mu, rho, L, V, g):
    """Calculates coating thickness and coated fiber diameter."""
    R = (die_diameter / 2) * 10**-6  # Die Radius (m)
    r = (entry_fiber_diameter / 2) * 10**-6  # Fiber Radius (m)
    k = r / R
    if k <= 0:
        return entry_fiber_diameter  # Return input value if k is invalid
    ln_k = math.log(k)

    # Pressure drop calculation
    delta_P = L * rho * g

    # Φ calculation
    Phi = (delta_P * R**2) / (8 * mu * L * V)

    # Calculate the coating thickness (t)
    term1 = Phi * (1 - k**4 + ((1 - k**2)**2) / ln_k)
    term2 = - (k**2 + (1 - k**2) / (2 * ln_k))  # Ensure valid sqrt input
    t = R * ((term1 + term2 + k**2)**0.5 - k)

    coated_fiber_diameter = entry_fiber_diameter + (t * 2 * 1e6)  # Convert thickness to microns
    return coated_fiber_diameter
def evaluate_viscosity(T, function_str):
    """Computes viscosity by evaluating the stored function string from config."""
    try:
        return eval(function_str, {"T": T, "math": math})
    except Exception as e:
        st.error(f"Error evaluating viscosity function: {e}")
        return None
# Load configuration
DATA_FOLDER = config.get("logs_directory", "./logs")
HISTORY_FILE = "history_log.csv"
PARTS_DIRECTORY = config.get("parts_directory", "./parts")
DEVELOPMENT_FILE = "development_process.csv"
DATASET_FOLDER = "./data_set_csv"

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
            "📈 Correlation & Outliers",
            "🛠️ Tower Parts",
            "📋 Protocols"
        ],
        "📚 Monitoring &  Research": [
            "🧪 SQL Lab",
            "🧪 Development Process",
        ],
        "🗂 Documentation ": [
            "📝 History Log",
            "✅ Closed Processes"
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
    if st.session_state.get("nav_group_select") not in GROUPS:
        st.session_state["nav_group_select"] = desired_group
    if st.session_state.get("tab_select") not in tab_labels:
        st.session_state["tab_select"] = desired_tab

    def _on_group_change():
        g = st.session_state.get("nav_group_select")
        last_by_group = st.session_state.get("nav_last_tab_by_group", {})
        next_tab = last_by_group.get(g, NAV_GROUPS[g][0])
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

df = pd.DataFrame()  # Initialize an empty DataFrame to avoid NameError

if tab_selection == "📊 Dashboard":
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".csv")]

    if not csv_files:
        st.error("No CSV files found in the directory.")
        st.stop()

    # newest first (by modified time)
    csv_files_sorted = sorted(
        csv_files,
        key=lambda fn: os.path.getmtime(os.path.join(DATA_FOLDER, fn)),
        reverse=True,
    )
    latest_file = csv_files_sorted[0]

    # ✅ Force default to latest ONLY when entering Dashboard
    if st.session_state.get("_last_tab_for_log_default") != "📊 Dashboard":
        st.session_state["dataset_select"] = latest_file

    st.session_state["_last_tab_for_log_default"] = "📊 Dashboard"

    selected_file = st.sidebar.selectbox(
        "Select a dataset",
        options=csv_files_sorted,
        key="dataset_select",
    )

    st.sidebar.caption(f"Latest: **{latest_file}**")

    df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
# Ensure df is only processed if it contains data
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
    orders_file: str = "draw_orders.csv",
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
    st.markdown("### 📋 Orders Table")

    show_cols = [
        "Status",
        "Priority",
        "Fiber Project",
        "Preform Number",
        "Desired Date",
        "Length (m)",
        "Spools",
        "Timestamp",
        "Notes",
        "Done CSV",
        "Done Description",
    ]
    show_cols = [c for c in show_cols if c in df_visible.columns]
    table_df = df_visible[show_cols].copy()

    table_df = table_df.sort_values(
        by=["Desired Date", "Timestamp"],
        ascending=[True, False],
    )

    def _status_style(val):
        bg = {
            "Pending": "#ffb020",
            "Scheduled": "#2d7ff9",
            "Done": "#2ecc71",
            "Failed": "#ff3b30",
            "In progress": "#b77bff",
        }.get(str(val), "#333")
        return f"background-color:{bg};color:black;font-weight:900;"

    styled = table_df.style
    if "Status" in table_df.columns:
        styled = styled.applymap(_status_style, subset=["Status"])

    st.dataframe(styled, use_container_width=True, height=height)
# ================== PROCESS SETUP (Coating + Iris + PID/TF) ==================
def render_create_draw_dataset_csv():
    st.subheader("🆕 Create New Draw Dataset CSV")

    csv_name1 = st.text_input(
        "Enter Unique CSV Name For Drawing Data Set Creation",
        "",
        key="create_draw_csv_name_input",
    )

    csv_name = csv_name1 + ".csv" if csv_name1 and not csv_name1.endswith(".csv") else csv_name1

    if st.button("Create New CSV for Data Program", key="create_draw_csv_btn"):
        if not csv_name:
            st.warning("Please enter a valid name for the CSV file.")
            return

        if not os.path.exists("data_set_csv"):
            os.makedirs("data_set_csv")

        csv_path = os.path.join("data_set_csv", csv_name)

        if os.path.exists(csv_path):
            st.warning(f"CSV file '{csv_name1}' already exists.")
            return

        columns = ["Parameter Name", "Value", "Units"]
        df_new = pd.DataFrame(columns=columns)

        new_rows = [
            {"Parameter Name": "Draw Name", "Value": csv_name1, "Units": "N/A"},
            {"Parameter Name": "Draw Date", "Value": pd.Timestamp.now(), "Units": "N/A"},
        ]

        df_new = pd.concat([df_new, pd.DataFrame(new_rows)], ignore_index=True)
        df_new.to_csv(csv_path, index=False)
        st.success(f"New CSV '{csv_name}' created in the 'data_set_csv' folder!")

        new_draw_entry = pd.DataFrame([{
            "Timestamp": pd.Timestamp.now(),
            "Type": "Draw History",
            "Draw Name": csv_name1,
            "First Coating": "N/A",
            "First Coating Temperature": "N/A",
            "First Coating Die Size": "N/A",
            "Second Coating": "N/A",
            "Second Coating Temperature": "N/A",
            "Second Coating Die Size": "N/A",
            "Fiber Diameter": "N/A"
        }])

        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
        else:
            history_df = pd.DataFrame(columns=[
                "Timestamp", "Type", "Draw Name",
                "First Coating", "First Coating Temperature", "First Coating Die Size",
                "Second Coating", "Second Coating Temperature", "Second Coating Die Size",
                "Fiber Diameter"
            ])

        history_df = pd.concat([history_df, new_draw_entry], ignore_index=True)
        history_df.to_csv(HISTORY_FILE, index=False)
        st.success(f"Draw history for {csv_name} added successfully!")
def _process_setup_buttons() -> str:
    """Returns which section to show: 'all' | 'coating' | 'iris' | 'pid'."""
    if "process_setup_view" not in st.session_state:
        st.session_state["process_setup_view"] = "all"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🧴 Coating", use_container_width=True):
            st.session_state["process_setup_view"] = "coating"
            st.rerun()
    with c2:
        if st.button("🔍 Iris", use_container_width=True):
            st.session_state["process_setup_view"] = "iris"
            st.rerun()
    with c3:
        if st.button("🎛 PID & TF", use_container_width=True):
            st.session_state["process_setup_view"] = "pid"
            st.rerun()
    with c4:
        if st.button("✅ All", use_container_width=True):
            st.session_state["process_setup_view"] = "all"
            st.rerun()

    return st.session_state["process_setup_view"]
def render_iris_selection_section():
    import os
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.title("🔍 Iris Selection")
    st.subheader("Iris Selection Tool")

    # ----------------------------
    # Inputs
    # ----------------------------
    st.markdown("### 📐 Preform geometry")

    is_octagonal = st.checkbox("✅ Octagonal preform (use F2F instead of circular diameter)", value=False, key="iris_octagonal_flag")

    if is_octagonal:
        f2f_mm = st.number_input("Enter Octagon Flat-to-Flat (F2F) (mm)", min_value=0.0, step=0.1, format="%.2f", key="iris_f2f_mm")
        preform_diameter = f2f_mm  # keep old variable name for compatibility
    else:
        preform_diameter = st.number_input("Enter Preform Diameter (mm)", min_value=0.0, step=0.1, format="%.2f", key="iris_circ_diam_mm")

    st.markdown("### 🐯 Tiger cut")
    tiger_cut = st.checkbox("Is it a Tiger?", value=False, key="iris_tiger_flag")
    cut_percentage = 0
    if tiger_cut:
        cut_percentage = st.number_input("Enter Cut Percentage (%)", min_value=0, max_value=100, value=20, step=1, key="iris_cut_pct")

    # ----------------------------
    # Geometry helpers
    # ----------------------------
    def area_circle_from_diameter(d_mm: float) -> float:
        return np.pi * (d_mm / 2.0) ** 2

    def area_octagon_from_f2f(d_f2f_mm: float) -> float:
        # A = 2(√2 - 1) d^2  (regular octagon from flat-to-flat)
        return 2.0 * (np.sqrt(2.0) - 1.0) * (d_f2f_mm ** 2)

    def apply_tiger_cut(area_mm2: float, cut_pct: float) -> float:
        return area_mm2 * (1.0 - cut_pct / 100.0)

    def effective_diameter_from_area(area_mm2: float) -> float:
        # equal-area round rod diameter
        return 2.0 * np.sqrt(area_mm2 / np.pi)

    # ----------------------------
    # Calculate areas + effective diameter
    # ----------------------------
    if preform_diameter <= 0:
        st.warning("Please enter a valid value (> 0).")
        return

    if is_octagonal:
        base_area = area_octagon_from_f2f(preform_diameter)
        st.write(f"**Octagonal Area (from F2F {preform_diameter:.2f} mm):** {base_area:.2f} mm²")
    else:
        base_area = area_circle_from_diameter(preform_diameter)
        st.write(f"**Circular Area (from D {preform_diameter:.2f} mm):** {base_area:.2f} mm²")

    adjusted_area = apply_tiger_cut(base_area, cut_percentage)
    st.write(f"**Adjusted Area:** {adjusted_area:.2f} mm²")

    effective_diameter = effective_diameter_from_area(adjusted_area)
    st.write(f"**Effective Diameter (equal-area round):** {effective_diameter:.2f} mm")

    # ----------------------------
    # Iris selection
    # ----------------------------
    iris_diameters = [round(x * 0.5, 1) for x in range(20, 91)]  # 10..45 step 0.5
    valid_iris = [d for d in iris_diameters if d > effective_diameter]

    if not valid_iris:
        st.warning("No iris diameter is larger than the effective preform diameter.")
        return

    # Choose best gap closest to 200 mm^2
    results = [(d, (np.pi / 4.0) * (d**2 - effective_diameter**2)) for d in valid_iris]
    best = min(results, key=lambda x: abs(x[1] - 200.0))

    selected_iris = st.selectbox("Select Iris Diameter", valid_iris, index=valid_iris.index(best[0]), key="iris_selected_diam")
    gap_area = (np.pi / 4.0) * (selected_iris**2 - effective_diameter**2)
    st.write(f"**Gap Area:** {gap_area:.2f} mm²")

    # ----------------------------
    # Save to CSV
    # ----------------------------
    st.markdown("### 💾 Save to dataset CSV")

    recent_csv_files = [f for f in os.listdir("data_set_csv") if f.endswith(".csv")]
    if not recent_csv_files:
        st.info("No CSV files found in data_set_csv/")
        return

    selected_csv = st.selectbox("Select CSV to Update", recent_csv_files, key="iris_select_csv_update")

    if st.button("Update Dataset CSV", key="iris_update_dataset_csv"):
        tiger_cut_value = cut_percentage if tiger_cut else 0
        oct_flag = 1 if is_octagonal else 0

        data_to_add = [
            {"Parameter Name": "Preform Diameter", "Value": ("" if is_octagonal else preform_diameter), "Units": "mm"},
            {"Parameter Name": "Octagonal Preform", "Value": oct_flag, "Units": "bool"},
            {"Parameter Name": "Octagonal F2F", "Value": (preform_diameter if is_octagonal else ""), "Units": "mm"},
            {"Parameter Name": "Octagonal Area", "Value": (base_area if is_octagonal else ""), "Units": "mm^2"},
            {"Parameter Name": "Tiger Cut", "Value": tiger_cut_value, "Units": "%"},
            {"Parameter Name": "Adjusted Area", "Value": adjusted_area, "Units": "mm^2"},
            {"Parameter Name": "Effective Preform Diameter", "Value": effective_diameter, "Units": "mm"},
            {"Parameter Name": "Selected Iris Diameter", "Value": selected_iris, "Units": "mm"},
            {"Parameter Name": "Gap Area", "Value": gap_area, "Units": "mm^2"},
        ]

        csv_path = os.path.join("data_set_csv", selected_csv)
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame(data_to_add)], ignore_index=True)
        df.to_csv(csv_path, index=False)

        st.success(f"CSV '{selected_csv}' updated (octagon/tiger + effective diameter saved).")
def render_coating_section():
    st.subheader("🧴 Coating")
    st.title("💧 Coating Calculation")

    # **User Input Section**
    st.subheader("Input Parameters")

    # Viscosity Fitting Parameters for Primary Coating
    # Viscosity function is now sourced from config_coating.json; UI inputs removed.

    # Viscosity Fitting Parameters for Secondary Coating
    # Viscosity function is now sourced from config_coating.json; UI inputs removed.
    entry_fiber_diameter = st.number_input("Entry Fiber Diameter (µm)", min_value=0.0, step=0.1, format="%.1f")
    if "primary_temperature" not in st.session_state:
        st.session_state.primary_temperature = 25.0
    if "secondary_temperature" not in st.session_state:
        st.session_state.secondary_temperature = 25.0

    primary_temperature = st.number_input("Primary Coating Temperature (°C)",
                                          value=st.session_state.primary_temperature, step=0.1,
                                          key="primary_temperature")
    secondary_temperature = st.number_input("Secondary Coating Temperature (°C)",
                                            value=st.session_state.secondary_temperature, step=0.1,
                                            key="secondary_temperature")
    # Removed st.rerun() to allow live updates of temperature values

    dies = config.get("dies")
    coatings = config.get("coatings")
    if not dies or not coatings:
        st.error("Dies and/or Coatings not configured in config.json")
        st.stop()

    # **Dropdowns for Die and Coating Selection**
    primary_die = st.selectbox("Select Primary Die", dies.keys())
    secondary_die = st.selectbox("Select Secondary Die", dies.keys())

    primary_coating = st.selectbox("Select Primary Coating", coatings.keys())
    secondary_coating = st.selectbox("Select Secondary Coating", coatings.keys())
    first_entry_die = st.number_input("First Coating Entry Die (µm)", min_value=0.0, step=0.1)
    second_entry_die = st.number_input("Second Coating Entry Die (µm)", min_value=0.0, step=0.1)

    # **Load Selected Die and Coating Data**
    primary_die_config = dies[primary_die]
    secondary_die_config = dies[secondary_die]
    primary_coating_config = coatings[primary_coating]
    secondary_coating_config = coatings[secondary_coating]

    # **Extract necessary parameters for calculations**
    try:
        primary_density = primary_coating_config.get("Density", None)
        primary_neck_length = primary_die_config.get("Neck_Length", 0.002)
        primary_die_diameter = primary_die_config["Die_Diameter"]

        secondary_density = secondary_coating_config.get("Density", None)
        secondary_neck_length = secondary_die_config.get("Neck_Length", 0.002)
        secondary_die_diameter = secondary_die_config["Die_Diameter"]

        primary_viscosity_function = primary_coating_config.get("viscosity_fit_params", {}).get("function", "T**0.5")
        secondary_viscosity_function = secondary_coating_config.get("viscosity_fit_params", {}).get("function",
                                                                                                    "T**0.5")

        primary_viscosity = evaluate_viscosity(primary_temperature, primary_viscosity_function)
        secondary_viscosity = evaluate_viscosity(secondary_temperature, secondary_viscosity_function)

        # Ensure no missing parameters
        if None in [primary_viscosity, primary_density, secondary_viscosity, secondary_density]:
            st.error("Viscosity values could not be computed. Please check the configuration file.")
            st.stop()



    except KeyError as e:
        st.error(f"Missing key in configuration: {e}")
        st.stop()

    # **Constants**
    V = 0.917  # Pulling speed (m/s)
    g = 9.8  # Gravity (m/s²)

    # Recalculate viscosity based on the updated temperature
    primary_viscosity = evaluate_viscosity(primary_temperature, primary_viscosity_function)
    secondary_viscosity = evaluate_viscosity(secondary_temperature, secondary_viscosity_function)

    # Compute coating thickness for Primary and Secondary coatings
    FC_diameter = calculate_coating_thickness(
        entry_fiber_diameter,
        primary_die_diameter,
        primary_viscosity,  # Ensure dynamically updated viscosity is used
        primary_density,
        primary_neck_length,
        V, g
    )

    SC_diameter = calculate_coating_thickness(
        FC_diameter,
        secondary_die_diameter,
        secondary_viscosity,  # Updated viscosity based on temperature
        secondary_density,
        secondary_neck_length,
        V, g
    )

    # **Display Computed Coating Dimensions**
    st.write("### Coating Dimensions")
    st.write(f"**Fiber Diameter:** {entry_fiber_diameter:.1f} µm")
    st.write(f"**First Coating Diameter:** {FC_diameter:.1f} µm - Using Die coat {primary_die} & {primary_coating}")
    st.write(
        f"**Second Coating Diameter:** {SC_diameter:.1f} µm - Using Die coat {secondary_die} & {secondary_coating}")

    st.subheader("Coating Info")
    st.write("---")

    # Organize coating info layout
    coating_col1, coating_col2 = st.columns([1, 2])

    with coating_col1:
        selected_coating_info = st.selectbox("Select Coating to View Details", list(coatings.keys()),
                                             key="coating_info_select")

    with coating_col2:
        if selected_coating_info:
            coating_info = coatings[selected_coating_info]

            # Styling for a better look
            st.markdown(
                f"""
                    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #ffffff; color: #000000;">
                        <h3 style="color: #4CAF50;">Coating Name: {selected_coating_info}</h3>
                        <p><b>Viscosity:</b> {coating_info.get('Viscosity', 'N/A')} Pa·s</p>
                        <p><b>Density:</b> {coating_info.get('Density', 'N/A')} kg/m³</p>
                        <p><b>Description:</b> {coating_info.get('Description', 'No description available')}</p>
                    </div>
                    """,
                unsafe_allow_html=True
            )
    recent_csv_files = [f for f in os.listdir('data_set_csv') if f.endswith(".csv")]
    selected_csv = st.selectbox("Select CSV to Update", recent_csv_files, key="coating_select_csv_update")
    if st.button("Update Dataset CSV", key="coating_update_dataset_csv"):
        if selected_csv:
            st.write(f"Selected CSV: {selected_csv}")
            # Use calculated die diameters from the coating calculation
            primary_die_main_diameter = primary_die_diameter
            secondary_die_main_diameter = secondary_die_diameter

            data_to_add = [
                {"Parameter Name": "Entry Fiber Diameter", "Value": entry_fiber_diameter, "Units": "µm"},
                {"Parameter Name": "First Coating Diameter (Theoretical)", "Value": FC_diameter, "Units": "µm"},
                {"Parameter Name": "Second Coating Diameter (Theoretical)", "Value": SC_diameter, "Units": "µm"},
                {"Parameter Name": "Primary Coating", "Value": primary_coating, "Units": ""},
                {"Parameter Name": "Secondary Coating", "Value": secondary_coating, "Units": ""},
                {"Parameter Name": "First Coating Entry Die", "Value": first_entry_die, "Units": "µm"},
                {"Parameter Name": "Second Coating Entry Die", "Value": second_entry_die, "Units": "µm"},
                {"Parameter Name": "Primary Coating Temperature", "Value": primary_temperature, "Units": "°C"},
                {"Parameter Name": "Secondary Coating Temperature", "Value": secondary_temperature, "Units": "°C"},
                {"Parameter Name": "Primary Die Diameter", "Value": primary_die_main_diameter, "Units": "µm"},
                {"Parameter Name": "Secondary Die Diameter", "Value": secondary_die_main_diameter, "Units": "µm"},
            ]
            csv_path = os.path.join('data_set_csv', selected_csv)
            try:
                df_csv = pd.read_csv(csv_path)
            except FileNotFoundError:
                st.error(f"CSV file '{selected_csv}' not found.")
                st.stop()
            new_rows = pd.DataFrame(data_to_add)
            df_csv = pd.concat([df_csv, new_rows], ignore_index=True)
            df_csv.to_csv(csv_path, index=False)
            st.success(f"CSV '{selected_csv}' updated with new data!")
def render_pid_tf_section():
    st.subheader("🎛 PID & TF")

    st.title("🔧 PID and TF Configuration")

    # Load previous configuration if exists
    pid_config_path = "pid_config.json"
    if os.path.exists(pid_config_path):
        with open(pid_config_path, "r") as f:
            pid_config = json.load(f)
    else:
        pid_config = {}

    st.subheader("PID and TF Settings")

    # Input fields for P Gain and I Gain
    p_gain = st.number_input("P Gain (Diameter Control)", min_value=0.0, step=0.1, value=pid_config.get("p_gain", 1.0))
    i_gain = st.number_input("I Gain (Diameter Control)", min_value=0.0, step=0.1, value=pid_config.get("i_gain", 1.0))
    # Winder Configuration
    winder_mode = st.selectbox("TF Mode", ["Winder", "Straight Mode"],
                               index=["Winder", "Straight Mode"].index(pid_config.get("winder_mode", "Winder")))

    # Increment Value for Winder
    increment_value = st.number_input("Increment Value [mm]", min_value=0.0, step=0.1,
                                      value=pid_config.get("increment_value", 0.5))

    # Select CSV to save the configuration
    selected_csv = st.selectbox("Select CSV to Save PID and TF Configuration",
                                [f for f in os.listdir('data_set_csv') if f.endswith('.csv')])

    if st.button("💾 Save PID and TF Configuration"):
        if not selected_csv:
            st.error("Please select a CSV file before saving.")
        else:
            pid_config["p_gain"] = p_gain
            pid_config["i_gain"] = i_gain
            pid_config["winder_mode"] = winder_mode
            pid_config["increment_value"] = increment_value

            # Save the PID configuration to a JSON file
            with open(pid_config_path, "w") as f:
                json.dump(pid_config, f, indent=4)

            st.success(f"PID and TF configuration saved to '{selected_csv}'!")

            # Save to the chosen CSV
            csv_path = os.path.join('data_set_csv', selected_csv)
            df_csv = pd.read_csv(csv_path)

            new_data = [
                {"Parameter Name": "P Gain (Diameter Control)", "Value": p_gain, "Units": ""},
                {"Parameter Name": "I Gain (Diameter Control)", "Value": i_gain, "Units": ""},
                {"Parameter Name": "TF Mode", "Value": winder_mode, "Units": ""},
                {"Parameter Name": "Increment TF Value", "Value": increment_value, "Units": "mm"}
            ]
            new_df = pd.DataFrame(new_data)
            df_csv = pd.concat([df_csv, new_df], ignore_index=True)

            # Save back to CSV
            df_csv.to_csv(csv_path, index=False)
            # st.success(f"PID and TF configuration saved to '{selected_csv}'!")
def render_schedule_home_minimal():
    st.subheader("📅 Schedule")

    SCHEDULE_FILE = "tower_schedule.csv"
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

    # Clean Description/Recurrence strings (safe, but the REAL fix is hovertemplate=None below)
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
                out_rows.append(r.to_dict())
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
                out_rows.append(r.to_dict())
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

    fig = px.timeline(
        filtered,
        x_start="Start DateTime",
        x_end="End DateTime",
        y="Event Type",
        color="Event Type",
        color_discrete_map=event_colors,
        title="Tower Schedule",
        hover_data={
            "Description": True,
            "Recurrence": True,
            "Start DateTime": True,
            "End DateTime": True,
        },
    )

    # ✅ This is the key line: kills the leaked %{customdata[0]} hovertemplate
    fig.update_traces(hovertemplate=None)

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

    ORDER_FILE = "part_orders.csv"

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

    CORR_METHOD = "spearman"          # robust
    TIME_WINDOW_SECONDS = 60          # corr point every 60s (if timestamp exists)
    ROW_WINDOW = 1500                 # if no timestamp -> window by rows
    MIN_POINTS_PER_WINDOW = 80
    MAX_NUMERIC_COLS = 28             # safety: pairs explode fast (28 => 378 pairs)
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
    st.caption(f"Showing pairs {a+1}–{b} of {len(pair_list)}")

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
                    f"{r.get('Component','')} — {r.get('Task','')}",
                    key=f"pre_{i}"
                )

        with colB:
            st.markdown("#### Post-Draw")
            if post.empty:
                st.info("No post-draw routine tasks found.")
            for i, r in post.iterrows():
                st.checkbox(
                    f"{r.get('Component','')} — {r.get('Task','')}",
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
    import pandas as pd
    import streamlit as st
    import datetime as dt

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
                y=[0]*len(df),
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
            mid = x0 + (x1 - x0)/2
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
):
    import streamlit as st
    import pandas as pd
    import datetime as dt
    import time
    import numpy as np

    if not st.button("✅ Apply 'Done Now' updates", type="primary"):
        return

    done_rows = edited[edited["Done_Now"] == True].copy()
    if done_rows.empty:
        st.info("No tasks selected.")
        return

    updated = 0
    problems = []

    for src, grp in done_rows.groupby("Source_File"):
        path = os.path.join(MAINT_FOLDER, src)
        try:
            raw = read_file(path)
            df_src = normalize_df(raw)
            df_src["Tracking_Mode_norm"] = df_src["Tracking_Mode"].apply(mode_norm)

            for _, r in grp.iterrows():
                mode = mode_norm(r["Tracking_Mode"])
                mask = (
                    df_src["Component"].astype(str).eq(str(r["Component"])) &
                    df_src["Task"].astype(str).eq(str(r["Task"]))
                )

                if not mask.any():
                    continue

                if mode == "hours":
                    df_src.loc[mask, "Last_Done_Hours"] = pick_current_hours(r["Hours_Source"])
                elif mode == "draws":
                    df_src.loc[mask, "Last_Done_Draw"] = int(current_draw_count)

                df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()
                updated += int(mask.sum())

            out = templateize_df(df_src, list(raw.columns))
            write_file(path, out)

        except Exception as e:
            problems.append((src, str(e)))

    st.success(f"Updated {updated} task(s).")

    # ---- Log to DuckDB ----
    now = dt.datetime.combine(current_date, dt.datetime.now().time())
    actions = []
    for _, r in done_rows.iterrows():
        actions.append({
            "action_id": int(time.time() * 1000),
            "action_ts": now,
            "component": r["Component"],
            "task": r["Task"],
            "task_id": str(r.get("Task_ID", "")),
            "tracking_mode": r["Tracking_Mode"],
            "hours_source": r.get("Hours_Source", ""),
            "done_date": current_date,
            "done_hours": None,
            "done_draw": None,
            "source_file": r["Source_File"],
            "actor": actor,
            "note": ""
        })

    if actions:
        df_act = pd.DataFrame(actions)
        con.register("tmp_actions", df_act)
        con.execute("INSERT INTO maintenance_actions SELECT * FROM tmp_actions")
        con.unregister("tmp_actions")

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
        s = f"{r.get('Source_File','')}|{r.get('Task_ID','')}|{r.get('Component','')}|{r.get('Task','')}"
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

def load_maintenance_files(MAINT_FOLDER):
    import os
    import numpy as np
    import pandas as pd
    import streamlit as st

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
                index=(components.index(row.get("Component")) if components and row.get("Component") in components else 0),
                key=f"edit_comp_{fault_id}"
            ) if components else st.text_input("Component / Part", value=str(row.get("Component", "")), key=f"edit_comp_txt_{fault_id}")
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

        area = st.text_input("Subsystem/Area", value=str(row.get("Subsystem/Area", "") or ""), key=f"edit_area_{fault_id}")
        title = st.text_input("Title", value=str(row.get("Title", "") or ""), key=f"edit_title_{fault_id}")
        desc = st.text_area("Description", value=str(row.get("Description", "") or ""), key=f"edit_desc_{fault_id}", height=140)

        c4, c5 = st.columns(2)
        with c4:
            immediate = st.text_area("Immediate Action", value=str(row.get("Immediate Action", "") or ""), key=f"edit_im_{fault_id}", height=90)
            owner = st.text_input("Owner", value=str(row.get("Owner", "") or ""), key=f"edit_owner_{fault_id}")
        with c5:
            root = st.text_area("Root Cause", value=str(row.get("Root Cause", "") or ""), key=f"edit_root_{fault_id}", height=90)
            related_draw = st.text_input("Related Draw", value=str(row.get("Related Draw", "") or ""), key=f"edit_draw_{fault_id}")

        corr = st.text_area("Corrective Action", value=str(row.get("Corrective Action", "") or ""), key=f"edit_corr_{fault_id}", height=80)
        prev = st.text_area("Preventive Action", value=str(row.get("Preventive Action", "") or ""), key=f"edit_prev_{fault_id}", height=80)
        links = st.text_input("Attachments/Links", value=str(row.get("Attachments/Links", "") or ""), key=f"edit_links_{fault_id}")
        notes = st.text_area("Notes", value=str(row.get("Notes", "") or ""), key=f"edit_notes_{fault_id}", height=80)

        closed_date = str(row.get("Closed Date", "") or "")
        if status == "Closed" and not closed_date.strip():
            closed_date = dt.date.today().isoformat()
        if status != "Closed":
            closed_date = ""

        colX, colY = st.columns([1, 1])
        with colX:
            save_btn = st.button("💾 Save changes", type="primary", use_container_width=True, key=f"save_fault_{fault_id}")
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

DATASET_DIR = "data_set_csv"
PID_CONFIG_PATH = "pid_config.json"


# ============================================================
# Dataset helpers
# ============================================================
def _ensure_dataset_dir():
    os.makedirs(DATASET_DIR, exist_ok=True)

def _list_dataset_csvs():
    _ensure_dataset_dir()
    return sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")])

def _most_recent_csv():
    _ensure_dataset_dir()
    files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
    if not files:
        return None
    latest_path = max(files, key=os.path.getmtime)
    return os.path.basename(latest_path)

def _append_rows_to_dataset_csv(selected_csv: str, rows: list[dict]):
    if not selected_csv:
        st.error("No CSV selected.")
        return

    _ensure_dataset_dir()
    csv_path = os.path.join(DATASET_DIR, selected_csv)

    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Parameter Name", "Value", "Units"]).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    for col in ["Parameter Name", "Value", "Units"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    df = df[["Parameter Name", "Value", "Units"]]
    new_rows = pd.DataFrame(rows)

    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(csv_path, index=False)


DATASET_DIR = "data_set_csv"
PID_CONFIG_PATH = "pid_config.json"
ORDERS_FILE = "draw_orders.csv"
PREFORMS_FILE = "preforms_inventory.csv"

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
def get_most_recent_dataset_csv(dataset_dir="data_set_csv"):
    if not os.path.exists(dataset_dir):
        return None
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    if not files:
        return None
    return os.path.basename(max(files, key=os.path.getmtime))
def mark_draw_order_done_by_dataset_csv(dataset_csv_filename: str):
    """
    Your draw_orders.csv schema uses:
      - Active CSV
      - Done CSV
      - Status
      - Done Description
    This will:
      1) Find row where Active CSV == dataset filename (preferred)
         else row where Done CSV == dataset filename (fallback)
      2) Set Status = "Done"
      3) Set Done CSV = dataset filename
      4) Set Done Description = "Saved from Dashboard"
      5) Set T&M Moved = True and timestamp (if columns exist)
    """
    if not os.path.exists(ORDERS_FILE):
        return False, f"{ORDERS_FILE} not found (couldn't mark order done)."

    orders = pd.read_csv(ORDERS_FILE)

    # Ensure columns exist (create if missing)
    for col in ["Status", "Active CSV", "Done CSV", "Done Description", "T&M Moved", "T&M Moved Timestamp"]:
        if col not in orders.columns:
            orders[col] = ""

    # Match row
    active = orders["Active CSV"].astype(str).str.strip()
    done = orders["Done CSV"].astype(str).str.strip()

    match = (active == dataset_csv_filename)
    if not match.any():
        match = (done == dataset_csv_filename)

    if not match.any():
        return False, f"No matching row found in draw_orders.csv for '{dataset_csv_filename}' (Active CSV / Done CSV)."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    orders.loc[match, "Status"] = "Done"
    orders.loc[match, "Done CSV"] = dataset_csv_filename
    orders.loc[match, "Done Description"] = done_description

    # Optional fields
    if "T&M Moved" in orders.columns:
        orders.loc[match, "T&M Moved"] = True
    if "T&M Moved Timestamp" in orders.columns:
        orders.loc[match, "T&M Moved Timestamp"] = now_str

    orders.to_csv(ORDERS_FILE, index=False)
    return True, "Order marked as Done (matched by Active CSV)."
def _ensure_dataset_dir():
    os.makedirs(DATASET_DIR, exist_ok=True)
def _list_dataset_csvs():
    _ensure_dataset_dir()
    return sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")])
def _most_recent_csv():
    _ensure_dataset_dir()
    files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
    if not files:
        return None
    latest_path = max(files, key=os.path.getmtime)
    return os.path.basename(latest_path)
def _append_rows_to_dataset_csv(selected_csv: str, rows: list[dict]):
    if not selected_csv:
        st.error("No CSV selected.")
        return

    _ensure_dataset_dir()
    csv_path = os.path.join(DATASET_DIR, selected_csv)

    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Parameter Name", "Value", "Units"]).to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    for col in ["Parameter Name", "Value", "Units"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    df = df[["Parameter Name", "Value", "Units"]]
    new_rows = pd.DataFrame(rows)

    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(csv_path, index=False)
# ============================================================
# IRIS (no CSV save inside)
# ============================================================
def render_iris_selection_section_collect():
    import numpy as np
    import streamlit as st

    st.subheader("🔍 Iris Selection Tool")
    iris_data = {}

    # ============================
    # Helpers
    # ============================
    def area_circle_from_diameter(d_mm: float) -> float:
        return np.pi * (d_mm / 2.0) ** 2

    def area_octagon_from_f2f(d_f2f_mm: float) -> float:
        # A = 2(√2 − 1) d²
        return 2.0 * (np.sqrt(2.0) - 1.0) * (d_f2f_mm ** 2)

    def apply_tiger_cut(area_mm2: float, cut_pct: float) -> float:
        return area_mm2 * (1.0 - cut_pct / 100.0)

    def effective_diameter_from_area(area_mm2: float) -> float:
        # Equal-area round rod
        return 2.0 * np.sqrt(area_mm2 / np.pi)

    def gap_area(iris_mm: float, eff_mm: float) -> float:
        return (np.pi / 4.0) * (iris_mm**2 - eff_mm**2)

    # ============================
    # INPUTS
    # ============================
    st.markdown("### 🧾 Inputs")

    preform_value_mm = st.number_input(
        "Preform Diameter / F2F (mm)",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        help="Used as circular diameter or octagonal F2F depending on options.",
        key="iris_preform_value_collect"
    )

    # ============================
    # OPTIONS
    # ============================
    with st.expander("⚙️ Options", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            is_octagonal = st.toggle(
                "📐 Octagonal preform (interpret value as F2F)",
                value=False,
                key="iris_octagonal_flag_collect"
            )

            tiger_cut = st.checkbox(
                "🐯 Tiger cut",
                value=False,
                key="iris_tiger_flag_collect"
            )

            cut_percentage = 0
            if tiger_cut:
                cut_percentage = st.number_input(
                    "Cut percentage (%)",
                    min_value=0,
                    max_value=100,
                    value=20,
                    step=1,
                    key="iris_cut_pct_collect"
                )

        with col2:
            pm_iris_system = st.checkbox(
                "🧲 PM iris system (auto iris = 37 mm)",
                value=False,
                key="iris_pm_system_collect"
            )

    if preform_value_mm <= 0:
        st.info("Enter a preform value to calculate.")
        return iris_data

    # ============================
    # GEOMETRY + AREA
    # ============================
    base_shape = "Octagonal (F2F)" if is_octagonal else "Circular"
    shape_label = f"{base_shape} + Tiger cut" if tiger_cut else base_shape

    if is_octagonal:
        base_area = area_octagon_from_f2f(preform_value_mm)
        base_label = "Octagonal Area"
    else:
        base_area = area_circle_from_diameter(preform_value_mm)
        base_label = "Circular Area"

    adjusted_area = apply_tiger_cut(base_area, cut_percentage)
    effective_diameter = effective_diameter_from_area(adjusted_area)

    # ============================
    # RESULTS
    # ============================
    st.markdown("### 📊 Geometry results")

    m1, m2, m3 = st.columns(3)
    m1.metric(base_label, f"{base_area:.2f} mm²")
    m2.metric("Adjusted Area", f"{adjusted_area:.2f} mm²")
    m3.metric("Effective Diameter", f"{effective_diameter:.2f} mm")

    st.caption(
        f"Shape: **{shape_label}** | "
        f"Tiger: **{cut_percentage if tiger_cut else 0}%** | "
        f"PM iris: **{'Yes (37 mm)' if pm_iris_system else 'No'}**"
    )

    # ============================
    # IRIS SELECTION
    # ============================
    st.markdown("### 🎯 Iris selection")

    if pm_iris_system:
        selected_iris = 37.0
        iris_mode = "PM Auto 37"
        st.success("PM iris enabled → Iris fixed to **37.0 mm**")
    else:
        iris_mode = "Manual"
        iris_diameters = [round(x * 0.5, 1) for x in range(20, 91)]
        valid_iris = [d for d in iris_diameters if d > effective_diameter]

        if not valid_iris:
            st.warning("No iris diameter is larger than the effective preform diameter.")
            return iris_data

        results = [(d, gap_area(d, effective_diameter)) for d in valid_iris]
        best = min(results, key=lambda x: abs(x[1] - 200.0))

        selected_iris = st.selectbox(
            "Select iris diameter (mm)",
            valid_iris,
            index=valid_iris.index(best[0]),
            key="iris_selected_diam_collect"
        )

    gap = max(0.0, gap_area(selected_iris, effective_diameter))

    g1, g2 = st.columns(2)
    g1.metric("Selected Iris", f"{selected_iris:.1f} mm")
    g2.metric("Gap Area", f"{gap:.2f} mm²")

    # ============================
    # RETURN DATA (FOR CSV SAVE)
    # ============================
    iris_data = {
        "Preform Input Value (mm)": float(preform_value_mm),
        "Preform Shape": shape_label,
        "Is Octagonal": bool(is_octagonal),

        "Tiger Preform": bool(tiger_cut),
        "Tiger Cut (%)": int(cut_percentage) if tiger_cut else 0,

        "Octagonal F2F (mm)": float(preform_value_mm) if is_octagonal else None,
        "Circular Diameter (mm)": float(preform_value_mm) if not is_octagonal else None,

        "PM Iris System": bool(pm_iris_system),
        "Iris Mode": iris_mode,

        "Base Area (mm^2)": float(base_area),
        "Adjusted Area (mm^2)": float(adjusted_area),
        "Effective Preform Diameter (mm)": float(effective_diameter),

        "Selected Iris Diameter (mm)": float(selected_iris),
        "Gap Area (mm^2)": float(gap),
    }

    return iris_data
# ============================================================
# COATING (no CSV save inside)
# requires: evaluate_viscosity(), calculate_coating_thickness()
# ============================================================
def render_coating_section_collect(config: dict) -> dict:
    st.subheader("🧴 Coating Calculation")

    dies = config.get("dies")
    coatings = config.get("coatings")
    if not dies or not coatings:
        st.error("Dies and/or Coatings not configured in config.")
        return {}

    st.markdown("**Entry Fiber Diameter (µm)**")

    # Preset buttons
    preset_cols = st.columns(3)
    with preset_cols[0]:
        if st.button("40 µm", key="preset_fiber_40"):
            st.session_state.ps_coat_entry_fiber = 40.0
    with preset_cols[1]:
        if st.button("125 µm", key="preset_fiber_125"):
            st.session_state.ps_coat_entry_fiber = 125.0
    with preset_cols[2]:
        if st.button("400 µm", key="preset_fiber_400"):
            st.session_state.ps_coat_entry_fiber = 400.0

    # Ensure default exists
    if "ps_coat_entry_fiber" not in st.session_state:
        st.session_state.ps_coat_entry_fiber = 125.0  # sensible default

    entry_fiber_diameter = st.number_input(
        "",
        min_value=0.0,
        step=0.1,
        format="%.1f",
        key="ps_coat_entry_fiber"
    )

    primary_temperature = st.number_input(
        "Primary Coating Temperature (°C)",
        value=25.0, step=0.1,
        key="ps_coat_primary_temp"
    )
    secondary_temperature = st.number_input(
        "Secondary Coating Temperature (°C)",
        value=25.0, step=0.1,
        key="ps_coat_secondary_temp"
    )

    c1, c2 = st.columns(2)
    with c1:
        primary_die = st.selectbox("Select Primary Die", list(dies.keys()), key="ps_coat_primary_die")
        primary_coating = st.selectbox("Select Primary Coating", list(coatings.keys()), key="ps_coat_primary_coating")
        first_entry_die = st.number_input("First Coating Entry Die (µm)", min_value=0.0, step=0.1, key="ps_coat_first_entry_die")
    with c2:
        secondary_die = st.selectbox("Select Secondary Die", list(dies.keys()), key="ps_coat_secondary_die")
        secondary_coating = st.selectbox("Select Secondary Coating", list(coatings.keys()), key="ps_coat_secondary_coating")
        second_entry_die = st.number_input("Second Coating Entry Die (µm)", min_value=0.0, step=0.1, key="ps_coat_second_entry_die")

    primary_die_config = dies[primary_die]
    secondary_die_config = dies[secondary_die]
    primary_coating_config = coatings[primary_coating]
    secondary_coating_config = coatings[secondary_coating]

    try:
        primary_density = primary_coating_config.get("Density", None)
        secondary_density = secondary_coating_config.get("Density", None)

        primary_neck_length = primary_die_config.get("Neck_Length", 0.002)
        secondary_neck_length = secondary_die_config.get("Neck_Length", 0.002)

        primary_die_diameter = primary_die_config["Die_Diameter"]
        secondary_die_diameter = secondary_die_config["Die_Diameter"]

        primary_visc_fn = primary_coating_config.get("viscosity_fit_params", {}).get("function", "T**0.5")
        secondary_visc_fn = secondary_coating_config.get("viscosity_fit_params", {}).get("function", "T**0.5")

        primary_viscosity = evaluate_viscosity(primary_temperature, primary_visc_fn)
        secondary_viscosity = evaluate_viscosity(secondary_temperature, secondary_visc_fn)

        if None in [primary_viscosity, primary_density, secondary_viscosity, secondary_density]:
            st.error("Viscosity/Density missing or not computable. Check config.")
            return {}
    except KeyError as e:
        st.error(f"Missing key in configuration: {e}")
        return {}

    V = 0.917
    g = 9.8

    fc_diameter = calculate_coating_thickness(
        entry_fiber_diameter, primary_die_diameter, primary_viscosity,
        primary_density, primary_neck_length, V, g
    )
    sc_diameter = calculate_coating_thickness(
        fc_diameter, secondary_die_diameter, secondary_viscosity,
        secondary_density, secondary_neck_length, V, g
    )

    st.write("### Coating Dimensions")
    st.write(f"First Coating Diameter: **{fc_diameter:.1f} µm**")
    st.write(f"Second Coating Diameter: **{sc_diameter:.1f} µm**")

    return {
        "entry_fiber_diameter": float(entry_fiber_diameter),
        "primary_temperature": float(primary_temperature),
        "secondary_temperature": float(secondary_temperature),
        "primary_die": primary_die,
        "secondary_die": secondary_die,
        "primary_die_diameter": float(primary_die_diameter),
        "secondary_die_diameter": float(secondary_die_diameter),
        "primary_coating": primary_coating,
        "secondary_coating": secondary_coating,
        "first_entry_die": float(first_entry_die),
        "second_entry_die": float(second_entry_die),
        "fc_diameter": float(fc_diameter),
        "sc_diameter": float(sc_diameter),
    }
# ============================================================
# PID/TF (no CSV save inside)
# ============================================================
def render_pid_tf_section_collect() -> dict:
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
# ============================================================
# FINAL SAVE (one click)
# ============================================================
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
def render_save_all_block(iris_data: dict, coating_data: dict, pid_data: dict, drum_data: dict):
    import pandas as pd
    import streamlit as st
    from datetime import datetime

    st.subheader("💾 Save Everything (one click)")

    _ensure_dataset_dir()
    csv_files = _list_dataset_csvs()
    latest = _most_recent_csv()

    # Build rows (start with timestamp)
    rows = [{
        "Parameter Name": "Process Setup Timestamp",
        "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Units": ""
    }]

    # ✅ Keep the same UI button, but DON'T save yet (Streamlit rerun issue)
    save_to_latest_clicked = st.button("⚡ Save ALL to MOST RECENT CSV", key="ps_saveall_to_latest")

    # =========================
    # IRIS rows (UPDATED KEYS)
    # =========================
    if iris_data:
        preform_val = iris_data.get("Preform Input Value (mm)")
        tiger_pct   = iris_data.get("Tiger Cut (%)", 0)
        eff_d       = iris_data.get("Effective Preform Diameter (mm)")
        sel_iris    = iris_data.get("Selected Iris Diameter (mm)")
        gap_area    = iris_data.get("Gap Area (mm^2)")

        is_oct      = iris_data.get("Is Octagonal", False)
        shape_lbl   = iris_data.get("Preform Shape", "")
        oct_f2f     = iris_data.get("Octagonal F2F (mm)")
        pm_sys      = iris_data.get("PM Iris System", False)
        iris_mode   = iris_data.get("Iris Mode", "")
        base_area   = iris_data.get("Base Area (mm^2)")
        adj_area    = iris_data.get("Adjusted Area (mm^2)")
        tiger_flag  = iris_data.get("Tiger Preform", False)
        circ_d      = iris_data.get("Circular Diameter (mm)")

        rows += [
            {"Parameter Name": "Preform Diameter", "Value": circ_d if circ_d is not None else "", "Units": "mm"},
            {"Parameter Name": "Preform Shape", "Value": shape_lbl, "Units": ""},

            {"Parameter Name": "Octagonal Preform", "Value": 1 if is_oct else 0, "Units": "bool"},
            {"Parameter Name": "Octagonal F2F", "Value": oct_f2f if oct_f2f is not None else "", "Units": "mm"},

            {"Parameter Name": "Tiger Preform", "Value": 1 if tiger_flag else 0, "Units": "bool"},
            {"Parameter Name": "Tiger Cut", "Value": tiger_pct, "Units": "%"},

            {"Parameter Name": "PM Iris System", "Value": 1 if pm_sys else 0, "Units": "bool"},
            {"Parameter Name": "Iris Mode", "Value": iris_mode, "Units": ""},

            {"Parameter Name": "Base Area", "Value": base_area, "Units": "mm^2"},
            {"Parameter Name": "Adjusted Area", "Value": adj_area, "Units": "mm^2"},
            {"Parameter Name": "Effective Preform Diameter", "Value": eff_d, "Units": "mm"},

            {"Parameter Name": "Selected Iris Diameter", "Value": sel_iris, "Units": "mm"},
            {"Parameter Name": "Iris Gap Area", "Value": gap_area, "Units": "mm^2"},
        ]

    # =========================
    # Coating rows (unchanged)
    # =========================
    if coating_data:
        rows += [
            {"Parameter Name": "Entry Fiber Diameter", "Value": coating_data.get("entry_fiber_diameter"), "Units": "µm"},
            {"Parameter Name": "First Coating Diameter (Theoretical)", "Value": coating_data.get("fc_diameter"), "Units": "µm"},
            {"Parameter Name": "Second Coating Diameter (Theoretical)", "Value": coating_data.get("sc_diameter"), "Units": "µm"},
            {"Parameter Name": "Primary Coating", "Value": coating_data.get("primary_coating"), "Units": ""},
            {"Parameter Name": "Secondary Coating", "Value": coating_data.get("secondary_coating"), "Units": ""},
            {"Parameter Name": "First Coating Entry Die", "Value": coating_data.get("first_entry_die"), "Units": "µm"},
            {"Parameter Name": "Second Coating Entry Die", "Value": coating_data.get("second_entry_die"), "Units": "µm"},
            {"Parameter Name": "Primary Coating Temperature", "Value": coating_data.get("primary_temperature"), "Units": "°C"},
            {"Parameter Name": "Secondary Coating Temperature", "Value": coating_data.get("secondary_temperature"), "Units": "°C"},
            {"Parameter Name": "Primary Die Diameter", "Value": coating_data.get("primary_die_diameter"), "Units": "µm"},
            {"Parameter Name": "Secondary Die Diameter", "Value": coating_data.get("secondary_die_diameter"), "Units": "µm"},
        ]

    # =========================
    # PID rows (unchanged)
    # =========================
    if pid_data:
        rows += [
            {"Parameter Name": "P Gain (Diameter Control)", "Value": pid_data.get("p_gain"), "Units": ""},
            {"Parameter Name": "I Gain (Diameter Control)", "Value": pid_data.get("i_gain"), "Units": ""},
            {"Parameter Name": "TF Mode", "Value": pid_data.get("winder_mode"), "Units": ""},
            {"Parameter Name": "Increment TF Value", "Value": pid_data.get("increment_value"), "Units": "mm"},
        ]

    # =========================
    # DRUM rows (new)
    # =========================
    if drum_data:
        rows += [
            {"Parameter Name": "Selected Drum", "Value": drum_data.get("Selected Drum"), "Units": ""}
        ]

    # Filter out None only (keep "" if you intentionally set blank)
    rows = [r for r in rows if r.get("Value") is not None]

    # UI (unchanged)
    colA, colB = st.columns([2, 1])
    with colA:
        selected_csv = st.selectbox(
            "Or choose a CSV from the list",
            options=[""] + csv_files,
            index=0,
            key="ps_saveall_selected_csv"
        )
    with colB:
        st.write("")
        st.caption(f"Most recent: **{latest if latest else 'None'}**")

    with st.expander("Preview what will be written", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ✅ Now saving happens AFTER rows are fully built
    if save_to_latest_clicked:
        if not latest:
            st.error("No CSV files found in data_set_csv/")
        else:
            _append_rows_to_dataset_csv(latest, rows)
            st.success(f"Saved ALL to most recent: {latest}")

    if st.button("💾 Save ALL to SELECTED CSV", key="ps_saveall_to_selected"):
        if not selected_csv:
            st.error("Pick a CSV first.")
        else:
            _append_rows_to_dataset_csv(selected_csv, rows)
            st.success(f"Saved ALL to: {selected_csv}")
# ============================================================
# TAB RENDER (matches your structure)
# ============================================================
def render_scheduled_quick_start(
    orders_df: pd.DataFrame,
    orders_file: str,
    data_set_dir: str = "data_set_csv",
    scheduled_status: str = "Scheduled",
    in_progress_status: str = "In Progress",
    key_prefix: str = "sched_qs",
):
    st.markdown("#### 🗓️ Scheduled quick start (Create CSV + set In Progress)")

    os.makedirs(data_set_dir, exist_ok=True)

    if orders_df is None or orders_df.empty:
        st.caption("No orders found.")
        return

    df = orders_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Ensure required columns exist
    required = ["Status", "Preform Number", "Fiber Project", "Desired Date", "Priority", "Active CSV"]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    # Filter scheduled
    df["Status"] = df["Status"].astype(str).str.strip()
    df_sched = df[df["Status"].str.lower() == str(scheduled_status).lower()].copy()

    if df_sched.empty:
        st.caption("No Scheduled orders right now.")
        return

    def _next_run_index(preform_num: str) -> int:
        preform_num = str(preform_num).strip()
        prefix = f"F{preform_num}_"
        existing_files = [
            f for f in os.listdir(data_set_dir)
            if f.startswith(prefix) and f.lower().endswith(".csv")
        ]
        runs = []
        for f in existing_files:
            try:
                tail = f[len(prefix):]
                n_str = tail.replace(".csv", "")
                runs.append(int(n_str))
            except Exception:
                pass
        return (max(runs) + 1) if runs else 1

    # Render each scheduled row (using original index)
    for idx in df_sched.index:
        fiber = str(df.at[idx, "Fiber Project"]).strip()
        preform = str(df.at[idx, "Preform Number"]).strip()
        dd = str(df.at[idx, "Desired Date"]).strip()
        prio = str(df.at[idx, "Priority"]).strip()

        header = f"#{idx} | {fiber} | Priority: {prio} | Preform: {preform} | {dd}".strip()
        with st.expander(header, expanded=False):

            if not preform:
                st.warning("This order has no Preform Number. Please fill it first.")
                continue

            first_work = st.checkbox(
                "First work on this preform? (try _1)",
                value=True,
                key=f"{key_prefix}_first_work_{idx}",
            )

            if first_work:
                run_idx = 1
                cand = os.path.join(data_set_dir, f"F{preform}_1.csv")
                if os.path.exists(cand):
                    run_idx = _next_run_index(preform)
                    st.warning(f"_1 already exists. I will create _{run_idx} instead.")
            else:
                run_idx = _next_run_index(preform)

            new_csv_name = f"F{preform}_{run_idx}.csv"
            csv_path = os.path.join(data_set_dir, new_csv_name)

            st.caption(f"CSV to be created: **{new_csv_name}**")

            if st.button("▶ Start (Create CSV + set In Progress)", key=f"{key_prefix}_start_{idx}"):
                if os.path.exists(csv_path):
                    st.error(f"CSV already exists: {new_csv_name}")
                    st.stop()

                # Create dataset CSV
                base_cols = ["Parameter Name", "Value", "Units"]
                df_new = pd.DataFrame(columns=base_cols)

                new_rows = [
                    {"Parameter Name": "Draw Name", "Value": new_csv_name.replace(".csv", ""), "Units": ""},
                    {"Parameter Name": "Draw Date", "Value": pd.Timestamp.now(), "Units": ""},
                    {"Parameter Name": "Order Index", "Value": idx, "Units": ""},
                    {"Parameter Name": "Preform Number", "Value": preform, "Units": ""},
                    {"Parameter Name": "Fiber Project", "Value": fiber, "Units": ""},
                ]

                df_new = pd.concat([df_new, pd.DataFrame(new_rows)], ignore_index=True)
                df_new.to_csv(csv_path, index=False)

                # Update order row
                df.at[idx, "Status"] = in_progress_status
                df.at[idx, "Active CSV"] = new_csv_name

                df.to_csv(orders_file, index=False)

                st.success(f"✅ Created {new_csv_name} and moved order #{idx} to {in_progress_status}.")
                st.rerun()

    st.markdown("---")
def render_process_setup_tab(config: dict):
    st.title("⚙️ Process Setup")
    st.caption("One-page setup for every draw: Create CSV + coating + iris + PID/TF — then save everything together")
    orders_file = "draw_orders.csv"
    if os.path.exists(orders_file):
        df_orders = pd.read_csv(orders_file, keep_default_na=False)
    else:
        df_orders = pd.DataFrame()

    render_scheduled_quick_start(
        orders_df=df_orders,
        orders_file=orders_file,
        data_set_dir="data_set_csv",
        key_prefix="process_setup_schedqs",
    )
    # Your existing creator
    render_create_draw_dataset_csv()

    st.markdown("---")
    view = _process_setup_buttons()  # expects "all", "coating", "iris", "pid"

    iris_data, coating_data, pid_data, drum_data = {}, {}, {}, {}

    if view in ("all", "coating"):
        st.markdown("---")
        coating_data = render_coating_section_collect(config)

    if view in ("all", "iris"):
        st.markdown("---")
        iris_data = render_iris_selection_section_collect()

    if view in ("all", "pid"):
        st.markdown("---")
        pid_data = render_pid_tf_section_collect()
    st.markdown("---")
    drum_data = render_drum_selection_section_collect()
    # Always show final save block
    st.markdown("---")
    render_save_all_block(iris_data, coating_data, pid_data, drum_data)
def _most_recent_dataset_csv(dataset_dir="data_set_csv"):
    if not os.path.exists(dataset_dir):
        return None
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    if not files:
        return None
    full = [os.path.join(dataset_dir, f) for f in files]
    return os.path.basename(max(full, key=os.path.getmtime))
def _read_dataset_csv(dataset_dir="data_set_csv", filename=None):
    if filename is None:
        filename = _most_recent_dataset_csv(dataset_dir)
    if not filename:
        return None, None
    path = os.path.join(dataset_dir, filename)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["Parameter Name", "Value", "Units"]:
        if c not in df.columns:
            df[c] = ""
    return df, filename
def _param_map(df_params: pd.DataFrame) -> dict:
    d = {}
    for _, r in df_params.iterrows():
        k = str(r.get("Parameter Name", "")).strip()
        if not k:
            continue
        d[k] = r.get("Value", "")
    return d
def _parse_steps(df_params: pd.DataFrame):
    d = _param_map(df_params)
    steps = []
    i = 1
    while True:
        ak = f"STEP {i} Action"
        lk = f"STEP {i} Length"
        if ak not in d or lk not in d:
            break
        action = str(d.get(ak, "")).strip().upper()
        try:
            length = float(d.get(lk))
        except Exception:
            length = None
        if action in ("SAVE", "CUT") and length is not None and length > 0:
            steps.append((action, float(length)))
        i += 1
    return steps
def _parse_zones_from_end(df_params: pd.DataFrame):
    """
    Reads:
      Zone i Start (from end) [km]
      Zone i End (from end)   [km]
    Returns list sorted by start:
      [{"i":1,"a":..,"b":..,"len":..}, ...]
    """
    d = _param_map(df_params)
    zones = []
    i = 1
    while True:
        ks = f"Zone {i} Start (from end)"
        ke = f"Zone {i} End (from end)"
        if ks not in d or ke not in d:
            break
        try:
            a = float(d[ks]); b = float(d[ke])
        except Exception:
            a = None; b = None
        if a is not None and b is not None:
            if b < a:
                a, b = b, a
            zones.append({"i": i, "a": a, "b": b, "len": (b - a)})
        i += 1
    zones.sort(key=lambda z: z["a"])
    return zones
def _parse_marked_zone_lengths(df_params: pd.DataFrame):
    """
    Reads:
      Marked Zone i Length [km]
    Returns list of lengths in order i=1..N
    """
    d = _param_map(df_params)
    out = []
    i = 1
    while True:
        k = f"Marked Zone {i} Length"
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
def _get_float(df_params: pd.DataFrame, name: str, default=0.0) -> float:
    hit = df_params.loc[df_params["Parameter Name"].astype(str) == name, "Value"]
    if hit.empty:
        return float(default)
    try:
        return float(hit.iloc[-1])
    except Exception:
        return float(default)
def render_tm_drum_fiber_visual_from_csv(df_params: pd.DataFrame, dataset_name: str):
    """
    Draws drum + fiber + zones/segments.
    Uses ONLY what exists in the dataset CSV (no log needed).
    Priority:
      1) Zone i Start/End (from end)  -> shows ALL zones accurately
      2) Marked Zone i Length         -> shows ALL zones sequentially
      3) STEP i Action/Length         -> fallback merged plan
    """
    total_km = _get_float(df_params, "Fiber Total Length (Log End)", 0.0)
    total_save = _get_float(df_params, "Total Saved Length", 0.0)
    total_cut  = _get_float(df_params, "Total Cut Length", 0.0)

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
        st.info("No zone information found in dataset CSV (no Zone-from-end, no Marked Zone Lengths, no STEP SAVE segments).")
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
                x=0.5*(x0p+x1p),
                y=0.58,
                text=f"Zone {z['i']}  {z['len']:.3f} km",
                showarrow=False
            )

        fig.add_trace(go.Scatter(
            x=[0.5*(x0p+x1p)],
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
    c3.metric("Total CUT (km)",  f"{total_cut:.6f}" if total_cut else "—")

    st.plotly_chart(fig, use_container_width=True)
def render_tm_home_section():
    import os
    import re
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    from datetime import datetime

    st.subheader("📦 T&M – Pending Transfer")
    st.caption("Draws completed but not yet transferred to T&M")

    ORDERS_FILE = "draw_orders.csv"
    DATASET_DIR = "data_set_csv"

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

    def _read_dataset_kv(csv_path: str) -> dict:
        """Reads dataset CSV (Parameter Name, Value, Units) -> {param_lower: value_str}"""
        try:
            dfx = pd.read_csv(csv_path, keep_default_na=False)
        except Exception:
            return {}

        if dfx is None or dfx.empty:
            return {}

        cols = {c.strip(): c for c in dfx.columns}
        pn_col = "Parameter Name" if "Parameter Name" in cols else None
        v_col = "Value" if "Value" in cols else None
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

    # -----------------------------
    # Plot helpers (use ONLY dataset CSV content)
    # -----------------------------
    def _param_map(df_params: pd.DataFrame) -> dict:
        d = {}
        for _, r in df_params.iterrows():
            k = str(r.get("Parameter Name", "")).strip()
            if not k:
                continue
            d[k] = r.get("Value", "")
        return d

    def _get_float(d: dict, key: str, default=0.0) -> float:
        try:
            v = d.get(key, default)
            return float(v) if v != "" else float(default)
        except Exception:
            return float(default)

    def _parse_zones_from_end(df_params: pd.DataFrame):
        d = _param_map(df_params)
        zones = []
        i = 1
        while True:
            ks = f"Zone {i} Start (from end)"
            ke = f"Zone {i} End (from end)"
            if ks not in d or ke not in d:
                break
            try:
                a = float(d[ks]); b = float(d[ke])
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
        d = _param_map(df_params)
        out = []
        i = 1
        while True:
            k = f"Marked Zone {i} Length"
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

    def _parse_steps(df_params: pd.DataFrame):
        d = _param_map(df_params)
        steps = []
        i = 1
        while True:
            ak = f"STEP {i} Action"
            lk = f"STEP {i} Length"
            if ak not in d or lk not in d:
                break
            action = str(d.get(ak, "")).strip().upper()
            try:
                length = float(d.get(lk))
            except Exception:
                length = None
            if action in ("SAVE", "CUT") and length is not None and length > 0:
                steps.append((action, float(length)))
            i += 1
        return steps

    def render_tm_drum_fiber_visual_from_csv(df_params: pd.DataFrame, csv_name: str):
        """
        Drum + fiber bar:
          - 0 km = fiber end at RIGHT side
          - drum at LEFT side (near total length)
          - CUT + SAVE segments shown across full fiber
          - NO hover
          - Each segment is labeled with the step number (#) that matches the table

        Uses ONLY dataset CSV.

        Priority for defining SAVE zones:
          1) Zone i Start/End (from end)
          2) Marked Zone i Length (sequential from 0)
          3) STEP SAVE segments (fallback)
        """
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        import streamlit as st

        d = _param_map(df_params)

        total_km = _get_float(d, "Fiber Total Length (Log End)", 0.0)
        total_save = _get_float(d, "Total Saved Length", 0.0)
        total_cut = _get_float(d, "Total Cut Length", 0.0)

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
            mask = p.str.contains(fr"^Good Zone {zone_i} Avg - ", regex=True, na=False) & p.str.contains(contains_name,
                                                                                                         na=False)
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
        # UI: plot + table
        # =========================
        st.markdown("**🧵 Fiber Map + Cut Plan (# matches the table)**")
        st.caption(f"Dataset: `{csv_name}`  |  0 km = fiber end (right)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total (km)", f"{total_km:.4f}")
        m2.metric("SAVE (km)", f"{total_save:.4f}" if total_save else "—")
        m3.metric("CUT (km)", f"{total_cut:.4f}" if total_cut else "—")

        left, right = st.columns([2.2, 1.3], vertical_alignment="top")

        with left:
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### 📋 Cut / Save Steps")
            show_cols = [c for c in ["#", "Action",  "Length (km)", "Start (km from end)", "End (km from end)",
                                     "Furnace Temp avg", "Tension avg", "Fiber Ø avg","Fiber inner coat Ø avg","Fiber Outer coat Ø avg"] if c in df_plan.columns]
            df_show = df_plan[show_cols].copy()

            st.dataframe(
                df_show,
                use_container_width=True,
                hide_index=True,
                height=260
            )
            st.caption("Numbers on the bar correspond to the # column here.")

    # -----------------------------
    # Cards
    # -----------------------------
    for idx, row in pending_tm.iterrows():
        draw_id = row.get("Preform Name") or row.get("Preform Number") or f"Row {idx}"
        done_csv = str(row.get("Done CSV") or "").strip()
        active_csv = str(row.get("Active CSV") or "").strip()

        csv_name = done_csv if done_csv else active_csv
        csv_path = os.path.join(DATASET_DIR, csv_name) if csv_name else ""

        done_desc = str(row.get("Done Description") or "").strip()

        kv = _read_dataset_kv(csv_path) if (csv_path and os.path.exists(csv_path)) else {}

        project = _pick(kv, ["Project", "Project Name", "Fiber Project", "Fiber name and number", "Fiber Name and Number"]) \
                  or str(row.get("Project Name") or "").strip()

        preform = _pick(kv, ["Preform Number", "Preform Name", "Preform", "Draw Name"]) \
                  or str(row.get("Preform Name") or row.get("Preform Number") or "").strip()

        fiber = _pick(kv, ["Draw Name"]) or str(row.get("Fiber Type") or row.get("Fiber Project") or "").strip()

        drum = _pick(kv, ["Drum", "Selected Drum"])

        project_disp = project if project else "—"
        preform_disp = preform if preform else "—"
        fiber_disp = fiber if fiber else "—"
        drum_disp = drum if drum else "—"
        csv_disp = csv_name if csv_name else "—"

        st.markdown("<div class='tm-card'>", unsafe_allow_html=True)
        st.markdown(f"**🧾 Draw:** {draw_id}", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='tm-meta'>
                Project: <b>{project_disp}</b>
                &nbsp; | &nbsp; Preform: <b>{preform_disp}</b>
                &nbsp; | &nbsp; Fiber: <b>{fiber_disp}</b>
                &nbsp; | &nbsp; Drum: <b>{drum_disp}</b>
                <br/>
                CSV: <code>{csv_disp}</code>
            </div>
            """,
            unsafe_allow_html=True
        )

        if done_desc:
            st.caption(f"Done notes: {done_desc}")

        # ✅ PUT THE VISUAL RIGHT HERE (inside each card)
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
def _most_recent_dataset_csv(dataset_dir="data_set_csv"):
    if not os.path.exists(dataset_dir):
        return None
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    if not files:
        return None
    full = [os.path.join(dataset_dir, f) for f in files]
    return os.path.basename(max(full, key=os.path.getmtime))
def _read_dataset_csv(dataset_dir="data_set_csv", filename=None):
    if filename is None:
        filename = _most_recent_dataset_csv(dataset_dir)
    if not filename:
        return None, None
    path = os.path.join(dataset_dir, filename)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, keep_default_na=False)
    # normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["Parameter Name", "Value", "Units"]:
        if c not in df.columns:
            df[c] = ""
    return df, filename
def _param_map(df_params: pd.DataFrame) -> dict:
    # If duplicates exist, keep the last occurrence
    d = {}
    for _, r in df_params.iterrows():
        k = str(r.get("Parameter Name", "")).strip()
        if not k:
            continue
        d[k] = r.get("Value", "")
    return d
def _parse_steps(df_params: pd.DataFrame):
    d = _param_map(df_params)
    steps = []
    i = 1
    while True:
        ak = f"STEP {i} Action"
        lk = f"STEP {i} Length"
        if ak not in d or lk not in d:
            break
        action = str(d.get(ak, "")).strip().upper()
        try:
            length = float(d.get(lk))
        except Exception:
            length = None
        if action in ("SAVE", "CUT") and length is not None and length > 0:
            steps.append((action, float(length)))
        i += 1
    return steps
def _get_value(df_params: pd.DataFrame, name: str, default=None):
    hit = df_params.loc[df_params["Parameter Name"].astype(str) == name, "Value"]
    if hit.empty:
        return default
    return hit.iloc[-1]
def _find_zone_avg_values(df_params: pd.DataFrame, wanted_cols: list):
    """
    Returns dict: {col_name: avg_across_zones}
    Looks for rows like: 'Good Zone {i} Avg - {col}'
    and averages across all found zones.
    """
    out = {}
    pnames = df_params["Parameter Name"].astype(str)

    for col in wanted_cols:
        # match: Good Zone 1 Avg - Tension   (case-insensitive contains)
        mask = pnames.str.contains(r"^Good Zone \d+ Avg - ", regex=True, na=False) & pnames.str.contains(re.escape(col), na=False)
        vals = pd.to_numeric(df_params.loc[mask, "Value"], errors="coerce").dropna()
        if not vals.empty:
            out[col] = float(vals.mean())
    return out
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
        total_len = float(_get_value(df_params, "Fiber Total Length (Log End)", 0.0) or 0.0)
    except Exception:
        total_len = 0.0
    try:
        total_save = float(_get_value(df_params, "Total Saved Length", 0.0) or 0.0)
    except Exception:
        total_save = 0.0
    try:
        total_cut = float(_get_value(df_params, "Total Cut Length", 0.0) or 0.0)
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
        st.metric("Total CUT (km)",  f"{total_cut:.6f}" if total_cut else "—")

        # Show key averages as small list
        if avg_map:
            st.markdown("**Key averages (good zones)**")
            for k, v in avg_map.items():
                st.write(f"- {k}: `{v:.4g}`")
# ------------------ Home Tab ------------------
if tab_selection == "🏠 Home":
    st.title("️ Tower Management Software")

    st.subheader("Isorad Tower Management Software")

    # ---------- CSS FIRST (so it affects everything below) ----------
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .css-1aumxhk {{ background-color: rgba(20, 20, 20, 0.90) !important; }}
        .css-1l02zno {{
            color: #FFFFFF !important;
            font-size: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .css-1d391kg, .css-qrbaxs, .css-1y4p8pa {{
            color: #FFFFFF !important;
            font-size: 18px;
            font-weight: 700;
        }}
        .css-1y4p8pa[aria-selected="true"] {{
            color: #FFD700 !important;
            font-weight: bold;
        }}
        .css-1y4p8pa:hover {{ color: #B0C4DE !important; }}
        h1 {{
            color: #FFFFFF !important;
            font-size: 38px !important;
            font-weight: bold !important;
            text-align: center;
            margin-top: 20px;
        }}
        h2 {{
            color: #DDDDDD !important;
            font-size: 24px !important;
            font-style: italic;
            text-align: center;
            margin-top: -10px;
        }}
        @media (prefers-color-scheme: light) {{
            h1 {{ color: #000000 !important; }}
            h2 {{ color: #333333 !important; }}
            .css-1l02zno {{ color: #000000 !important; }}
            .css-1d391kg, .css-qrbaxs, .css-1y4p8pa {{ color: #000000 !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------- 1) SCHEDULE FIRST ----------
    render_schedule_home_minimal()

    st.markdown("---")
    render_home_draw_orders_overview()
    render_tm_home_section()  # <-- use the correct function name you have

    st.markdown("---")
    # ---------- 2) MAINTENANCE OVERVIEW ----------
    st.subheader("🧰 Maintenance Overview")

    overdue = st.session_state.get("maint_overdue")
    due_soon = st.session_state.get("maint_due_soon")

    if overdue is None or due_soon is None:
        import datetime as dt
        base_dir = os.getcwd()
        overdue, due_soon = maintenance_quick_counts(
            base_dir=base_dir,
            current_date=dt.date.today(),
            furnace_hours=st.session_state.get("furnace_hours", 0.0),
            uv1_hours=st.session_state.get("uv1_hours", 0.0),
            uv2_hours=st.session_state.get("uv2_hours", 0.0),
            warn_days=14,
            warn_hours=50.0,
        )

    c1, c2 = st.columns(2)
    c1.metric("🔴 Overdue", int(overdue))
    c2.metric("🟠 Due soon", int(due_soon))

    st.markdown("---")

    # ---------- 2) PARTS NEEDED ----------
    render_parts_orders_home_all()
    st.markdown("---")
    # ---------- 3) T&MMANAGEMENT ----------
# ------------------ Process Tab ------------------
elif tab_selection == "⚙️ Process Setup":
    render_process_setup_tab(config)
# ------------------ Dashboard Tab ------------------
elif tab_selection == "📊 Dashboard":
    st.title(f"📊 Draw Tower Logs Dashboard - {selected_file}")


    # -----------------------------
    # Helpers (minimal, local)
    # -----------------------------
    ORDERS_FILE = "draw_orders.csv"


    def _find_length_col(cols):
        cols_map = {str(c).strip().lower(): c for c in cols}

        exact = [
            "fiber length", "fibre length",
            "fiber_length", "fibre_length",
            "draw length", "line length", "spool length",
        ]
        for k in exact:
            if k in cols_map:
                return cols_map[k]

        for cl_lower, orig in cols_map.items():
            if ("length" in cl_lower) and (("fiber" in cl_lower) or ("fibre" in cl_lower)):
                return orig

        length_like = [orig for cl_lower, orig in cols_map.items() if "length" in cl_lower]
        if len(length_like) == 1:
            return length_like[0]

        return None


    def build_cut_save_steps_km(df_work, x_axis, good_zones, fiber_len_col):
        """
        Returns:
          total_km: float
          save_segments_km: [(a,b), ...]  # km from end, merged, sorted
          cut_segments_km:  [(a,b), ...]  # complement, sorted
          steps: [("CUT", length_km), ("SAVE", length_km), ...] sorted from end outward
        """
        import numpy as np
        import pandas as pd

        # ---- length unit -> km factor (best effort)
        name = str(fiber_len_col).lower()
        if "km" in name:
            to_km = 1.0
        elif "cm" in name:
            to_km = 1e-5
        else:
            to_km = 1e-3  # assume meters by default

        tmp = df_work[[x_axis, fiber_len_col]].copy()
        tmp = tmp.dropna(subset=[x_axis]).sort_values(by=x_axis)
        tmp[fiber_len_col] = pd.to_numeric(tmp[fiber_len_col], errors="coerce")
        tmp = tmp.dropna(subset=[fiber_len_col])

        if tmp.empty:
            return None

        L_end = float(tmp[fiber_len_col].iloc[-1])
        total_km = L_end * to_km

        # Interpolation x
        x_series = tmp[x_axis]
        if pd.api.types.is_datetime64_any_dtype(x_series):
            x_num = (x_series.view("int64") / 1e9).to_numpy(dtype=float)

            def x_to_num(v):
                return pd.to_datetime(v).value / 1e9
        else:
            x_num = pd.to_numeric(x_series, errors="coerce").to_numpy(dtype=float)

            def x_to_num(v):
                return float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])

        L_arr = tmp[fiber_len_col].to_numpy(dtype=float)

        ok = np.isfinite(x_num) & np.isfinite(L_arr)
        x_num = x_num[ok]
        L_arr = L_arr[ok]
        if len(x_num) < 2:
            return None

        # ---- zones -> SAVE segments in km from end
        save_segments = []
        for (zs, ze) in good_zones:
            xs = x_to_num(zs)
            xe = x_to_num(ze)
            if not (np.isfinite(xs) and np.isfinite(xe)):
                continue
            if xe < xs:
                xs, xe = xe, xs

            Ls = float(np.interp(xs, x_num, L_arr))
            Le = float(np.interp(xe, x_num, L_arr))
            if Le < Ls:
                Ls, Le = Le, Ls

            a = (L_end - Le) * to_km  # near end
            b = (L_end - Ls) * to_km  # farther from end

            a = max(0.0, min(total_km, a))
            b = max(0.0, min(total_km, b))
            if b - a <= 0:
                continue

            save_segments.append((a, b))

        # No save zones => all cut
        if not save_segments:
            cut_segments = [(0.0, total_km)]
            steps = [("CUT", total_km)]
            return total_km, [], cut_segments, steps

        # merge overlaps
        save_segments.sort(key=lambda t: t[0])
        merged = []
        for a, b in save_segments:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        save_segments = [(a, b) for a, b in merged]

        # complement -> cuts
        cut_segments = []
        cur = 0.0
        for a, b in save_segments:
            if a > cur:
                cut_segments.append((cur, a))
            cur = max(cur, b)
        if cur < total_km:
            cut_segments.append((cur, total_km))

        # build ordered steps from end outward by slicing [0,total] into alternating segments
        # We can just merge both lists into boundaries then label membership
        steps = []
        all_bounds = [0.0, total_km]
        for a, b in save_segments:
            all_bounds.extend([a, b])
        all_bounds = sorted(set([float(x) for x in all_bounds]))

        def in_save(mid):
            for a, b in save_segments:
                if a <= mid <= b:
                    return True
            return False

        for i in range(len(all_bounds) - 1):
            a = all_bounds[i]
            b = all_bounds[i + 1]
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            action = "SAVE" if in_save(mid) else "CUT"
            length = b - a
            # merge consecutive same-action steps
            if steps and steps[-1][0] == action:
                steps[-1] = (action, steps[-1][1] + length)
            else:
                steps.append((action, length))

        return total_km, save_segments, cut_segments, steps
    def get_most_recent_dataset_csv(dataset_dir="data_set_csv"):
        if not os.path.exists(dataset_dir):
            return None
        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
        if not files:
            return None
        return os.path.basename(max(files, key=os.path.getmtime))

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

    def mark_draw_order_done_by_dataset_csv(dataset_csv_filename: str, done_desc: str, preform_len_after_cm: float):
        if not os.path.exists(ORDERS_FILE):
            return False, f"{ORDERS_FILE} not found (couldn't mark order done)."

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
            "Preform Length After Draw (m)": "",
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

        orders.loc[match, "Status"] = "Done"
        orders.loc[match, "Done CSV"] = os.path.basename(dataset_csv_filename)
        orders.loc[match, "Done Description"] = str(done_desc).strip()
        orders.loc[match, "Preform Length After Draw (cm)"] = float(preform_len_after_cm)

        if "T&M Moved Timestamp" in orders.columns:
            orders.loc[match, "T&M Moved Timestamp"] = now_str

        orders.to_csv(ORDERS_FILE, index=False)
        return True, "Order marked as Done."

    # -----------------------------
    # Zones state
    # -----------------------------
    if "good_zones" not in st.session_state:
        st.session_state["good_zones"] = []

    # ==========================================================
    # Plot controls (can be empty WITHOUT stopping the tab)
    # ==========================================================
    column_options = df.columns.tolist()

    x_axis = st.selectbox("Select X-axis", column_options, key="x_axis_dash")
    y_axes = st.multiselect(
        "Select Y-axis column(s)",
        options=column_options,
        default=[],
        key="y_axes_dash_multi",
    )

    # Build df_work always (Math Lab uses fixed x too)
    df_work = df.copy()

    # Fix X dtype (datetime/numeric) BEFORE any plotting or math time axis
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

    # ==========================================================
    # MAIN PLOT AREA (only if user selected Y columns)
    # ==========================================================
    if not y_axes:
        st.info("Select one or more **Y-axis** columns to show the plot + zones. (Math Lab is below.)")
    else:
        filtered_df = df_work.dropna(subset=[x_axis] + y_axes).sort_values(by=x_axis)

        if filtered_df.empty:
            st.warning("No data to plot after filtering NA values for the selected X/Y columns.")
        else:
            # -----------------------------
            # Zone slider
            # -----------------------------
            st.subheader("🟩 Zone Marker")
            st.caption("Use the slider to pick a range, then click 'Add Selected Zone'.")
            time_range = None

            if pd.api.types.is_datetime64_any_dtype(filtered_df[x_axis]):
                time_min = filtered_df[x_axis].min().to_pydatetime()
                time_max = filtered_df[x_axis].max().to_pydatetime()
                time_range = st.slider(
                    "Select Time Range for Good Zone",
                    min_value=time_min,
                    max_value=time_max,
                    value=(time_min, time_max),
                    step=pd.Timedelta(seconds=1).to_pytimedelta(),
                    format="HH:mm:ss",
                    key="dash_zone_slider_dt",
                )
            elif pd.api.types.is_numeric_dtype(filtered_df[x_axis]):
                x_min = float(filtered_df[x_axis].min())
                x_max = float(filtered_df[x_axis].max())
                step = (x_max - x_min) / 1000.0 if x_max > x_min else 1.0
                if step <= 0:
                    step = 1.0
                time_range = st.slider(
                    f"Select Range for Good Zone ({x_axis})",
                    min_value=x_min,
                    max_value=x_max,
                    value=(x_min, x_max),
                    step=step,
                    key="dash_zone_slider_num",
                )
            else:
                i0, i1 = st.slider(
                    "Select Index Range for Good Zone",
                    min_value=0,
                    max_value=max(0, len(filtered_df) - 1),
                    value=(0, max(0, len(filtered_df) - 1)),
                    step=1,
                    key="dash_zone_slider_idx",
                )
                xs = filtered_df[x_axis].iloc[int(i0)]
                xe = filtered_df[x_axis].iloc[int(i1)]
                time_range = (xs, xe)

            # -----------------------------
            # Plot (wide feel) + legend under plot
            # -----------------------------
            st.subheader("📈 Plot")

            fig = go.Figure()
            for y_col in y_axes:
                fig.add_trace(go.Scatter(
                    x=filtered_df[x_axis],
                    y=filtered_df[y_col],
                    mode="lines+markers",
                    name=y_col,
                ))

            for start, end in st.session_state["good_zones"]:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="green", opacity=0.3, line_width=0,
                    annotation_text="Good Zone", annotation_position="top left"
                )

            if time_range:
                fig.add_vrect(
                    x0=time_range[0], x1=time_range[1],
                    fillcolor="blue", opacity=0.2, line_width=1,
                    line_dash="dot",
                    annotation_text="Selected", annotation_position="top right"
                )

            fig.update_layout(
                title=f"{' , '.join(y_axes)} vs {x_axis}",
                margin=dict(l=10, r=10, t=55, b=10),
                height=620,
                xaxis_title=x_axis,
                yaxis_title="Values",
                legend=dict(
                    title="Y columns",
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            if time_range and st.button("➕ Add Selected Zone", key="dash_add_zone"):
                st.session_state["good_zones"].append(time_range)
                st.success(f"Zone added: {time_range[0]} to {time_range[1]}")
                st.rerun()

            # -----------------------------
            # Summary (per selected y)
            # -----------------------------
            if st.session_state["good_zones"]:
                st.write("### ✅ Good Zones Summary")

                summary_rows = []
                combined = {y: [] for y in y_axes}

                for i, (start, end) in enumerate(st.session_state["good_zones"]):
                    zone_data = filtered_df[(filtered_df[x_axis] >= start) & (filtered_df[x_axis] <= end)]
                    if zone_data.empty:
                        continue
                    for y_col in y_axes:
                        vals = pd.to_numeric(zone_data[y_col], errors="coerce").dropna()
                        if vals.empty:
                            continue
                        summary_rows.append({
                            "Zone": f"Zone {i+1}",
                            "Y": y_col,
                            "Start": start,
                            "End": end,
                            "Avg": float(vals.mean()),
                            "Min": float(vals.min()),
                            "Max": float(vals.max()),
                        })
                        combined[y_col].extend(vals.values.tolist())

                if summary_rows:
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
                    st.markdown("#### 📊 Combined Stats (all zones)")
                    for y_col in y_axes:
                        vals = combined.get(y_col, [])
                        if vals:
                            s = pd.Series(vals, dtype=float)
                            st.write(f"**{y_col}**  |  Avg: {s.mean():.4f}   Min: {s.min():.4f}   Max: {s.max():.4f}")
                else:
                    st.info("No valid zone stats for the selected columns.")

            # ==========================================================
            # 💾 SAVE + MARK DONE
            # ==========================================================
            st.markdown("---")
            st.subheader("💾 Save to Dataset CSV + ✅ Mark Order Done")

            recent_csv_files = sorted([f for f in os.listdir("data_set_csv") if f.lower().endswith(".csv")]) if os.path.exists("data_set_csv") else []
            latest_csv = get_most_recent_dataset_csv("data_set_csv")
            st.caption(f"Most recent dataset CSV: **{latest_csv if latest_csv else 'None'}**")

            done_desc = st.text_area(
                "Done description (what happened / notes)",
                value=st.session_state.get("dash_done_desc", ""),
                key="dash_done_desc",
                height=90
            )
            preform_len_after_cm = st.number_input(
                "Preform length after draw (cm)",
                min_value=0.0,
                value=float(st.session_state.get("dash_preform_len_after_cm", 0.0)),
                step=0.5,
                format="%.1f",
                key="dash_preform_len_after_cm",
            )
            if float(preform_len_after_cm) <= 0:
                st.warning("Please enter **Preform length after draw (cm)** above to enable saving/marking done.")

            do_save_latest = st.button(
                "⚡ Save to MOST RECENT CSV & Mark Done",
                key="dash_save_latest_btn",
                disabled=(
                    (latest_csv is None)
                    or (not str(done_desc).strip())
                    or (float(preform_len_after_cm) <= 0)
                ),
                use_container_width=True
            )

            st.write("**Or choose a dataset CSV**")
            selected_csv = st.selectbox(
                "Choose a dataset CSV",
                options=[""] + recent_csv_files,
                index=0,
                key="dashboard_select_csv_update",
                label_visibility="collapsed"
            )

            do_save_selected = st.button(
                "💾 Save Zones Summary (selected CSV)",
                key="dash_save_selected_btn",
                disabled=(
                    (selected_csv == "")
                    or (not str(done_desc).strip())
                    or (float(preform_len_after_cm) <= 0)
                ),
                use_container_width=True
            )

            target_csv = None
            if do_save_latest:
                target_csv = latest_csv
            elif do_save_selected:
                target_csv = selected_csv

            if target_csv:
                if not str(done_desc).strip():
                    st.error("Please write Done description before saving + marking Done.")
                elif float(preform_len_after_cm) <= 0:
                    st.error("Please enter **Preform length after draw (m)** (must be > 0).")
                else:
                    csv_path = os.path.join("data_set_csv", target_csv)
                    try:
                        df_csv = pd.read_csv(csv_path, keep_default_na=False)
                    except FileNotFoundError:
                        st.error(f"CSV file '{target_csv}' not found.")
                        df_csv = None

                    if df_csv is not None:
                        import numpy as np
                        import pandas as pd
                        import streamlit as st

                        # -------------------------------------------------
                        # Ensure required columns
                        # -------------------------------------------------
                        for c in ["Parameter Name", "Value", "Units"]:
                            if c not in df_csv.columns:
                                df_csv[c] = ""

                        # -------------------------------------------------
                        # Start building rows
                        # -------------------------------------------------
                        data_to_add = [
                            {"Parameter Name": "Log File Name", "Value": selected_file, "Units": ""}
                        ]

                        # -------------------------------------------------
                        # Save zone boundaries (traceability)
                        # -------------------------------------------------
                        for i, (start, end) in enumerate(st.session_state["good_zones"], start=1):
                            data_to_add.extend([
                                {"Parameter Name": f"Zone {i} Start", "Value": start, "Units": ""},
                                {"Parameter Name": f"Zone {i} End", "Value": end, "Units": ""},
                            ])

                        # -------------------------------------------------
                        # ✅ Restore: per-zone stats for ALL numeric log columns
                        # (this is the “missing data” you complained about)
                        # -------------------------------------------------
                        try:
                            numeric_cols_all = filtered_df.select_dtypes(include="number").columns.tolist()
                            # remove x-axis if numeric
                            if x_axis in numeric_cols_all:
                                numeric_cols_all.remove(x_axis)
                            # remove obvious junk cols
                            for drop_col in ["index", "Index", "__index__", "Unnamed: 0"]:
                                if drop_col in numeric_cols_all:
                                    numeric_cols_all.remove(drop_col)

                            for zone_idx, (zs, ze) in enumerate(st.session_state["good_zones"], start=1):
                                zone_data = filtered_df[(filtered_df[x_axis] >= zs) & (filtered_df[x_axis] <= ze)]
                                if zone_data.empty:
                                    continue

                                for col in numeric_cols_all:
                                    vals = pd.to_numeric(zone_data[col], errors="coerce").dropna()
                                    if vals.empty:
                                        continue

                                    s = pd.Series(vals, dtype=float)
                                    data_to_add.extend([
                                        {"Parameter Name": f"Good Zone {zone_idx} Avg - {col}",
                                         "Value": float(s.mean()), "Units": ""},
                                        {"Parameter Name": f"Good Zone {zone_idx} Min - {col}", "Value": float(s.min()),
                                         "Units": ""},
                                        {"Parameter Name": f"Good Zone {zone_idx} Max - {col}", "Value": float(s.max()),
                                         "Units": ""},
                                    ])
                        except Exception as e:
                            st.warning(f"Zone stats were not saved due to an error: {e}")


                        # -------------------------------------------------
                        # Helper: find Fiber Length column (already KM)
                        # -------------------------------------------------
                        def _find_length_col(cols):
                            cols_map = {str(c).strip().lower(): c for c in cols}
                            exact = [
                                "fiber length", "fibre length",
                                "fiber_length", "fibre_length",
                                "draw length", "line length", "spool length",
                            ]
                            for k in exact:
                                if k in cols_map:
                                    return cols_map[k]
                            for cl, orig in cols_map.items():
                                if "length" in cl and ("fiber" in cl or "fibre" in cl):
                                    return orig
                            length_like = [orig for cl, orig in cols_map.items() if "length" in cl]
                            return length_like[0] if len(length_like) == 1 else None


                        fiber_len_col = _find_length_col(df_work.columns)

                        # Always store how many zones were marked (as-drawn)
                        data_to_add.append({
                            "Parameter Name": "Total Marked Good Zones",
                            "Value": int(len(st.session_state["good_zones"])),
                            "Units": "count"
                        })

                        if fiber_len_col is None:
                            st.warning("Fiber Length column not found – CUT/SAVE steps not written.")
                        else:
                            # -------------------------------------------------
                            # Build mapping x_axis -> Fiber Length (KM)
                            # -------------------------------------------------
                            tmp = df_work[[x_axis, fiber_len_col]].copy()
                            tmp = tmp.dropna(subset=[x_axis]).sort_values(by=x_axis)
                            tmp[fiber_len_col] = pd.to_numeric(tmp[fiber_len_col], errors="coerce")
                            tmp = tmp.dropna(subset=[fiber_len_col])

                            if tmp.empty or len(tmp) < 2:
                                st.warning("Not enough valid Fiber Length data to compute CUT/SAVE steps.")
                            else:
                                # Fiber length is ALREADY in KM
                                L_end_km = float(tmp[fiber_len_col].iloc[-1])
                                total_km = L_end_km

                                # numeric x for interpolation
                                if pd.api.types.is_datetime64_any_dtype(tmp[x_axis]):
                                    x_num = (tmp[x_axis].view("int64") / 1e9).to_numpy(float)


                                    def x_to_num(v):
                                        return pd.to_datetime(v).value / 1e9
                                else:
                                    x_num = pd.to_numeric(tmp[x_axis], errors="coerce").to_numpy(float)


                                    def x_to_num(v):
                                        return float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])

                                L_arr_km = tmp[fiber_len_col].to_numpy(float)

                                # -------------------------------------------------
                                # Zones -> SAVE segments in km from END (0=end)
                                # -------------------------------------------------
                                save_segments = []
                                for (zs, ze) in st.session_state["good_zones"]:
                                    xs, xe = x_to_num(zs), x_to_num(ze)
                                    if not (np.isfinite(xs) and np.isfinite(xe)):
                                        continue
                                    if xe < xs:
                                        xs, xe = xe, xs

                                    Ls_km = float(np.interp(xs, x_num, L_arr_km))
                                    Le_km = float(np.interp(xe, x_num, L_arr_km))
                                    if Le_km < Ls_km:
                                        Ls_km, Le_km = Le_km, Ls_km

                                    # From-end coordinates
                                    a = (L_end_km - Le_km)  # near end
                                    b = (L_end_km - Ls_km)  # farther from end
                                    a = max(0.0, min(total_km, a))
                                    b = max(0.0, min(total_km, b))
                                    if b > a:
                                        save_segments.append((a, b))

                                # merge overlaps
                                save_segments.sort()
                                merged = []
                                for a, b in save_segments:
                                    if not merged or a > merged[-1][1]:
                                        merged.append([a, b])
                                    else:
                                        merged[-1][1] = max(merged[-1][1], b)
                                save_segments = [(a, b) for a, b in merged]



                                # Complement -> CUT segments
                                cut_segments = []
                                cur = 0.0
                                for a, b in save_segments:
                                    if a > cur:
                                        cut_segments.append((cur, a))
                                    cur = max(cur, b)
                                if cur < total_km:
                                    cut_segments.append((cur, total_km))

                                # Build STEP list (length only)
                                steps = []
                                bounds = [0.0, total_km]
                                for a, b in save_segments:
                                    bounds.extend([a, b])
                                bounds = sorted(set(bounds))


                                def is_save(mid):
                                    return any(a <= mid <= b for a, b in save_segments)


                                for i in range(len(bounds) - 1):
                                    a, b = bounds[i], bounds[i + 1]
                                    if b <= a:
                                        continue
                                    action = "SAVE" if is_save(0.5 * (a + b)) else "CUT"
                                    length = b - a
                                    if steps and steps[-1][0] == action:
                                        steps[-1] = (action, steps[-1][1] + length)
                                    else:
                                        steps.append((action, length))

                                # Totals
                                total_save_km = sum(b - a for a, b in save_segments)
                                total_cut_km = sum(b - a for a, b in cut_segments)

                                data_to_add.extend([
                                    {"Parameter Name": "Fiber Total Length (Log End)", "Value": float(total_km),
                                     "Units": "km"},
                                    {"Parameter Name": "Total Saved Length", "Value": float(total_save_km),
                                     "Units": "km"},
                                    {"Parameter Name": "Total Cut Length", "Value": float(total_cut_km), "Units": "km"},
                                ])


                                # STEP i (Action + Length only)
                                for i, (action, length) in enumerate(steps, start=1):
                                    data_to_add.extend([
                                        {"Parameter Name": f"STEP {i} Action", "Value": action, "Units": ""},
                                        {"Parameter Name": f"STEP {i} Length", "Value": float(length), "Units": "km"},
                                    ])

                        # -------------------------------------------------
                        # Done metadata
                        # -------------------------------------------------
                        data_to_add.extend([
                            {"Parameter Name": "Preform Length After Draw", "Value": float(preform_len_after_cm),
                             "Units": "cm"},
                            {"Parameter Name": "Done Description", "Value": str(done_desc).strip(), "Units": ""},
                            {"Parameter Name": "Done Timestamp", "Value": pd.Timestamp.now(), "Units": ""},
                        ])

                        # -------------------------------------------------
                        # Write dataset CSV
                        # -------------------------------------------------
                        df_csv = pd.concat([df_csv, pd.DataFrame(data_to_add)], ignore_index=True)
                        df_csv.to_csv(csv_path, index=False)
                        st.success(f"CSV '{target_csv}' updated!")

                        ok, msg = mark_draw_order_done_by_dataset_csv(target_csv, done_desc,
                                                                      float(preform_len_after_cm))
                        if ok:
                            st.success(msg)
                            append_preform_length(
                                preform_name=df_csv.loc[df_csv["Parameter Name"] == "Preform Number", "Value"].iloc[0],
                                length_cm=float(preform_len_after_cm),
                                source_draw=target_csv.replace(".csv", "")
                            )
                        else:
                            st.warning(msg)

    # ==========================================================
    # 🧮 MATH LAB (single expander, NO nested expanders)
    # ==========================================================
    st.markdown("---")
    with st.expander("🧮 Math Lab (advanced)", expanded=False):

        # ---------- A) f(x,y) ----------
        st.subheader("A) f(x,y) vs time")

        math_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(math_numeric_cols) < 1:
            st.info("No numeric columns found in this log.")
        else:
            m1, m2, m3 = st.columns([1, 1, 2])

            with m1:
                math_x_col = st.selectbox("Math X column", math_numeric_cols, key="dash_math_end_x_col")

            with m2:
                math_y_col = st.selectbox("Math Y column (optional)", ["None"] + math_numeric_cols, key="dash_math_end_y_col")

            with m3:
                default_expr = "x ** y" if math_y_col != "None" else "x"
                math_expr = st.text_input(
                    "Expression (use x, y and np)",
                    value=st.session_state.get("dash_math_end_expr", default_expr),
                    key="dash_math_end_expr_input"
                )
                st.session_state["dash_math_end_expr"] = math_expr
                st.caption("Examples: `x**y`, `x*y`, `np.log(x)`, `np.sqrt(x+y)`, `(x-y)/x`")

            show_math_preview = st.checkbox("Show f(x,y) preview table", value=False, key="dash_math_preview_chk")

            math_df = df.copy()
            math_df[x_axis] = df_work[x_axis]  # fixed x-axis

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

                    if math_plot_df.empty:
                        st.warning("No data to plot after NA cleaning.")
                    else:
                        fig_math = px.line(
                            math_plot_df,
                            x=x_axis,
                            y="__math_result__",
                            markers=False,
                            title=f"Math Lab: f(x,y) vs {x_axis} | f={math_expr} | x={math_x_col}, y={math_y_col}"
                        )
                        st.plotly_chart(fig_math, use_container_width=True)

                        if show_math_preview:
                            show_cols = [x_axis, math_x_col]
                            if math_y_col != "None":
                                show_cols.append(math_y_col)
                            show_cols.append("__math_result__")
                            st.dataframe(math_plot_df[show_cols].head(300), use_container_width=True)

            except Exception as e:
                st.error(f"Math Lab error: {e}")

        st.markdown("---")

        # ---------- B) Derivative / Integral ----------
        st.subheader("B) Single Column Derivative / Integral")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 1:
            st.info("No numeric columns found in this log.")
        else:
            b1, b2, b3 = st.columns([1, 1, 2])

            with b1:
                base_col = st.selectbox("Column", numeric_cols, key="dash_single_base_col")

            with b2:
                smooth_win = st.number_input(
                    "Smoothing window (points, 1 = none)",
                    min_value=1, max_value=501, value=1, step=2,
                    key="dash_single_smooth"
                )

            with b3:
                show_derivative = st.checkbox("Show derivative d/dt", value=True, key="dash_single_show_deriv")
                show_integral = st.checkbox("Show integral ∫ y dt", value=False, key="dash_single_show_integ")

            show_single_preview = st.checkbox("Show derivative/integral preview table", value=False, key="dash_single_preview_chk")

            s_df = df[[base_col]].copy()
            s_df[x_axis] = df_work[x_axis]
            s_df[base_col] = pd.to_numeric(s_df[base_col], errors="coerce")
            s_df = s_df.dropna(subset=[x_axis, base_col]).sort_values(by=x_axis)

            if s_df.empty:
                st.warning("No data available for this column after NA cleaning.")
            else:
                if pd.api.types.is_datetime64_any_dtype(s_df[x_axis]):
                    t_sec = (s_df[x_axis] - s_df[x_axis].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
                else:
                    t_sec = pd.to_numeric(s_df[x_axis], errors="coerce").to_numpy(dtype=float)

                y_vals = s_df[base_col].to_numpy(dtype=float)

                def moving_average(a, w: int):
                    if w <= 1:
                        return a
                    w = int(w)
                    kernel = np.ones(w) / w
                    return np.convolve(a, kernel, mode="same")

                y_s = moving_average(y_vals, int(smooth_win))

                plot_df = s_df.copy()
                plot_df["__y__"] = y_s

                if show_derivative:
                    plot_df["__d_dt__"] = np.gradient(y_s, t_sec)

                if show_integral:
                    integ = np.zeros_like(y_s, dtype=float)
                    dt = np.diff(t_sec)
                    integ[1:] = np.cumsum((y_s[1:] + y_s[:-1]) * 0.5 * dt)
                    plot_df["__int_dt__"] = integ

                y_series = ["__y__"]
                labels = {"__y__": base_col}
                if show_derivative:
                    y_series.append("__d_dt__")
                    labels["__d_dt__"] = f"d({base_col})/dt"
                if show_integral:
                    y_series.append("__int_dt__")
                    labels["__int_dt__"] = f"∫ {base_col} dt"

                fig_single = px.line(
                    plot_df,
                    x=x_axis,
                    y=y_series,
                    markers=False,
                    title=f"{base_col} (and optional derivative/integral) vs {x_axis}"
                )
                fig_single.for_each_trace(lambda tr: tr.update(name=labels.get(tr.name, tr.name)))
                st.plotly_chart(fig_single, use_container_width=True)

                if show_single_preview:
                    show_cols = [x_axis, base_col, "__y__"]
                    if show_derivative:
                        show_cols.append("__d_dt__")
                    if show_integral:
                        show_cols.append("__int_dt__")
                    st.dataframe(plot_df[show_cols].head(300), use_container_width=True)
# ------------------ Consumables Tab ------------------
elif tab_selection == "🍃 Tower state - Consumables and dies":
    log_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv") or f.endswith(".xlsx")]
    selected_log_file = st.sidebar.selectbox("Select Log File for Gas Calculation", log_files, key="log_file_select")

    # Load saved stock levels if they exist
    stock_path = "stock_levels.json"
    if os.path.exists(stock_path):
        with open(stock_path, "r") as f:
            try:
                saved_stock = json.load(f)
                gas_stock = saved_stock.get("gas_stock", 0.0)
                coating_stock = saved_stock.get("coating_stock", 0.0)
            except Exception:
                gas_stock = 0.0
                coating_stock = 0.0
    else:
        gas_stock = 0.0
        coating_stock = 0.0

    st.title("🍃 Consumables")
    st.subheader("Coating Containers & Argon Vessel Visualization")
    with open("config_coating.json", "r") as config_file:
        config = json.load(config_file)
    coatings = config.get("coatings", {})

    st.markdown("---")
    st.subheader("🏷️ Coating Stock by Type")

    # Load or initialize stock levels for each coating type
    stock_file = "coating_type_stock.json"
    if os.path.exists(stock_file):
        with open(stock_file, "r") as f:
            try:
                coating_type_stock = json.load(f)
            except Exception:
                coating_type_stock = {ctype: 0.0 for ctype in coatings.keys()}
    else:
        coating_type_stock = {ctype: 0.0 for ctype in coatings.keys()}

    # Display and update coating stock per type with vessel-style visuals
    coating_types = list(coatings.keys())
    rows = [coating_types[i:i + 4] for i in range(0, len(coating_types), 4)]
    updated_stock = {}

    for row in rows:
        cols = st.columns(len(row))
        for i, coating_type in enumerate(row):
            with cols[i]:
                current_value = coating_type_stock.get(coating_type, 0.0)
                updated_stock[coating_type] = st.slider(
                    f"{coating_type}", min_value=0.0, max_value=40.0, value=float(current_value), step=0.1, key=f"stock_{coating_type}"
                )
                fill_height = int((updated_stock[coating_type] / 40) * 100)
                st.markdown(
                    f"""
                    <div style='height: 120px; width: 30px; border: 1px solid black; margin: auto; position: relative; background: #eee;'>
                        <div style='position: absolute; bottom: 0; height: {fill_height}%; width: 100%; background: #4CAF50;'></div>
                    </div>
                <p style='text-align: center;'>{updated_stock[coating_type]:.1f} kg</p>
                    """,
                    unsafe_allow_html=True
                )

    if st.button("💾 Save Coating Stock by Type"):
        with open(stock_file, "w") as f:
            json.dump(updated_stock, f, indent=4)
        st.success("Coating stock levels saved!")

    st.markdown("### 🧪 Coating Containers (A, B, C, D)")
    container_cols = st.columns(4)
    container_labels = ["A", "B", "C", "D"]
    container_levels = {}
    container_temps = {}

    import os

    CONFIG_PATH = "container_config.json"

    # Load saved configuration if it exists
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            try:
                saved_config = json.load(f)
                if not isinstance(saved_config, dict):
                    saved_config = {}
            except Exception:
                saved_config = {}
    else:
        saved_config = {}

    # Use saved values if available
    for label in container_labels:
        st.session_state.setdefault(f"level_{label}", saved_config.get(label, {}).get("level", 50))
        st.session_state.setdefault(f"coating_type_{label}", saved_config.get(label, {}).get("type", ""))
        st.session_state.setdefault(f"temp_{label}", saved_config.get(label, {}).get("temp", 25.0))

    for col, label in zip(container_cols, container_labels):
        with col:
            st.markdown(f"**Container {label}**")
            level_key = f"level_{label}"
            type_key = f"coating_type_{label}"
            temp_key = f"temp_{label}"

            # Input controls managed by Streamlit defaults
            default_level = updated_stock.get(label, saved_config.get(label, {}).get("level", 0.0))
            level = st.slider(f"Fill Level {label} (kg)", min_value=0.0, max_value=4.0, value=float(default_level), step=0.1, key=level_key)
            coating_options = list(coatings.keys())
            default_type = st.session_state.get(type_key, "")
            if default_type not in coating_options:
                default_type = coating_options[0] if coating_options else ""
            st.session_state[type_key] = default_type
            coating_type = st.selectbox(f"Coating Type for {label}", options=coating_options, key=type_key)
            temperature = st.number_input(f"Temperature for {label} (°C)", min_value=0.0, step=0.1, key=temp_key)

            # Store values
            container_levels[label] = level
            container_temps[label] = temperature

            # Progress bar
            fill_height = int((level / 4.0) * 100)
            st.markdown(
                f"""
                <div style='height: 120px; width: 30px; border: 1px solid black; margin: auto; position: relative; background: #eee;'>
                    <div style='position: absolute; bottom: 0; height: {fill_height}%; width: 100%; background: #4CAF50;'></div>
                </div>
                <p style='text-align: center;'>{level:.1f} kg</p>
                """,
                unsafe_allow_html=True
            )
            refill_checkbox = st.checkbox(f"Refill Container {label}?", key=f"refill_{label}")
            if refill_checkbox:
                refill_kg = st.number_input(f"Amount to Refill (kg)", min_value=0.0, step=0.1, key=f"refill_kg_{label}")
                if st.button(f"💾 Confirm Refill {label}"):

                    coating_type = st.session_state[type_key]
                    stock_file = "coating_type_stock.json"
                    if os.path.exists(stock_file):
                        with open(stock_file, "r") as f:
                            coating_type_stock = json.load(f)
                    else:
                        coating_type_stock = {}

                    current_stock = coating_type_stock.get(coating_type, 0.0)
                    coating_type_stock[coating_type] = max(0.0, current_stock - refill_kg)

                    with open(stock_file, "w") as f:
                        json.dump(coating_type_stock, f, indent=4)
                    updated_stock[coating_type] = coating_type_stock[coating_type]
                    # Save refill info to config and update session state
                    if os.path.exists(CONFIG_PATH):
                        with open(CONFIG_PATH, "r") as f:
                            config_data = json.load(f)
                    else:
                        config_data = {}

                    new_level = min(4.0, st.session_state[level_key] + refill_kg)
                    config_data[label] = {
                        "level": new_level,
                        "type": coating_type,
                        "temp": st.session_state[temp_key]
                    }
                    with open(CONFIG_PATH, "w") as f:
                        json.dump(config_data, f, indent=4)

                    # Instead of trying to assign to st.session_state[level_key], assign to updated_stock
                    updated_stock[label] = new_level

                    st.rerun()
    st.subheader("🔥 Coating Heater Temperatures")

    heater_config_path = "heater_config.json"
    if os.path.exists(heater_config_path):
        with open(heater_config_path, "r") as f:
            try:
                saved_heater_config = json.load(f)
            except Exception:
                saved_heater_config = {}
    else:
        saved_heater_config = {}

    main_default = saved_heater_config.get("main_heater_temp", 0.0)
    secondary_default = saved_heater_config.get("secondary_heater_temp", 0.0)

    main_heater_temp = st.number_input("Main Coating Heater Temperature (°C)", min_value=0.0, step=0.1, value=main_default, key="main_heater_temp_value")
    secondary_heater_temp = st.number_input("Secondary Coating Heater Temperature (°C)", min_value=0.0, step=0.1, value=secondary_default, key="secondary_heater_temp_value")

    if st.button("💾 Save Heater Temperature Configuration"):
        heater_config = {
            "main_heater_temp": main_heater_temp,
            "secondary_heater_temp": secondary_heater_temp
        }
        with open("heater_config.json", "w") as f:
            json.dump(heater_config, f, indent=4)
    st.success("Heater temperature configuration saved!")
    # Gas calculation logic
    if selected_log_file:
        # Display the "Calculate Gas Spent" button only if a file is selected
        if st.button("Calculate Gas Spent"):
            log_file_path = os.path.join(DATA_FOLDER, selected_log_file)
            if selected_log_file.endswith(".csv"):
                log_data = pd.read_csv(log_file_path)
            else:
                log_data = pd.read_excel(log_file_path)

            # Ensure "Date/Time" column is parsed correctly
            def try_parse_datetime(dt_str):
                try:
                    return pd.to_datetime(dt_str, errors='coerce')  # Use 'coerce' to handle invalid timestamps
                except Exception:
                    return pd.NaT

            # Try parsing the "Date/Time" column
            if "Date/Time" in log_data.columns:
                log_data["Date/Time"] = log_data["Date/Time"].apply(try_parse_datetime)
            else:
                st.error("No 'Date/Time' column found in the data.")
                st.stop()

            # Check if valid timestamps exist after parsing
            if log_data["Date/Time"].isna().all():
                st.error("No valid timestamps found in the log data.")
                st.dataframe(log_data)
                st.stop()

            # Gas calculation logic
            total_gas = 0.0
            mfc_columns = ["Furnace MFC1 Actual", "Furnace MFC2 Actual", "Furnace MFC3 Actual", "Furnace MFC4 Actual"]
            if all(col in log_data.columns for col in mfc_columns):
                log_data["Total Flow"] = log_data[mfc_columns].sum(axis=1)
                time_column = "Date/Time"
                log_data[time_column] = log_data[time_column].apply(try_parse_datetime)

                log_data['Time Difference'] = log_data[time_column].diff().dt.total_seconds() / 60.0  # in minutes
                total_gas = 0
                for i in range(1, len(log_data)):
                    flow_avg = (log_data['Total Flow'].iloc[i - 1] + log_data['Total Flow'].iloc[i]) / 2
                    time_diff = log_data['Time Difference'].iloc[i]
                    total_gas += flow_avg * time_diff

            # Show the result of the gas calculation
            st.success(f"Gas calculation completed! Total Gas Spent: {total_gas:.2f} liters")

            # Option to save the result to the CSV file
            save_to_csv = st.checkbox("Save Gas Calculation to CSV", key="save_gas_to_csv")

            if save_to_csv:
                # Let the user choose a CSV file to append the result
                csv_files = [f for f in os.listdir('data_set_csv') if f.endswith('.csv')]
                selected_csv = st.selectbox("Select CSV to Save Gas Calculation", csv_files)

                if selected_csv:
                    csv_path = os.path.join('data_set_csv', selected_csv)

                    try:
                        df_csv = pd.read_csv(csv_path)
                    except FileNotFoundError:
                        st.error(f"CSV file '{selected_csv}' not found.")
                        st.stop()

                    # Prepare the new data to append
                    new_data = [
                        {"Parameter Name": "Total Gas Spent", "Value": total_gas, "Units": "liters"}
                    ]
                    new_df = pd.DataFrame(new_data)

                    # Append the new data to the existing DataFrame
                    df_csv = pd.concat([df_csv, new_df], ignore_index=True)

                    # Save the updated CSV
                    df_csv.to_csv(csv_path, index=False)
                    st.success(f"Gas calculation saved to '{selected_csv}'!")
        else:
            st.warning("Please select a CSV file before calculating the gas.")
    else:
        st.warning("Please select a CSV file for calculation.")
    # =========================
    # 🔩 Dies System (names from JSON keys) — auto save/load
    # =========================
    st.markdown("---")
    st.subheader("🔩 Dies System")

    DIES_CONFIG_PATH = "dies_6station.json"

    # Default template if file doesn't exist yet
    default_cfg = {
        f"Station {i}": {
            "entry_die_um": 0.0,
            "primary_die_um": 0.0,
            "primary_on_tower": False,
            "secondary_on_tower": False
        } for i in range(1, 7)
    }

    # Load or create
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

    # IMPORTANT: station names come from the JSON keys (your names)
    station_names = list(dies_cfg.keys())

    # Init session_state based on JSON
    for name in station_names:
        safe_key = name.replace(" ", "_").replace("/", "_")
        st.session_state.setdefault(f"dies_entry_{safe_key}", float(dies_cfg.get(name, {}).get("entry_die_um", 0.0)))
        st.session_state.setdefault(f"dies_primary_{safe_key}",
                                    float(dies_cfg.get(name, {}).get("primary_die_um", 0.0)))
        st.session_state.setdefault(f"dies_primary_on_{safe_key}",
                                    bool(dies_cfg.get(name, {}).get("primary_on_tower", False)))
        st.session_state.setdefault(f"dies_secondary_on_{safe_key}",
                                    bool(dies_cfg.get(name, {}).get("secondary_on_tower", False)))

    # Layout: 3 per row
    rows = [station_names[i:i + 3] for i in range(0, len(station_names), 3)]
    updated_dies_cfg = {}

    for row in rows:
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            safe_key = name.replace(" ", "_").replace("/", "_")

            with col:
                st.markdown(f"### {name}")

                entry_um = st.number_input(
                    "Entry die (µm)",
                    min_value=0.0,
                    step=1.0,
                    format="%.1f",
                    key=f"dies_entry_{safe_key}",
                )

                primary_um = st.number_input(
                    "Primary die (µm)",
                    min_value=0.0,
                    step=1.0,
                    format="%.1f",
                    key=f"dies_primary_{safe_key}",
                )

                primary_on = st.checkbox("Primary on tower", key=f"dies_primary_on_{safe_key}")
                secondary_on = st.checkbox("Secondary on tower", key=f"dies_secondary_on_{safe_key}")

                updated_dies_cfg[name] = {
                    "entry_die_um": float(entry_um),
                    "primary_die_um": float(primary_um),
                    "primary_on_tower": bool(primary_on),
                    "secondary_on_tower": bool(secondary_on),
                }

                st.caption(f"Entry: **{entry_um:.1f} µm** | Primary: **{primary_um:.1f} µm**")

    # Auto-save every rerun
    try:
        with open(DIES_CONFIG_PATH, "w") as f:
            json.dump(updated_dies_cfg, f, indent=4)
        st.caption(f"Auto-saved to `{DIES_CONFIG_PATH}`")
    except Exception as e:
        st.error(f"Failed to save dies config: {e}")
# ------------------ History Log Tab ------------------
elif tab_selection == "📝 History Log":
    st.title("📝 History Log")
    st.sidebar.title("History Log Management")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
        if 'Status' not in history_df.columns:
            history_df['Status'] = 'Not Yet Addressed'

        # Define relevant columns for each history type
        draw_history_fields = ["Draw Name", "Timestamp"]

        problem_history_fields = ["Description", "Status"]

        maintenance_history_fields = ["Part Changed", "Notes"]

        fields_mapping = {
            "Draw History": draw_history_fields,
            "Problem History": problem_history_fields,
            "Maintenance History": maintenance_history_fields
        }

        # Ensure column names are unique
        def make_column_names_unique(columns):
            seen = {}
            new_columns = []
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)
            return new_columns

        history_df.columns = make_column_names_unique(history_df.columns.tolist())

        # Sidebar Selection for History Type
        history_type = st.sidebar.radio("Select History Type", [ "Draw History", "Problem History", "Maintenance History"], key="history_type_select")

        if history_type == "All":
            # Show all history logs with separate tables & plots
            for log_type, fields in zip(["Draw History", "Problem History", "Maintenance History"],
                                        [draw_history_fields, problem_history_fields, maintenance_history_fields]):
                st.write(f"## {log_type}")

                if log_type == "Problem History":
                    filtered_df = history_df[(history_df["Type"] == log_type) & (history_df["Status"] != "Fixed")]
                else:
                    filtered_df = history_df[history_df["Type"] == log_type]
                if not filtered_df.empty:
                    st.write(f"### {log_type} Table")
                    st.data_editor(filtered_df[fields], height=200, use_container_width=True)


                else:
                    st.warning(f"No records found for {log_type}")

        else:
            # Show only selected history type
            if history_type == "Problem History":
                filtered_df = history_df[(history_df["Type"] == history_type) & (history_df["Status"] != "Fixed")]
            else:
                filtered_df = history_df[history_df["Type"] == history_type]
            if not filtered_df.empty:
                st.write(f"### {history_type} Table")
                st.data_editor(filtered_df[fields_mapping[history_type]], height=200, use_container_width=True)


            else:
                st.warning(f"No records found for {history_type}")

        # ------------------ Add Event Form ------------------
        with open("config_coating.json", "r") as config_file:
            config = json.load(config_file)
        # Extract coatings and dies dictionaries
        coatings = config.get("coatings", {})
        dies = config.get("dies", {})

        if history_type == "Draw History":
            # st.sidebar.subheader("Draw History")
            # Load all CSV files from the dataset folder and combine them
            data_set_files = [f for f in os.listdir('data_set_csv') if f.endswith('.csv')]
            folder_data = []
            for file in data_set_files:
                csv_data = pd.read_csv(os.path.join('data_set_csv', file), header=None)
                if not csv_data.empty:
                    csv_data.columns = ['Parameter Name', 'Value', 'Units']
                    csv_data['Draw Name'] = file.replace('.csv', '')
                    folder_data.append(csv_data)
            if folder_data:
                all_data = pd.concat(folder_data, ignore_index=True)
                st.write("### Combined Draw History from All CSV Files")
                # Let the user select parameters to display from the combined data
                parameters_to_display = st.multiselect("Select Parameters to Display", all_data["Parameter Name"].unique().tolist())
                if parameters_to_display:
                    filtered_data = all_data[all_data["Parameter Name"].isin(parameters_to_display)]
                    st.dataframe(filtered_data, height=300, use_container_width=True)
                    # Optionally, allow detailed view per draw entry
                    selected_draw = st.selectbox("Select a Draw Entry", filtered_data["Parameter Name"].tolist())
                    if selected_draw:
                        selected_draw_data = filtered_data[filtered_data["Parameter Name"] == selected_draw].iloc[0]
                        st.write(f"**Selected Data for {selected_draw}:**")
                        st.write(f"**Value:** {selected_draw_data['Value']} {selected_draw_data['Units']}")
                else:
                    st.warning("No parameters selected.")
            else:
                st.warning("No CSV files found in the folder.")
        elif history_type == "Maintenance History":
            st.sidebar.subheader("Add Maintenance History Entry")

            # Checkbox to indicate if a part was changed
            part_changed_checkbox = st.sidebar.checkbox("Was a part changed?")

            # Show part name input only if checked
            part_changed = ""

            if part_changed_checkbox:
                part_changed = st.sidebar.text_input("Part Changed")

            maintenance_notes = st.sidebar.text_area("Maintenance Details")

            if st.sidebar.button("Save Maintenance History"):
                new_entry = pd.DataFrame([{
                    "Timestamp": pd.Timestamp.now(),
                    "Type": "Maintenance History",
                    "Part Changed": part_changed if part_changed_checkbox else "N/A",
                    "Notes": maintenance_notes
                }])

                history_df = pd.concat([history_df, new_entry], ignore_index=True)
                history_df.to_csv(HISTORY_FILE, index=False)
                st.sidebar.success("Maintenance history saved!")
        elif history_type == "Problem History":
            st.sidebar.subheader("Add or Update Problem History Entry")
            problem_action = st.sidebar.radio("Select Action", ["Add New Problem", "Update Existing Problem"], index=0)
            if problem_action == "Add New Problem":
                problem_description = st.sidebar.text_area("Describe the Problem")
                problem_status = st.sidebar.selectbox("Problem Status", ["Not Yet Addressed", "Waiting for Parts", "Fixed"])
                if st.sidebar.button("Save Problem History"):
                    new_entry = pd.DataFrame([{
                        "Timestamp": pd.Timestamp.now(),
                        "Type": "Problem History",
                        "Description": problem_description,
                        "Status": problem_status
                    }])

                    history_df = pd.concat([history_df, new_entry], ignore_index=True)
                    history_df.to_csv(HISTORY_FILE, index=False)
                    #st.rerun()
            elif problem_action == "Update Existing Problem":
                filtered_df = history_df[(history_df["Type"] == "Problem History") & (history_df["Status"] != "Fixed")]
                if not filtered_df.empty:
                    selected_problem = st.sidebar.selectbox("Select Problem to Update", filtered_df["Description"])
                    selected_index = filtered_df.index[filtered_df["Description"] == selected_problem].tolist()[0]

                    new_status = st.sidebar.selectbox("Update Problem Status",
                                                      ["Not Yet Addressed", "Waiting for Parts", "Fixed"],
                                                      index=["Not Yet Addressed", "Waiting for Parts", "Fixed"].index(
                                                          filtered_df.at[selected_index, "Status"]))

                    if st.sidebar.button("Update Problem Status"):
                        history_df.at[selected_index, "Status"] = new_status
                        history_df.to_csv(HISTORY_FILE, index=False)
                        #st.rerun()
                else:
                    st.sidebar.info("No existing problem entries to update.")
    else:
        st.warning("No history logs found. You can add new records using the form below.")# Ensure df is only processed if it contains data
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

        # ------------------ Schedule Tab ------------------
        if tab_selection == "📅 Schedule":
            st.title("📅 Tower Schedule")
            st.sidebar.title("Schedule Management")

            SCHEDULE_FILE = "tower_schedule.csv"
            required_columns = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]
            if not os.path.exists(SCHEDULE_FILE):
                pd.DataFrame(columns=required_columns).to_csv(SCHEDULE_FILE, index=False)
                st.warning("Schedule file was empty. New file with required columns created.")
            else:
                schedule_df = pd.read_csv(SCHEDULE_FILE)
                missing_columns = [col for col in required_columns if col not in schedule_df.columns]
                if missing_columns:
                    st.error(f"Missing columns in schedule file: {missing_columns}")
                    st.stop()
                else:
                    # Clean column names by stripping extra spaces
                    schedule_df.columns = schedule_df.columns.str.strip()

                    # Parse 'Start DateTime' and 'End DateTime' columns
                    try:
                        schedule_df['Start DateTime'] = pd.to_datetime(schedule_df['Start DateTime'], errors='coerce')
                        schedule_df['End DateTime'] = pd.to_datetime(schedule_df['End DateTime'], errors='coerce')
                    except Exception as e:
                        st.error(f"Error parsing datetime columns: {e}")
                        st.stop()

                    # Check if 'Start DateTime' and 'End DateTime' columns are valid
                    if schedule_df['Start DateTime'].isna().all() or schedule_df['End DateTime'].isna().all():
                        st.error("One or both datetime columns ('Start DateTime', 'End DateTime') could not be parsed. Please check the data.")
                        st.stop()

                    # Apply date filtering safely
                    start_filter = st.sidebar.date_input("Start Date", pd.Timestamp.now().date(), key="schedule_start_date")
                    end_filter = st.sidebar.date_input("End Date", (pd.Timestamp.now() + pd.DateOffset(weeks=1)).date(), key="schedule_end_date")

                    start_datetime = schedule_df['Start DateTime']
                    end_datetime = schedule_df['End DateTime']

                    # Apply filtering based on user-selected date range
                    filtered_schedule = schedule_df[
                        (start_datetime >= pd.to_datetime(start_filter)) &
                        (end_datetime <= pd.to_datetime(end_filter))
                    ]

                    # Display schedule as a timeline
                    st.write("### Schedule Timeline")
                    event_colors = {
                        "Maintenance": "blue",
                        "Drawing": "green",
                        "Stop": "red",
                        "Management Event": "purple"  # New color for the management event
                    }
                    if not filtered_schedule.empty:
                        fig = px.timeline(
                            filtered_schedule,
                            x_start="Start DateTime",
                            x_end="End DateTime",
                            y="Event Type",
                            color="Event Type",
                            title="Tower Schedule",
                            color_discrete_map=event_colors
                        )
                        st.plotly_chart(fig, use_container_width=True)


                    st.write("### Current Schedule")
                    st.data_editor(schedule_df, height=300, use_container_width=True)
                    # Add new event form
                    st.sidebar.subheader("Add New Event")
                    event_description = st.sidebar.text_area("Event Description")
                    event_type = st.sidebar.selectbox("Select Event Type", ["Maintenance", "Drawing", "Stop","Management Event"])
                    deadline_date = None
                    if event_type == "Management Event":
                        deadline_date = st.sidebar.date_input("Deadline Date")
                    start_date = st.sidebar.date_input("Start Date", pd.Timestamp.now().date())
                    start_time = st.sidebar.time_input("Start Time")
                    end_date = st.sidebar.date_input("End Date", pd.Timestamp.now().date())
                    end_time = st.sidebar.time_input("End Time")
                    recurrence = st.sidebar.selectbox("Recurrence", ["None", "Weekly", "Monthly", "Yearly"])

                    start_datetime = pd.to_datetime(f"{start_date} {start_time}")
                    end_datetime = pd.to_datetime(f"{end_date} {end_time}")

                    if st.sidebar.button("Add Event"):
                        new_event = pd.DataFrame([{
                            "Event Type": event_type,
                            "Start DateTime": start_datetime,
                            "End DateTime": end_datetime,
                            "Description": event_description,
                            "Recurrence": recurrence,
                            "Deadline Date": deadline_date if event_type == "Management Event" else None
                            # Add deadline only for Management Event
                        }])

                        full_schedule_df = pd.read_csv(SCHEDULE_FILE)
                        full_schedule_df = pd.concat([full_schedule_df, new_event], ignore_index=True)
                        full_schedule_df.to_csv(SCHEDULE_FILE, index=False)

                        st.sidebar.success("Event added to schedule!")
# ------------------ Schedule Tab ------------------
elif tab_selection == "📅 Schedule":
    st.title("📅 Tower Schedule")
    st.sidebar.title("Schedule Management")

    SCHEDULE_FILE = "tower_schedule.csv"
    required_columns = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    # --- Ensure schedule file exists + has required columns (works with empty/old files) ---
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

    # --- Clean leaked Plotly template text from Description (prevents showing %{customdata[0]} in hover) ---
    schedule_df["Description"] = (
        schedule_df["Description"]
        .astype(str)
        .str.replace(r"%\{.*?\}", "", regex=True)
        .str.replace("Description=", "", regex=False)
        .str.strip()
    )

    # ----------------------------
    # Sidebar date filter
    # ----------------------------
    start_filter = st.sidebar.date_input(
        "Start Date", pd.Timestamp.now().date(), key="schedule_start_date"
    )
    end_filter = st.sidebar.date_input(
        "End Date", (pd.Timestamp.now() + pd.DateOffset(weeks=1)).date(), key="schedule_end_date"
    )

    range_start = pd.to_datetime(start_filter)
    range_end = pd.to_datetime(end_filter) + pd.to_timedelta(1, unit="day")  # include the end day

    # Filter safely (ignore NaT rows)
    base = schedule_df.dropna(subset=["Start DateTime", "End DateTime"]).copy()

    # -------------------------------------------------
    # Expand recurring events so they "show all"
    # -------------------------------------------------
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
        rec = str(row.get("Recurrence", "")).strip()
        start_dt = row["Start DateTime"]
        end_dt = row["End DateTime"]

        # If no recurrence -> keep as single event
        if rec in ["", "None", "none", "NONE"]:
            expanded_rows.append(row.to_dict())
            continue

        # Duration stays constant for each occurrence
        duration = end_dt - start_dt

        # Move occurrence start forward until it can intersect the window
        occ_start = start_dt
        occ_end = occ_start + duration

        # If the base event is far in the past, fast-forward occurrences
        # until it reaches near the range (avoid infinite loops)
        safety = 0
        while occ_end < range_start and safety < 5000:
            occ_start = _next_dt(occ_start, rec)
            occ_end = occ_start + duration
            safety += 1

        # Now generate occurrences while within window
        safety = 0
        while occ_start <= range_end and safety < 5000:
            new_row = row.to_dict()
            new_row["Start DateTime"] = occ_start
            new_row["End DateTime"] = occ_end
            # Optional: mark that it's an occurrence (keeps description same but more clear)
            # new_row["Description"] = f"{new_row['Description']} (recurring)"
            expanded_rows.append(new_row)

            occ_start = _next_dt(occ_start, rec)
            occ_end = occ_start + duration
            safety += 1

    expanded_df = pd.DataFrame(expanded_rows)
    if not expanded_df.empty:
        expanded_df["Start DateTime"] = pd.to_datetime(expanded_df["Start DateTime"], errors="coerce")
        expanded_df["End DateTime"] = pd.to_datetime(expanded_df["End DateTime"], errors="coerce")
        expanded_df = expanded_df.dropna(subset=["Start DateTime", "End DateTime"])

    # --------------------------------
    # Filter (OVERLAP logic) on expanded
    # --------------------------------
    # event intersects [range_start, range_end]
    filtered_schedule = expanded_df[
        (expanded_df["End DateTime"] >= range_start) &
        (expanded_df["Start DateTime"] <= range_end)
    ].copy()

    # --- Timeline ---
    st.write("### Schedule Timeline")
    event_colors = {
        "Maintenance": "blue",
        "Drawing": "green",
        "Stop": "red",
        "Management Event": "purple",
    }

    if not filtered_schedule.empty:
        # Build timeline with explicit custom_data so hover is stable
        fig = px.timeline(
            filtered_schedule,
            x_start="Start DateTime",
            x_end="End DateTime",
            y="Event Type",
            color="Event Type",
            title="Tower Schedule",
            color_discrete_map=event_colors,
            custom_data=["Description", "Recurrence"],
        )

        # Force a clean hover that always includes Description + Recurrence
        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Start: %{x|%Y-%m-%d %H:%M}<br>"
                "End: %{x_end|%Y-%m-%d %H:%M}<br>"
                "Recurrence: %{customdata[1]}<br>"
                "Description: %{customdata[0]}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events in the selected date range.")

    # --- Current schedule table (MASTER, not expanded) ---
    st.write("### Current Schedule (Master)")
    st.data_editor(schedule_df, height=300, use_container_width=True)

    # --- Add new event ---
    st.sidebar.subheader("Add New Event")
    event_description = st.sidebar.text_area("Event Description", key="sched_desc")
    event_type = st.sidebar.selectbox(
        "Select Event Type", ["Maintenance", "Drawing", "Stop", "Management Event"], key="sched_type"
    )
    start_date = st.sidebar.date_input("Start Date", pd.Timestamp.now().date(), key="sched_start_date2")
    start_time = st.sidebar.time_input("Start Time", key="sched_start_time")
    end_date = st.sidebar.date_input("End Date", pd.Timestamp.now().date(), key="sched_end_date2")
    end_time = st.sidebar.time_input("End Time", key="sched_end_time")
    recurrence = st.sidebar.selectbox("Recurrence", ["None", "Weekly", "Monthly", "Yearly"], key="sched_recur")

    start_datetime = pd.to_datetime(f"{start_date} {start_time}")
    end_datetime = pd.to_datetime(f"{end_date} {end_time}")

    if st.sidebar.button("Add Event", key="sched_add_btn"):
        new_event = pd.DataFrame([{
            "Event Type": event_type,
            "Start DateTime": start_datetime,
            "End DateTime": end_datetime,
            "Description": event_description,
            "Recurrence": recurrence,
        }])

        full_schedule_df = pd.read_csv(SCHEDULE_FILE)
        for c in required_columns:
            if c not in full_schedule_df.columns:
                full_schedule_df[c] = ""
        full_schedule_df = full_schedule_df[required_columns]

        full_schedule_df = pd.concat([full_schedule_df, new_event], ignore_index=True)
        full_schedule_df.to_csv(SCHEDULE_FILE, index=False)
        st.sidebar.success("Event added to schedule!")
        st.rerun()

    # --- Delete event ---
    st.sidebar.subheader("Delete Event")
    if not schedule_df.empty:
        delete_options = [
            f"{i}: {schedule_df.loc[i, 'Event Type']} | {schedule_df.loc[i, 'Start DateTime']} | {schedule_df.loc[i, 'Description']}"
            for i in schedule_df.index
        ]
        to_delete = st.sidebar.selectbox("Select Event to Delete", delete_options, key="sched_del_select")
        del_idx = int(to_delete.split(":")[0])

        if st.sidebar.button("Delete Event", key="sched_del_btn"):
            schedule_df = schedule_df.drop(index=del_idx).reset_index(drop=True)
            schedule_df.to_csv(SCHEDULE_FILE, index=False)
            st.sidebar.success("Event deleted successfully!")
            st.rerun()
    else:
        st.sidebar.info("No events available for deletion.")
# ------------------ Draw order Tab ------------------
elif tab_selection == "📦 Order Draw":
    st.title("📦 Order Draw")

    import os
    import datetime as dt
    import pandas as pd
    import streamlit as st

    orders_file = "draw_orders.csv"
    SCHEDULE_FILE = "tower_schedule.csv"
    schedule_required_cols = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    # Password for scheduling
    SCHEDULE_PASSWORD = "DORON"

    # New column (replaces Spools)
    GOOD_ZONES_COL = "Good Zones Count (required length zones)"

    # Ensure schedule file exists
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=schedule_required_cols).to_csv(SCHEDULE_FILE, index=False)

    # =========================
    # Helper (kept, harmless even if not used here)
    # =========================
    def _append_done_desc_to_dataset_csv(dataset_dir: str, csv_filename: str, done_desc: str):
        try:
            csv_filename = str(csv_filename or "").strip()
            done_desc = str(done_desc or "").strip()

            if not csv_filename or not done_desc:
                return False, "Missing CSV filename or Done Description."

            csv_path = os.path.join(dataset_dir, csv_filename)
            if not os.path.exists(csv_path):
                return False, f"Dataset CSV not found: {csv_filename}"

            df_csv = pd.read_csv(csv_path, keep_default_na=False)
            for c in ["Parameter Name", "Value", "Units"]:
                if c not in df_csv.columns:
                    df_csv[c] = ""

            new_rows = pd.DataFrame([
                {"Parameter Name": "Done Description", "Value": done_desc, "Units": ""},
                {"Parameter Name": "Done Timestamp", "Value": pd.Timestamp.now(), "Units": ""},
            ])

            df_csv = pd.concat([df_csv, new_rows], ignore_index=True)
            df_csv.to_csv(csv_path, index=False)
            return True, f"Saved Done Description into {csv_filename}"
        except Exception as e:
            return False, f"Failed writing Done Description into dataset CSV: {e}"

    # =========================
    # 1) TABLE FIRST
    # =========================
    st.subheader("📋 Existing Draw Orders")

    if not os.path.exists(orders_file):
        st.info("No orders submitted yet.")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(orders_file, keep_default_na=False)

    # ---- Fix types / normalize ----
    if not df.empty:
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        for _c in [
            "Done CSV",
            "Done Description",
            "T&M Moved Timestamp",
            "Notes",
            "Fiber Project",
            "Order Opener",
            "Preform Number",
            "Active CSV",
        ]:
            if _c in df.columns:
                df[_c] = df[_c].astype(str).replace({"nan": "", "None": ""}).fillna("")

        # ---- Fix missing columns ----
        for col, default in {
            "Status": "Pending",
            "Priority": "Normal",
            "Fiber Project": "",
            "Order Opener": "",
            "Preform Number": "",
            "Done CSV": "",
            "Done Description": "",
            "Active CSV": "",
            "T&M Moved": False,
            "T&M Moved Timestamp": "",
            "Required Length (m) (for T&M+costumer)": "",
            GOOD_ZONES_COL: "",  # NEW
        }.items():
            if col not in df.columns:
                df[col] = default

        # Backward-compat: migrate old "Length (m)" -> new required length column (if new empty)
        if "Length (m)" in df.columns and "Required Length (m) (for T&M+costumer)" in df.columns:
            _new = "Required Length (m) (for T&M+costumer)"
            mask = df[_new].astype(str).str.strip().eq("") & df["Length (m)"].astype(str).str.strip().ne("")
            if mask.any():
                df.loc[mask, _new] = df.loc[mask, "Length (m)"]

        # Backward-compat: migrate old "Spools" -> GOOD_ZONES_COL (if new empty)
        if "Spools" in df.columns and GOOD_ZONES_COL in df.columns:
            mask = df[GOOD_ZONES_COL].astype(str).str.strip().eq("") & df["Spools"].astype(str).str.strip().ne("")
            if mask.any():
                df.loc[mask, GOOD_ZONES_COL] = df.loc[mask, "Spools"]

        # ---- Reorder Columns ----
        desired_order = [
            "Status",
            "Priority",
            "Order Opener",
            "Preform Number",
            "Fiber Project",
            "Timestamp",
            "Fiber Diameter (µm)",
            "Main Coating Diameter (µm)",
            "Secondary Coating Diameter (µm)",
            "Tension (g)",
            "Draw Speed (m/min)",
            "Required Length (m) (for T&M+costumer)",
            GOOD_ZONES_COL,  # NEW (replaces Spools)
            "Main Coating",
            "Secondary Coating",
            "Notes",
            "Active CSV",
            "Done CSV",
            "Done Description",
        ]
        other_cols = [c for c in df.columns if c not in desired_order]
        df = df[[c for c in desired_order if c in df.columns] + other_cols]

        # ---- Hide items moved to T&M ----
        if "T&M Moved" in df.columns:
            df["T&M Moved"] = df["T&M Moved"].apply(
                lambda x: str(x).strip().lower() in ("true", "1", "yes", "y", "moved")
            )

        df_visible = df.copy()
        if "T&M Moved" in df_visible.columns:
            df_visible = df_visible[~df_visible["T&M Moved"]].copy()

        # ---- Color formatting ----
        def color_status(val):
            s = str(val).strip()
            colors = {
                "Pending": "orange",
                "In Progress": "dodgerblue",
                "Scheduled": "teal",
                "Failed": "red",
                "Done": "green",
            }
            return f"color: {colors.get(s, 'black')}; font-weight: bold"

        def color_priority(val):
            p = str(val).strip()
            colors = {"Low": "gray", "Normal": "black", "High": "crimson"}
            return f"color: {colors.get(p, 'black')}; font-weight: bold"

        styled_df = (
            df_visible.style
            .applymap(color_status, subset=["Status"] if "Status" in df_visible.columns else None)
            .applymap(color_priority, subset=["Priority"] if "Priority" in df_visible.columns else None)
        )
        st.dataframe(styled_df, use_container_width=True)
    else:
        df_visible = pd.DataFrame()

    # =========================
    # 2) PENDING LIST → SCHEDULE
    # =========================
    st.markdown("---")
    # =====================================================
    # 🟠 PENDING ORDERS → SCHEDULE
    #   • Only Pending
    #   • If Preform Number == 0 → MUST enter real preform
    #   • Saves real preform back into draw_orders.csv
    # =====================================================

    st.markdown("### 🟠 Pending Orders → Schedule")

    df_pending = df_visible.copy()
    if "Status" in df_pending.columns:
        df_pending["Status"] = df_pending["Status"].astype(str).str.strip()
        df_pending = df_pending[df_pending["Status"].str.lower() == "pending"].copy()

    if df_pending.empty:
        st.info("No Pending orders to schedule.")
    else:
        for idx in df_pending.index:
            fiber = str(df.loc[idx, "Fiber Project"]) if "Fiber Project" in df.columns else ""
            preform = str(df.loc[idx, "Preform Number"]).strip()
            prio = str(df.loc[idx, "Priority"]).strip()
            length_m = df.loc[idx, "Length (m)"] if "Length (m)" in df.columns else ""
            notes_txt = str(df.loc[idx, "Notes"]) if "Notes" in df.columns else ""

            header = f"#{idx} | {fiber} | Priority: {prio} | Preform: {preform}"
            with st.expander(header, expanded=False):

                # ---------------------------------
                # REAL PREFORM ENFORCEMENT
                # ---------------------------------
                need_real_preform = (preform == "0" or preform == "")
                real_preform = preform

                if need_real_preform:
                    st.warning("⚠️ This order has no real preform yet.")
                    real_preform = st.text_input(
                        "Enter REAL Preform Number to continue",
                        key=f"real_preform_input_{idx}",
                        placeholder="e.g. P0921",
                    )

                can_schedule = bool(real_preform.strip())

                # ---------------------------------
                # TIME PRESET
                # ---------------------------------
                preset = st.radio(
                    "Time preset",
                    ["All day (08:00–16:00)", "Before lunch (08:00–12:00)", "After lunch (12:00–16:00)"],
                    horizontal=True,
                    key=f"pending_sched_preset_{idx}",
                )

                if preset.startswith("All day"):
                    start_t = dt.time(8, 0)
                    dur_min = 8 * 60
                elif preset.startswith("Before"):
                    start_t = dt.time(8, 0)
                    dur_min = 4 * 60
                else:
                    start_t = dt.time(12, 0)
                    dur_min = 4 * 60

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    sched_date = st.date_input(
                        "Schedule Date",
                        value=pd.Timestamp.today().date(),
                        key=f"pending_sched_date_{idx}",
                    )
                with c2:
                    sched_start = st.time_input(
                        "Start Time",
                        value=start_t,
                        key=f"pending_sched_start_{idx}",
                    )
                with c3:
                    sched_dur = st.number_input(
                        "Duration (min)",
                        min_value=1,
                        step=5,
                        value=int(dur_min),
                        key=f"pending_sched_dur_{idx}",
                    )

                start_dt = pd.to_datetime(f"{sched_date} {sched_start}")
                end_dt = start_dt + pd.to_timedelta(int(sched_dur), unit="m")

                # ---------------------------------
                # PASSWORD
                # ---------------------------------
                pwd = st.text_input(
                    "Scheduling password",
                    type="password",
                    key=f"pending_sched_pwd_{idx}",
                )
                pwd_ok = (pwd == SCHEDULE_PASSWORD)

                if pwd and not pwd_ok:
                    st.error("Wrong password ❌")

                # ---------------------------------
                # SCHEDULE BUTTON
                # ---------------------------------
                if st.button(
                        "📅 Schedule Order",
                        key=f"pending_schedule_btn_{idx}",
                        disabled=not (can_schedule and pwd_ok),
                ):
                    # 1️⃣ Save REAL preform back to orders
                    df.at[idx, "Preform Number"] = real_preform.strip()
                    df.at[idx, "Status"] = "Scheduled"
                    df.to_csv(orders_file, index=False)

                    # 2️⃣ Write schedule event
                    existing = pd.read_csv(SCHEDULE_FILE) if os.path.exists(SCHEDULE_FILE) else pd.DataFrame()
                    for c in schedule_required_cols:
                        if c not in existing.columns:
                            existing[c] = ""
                    existing = existing[schedule_required_cols]

                    desc_lines = [
                        f"ORDER #{idx} | Priority: {prio}",
                        f"Fiber: {fiber} | Preform: {real_preform.strip()}",
                        f"Length: {length_m} m",
                    ]
                    if notes_txt:
                        desc_lines.append(f"Notes: {notes_txt}")

                    new_event = pd.DataFrame([{
                        "Event Type": "Drawing",
                        "Start DateTime": start_dt,
                        "End DateTime": end_dt,
                        "Description": " | ".join(desc_lines),
                        "Recurrence": "None",
                    }])

                    pd.concat([existing, new_event], ignore_index=True).to_csv(SCHEDULE_FILE, index=False)

                    st.success(f"✅ Order #{idx} scheduled with Preform {real_preform}")
                    st.rerun()

    # =========================
    # 3) CREATE NEW ORDER LAST
    # =========================
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
        with st.container(border=True):
            st.markdown("#### ✅ Required fields")

            r1, r2, r3 = st.columns([1.1, 1.1, 1.1])
            with r1:
                preform_name = st.text_input("Preform Number (0 for not yet exist) *", key="order_preform_name")
            with r2:
                fiber_type = st.text_input("Fiber Project *", key="order_fiber_type")
            with r3:
                priority = st.selectbox("Priority *", ["Low", "Normal", "High"], index=1, key="order_priority")

            r4, r5, r6 = st.columns([1.1, 1.1, 1.1])
            with r4:
                length_required = st.number_input(
                    "Required Length (m) (for T&M+costumer) *",
                    min_value=0.0,
                    key="order_length_required_required",
                )
            with r5:
                good_zones = st.number_input(
                    f"{GOOD_ZONES_COL} *",
                    min_value=1,
                    step=1,
                    value=1,
                    key="order_good_zones_required",
                )
            with r6:
                order_opener = st.text_input("Order Opened By *", key="order_opener")

            st.markdown("#### 🧪 Process targets (optional)")
            p1, p2, p3 = st.columns(3)
            with p1:
                fiber_diameter = st.number_input("Fiber Diameter (µm)", min_value=0.0, key="order_fiber_diam")
                tension = st.number_input("Tension (g)", min_value=0.0, key="order_tension")
            with p2:
                diameter_main = st.number_input("Main Coating Diameter (µm)", min_value=0.0, key="order_main_diam")
                draw_speed = st.number_input("Draw Speed (m/min)", min_value=0.0, key="order_speed")
            with p3:
                diameter_secondary = st.number_input("Secondary Coating Diameter (µm)", min_value=0.0, key="order_sec_diam")

            st.markdown("#### 🧴 Materials (optional)")
            m1, m2 = st.columns([1.1, 1.1])
            with m1:
                coating_main = st.text_input("Main Coating Type", key="order_coating_main")
            with m2:
                coating_secondary = st.text_input("Secondary Coating Type", key="order_coating_secondary")

            notes = st.text_area("Additional Notes / Instructions", key="order_notes")

            st.markdown("#### 📅 Optional: schedule immediately (password protected)")
            schedule_now = st.checkbox("Schedule now", value=False, key="order_schedule_now_cb")
            sched_ok = False

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

                cA, cB, cC = st.columns([1, 1, 1])
                with cA:
                    sched_date_new = st.date_input("Schedule Date", value=default_date, key="order_create_sched_date")
                with cB:
                    sched_start_new = st.time_input("Start Time", value=preset_start, key="order_create_sched_start")
                with cC:
                    sched_dur_new = st.number_input(
                        "Duration (min)", min_value=1, step=5, value=int(preset_duration), key="order_create_sched_dur"
                    )

                start_dt_new = pd.to_datetime(f"{sched_date_new} {sched_start_new}")
                end_dt_new = start_dt_new + pd.to_timedelta(int(sched_dur_new), unit="m")

            colA, colB = st.columns([1, 1])
            with colA:
                submit = st.button("📤 Submit Draw Order", key="order_submit_btn")
            with colB:
                cancel = st.button("❌ Cancel", key="order_cancel_btn")

            if cancel:
                st.session_state["show_new_order_form"] = False
                st.rerun()

            if submit:
                missing = []
                if not str(preform_name).strip():
                    missing.append("Preform Number")
                if not str(fiber_type).strip():
                    missing.append("Fiber Project")
                if not str(order_opener).strip():
                    missing.append("Order Opened By")
                if float(length_required) <= 0:
                    missing.append("Required Length (m) (for T&M+costumer)")
                if int(good_zones) <= 0:
                    missing.append(GOOD_ZONES_COL)

                if missing:
                    st.error("Please fill required fields: " + ", ".join(missing))
                    st.stop()

                order_data = {
                    "Status": "Pending",
                    "Priority": priority,
                    "Order Opener": order_opener,
                    "Preform Number": preform_name,
                    "Fiber Project": fiber_type,
                    "Timestamp": pd.Timestamp.now(),
                    "Fiber Diameter (µm)": fiber_diameter,
                    "Main Coating Diameter (µm)": diameter_main,
                    "Secondary Coating Diameter (µm)": diameter_secondary,
                    "Tension (g)": tension,
                    "Draw Speed (m/min)": draw_speed,
                    "Required Length (m) (for T&M+costumer)": float(length_required),
                    GOOD_ZONES_COL: int(good_zones),
                    "Main Coating": coating_main,
                    "Secondary Coating": coating_secondary,
                    "Notes": notes,
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

                if schedule_now:
                    if not sched_ok:
                        st.error("Order saved, but NOT scheduled (password missing/wrong).")
                    else:
                        existing = pd.read_csv(SCHEDULE_FILE) if os.path.exists(SCHEDULE_FILE) else pd.DataFrame()
                        for c in schedule_required_cols:
                            if c not in existing.columns:
                                existing[c] = ""
                        existing = existing[schedule_required_cols]

                        desc_lines = [
                            f"ORDER #{new_idx} | Priority: {priority}",
                            f"Fiber: {fiber_type} | Preform: {preform_name}",
                            f"Required Length: {length_required} m | Good Zones Count: {int(good_zones)}",
                        ]
                        if notes and str(notes).strip():
                            desc_lines.append(f"Notes: {str(notes).strip()}")
                        event_description = " | ".join([x for x in desc_lines if str(x).strip() != ""])

                        new_event = pd.DataFrame([{
                            "Event Type": "Drawing",
                            "Start DateTime": start_dt_new,
                            "End DateTime": end_dt_new,
                            "Description": event_description,
                            "Recurrence": "None",
                        }])

                        updated = pd.concat([existing, new_event], ignore_index=True)
                        updated.to_csv(SCHEDULE_FILE, index=False)

                        new_df.at[new_idx, "Status"] = "Scheduled"
                        new_df.to_csv(orders_file, index=False)

                        st.success("✅ Order saved + scheduled (and status set to Scheduled).")

                st.session_state["show_new_order_form"] = False
                st.success("✅ Draw order submitted!")
                st.rerun()
# ------------------ Closed Processes Tab ------------------
elif tab_selection == "✅ Closed Processes":
    # Define the CLOSED_PROCESSES_FILE path
    CLOSED_PROCESSES_FILE = "closed_processes.csv"
    st.title("✅ Closed Processes")
    st.write("Manage products that are finalized and ready for drawing.")

    # Check if the CSV file exists and if it's empty
    if not os.path.exists(CLOSED_PROCESSES_FILE) or os.stat(CLOSED_PROCESSES_FILE).st_size == 0:
        # Define columns for the blank CSV
        columns = ["Product Name", "Furnace Temperature (°C)", "Tension (g)", "Drawing Speed (m/min)",
                   "Coating Type (Main)", "Coating Type (Secondary)", "Entry Die (Main)", "Entry Die (Secondary)",
                   "Primary Die (Main)", "Primary Die (Secondary)", "Coating Diameter (Main, µm)",
                   "Coating Diameter (Secondary, µm)", "Coating Temperature (Main, °C)",
                   "Coating Temperature (Secondary, °C)", "Fiber Diameter (µm)", "P Gain for Diameter Control",
                   "I Gain for Diameter Control", "Process Description", "Recipe Name", "Process Type", "TF Mode",
                   "TF Increment (mm)", "Core-Clad Ratio"]

        # Create the CSV with the above columns
        pd.DataFrame(columns=columns).to_csv(CLOSED_PROCESSES_FILE, index=False)
        st.warning("CSV file is empty or doesn't exist. A new blank file has been created.")

    # Load the closed processes file
    closed_df = pd.read_csv(CLOSED_PROCESSES_FILE)

    # Load the configuration from the JSON file
    with open("config_coating.json", "r") as config_file:
        config = json.load(config_file)

    # Die and coating selections
    dies = config.get("dies", {})
    coatings = config.get("coatings", {})
    process_types = ["PM", "NPM", "Other"]  # List of process types

    # Sidebar options for adding a new or updating an existing process
    action = st.sidebar.radio("Select Action", ["Add New Process", "Update Existing Process"])

    # **Add New Process**
    if action == "Add New Process":
        st.sidebar.subheader("Add New Closed Process")
        product_name = st.sidebar.text_input("Product Name")
        process_type = st.sidebar.selectbox("Process Type", process_types)  # Move Process Type here
        core_clad_ratio = st.sidebar.text_input("Core-Clad Ratio")
        furnace_temperature = st.sidebar.number_input("Furnace Temperature (°C)", min_value=0.0, step=0.1)
        tension = st.sidebar.number_input("Tension (g)", min_value=0.0, step=0.1)
        drawing_speed = st.sidebar.number_input("Drawing Speed (m/min)", min_value=0.0, step=0.1)

        # Coating Type Inputs
        coating_type_main = st.sidebar.selectbox("Coating Type (Main)", list(coatings.keys()))
        coating_type_secondary = st.sidebar.selectbox("Coating Type (Secondary)", list(coatings.keys()))

        # Die Inputs (Entry and Primary Dies)
        entry_die_main = st.sidebar.number_input("Entry Die (Main, µm)", min_value=0.0, step=0.1)
        entry_die_secondary = st.sidebar.number_input("Entry Die (Secondary, µm)", min_value=0.0, step=0.1)
        primary_die_main = st.sidebar.selectbox("Primary Die (Main)", list(dies.keys()))
        primary_die_secondary = st.sidebar.selectbox("Primary Die (Secondary)", list(dies.keys()))

        # Coating Diameter Inputs
        coating_diameter_main = st.sidebar.number_input("Coating Diameter (Main, µm)", min_value=0.0, step=0.1)
        coating_diameter_secondary = st.sidebar.number_input("Coating Diameter (Secondary, µm)", min_value=0.0,
                                                             step=0.1)

        # Coating Temperature Inputs
        coating_temperature_main = st.sidebar.number_input("Coating Temperature (Main, °C)", min_value=0.0, step=0.1)
        coating_temperature_secondary = st.sidebar.number_input("Coating Temperature (Secondary, °C)", min_value=0.0,
                                                                step=0.1)

        # Fiber Diameter and Control Inputs
        fiber_diameter = st.sidebar.number_input("Fiber Diameter (µm)", min_value=0.0, step=0.1)
        p_gain = st.sidebar.number_input("P Gain for Diameter Control", min_value=0.0, step=0.1)
        i_gain = st.sidebar.number_input("I Gain for Diameter Control", min_value=0.0, step=0.1)

        # TF Mode and Increment Inputs (Sidebar - before Description and Recipe)
        tf_mode = st.sidebar.selectbox("TF Mode", ["Winder", "Straight Mode"],
                                       index=["Winder", "Straight Mode"].index("Winder"))
        tf_increment = st.sidebar.number_input("TF Increment (mm)", min_value=0.0, step=0.01, value=0.1)

        # Process Description and Recipe Name
        process_description = st.sidebar.text_area("Process Description")
        recipe_name = st.sidebar.text_input("Recipe Name")

        if st.sidebar.button("Add New Process"):
            new_entry = pd.DataFrame([{
                "Product Name": product_name,
                "Process Type": process_type,  # User-selected process type
                "Furnace Temperature (°C)": furnace_temperature,
                "Tension (g)": tension,
                "Drawing Speed (m/min)": drawing_speed,
                "Coating Type (Main)": coating_type_main,
                "Coating Type (Secondary)": coating_type_secondary,
                "Entry Die (Main)": entry_die_main,
                "Entry Die (Secondary)": entry_die_secondary,
                "Primary Die (Main)": primary_die_main,
                "Primary Die (Secondary)": primary_die_secondary,
                "Coating Diameter (Main, µm)": coating_diameter_main,
                "Coating Diameter (Secondary, µm)": coating_diameter_secondary,
                "Coating Temperature (Main, °C)": coating_temperature_main,
                "Coating Temperature (Secondary, °C)": coating_temperature_secondary,
                "Fiber Diameter (µm)": fiber_diameter,
                "P Gain for Diameter Control": p_gain,
                "I Gain for Diameter Control": i_gain,
                "Process Description": process_description,
                "Recipe Name": recipe_name,
                "TF Mode": tf_mode,
                "TF Increment (mm)": tf_increment,
                "Core-Clad Ratio": core_clad_ratio  # New input
            }])

            # Append the new entry to the closed processes DataFrame and save it
            closed_df = pd.concat([closed_df, new_entry], ignore_index=True)
            closed_df.to_csv(CLOSED_PROCESSES_FILE, index=False)
            st.sidebar.success(f"New process '{product_name}' added successfully!")

    # **Update Existing Process**
    elif action == "Update Existing Process":
        st.sidebar.subheader("Update Existing Closed Process")
        closed_process_name = st.sidebar.selectbox("Select Process to Update", closed_df["Product Name"].tolist())

        if closed_process_name:
            matching_process = closed_df[closed_df["Product Name"] == closed_process_name]

            if not matching_process.empty:
                selected_process = matching_process.iloc[0]

                # Display current values of the closed process
                st.sidebar.write(f"Updating {closed_process_name}")
                product_name = st.sidebar.text_input("Product Name", value=selected_process["Product Name"])
                process_type = st.sidebar.selectbox("Process Type", process_types,
                                                    index=process_types.index(selected_process["Process Type"]))
                core_clad_ratio = st.sidebar.text_input("Core-Clad Ratio", value=selected_process["Core-Clad Ratio"])
                furnace_temperature = st.sidebar.number_input("Furnace Temperature (°C)", min_value=0.0, step=0.1,
                                                              value=selected_process["Furnace Temperature (°C)"])
                tension = st.sidebar.number_input("Tension (g)", min_value=0.0, step=0.1,
                                                  value=selected_process["Tension (g)"])
                drawing_speed = st.sidebar.number_input("Drawing Speed (m/min)", min_value=0.0, step=0.1,
                                                        value=selected_process["Drawing Speed (m/min)"])

                # Coating Type Inputs
                coating_type_main = st.sidebar.selectbox("Coating Type (Main)", list(coatings.keys()),
                                                         index=list(coatings.keys()).index(
                                                             selected_process["Coating Type (Main)"]))
                coating_type_secondary = st.sidebar.selectbox("Coating Type (Secondary)", list(coatings.keys()),
                                                              index=list(coatings.keys()).index(
                                                                  selected_process["Coating Type (Secondary)"]))

                # Die Inputs (Entry and Primary Dies)
                entry_die_main = st.sidebar.number_input("Entry Die (Main, µm)", min_value=0.0, step=0.1,
                                                         value=selected_process["Entry Die (Main)"])
                entry_die_secondary = st.sidebar.number_input("Entry Die (Secondary, µm)", min_value=0.0, step=0.1,
                                                              value=selected_process["Entry Die (Secondary)"])
                primary_die_main = st.sidebar.selectbox("Primary Die (Main)", list(dies.keys()),
                                                        index=list(dies.keys()).index(
                                                            selected_process["Primary Die (Main)"]))
                primary_die_secondary = st.sidebar.selectbox("Primary Die (Secondary)", list(dies.keys()),
                                                             index=list(dies.keys()).index(
                                                                 selected_process["Primary Die (Secondary)"]))

                # Coating Diameter Inputs
                coating_diameter_main = st.sidebar.number_input("Coating Diameter (Main, µm)", min_value=0.0, step=0.1,
                                                                value=selected_process["Coating Diameter (Main, µm)"])
                coating_diameter_secondary = st.sidebar.number_input("Coating Diameter (Secondary, µm)", min_value=0.0,
                                                                     step=0.1, value=selected_process[
                        "Coating Diameter (Secondary, µm)"])

                # Coating Temperature Inputs
                coating_temperature_main = st.sidebar.number_input("Coating Temperature (Main, °C)", min_value=0.0,
                                                                   step=0.1, value=selected_process[
                        "Coating Temperature (Main, °C)"])
                coating_temperature_secondary = st.sidebar.number_input("Coating Temperature (Secondary, °C)",
                                                                        min_value=0.0, step=0.1, value=selected_process[
                        "Coating Temperature (Secondary, °C)"])

                # Fiber Diameter and Control Inputs
                fiber_diameter = st.sidebar.number_input("Fiber Diameter (µm)", min_value=0.0, step=0.1,
                                                         value=selected_process["Fiber Diameter (µm)"])
                p_gain = st.sidebar.number_input("P Gain for Diameter Control", min_value=0.0, step=0.1,
                                                 value=selected_process["P Gain for Diameter Control"])
                i_gain = st.sidebar.number_input("I Gain for Diameter Control", min_value=0.0, step=0.1,
                                                 value=selected_process["I Gain for Diameter Control"])

                # TF Mode and Increment Inputs (Sidebar - before Description and Recipe)
                tf_mode = st.sidebar.selectbox("TF Mode", ["Winder", "Straight Mode"],
                                               index=["Winder", "Straight Mode"].index(selected_process["TF Mode"]))
                tf_increment = st.sidebar.number_input("TF Increment (mm)", min_value=0.0, step=0.01,
                                                       value=selected_process["TF Increment (mm)"])

                # Process Description and Recipe Name
                process_description = st.sidebar.text_area("Process Description",
                                                           value=selected_process["Process Description"])
                recipe_name = st.sidebar.text_input("Recipe Name", value=selected_process["Recipe Name"])

                if st.sidebar.button("Update Product"):
                    # Prepare updated entry with all the values
                    updated_entry = {
                        "Product Name": product_name,
                        "Process Type": process_type,
                        "Furnace Temperature (°C)": furnace_temperature,
                        "Tension (g)": tension,
                        "Drawing Speed (m/min)": drawing_speed,
                        "Coating Type (Main)": coating_type_main,
                        "Coating Type (Secondary)": coating_type_secondary,
                        "Entry Die (Main)": entry_die_main,
                        "Entry Die (Secondary)": entry_die_secondary,
                        "Primary Die (Main)": primary_die_main,
                        "Primary Die (Secondary)": primary_die_secondary,
                        "Coating Diameter (Main, µm)": coating_diameter_main,
                        "Coating Diameter (Secondary, µm)": coating_diameter_secondary,
                        "Coating Temperature (Main, °C)": coating_temperature_main,
                        "Coating Temperature (Secondary, °C)": coating_temperature_secondary,
                        "Fiber Diameter (µm)": fiber_diameter,
                        "P Gain for Diameter Control": p_gain,
                        "I Gain for Diameter Control": i_gain,
                        "Process Description": process_description,
                        "Recipe Name": recipe_name,
                        "TF Mode": tf_mode,
                        "TF Increment (mm)": tf_increment,
                        "Core-Clad Ratio": core_clad_ratio
                    }

                    # Find the index of the product name and update it
                    closed_df.loc[closed_df["Product Name"] == closed_process_name, updated_entry.keys()] = list(
                        updated_entry.values())

                    closed_df.to_csv(CLOSED_PROCESSES_FILE, index=False)
                    st.sidebar.success(f"Product '{product_name}' updated successfully!")

            else:
                st.sidebar.error(f"No process found with the name '{closed_process_name}'.")

        # Remove duplicates and display the cleaned table

    # Remove duplicates and display the cleaned table
    closed_df_clean = closed_df.drop_duplicates(
        subset=["Product Name", "Coating Type (Main)", "Coating Type (Secondary)"])

    # Reorganize the columns as requested
    closed_df_clean = closed_df_clean[[
        "Product Name", "Process Type", "Core-Clad Ratio", "Fiber Diameter (µm)",
        "Coating Diameter (Main, µm)", "Coating Diameter (Secondary, µm)", "Drawing Speed (m/min)",
        "Furnace Temperature (°C)", "Tension (g)", "TF Mode", "TF Increment (mm)"
    ]]

    st.write("### Cleaned Closed Products Table")
    st.dataframe(closed_df_clean, height=300, use_container_width=True)
# ------------------ Tower Parts Tab ------------------
elif tab_selection == "🛠️ Tower Parts":
    st.title("🛠️ Tower Parts Management")
    st.caption("Parts orders + docs (no sidebar forms)")

    ORDER_FILE = "part_orders.csv"
    archive_file = "archived_orders.csv"

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
        orders_df = pd.read_csv(ORDER_FILE)
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

    # ---------------- Projects list ----------------
    project_options = ["None"]
    try:
        projects_df = pd.read_csv(DEVELOPMENT_FILE)
        if "Project Name" in projects_df.columns:
            project_options += sorted(list(pd.Series(projects_df["Project Name"]).dropna().astype(str).unique()))
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
                    selected_project = st.selectbox("Project", project_options)
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
                            "Project",
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
                    archived_df = pd.read_csv(archive_file)
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
            archived_df = pd.read_csv(archive_file)
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
    # Parts Datasheet (Hierarchical View)
    # =========================
    st.write("### 📚 Parts Datasheet (Hierarchical View)")

    def display_directory(current_path, level=0):
        try:
            items = sorted(os.listdir(current_path))
        except Exception as e:
            st.error(f"Error accessing {current_path}: {e}")
            return

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

        if selected_folder:
            display_directory(os.path.join(current_path, selected_folder), level + 1)

        for file_path in files:
            file_name = os.path.basename(file_path)
            if st.button(f"📄 Open {file_name}", key=f"open_{file_path}"):
                os.system(f"open {file_path}")  # macOS. Linux: xdg-open, Windows: start

    if os.path.exists(PARTS_DIRECTORY) and os.listdir(PARTS_DIRECTORY):
        display_directory(PARTS_DIRECTORY)
    else:
        st.info("No parts documents found in PARTS_DIRECTORY.")
# ------------------ Development Tab ------------------
elif tab_selection == "🧪 Development Process":
    import os
    import json
    import pandas as pd
    import streamlit as st
    from datetime import datetime

    st.title("🧪 Development Process")

    # =========================
    # Files / folders
    # =========================
    PROJECTS_FILE     = "development_projects.csv"
    EXPERIMENTS_FILE  = "development_experiments.csv"
    UPDATES_FILE      = "experiment_updates.csv"
    DATASET_DIR       = "data_set_csv"
    MEDIA_ROOT        = "development_media"

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
                "Result Images",     # ; separated paths
                "Image Captions"     # JSON dict: {filename: caption}
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
                df[c] = "" if c not in ["Archived", "Is Drawing"] else False
                changed = True
        if changed:
            df.to_csv(path, index=False)

    _ensure_files()
    _ensure_columns(PROJECTS_FILE, ["Project Name", "Project Purpose", "Target", "Created At", "Archived"])
    _ensure_columns(EXPERIMENTS_FILE, [
        "Project Name", "Experiment Title", "Date", "Researcher", "Methods", "Purpose",
        "Observations", "Results", "Is Drawing", "Drawing Details", "Draw CSV",
        "Result Images", "Image Captions"
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
        if "Result Images" not in df.columns:
            df["Result Images"] = ""
        if "Image Captions" not in df.columns:
            df["Image Captions"] = ""
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

    def parse_img_list(s):
        if not isinstance(s, str) or not s.strip():
            return []
        return [x for x in s.split(";") if x.strip()]

    def join_img_list(lst):
        return ";".join(lst)

    def parse_captions(s):
        if not isinstance(s, str) or not s.strip():
            return {}
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except:
            return {}

    def dump_captions(d):
        try:
            return json.dumps(d, ensure_ascii=False)
        except:
            return ""

    def list_dataset_csvs_newest_first():
        if not os.path.isdir(DATASET_DIR):
            return []
        files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
        return sorted(files, key=lambda fn: os.path.getmtime(os.path.join(DATASET_DIR, fn)), reverse=True)

    # =========================
    # Session defaults
    # =========================
    if "dev_view_mode_main" not in st.session_state:
        st.session_state["dev_view_mode_main"] = "Active"
    if "dev_selected_project" not in st.session_state:
        st.session_state["dev_selected_project"] = ""
    if "dev_show_add_experiment" not in st.session_state:
        st.session_state["dev_show_add_experiment"] = False

    # =========================
    # Top bar UI
    # =========================
    projects_df = load_projects()
    top1, top2, top3, top4 = st.columns([1.15, 1.15, 1.25, 2.8])

    # ---- Add Project ----
    with top1.popover("➕ Add New Project"):
        with st.form("add_project_pop_form", clear_on_submit=True):
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
                st.rerun()

    # ---- Select Project ----
    with top2.popover("📂 Select Project"):
        view_mode = st.radio("View", ["Active", "Archived"], horizontal=True, key="dev_view_mode_main")
        projects_df = load_projects()
        filtered = projects_df[projects_df["Archived"] == (view_mode == "Archived")]
        options = [""] + filtered["Project Name"].dropna().unique().tolist()

        st.selectbox("Choose a Project", options, key="dev_selected_project")
        if st.session_state.get("dev_selected_project"):
            st.success(f"Selected: {st.session_state['dev_selected_project']}")

    selected_project = st.session_state.get("dev_selected_project", "")

    # ---- Manage Project ----
    with top3.popover("📦 Manage Project"):
        if not selected_project:
            st.info("Select a project first.")
        else:
            projects_df = load_projects()
            row = projects_df[projects_df["Project Name"] == selected_project]
            if row.empty:
                st.warning("Project not found (maybe deleted).")
            else:
                is_archived = bool(row.iloc[0].get("Archived", False))
                cA, cB = st.columns(2)

                if not is_archived:
                    if cA.button("🗄️ Archive", use_container_width=True):
                        projects_df.loc[projects_df["Project Name"] == selected_project, "Archived"] = True
                        save_projects(projects_df)
                        st.success("Archived.")
                        st.rerun()
                else:
                    if cA.button("♻️ Restore", use_container_width=True):
                        projects_df.loc[projects_df["Project Name"] == selected_project, "Archived"] = False
                        save_projects(projects_df)
                        st.success("Restored.")
                        st.rerun()

                if cB.button("🗑️ Delete", use_container_width=True):
                    exp_df = load_experiments()
                    upd_df = load_updates()

                    projects_df = projects_df[projects_df["Project Name"] != selected_project]
                    exp_df = exp_df[exp_df["Project Name"] != selected_project]
                    upd_df = upd_df[upd_df["Project Name"] != selected_project]

                    save_projects(projects_df)
                    save_experiments(exp_df)
                    save_updates(upd_df)

                    st.session_state["dev_selected_project"] = ""
                    st.warning("Deleted permanently.")
                    st.rerun()

    # ---- Fold / open Add Experiment ----
    with top4:
        if selected_project:
            label = "➖ Hide Add Experiment" if st.session_state["dev_show_add_experiment"] else "➕ Add Experiment"
            if st.button(label, use_container_width=True):
                st.session_state["dev_show_add_experiment"] = not st.session_state["dev_show_add_experiment"]
        else:
            st.caption("Select a project to add experiments")

    # =========================
    # Main content
    # =========================
    if not selected_project:
        st.info("Use **📂 Select Project** to start.")
        st.stop()

    projects_df = load_projects()
    proj_row = projects_df[projects_df["Project Name"] == selected_project]
    if proj_row.empty:
        st.warning("Selected project not found.")
        st.stop()

    proj = proj_row.iloc[0]
    st.info(f"📌 Current project: **{selected_project}**")
    st.subheader("📌 Project Details")
    st.markdown(f"**Project Purpose:** {proj.get('Project Purpose', 'N/A')}")
    st.markdown(f"**Target:** {proj.get('Target', 'N/A')}")
    st.caption(f"Created at: {proj.get('Created At', '')} | Archived: {bool(proj.get('Archived', False))}")

    st.divider()

    # =========================
    # Add Experiment (LIVE drawing UI + newest CSV first + image descriptions)
    # =========================
    if st.session_state.get("dev_show_add_experiment", False):
        with st.expander("➕ Add Experiment", expanded=True):

            # ---- Live drawing toggle OUTSIDE the form ----
            is_drawing_live = st.checkbox("Is this a Drawing?", key=f"newexp_is_drawing__{selected_project}")

            drawing_details_live = ""
            draw_csv_live = ""

            if is_drawing_live:
                drawing_details_live = st.text_area(
                    "Drawing Details",
                    height=90,
                    key=f"newexp_drawing_details__{selected_project}"
                )

                dataset_files = list_dataset_csvs_newest_first()
                if not dataset_files:
                    st.info("No CSV files found in data_set_csv/")
                else:
                    newest = dataset_files[0]
                    st.caption(f"Newest CSV: **{newest}**")

                    # default to newest (index=1 because first option is "")
                    draw_csv_live = st.selectbox(
                        "Select Draw CSV (newest first)",
                        [""] + dataset_files,
                        index=1,
                        key=f"newexp_draw_csv__{selected_project}"
                    )

            st.divider()

            # ---- Upload images + descriptions OUTSIDE the form (so it shows immediately) ----
            st.markdown("### 📷 Attach photos now (optional)")
            uploaded_new_imgs = st.file_uploader(
                "Drag & drop images (png/jpg/webp)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key=f"newexp_imgs__{selected_project}"
            )

            caption_inputs = {}
            if uploaded_new_imgs:
                st.markdown("### 📝 Photo descriptions")
                st.caption("These descriptions will be shown under each photo.")
                for f in uploaded_new_imgs:
                    caption_inputs[f.name] = st.text_area(
                        f"Description for {f.name}",
                        height=80,
                        key=f"newexp_caption__{selected_project}__{f.name}"
                    )

            st.divider()

            # ---- Form for experiment fields + Save button ----
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
                        saved_img_paths = []
                        captions_map = {}

                        if uploaded_new_imgs:
                            media_dir = exp_media_dir(selected_project, experiment_title.strip(), exp_date_str)
                            for f in uploaded_new_imgs:
                                out_path = os.path.join(media_dir, f.name)
                                try:
                                    with open(out_path, "wb") as w:
                                        w.write(f.getbuffer())
                                    saved_img_paths.append(out_path)

                                    # ✅ captions keyed by filename (robust)
                                    captions_map[os.path.basename(out_path)] = (caption_inputs.get(f.name, "") or "").strip()
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
                            "Result Images": join_img_list(saved_img_paths) if saved_img_paths else "",
                            "Image Captions": dump_captions(captions_map) if captions_map else ""
                        }])

                        exp_df = pd.concat([exp_df, new_exp], ignore_index=True)
                        save_experiments(exp_df)

                        st.success(f"Experiment saved. Images attached: {len(saved_img_paths)}")
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
                st.write(f"**Researcher:** {exp.get('Researcher', 'N/A')}")
                st.write(f"**Methods:** {exp.get('Methods', 'N/A')}")
                st.write(f"**Purpose:** {exp.get('Purpose', 'N/A')}")
                st.write(f"**Observations:** {exp.get('Observations', 'N/A')}")
                st.write(f"**Results:** {exp.get('Results', 'N/A')}")

                # ---- Drawing
                if bool(exp.get("Is Drawing", False)):
                    st.markdown("#### 🧵 Drawing")
                    st.write(f"**Drawing Details:** {exp.get('Drawing Details', '')}")

                    draw_csv_name = str(exp.get("Draw CSV", "")).strip()
                    if draw_csv_name:
                        st.write(f"**Draw CSV:** `{draw_csv_name}`")
                        csv_path = os.path.join(DATASET_DIR, draw_csv_name)
                        if os.path.exists(csv_path):
                            if st.button("📄 Load & View Draw CSV", key=f"load_draw__{expander_key}"):
                                try:
                                    df_draw = load_draw_csv(csv_path)
                                    st.dataframe(df_draw, use_container_width=True, height=320)
                                except Exception as e:
                                    st.error(f"Failed to load draw CSV: {e}")
                        else:
                            st.warning("Draw CSV file not found in data_set_csv/")
                    else:
                        st.info("No draw CSV attached to this drawing experiment.")

                st.divider()

                # ---- Images + captions
                st.markdown("#### 🖼️ Results Images")

                existing_imgs = parse_img_list(exp.get("Result Images", ""))
                captions_map = parse_captions(exp.get("Image Captions", ""))

                if existing_imgs:
                    captions_list = []
                    for p in existing_imgs:
                        fn = os.path.basename(p)
                        cap = (captions_map.get(fn, "") or "").strip()
                        captions_list.append(cap if cap else fn)

                    st.image(existing_imgs, caption=captions_list, use_container_width=True)
                else:
                    st.caption("No images saved yet.")

                # ---- Add more images + descriptions
                st.markdown("**➕ Add more images**")
                media_dir = exp_media_dir(selected_project, exp_title, exp_date)

                uploaded_imgs = st.file_uploader(
                    "Drop more images here (png/jpg/webp)",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key=f"img_uploader__{expander_key}"
                )

                new_caps = {}
                if uploaded_imgs:
                    st.markdown("### 📝 Descriptions for new images")
                    for f in uploaded_imgs:
                        new_caps[f.name] = st.text_area(
                            f"Description for {f.name}",
                            height=80,
                            key=f"more_caption__{expander_key}__{f.name}"
                        )

                    if st.button("💾 Save new images", key=f"save_more_imgs__{expander_key}"):
                        new_paths = []
                        for f in uploaded_imgs:
                            out_path = os.path.join(media_dir, f.name)
                            try:
                                with open(out_path, "wb") as w:
                                    w.write(f.getbuffer())
                                new_paths.append(out_path)

                                # ✅ captions keyed by filename
                                captions_map[os.path.basename(out_path)] = (new_caps.get(f.name, "") or "").strip()
                            except Exception as e:
                                st.error(f"Failed saving {f.name}: {e}")

                        if new_paths:
                            exp_df2 = load_experiments()
                            mask = (
                                (exp_df2["Project Name"] == selected_project) &
                                (exp_df2["Experiment Title"].astype(str) == exp_title) &
                                (exp_df2["Date"].astype(str) == exp_date)
                            )
                            merged = existing_imgs + new_paths
                            exp_df2.loc[mask, "Result Images"] = join_img_list(merged)
                            exp_df2.loc[mask, "Image Captions"] = dump_captions(captions_map)
                            save_experiments(exp_df2)

                            st.success(f"Saved {len(new_paths)} image(s).")
                            st.rerun()

                st.divider()

                # ---- Updates history
                upd_df = load_updates()
                exp_updates = upd_df[
                    (upd_df["Project Name"] == selected_project) &
                    (upd_df["Experiment Title"] == exp_title)
                ].copy()

                st.markdown("#### 📜 Progress Updates")
                if exp_updates.empty:
                    st.caption("No updates yet.")
                else:
                    exp_updates["Update_sort"] = pd.to_datetime(exp_updates["Update Date"], errors="coerce")
                    exp_updates = exp_updates.sort_values("Update_sort", ascending=True)
                    for _, u in exp_updates.iterrows():
                        st.write(f"📅 **{u.get('Update Date','')}** — **{u.get('Researcher','')}**: {u.get('Update Notes','')}")

                # ---- Add update
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

    # =========================
    # Project conclusion
    # =========================
    st.subheader("📢 Project Conclusion")
    conclusion_file = f"project_conclusion__{selected_project.replace(' ', '_')}.txt"

    existing = ""
    if os.path.exists(conclusion_file):
        try:
            existing = open(conclusion_file, "r", encoding="utf-8").read()
        except:
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
    st.title("📋 Protocols")
    st.subheader("Manage Tower Protocols")

    PROTOCOLS_FILE = "protocols.json"

    # Load protocols from file if they exist
    if os.path.exists(PROTOCOLS_FILE):
        with open(PROTOCOLS_FILE, "r") as file:
            st.session_state["protocols"] = json.load(file)

    if "protocols" not in st.session_state:
        st.session_state["protocols"] = []

    # Ensure each protocol has a 'sub_type' key
    for protocol in st.session_state["protocols"]:
        if "sub_type" not in protocol:
            protocol["sub_type"] = "Instructions"  # Default to "Instructions" if not provided


    # Organize protocols by type and display each type as a subtitle with its protocols listed below
    protocol_types = ["Drawings", "Maintenance", "Tower Regular Operations"]
    for protocol_type in protocol_types:
        st.subheader(f"Protocols for {protocol_type}")
        filtered_protocols = [p for p in st.session_state["protocols"] if p.get("type") == protocol_type]
        if filtered_protocols:
            for protocol in filtered_protocols:
                with st.expander(protocol["name"]):
                    st.write(f"Type: {protocol['type']}")
                    if protocol["sub_type"] == "Checklist":
                        checklist_items = [item.strip() for item in protocol["instructions"].split("\n") if item.strip()]
                        if checklist_items:
                            # Provide a unique key for each checkbox
                            checkbox_values = [st.checkbox(item, key=f"{protocol['name']}_{item}") for item in checklist_items]
                            if all(checkbox_values):
                                st.success(f"All items in {protocol['name']} checklist are completed!")
                        else:
                            st.info("No checklist items available.")
                    else:
                        st.markdown("Instructions:\n" + protocol["instructions"].replace("\n", "  \n"))

                    # Individual delete button for protocol
                    delete_button = st.button(f"Delete {protocol['name']}", key=f"delete_{protocol['name']}")
                    if delete_button:
                        st.session_state["protocols"].remove(protocol)
                        # Save the updated protocols list to file
                        with open(PROTOCOLS_FILE, "w") as file:
                            json.dump(st.session_state["protocols"], file, indent=4)
                        st.success(f"Protocol '{protocol['name']}' deleted successfully!")
                        st.rerun()  # Immediately refresh the list
        else:
            st.warning(f"No protocols available for {protocol_type}")

    st.markdown("---")

    # Protocol creation section
    create_new = st.checkbox("Create New Protocol", key="create_new_protocol_checkbox")
    if create_new:
        with st.form(key="new_protocol_form"):
            protocol_name = st.text_input("Enter Protocol Name")
            protocol_type = st.selectbox("Select Protocol Type", protocol_types, key="protocol_type_select_create")

            checklist_or_instructions = st.selectbox("Select Protocol Sub-Type", ["Checklist", "Instructions"],
                                                     key="checklist_or_instructions_create")

            protocol_instructions = st.text_area("Enter Protocol Instructions")
            submit_button = st.form_submit_button(label="Add Protocol")
            if submit_button:
                if protocol_name and protocol_instructions:
                    new_protocol = {"name": protocol_name, "type": protocol_type, "sub_type": checklist_or_instructions,
                                    "instructions": protocol_instructions}
                    st.session_state["protocols"].append(new_protocol)
                    # Save updated protocols list to file
                    with open(PROTOCOLS_FILE, "w") as file:
                        json.dump(st.session_state["protocols"], file, indent=4)
                    # Immediately update the protocols list without page refresh
                    st.session_state["protocols"] = st.session_state["protocols"]
                    st.success(f"Protocol '{protocol_name}' added successfully!")
                    st.rerun()  # Immediately refresh the list
                else:
                    st.error("Please fill out all fields.")
    st.markdown("---")
    # Now display the delete checkbox and delete button at the end of the page
    delete_mode = st.checkbox("Delete Protocols", key="delete_mode")
    if delete_mode:
        protocol_names = [protocol["name"] for protocol in st.session_state["protocols"]]
        selected_for_deletion = st.multiselect("Select protocols to delete", options=protocol_names, key="protocols_to_delete")
        if selected_for_deletion:
            if st.button("Delete Selected Protocols"):
                st.session_state["protocols"] = [protocol for protocol in st.session_state["protocols"] if protocol["name"] not in selected_for_deletion]
                with open(PROTOCOLS_FILE, "w") as file:
                    json.dump(st.session_state["protocols"], file, indent=4)
                st.success(f"Deleted {len(selected_for_deletion)} protocol(s) successfully!")
                st.rerun()
# ------------------ Maintenance Tab ------------------
elif tab_selection == "🧰 Maintenance":
    import os, sys, json, pathlib, subprocess, time, hashlib
    import datetime as dt

    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    import duckdb

    # =========================================================
    # Persistent DuckDB (shared with SQL Lab)
    # =========================================================
    DB_PATH = os.path.join(os.getcwd(), "tower.duckdb")
    if "tower_con" not in st.session_state:
        st.session_state.tower_con = duckdb.connect(DB_PATH)
    con = st.session_state.tower_con

    st.title("🧰 Maintenance")
    st.caption(
        "Auto-loads ALL maintenance files from /maintenance. "
        "Furnace + UV1 + UV2 hours are persisted to maintenance/_app_state.json. "
        "New draw detection shows an inline Pre/Post checklist."
    )

    # =========================================================
    # Paths
    # =========================================================
    BASE_DIR = os.getcwd()
    MAINT_FOLDER = os.path.join(BASE_DIR, "maintenance")
    DRAW_FOLDER = os.path.join(BASE_DIR, "data_set_csv")
    STATE_PATH = os.path.join(MAINT_FOLDER, "_app_state.json")
    os.makedirs(MAINT_FOLDER, exist_ok=True)

    # =========================================================
    # Create DB tables (tasks snapshot + actions log)
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
    # Routine split (Tracking_Mode == event)
    # =========================================================
    def pick_pre_post_tasks(maintenance_df: pd.DataFrame):
        df = maintenance_df.copy()
        for c in ["Task", "Notes", "Procedure_Summary", "Tracking_Mode", "Component", "Source_File", "Task_ID"]:
            if c not in df.columns:
                df[c] = ""
            df[c] = df[c].fillna("").astype(str)

        routine = df[df["Tracking_Mode"].str.strip().str.lower() == "event"].copy()

        text = (routine["Task"] + " " + routine["Notes"] + " " + routine["Procedure_Summary"]).str.lower()
        is_pre = text.str.contains(r"\bpre\b|\bbefore\b|\bstartup\b|\bstart up\b|\bstart-up\b")
        is_post = text.str.contains(r"\bpost\b|\bafter\b|\bshutdown\b|\bshut down\b|\bshut-down\b|\bend\b")

        pre = routine[is_pre].copy()
        post = routine[is_post].copy()
        other = routine[~(is_pre | is_post)].copy()
        pre = pd.concat([pre, other], ignore_index=True)

        return pre, post

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
        "tracking mode": "Tracking_Mode",       # calendar / hours / event / draws
        "hours source": "Hours_Source",         # FURNACE / UV1 / UV2
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
        # mirror for home tab
        st.session_state["furnace_hours"] = state["furnace_hours"]
        st.session_state["uv1_hours"] = state["uv1_hours"]
        st.session_state["uv2_hours"] = state["uv2_hours"]

    default_furnace = float(state.get("furnace_hours", 0.0) or 0.0)
    default_uv1 = float(state.get("uv1_hours", 0.0) or 0.0)
    default_uv2 = float(state.get("uv2_hours", 0.0) or 0.0)
    default_warn_days = int(state.get("warn_days", 14) or 14)
    default_warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

    st.subheader("Current status inputs (saved)")
    # Only show the disabled date_input for today's date, no "Real today" or "Saved date" UI.
    current_date = dt.date.today()
    st.date_input(
        "Today",
        value=current_date,
        disabled=True
    )
    c2, c3, c4, c5 = st.columns([1, 1, 1, 1])
    with c2:
        furnace_hours = st.number_input("Furnace hours", min_value=0.0, value=default_furnace, step=1.0,
                                        key="maint_furnace_hours", on_change=_persist_inputs)
    with c3:
        uv1_hours = st.number_input("UV System 1 hours", min_value=0.0, value=default_uv1, step=1.0,
                                    key="maint_uv1_hours", on_change=_persist_inputs)
    with c4:
        uv2_hours = st.number_input("UV System 2 hours", min_value=0.0, value=default_uv2, step=1.0,
                                    key="maint_uv2_hours", on_change=_persist_inputs)
    with c5:
        warn_days = st.number_input("Warn if due within (days)", min_value=0, value=default_warn_days, step=1,
                                    key="maint_warn_days", on_change=_persist_inputs)

    warn_hours = st.number_input("Warn if due within (hours)", min_value=0.0, value=default_warn_hours, step=1.0,
                                 key="maint_warn_hours", on_change=_persist_inputs)

    _persist_inputs()

    st.caption("Hours-based tasks use **Hours Source**: FURNACE / UV1 / UV2. If empty → defaults to FURNACE.")

    # =========================================================
    # Actor (fixed session_state usage)
    # =========================================================
    if "maint_actor" not in st.session_state:
        st.session_state["maint_actor"] = "operator"

    st.text_input(
        "Actor / operator name (for DB history)",
        key="maint_actor"
    )
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

    # Publish quick counts for Home tab
    st.session_state["maint_overdue"] = int((dfm["Status"] == "OVERDUE").sum())
    st.session_state["maint_due_soon"] = int((dfm["Status"] == "DUE SOON").sum())

    # =========================================================
    # Sync "maintenance_tasks" snapshot into DuckDB
    # =========================================================
    render_maintenance_tasks_snapshot(dfm, con)

    # =========================================================
    # INLINE "NEW DRAW DETECTED" PANEL
    # =========================================================
    render_new_draw_checklist(
        dfm,
        current_draw_count,
        state,
        STATE_PATH,
        save_state,
    )

    # =========================================================
    # Dashboard summary
    # =========================================================
    render_maintenance_dashboard_metrics(dfm)

    # =========================================================
    # FUTURE TIMELINE VIEW (BUTTONS)
    # =========================================================
    horizon_hours, horizon_days, horizon_draws = render_maintenance_horizon_selector(current_draw_count)

    # =========================================================
    # Roadmap Plotly helpers
    # =========================================================
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

    # =========================================================
    # Mark Done (updates source Excel/CSV) + LOG into DuckDB
    # =========================================================
    edited = render_maintenance_done_editor(dfm)

    render_maintenance_apply_done(
        edited,
        dfm=dfm,
        current_date=current_date,
        current_draw_count=current_draw_count,
        actor=actor,
        MAINT_FOLDER=MAINT_FOLDER,
        STATE_PATH=STATE_PATH,
        con=con,
        read_file=read_file,
        write_file=write_file,
        normalize_df=normalize_df,
        templateize_df=templateize_df,
        pick_current_hours=pick_current_hours,
        mode_norm=mode_norm,
    )

    # =========================================================
    # DB viewer (dates displayed clean)
    # =========================================================
    render_maintenance_history(con)
    # =========================================================
    # Load report
    # =========================================================
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
        "Auto-computes correlation between numeric columns across logs over time, "
        "incrementally cached. Flags correlation-break outliers."
    )

    BASE_DIR = os.getcwd()
    LOGS_FOLDER = os.path.join(BASE_DIR, "logs")
    MAINT_FOLDER = os.path.join(BASE_DIR, "maintenance")
    os.makedirs(MAINT_FOLDER, exist_ok=True)

    render_corr_outliers_tab(DRAW_FOLDER=LOGS_FOLDER, MAINT_FOLDER=MAINT_FOLDER)
# ------------------ SQL Lab ------------------
elif tab_selection == "🧪 SQL Lab":
    import os, glob, re
    import duckdb
    import pandas as pd
    import numpy as np
    import streamlit as st
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # =========================================================
    # 🧪 SQL Lab – Draws + DONE Maintenance (refactored)
    # =========================================================
    st.title("🧪 SQL Lab – One filter for Draws + DONE Maintenance")
    st.caption(
        "Maintenance shows ONLY DONE events, searchable by the **maintenance Excel/CSV file name**. "
        "Filter → Run → Visual Lab → Math Lab."
    )

    BASE_DIR = os.getcwd()
    DATASET_DIR = os.path.join(BASE_DIR, "data_set_csv")
    DB_PATH = os.path.join(BASE_DIR, "tower.duckdb")

    # =========================================================
    # Persistent DuckDB
    # =========================================================
    if "tower_con" not in st.session_state:
        st.session_state.tower_con = duckdb.connect(DB_PATH)
    con = st.session_state.tower_con

    # =========================================================
    # Helpers
    # =========================================================
    def esc(s): return (s or "").replace("'", "''")
    def lit(s): return "'" + esc(str(s)) + "'"

    def is_num(x):
        try:
            float(str(x))
            return True
        except Exception:
            return False

    def fmt_num(x, nd=3):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        try:
            xf = float(x)
            if abs(xf) >= 1000: return f"{xf:,.0f}"
            if abs(xf) >= 100:  return f"{xf:,.1f}"
            if abs(xf) >= 10:   return f"{xf:,.2f}"
            return f"{xf:,.{nd}f}"
        except Exception:
            return str(x)

    def _mtime(path):
        try:
            if isinstance(path, str) and path.strip():
                return pd.to_datetime(os.path.getmtime(path), unit="s")
        except Exception:
            pass
        return pd.NaT

    # =========================================================
    # Ensure maintenance_actions exists
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

    # =========================================================
    # Views builder (keep your schema: datasets_kv has _draw/_file etc.)
    # =========================================================
    def build_datasets_view_from_disk():
        files = glob.glob(os.path.join(DATASET_DIR, "**", "*.csv"), recursive=True)
        files = [f for f in files if os.path.isfile(f)]

        if not files:
            con.execute("""
                CREATE OR REPLACE VIEW datasets_kv AS
                SELECT
                    'dataset'::VARCHAR AS source_kind,
                    NULL::VARCHAR AS source_file,
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
        files_sql = "[" + ",".join(lit(f) for f in files) + "]"

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
                NULL::VARCHAR AS source_file,
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

    def build_maintenance_kv_view():
        con.execute("""
            CREATE OR REPLACE VIEW maintenance_kv AS
            SELECT
                'maintenance'::VARCHAR AS source_kind,
                COALESCE(source_file,'')::VARCHAR AS source_file,
                action_ts AS event_ts,
                CAST(action_id AS VARCHAR) AS event_id,
                NULL::VARCHAR AS _draw,
                NULL::VARCHAR AS _file,
                NULL::VARCHAR AS filename,
                COALESCE(source_file,'')::VARCHAR AS "Parameter Name",
                (
                    trim(COALESCE(component,'') || ' — ' || COALESCE(task,'')) ||
                    CASE WHEN COALESCE(note,'') <> '' THEN (' | ' || note) ELSE '' END
                )::VARCHAR AS "Value",
                ''::VARCHAR AS "Units"
            FROM maintenance_actions
            WHERE action_ts IS NOT NULL
              AND source_file IS NOT NULL
              AND trim(source_file) <> '';
        """)

    def build_all_kv_views():
        n = build_datasets_view_from_disk()
        build_maintenance_kv_view()
        con.execute("""
            CREATE OR REPLACE VIEW all_kv AS
            SELECT
                source_kind, source_file, event_ts, event_id, _draw, _file, filename, "Parameter Name", "Value", "Units"
            FROM datasets_kv
            UNION ALL
            SELECT
                source_kind, source_file, event_ts, event_id, _draw, _file, filename, "Parameter Name", "Value", "Units"
            FROM maintenance_kv;
        """)
        return n

    # =========================================================
    # Render: indexing controls
    # =========================================================
    def render_indexing_controls():
        st.subheader("📁 Source indexing")
        cA, cB = st.columns(2)

        if cA.button("🔄 Rebuild views", use_container_width=True, key="sql_rebuild_all"):
            for k in ["sql_df_all", "sql_events_wide", "ds_conditions"]:
                st.session_state.pop(k, None)
            n = build_all_kv_views()
            st.success(f"Rebuilt. Loaded {n} dataset CSV(s). Maintenance DONE events available if logged.")

        if cB.button("🧹 Reset DB connection", use_container_width=True, key="sql_reset_con"):
            try:
                con.close()
            except Exception:
                pass
            st.session_state.tower_con = duckdb.connect(DB_PATH)
            for k in list(st.session_state.keys()):
                if k.startswith("sql_") or k.startswith("math_") or k.startswith("param_") or k.startswith("time_"):
                    st.session_state.pop(k, None)
            st.success("DB connection reset.")
            st.stop()

        # Ensure views exist
        try:
            con.execute("SELECT COUNT(*) FROM all_kv").fetchone()
        except Exception:
            n = build_all_kv_views()
            st.info(f"Auto-built views. Loaded {n} dataset CSV(s).")

    # =========================================================
    # Render: filter builder
    # =========================================================
    def render_filter_builder():
        params_df = con.execute("""
            SELECT DISTINCT "Parameter Name"
            FROM all_kv
            WHERE "Parameter Name" IS NOT NULL AND trim("Parameter Name") <> ''
            ORDER BY 1
        """).fetchdf()

        all_params = params_df["Parameter Name"].astype(str).tolist() if not params_df.empty else []
        if not all_params:
            st.warning("No parameters found.")
            st.stop()

        st.subheader("🧱 Filter conditions")

        if "ds_conditions" not in st.session_state:
            st.session_state.ds_conditions = []

        param_search = st.text_input(
            "Search parameter (live filter)",
            placeholder="Type… e.g. diameter, tension, Capstan, UV, furnace…",
            key="param_search",
        )
        filt = (param_search or "").strip().lower()
        shown_params = [p for p in all_params if filt in p.lower()] if filt else all_params

        c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
        p = c1.selectbox("Parameter Name", shown_params, key="sql_param_name")
        op = c2.selectbox("Op", ["any", "=", "!=", ">", ">=", "<", "<=", "between", "contains"], key="sql_op")
        v1 = c3.text_input("Value 1", key="sql_v1")
        v2 = c4.text_input("Value 2 (between)", key="sql_v2")
        joiner = st.selectbox("Join with", ["AND", "OR"], key="sql_joiner")

        st.markdown("#### 🗓️ Time filter (optional)")
        time_on = st.checkbox("Enable time filter", value=False, key="time_filter_on")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            t_start = st.date_input("From", value=None, key="time_filter_start")
        with tcol2:
            t_end = st.date_input("To", value=None, key="time_filter_end")

        def build_cond(p, op, v1, v2):
            base = f"\"Parameter Name\"={lit(p)}"
            v1 = (v1 or "").strip()
            v2 = (v2 or "").strip()

            if op == "any":
                return f"({base})"

            if op == "contains":
                if not v1:
                    return None
                return f"({base} AND CAST(\"Value\" AS VARCHAR) ILIKE '%{esc(v1)}%')"

            if op == "between":
                if not v1 or not v2:
                    return None
                if is_num(v1) and is_num(v2):
                    return f"({base} AND TRY_CAST(\"Value\" AS DOUBLE) BETWEEN {v1} AND {v2})"
                return f"({base} AND \"Value\" BETWEEN {lit(v1)} AND {lit(v2)})"

            if not v1:
                return None

            if is_num(v1):
                return f"({base} AND TRY_CAST(\"Value\" AS DOUBLE) {op} {v1})"
            return f"({base} AND \"Value\" {op} {lit(v1)})"

        b1, b2, b3 = st.columns(3)
        if b1.button("➕ Add condition", use_container_width=True, key="sql_add"):
            cond = build_cond(p, op, v1, v2)
            if not cond:
                st.warning("Condition not complete.")
            else:
                if st.session_state.ds_conditions:
                    st.session_state.ds_conditions.append(f"{joiner} {cond}")
                else:
                    st.session_state.ds_conditions.append(cond)

        if b2.button("↩ Remove last", use_container_width=True, key="sql_pop") and st.session_state.ds_conditions:
            st.session_state.ds_conditions.pop()

        if b3.button("🧹 Clear", use_container_width=True, key="sql_clear"):
            st.session_state.ds_conditions = []

        where_parts = []
        if st.session_state.ds_conditions:
            where_parts.append("\n  ".join(st.session_state.ds_conditions))

        if time_on and t_start and t_end:
            start_ts = pd.Timestamp(t_start)
            end_ts = pd.Timestamp(t_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            where_parts.append(f"(event_ts BETWEEN {lit(start_ts)} AND {lit(end_ts)})")

        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + "\n  AND ".join([f"({w})" for w in where_parts])

        st.code(where_sql or "-- no WHERE", language="sql")
        return where_sql

    # =========================================================
    # Render: Run filter
    # =========================================================
    def render_run_filter(where_sql: str):
        st.subheader("▶ Run filter")

        sql = f"""
        SELECT
            source_kind,
            source_file,
            event_ts,
            event_id,
            _draw,
            filename,
            "Parameter Name",
            "Value",
            "Units"
        FROM all_kv
        {where_sql}
        ORDER BY COALESCE(event_ts, TIMESTAMP '1900-01-01') ASC, source_kind, event_id, "Parameter Name";
        """
        st.code(sql, language="sql")

        if st.button("▶ Run filter", use_container_width=True, key="sql_run"):
            df = con.execute(sql).fetchdf()
            st.session_state.sql_df_all = df
            st.session_state.pop("sql_events_wide", None)
            st.success(f"{len(df):,} rows matched")
            st.dataframe(df, use_container_width=True, height=380)

        if "sql_df_all" not in st.session_state:
            st.stop()

        df_all = st.session_state.sql_df_all
        if df_all.empty:
            st.warning("No rows matched.")
            st.stop()

        return df_all

    # =========================================================
    # Render: Visual Lab (FIXED)
    # - Always pull FULL draw params for matched draw_ids using event_id
    # - Maintenance plotted as vertical lines (no y=1 dots)
    # =========================================================
    # =========================================================
    # Visual Lab (events over time) + CLICK to inspect
    # =========================================================
    import plotly.graph_objects as go


    def _mtime(path):
        try:
            if isinstance(path, str) and path.strip():
                return pd.to_datetime(os.path.getmtime(path), unit="s")
        except Exception:
            pass
        return pd.NaT



    def build_visual_dataset_plus_maint(
        con,
        df_all: pd.DataFrame,
        use_time_fallback: bool = True,
        include_nearby_draws_when_maintenance_only: bool = True,
        nearby_pad_days: int = 7,
    ):
        """Build a KV + WIDE dataset for plotting.

        - If the filter matched dataset rows, we pull full KV for those matched draw ids.
        - If the filter matched ONLY maintenance rows, we can (optionally) also pull nearby draws
          by time window (±pad days around the matched maintenance times, or around the time filter).

        Returns:
          d_kv: KV rows for plotting (full draw params + matched maintenance events)
          wide: wide table indexed by event
        """
        df_f = df_all.copy()
        df_f["event_ts"] = pd.to_datetime(df_f.get("event_ts"), errors="coerce")

        # 1) matched draw ids from filter (if any)
        ds_rows = df_f.loc[df_f.get("source_kind").astype(str).eq("dataset")].copy() if "source_kind" in df_f.columns else pd.DataFrame()
        matched_draw_ids: list[str] = []
        if not ds_rows.empty:
            if "_draw" in ds_rows.columns:
                matched_draw_ids = ds_rows["_draw"].dropna().astype(str).tolist()
            if (not matched_draw_ids) and "event_id" in ds_rows.columns:
                matched_draw_ids = ds_rows["event_id"].dropna().astype(str).tolist()
            matched_draw_ids = sorted(list(dict.fromkeys([x.strip() for x in matched_draw_ids if str(x).strip()])))

        # 2) matched maintenance rows from filter
        df_maint_matched = df_f.loc[df_f.get("source_kind").astype(str).eq("maintenance")].copy() if "source_kind" in df_f.columns else pd.DataFrame()

        # 3) If filter matched only maintenance -> optionally pull nearby draws
        draw_ids_to_pull: list[str] = list(matched_draw_ids)
        if (not matched_draw_ids) and (not df_maint_matched.empty) and include_nearby_draws_when_maintenance_only:
            window_start = None
            window_end = None

            # If user enabled the time filter, use that exact window
            if st.session_state.get("time_filter_on", False) and st.session_state.get("time_filter_start") and st.session_state.get("time_filter_end"):
                window_start = pd.Timestamp(st.session_state.get("time_filter_start"))
                window_end = pd.Timestamp(st.session_state.get("time_filter_end")) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            else:
                maint_times = pd.to_datetime(df_maint_matched["event_ts"], errors="coerce").dropna()
                if not maint_times.empty:
                    window_start = maint_times.min() - pd.Timedelta(days=int(nearby_pad_days))
                    window_end = maint_times.max() + pd.Timedelta(days=int(nearby_pad_days))

            if window_start is not None and window_end is not None:
                try:
                    cand = con.execute(f"""
                        SELECT DISTINCT event_id
                        FROM datasets_kv
                        WHERE event_ts BETWEEN {lit(window_start)} AND {lit(window_end)}
                    """).fetchdf()
                    if cand is not None and not cand.empty:
                        draw_ids_to_pull = cand["event_id"].dropna().astype(str).unique().tolist()
                except Exception:
                    draw_ids_to_pull = []

                # Optional fallback using mtime if event_ts missing in datasets
                if use_time_fallback:
                    try:
                        all_draws = con.execute("""
                            SELECT DISTINCT event_id, filename, event_ts
                            FROM datasets_kv
                        """).fetchdf()
                        all_draws["event_ts"] = pd.to_datetime(all_draws["event_ts"], errors="coerce")
                        all_draws["_mtime"] = all_draws["filename"].astype(str).apply(_mtime)
                        all_draws["_t"] = all_draws["event_ts"].fillna(all_draws["_mtime"])
                        extra = all_draws[(all_draws["_t"].notna()) & (all_draws["_t"].between(window_start, window_end))].copy()
                        if not extra.empty:
                            draw_ids_to_pull = sorted(set(draw_ids_to_pull).union(set(extra["event_id"].dropna().astype(str).tolist())))
                    except Exception:
                        pass

        # 4) Pull ALL params for selected draws (full KV)
        df_draw_full = pd.DataFrame()
        if draw_ids_to_pull:
            draws_sql = "(" + ",".join(lit(d) for d in draw_ids_to_pull) + ")"

            # Prefer match by _draw
            try:
                df_draw_full = con.execute(f"""
                    SELECT
                        source_kind, source_file, event_ts, event_id, _draw, filename,
                        "Parameter Name", "Value", "Units"
                    FROM datasets_kv
                    WHERE CAST(_draw AS VARCHAR) IN {draws_sql}
                """).fetchdf()
            except Exception:
                df_draw_full = pd.DataFrame()

            # Fallback match by event_id
            if df_draw_full.empty:
                try:
                    df_draw_full = con.execute(f"""
                        SELECT
                            source_kind, source_file, event_ts, event_id, _draw, filename,
                            "Parameter Name", "Value", "Units"
                        FROM datasets_kv
                        WHERE CAST(event_id AS VARCHAR) IN {draws_sql}
                """).fetchdf()
                except Exception:
                    df_draw_full = pd.DataFrame()

        # 5) Combine full draws + matched maintenance
        d = pd.concat([df_draw_full, df_maint_matched], ignore_index=True)
        d["event_ts"] = pd.to_datetime(d.get("event_ts"), errors="coerce")

        # Fill dataset times by mtime if missing
        mask_ds = d.get("source_kind").astype(str).eq("dataset") if "source_kind" in d.columns else pd.Series([False] * len(d))
        if "filename" in d.columns and use_time_fallback:
            d.loc[mask_ds, "event_ts"] = d.loc[mask_ds, "event_ts"].fillna(
                d.loc[mask_ds, "filename"].astype(str).apply(_mtime)
            )

        # IMPORTANT: pivot drops NaN in index cols, so ensure these are not-null
        if "source_file" in d.columns:
            d["source_file"] = d["source_file"].fillna("").astype(str)

        d = d[d["event_ts"].notna()].copy()
        if d.empty:
            return pd.DataFrame(), pd.DataFrame()

        d["event_key"] = d["source_kind"].astype(str) + ":" + d["event_id"].astype(str)

        wide = (
            d.pivot_table(
                index=["event_ts", "event_key", "source_kind", "source_file"],
                columns="Parameter Name",
                values="Value",
                aggfunc="first",
            )
            .reset_index()
            .sort_values("event_ts")
        )

        return d, wide


    def render_event_details(con, event_key: str):
        """
        event_key format: 'dataset:<id>' or 'maintenance:<action_id>'
        """
        if not event_key or ":" not in event_key:
            return

        kind, eid = event_key.split(":", 1)
        kind = (kind or "").strip()

        st.markdown("### 🔎 Event details")
        st.caption(f"Selected: **{event_key}**")

        if kind == "dataset":
            # Show full draw KV (Parameter Name / Value / Units)
            df_kv = con.execute(f"""
                SELECT
                    event_ts,
                    event_id,
                    _draw,
                    filename,
                    "Parameter Name",
                    "Value",
                    "Units"
                FROM datasets_kv
                WHERE CAST(_draw AS VARCHAR) = {lit(eid)}
                   OR CAST(event_id AS VARCHAR) = {lit(eid)}
                ORDER BY "Parameter Name"
            """).fetchdf()

            if df_kv.empty:
                st.warning("No KV rows found for this draw id.")
                return

            # Nice header
            top = df_kv.head(1)
            draw_name = top["event_id"].iloc[0] if "event_id" in top.columns else eid
            ts = top["event_ts"].iloc[0] if "event_ts" in top.columns else None
            st.markdown(f"**Draw:** `{draw_name}`")
            if ts is not None:
                st.caption(f"Time: {ts}")

            st.dataframe(
                df_kv[["Parameter Name", "Value", "Units"]],
                use_container_width=True,
                height=420
            )

        elif kind == "maintenance":
            # Show maintenance action row
            try:
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
                    WHERE CAST(action_id AS VARCHAR) = {lit(eid)}
                    LIMIT 1
                """).fetchdf()
            except Exception as e:
                st.error(f"Failed to read maintenance action: {e}")
                return

            if df_act.empty:
                st.warning("Maintenance action not found.")
                return

            st.markdown("**Maintenance action**")
            st.dataframe(df_act, use_container_width=True, height=140)

            # Also show KV-style view (same parameter name = source_file)
            try:
                df_kv = con.execute(f"""
                    SELECT
                        action_ts AS event_ts,
                        source_file AS "Parameter Name",
                        (trim(COALESCE(component,'') || ' — ' || COALESCE(task,'')) ||
                         CASE WHEN COALESCE(note,'') <> '' THEN (' | ' || note) ELSE '' END
                        ) AS "Value",
                        '' AS "Units"
                    FROM maintenance_actions
                    WHERE CAST(action_id AS VARCHAR) = {lit(eid)}
                """).fetchdf()
                st.markdown("**KV view**")
                st.dataframe(df_kv[["Parameter Name", "Value", "Units"]], use_container_width=True, height=120)
            except Exception:
                pass

        else:
            st.info("Unknown event type.")


    def render_visual_lab_clickable(con, df_all):
        st.subheader("📈 Visual Lab (events over time)")
        wide_out = pd.DataFrame()
        numeric_out = []

        # If the filter matched only maintenance, optionally include nearby draws so that
        # numeric parameters are available in Visual Lab + Math Lab.
        has_dataset_match = False
        try:
            has_dataset_match = df_all["source_kind"].astype(str).eq("dataset").any()
        except Exception:
            has_dataset_match = False

        include_nearby = True
        pad_days = 7
        if not has_dataset_match:
            st.caption("Filter matched maintenance only. You can optionally include nearby draws to plot numeric parameters.")
            c_near1, c_near2 = st.columns([1, 1])
            with c_near1:
                include_nearby = st.checkbox(
                    "Include nearby draws",
                    value=bool(st.session_state.get("sql_include_nearby_draws", True)),
                    key="sql_include_nearby_draws",
                )
            with c_near2:
                pad_days = st.number_input(
                    "± pad days",
                    min_value=0,
                    max_value=365,
                    value=int(st.session_state.get("sql_nearby_pad_days", 7)),
                    step=1,
                    key="sql_nearby_pad_days",
                )

        d_kv, wide = build_visual_dataset_plus_maint(
            con,
            df_all,
            use_time_fallback=True,
            include_nearby_draws_when_maintenance_only=include_nearby,
            nearby_pad_days=int(pad_days),
        )
        if wide is None or wide.empty:
            st.warning("No events with valid time to plot.")
            return wide_out, numeric_out

        st.caption(f"Events with valid time: **{len(wide):,}**")

        # Explain what is included (helps keep Math Lab expectations consistent)
        try:
            n_ds = int(wide["source_kind"].astype(str).eq("dataset").sum())
            n_m = int(wide["source_kind"].astype(str).eq("maintenance").sum())
            st.caption(f"Included events → datasets: **{n_ds}**, maintenance: **{n_m}**")
        except Exception:
            pass

        META = {"event_ts", "event_key", "source_kind", "source_file"}
        all_plot_params = [c for c in wide.columns if c not in META]

        # Filter params used (for your mode toggle)
        filter_params_used = sorted(df_all["Parameter Name"].dropna().astype(str).unique().tolist())
        filter_plot_params = [c for c in filter_params_used if c in wide.columns]

        # Numeric candidates (for pooling / math lab)
        numeric_all = [c for c in all_plot_params if pd.to_numeric(wide[c], errors="coerce").notna().sum() > 0]
        wide_out = wide
        numeric_out = numeric_all

        st.markdown("### Choose parameters to plot (raw)")

        mode = st.radio(
            "Parameter picker mode",
            ["Only parameters used in the filter", "Any parameter from matched events (after filter)"],
            horizontal=True,
            key="sql_pick_mode",
        )

        if mode == "Only parameters used in the filter":
            pool = list(filter_plot_params)
            # enrich pool so you can still plot real draw numeric signals
            if len(pool) < 3 and len(numeric_all) > 0:
                pool = sorted(set(pool + numeric_all))
        else:
            pool = list(all_plot_params)

        chosen = st.multiselect("Parameters to plot", pool, key="sql_plot_params")

        # --- build plotly fig ---
        fig = go.Figure()

        # We’ll keep an event selection even if user plots only maintenance markers
        # Create a hidden “event markers” base trace (clickable)
        fig.add_trace(go.Scatter(
            x=wide["event_ts"],
            y=[0] * len(wide),
            mode="markers",
            marker=dict(size=10, opacity=0.0),
            customdata=wide["event_key"],
            name="events",
            hoverinfo="skip",
            showlegend=False
        ))

        y_min, y_max = None, None

        # Plot numeric series (clickable points with customdata)
        for col in (chosen or []):
            s_num = pd.to_numeric(wide[col], errors="coerce")
            if s_num.notna().sum() > 0:
                fig.add_trace(go.Scatter(
                    x=wide["event_ts"],
                    y=s_num,
                    mode="lines+markers",
                    name=col,
                    customdata=wide["event_key"],
                    hovertemplate=(
                        f"<b>{col}</b><br>"
                        "Time: %{x}<br>"
                        "Value: %{y}<br>"
                        "Event: %{customdata}<extra></extra>"
                    )
                ))
                _mn = float(s_num.min())
                _mx = float(s_num.max())
                y_min = _mn if y_min is None else min(y_min, _mn)
                y_max = _mx if y_max is None else max(y_max, _mx)

        # Maintenance/non-numeric markers: dashed vlines + a bottom marker trace (clickable)
        for col in (chosen or []):
            s_raw = wide[col]
            s_num = pd.to_numeric(s_raw, errors="coerce")
            if s_num.notna().sum() == 0:
                mask = s_raw.notna() & (s_raw.astype(str).str.strip() != "")
                if mask.any():
                    times = wide.loc[mask, "event_ts"]
                    keys = wide.loc[mask, "event_key"]

                    # vlines (not clickable)
                    for t in times.tolist():
                        fig.add_vline(x=t, line_width=1, line_dash="dash", opacity=0.35)

                    # clickable markers at bottom
                    base_y = 0.0
                    if y_min is not None and y_max is not None and y_max > y_min:
                        base_y = y_min - 0.05 * (y_max - y_min)

                    fig.add_trace(go.Scatter(
                        x=times,
                        y=[base_y] * len(times),
                        mode="markers",
                        marker=dict(size=10, symbol="x"),
                        name=col,
                        customdata=keys,
                        hovertemplate=(
                            f"<b>{col}</b><br>"
                            "Time: %{x}<br>"
                            "Event: %{customdata}<extra></extra>"
                        )
                    ))

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            title="Matched events over time (click a point to inspect)",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="closest",
        )

        # --- render + selection ---
        sel = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            key="sql_vis_plot"
        )

        # robust extraction of selected event_key
        selected_key = None
        try:
            if isinstance(sel, dict):
                pts = sel.get("selection", {}).get("points", sel.get("points", []))
                if pts:
                    selected_key = pts[0].get("customdata")
            else:
                # Streamlit sometimes returns an object with .selection
                pts = getattr(getattr(sel, "selection", None), "points", None)
                if pts:
                    selected_key = pts[0].get("customdata")
        except Exception:
            selected_key = None

        if selected_key:
            st.session_state["sql_selected_event_key"] = selected_key

        # show details if we have a selection saved
        if st.session_state.get("sql_selected_event_key"):
            with st.expander("📌 Clicked event details", expanded=True):
                render_event_details(con, st.session_state["sql_selected_event_key"])

        return wide_out, numeric_out



    # =========================================================
    # Render: Math Lab
    # =========================================================
    def render_math_lab(wide: pd.DataFrame, numeric_all: list):
        st.subheader("🧮 Math Lab (derived signals)")
        try:
            n_ds = int(wide["source_kind"].astype(str).eq("dataset").sum()) if "source_kind" in wide.columns else 0
            n_m = int(wide["source_kind"].astype(str).eq("maintenance").sum()) if "source_kind" in wide.columns else 0
            st.caption(f"Math is computed on the same included events as Visual Lab → datasets: {n_ds}, maintenance: {n_m}")
        except Exception:
            pass

        if not numeric_all:
            st.info("No numeric parameters available for Math Lab.")
            st.stop()

        st.caption(
            "Write expressions using **A**, **B**, **C** and **numpy** (**np**). "
            "Examples: `A`, `A/B`, `np.log10(A)`, `np.abs(A)`, `np.sqrt(A)`."
        )

        var_count = st.radio("How many parameters?", [1, 2, 3], horizontal=True, key="math_var_count")
        A_name = st.selectbox("A (parameter)", numeric_all, key="math_A_name")

        B_name = None
        C_name = None
        if var_count >= 2:
            B_name = st.selectbox("B (parameter)", [p for p in numeric_all if p != A_name], key="math_B_name")
        if var_count >= 3:
            C_name = st.selectbox("C (parameter)", [p for p in numeric_all if p not in (A_name, B_name)], key="math_C_name")

        if "math_expr" not in st.session_state or not str(st.session_state["math_expr"]).strip():
            st.session_state["math_expr"] = "A"

        expr = st.text_input("Expression", value=str(st.session_state["math_expr"]), key="math_expr_input")
        st.session_state["math_expr"] = expr
        expr = (expr or "").strip()

        if expr == "":
            st.info("Type an expression to compute, e.g. `A`, `A/B`, `np.log10(A)`.")
            st.stop()

        if not re.fullmatch(r"[0-9A-Za-z_\.\+\-\*\/\(\)\s,]+", expr):
            st.error("Expression contains unsupported characters.")
            st.stop()

        def series_of(name):
            if name is None:
                return pd.Series([np.nan] * len(wide), index=wide.index)
            return pd.to_numeric(wide[name], errors="coerce").astype(float)

        A = series_of(A_name)
        B = series_of(B_name) if var_count >= 2 else pd.Series([np.nan] * len(wide), index=wide.index)
        C = series_of(C_name) if var_count >= 3 else pd.Series([np.nan] * len(wide), index=wide.index)

        allowed = {"np": np, "A": A, "B": B, "C": C}

        try:
            Y = eval(expr, {"__builtins__": {}}, allowed)
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

        out = wide[["event_ts", "event_key", "source_kind", "source_file"]].copy()
        out["math"] = Y
        out = out.dropna(subset=["math"]).sort_values("event_ts")

        if out.empty:
            st.warning("No valid values after applying the math expression.")
            st.stop()

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        connect = st.checkbox("Connect points", value=True, key="math_connect_points")
        ax2.plot(
            out["event_ts"],
            out["math"],
            marker="o",
            linestyle="-" if connect else "None",
            label=expr,
        )
        locator2 = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter2 = mdates.ConciseDateFormatter(locator2)
        ax2.xaxis.set_major_locator(locator2)
        ax2.xaxis.set_major_formatter(formatter2)
        ax2.tick_params(axis="x", rotation=30)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")
        ax2.set_title("Math Lab")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)

        with st.expander("Show Math table"):
            st.dataframe(out, use_container_width=True, height=420)

    # =========================================================
    # MAIN FLOW
    # =========================================================
    render_indexing_controls()
    where_sql = render_filter_builder()
    df_all = render_run_filter(where_sql)
    wide, numeric_all = render_visual_lab_clickable(con, df_all)
    render_math_lab(wide, numeric_all)
