import streamlit as st
import base64
import pandas as pd
import plotly.express as px
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime as dt
import numpy as np
import plotly.graph_objects as go
CSV_SELECTION_FILE = "selected_csv.json"

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
            "📦 Order Draw"
        ],
        "⚙️ Operations": [
            "🍃 Tower state - Consumables and dies",
            "⚙️ Process Setup",
            "🧰 Maintenance",
            "🛠️ Tower Parts",
            "📋 Protocols"
        ],
        "📚 Monitoring &  Research": [
            "📊 Dashboard",
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
if tab_selection == "📊 Dashboard":
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if csv_files:
        selected_file = st.sidebar.selectbox("Select a dataset", csv_files, key="dataset_select")
        df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
    else:
        st.error("No CSV files found in the directory.")
        st.stop()
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
    # --- LEFT-ALIGNED TITLE (same look as "Orders Table") ---
    st.markdown(
        f"""
        <div style="
            font-size: 1.5rem;
            font-weight: 700;
            color: rgba(255,255,255,0.95);
            font-style: normal;
            margin-top: 0.2em;
            margin-bottom: 0.6em;
            text-align: left;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _card(title_txt, value, border_color, emoji=""):
        st.markdown(
            f"""
            <div style="
                width: 100%;
                min-height: 140px;
                background: rgba(0,0,0,0.35);
                border: 2px solid {border_color};
                border-radius: 18px;
                padding: 14px 14px;
                text-align: center;
                box-shadow: 0 6px 18px rgba(0,0,0,0.25);
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                gap: 6px;
            ">
                <div style="
                    font-size: 18px;
                    font-weight: 800;
                    color: white;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    max-width: 100%;
                ">
                    {emoji} {title_txt}
                </div>
                <div style="
                    font-size: 44px;
                    font-weight: 900;
                    color: white;
                    line-height: 1;
                ">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not os.path.exists(orders_file):
        st.info("No orders submitted yet.")
        return

    # Keep empty strings as empty (avoid NaN strings)
    df = pd.read_csv(orders_file, keep_default_na=False)

    # Ensure key columns exist (including Done + T&M fields)
    for col, default in {
        "Status": "Pending",
        "Priority": "Normal",
        "Fiber name and number": "",
        "Preform Name": "",
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

    # Parse time columns safely
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Desired Date"] = pd.to_datetime(df["Desired Date"], errors="coerce").dt.date

    # Normalize strings (avoid 'nan' showing in UI)
    for _c in [
        "Status",
        "Priority",
        "Fiber name and number",
        "Preform Name",
        "Notes",
        "Done CSV",
        "Done Description",
        "T&M Moved Timestamp",
    ]:
        if _c in df.columns:
            df[_c] = df[_c].astype(str).replace({"nan": "", "None": ""}).fillna("").str.strip()

    # Normalize T&M moved flag robustly (handles True/False, "True"/"False", 1/0, etc.)
    if "T&M Moved" in df.columns:
        df["T&M Moved"] = df["T&M Moved"].apply(
            lambda x: str(x).strip().lower() in ("true", "1", "yes", "y", "moved")
        )

    # Visible orders = not moved to T&M
    df_visible = df.copy()
    if "T&M Moved" in df_visible.columns:
        df_visible = df_visible[~df_visible["T&M Moved"]].copy()

    # KPI counts based ONLY on visible orders (so Done decreases when moved to T&M)
    def _count(status_name: str) -> int:
        if "Status" not in df_visible.columns:
            return 0
        return int((df_visible["Status"].astype(str).str.strip() == status_name).sum())

    pending = _count("Pending")
    scheduled = _count("Scheduled")
    done = _count("Done")
    failed = _count("Failed")

    # --- Cards row ---
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        _card("Pending", pending, "orange", "🟠")
    with c2:
        _card("Scheduled", scheduled, "dodgerblue", "🗓️")
    with c3:
        _card("Done", done, "limegreen", "✅")
    with c4:
        _card("Failed", failed, "crimson", "❌")

    st.markdown("### 📋 Orders Table")

    show_cols = [
        "Status",
        "Priority",
        "Fiber name and number",
        "Preform Name",
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

    # Sort: soonest desired date first, then newest timestamp
    _sort_date = pd.to_datetime(df_visible["Desired Date"], errors="coerce")
    _sort_ts = pd.to_datetime(df_visible["Timestamp"], errors="coerce")
    table_df["_sort_date"] = _sort_date
    table_df["_sort_ts"] = _sort_ts
    table_df = table_df.sort_values(by=["_sort_date", "_sort_ts"], ascending=[True, False])
    table_df = table_df.drop(columns=["_sort_date", "_sort_ts"])

    def _status_style(val):
        s = str(val).strip()
        bg = {
            "Pending": "#ffb020",
            "Scheduled": "#2d7ff9",
            "Done": "#2ecc71",
            "Failed": "#ff3b30",
        }.get(s, "#2c2c2c")
        return f"background-color: {bg}; color: black; font-weight: 900;"

    def _priority_style(val):
        p = str(val).strip()
        bg = {"Low": "#b0b0b0", "Normal": "#ffffff", "High": "#ff5a5a"}.get(p, "#ffffff")
        return f"background-color: {bg}; color: black; font-weight: 800;"

    styled = table_df.style
    if "Status" in table_df.columns:
        styled = styled.applymap(_status_style, subset=["Status"])
    if "Priority" in table_df.columns:
        styled = styled.applymap(_priority_style, subset=["Priority"])

    st.dataframe(styled, use_container_width=True, height=height)

    # -----------------------------
    # T&M Report section
    # -----------------------------
    st.markdown("### 🧾 T&M Report")

    # DONE-only inbox (and not moved)
    df_done = df_visible.copy()
    if "Status" in df_done.columns:
        df_done["Status"] = df_done["Status"].astype(str).str.strip()
        df_done = df_done[df_done["Status"].str.lower() == "done"].copy()
    else:
        df_done = df_done.iloc[0:0].copy()

    order_options = []
    for i in df_done.index:
        ft = str(df_done.loc[i, "Fiber name and number"]) if "Fiber name and number" in df_done.columns else ""
        dd = str(df_done.loc[i, "Desired Date"]) if "Desired Date" in df_done.columns else ""
        order_options.append(f"{i}: {ft} | {dd} | Done")

    if not order_options:
        st.info("No DONE orders available for T&M reporting.")
        return

    # Prevent stale selection from older option lists
    tm_key = "home_tm_order_select_done_only"
    if tm_key in st.session_state and st.session_state[tm_key] not in order_options:
        st.session_state.pop(tm_key, None)

    rep_sel = st.selectbox("Select an order for report", order_options, key=tm_key)
    rep_idx = int(str(rep_sel).split(":")[0])

    done_csv = str(df.loc[rep_idx, "Done CSV"]).strip() if "Done CSV" in df.columns else ""
    done_desc = str(df.loc[rep_idx, "Done Description"]).strip() if "Done Description" in df.columns else ""
    if done_desc:
        st.caption(f"Done description: {done_desc}")

    btn_cols = st.columns([1, 1, 3])
    with btn_cols[0]:
        show_report = st.button("🧾 Open T&M Report", key=f"home_tm_open_{rep_idx}")
    with btn_cols[1]:
        move_to_tm = st.button("📦 Move to T&M", key=f"home_tm_move_{rep_idx}")

    if move_to_tm:
        df.at[rep_idx, "T&M Moved"] = True
        df.at[rep_idx, "T&M Moved Timestamp"] = str(pd.Timestamp.now())
        df.to_csv(orders_file, index=False)
        st.success("Moved to T&M. This order is now hidden from the tables.")
        st.rerun()

    if show_report:
        if not done_csv:
            st.warning("This order has no 'Done CSV' attached yet. Mark it Done in 📦 Order Draw and attach the CSV.")
            return

        csv_path = os.path.join("data_set_csv", done_csv)
        if not os.path.exists(csv_path):
            st.error(f"CSV not found: {csv_path}")
            return

        report_df = pd.read_csv(csv_path)
        st.success(f"Showing report: {done_csv}")
        st.dataframe(report_df, use_container_width=True, height=420)

        try:
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Download CSV",
                    data=f,
                    file_name=done_csv,
                    mime="text/csv",
                    key=f"home_tm_dl_{rep_idx}",
                )
        except Exception:
            pass
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

    st.title("🔍 Iris Selection")
    st.subheader("Iris Selection Tool")

    # Input Preform Diameter
    preform_diameter = st.number_input("Enter Preform Diameter (mm)", min_value=0.0, step=0.1, format="%.2f")
    # Add checkbox for "Tiger" and input for cut percentage
    tiger_cut = st.checkbox("Is it a Tiger?", value=False)
    cut_percentage = 0
    if tiger_cut:
        cut_percentage = st.number_input("Enter Cut Percentage", min_value=0, max_value=100, value=20, step=1)

    # Calculate the adjusted area based on the cut
    def calculate_adjusted_area(diameter, cut_percentage):
        original_area = np.pi * (diameter / 2) ** 2
        adjusted_area = original_area * (1 - cut_percentage / 100)
        return adjusted_area

    if preform_diameter > 0:
        adjusted_area = calculate_adjusted_area(preform_diameter, cut_percentage)
        st.write(f"Adjusted Area (with {cut_percentage}% cut): {adjusted_area:.2f} mm²")
        effective_diameter = 2 * np.sqrt(adjusted_area / np.pi)
        st.write(f"Effective Preform Diameter after cut: {effective_diameter:.2f} mm")
    else:
        st.warning("Please enter a valid preform diameter.")

    iris_diameters = [round(x * 0.5, 1) for x in range(20, 91)]  # Iris diameters from 10 mm to 45 mm

    # Validate and compute the best iris diameter based on the effective preform diameter
    if preform_diameter > 0 and iris_diameters:
        valid_iris = [d for d in iris_diameters if d > effective_diameter]
        if valid_iris:
            # Calculate the best iris diameter that gives the gap closest to 200
            results = [(d, (np.pi / 4) * (d ** 2 - effective_diameter ** 2)) for d in valid_iris]
            best = min(results, key=lambda x: abs(x[1] - 200))  # Find the iris diameter with gap closest to 200

            # Display the best matching iris diameter
            st.write(f"**Best Matching Iris Diameter:** {best[0]:.2f} mm")
            st.write(f"**Calculated Gap:** {best[1]:.2f} mm")

            # Allow manual override of iris selection
            selected_iris = st.selectbox("Or select a different iris diameter", valid_iris,
                                         index=valid_iris.index(best[0]))
            manual_gap = (np.pi / 4) * (selected_iris ** 2 - effective_diameter ** 2)
            st.write(
                f"**Manual Selection - Iris Diameter:** {selected_iris:.2f} mm, **Calculated Gap:** {manual_gap:.2f} mm")

            # Display the Preform Diameter and Selected Iris Data
            st.write(f"**Preform Diameter:** {preform_diameter:.2f} mm")

            # Allow user to select the CSV to update
            recent_csv_files = [f for f in os.listdir('data_set_csv') if f.endswith(".csv")]
            selected_csv = st.selectbox("Select CSV to Update", recent_csv_files, key="iris_select_csv_update")

            # Show update button only after CSV is selected
            if selected_csv:
                if st.button("Update Dataset CSV", key="iris_update_dataset_csv"):
                    st.write(f"Selected CSV: {selected_csv}")
                    tiger_cut_value = cut_percentage if tiger_cut else 0  # Set the tiger cut value
                    data_to_add = [
                        {"Parameter Name": "Preform Diameter", "Value": preform_diameter, "Units": "mm"},
                        {"Parameter Name": "Tiger Cut", "Value": tiger_cut_value, "Units": "%"},
                        {"Parameter Name": "Selected Iris Diameter", "Value": selected_iris, "Units": "mm"},
                    ]

                    # Load the selected CSV
                    csv_path = os.path.join('data_set_csv', selected_csv)
                    try:
                        df = pd.read_csv(csv_path)
                    except FileNotFoundError:
                        st.error(f"CSV file '{selected_csv}' not found.")
                        st.stop()

                    # Append new rows with the data
                    new_rows = pd.DataFrame(data_to_add)
                    df = pd.concat([df, new_rows], ignore_index=True)

                    # Save the updated CSV back to the 'data_set_csv' folder
                    df.to_csv(csv_path, index=False)
                    st.success(f"CSV '{selected_csv}' updated with new data!")
        else:
            st.warning("No iris diameter is larger than the preform diameter.")
    else:
        st.info("Please enter a preform diameter and provide valid iris diameters.")
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

    column_order = [
        "Status",
        "Part Name",
        "Serial Number",
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

    for col in column_order:
        if col not in orders_df.columns:
            orders_df[col] = ""

    orders_df = orders_df[column_order].copy().fillna("")
    orders_df["Status"] = orders_df["Status"].astype(str).str.strip()

    if orders_df.empty:
        st.warning("No orders have been placed yet.")
        return

    # ---------------- Counts ----------------
    status_lower = orders_df["Status"].str.lower()
    needed_count = int((status_lower == "needed").sum())
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
        .kpi-needed{
            border: 2px solid rgba(255, 80, 80, 0.95) !important;
            box-shadow: 0 0 18px rgba(255, 80, 80, 0.85);
            background: rgba(255, 80, 80, 0.22);
        }
        .kpi-received{
            border: 2px solid rgba(80, 255, 120, 0.95) !important;
            box-shadow: 0 0 18px rgba(80, 255, 120, 0.85);
            background: rgba(80, 255, 120, 0.22);
        }

        /* ================= FULL GLASS TABLE (AG-GRID) =================
           NOTE: Streamlit dataframe uses AG-Grid internally.
           We make the whole grid semi-transparent so the background image shows through.
        */

        /* Outer shell */
        div[data-testid="stDataFrame"]{
            background: transparent !important;
        }

        /* Wrapper around the grid */
        div[data-testid="stDataFrame"] > div{
            background: rgba(0,0,0,0.28) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 18px !important;
            padding: 10px !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25) !important;
        }

        /* Make all AG-grid base layers transparent */
        .ag-root-wrapper, .ag-root, .ag-body-viewport, .ag-center-cols-viewport,
        .ag-center-cols-container, .ag-floating-top, .ag-floating-bottom,
        .ag-pinned-left-cols-container, .ag-pinned-right-cols-container,
        .ag-row, .ag-row-odd, .ag-row-even{
            background: transparent !important;
        }

        /* Header glass */
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

        /* Cells glass fill (THIS is what makes the full table look like the KPI cards) */
        .ag-cell{
            background: rgba(0,0,0,0.14) !important;
            color: rgba(255,255,255,0.92) !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
            border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        }

        /* Hover */
        .ag-row-hover .ag-cell{
            background: rgba(255,255,255,0.06) !important;
        }

        /* Empty area below rows */
        .ag-body-viewport{
            background: rgba(0,0,0,0.14) !important;
        }

        /* Scrollbars (optional nicer look) */
        ::-webkit-scrollbar { height: 10px; width: 10px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- KPI Cards (symmetric) ----------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    needed_class = "kpi-card kpi-needed" if needed_count > 0 else "kpi-card"
    with c1:
        st.markdown(
            f"""
            <div class="{needed_class}">
                <div class="kpi-title">🔴 Needed</div>
                <div class="kpi-value">{needed_count}</div>
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
    status_order = ["Needed", "Approved", "Ordered", "Shipped", "Received", "Installed"]
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
    def highlight_status(row):
        color_map = {
            "Needed": "background-color: lightcoral; color: black; font-weight: 900;",
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
# ------------------ Process Tab ------------------
elif tab_selection == "⚙️ Process Setup":
    st.title("⚙️ Process Setup")
    st.caption("One-page setup for every draw: Create CV + coating + iris + PID/TF")
    render_create_draw_dataset_csv()
    st.markdown("---")
    view = _process_setup_buttons()

    # Option 2: vertical sections
    if view in ("all", "coating"):
        st.markdown("---")
        render_coating_section()

    if view in ("all", "iris"):
        st.markdown("---")
        render_iris_selection_section()

    if view in ("all", "pid"):
        st.markdown("---")
        render_pid_tf_section()
# ------------------ Dashboard Tab ------------------
elif tab_selection == "📊 Dashboard":
    st.title(f"📊 Draw Tower Logs Dashboard - {selected_file}")

    # Sidebar options
    show_corr_matrix = st.sidebar.checkbox("Show Correlation Matrix")
    column_options = df.columns.tolist()

    # Plot axis selections
    x_axis = st.selectbox("Select X-axis", column_options, key="x_axis_dash")
    y_axis = st.selectbox("Select Y-axis", column_options, key="y_axis_dash")

    # Drop NA and sort by x
    filtered_df = df.dropna(subset=[x_axis, y_axis]).sort_values(by=x_axis)

    # ---- Zone selection slider (works for datetime, numeric, or index) ----
    st.subheader("🟩 Zone Marker")
    st.caption("Use the slider to pick a range, then click 'Add Selected Zone'.")
    time_range = None

    # datetime x-axis
    if np.issubdtype(filtered_df[x_axis].dtype, np.datetime64):
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

    # numeric x-axis
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

    # fallback: index slider
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

    # Build plot
    st.subheader("📈 Plot")
    fig = px.line(filtered_df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", markers=True)

    # Green rectangles for saved zones
    for start, end in st.session_state["good_zones"]:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="green", opacity=0.3, line_width=0,
            annotation_text="Good Zone", annotation_position="top left"
        )

    # Blue rectangle for live selection
    if time_range:
        fig.add_vrect(
            x0=time_range[0], x1=time_range[1],
            fillcolor="blue", opacity=0.2, line_width=1,
            line_dash="dot",
            annotation_text="Selected", annotation_position="top right"
        )

    # Final render
    st.plotly_chart(fig, use_container_width=True)

    # Add new zone
    if time_range and st.button("➕ Add Selected Zone", key="dash_add_zone"):
        st.session_state["good_zones"].append(time_range)
        st.success(f"Zone added: {time_range[0]} to {time_range[1]}")

    # Summary section
    if st.session_state["good_zones"]:
        st.write("### ✅ Good Zones Summary")

        summary_data = []
        all_values = []

        for i, (start, end) in enumerate(st.session_state["good_zones"]):
            zone_data = filtered_df[(filtered_df[x_axis] >= start) & (filtered_df[x_axis] <= end)]
            if not zone_data.empty:
                summary_data.append({
                    "Zone": f"Zone {i+1}",
                    "Start": start,
                    "End": end,
                    "Avg": zone_data[y_axis].mean(),
                    "Min": zone_data[y_axis].min(),
                    "Max": zone_data[y_axis].max()
                })
                all_values.extend(pd.to_numeric(zone_data[y_axis], errors='coerce').dropna().values)

        if all_values:
            st.markdown("#### 📊 Combined Stats")
            st.write(f"**Start:** {min(all_values):.4f}")
            st.write(f"**End:** {max(all_values):.4f}")
            st.write(f"**Average:** {pd.Series(all_values).mean():.4f}")
            st.write(f"**Min:** {min(all_values):.4f}")
            st.write(f"**Max:** {max(all_values):.4f}")

        st.dataframe(pd.DataFrame(summary_data))

    # CSV Save section
    recent_csv_files = [f for f in os.listdir('data_set_csv') if f.endswith(".csv")]
    selected_csv = st.selectbox("Select CSV to Update", recent_csv_files, key="dashboard_select_csv_update")

    if selected_csv and st.button("💾 Save Zones Summary"):
        csv_path = os.path.join('data_set_csv', selected_csv)
        try:
            df_csv = pd.read_csv(csv_path)
        except FileNotFoundError:
            st.error(f"CSV file '{selected_csv}' not found.")
            st.stop()

        data_to_add = [{"Parameter Name": "Log File Name", "Value": selected_file, "Units": ""}]
        for i, (start, end) in enumerate(st.session_state["good_zones"]):
            data_to_add.extend([
                {"Parameter Name": f"Zone {i+1} Start", "Value": start, "Units": ""},
                {"Parameter Name": f"Zone {i+1} End", "Value": end, "Units": ""}
            ])

            zone_data = df[(df["Date/Time"] >= pd.to_datetime(start)) & (df["Date/Time"] <= pd.to_datetime(end))]
            if not zone_data.empty:
                for param in ["Fibre Length", "Pf Process Position"]:
                    if param in zone_data.columns:
                        data_to_add.extend([
                            {"Parameter Name": f"Zone {i+1} {param} at Start", "Value": zone_data.iloc[0][param], "Units": "km" if "Fibre" in param else "mm"},
                            {"Parameter Name": f"Zone {i+1} {param} at End", "Value": zone_data.iloc[-1][param], "Units": "km" if "Fibre" in param else "mm"}
                        ])
                for param in ["Bare Fibre Diameter", "Coated Inner Diameter", "Coated Outer Diameter"]:
                    if param in zone_data.columns:
                        data_to_add.extend([
                            {"Parameter Name": f"Zone {i + 1} Avg ({param})", "Value": zone_data[param].mean(), "Units": "µm"},
                            {"Parameter Name": f"Zone {i + 1} Min ({param})", "Value": zone_data[param].min(), "Units": "µm"},
                            {"Parameter Name": f"Zone {i + 1} Max ({param})", "Value": zone_data[param].max(), "Units": "µm"}
                        ])
                for param in ["Capstan Speed"]:
                    if param in zone_data.columns:
                        data_to_add.extend([
                            {"Parameter Name": f"Zone {i + 1} Avg ({param})", "Value": zone_data[param].mean(),
                             "Units": "m/min"},
                            {"Parameter Name": f"Zone {i + 1} Min ({param})", "Value": zone_data[param].min(),
                             "Units": "m/min"},
                            {"Parameter Name": f"Zone {i + 1} Max ({param})", "Value": zone_data[param].max(),
                             "Units": "m/min"}
                        ])
                for param in ["Tension N"]:
                    if param in zone_data.columns:
                        data_to_add.extend([
                            {"Parameter Name": f"Zone {i + 1} Avg ({param})", "Value": zone_data[param].mean(),
                             "Units": "g"},
                            {"Parameter Name": f"Zone {i + 1} Min ({param})", "Value": zone_data[param].min(),
                             "Units": "g"},
                            {"Parameter Name": f"Zone {i + 1} Max ({param})", "Value": zone_data[param].max(),
                             "Units": "g"}
                        ])
                for param in ["Furnace DegC Actual"]:
                    if param in zone_data.columns:
                        data_to_add.extend([
                            {"Parameter Name": f"Zone {i + 1} Avg ({param})", "Value": zone_data[param].mean(),
                             "Units": "C"},
                            {"Parameter Name": f"Zone {i + 1} Min ({param})", "Value": zone_data[param].min(),
                             "Units": "C"},
                            {"Parameter Name": f"Zone {i + 1} Max ({param})", "Value": zone_data[param].max(),
                             "Units": "C"}
                        ])
        df_csv = pd.concat([df_csv, pd.DataFrame(data_to_add)], ignore_index=True)
        df_csv.to_csv(csv_path, index=False)
        st.success(f"CSV '{selected_csv}' updated!")

    # Show raw data
    st.write("### 🧾 Raw Data Preview")
    st.data_editor(df, height=300, use_container_width=True)

    # Correlation matrix
    if show_corr_matrix:
        st.write("### 🔗 Correlation Matrix")
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            fig_corr, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig_corr)
        else:
            st.warning("No numerical columns available.")
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
    st.write("### Schedule Timeline (Recurring Expanded)")
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

    orders_file = "draw_orders.csv"
    SCHEDULE_FILE = "tower_schedule.csv"
    schedule_required_cols = ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"]

    # Make sure schedule file exists (needed for move-to-schedule)
    if not os.path.exists(SCHEDULE_FILE):
        pd.DataFrame(columns=schedule_required_cols).to_csv(SCHEDULE_FILE, index=False)

    # =========================
    # 1) TABLE FIRST
    # =========================
    st.subheader("📋 Existing Draw Orders")

    if os.path.exists(orders_file):
        # Keep empty strings as empty (avoid NaN for the Done fields)
        df = pd.read_csv(orders_file, keep_default_na=False)

        # ---- Fix types ----
        if "Desired Date" in df.columns:
            df["Desired Date"] = pd.to_datetime(df["Desired Date"], errors="coerce").dt.date
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # Normalize common text columns (prevents showing 'nan' in the UI)
        for _c in [
            "Done CSV",
            "Done Description",
            "T&M Moved Timestamp",
            "Notes",
            "Fiber name and number",
            "Order Opener",
            "Preform Name",
        ]:
            if _c in df.columns:
                df[_c] = df[_c].astype(str).replace({"nan": "", "None": ""}).fillna("")

        # ---- Fix missing columns ----
        for col, default in {
            "Status": "Pending",
            "Priority": "Normal",
            "Fiber name and number": "",
            "Order Opener": "",
            "Preform Name": "",
            "Done CSV": "",
            "Done Description": "",
            "T&M Moved": False,
            "T&M Moved Timestamp": "",
        }.items():
            if col not in df.columns:
                df[col] = default

        # ---- Reorder Columns ----
        desired_order = [
            "Status",
            "Priority",
            "Order Opener",
            "Preform Name",
            "Fiber name and number",
            "Timestamp",
            "Desired Date",
            "Fiber Diameter (µm)",
            "Main Coating Diameter (µm)",
            "Secondary Coating Diameter (µm)",
            "Tension (g)",
            "Draw Speed (m/min)",
            "Length (m)",
            "Main Coating",
            "Secondary Coating",
            "Spools",
            "Notes",
            "Done CSV",
            "Done Description",
        ]
        other_cols = [c for c in df.columns if c not in desired_order]
        df = df[[c for c in desired_order if c in df.columns] + other_cols]

        # ---- Hide items that were moved to T&M ----
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
            .applymap(color_status, subset=["Status"] if "Status" in df.columns else None)
            .applymap(color_priority, subset=["Priority"] if "Priority" in df.columns else None)
        )
        st.dataframe(styled_df, use_container_width=True)

        # =========================
        # 2) UPDATE STATUS
        # =========================
        st.markdown("### 🔄 Update Status for a Specific Order")

        order_options = []
        for i in df_visible.index:
            ft = df.loc[i, "Fiber name and number"] if "Fiber name and number" in df.columns else ""
            ts = df.loc[i, "Timestamp"] if "Timestamp" in df.columns else ""
            order_options.append(f"{i}: {ft} | {ts}")

        if not order_options:
            st.info("No visible orders (all orders may have been moved to T&M).")
            st.stop()

        selected = st.selectbox("Select an Order", order_options, key="order_status_select")
        selected_index = int(selected.split(":")[0])

        if "T&M Moved" in df.columns and str(df.loc[selected_index, "T&M Moved"]).strip().lower() in ("true", "1", "yes", "y", "moved"):
            st.warning("This order was moved to T&M and is hidden/locked. It can’t be edited here.")
            st.stop()

        all_statuses = ["Pending", "Scheduled", "Failed", "Done"]
        current_status = str(df.loc[selected_index, "Status"]).strip()

        if current_status not in all_statuses and current_status != "":
            all_statuses.insert(0, current_status)

        new_status = st.selectbox(
            "New Status",
            all_statuses,
            index=all_statuses.index(current_status) if current_status in all_statuses else 0,
            key=f"order_new_status_{selected_index}",
        )

        # ---------------------------
        # Completion info (shown always, enabled only when Done)
        # ---------------------------
        done_mode = str(new_status).strip() == "Done"
        done_csv_choice = ""
        done_desc_text = ""

        # Ensure dataset folder exists
        data_set_dir = "data_set_csv"
        os.makedirs(data_set_dir, exist_ok=True)

        with st.expander("✅ Completion info (required for Done)", expanded=done_mode):
            csv_files = [f for f in os.listdir(data_set_dir) if f.lower().endswith(".csv")]

            # Sort newest first (by modified time)
            csv_files_sorted = sorted(
                csv_files,
                key=lambda fn: os.path.getmtime(os.path.join(data_set_dir, fn)) if os.path.exists(os.path.join(data_set_dir, fn)) else 0,
                reverse=True,
            )

            # Add a blank option so the user can see 'not selected'
            options = [""] + csv_files_sorted

            current_done_csv = str(df.loc[selected_index, "Done CSV"]) if "Done CSV" in df.columns else ""
            default_idx = options.index(current_done_csv) if current_done_csv in options else 0

            done_csv_choice = st.selectbox(
                "Select result CSV (newest first)",
                options,
                index=default_idx,
                key=f"order_done_csv_select_{selected_index}",
                disabled=not done_mode,
                help="Put the draw result CSV inside the folder: data_set_csv/",
            )

            done_desc_text = st.text_area(
                "Done description (what happened / notes)",
                value=str(df.loc[selected_index, "Done Description"]) if "Done Description" in df.columns else "",
                key=f"order_done_desc_{selected_index}",
                height=80,
                disabled=not done_mode,
            )

            if done_mode and not csv_files_sorted:
                st.warning("No CSV files found in 'data_set_csv'. Add the draw result CSV there, then pick it here.")

            if done_mode:
                st.caption(f"Found {len(csv_files_sorted)} CSV file(s) in '{data_set_dir}'.")

        if st.button("✅ Update Status", key="order_update_status_btn"):
            # If Done -> require completion info BEFORE changing status
            if str(new_status).strip() == "Done" and not str(done_csv_choice).strip():
                st.error("To set status to 'Done', please select a result CSV (newest-first list) from 'data_set_csv'.")
                st.stop()

            df.at[selected_index, "Status"] = new_status

            # If Done -> also store completion info
            if str(new_status).strip() == "Done":
                # Ensure columns exist even if file was old
                if "Done CSV" not in df.columns:
                    df["Done CSV"] = ""
                if "Done Description" not in df.columns:
                    df["Done Description"] = ""

                df.at[selected_index, "Done CSV"] = str(done_csv_choice).strip()
                df.at[selected_index, "Done Description"] = str(done_desc_text).strip()
                st.info(f"Attached CSV: {str(done_csv_choice).strip()}")

            # Final cleanup before save
            for _c in ["Done CSV", "Done Description"]:
                if _c in df.columns:
                    df[_c] = df[_c].astype(str).replace({"nan": "", "None": ""}).fillna("")

            df.to_csv(orders_file, index=False)
            st.success(f"Status updated for order {selected_index}.")
            st.rerun()

        # =========================
        # 2.5) MOVE TO SCHEDULE
        # =========================
        st.markdown("### 📅 Move Selected Order to Schedule")

        default_date = pd.Timestamp.today().date()

        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            sched_date = st.date_input("Schedule Date", value=default_date, key="move_sched_date")
        with colB:
            sched_start_time = st.time_input("Start Time", key="move_sched_start_time")
        with colC:
            duration_min = st.number_input(
                "Duration (min)", min_value=1, step=5, value=60, key="move_sched_duration"
            )

        start_dt = pd.to_datetime(f"{sched_date} {sched_start_time}")
        end_dt = start_dt + pd.to_timedelta(int(duration_min), unit="m")

        fiber = str(df.loc[selected_index, "Fiber name and number"]) if "Fiber name and number" in df.columns else ""
        preform = str(df.loc[selected_index, "Preform Name"]) if "Preform Name" in df.columns else ""
        prio = str(df.loc[selected_index, "Priority"]) if "Priority" in df.columns else ""
        length_m = df.loc[selected_index, "Length (m)"] if "Length (m)" in df.columns else ""
        spools = df.loc[selected_index, "Spools"] if "Spools" in df.columns else ""
        notes_txt = str(df.loc[selected_index, "Notes"]) if "Notes" in df.columns else ""

        desc_lines = [
            f"ORDER #{selected_index} | Priority: {prio}",
            f"Fiber: {fiber} | Preform: {preform}",
            f"Length: {length_m} m | Spools: {spools}",
        ]
        if notes_txt and notes_txt.strip():
            desc_lines.append(f"Notes: {notes_txt.strip()}")

        event_description = " | ".join([x for x in desc_lines if str(x).strip() != ""])

        existing = pd.read_csv(SCHEDULE_FILE)
        for c in schedule_required_cols:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[schedule_required_cols]

        already_exists = False
        if not existing.empty:
            already_exists = existing["Description"].astype(str).str.contains(
                f"ORDER #{selected_index}", na=False
            ).any()

        col1, col2 = st.columns([1, 1])
        with col1:
            mark_order_scheduled = st.checkbox(
                "Also mark order status as 'Scheduled'", value=True, key="mark_scheduled_cb"
            )
        with col2:
            allow_duplicates = st.checkbox("Allow duplicates", value=False, key="allow_duplicates_cb")

        btn_label = "📅 Move to Schedule" if not already_exists else "⚠️ Already scheduled (click to add anyway)"
        if st.button(btn_label, key="move_to_schedule_btn"):
            if already_exists and not allow_duplicates:
                st.warning("This order already exists in the schedule. Enable 'Allow duplicates' to add another event.")
            else:
                new_event = pd.DataFrame([{
                    "Event Type": "Drawing",
                    "Start DateTime": start_dt,
                    "End DateTime": end_dt,
                    "Description": event_description,
                    "Recurrence": "None",
                }])

                updated = pd.concat([existing, new_event], ignore_index=True)
                updated.to_csv(SCHEDULE_FILE, index=False)

                if mark_order_scheduled:
                    df.at[selected_index, "Status"] = "Scheduled"
                    df.to_csv(orders_file, index=False)

                st.success("✅ Order moved to schedule!")
                st.rerun()

    else:
        st.info("No orders submitted yet.")

    st.divider()

    # =========================
    # 3) NEW ENTRY LAST
    # =========================
    st.subheader("📝 Enter Draw Order Details")

    order_opener = st.text_input("Order Opened By", key="order_opener")
    priority = st.selectbox("Priority", ["Low", "Normal", "High"], key="order_priority")
    fiber_type = st.text_input("Fiber name and number", key="order_fiber_type")
    preform_name = st.text_input("Preform Name", key="order_preform_name")

    fiber_diameter = st.number_input("Fiber Diameter (µm)", min_value=0.0, key="order_fiber_diam")
    diameter_main = st.number_input("Main Coating Diameter (µm)", min_value=0.0, key="order_main_diam")
    diameter_secondary = st.number_input("Secondary Coating Diameter (µm)", min_value=0.0, key="order_sec_diam")

    tension = st.number_input("Tension (g)", min_value=0.0, key="order_tension")
    draw_speed = st.number_input("Draw Speed (m/min)", min_value=0.0, key="order_speed")
    length_required = st.number_input("Required Length (m)", min_value=0.0, key="order_length")

    coating_main = st.text_input("Main Coating Type", key="order_coating_main")
    coating_secondary = st.text_input("Secondary Coating Type", key="order_coating_secondary")

    num_spools = st.number_input("Number of Spools", min_value=1, step=1, key="order_spools")
    desired_date = st.date_input("Desired Draw Date", key="order_date")
    notes = st.text_area("Additional Notes / Instructions", key="order_notes")

    submit = st.button("📤 Submit Draw Order", key="order_submit_btn")

    if submit:
        order_data = {
            "Status": "Pending",
            "Priority": priority,
            "Order Opener": order_opener,
            "Preform Name": preform_name,
            "Fiber name and number": fiber_type,
            "Timestamp": pd.Timestamp.now(),
            "Desired Date": desired_date,
            "Fiber Diameter (µm)": fiber_diameter,
            "Main Coating Diameter (µm)": diameter_main,
            "Secondary Coating Diameter (µm)": diameter_secondary,
            "Tension (g)": tension,
            "Draw Speed (m/min)": draw_speed,
            "Length (m)": length_required,
            "Main Coating": coating_main,
            "Secondary Coating": coating_secondary,
            "Spools": num_spools,
            "Notes": notes,
        }

        if os.path.exists(orders_file):
            old = pd.read_csv(orders_file)
            new_df = pd.concat([old, pd.DataFrame([order_data])], ignore_index=True)
        else:
            new_df = pd.DataFrame([order_data])

        new_df.to_csv(orders_file, index=False)
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
    st.write("### Order Tracking")
    st.sidebar.title("Order Parts Management")

    ORDER_FILE = "part_orders.csv"

    if os.path.exists(ORDER_FILE):
        orders_df = pd.read_csv(ORDER_FILE)
    else:
        orders_df = pd.DataFrame(
            columns=["Part Name", "Serial Number", "Purpose", "Details", "Date Ordered", "Company", "Status"]
        )

    action = st.sidebar.radio("Manage Orders", ["Add New Order", "Update Existing Order"], key="order_action")

    if action == "Add New Order":
        part_name = st.sidebar.text_input("Part Name")
        serial_number = st.sidebar.text_input("Serial Number")
        purpose = st.sidebar.text_area("Purpose of Order")
        Details = st.sidebar.text_area("Details for New/Replacement")

        opened_by = st.sidebar.text_input("Opened By")

        # Load the list of projects from the development file
        projects_df = pd.read_csv(DEVELOPMENT_FILE)
        project_options = ["None"] + list(pd.Series(projects_df["Project Name"]).unique())  # Ensure unique projects
        selected_project = st.sidebar.selectbox("Select Project for This Part", project_options)

        if st.sidebar.button("Save Order"):
            new_order = pd.DataFrame([{
                "Part Name": part_name,
                "Serial Number": serial_number,
                "Purpose": purpose,
                "Details": Details,
                #"Date Ordered": date_ordered.strftime("%Y-%m-%d") if date_ordered else "",
                #"Company": company,
                "Opened By": opened_by,
                "Status": "Needed",
                "Project Name": selected_project if selected_project != "None" else ""  # If "None", leave empty
            }])

            orders_df = pd.concat([orders_df, new_order], ignore_index=True)
            orders_df.to_csv(ORDER_FILE, index=False)
            st.sidebar.success("Order saved!")

    elif action == "Update Existing Order":
        if not orders_df.empty:
            order_to_update = st.sidebar.selectbox("Select an Order to Update",
                                                   orders_df["Part Name"] + " - " + orders_df["Serial Number"].astype(str))

            order_index = orders_df[
                (orders_df["Part Name"] + " - " + orders_df["Serial Number"].astype(str)) == order_to_update].index[0]
            # Input fields for Part Name and Serial Number
            updated_part_name = st.sidebar.text_input("Update Part Name", value=orders_df.at[order_index, "Part Name"])
            updated_serial_number = st.sidebar.text_input("Update Serial Number", value=orders_df.at[order_index, "Serial Number"])
            new_status = st.sidebar.selectbox("Update Order Status",
                                              ["Needed", "Approved", "Ordered", "Shipped", "Received", "Installed"],
                                              index=["Needed", "Approved", "Ordered", "Shipped", "Received", "Installed"].index(
                                                  orders_df.at[order_index, "Status"]))
            approval_date = st.sidebar.date_input("Date of Approval")
            ordered_by = st.sidebar.text_input("Ordered By")
            # Ensure "Date Ordered" is either valid or set it to today's date
            date_ordered_value = pd.to_datetime(orders_df.at[order_index, "Date Ordered"], errors='coerce') if orders_df.at[order_index, "Date Ordered"] else pd.Timestamp.today()

            # Use a default value if the "Date Ordered" is invalid (NaT)
            if pd.isna(date_ordered_value):
                date_ordered_value = pd.Timestamp.today()

            date_ordered = st.sidebar.date_input("Date of Order", value=date_ordered_value)
            company = st.sidebar.text_input("Company Ordered From", value=orders_df.at[order_index, "Company"] if "Company" in orders_df.columns else "")
            approved_value = orders_df.at[order_index, "Approved"] if "Approved" in orders_df.columns else "No"
            approved = st.sidebar.selectbox("Approved", ["No", "Yes"], index=0 if approved_value == "No" else 1)
            approved_by = st.sidebar.text_input("Approved By", value=orders_df.at[order_index, "Approved By"] if "Approved By" in orders_df.columns else "")

            if st.sidebar.button("Update Order"):
                orders_df.at[order_index, "Part Name"] = updated_part_name  # Update Part Name
                orders_df.at[order_index, "Serial Number"] = updated_serial_number  # Update Serial Number
                orders_df.at[order_index, "Status"] = new_status
                if new_status == "Approved" and pd.isna(orders_df.at[order_index, "Approval Date"]):
                    approval_date = st.sidebar.date_input("Date of Approval", value=pd.Timestamp.today())  # Set today's date if not already set
                else:
                    approval_date = orders_df.at[order_index, "Approval Date"]  # Keep the existing approval date if it's already set
                orders_df.at[order_index, "Approval Date"] = approval_date
                orders_df.at[order_index, "Ordered By"] = ordered_by
                orders_df.at[order_index, "Company"] = company
                orders_df.at[order_index, "Approved"] = approved
                orders_df.at[order_index, "Approved By"] = approved_by
                orders_df.at[order_index, "Date Ordered"] = date_ordered.strftime("%Y-%m-%d")  # Update Date Ordered column with the new date
                orders_df.to_csv(ORDER_FILE, index=False)
                st.sidebar.success("Order updated!")

    if not orders_df.empty:
        # Define the new column order
        column_order = [
            "Status",
            "Part Name",
            "Serial Number",
            "Details",
            "Approved",
            "Approved By",
            "Approval Date",
            "Ordered By",
            "Date Ordered",
            "Company",

        ]
        # Reorganize the DataFrame columns
        orders_df = orders_df[column_order]
        # Remove "Date of Approval" if still present
        orders_df = orders_df.drop(columns=["Date of Approval"], errors='ignore')

        # Color-coding based on Status
        def highlight_status(row):
            color_map = {
            "Needed": "background-color: lightcoral; color: black",
            "Approved": "background-color: lightgreen; color: black",
            "Ordered": "background-color: lightyellow; color: black",
            "Shipped": "background-color: lightblue; color: black",
            "Received": "background-color: green; color: black",
            "Installed": "background-color: lightgray; color: black",
            }
            return [color_map.get(row["Status"], "")] + [""] * (len(row) - 1)

        # Sort the DataFrame by 'Status' so 'Needed' items come first
        status_order = ["Needed","Approved", "Ordered", "Shipped", "Received", "Installed"]
        orders_df['Status'] = pd.Categorical(orders_df['Status'], categories=status_order, ordered=True)
        orders_df = orders_df.sort_values('Status')

        st.dataframe(orders_df.style.apply(highlight_status, axis=1), height=400, use_container_width=True)

    else:
        st.warning("No orders have been placed yet.")


        # Archive installed parts
    if st.button("📦 Archive Installed Orders"):
            archive_file = "archived_orders.csv"
            installed_df = orders_df[orders_df["Status"].str.strip().str.lower() == "installed"]
            remaining_df = orders_df[orders_df["Status"].str.strip().str.lower() != "installed"]

            if not installed_df.empty:
                if os.path.exists(archive_file):
                    archived_df = pd.read_csv(archive_file)
                    archived_df = pd.concat([archived_df, installed_df], ignore_index=True)
                else:
                    archived_df = installed_df

                archived_df.to_csv(archive_file, index=False)
                remaining_df.to_csv(ORDER_FILE, index=False)
                orders_df = remaining_df
                st.success(f"{len(installed_df)} installed order(s) archived.")
            else:
                st.info("No installed parts to archive.")

        # Button to view archived orders
    if st.button("📂 View Archived Orders"):
            archive_file = "archived_orders.csv"
            if os.path.exists(archive_file):
                archived_df = pd.read_csv(archive_file)
                if not archived_df.empty:
                    st.write("### Archived Orders")

                    # Reorganize columns for the archived table to match the Order Tracking table
                    column_order = [
                        "Status", "Part Name", "Serial Number", "Details", "Approved", "Approved By", "Approval Date",
                        "Ordered By", "Date Ordered", "Company"
                    ]
                    archived_df = archived_df[column_order]
                    st.dataframe(archived_df, height=300, use_container_width=True)
                else:
                    st.info("The archive is currently empty.")
            else:
                st.info("Archive file does not exist yet.")

    # ------------------ Parts Datasheet (Hierarchical View) Section ------------------
    st.write("### Parts Datasheet (Hierarchical View)")


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

        selected_folder = st.selectbox(f"📂 Select folder in {os.path.basename(current_path)}:", [""] + folder_options, key=f"folder_{level}")

        if selected_folder:
            display_directory(os.path.join(current_path, selected_folder), level + 1)

        for file_path in files:
            file_name = os.path.basename(file_path)
            if st.button(f"📄 Open {file_name}", key=file_path):
                os.system(f"open {file_path}")  # For macOS, use `xdg-open` for Linux, `start` for Windows

    if os.path.exists(PARTS_DIRECTORY) and os.listdir(PARTS_DIRECTORY):
        display_directory(PARTS_DIRECTORY)
        # Add a delete option
    delete_row = st.selectbox("Select a part to delete", orders_df["Part Name"].tolist(), key="delete_part")
    if st.button("Delete Selected Part"):
            orders_df = orders_df[orders_df["Part Name"] != delete_row]
            orders_df.to_csv(ORDER_FILE, index=False)
            st.success(f"Deleted part: {delete_row}")
# ------------------ Development Tab ------------------
elif tab_selection == "🧪 Development Process":
    st.title("🧪 Development Process")
    st.sidebar.title("Manage R&D Projects")
    UPDATES_FILE = "experiment_updates.csv"
    if not os.path.exists(UPDATES_FILE):
        pd.DataFrame(columns=["Experiment Title", "Update Date", "Researcher", "Update Notes"]).to_csv(UPDATES_FILE,
                                                                                                       index=False)

    if not os.path.exists(DEVELOPMENT_FILE):
        pd.DataFrame(columns=["Project Name", "Project Purpose", "Target"]).to_csv(DEVELOPMENT_FILE, index=False)

    dev_df = pd.read_csv(DEVELOPMENT_FILE)
    archived_file = "archived_projects.csv"



    # ---- Add New Project ----
    st.sidebar.subheader("➕ Add New Project")
    new_project_name = st.sidebar.text_input("Project Name")
    new_project_purpose = st.sidebar.text_area("Project Purpose")
    new_project_target = st.sidebar.text_area("Target")

    if st.sidebar.button("Create Project"):
        if new_project_name:
            new_project_entry = pd.DataFrame([{
                "Project Name": new_project_name,
                "Project Purpose": new_project_purpose,
                "Target": new_project_target
            }])
            dev_df = pd.concat([dev_df, new_project_entry], ignore_index=True)
            dev_df.to_csv(DEVELOPMENT_FILE, index=False)
            st.sidebar.success("Project created successfully!")
            #st.rerun()
        else:
            st.sidebar.error("Project Name is required!")

    # ---- Show List of Projects ----
    st.sidebar.subheader("📂 Select a Project")
    active_projects = dev_df[~dev_df["Project Name"].isin(
        pd.read_csv("archived_projects.csv")["Project Name"].tolist()
    )] if os.path.exists("archived_projects.csv") else dev_df
    # Restore project from archive shortcut
    if "restored_project" in st.session_state:
        selected_project = st.session_state["restored_project"]
        # Refresh active_projects to include the restored project
        active_projects = dev_df[~dev_df["Project Name"].isin(
            pd.read_csv("archived_projects.csv")["Project Name"].tolist()
        )] if os.path.exists("archived_projects.csv") else dev_df
        del st.session_state["restored_project"]
    else:
        selected_project = st.sidebar.selectbox(
            "Choose a Project",
            [""] + active_projects["Project Name"].unique().tolist(),  # Add an empty string as the first option
        )
    if selected_project:
        st.subheader(f"Project Details: {selected_project}")

        # Retrieve project details
        project_rows = dev_df[dev_df["Project Name"] == selected_project]
        if not project_rows.empty:
            project_data = project_rows.iloc[0]
            st.write(f"**Project Purpose:** {project_data.get('Project Purpose', 'N/A')}")
            st.write(f"**Target:** {project_data.get('Target', 'N/A')}")
        else:
            st.warning(f"No project data found for '{selected_project}'. It may have been removed or archived.")

        # Display experiment details and draw data (CSV)
        project_experiments = dev_df[
            (dev_df["Project Name"] == selected_project) &
            (dev_df["Experiment Title"].notna()) &
            (dev_df["Date"].notna())
            ]


    # ---- Archive or Delete Project ----
    st.sidebar.subheader("📦 Manage Project")
    if selected_project:
        if st.sidebar.button("🗄️ Archive Project"):
            archived_df = pd.read_csv("archived_projects.csv") if os.path.exists("archived_projects.csv") else pd.DataFrame(columns=dev_df.columns)
            archived_df = pd.concat([archived_df, dev_df[dev_df["Project Name"] == selected_project]], ignore_index=True)
            archived_df.to_csv("archived_projects.csv", index=False)

            dev_df = dev_df[dev_df["Project Name"] != selected_project]  # Remove from active list
            dev_df.to_csv(DEVELOPMENT_FILE, index=False)
            st.sidebar.success("Project archived successfully!")
            #st.rerun()

        if st.sidebar.button("🗑️ Delete Project"):
            dev_df = dev_df[dev_df["Project Name"] != selected_project]  # Remove from list
            dev_df.to_csv(DEVELOPMENT_FILE, index=False)
            st.sidebar.warning("Project deleted permanently!")
            #st.rerun()
            # ---- Display Archived Projects ----
            st.subheader("📦 Archived Projects")
            archived_file = "archived_projects.csv"
            if os.path.exists(archived_file):
                archived_projects_df = pd.read_csv(archived_file)
                archived_projects = archived_projects_df["Project Name"].unique().tolist()
                selected_archived = st.selectbox("Select Archived Project", [""] + archived_projects,
                                                 key="archived_project_select")
                if selected_archived:
                    st.markdown(f"## 📋 Project Details: {selected_archived}")
                    archived_project_data = archived_projects_df[
                        archived_projects_df["Project Name"] == selected_archived]
                    if not archived_project_data.empty:
                        first_entry = archived_project_data.iloc[0]
                        st.markdown(f"**Project Purpose:** {first_entry.get('Project Purpose', 'N/A')}")
                        st.markdown(f"**Target:** {first_entry.get('Target', 'N/A')}")

                        experiments = archived_project_data[
                            archived_project_data["Experiment Title"].notna() &
                            archived_project_data["Date"].notna()
                            ]
                        if not experiments.empty:
                            st.subheader("🔬 Archived Experiments")
                            for _, exp in experiments.iterrows():
                                with st.expander(f"🧪 {exp['Experiment Title']} ({exp['Date']})"):
                                    st.markdown(f"**Researcher:** {exp.get('Researcher', 'N/A')}")
                                    st.markdown(f"**Methods:** {exp.get('Methods', 'N/A')}")
                                    st.markdown(f"**Purpose:** {exp.get('Purpose', 'N/A')}")
                                    st.markdown(f"**Observations:** {exp.get('Observations', 'N/A')}")
                                    st.markdown(f"**Results:** {exp.get('Results', 'N/A')}")
                    else:
                        st.warning("No data found for selected archived project.")
            else:
                st.info("No archived projects file available.")
        # ---- Add New Experiment ----
        show_add_experiment = st.checkbox("➕ Add Experiment to Project")
        # ---- Inside "Add Experiment" Section ----
        # Inside the Add Experiment section
        # Inside "Add Experiment" section
        # Inside "Add Experiment" section
        if show_add_experiment:
            st.subheader("➕ Add Experiment to Project")
            experiment_title = st.text_input("Experiment Title")
            methods = st.text_area("Methods")
            purpose = st.text_area("Experiment Purpose")
            date = st.date_input("Date")
            researcher = st.text_input("Researcher Name")
            observations = st.text_area("Observations")
            results = st.text_area("Results")
            show_drawing = st.checkbox("Is this a Drawing?", key=f"show_drawing_{selected_project}")

            if show_drawing:
                drawing_details = st.text_area("Enter Drawing Details", key=f"drawing_details_{selected_project}")

                # Check if the CSV is already stored in session state
                if 'selected_csv' in st.session_state and st.session_state.selected_csv:
                    selected_csv = st.session_state.selected_csv
                    st.write(f"Selected CSV: {selected_csv}")
                else:
                    # If not already selected, allow the user to choose a CSV from the dataset
                    dataset_files = [f for f in os.listdir('data_set_csv') if f.endswith('.csv')]



                    selected_csv = st.selectbox("Select CSV for Drawing Data", dataset_files, key="select_csv")
                # Load the CSV file and display it if selected
                if selected_csv:
                    csv_path = os.path.join('data_set_csv', selected_csv)
                    try:
                        draw_data = pd.read_csv(csv_path)
                        st.write("### CSV Data")
                        st.dataframe(draw_data)  # Display the CSV data as a table
                        # Store the CSV data in session state to persist it for this experiment
                        st.session_state.selected_csv_data = draw_data
                    except Exception as e:
                        st.error(f"Failed to load CSV: {e}")
            else:
                drawing_details = ""

            # Add Experiment Button
            if st.button("Add Experiment"):
                if experiment_title and date:
                    new_experiment = pd.DataFrame([{
                        "Project Name": selected_project,
                        "Experiment Title": experiment_title,
                        "Methods": methods,
                        "Purpose": purpose,
                        "Date": date.strftime("%Y-%m-%d"),
                        "Researcher": researcher,
                        "Observations": observations,
                        "Results": results,
                        "Is Drawing": show_drawing,
                        "Drawing Details": drawing_details if show_drawing else "",
                        "Draw Name": selected_csv.replace('.csv', '') if selected_csv else "",
                    }])

                    # Store CSV data along with experiment
                    if 'selected_csv_data' in st.session_state:
                        new_experiment["Draw Table"] = [st.session_state.selected_csv_data.to_dict(orient='records')]

                    dev_df = pd.read_csv(DEVELOPMENT_FILE) if os.path.exists(DEVELOPMENT_FILE) else pd.DataFrame(
                        columns=new_experiment.columns)
                    dev_df = pd.concat([dev_df, new_experiment], ignore_index=True)
                    dev_df.to_csv(DEVELOPMENT_FILE, index=False)
                    st.success("Experiment added successfully!")
                else:
                    st.warning("Please provide at least a title and date for the experiment.")

        # Display existing experiments
        # Display existing experiments
        project_experiments = dev_df[
            (dev_df["Project Name"] == selected_project) &
            (dev_df["Experiment Title"].notna()) &
            (dev_df["Date"].notna())
            ]

        if not project_experiments.empty:
            st.subheader("🔬 Experiments Conducted")
            for _, exp in project_experiments.iterrows():
                with st.expander(f"🧪 {exp['Experiment Title']} ({exp['Date']})"):
                    st.write(f"**Researcher:** {exp.get('Researcher', 'N/A')}")
                    st.write(f"**Methods:** {exp.get('Methods', 'N/A')}")
                    st.write(f"**Purpose:** {exp.get('Purpose', 'N/A')}")
                    st.write(f"**Observations:** {exp.get('Observations', 'N/A')}")
                    st.write(f"**Results:** {exp.get('Results', 'N/A')}")

                    # Check if 'Draw Name' exists before attempting to access it
                    if 'Draw Name' in exp:
                        st.write(f"**Drawing Name:** {exp['Draw Name']}")
                    else:
                        st.warning("No drawing name available for this experiment.")

                    # If CSV data is available for this experiment (stored in session state)
                    if 'selected_csv_data' in st.session_state:
                        st.write("### Draw Data (CSV) for this Experiment")
                        st.dataframe(st.session_state.selected_csv_data)

                    # Experiment updates
                    updates_df = pd.read_csv(UPDATES_FILE) if os.path.exists(UPDATES_FILE) else pd.DataFrame(
                        columns=["Experiment Title", "Update Date", "Researcher", "Update Notes"])
                    exp_updates = updates_df[updates_df["Experiment Title"] == exp["Experiment Title"]]

                    if not exp_updates.empty:
                        st.subheader("📜 Experiment Progress Updates")
                        for _, update in exp_updates.sort_values("Update Date").iterrows():
                            st.write(
                                f"📅 **{update['Update Date']}** - {update['Researcher']}: {update['Update Notes']}")

                    # ---- Update Experiment Progress Over Time ----
                    st.subheader("🔄 Update Experiment Progress")
                    update_researcher = st.text_input(f"Your name for update on {exp['Experiment Title']}",
                                                      key=f"researcher_{exp['Experiment Title']}")
                    update_notes = st.text_area(f"Add new progress update for {exp['Experiment Title']}",
                                                key=f"update_{exp['Experiment Title']}")
                    if st.button(f"Update {exp['Experiment Title']}", key=f"update_button_{exp['Experiment Title']}"):
                        new_update = pd.DataFrame([{
                            "Experiment Title": exp["Experiment Title"],
                            "Update Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                            "Researcher": update_researcher,
                            "Update Notes": update_notes
                        }])
                        updates_df = pd.concat([updates_df, new_update], ignore_index=True)
                        updates_df.to_csv(UPDATES_FILE, index=False)
                        st.success(f"Update added to {exp['Experiment Title']}!")
                        st.rerun()
        # ---- Final Conclusion for the Project ----
        st.subheader("📢 Project Conclusion")
        conclusion = st.text_area("Enter conclusion and final summary for this project",
                                  key=f"conclusion_{selected_project}")
    st.subheader("📦 Archived Projects")
    # Render quick access buttons to archived project views
    if os.path.exists(archived_file):
        archived_projects_df = pd.read_csv(archived_file)
        archived_projects = archived_projects_df["Project Name"].unique().tolist()
        selected_archived = st.selectbox("Select Archived Project", [""] + archived_projects,
                                         key="archived_project_select")
        if selected_archived:
            st.markdown(f"## 📋 Project Details: {selected_archived}")
            archived_project_data = archived_projects_df[archived_projects_df["Project Name"] == selected_archived]
            if not archived_project_data.empty:
                first_entry = archived_project_data.iloc[0]
                st.markdown(f"**Project Purpose:** {first_entry.get('Project Purpose', 'N/A')}")
                st.markdown(f"**Target:** {first_entry.get('Target', 'N/A')}")

                experiments = archived_project_data[
                    archived_project_data["Experiment Title"].notna() &
                    archived_project_data["Date"].notna()
                    ]
                if not experiments.empty:
                    st.subheader("🔬 Archived Experiments")
                    for _, exp in experiments.iterrows():
                        with st.expander(f"🧪 {exp['Experiment Title']} ({exp['Date']})"):
                            st.markdown(f"**Researcher:** {exp.get('Researcher', 'N/A')}")
                            st.markdown(f"**Methods:** {exp.get('Methods', 'N/A')}")
                            st.markdown(f"**Purpose:** {exp.get('Purpose', 'N/A')}")
                            st.markdown(f"**Observations:** {exp.get('Observations', 'N/A')}")
                            st.markdown(f"**Results:** {exp.get('Results', 'N/A')}")
            else:
                st.warning("No data found for selected archived project.")
    else:
        st.info("No archived projects file available.")
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
        d = st.session_state.get("maint_today", dt.date.today())
        state["current_date"] = d.isoformat() if isinstance(d, (dt.date, dt.datetime)) else str(d)

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

    default_date = dt.date.today()
    if isinstance(state.get("current_date"), str):
        try:
            default_date = dt.date.fromisoformat(state["current_date"])
        except Exception:
            default_date = dt.date.today()

    default_furnace = float(state.get("furnace_hours", 0.0) or 0.0)
    default_uv1 = float(state.get("uv1_hours", 0.0) or 0.0)
    default_uv2 = float(state.get("uv2_hours", 0.0) or 0.0)
    default_warn_days = int(state.get("warn_days", 14) or 14)
    default_warn_hours = float(state.get("warn_hours", 50.0) or 50.0)

    st.subheader("Current status inputs (saved)")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

    with c1:
        current_date = st.date_input("Today", value=default_date, key="maint_today", on_change=_persist_inputs)
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
    # INLINE "NEW DRAW DETECTED" PANEL
    # =========================================================
    last_draw_count = int(state.get("last_draw_count", current_draw_count))
    new_draws = current_draw_count - last_draw_count

    st.caption(
        f"📦 Draw CSVs in data_set_csv: **{current_draw_count}**"
        + (f"  |  🆕 new since last run: **{new_draws}**" if new_draws > 0 else "")
    )

    if new_draws > 0:
        st.warning(f"🆕 New draw detected! {new_draws} new draw CSV file(s) were added.")
        pre_df, post_df = pick_pre_post_tasks(dfm)

        with st.container(border=True):
            st.markdown("### ✅ Routine checklist (Pre / Post)")
            st.caption("From maintenance tasks where Tracking Mode = 'event'. (Keyword-based split for now.)")
            colA, colB = st.columns(2)

            with colA:
                st.markdown("#### Pre-Draw")
                if pre_df.empty:
                    st.info("No routine tasks found (Tracking_Mode='event'). Add routine tasks in maintenance Excel.")
                else:
                    for i, r in pre_df.iterrows():
                        label = f"{str(r.get('Component','')).strip()} — {str(r.get('Task','')).strip()}"
                        st.checkbox(label, key=f"inline_pre_{r.get('Source_File','')}_{r.get('Task_ID','')}_{i}")

            with colB:
                st.markdown("#### Post-Draw")
                if post_df.empty:
                    st.info("No post tasks matched keywords. Add 'post/after/shutdown' in Task/Notes or add a Phase column later.")
                else:
                    for i, r in post_df.iterrows():
                        label = f"{str(r.get('Component','')).strip()} — {str(r.get('Task','')).strip()}"
                        st.checkbox(label, key=f"inline_post_{r.get('Source_File','')}_{r.get('Task_ID','')}_{i}")

            st.divider()
            if st.button("✅ Acknowledge checklist (hide until next new draw)", type="primary", key="ack_draw_checklist_inline"):
                state["last_draw_count"] = current_draw_count
                save_state(STATE_PATH, state)
                st.success("Acknowledged. This panel will stay hidden until a new draw CSV is added.")
                st.rerun()
    else:
        state["last_draw_count"] = current_draw_count
        save_state(STATE_PATH, state)

    # =========================================================
    # Dashboard summary
    # =========================================================
    st.subheader("Dashboard")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("OVERDUE", int((dfm["Status"] == "OVERDUE").sum()))
    s2.metric("DUE SOON", int((dfm["Status"] == "DUE SOON").sum()))
    s3.metric("ROUTINE (every activity)", int((dfm["Status"] == "ROUTINE").sum()))
    s4.metric("OK", int((dfm["Status"] == "OK").sum()))

    # =========================================================
    # FUTURE TIMELINE VIEW (BUTTONS)  - prettier & symmetric
    # =========================================================
    st.subheader("📅 Future schedule view")

    # --- UI polish (applies to buttons on the page) ---
    st.markdown(
        """
        <style>
        /* Make all Streamlit buttons consistent + no text wrapping */
        div.stButton > button {
            width: 100%;
            height: 44px;
            border-radius: 12px;
            font-weight: 600;
            padding: 0 10px;
        }
        div.stButton > button p {
            white-space: nowrap !important;
            margin: 0 !important;
        }
        /* Slightly tighter column gaps */
        div[data-testid="column"] { padding-left: 0.35rem; padding-right: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # defaults (small)
    if "maint_horizon_hours" not in st.session_state:
        st.session_state["maint_horizon_hours"] = 10
    if "maint_horizon_days" not in st.session_state:
        st.session_state["maint_horizon_days"] = 7
    if "maint_horizon_draws" not in st.session_state:
        st.session_state["maint_horizon_draws"] = 5


    def _button_group(title: str, options, current_value, key_prefix: str):
        st.caption(title)
        cols = st.columns(len(options), gap="small")
        for col, (label, value) in zip(cols, options):
            btn_type = "primary" if value == current_value else "secondary"
            if col.button(label, key=f"{key_prefix}_{value}", type=btn_type, use_container_width=True):
                return value
        return current_value


    g1, g2, g3 = st.columns(3, gap="large")

    with g1:
        st.session_state["maint_horizon_hours"] = _button_group(
            "Hours horizon",
            options=[("10", 10), ("50", 50), ("100", 100)],
            current_value=st.session_state["maint_horizon_hours"],
            key_prefix="mh"
        )

    with g2:
        st.session_state["maint_horizon_days"] = _button_group(
            "Calendar horizon",
            options=[("Week", 7), ("Month", 30), ("3 \n Months", 90)],
            current_value=st.session_state["maint_horizon_days"],
            key_prefix="md"
        )

    with g3:
        st.session_state["maint_horizon_draws"] = _button_group(
            "Draw horizon",
            options=[("5", 5), ("10", 10), ("50", 50)],
            current_value=st.session_state["maint_horizon_draws"],
            key_prefix="mD"
        )

    horizon_hours = int(st.session_state["maint_horizon_hours"])
    horizon_days = int(st.session_state["maint_horizon_days"])
    horizon_draws = int(st.session_state["maint_horizon_draws"])

    st.caption(
        f"📦 Now: **{current_draw_count}** draws → Horizon: **{horizon_draws}** draws → Up to draw **#{current_draw_count + horizon_draws}**"
    )

    # =========================================================
    # Roadmap Plotly helpers (same style for all)
    # =========================================================
    def _status_color(s: str) -> str:
        s = "" if s is None else str(s).strip().upper()
        if s == "OVERDUE":
            return "#ff4d4d"
        if s == "DUE SOON":
            return "#ffcc00"
        return "#66ff99"

    def _roadmap_figure(x0, x1, title, x_label, points_df, x_col, hover_col):
        """
        Draws a clean 1-line roadmap with a NOW marker and due points.
        Works for numeric axes and datetime axes.
        """
        fig = go.Figure()

        # baseline
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[0, 0],
            mode="lines",
            line=dict(width=6, color="rgba(180,180,180,0.20)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        # NOW marker
        fig.add_vline(x=x0, line_width=2, line_dash="dash", line_color="rgba(255,255,255,0.75)")

        # points
        if points_df is not None and not points_df.empty:
            fig.add_trace(go.Scatter(
                x=points_df[x_col],
                y=[0] * len(points_df),
                mode="markers",
                marker=dict(
                    size=13,
                    color=[_status_color(s) for s in points_df["Status"].tolist()],
                    line=dict(width=1, color="rgba(255,255,255,0.55)")
                ),
                hovertemplate="%{text}<extra></extra>",
                text=points_df[hover_col].tolist(),
                showlegend=False
            ))
        else:
            # midpoint label (Timestamp-safe)
            if isinstance(x0, (pd.Timestamp, dt.date, dt.datetime)) or isinstance(x1, (pd.Timestamp, dt.date, dt.datetime)):
                x_mid = pd.Timestamp(x0) + (pd.Timestamp(x1) - pd.Timestamp(x0)) / 2
            else:
                x_mid = (float(x0) + float(x1)) / 2.0

            fig.add_annotation(
                x=x_mid,
                y=0,
                text="No tasks in horizon",
                showarrow=False,
                font=dict(size=14, color="rgba(255,255,255,0.70)"),
                yshift=26
            )

        fig.update_layout(
            title=title,
            height=220,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(visible=False, range=[-1, 1]),
            xaxis=dict(
                title=x_label,
                range=[x0, x1],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False,
            ),
        )
        return fig

    # =========================================================
    # Build due datasets for plotting
    # =========================================================
    def _norm_group(src):
        s = "" if src is None or pd.isna(src) else str(src).strip().lower()
        if s in ("uv1", "uv 1", "uv_system_1", "uv system 1", "uv-system-1", "system1", "system 1"):
            return "UV1"
        if s in ("uv2", "uv 2", "uv_system_2", "uv system 2", "uv-system-2", "system2", "system 2"):
            return "UV2"
        return "FURNACE"

    def _hours_due_df(df_all):
        d = df_all[df_all["Tracking_Mode_norm"].eq("hours")].copy()
        d["Due_Hours"] = pd.to_numeric(d["Next_Due_Hours"], errors="coerce")
        d = d[d["Due_Hours"].notna()].copy()
        d["Group"] = d["Hours_Source"].apply(_norm_group)
        d["Label"] = (
            d["Component"].fillna("").astype(str).str.strip()
            + " — "
            + d["Task"].fillna("").astype(str).str.strip()
        ).str.strip(" —")
        d["Hover"] = d["Label"] + "<br>Status: " + d["Status"].astype(str) + "<br>Due: " + d["Due_Hours"].astype(str)
        return d

    def _calendar_due_df(df_all):
        d = df_all[df_all["Tracking_Mode_norm"].eq("calendar")].copy()
        d["Due_Date"] = pd.to_datetime(d["Next_Due_Date"], errors="coerce")
        d = d[d["Due_Date"].notna()].copy()
        d["Label"] = (
            d["Component"].fillna("").astype(str).str.strip()
            + " — "
            + d["Task"].fillna("").astype(str).str.strip()
        ).str.strip(" —")
        d["Hover"] = d["Label"] + "<br>Status: " + d["Status"].astype(str) + "<br>Due: " + d["Due_Date"].dt.date.astype(str)
        return d

    def _draw_due_df(df_all):
        d = df_all[df_all["Tracking_Mode_norm"].eq("draws")].copy()
        d["Due_Draw"] = pd.to_numeric(d["Next_Due_Draw"], errors="coerce")
        d = d[d["Due_Draw"].notna()].copy()
        d["Label"] = (
            d["Component"].fillna("").astype(str).str.strip()
            + " — "
            + d["Task"].fillna("").astype(str).str.strip()
        ).str.strip(" —")
        d["Hover"] = d["Label"] + "<br>Status: " + d["Status"].astype(str) + "<br>Due draw: " + d["Due_Draw"].astype(int).astype(str)
        return d

    df_hours_due = _hours_due_df(dfm)
    df_cal_due = _calendar_due_df(dfm)
    df_draw_due = _draw_due_df(dfm)

    # =========================================================
    # Render plots (same nice style)
    # =========================================================
    st.markdown("### 🔥 Furnace / 💡 UV1 / 💡 UV2 (hours timelines)")
    t1, t2, t3 = st.columns(3)

    # Furnace
    with t1:
        now_h = float(furnace_hours)
        x0 = now_h
        x1 = now_h + float(horizon_hours)
        pts = df_hours_due[(df_hours_due["Group"] == "FURNACE") & (df_hours_due["Due_Hours"].between(x0, x1))].copy()
        st.plotly_chart(
            _roadmap_figure(x0, x1, "FURNACE (next hours)", "Hours", pts, "Due_Hours", "Hover"),
            use_container_width=True
        )

    # UV1
    with t2:
        now_h = float(uv1_hours)
        x0 = now_h
        x1 = now_h + float(horizon_hours)
        pts = df_hours_due[(df_hours_due["Group"] == "UV1") & (df_hours_due["Due_Hours"].between(x0, x1))].copy()
        st.plotly_chart(
            _roadmap_figure(x0, x1, "UV1 (next hours)", "Hours", pts, "Due_Hours", "Hover"),
            use_container_width=True
        )

    # UV2
    with t3:
        now_h = float(uv2_hours)
        x0 = now_h
        x1 = now_h + float(horizon_hours)
        pts = df_hours_due[(df_hours_due["Group"] == "UV2") & (df_hours_due["Due_Hours"].between(x0, x1))].copy()
        st.plotly_chart(
            _roadmap_figure(x0, x1, "UV2 (next hours)", "Hours", pts, "Due_Hours", "Hover"),
            use_container_width=True
        )

    # Draw number timeline
    st.markdown("### 🧵 Draw number timeline")
    d0 = int(current_draw_count)
    d1 = int(current_draw_count + horizon_draws)
    pts = df_draw_due[df_draw_due["Due_Draw"].between(d0, d1)].copy()
    st.plotly_chart(
        _roadmap_figure(d0, d1, f"Draw-based due points (next {horizon_draws} draws)", "Draw number", pts, "Due_Draw", "Hover"),
        use_container_width=True
    )

    # Calendar timeline
    st.markdown("### 🗓️ Calendar timeline")
    now_ts = pd.Timestamp(current_date)
    end_ts = now_ts + pd.Timedelta(days=int(horizon_days))
    pts = df_cal_due[(df_cal_due["Due_Date"] >= now_ts) & (df_cal_due["Due_Date"] <= end_ts)].copy()
    st.plotly_chart(
        _roadmap_figure(now_ts, end_ts, f"Calendar due points (next {horizon_days} days)", "Date", pts, "Due_Date", "Hover"),
        use_container_width=True
    )

    # =========================================================
    # Mark Done (updates source Excel/CSV)
    # =========================================================
    st.subheader("Mark tasks as done (updates the source Excel/CSV)")

    focus_default = ["OVERDUE", "DUE SOON", "ROUTINE"]
    focus_status = st.multiselect(
        "Work on these statuses",
        ["OVERDUE", "DUE SOON", "ROUTINE", "OK"],
        default=focus_default,
        key="maint_focus_status"
    )

    work = dfm[dfm["Status"].isin(focus_status)].copy().sort_values(["Status", "Component", "Task"])
    work["Done_Now"] = False

    editor_cols = [
        "Done_Now",
        "Status", "Component", "Task", "Task_ID",
        "Tracking_Mode", "Hours_Source", "Current_Hours_For_Task",
        "Interval_Value", "Interval_Unit",
        "Last_Done_Date", "Last_Done_Hours", "Last_Done_Draw",
        "Next_Due_Date", "Next_Due_Hours", "Next_Due_Draw",
        "Manual_Name", "Page", "Document",
        "Owner", "Source_File"
    ]
    editor_cols = [c for c in editor_cols if c in work.columns]

    edited = st.data_editor(
        work[editor_cols],
        use_container_width=True,
        height=420,
        column_config={
            "Done_Now": st.column_config.CheckboxColumn("Done now", help="Tick tasks you completed."),
        },
        disabled=[c for c in editor_cols if c != "Done_Now"],
        key="maint_editor"
    )

    def apply_updates_to_sources(done_rows: pd.DataFrame):
        if done_rows.empty:
            return 0, []

        updated_rows = 0
        problems = []

        for src, grp in done_rows.groupby("Source_File"):
            path = os.path.join(MAINT_FOLDER, src)
            try:
                raw = read_file(path)
                if raw is None or raw.empty:
                    problems.append((src, "File empty/unreadable"))
                    continue

                original_cols = list(raw.columns)
                df_src = normalize_df(raw)
                df_src["Tracking_Mode_norm"] = df_src["Tracking_Mode"].apply(mode_norm)

                for _, r in grp.iterrows():
                    tid = r.get("Task_ID", np.nan)
                    comp = str(r.get("Component", "")).strip()
                    task = str(r.get("Task", "")).strip()
                    mode = mode_norm(r.get("Tracking_Mode", ""))
                    hs = r.get("Hours_Source", "")

                    if tid is not None and not pd.isna(tid) and str(tid).strip() != "" and "Task_ID" in df_src.columns:
                        mask = df_src["Task_ID"].astype(str).str.strip() == str(tid).strip()
                    else:
                        mask = (
                            df_src["Component"].astype(str).str.strip().eq(comp) &
                            df_src["Task"].astype(str).str.strip().eq(task)
                        )

                    if not mask.any():
                        problems.append((src, f"Not found: {comp} / {task} / Task_ID={tid}"))
                        continue

                    if mode == "hours":
                        cur_hours = pick_current_hours(hs)
                        df_src.loc[mask, "Last_Done_Hours"] = float(cur_hours)
                        df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()
                    elif mode == "draws":
                        df_src.loc[mask, "Last_Done_Draw"] = int(current_draw_count)
                        df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()
                    else:
                        df_src.loc[mask, "Last_Done_Date"] = current_date.isoformat()

                    updated_rows += int(mask.sum())

                out = templateize_df(df_src.drop(columns=["Tracking_Mode_norm"], errors="ignore"), original_cols)
                write_file(path, out)

            except Exception as e:
                problems.append((src, str(e)))

        return updated_rows, problems

    if st.button("✅ Apply 'Done Now' updates", type="primary", key="maint_apply_done"):
        done_rows = edited[edited["Done_Now"] == True].copy()
        if done_rows.empty:
            st.info("No tasks selected.")
        else:
            updated_rows, problems = apply_updates_to_sources(done_rows)
            st.success(f"Updated {updated_rows} task row(s) in maintenance files.")
            if problems:
                st.warning("Some updates had issues:")
                st.dataframe(pd.DataFrame(problems, columns=["File", "Problem"]), use_container_width=True)
            st.rerun()

    # =========================================================
    # Load report
    # =========================================================
    with st.expander("Load report"):
        st.write("Loaded files:", sorted(files))
        if load_errors:
            st.warning("Some files failed to load:")
            st.dataframe(pd.DataFrame(load_errors, columns=["File", "Error"]), use_container_width=True)

    # =========================================================
    # MANUALS BROWSER (files OR folders)
    # Folder: ./manuals
    # =========================================================
    import pathlib
    import subprocess
    import sys

    MANUALS_FOLDER = os.path.join(BASE_DIR, "manuals")
    os.makedirs(MANUALS_FOLDER, exist_ok=True)

    st.subheader("📚 Manuals")
    st.caption(f"Put manuals directly in: {MANUALS_FOLDER} (PDF, DOCX, XLSX, images…)")


    def open_file_cross_platform(path: str):
        path = os.path.abspath(path)
        try:
            if sys.platform.startswith("darwin"):  # macOS
                subprocess.Popen(["open", path])
            elif os.name == "nt":  # Windows
                os.startfile(path)  # type: ignore
            else:  # Linux
                subprocess.Popen(["xdg-open", path])
            return True, ""
        except Exception as e:
            return False, str(e)


    def list_files(folder: str):
        exts = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv",
                ".txt", ".png", ".jpg", ".jpeg")
        p = pathlib.Path(folder)
        return sorted(
            [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts and not x.name.startswith("~$")],
            key=lambda x: x.name.lower()
        )


    def list_subfolders(folder: str):
        p = pathlib.Path(folder)
        return sorted([x.name for x in p.iterdir() if x.is_dir()])


    # --- folder selector ---
    subfolders = list_subfolders(MANUALS_FOLDER)
    folder_options = ["(root)"] + subfolders

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_folder = st.selectbox("Location", folder_options, key="manuals_location")

    if selected_folder == "(root)":
        active_path = MANUALS_FOLDER
    else:
        active_path = os.path.join(MANUALS_FOLDER, selected_folder)

    files = list_files(active_path)

    with c2:
        if not files:
            st.info("No manuals found here.")
        else:
            selected_file = st.selectbox(
                "Select manual",
                [f.name for f in files],
                key="manuals_file"
            )
            full_path = os.path.join(active_path, selected_file)

            colA, colB = st.columns([1, 2])
            with colA:
                if st.button("📄 Open", key="manuals_open", use_container_width=True):
                    ok, err = open_file_cross_platform(full_path)
                    if ok:
                        st.success("Opened")
                    else:
                        st.error(err)

            with colB:
                st.code(full_path, language="text")