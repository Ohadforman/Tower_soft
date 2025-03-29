import streamlit as st
## 🔧 Required for capturing plot clicks (install with `pip install streamlit-plotly-events`)
from streamlit_plotly_events import plotly_events  # Ensure package is installed via: pip install streamlit-plotly-events
import base64
import pandas as pd
import plotly.express as px
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import math


DEVELOPMENT_FILE = "development_process.csv"

tab_labels = [
    "🏠 Home",
    #"📐 Calculators",
    #"📊 Logs & Visualization",
    "📅 Schedule",
    "🛠️ Tower Parts",
    "🧪 Development Process",
    #"📂 Archive",
    "💧 Coating",
    "🔍 Iris Selection",
    "📊 Dashboard",
    "📝 History Log",
    "✅ Closed Processes",
    "📋 Protocols"
]
if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = None

if "tab_select" not in st.session_state:
    st.session_state["tab_select"] = "🏠 Home"
if "last_tab" not in st.session_state:
    st.session_state["last_tab"] = "🏠 Home"
    if "tab_labels" not in st.session_state:
        st.session_state["tab_labels"] = tab_labels

# Use the last selected tab unless explicitly switching
if st.session_state.get("selected_tab"):
    st.session_state["tab_select"] = st.session_state["selected_tab"]
    st.session_state["last_tab"] = st.session_state["selected_tab"]
st.session_state["selected_tab"] = None
if "good_zones" not in st.session_state:
    st.session_state["good_zones"] = []

 # Initialize tab navigation state to avoid jumps back to "🏠 Home" on rerun

def switch_tab(tab_name, parent_tab=None):
    if parent_tab:
        st.session_state["parent_tab"] = parent_tab  # Ensure parent tab is stored persistently
    if tab_name not in ["💧 Coating", "🔍 Iris Selection", "📊 Dashboard", "📝 History Log", "✅ Closed Processes"]:
        st.session_state["last_tab"] = tab_name
    st.session_state["selected_tab"] = tab_name
    st.rerun()

def get_base64_image(image_path):
    """Encodes an image to base64 format for inline CSS."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# Ensure the image file exists in the project folder
image_base64 = get_base64_image("Group-25.jpg")
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
with open("config_coating.json", "r") as config_file:
    config = json.load(config_file)

DATA_FOLDER = config.get("logs_directory", "./logs")
HISTORY_FILE = "history_log.csv"
PARTS_DIRECTORY = config.get("parts_directory", "./parts")

tab_labels = [
    "🏠 Home",
    #"📐 Calculators",
    #"📊 Logs & Visualization",
    "📅 Schedule",
    "🛠️ Tower Parts",
    "🧪 Development Process",
    #"📂 Archive",
    "💧 Coating",
    "🔍 Iris Selection",
    "📊 Dashboard",
    "📝 History Log",
    "✅ Closed Processes",
    "📋 Protocols"
]

if st.session_state.get("selected_tab"):
    tab_selection = st.session_state["selected_tab"]
    st.session_state["last_tab"] = tab_selection
    st.session_state["selected_tab"] = None
else:
    default_tab = st.session_state.get("last_tab", "🏠 Home")
    if default_tab not in tab_labels:
        default_tab = "🏠 Home"
        st.session_state["last_tab"] = default_tab
    tab_selection = st.sidebar.radio("Navigation", tab_labels, key="tab_select", index=tab_labels.index(default_tab))
## Parent Tab Quick Links Navigation
if tab_selection == "📐 Calculators":
    st.title("📐 Calculators")
    st.subheader("Select a tool to perform calculations")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💧 Coating Calculation", key="btn_coating"):
            switch_tab("💧 Coating", parent_tab="📐 Calculators")
    with col2:
        if st.button("🔍 Iris Selection", key="btn_iris"):
            switch_tab("🔍 Iris Selection", parent_tab="📐 Calculators")
elif tab_selection == "📊 Logs & Visualization":
    st.title("📊 Logs & Visualization")
    st.subheader("Analyze and visualize tower operations")
    if st.button("📊 Open Dashboard", key="btn_dashboard"):
        switch_tab("📊 Dashboard", parent_tab="📊 Logs & Visualization")

elif tab_selection == "📂 Archive":
    st.title("📂 Archive")
    st.subheader("View historical records and closed processes")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 View History Log", key="btn_history"):
            switch_tab("📝 History Log", parent_tab="📂 Archive")
    with col2:
        if st.button("✅ View Closed Processes", key="btn_closed"):
            switch_tab("✅ Closed Processes", parent_tab="📂 Archive")

    st.subheader("📦 Archived Projects")
    archived_file = "archived_projects.csv"

    # Render quick access buttons to archived project views
    if os.path.exists(archived_file):
        archived_projects_df = pd.read_csv(archived_file)
        archived_projects = archived_projects_df["Project Name"].unique().tolist()
        selected_archived = st.selectbox("Select Archived Project", [""] + archived_projects, key="archived_project_select")
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

# ------------------ Quick Link Sections Return Button ------------------
if tab_selection in ["💧 Coating", "🔍 Iris Selection", "📊 Dashboard", "📝 History Log", "✅ Closed Processes"]:
    parent_tab = st.session_state.get("parent_tab", None)
    if parent_tab:
        if st.button(f"⬅️ Return to {parent_tab}", key=f"return_{parent_tab}"):
            st.session_state["selected_tab"] = parent_tab
            st.session_state["last_tab"] = parent_tab
            st.session_state["tab_select"] = parent_tab
            del st.session_state["parent_tab"]  # Remove after successful navigation
            st.rerun()

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

# ------------------ Home Tab ------------------
if tab_selection == "🏠 Home":
    st.title("️ Tower Management Software")
    st.subheader("A Smart Way to Manage Tower Operations Efficiently")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        /* Sidebar background */
        .css-1aumxhk {{
            background-color: rgba(20, 20, 20, 0.90) !important;
        }}
        /* Sidebar title */
        .css-1l02zno {{
            color: #FFFFFF !important; /* Force white text for dark themes */
            font-size: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        /* Sidebar text */
        .css-1d391kg, .css-qrbaxs, .css-1y4p8pa {{
            color: #FFFFFF !important; /* Ensure white text */
            font-size: 18px;
            font-weight: 700;
        }}
        /* Active Sidebar Link */
        .css-1y4p8pa[aria-selected="true"] {{
            color: #FFD700 !important; /* Gold for active selection */
            font-weight: bold;
        }}
        /* Hover Effect */
        .css-1y4p8pa:hover {{
            color: #B0C4DE !important; /* Light steel blue hover effect */
        }}
        /* Main Title Styling */
        h1 {{
            color: #FFFFFF !important; /* White for contrast */
            font-size: 38px !important;
            font-weight: bold !important;
            text-align: center;
            margin-top: 20px;
        }}
        /* Subtitle Styling */
        h2 {{
            color: #DDDDDD !important; /* Light gray for dark themes */
            font-size: 24px !important;
            font-style: italic;
            text-align: center;
            margin-top: -10px;
        }}
        /* Fallback for Light Mode */
        @media (prefers-color-scheme: light) {{
            h1 {{
                color: #000000 !important; /* Black for light themes */
            }}
            h2 {{
                color: #333333 !important; /* Dark gray for subtitle */
            }}
            .css-1l02zno {{
                color: #000000 !important;
            }}
            .css-1d391kg, .css-qrbaxs, .css-1y4p8pa {{
                color: #000000 !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# ------------------ Dashboard Tab ------------------
elif tab_selection == "📊 Dashboard":
    st.title(f"Draw Tower Logs Dashboard - {selected_file}")

    # Dashboard-specific sidebar controls
    st.sidebar.title("Data Selection")
    plot_type = st.sidebar.radio("Select Plot Type", ["2D", "3D"], key="plot_type")

    if plot_type == "3D":
        x_axis = st.sidebar.selectbox("Select X-axis", column_options, key="x_axis_select")
        y_axis = st.sidebar.selectbox("Select Y-axis", column_options, key="y_axis_select")
        z_axis = st.sidebar.selectbox("Select Z-axis", column_options, key="z_axis_select")
        if x_axis in df.columns and y_axis in df.columns and z_axis in df.columns:
            plot_df_3d = df[[x_axis, y_axis, z_axis]].dropna().copy()
            fig_3d = px.scatter_3d(plot_df_3d, x=x_axis, y=y_axis, z=z_axis, title=f"3D Plot: {z_axis} vs {y_axis} vs {x_axis}")
            st.plotly_chart(fig_3d, use_container_width=True, key="plot_3d")
        else:
            st.warning("Please select valid columns for X, Y, and Z axes.")

    show_corr_matrix = st.sidebar.checkbox("Show Correlation Matrix", key="corr_matrix")


    st.subheader("📏 Mark Good Zones (Click on the Plot)")

    # Create unified plot(s) for selected axes (2D only)
    if plot_type == "2D":
        if "plot_configs" not in st.session_state:
            default_x = column_options[0] if column_options else ""
            default_y = column_options[0] if column_options else ""
            st.session_state["plot_configs"] = [{"x_axis": default_x, "y_axis": default_y}]

        if st.sidebar.button("Add Plot", key="add_plot_button"):
            default_x = column_options[0] if column_options else ""
            default_y = column_options[0] if column_options else ""
            st.session_state["plot_configs"].append({"x_axis": default_x, "y_axis": default_y})

        for i, config in enumerate(st.session_state["plot_configs"]):
            config["x_axis"] = st.sidebar.selectbox(
                f"Select X-axis for Plot {i+1}", column_options,
                key=f"x_axis_select_{i}",
                index=column_options.index(config["x_axis"]) if config["x_axis"] in column_options else 0
            )
            config["y_axis"] = st.sidebar.selectbox(
                f"Select Y-axis for Plot {i+1}", column_options,
                key=f"y_axis_select_{i}",
                index=column_options.index(config["y_axis"]) if config["y_axis"] in column_options else 0
            )

            if config["x_axis"] in df.columns and config["y_axis"] in df.columns:
                filtered_df = df.dropna(subset=[config["x_axis"], config["y_axis"]])
                fig_plot = px.line(
                    filtered_df,
                    x=config["x_axis"],
                    y=config["y_axis"],
                    title=f"{config['y_axis']} vs {config['x_axis']} (Plot {i+1})",
                    markers=True
                )

                for start, end in st.session_state.get("good_zones", []):
                    fig_plot.add_vrect(
                        x0=start, x1=end,
                        fillcolor="green", opacity=0.3, line_width=0,
                        annotation_text="Good Zone", annotation_position="top left"
                    )
                selected_points_raw = plotly_events(
                    fig_plot, click_event=True, hover_event=False,
                    key=f"zone_click_plot_{i}"
                )
                st.plotly_chart(fig_plot, use_container_width=True, key=f"final_plot_{i}")
            else:
                st.warning(f"Plot {i+1}: Selected axes are not valid columns in the dataset.")

    if "zone_click_mode" not in st.session_state:
        st.session_state["zone_click_mode"] = "Start"
    if "current_zone" not in st.session_state:
        st.session_state["current_zone"] = {}
    zone_mode = st.session_state["zone_click_mode"]
    st.write("### Click to Mark Zone")
    if selected_points_raw:
        st.write(f"Clicked Point X: {selected_points_raw[0]['x']}")
        if st.button("Mark Zone", key="mark_zone_button"):
            raw_time = selected_points_raw[0]["x"]
            try:
                point_time = pd.to_datetime(raw_time, errors='raise')
            except Exception:
                try:
                    if isinstance(raw_time, str) and len(raw_time.split(":")[-1]) > 2:
                        parts = raw_time.rsplit(":", 1)
                        fixed_time = parts[0] + ":" + parts[1][:2] + "." + parts[1][2:]
                        point_time = pd.to_datetime(fixed_time)
                    else:
                        raise
                except Exception as e:
                    st.error(f"⚠️ Failed to parse timestamp: {raw_time}\n\nError: {e}")
                    st.stop()
            if zone_mode == "Start":
                st.session_state["current_zone"]["start"] = point_time
                st.success(f"Start of zone set to: {point_time}")
            elif zone_mode == "End":
                st.session_state["current_zone"]["end"] = point_time
                st.success(f"End of zone set to: {point_time}")

            if "start" in st.session_state["current_zone"] and "end" in st.session_state["current_zone"]:
                zone = (
                    min(st.session_state["current_zone"]["start"], st.session_state["current_zone"]["end"]),
                    max(st.session_state["current_zone"]["start"], st.session_state["current_zone"]["end"])
                )
                st.session_state["good_zones"].append(zone)
                st.success(f"Zone added from {zone[0]} to {zone[1]}")
                st.session_state["current_zone"] = {}
    if "good_zones" not in st.session_state:
        st.session_state["good_zones"] = []

    st.write("## Zone Selection Mode")
    zone_mode = st.radio("Click Mode", ["Start", "End"], key="zone_click_mode")
    selected_points = []

    if selected_points and st.button("Mark Zone"):
        raw_time = selected_points[0]["x"]
        try:
            point_time = pd.to_datetime(raw_time, errors='raise')
        except Exception:
            try:
                # Fix malformed timestamps like "13:16:48813" → "13:16:48.813"
                if isinstance(raw_time, str) and len(raw_time.split(":")[-1]) > 2:
                    parts = raw_time.rsplit(":", 1)
                    fixed_time = parts[0] + ":" + parts[1][:2] + "." + parts[1][2:]
                    point_time = pd.to_datetime(fixed_time)
                else:
                    raise
            except Exception as e:
                st.error(f"⚠️ Failed to parse timestamp: {raw_time}\n\nError: {e}")
                st.stop()
        if zone_mode == "Start":
            st.session_state["current_zone"]["start"] = point_time
            st.success(f"Start of zone set to: {point_time}")
        elif zone_mode == "End":
            st.session_state["current_zone"]["end"] = point_time
            st.success(f"End of zone set to: {point_time}")

        # When both start and end are selected, add to good_zones
        if "start" in st.session_state["current_zone"] and "end" in st.session_state["current_zone"]:
            zone = (
                min(st.session_state["current_zone"]["start"], st.session_state["current_zone"]["end"]),
                max(st.session_state["current_zone"]["start"], st.session_state["current_zone"]["end"])
            )
            st.session_state["good_zones"].append(zone)
            st.success(f"Zone added from {zone[0]} to {zone[1]}")
            st.session_state["current_zone"] = {}

    if st.session_state["good_zones"]:
        st.write("### Good Zones Summary")
        summary_data = []
        global_start = None
        global_end = None
        all_values = []

        for start, end in st.session_state["good_zones"]:
            if "Date/Time" in df.columns:
                zone_data = df[(df["Date/Time"] >= pd.to_datetime(start)) & (df["Date/Time"] <= pd.to_datetime(end))]
                if not zone_data.empty:
                    global_start = min(global_start, start) if global_start else start
                    global_end = max(global_end, end) if global_end else end
                    y_axis_selected = st.session_state["plot_configs"][0]["y_axis"] if st.session_state.get("plot_configs") else (column_options[0] if column_options else "")
                    all_values.extend(zone_data[y_axis_selected].tolist())

        if all_values:
            st.markdown("### 📊 Combined Good Zone Stats")
            st.write(f"**Start:** {global_start}")
            st.write(f"**End:** {global_end}")
            st.write(f"**Average:** {pd.Series(all_values).mean():.4f}")
            st.write(f"**Min:** {min(all_values):.4f}")
            st.write(f"**Max:** {max(all_values):.4f}")
        for i, (start, end) in enumerate(st.session_state["good_zones"]):
            if "Date/Time" in df.columns:
                zone_data = df[(df["Date/Time"] >= pd.to_datetime(start)) & (df["Date/Time"] <= pd.to_datetime(end))]
                if not zone_data.empty:
                    y_axis_selected = st.session_state["plot_configs"][0]["y_axis"] if st.session_state.get(
                        "plot_configs") else (column_options[0] if column_options else "")
                    summary = {
                        "Zone": f"Zone {i + 1}",
                        "Start": start,
                        "End": end,
                        "Avg": zone_data[y_axis_selected].mean(),
                        "Min": zone_data[y_axis_selected].min(),
                        "Max": zone_data[y_axis_selected].max()
                    }
                    summary_data.append(summary)
            else:
                st.warning("The selected dataset does not contain a 'Date/Time' column.")
                break

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)

        if st.button("Save Zone Summary"):
            summary_df.to_csv("good_zone_summary.csv", index=False)
            st.success("Zone summary saved to good_zone_summary.csv")

    # Display log data
    st.write("### Log Data")
    st.data_editor(df, height=300, width=1000, use_container_width=True)

    # Display correlation matrix if checked
    if show_corr_matrix:
        st.write("### Correlation Matrix Heatmap")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr,
                        annot_kws={"size": 8})
            ax_corr.set_xticklabels(ax_corr.get_xticklabels(), fontsize=10, rotation=45, ha="right")
            ax_corr.set_yticklabels(ax_corr.get_yticklabels(), fontsize=10, rotation=0)
            st.pyplot(fig_corr)
        else:
            st.warning("No numerical columns available for correlation analysis.")
# ------------------ Coating Log Tab ------------------
elif tab_selection == "💧 Coating":
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

    primary_temperature = st.number_input("Primary Coating Temperature (°C)", value=st.session_state.primary_temperature, step=0.1, key="primary_temperature")
    secondary_temperature = st.number_input("Secondary Coating Temperature (°C)", value=st.session_state.secondary_temperature, step=0.1, key="secondary_temperature")
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
        secondary_viscosity_function = secondary_coating_config.get("viscosity_fit_params", {}).get("function", "T**0.5")

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
    st.write(f"**Second Coating Diameter:** {SC_diameter:.1f} µm - Using Die coat {secondary_die} & {secondary_coating}")

    st.subheader("Coating Info")
    st.write("---")

    # Organize coating info layout
    coating_col1, coating_col2 = st.columns([1, 2])

    with coating_col1:
        selected_coating_info = st.selectbox("Select Coating to View Details", list(coatings.keys()), key="coating_info_select")

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
        draw_history_fields = ["Draw Name", "First Coating", "First Coating Temperature", "First Coating Die Size",
                               "Second Coating", "Second Coating Temperature", "Second Coating Die Size", "Fiber Diameter"]

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
        history_type = st.sidebar.radio("Select History Type", ["All", "Draw History", "Problem History", "Maintenance History"], key="history_type_select")

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

                    st.write(f"### {log_type} Timeline")
                    fig = px.scatter(
                        filtered_df,
                        x="Timestamp",
                        y="Type",
                        color="Type",
                        opacity=0.8,
                        title=f"{log_type} Timeline"
                    )
                    st.plotly_chart(fig, use_container_width=True)
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

                st.write(f"### {history_type} Timeline")
                fig = px.scatter(
                    filtered_df,
                    x="Timestamp",
                    y="Type",
                    color="Type",
                    opacity=0.8,
                    title=f"{history_type} Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No records found for {history_type}")

        # ------------------ Add Event Form ------------------
        with open("config_coating.json", "r") as config_file:
            config = json.load(config_file)
        # Extract coatings and dies dictionaries
        coatings = config.get("coatings", {})
        dies = config.get("dies", {})

        if history_type == "Draw History":
            st.sidebar.subheader("Add Draw Entry")
            draw_name = st.sidebar.text_input("Draw Name")
            first_coating = st.sidebar.selectbox("Select First Coating", list(coatings.keys()))
            first_coating_temp = st.sidebar.number_input("First Coating Temperature (°C)", min_value=0.0, step=0.1)
            first_die = st.sidebar.selectbox("Select First Die", list(dies.keys()))
            second_coating = st.sidebar.selectbox("Select Second Coating", list(coatings.keys()))
            second_coating_temp = st.sidebar.number_input("Second Coating Temperature (°C)", min_value=0.0, step=0.1)
            second_die = st.sidebar.selectbox("Select Second Die", list(dies.keys()))
            fiber_diameter = st.sidebar.number_input("Fiber Diameter (µm)", min_value=0.0, step=0.1)
            first_entry_die = st.sidebar.number_input("First Coating Entry Die (mm)", min_value=0.0, step=0.1)
            second_entry_die = st.sidebar.number_input("Second Coating Entry Die (mm)", min_value=0.0, step=0.1)

            if st.sidebar.button("Save Draw History"):
                new_entry = pd.DataFrame([{
                    "Timestamp": pd.Timestamp.now(),
                    "Type": "Draw History",
                    "Draw Name": draw_name,
                    "First Coating": first_coating,
                    "First Coating Temperature": first_coating_temp,
                    "First Die": first_die,
                    "Second Coating": second_coating,
                    "Second Coating Temperature": second_coating_temp,
                    "Second Die": second_die,
                    "Fiber Diameter": fiber_diameter,
                    "First Coating Entry Die": first_entry_die,
                    "Second Coating Entry Die": second_entry_die,
                }])

                history_df = pd.concat([history_df, new_entry], ignore_index=True)
                history_df.to_csv(HISTORY_FILE, index=False)
                st.sidebar.success("Draw history saved!")


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

        elif history_type == "Maintenance History":
            part_changed = st.sidebar.text_input("Part Changed")
            maintenance_notes = st.sidebar.text_area("Maintenance Details")

            if st.sidebar.button("Save Maintenance History"):
                new_entry = pd.DataFrame([{
                    "Timestamp": pd.Timestamp.now(),
                    "Type": "Maintenance History",
                    "Part Changed": part_changed,
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
                    event_colors = {"Maintenance": "blue", "Drawing": "green", "Stop": "red"}

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
                    event_type = st.sidebar.selectbox("Select Event Type", ["Maintenance", "Drawing", "Stop"])
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
                            "Recurrence": recurrence
                        }])

                        full_schedule_df = pd.read_csv(SCHEDULE_FILE)
                        full_schedule_df = pd.concat([full_schedule_df, new_event], ignore_index=True)
                        full_schedule_df.to_csv(SCHEDULE_FILE, index=False)

                        st.sidebar.success("Event added to schedule!")
# ------------------ Schedule Tab ------------------
elif tab_selection == "📅 Schedule":
    st.title("📅 Tower Schedule")
    st.sidebar.title("Schedule Management")

    #make it work with no events in schedule

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
            event_colors = {"Maintenance": "blue", "Drawing": "green", "Stop": "red"}

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
            event_type = st.sidebar.selectbox("Select Event Type", ["Maintenance", "Drawing", "Stop"])
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
                    "Recurrence": recurrence
                }])

                full_schedule_df = pd.read_csv(SCHEDULE_FILE)
                full_schedule_df = pd.concat([full_schedule_df, new_event], ignore_index=True)
                full_schedule_df.to_csv(SCHEDULE_FILE, index=False)

                st.sidebar.success("Event added to schedule!")
            st.sidebar.subheader("Delete Event")
            if not schedule_df.empty:
                event_to_delete = st.sidebar.selectbox("Select Event to Delete", schedule_df["Description"].tolist())
                if st.sidebar.button("Delete Event"):
                    # Delete the selected event
                    schedule_df = schedule_df[schedule_df["Description"] != event_to_delete]
                    schedule_df.to_csv(SCHEDULE_FILE, index=False)  # Save the updated schedule
                    st.sidebar.success("Event deleted successfully!")
            else:
                st.sidebar.info("No events available for deletion.")
# ------------------ Closed Log Tab ------------------
elif tab_selection == "✅ Closed Processes":
    st.title("✅ Closed Processes")
    st.write("Manage products that are finalized and ready for drawing.")

    CLOSED_PROCESSES_FILE = "closed_processes.csv"

    if not os.path.exists(CLOSED_PROCESSES_FILE):
        pd.DataFrame(columns=[
            "Product Name", "Temperature (°C)", "Velocity (m/s)", "Tension (N)",
            "Entry Diameter (µm)", "First Coating Diameter (µm)", "Second Coating Diameter (µm)", "Additional Info"
        ]).to_csv(CLOSED_PROCESSES_FILE, index=False)

    closed_df = pd.read_csv(CLOSED_PROCESSES_FILE)

    st.write("### Closed Products Table")
    st.data_editor(closed_df, height=400, use_container_width=True)

    st.sidebar.subheader("Add New Closed Process")
    product_name = st.sidebar.text_input("Product Name")
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    velocity = st.sidebar.number_input("Velocity (m/s)", min_value=0.0, step=0.1)
    tension = st.sidebar.number_input("Tension (N)", min_value=0.0, step=0.1)
    entry_diameter = st.sidebar.number_input("Entry Diameter (µm)", min_value=0.0, step=0.1)
    first_coating_diameter = st.sidebar.number_input("First Coating Diameter (µm)", min_value=0.0, step=0.1)
    second_coating_diameter = st.sidebar.number_input("Second Coating Diameter (µm)", min_value=0.0, step=0.1)
    additional_info = st.sidebar.text_area("Additional Info")

    if st.sidebar.button("Add Product"):
        new_entry = pd.DataFrame([{
            "Product Name": product_name,
            "Temperature (°C)": temperature,
            "Velocity (m/s)": velocity,
            "Tension (N)": tension,
            "Entry Diameter (µm)": entry_diameter,
            "First Coating Diameter (µm)": first_coating_diameter,
            "Second Coating Diameter (µm)": second_coating_diameter,
            "Additional Info": additional_info
        }])

        closed_df = pd.concat([closed_df, new_entry], ignore_index=True)
        closed_df.to_csv(CLOSED_PROCESSES_FILE, index=False)
        st.sidebar.success("Product added successfully!")
        #st.rerun()

    st.sidebar.subheader("Delete Closed Process")
    if not closed_df.empty:
        delete_product = st.sidebar.selectbox("Select Product to Delete", closed_df["Product Name"].tolist())
        if st.sidebar.button("Delete Product"):
            closed_df = closed_df[closed_df["Product Name"] != delete_product]
            closed_df.to_csv(CLOSED_PROCESSES_FILE, index=False)
            st.sidebar.success("Product deleted successfully!")
            #st.rerun()
    else:
        st.sidebar.info("No products available for deletion.")
# ------------------ Tower Parts Tab ------------------
elif tab_selection == "🛠️ Tower Parts":
    st.title("🛠️ Tower Parts Management")
    st.sidebar.title("Order Parts Management")

    ORDER_FILE = "part_orders.csv"

    if os.path.exists(ORDER_FILE):
        orders_df = pd.read_csv(ORDER_FILE)
    else:
        orders_df = pd.DataFrame(
            columns=["Part Name", "Serial Number", "Purpose", "Reason", "Date Ordered", "Company", "Status"]
        )

    action = st.sidebar.radio("Manage Orders", ["Add New Order", "Update Existing Order"], key="order_action")

    if action == "Add New Order":
        part_name = st.sidebar.text_input("Part Name")
        serial_number = st.sidebar.text_input("Serial Number")
        purpose = st.sidebar.text_area("Purpose of Order")
        reason = st.sidebar.text_area("Reason for New/Replacement")
        date_ordered = st.sidebar.date_input("Date of Order (if ordered)")
        company = st.sidebar.text_input("Company Ordered From")
        opened_by = st.sidebar.text_input("Opened By")

        if st.sidebar.button("Save Order"):
            new_order = pd.DataFrame([{
                "Part Name": part_name,
                "Serial Number": serial_number,
                "Purpose": purpose,
                "Reason": reason,
                "Date Ordered": date_ordered.strftime("%Y-%m-%d") if date_ordered else "",
                "Company": company,
                "Opened By": opened_by,
                "Status": "Needed"
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
            new_status = st.sidebar.selectbox("Update Order Status",
                                              ["Needed", "Ordered", "Shipped", "Received", "Installed"],
                                              index=["Needed", "Ordered", "Shipped", "Received", "Installed"].index(
                                                  orders_df.at[order_index, "Status"]))

            if st.sidebar.button("Update Order"):
                orders_df.at[order_index, "Status"] = new_status
                orders_df.to_csv(ORDER_FILE, index=False)
                st.sidebar.success("Order updated!")

    st.write("### Order Tracking")
    if not orders_df.empty:
        st.data_editor(orders_df, height=400, use_container_width=True)
    else:
        st.warning("No orders have been placed yet.")

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
# ------------------ Draw Archive Tab ------------------
elif tab_selection == "🔍 Iris Selection":
    st.title("🔍 Iris Selection")
    st.subheader("Iris Selection Tool")
    preform_diameter = st.number_input("Enter Preform Diameter (mm)", min_value=0.0, step=0.1, format="%.2f")

    iris_diameters = [round(x * 0.5, 1) for x in range(20, 91)]

    if preform_diameter > 0 and iris_diameters:
        valid_iris = [d for d in iris_diameters if d > preform_diameter]
        if valid_iris:
            results = [(d, (d - preform_diameter) / 2) for d in valid_iris]
            best = min(results, key=lambda x: abs(x[1] - 0.2))
            st.write(f"Best Matching Iris Diameter: {best[0]:.2f} mm")
            st.write(f"Calculated Free Space: {best[1]:.2f} mm")
        else:
            st.warning("No iris diameter is larger than the preform diameter.")
    else:
        st.info("Please enter a preform diameter and provide valid iris diameters.")
# ------------------ Development Tab ------------------
elif tab_selection == "🧪 Development Process":
    st.title("🧪 Development Process")
    st.sidebar.title("Manage R&D Projects")
    st.subheader("📦 Archived Projects")
    archived_file = "archived_projects.csv"

    # Render quick access buttons to archived project views
    if os.path.exists(archived_file):
        archived_projects_df = pd.read_csv(archived_file)
        archived_projects = archived_projects_df["Project Name"].unique().tolist()
        selected_archived = st.selectbox("Select Archived Project", [""] + archived_projects, key="archived_project_select")
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
    UPDATES_FILE = "experiment_updates.csv"
    if not os.path.exists(UPDATES_FILE):
        pd.DataFrame(columns=["Experiment Title", "Update Date", "Researcher", "Update Notes"]).to_csv(UPDATES_FILE,
                                                                                                       index=False)

    if not os.path.exists(DEVELOPMENT_FILE):
        pd.DataFrame(columns=["Project Name", "Project Purpose", "Target"]).to_csv(DEVELOPMENT_FILE, index=False)

    dev_df = pd.read_csv(DEVELOPMENT_FILE)

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
        selected_project = st.sidebar.selectbox("Choose a Project", active_projects["Project Name"].unique().tolist())

    if selected_project:
        st.subheader(f"Project Details: {selected_project}")

        # Load project details safely
        project_rows = dev_df[dev_df["Project Name"] == selected_project]
        if not project_rows.empty:
            project_data = project_rows.iloc[0]
            st.write(f"**Project Purpose:** {project_data.get('Project Purpose', 'N/A')}")
            st.write(f"**Target:** {project_data.get('Target', 'N/A')}")
        else:
            st.warning(f"No project data found for '{selected_project}'. It may have been removed or archived.")

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
        if show_add_experiment:
            st.subheader("➕ Add Experiment to Project")
            experiment_title = st.text_input("Experiment Title")
            methods = st.text_area("Methods")
            purpose = st.text_area("Experiment Purpose")
            date = st.date_input("Date")
            researcher = st.text_input("Researcher Name")
            observations = st.text_area("Observations")
            results = st.text_area("Results")

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
                        "Results": results
                    }])
                    dev_df = pd.concat([dev_df, new_experiment], ignore_index=True)
                    dev_df.to_csv(DEVELOPMENT_FILE, index=False)
                    st.success("Experiment added successfully!")
                    st.rerun()
                else:
                    st.warning("Please provide at least a title and date for the experiment.")

        # ---- Display Existing Experiments ----
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

                    # Load experiment updates
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
                    update_researcher = st.text_input(f"Your name for update on {exp['Experiment Title']}", key=f"researcher_{exp['Experiment Title']}")
                    update_notes = st.text_area(f"Add new progress update for {exp['Experiment Title']}", key=f"update_{exp['Experiment Title']}")
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
# ------------------ Protocols Tab ------------------
PROTOCOLS_FILE = "protocols.json"
if os.path.exists(PROTOCOLS_FILE):
    with open(PROTOCOLS_FILE, "r") as file:
        st.session_state["protocols"] = json.load(file)

if tab_selection == "📋 Protocols":
    st.title("📋 Protocols")
    st.subheader("Manage Tower Protocols")

    if "protocols" not in st.session_state:
        st.session_state["protocols"] = []

    # Load protocols from file if they exist
    if os.path.exists(PROTOCOLS_FILE):
        with open(PROTOCOLS_FILE, "r") as file:
            st.session_state["protocols"] = json.load(file)

    selected_protocol = ""
    if st.session_state["protocols"]:
        selected_protocol = st.selectbox("Select a Protocol", [""] + [p["name"] for p in st.session_state["protocols"]])
        if selected_protocol:
            protocol = next(p for p in st.session_state["protocols"] if p["name"] == selected_protocol)
            st.markdown(f"**{protocol['name']}**")
            st.write(f"Type: {protocol['type']}")
            if protocol["type"] == "Checklist":
                checklist_items = [item.strip() for item in protocol["instructions"].split("\n") if item.strip()]
                if checklist_items:
                    checkbox_values = [st.checkbox(item) for item in checklist_items]
                    if all(checkbox_values):
                        st.success(f"All items in {protocol['name']} checklist are completed!")
                else:
                    st.info("No checklist items available.")
            else:
                st.markdown("Instructions:\n" + protocol["instructions"].replace("\n", "  \n"))

    if selected_protocol:
        st.markdown("---")
        update_protocol = st.checkbox("Update Protocol", key="update_protocol_checkbox")
        if update_protocol:
            with st.form(key="update_protocol_form"):
                new_protocol_name = st.text_input("New Protocol Name", value=protocol["name"])
                new_protocol_type = st.radio("New Protocol Type", ["Checklist", "Instructions"],
                                             index=["Checklist", "Instructions"].index(protocol["type"]))
                new_protocol_instructions = st.text_area("New Protocol Instructions", value=protocol["instructions"])
                update_button = st.form_submit_button(label="Update Protocol")
                if update_button:
                    protocol["name"] = new_protocol_name
                    protocol["type"] = new_protocol_type
                    protocol["instructions"] = new_protocol_instructions
                    # Save updated protocols list to file
                    with open(PROTOCOLS_FILE, "w") as file:
                        json.dump(st.session_state["protocols"], file, indent=4)
                    st.success(f"Protocol '{new_protocol_name}' updated successfully!")
                    st.rerun()  # Immediately refresh the list

        delete_protocol = st.checkbox("Delete Protocol", key="delete_protocol_checkbox")
        if delete_protocol:
            if st.button(f"Delete {protocol['name']}"):
                st.session_state["protocols"].remove(protocol)
                # Save updated protocols list to file
                with open(PROTOCOLS_FILE, "w") as file:
                    json.dump(st.session_state["protocols"], file, indent=4)
                st.success(f"Protocol '{protocol['name']}' deleted successfully!")
                st.rerun()  # Immediately refresh the list

    else:
        st.info("No protocols available.")

    if not selected_protocol:
        create_new = st.checkbox("Create New Protocol", key="create_new_protocol_checkbox")
        if create_new:
            with st.form(key="new_protocol_form"):
                protocol_name = st.text_input("Enter Protocol Name")
                protocol_type = st.radio("Select Protocol Type", ["Checklist", "Instructions"])
                protocol_instructions = st.text_area("Enter Protocol Instructions")
                submit_button = st.form_submit_button(label="Add Protocol")
                if submit_button:
                    if protocol_name and protocol_instructions:
                        new_protocol = {"name": protocol_name, "type": protocol_type, "instructions": protocol_instructions}
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