import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from recent_csv import get_most_recent_csv  # Import function from recent_csv.py

# Load directory configuration from JSON
CONFIG_FILE = "directory_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

def save_plot_report_and_dial_pages(csv_file, output_directory):
    """
    Generates a fiber quality report and saves it in a specified directory.

    :param csv_file: Path to the CSV file containing numeric data.
    :param output_directory: Directory where the PDF report will be saved.
    """

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Define output PDF path
    pdf_filename = os.path.join(output_directory, config["fiber_report_pdf"])

    # Define Good Fiber Zones
    good_fiber_zones = [
        ("04/12/2024 12:11:47750", "04/12/2024 12:40:47750"),
    ]

    fiber_z_col = "Fibre Length"
    preform_z_col = "Pf Process Position"

    other_sensors = [
        "Bare Fibre Diameter",
        "Diameter Error",
        "Capstan Speed",
        "Furnace DegC Actual",
        "Preform Speed Actual",
        "Cane Speed Actual",
        "Tension N",
        "Trend Marker",
        "Coated Outer Diameter",
        "Poly X Diameter",
        "Poly Y Diameter",
        "Poly Major Value",
        "Poly Minor Value",
        "Diameter Deviation Gauge 2",
        "UV 1 Intensity",
        "UV 2 Intensity"
    ]

    # Load data
    df = pd.read_csv(csv_file, parse_dates=["Date/Time"])

    # Replace infinities and drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[fiber_z_col, preform_z_col], inplace=True)

    # Compute min/max Fiber Z for good zones
    min_fiber_z, max_fiber_z = float('inf'), float('-inf')

    for start_time, end_time in good_fiber_zones:
        df_filtered = df[(df["Date/Time"] >= start_time) & (df["Date/Time"] <= end_time)]
        if df_filtered.empty:
            print(f"Warning: No data found for time range {start_time} to {end_time}")
            continue
        min_fiber_z = min(min_fiber_z, df_filtered[fiber_z_col].min())
        max_fiber_z = max(max_fiber_z, df_filtered[fiber_z_col].max())

    # Ensure valid limits
    if min_fiber_z == float('inf') or max_fiber_z == float('-inf'):
        raise ValueError("No valid Fiber Z values found in the dataset for the selected time zones.")

    print(f"min_fiber_z: {min_fiber_z}, max_fiber_z: {max_fiber_z}")

    # Set font properties
    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = 12

    # Create PDF report
    with PdfPages(pdf_filename) as pdf:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twiny()
        zone_colors = ["lightcoral", "lightgreen", "lightblue"]

        for idx, (start_time, end_time) in enumerate(good_fiber_zones, start=1):
            df_filtered = df[(df["Date/Time"] >= start_time) & (df["Date/Time"] <= end_time)]
            if df_filtered.empty:
                continue
            ax1.axvspan(df_filtered[fiber_z_col].min(), df_filtered[fiber_z_col].max(),
                        color=zone_colors[idx % len(zone_colors)], alpha=0.5, label=f"Good Zone {idx}")

        ax1.set_xlim(min_fiber_z, max_fiber_z)
        ax1.set_xlabel("Fiber Z (Measured Data)", fontsize=14, fontweight="bold")
        ax1.set_title("Good Fiber Zones with Dual X-Axis (Fiber Z & Preform Z)", fontsize=16, fontweight="bold")

        # Set Preform Z on second x-axis **CORRECTLY**
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(df[fiber_z_col].iloc[::len(df) // 10])  # Use evenly spaced ticks
        ax2.set_xticklabels([f"{val:.2f}" for val in df[preform_z_col].iloc[::len(df) // 10]])  # Directly use Preform Z values
        ax2.set_xlabel("Preform Z", fontsize=14, fontweight="bold")

        ax1.legend(fontsize=12, loc="upper left")
        ax1.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Individual zone reports
        for idx, (start_time, end_time) in enumerate(good_fiber_zones, start=1):
            df_filtered = df[(df["Date/Time"] >= start_time) & (df["Date/Time"] <= end_time)]
            if df_filtered.empty:
                continue

            min_fiber_z_zone = df_filtered[fiber_z_col].min()
            max_fiber_z_zone = df_filtered[fiber_z_col].max()

            sensor_averages = {sensor: df_filtered[sensor].mean() for sensor in other_sensors}
            sensor_std_dev = {sensor: df_filtered[sensor].std() for sensor in other_sensors}

            d1 = sensor_averages.get("Bare Fibre Diameter", 50)
            d2 = sensor_averages.get("Coated Inner Diameter", 100)
            d3 = sensor_averages.get("Coated Outer Diameter", 150)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            report_text = f"\n=== Good Fiber Zone {idx} (Fiber Z: {min_fiber_z_zone:.2f} - {max_fiber_z_zone:.2f}) ===\n" + "=" * 40 + "\n"
            for sensor in other_sensors:
                report_text += f"{sensor}: {sensor_averages[sensor]:.2f} ± {sensor_std_dev[sensor]:.2f}\n"
            report_text += "\n===T&M Section===\nNew fiber name=__________\nCore Diameter(µm) = ______\nClad Diameter(µm) = ______\nFirst coating Diameter(µm) = ______\nSecond coating Diameter(µm) = ______\nBirefringence=_____"

            axs[0].text(0.01, 0.99, report_text, fontsize=14, ha='left', va='top', family="Times New Roman", fontweight="bold", wrap=True)
            axs[0].axis("off")

            colors = ['red', 'blue', 'green']
            labels = [f"Clad: {d1:.2f} µm", f"First Coating: {d2:.2f} µm", f"Second Coating: {d3:.2f} µm"]
            for r, color, label in zip([d1 / 2, d2 / 2, d3 / 2], colors, labels):
                axs[1].add_patch(plt.Circle((0, 0), r, fill=False, linewidth=2, edgecolor=color, label=label))
            axs[1].set_xlim(-d3 / 2 - 10, d3 / 2 + 10)
            axs[1].set_ylim(-d3 / 2 - 10, d3 / 2 + 10)
            axs[1].set_aspect('equal')
            axs[1].legend(loc="upper right", fontsize=10)
            axs[1].grid(True)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved as: {pdf_filename}")

# **Find the most recent CSV and generate a folder for the PDF output**
folder_path = config["logs_directory"]
recent_csv = get_most_recent_csv()
folder_out_name = os.path.splitext(recent_csv)[0]
output_directory = os.path.join(config["output_folders"], folder_out_name)

csv_file = os.path.join(folder_path, recent_csv)
save_plot_report_and_dial_pages(csv_file, output_directory)
