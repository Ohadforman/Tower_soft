import os
import csv
import numpy as np
import pandas as pd
from fpdf import FPDF
from pathlib import Path
from recent_csv import get_most_recent_csv  # Import the function
import json

# Load directory configuration from JSON
CONFIG_FILE = "directory_config.json"
if Path(CONFIG_FILE).exists():
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

# Load coating data from JSON for flexibility
COATING_DATA_FILE = "coating_data.json"
if Path(COATING_DATA_FILE).exists():
    with open(COATING_DATA_FILE, "r") as file:
        raw_coating_data = json.load(file)

    # Convert viscosity expressions from strings to functions
    coating_data = {}
    for key, values in raw_coating_data.items():
        coating_data[key] = {
            "density": values["density"],
            "viscosity": eval(values["viscosity_function"])
        }
else:
    coating_data = {}


def estimate_coating_diameter(die_diameter, viscosity, density, exponent=0.15):
    """Estimate coating thickness based on viscosity, density, and die diameter."""
    return die_diameter * ((viscosity / density) ** exponent)


def get_user_inputs():
    """ Get user inputs for coating parameters with error handling. """
    print("\n=== Enter Coating Parameters ===")
    try:
        main_die_diameter = float(input("Enter Main Coating Die Diameter (µm): "))
        main_entry_die_diameter = float(input("Enter Main Coating Entry Die Diameter (µm): "))
        print("\nAvailable Coatings:", list(coating_data.keys()))
        main_coating = input("Enter Main Coating Name: ")
        if main_coating not in coating_data:
            raise ValueError("Invalid coating choice!")
        main_density = coating_data[main_coating]["density"]
        main_coating_temp = float(input("Enter Main Coating Temperature (°C): "))
        main_viscosity = coating_data[main_coating]["viscosity"](main_coating_temp)
        main_coating_thickness = estimate_coating_diameter(main_die_diameter, main_viscosity, main_density)
        sec_die_diameter = float(input("\nEnter Secondary Coating Die Diameter (µm): "))
        sec_entry_die_diameter = float(input("Enter Secondary Coating Entry Die Diameter (µm): "))
        print("\nAvailable Coatings:", list(coating_data.keys()))
        sec_coating = input("Enter Secondary Coating Name: ")
        if sec_coating not in coating_data:
            raise ValueError("Invalid coating choice!")
        sec_density = coating_data[sec_coating]["density"]
        sec_coating_temp = float(input("Enter Secondary Coating Temperature (°C): "))
        sec_viscosity = coating_data[sec_coating]["viscosity"](sec_coating_temp)
        sec_coating_thickness = estimate_coating_diameter(sec_die_diameter, sec_viscosity, sec_density)
    except ValueError as e:
        print(f"⚠️ Input Error: {e}")
        return None
    return {
        "Main Coating Die Diameter (µm)": main_die_diameter,
        "Main Entry Die Diameter (µm)": main_entry_die_diameter,
        "Main Density (g/cm³)": main_density,
        "Main Coating": main_coating,
        "Main Coating Temp (°C)": main_coating_temp,
        "Main Viscosity (mPa·s)": round(main_viscosity, 3),
        "Main Estimated Thickness (µm)": round(main_coating_thickness, 2),
        "Secondary Coating Die Diameter (µm)": sec_die_diameter,
        "Secondary Entry Die Diameter (µm)": sec_entry_die_diameter,
        "Secondary Density (g/cm³)": sec_density,
        "Secondary Coating": sec_coating,
        "Secondary Coating Temp (°C)": sec_coating_temp,
        "Secondary Viscosity (mPa·s)": round(sec_viscosity, 3),
        "Secondary Estimated Thickness (µm)": round(sec_coating_thickness, 2)
    }


def save_to_csv(data):
    """ Save coating data to a CSV file based on config directory. """
    csv_filename = Path(config["coating_data_csv"])
    file_exists = csv_filename.exists()
    with csv_filename.open(mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    print(f"✅ Coating data saved to: {csv_filename}")


def generate_coating_report(data, output_directory):
    """ Generate a PDF report with coating data."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_filename = output_directory / config["coating_report_pdf"]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Coating Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Main Coating", ln=True)
    pdf.set_font("Arial", "", 12)
    for key in data:
        if "Main" in key:
            pdf.cell(200, 10, f"{key}: {data[key]}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Secondary Coating", ln=True)
    pdf.set_font("Arial", "", 12)
    for key in data:
        if "Secondary" in key:
            pdf.cell(200, 10, f"{key}: {data[key]}", ln=True)
    pdf.output(str(pdf_filename))
    print(f"✅ Coating Report Generated: {pdf_filename}")


if __name__ == "__main__":
    coating_data = get_user_inputs()
    if coating_data:
        save_to_csv(coating_data)  # Save CSV using config path
        recent_csv = get_most_recent_csv()
        folder_out_name = os.path.splitext(recent_csv)[0] if recent_csv else "default_output"
        output_directory = Path(config["output_folders"]) / folder_out_name
        generate_coating_report(coating_data, output_directory)
