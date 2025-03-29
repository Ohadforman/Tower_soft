import pandas as pd
import json
import os
from recent_csv import get_most_recent_csv  # Import the function

# Load directory configuration from JSON
CONFIG_FILE = "directory_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

def replace_column_names(csv_input_path, csv_output_path, mapping):
    """
    Reads a CSV file, renames columns based on a mapping dictionary, and saves the updated CSV.
    """
    # Load the CSV file, skipping the first row if it contains an extra header
    df = pd.read_csv(csv_input_path, delimiter=",", skiprows=1)

    # Ensure 'Date/Time' is retained if it's in the file but not in the mapping
    if 'Date/Time' in df.columns and 'Date/Time' not in mapping:
        mapping = {**{'Date/Time': 'Date/Time'}, **mapping}  # Preserve 'Date/Time'

    # Clean column names to remove potential typos or extra characters
    df.columns = df.columns.str.strip()

    # Rename the columns using the mapping
    df.rename(columns=mapping, inplace=True)

    # Drop empty columns that might have been created due to extra commas
    df.dropna(axis=1, how='all', inplace=True)

    # Save the updated CSV file
    df.to_csv(csv_output_path, index=False)

    print(f"✅ Updated CSV saved as: {csv_output_path}")


# **Find the most recent CSV and generate a folder for the merged PDF output**
folder_path = config["logs_directory"]  # Use path from JSON
recent_csv = get_most_recent_csv()  # Get most recent CSV filename

# Paths for input and output
csv_input_path = os.path.join(folder_path, recent_csv)
csv_output_path = os.path.join(folder_path, f'modified_{recent_csv}')  # Create a new file with modified name

# **Define the mapping dictionary**
mapping = {
    "[plc]BareFibreDiaDisplay": "Bare Fibre Diameter",
    "[plc]diadevbaremv": "Diameter Error",
    "[plc]PfProcessPsn": "Pf Process Position",
    "[plc]cpspdactval": "Capstan Speed",
    "[plc]CpLenTareVal": "Fibre Length",
    "[plc]GoodFibreStartState": "Good Fibre State",
    "[plc]FrnTmpMv": "Furnace DegC Actual",
    "[plc]FrnPwrMv": "Furnace Power",
    "[plc]pfspdactval": "Preform Speed Actual",
    "[plc]FrnMFC1MV": "Furnace MFC1 Actual",
    "[plc]FrnMFC2MV": "Furnace MFC2 Actual",
    "[plc]FrnMFC3MV": "Furnace MFC3 Actual",
    "[plc]FrnMFC4MV": "Furnace MFC4 Actual",
    "[plc]CaneSpdActVal": "Cane Speed Actual",
    "[plc]TenTareVal": "Tension N",
    "[plc]TrendMarkerPulse": "Trend Marker",
    "[plc]CoatedOuterFibreDiaDisplay": "Coated Outer Diameter",
    "[plc]FrnMFC1SP": "Furnace MFC1 Set",
    "[plc]FrnMFC3SP": "Furnace MFC3 Set",
    "[plc]FrnMFC4SP": "Furnace MFC4 Set",
    "[plc]FrnMFC2SP": "Furnace MFC2 Set",
    "[plc]FrnTmpSP": "Furnace DegC Set",
    "[plc]UVHiLo[0].status": "UV1 Status",
    "[plc]UVHiLO[1].Status": "UV2 Status",
    "plc]PolyXDia": "Poly X Diameter",
    "plc]PolyYDia": "Poly Y Diameter",
    "[plc]PloyMajorAxis": "Poly Major Value",
    "[plc]PolyMinorAxis": "Poly Minor Value",
    "[plc]DiaDevXYBareMV": "Diameter Deviation Gauge 2",
    "[plc]CoatedInnerFibreDiaDisplay": "Coated Inner Diameter",
    "[plc]UVLamp1Intensity": "UV 1 Intensity",
    "[plc]UVLamp2Intensity": "UV 2 Intensity"
}

# Run the function
replace_column_names(csv_input_path, csv_output_path, mapping)
