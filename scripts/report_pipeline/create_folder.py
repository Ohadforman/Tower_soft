import os
import json
from pathlib import Path
from recent_csv import get_most_recent_csv  # Import the function from recent_csv.py

# Load directory configuration from JSON
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "directory_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")


def create_folder_from_csv():
    """
    Creates a folder named after the most recent CSV file (without extension),
    using paths specified in the JSON configuration.
    """
    base_directory = config["output_folders"]

    recent_csv = get_most_recent_csv()

    if recent_csv:
        folder_name = os.path.splitext(recent_csv)[0]  # Remove .csv extension
        new_folder_path = os.path.join(base_directory, folder_name)

        try:
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"✅ Folder created: {new_folder_path}")
        except Exception as e:
            print(f"❌ Error creating folder: {e}")
    else:
        print("⚠️ No CSV files found. No folder created.")


# Execute function
create_folder_from_csv()
