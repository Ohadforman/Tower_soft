import os
import glob
import json

# Load directory configuration from JSON
CONFIG_FILE = "directory_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

def get_most_recent_csv():
    """
    Finds the most recent CSV file in the folder specified in the JSON configuration.

    :return: The name of the most recent CSV file or None if no CSV is found.
    """
    folder_path = config["logs_directory"]  # Get directory from JSON

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # If no CSV files found, return None
    if not csv_files:
        return None

    # Find the most recent CSV file based on modification time
    most_recent_csv = max(csv_files, key=os.path.getmtime)

    return os.path.basename(most_recent_csv)  # Return only the file name


# Example usage
recent_csv = get_most_recent_csv()

if recent_csv:
    print(f"Most recent CSV file: {recent_csv}")
else:
    print("No CSV files found in the folder.")
