import json
import os

CSV_SELECTION_FILE = "selected_csv.json"


def save_selected_csv(selected_csv: str, path: str = CSV_SELECTION_FILE):
    """Save the selected CSV filename/path in a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"selected_csv": selected_csv}, f)


def load_selected_csv(path: str = CSV_SELECTION_FILE):
    """Load the selected CSV filename/path from the JSON file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("selected_csv")
        except Exception:
            return None
    return None