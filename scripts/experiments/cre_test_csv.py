import os
import pandas as pd
from datetime import datetime, timedelta
import random

out_dir = "data_set_csv"
os.makedirs(out_dir, exist_ok=True)

base = datetime(2025, 1, 1, 8, 0, 0)

for i in range(10):
    draw_name = f"sql_lab_demo_{i+1}"
    draw_date = (base + timedelta(days=i*7)).isoformat(sep=" ")

    rows = [
        ("Draw Name", draw_name, ""),
        ("Draw Date", draw_date, ""),
        ("Entry Fiber Diameter", str(120 + random.uniform(-60, 60)), "um"),
        ("Second Coating Diameter (Theoretical)", str(120 + random.uniform(-60, 60)), "um"),
        ("P Gain", str(round(random.uniform(4, 8), 3)), ""),
        ("I Gain", str(round(random.uniform(9, 22.2), 3)), ""),
        ("TF Mode", random.choice(["Winder", "Straight Mode"]), ""),
    ]

    df = pd.DataFrame(rows, columns=["Parameter Name", "Value", "Units"])
    df.to_csv(os.path.join(out_dir, f"{draw_name}.csv"), index=False)

print("✅ Created 5 demo dataset CSVs in data_set_csv/")