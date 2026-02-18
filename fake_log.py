import os
import pandas as pd
import numpy as np
import datetime as dt

# 👇 adjust if needed
LOGS_FOLDER = "logs"   # or put full path manually

os.makedirs(LOGS_FOLDER, exist_ok=True)

# Generate 2 hours of fake data, every 5 seconds
start = dt.datetime.now() - dt.timedelta(hours=2)
rows = 2 * 60 * 60 // 5  # 2 hours, 5 sec step

times = [start + dt.timedelta(seconds=5*i) for i in range(int(rows))]

df = pd.DataFrame({
    "Date/Time": times,
    "Furnace MFC1 Actual": np.random.normal(5.0, 0.1, len(times)),  # ~5 SCCM
    "Furnace MFC2 Actual": np.random.normal(1.0, 0.05, len(times)), # ~1 SCCM
    "Furnace MFC3 Actual": np.random.normal(4.8, 0.1, len(times)),
    "Furnace MFC4 Actual": np.random.normal(1.0, 0.05, len(times)),
})

file_path = os.path.join(LOGS_FOLDER, f"FAKE_LOG_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
df.to_csv(file_path, index=False)

print("Created:", file_path)