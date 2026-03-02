# helpers/style_utils.py
from __future__ import annotations


def color_status(val):
    s = str(val).strip()
    colors = {"Pending": "orange", "In Progress": "dodgerblue", "Scheduled": "teal", "Failed": "red", "Done": "green"}
    return f"color: {colors.get(s, 'black')}; font-weight: bold"


def color_priority(val):
    p = str(val).strip()
    colors = {"Low": "gray", "Normal": "black", "High": "crimson"}
    return f"color: {colors.get(p, 'black')}; font-weight: bold"

