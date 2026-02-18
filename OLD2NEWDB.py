#!/usr/bin/env python3
"""
Legacy Tower DB (Excel) -> New-format Draw CSVs
Mapping-first version (easy to see/modify), including zone blocks
that match the "new CSV" naming style:

Zone 1 Start
Zone 1 End
Marked Zone 1 Avg - ...
Marked Zone 1 Min - ...
Marked Zone 1 Max - ...
(repeat for Zone 2, Zone 3)

Output CSV columns:
Parameter Name,Value,Units
"""

from __future__ import annotations

import re
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd


# =========================
# CONFIG — EDIT HERE
# =========================

EXCEL_PATH = "Old tower DB.xlsx"
SHEET_NAME = "Drawing_data_base"
DATA_START_ROW = 35

OUT_DIR = Path("legacy_converted_out")

DRAW_HOUR = 12
DONE_HOUR = 13

DONE_OK_TEXT = "OK"
OLD_DONE_DESC_COL = "Done description"

COATING_MAP = {
    "DCOF": ("OF-136", "DS-2015"),
    # "SCOF": ("???", "???"),  # fill later if desired
}

PM_DETECT_COLS = ["Drawing Purpus", "Preform Shape", "Iris Selected"]
OCTA_KEYWORDS = ["octa", "oct", "octagonal"]

# Old zone columns
ZONE_FIBER_START_COL = {1: "Start of good zone 1", 2: "Start of good zone 2", 3: "Start of good zone 3"}
ZONE_FIBER_END_COL   = {1: "End good Zone 1",      2: "End good Zone 2",      3: "End good Zone 3"}

# paired preform (cm) columns (unnamed in header row)
ZONE_PREFORM_PAIR_COLS = {
    1: ("Unnamed_60", "Unnamed_62"),
    2: ("Unnamed_64", "Unnamed_66"),
    3: ("Unnamed_68", "Unnamed_70"),
}


# =========================
# Helpers
# =========================

_num_re = re.compile(r"[-+]?\d*\.?\d+")


def _make_unique(headers: List[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, h in enumerate(headers):
        if h is None or (isinstance(h, float) and math.isnan(h)) or str(h).strip() == "":
            name = f"Unnamed_{i}"
        else:
            name = str(h).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        out.append(name)
    return out


def to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "nan":
        return None
    m = _num_re.search(s)
    return float(m.group()) if m else None


def legacy_dt(date_val: Any, hour: int) -> str:
    d = pd.to_datetime(date_val, errors="coerce")
    return "" if pd.isna(d) else d.strftime(f"%Y-%m-%d {hour:02d}:00:00")


def has_any_text(x: Any) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip()
    return s not in ("", "-", "nan", "NaN")


def contains_pm(row: pd.Series) -> bool:
    for c in PM_DETECT_COLS:
        if "pm" in str(row.get(c, "")).lower():
            return True
    return False


def is_octa(row: pd.Series) -> bool:
    purp = str(row.get("Drawing Purpus", "")).lower()
    shape = str(row.get("Preform Shape", "")).lower()
    return any(k in purp for k in OCTA_KEYWORDS) or any(k in shape for k in OCTA_KEYWORDS)


def fiber_geometry_type(row: pd.Series) -> str:
    if contains_pm(row):
        return "PM"
    if is_octa(row):
        return "Octagonal"
    return "Round"


def map_coatings(row: pd.Series) -> Tuple[str, str]:
    ct = str(row.get("Coating type", "")).strip().upper()
    return COATING_MAP.get(ct, ("", ""))


def add(out: List[Dict[str, Any]], name: str, value: Any, units: str = "") -> None:
    # Keep explicit "" values (so you can force empty rows to exist)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return
    out.append({"Parameter Name": name, "Value": value, "Units": units})


def add_empty(out: List[Dict[str, Any]], name: str, units: str = "") -> None:
    """Force a row to exist even if empty (optional)."""
    out.append({"Parameter Name": name, "Value": "", "Units": units})


# =========================
# Global mapping (non-zones)
# =========================

DIRECT_MAP: Dict[str, Tuple[str, str]] = {
    "Preform Number": ("Preform ID", ""),
    "Order Notes": ("Drawing Purpus", ""),

    "Tension (g)": ("T (g)", "g"),
    "Draw Speed (m/min)": ("Vn (m/min)", "m/min"),

    "Tension Actual (g)": ("T (g) - Actual", "g"),
    "Draw Speed Actual (m/min)": ("Vn (m/min) - Actual", "m/min"),

    "Fiber Diameter (µm)": ("Fiber dimension 125/250/400", "µm"),
    "Entry Fiber Diameter (µm)": ("Bare fiber diameter", "µm"),
    "Main Coating Diameter (µm)": ("Primary coating diameter", "µm"),
    "Secondary Coating Diameter (µm)": ("Secondary coating diameter", "µm"),

    "Furnace DegC Set": ("Final Process temp", "°C"),

    "Preform Shape": ("Preform Shape", ""),
    "Preform Diameter": ("Preform Diameter F2F", "mm"),

    "Total Fiber Length (m)": ("Total Fiber", "m"),
    "Total Good Fiber Length (m)": ("total good fiber", "m"),

    "Cut 1": ("Cut 1", "m"),
    "Good fiber 1": ("Good fiber 1", "m"),
    "Cut 2": ("Cut 2", "m"),
    "Good fiber 2": ("Good fiber 2", "m"),
    "Cut 3": ("Cut 3", "m"),
    "Good fiber 3": ("Good fiber 3", "m"),

    "Required Length (m) (for T&M+costumer)": ("Total good zone fiber requested", "m"),
}


def add_derived(out: List[Dict[str, Any]], row: pd.Series) -> None:
    dt12 = legacy_dt(row.get("Date"), DRAW_HOUR)
    dt13 = legacy_dt(row.get("Date"), DONE_HOUR)

    preform = str(row.get("Preform ID", "")).strip()
    draw_count = int(row.get("_count", 1))
    draw_name = f"{preform}F_{draw_count}"

    # Done Description -> OK if any text, else blank
    done_desc = DONE_OK_TEXT if has_any_text(row.get(OLD_DONE_DESC_COL)) else ""

    main_c, sec_c = map_coatings(row)
    pm_iris = 1 if contains_pm(row) else 0
    octa = 1 if is_octa(row) else 0
    octa_f2f = to_float(row.get("Preform Diameter F2F")) if octa else 0.0

    add(out, "Draw Name", draw_name, "")
    add(out, "Draw Date", dt12, "")
    add(out, "Process Setup Timestamp", dt12, "")

    add(out, "Priority", "Normal", "")
    add(out, "Order Index", "", "")
    add(out, "Fiber Project", "", "")
    add(out, "Order Opener", "", "")

    add(out, "Fiber Geometry Type", fiber_geometry_type(row), "")

    # Legacy tiger = 0
    add(out, "Tiger Cut (%)", 0.0, "%")
    add(out, "Tiger Preform", 0, "bool")
    add(out, "Tiger Cut", 0.0, "%")

    add(out, "Octagonal Preform", octa, "bool")
    add(out, "Octagonal F2F (mm)", octa_f2f, "mm")

    add(out, "PM Iris System", pm_iris, "bool")
    add(out, "Iris Mode", "Manual", "")
    add(out, "Selected Iris Diameter", to_float(row.get("Iris Selected")), "mm")

    add(out, "Main Coating", main_c, "")
    add(out, "Secondary Coating", sec_c, "")

    # Not present in legacy (keep as empty rows if you want strict schema)
    add(out, "Main Coating Temperature (°C)", "", "°C")
    add(out, "Secondary Coating Temperature (°C)", "", "°C")

    # Done rules you requested
    add(out, "Done Description", done_desc, "")
    add(out, "Done Timestamp", dt13, "")


# =========================
# Zone mapping — THIS is the key part you asked for
# =========================

def choose_speed(row: pd.Series) -> Optional[float]:
    """Actual preferred else setpoint."""
    v_act = to_float(row.get("Vn (m/min) - Actual"))
    return v_act if v_act is not None else to_float(row.get("Vn (m/min)"))

def choose_tension_g(row: pd.Series) -> Optional[float]:
    """Actual preferred else setpoint; legacy stored grams even if field says 'Tension N'."""
    t_act = to_float(row.get("T (g) - Actual"))
    return t_act if t_act is not None else to_float(row.get("T (g)"))

def choose_furnace(row: pd.Series) -> Optional[float]:
    """Legacy only has one temperature column; treat it as both set & actual."""
    return to_float(row.get("Final Process temp"))

def choose_bare_um(row: pd.Series) -> Optional[float]:
    return to_float(row.get("Bare fiber diameter"))

def choose_pf_stats_from_zone(z: Dict[str, Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    sp = z.get("sp_cm")
    ep = z.get("ep_cm")
    if sp is None or ep is None:
        return None, None, None
    mn = min(sp, ep)
    mx = max(sp, ep)
    avg = (mn + mx) / 2.0
    return avg, mn, mx


# This list defines EXACTLY which "Marked Zone ..." rows you generate.
# Each item: (suffix, value_getter, units)
# We will generate Avg/Min/Max rows for each.
ZONE_METRICS: List[Tuple[str, Callable[[pd.Series], Optional[float]], str]] = [
    ("Bare Fibre Diameter", choose_bare_um, ""),   # new CSV often leaves units blank for these stats
    ("Furnace DegC Actual", choose_furnace, ""),
    ("Capstan Speed",       choose_speed,  ""),
    # IMPORTANT: name says "Tension N" but legacy value is grams
    ("Tension N",           choose_tension_g, "g"),
]

def parse_zones(row: pd.Series) -> List[Dict[str, Optional[float]]]:
    zones = []
    for i in (1, 2, 3):
        sf = to_float(row.get(ZONE_FIBER_START_COL[i]))
        ef = to_float(row.get(ZONE_FIBER_END_COL[i]))
        if sf is None or ef is None:
            continue
        pstart_col, pend_col = ZONE_PREFORM_PAIR_COLS[i]
        zones.append({
            "i": i,
            "sf_km": sf,
            "ef_km": ef,
            "sp_cm": to_float(row.get(pstart_col)),
            "ep_cm": to_float(row.get(pend_col)),
        })
    return zones

def add_zones(out: List[Dict[str, Any]], row: pd.Series) -> None:
    zones = parse_zones(row)

    add(out, "Good Zones Count", len(zones), "count")
    add(out, "Good Zones X Column", "Fibre Length", "")

    total_f_m = to_float(row.get("Total Fiber"))
    L_end_km = (total_f_m / 1000.0) if total_f_m is not None else None

    for idx, z in enumerate(zones, start=1):
        sf = z["sf_km"]
        ef = z["ef_km"]
        fl_min = min(sf, ef)
        fl_max = max(sf, ef)

        # Match new style: Zone i Start/End exist
        add(out, f"Zone {idx} Start", f"legacy_fibre_km={sf}", "index/label")
        add(out, f"Zone {idx} End",   f"legacy_fibre_km={ef}", "index/label")

        # "Good Zone ..." plan fields (optional but useful)
        if L_end_km is not None:
            kfe_start = max(0.0, L_end_km - fl_max)
            kfe_end = max(0.0, L_end_km - fl_min)
            add(out, f"Good Zone {idx} (km from end) Start", kfe_start, "km")
            add(out, f"Good Zone {idx} (km from end) End",   kfe_end,   "km")
            add(out, f"Good Zone {idx} Length", max(0.0, kfe_end - kfe_start), "km")

        add(out, f"Good Zone {idx} Fibre Length Min", fl_min, "km")
        add(out, f"Good Zone {idx} Fibre Length Max", fl_max, "km")

        # Now the exact "Marked Zone i Avg/Min/Max - ..." style
        for suffix, getter, unit in ZONE_METRICS:
            v = getter(row)
            if v is None:
                continue
            add(out, f"Marked Zone {idx} Avg - {suffix}", v, unit)
            add(out, f"Marked Zone {idx} Min - {suffix}", v, unit)
            add(out, f"Marked Zone {idx} Max - {suffix}", v, unit)

        # Pf Process Position uses per-zone paired preform cm values
        pf_avg, pf_min, pf_max = choose_pf_stats_from_zone(z)
        if pf_avg is not None:
            add(out, f"Marked Zone {idx} Avg - Pf Process Position", pf_avg, "cm")
            add(out, f"Marked Zone {idx} Min - Pf Process Position", pf_min, "cm")
            add(out, f"Marked Zone {idx} Max - Pf Process Position", pf_max, "cm")


# =========================
# IO
# =========================

def load_legacy_excel(excel_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
    headers = _make_unique(df_raw.iloc[0].tolist())

    df = df_raw.iloc[DATA_START_ROW:].copy()
    df.columns = headers
    df = df[df["Preform ID"].notna()].copy().reset_index(drop=True)

    df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Preform ID", "_date"]).reset_index(drop=True)
    df["_count"] = df.groupby("Preform ID").cumcount() + 1
    df = df.sort_values(["_date", "Preform ID", "_count"]).reset_index(drop=True)
    return df


def convert_row(row: pd.Series) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []

    # Derived block first
    add_derived(out, row)

    # Direct mappings
    for new_name, (old_col, unit) in DIRECT_MAP.items():
        add(out, new_name, row.get(old_col), unit)

    # Zones block (new-style naming)
    add_zones(out, row)

    return pd.DataFrame(out, columns=["Parameter Name", "Value", "Units"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_legacy_excel(EXCEL_PATH)

    written: List[Path] = []
    for _, row in df.iterrows():
        preform = str(row.get("Preform ID", "")).strip()
        date_str = pd.to_datetime(row["_date"]).strftime("%Y%m%d") if pd.notna(row["_date"]) else "unknown"
        count = int(row["_count"])
        out_name = f"{preform}_{date_str}_{count}.csv"
        out_path = OUT_DIR / out_name

        out_df = convert_row(row)
        out_df.to_csv(out_path, index=False)
        written.append(out_path)

    zip_path = OUT_DIR.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in written:
            z.write(p, arcname=p.name)

    print(f"✅ Converted {len(written)} draws into: {OUT_DIR}")
    print(f"📦 Zip: {zip_path}")


if __name__ == "__main__":
    main()