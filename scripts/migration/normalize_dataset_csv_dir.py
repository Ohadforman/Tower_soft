#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data_set_csv"
PROJECTS_TEMPLATES = ROOT / "data" / "projects_fiber_templates.csv"

PROJECT_ALIAS_MAP = {
    "ELOP - LARA": "LARA - ELOP",
    "MKS": "ELOP - 130/10",
}

GEOMETRY_ALIAS_MAP = {
    "ROUND": "ROUND",
    "ROUND ": "ROUND",
    "CIRCULAR": "ROUND",
    "OCTAGONAL": "Octagonal",
    "OCTAGON": "Octagonal",
    "PANDA - PM": "PANDA - PM",
    "STEP INDEX": "STEP INDEX",
    "TIGER - PM": "TIGER - PM",
    "PM": "PM",
}

COATING_ALIAS_MAP = {
    "OF - 136": "Coating_OF_136",
    "OF-136": "Coating_OF_136",
    "COATING_OF_136": "Coating_OF_136",
    "DP-1032": "DP-1032",
    "DS-2015": "DS-2015",
    "DS-2042": "DS-2042",
}


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = safe_str(value)
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(m.group()) if m else None


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value)


def normalize_project(raw: Any) -> str:
    text = safe_str(raw)
    return PROJECT_ALIAS_MAP.get(text, text)


def normalize_geometry(raw: Any) -> str:
    text = safe_str(raw)
    if not text:
        return ""
    upper = text.upper()
    if upper in GEOMETRY_ALIAS_MAP:
        return GEOMETRY_ALIAS_MAP[upper]
    if "TIGER" in upper:
        return "TIGER - PM" if "PM" in upper else "TIGER"
    if "PANDA" in upper:
        return "PANDA - PM"
    if "OCTA" in upper:
        return "Octagonal"
    if "ROUND" in upper or "CIRC" in upper:
        return "ROUND"
    return text


def normalize_coating(raw: Any) -> str:
    text = safe_str(raw)
    return COATING_ALIAS_MAP.get(text.upper(), text)


def load_project_templates() -> pd.DataFrame:
    if not PROJECTS_TEMPLATES.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROJECTS_TEMPLATES)
    if "Fiber Project" in df.columns:
        df["Fiber Project"] = df["Fiber Project"].astype(str).str.strip().map(normalize_project)
    return df


PROJECT_TEMPLATES_DF = load_project_templates()


def template_for_project(project: str) -> pd.Series | None:
    if PROJECT_TEMPLATES_DF.empty or not project:
        return None
    hit = PROJECT_TEMPLATES_DF[PROJECT_TEMPLATES_DF["Fiber Project"].astype(str).str.strip().eq(project.strip())]
    if hit.empty:
        return None
    return hit.iloc[0]


def read_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 and row[:3] == ["Parameter Name", "Value", "Units"]:
                continue
            name = row[0] if len(row) > 0 else ""
            value = row[1] if len(row) > 1 else ""
            units = row[2] if len(row) > 2 else ""
            rows.append((name, value, units))
    return rows


def is_already_new(text: str) -> bool:
    return "=== ORDER PARAMETERS ===" in text and "### DASHBOARD ZONES" in text and "Zone 1 | " in text


def needs_conversion(text: str) -> bool:
    if is_already_new(text):
        return False
    if "=== ORDER PARAMETERS ===" not in text:
        return True
    if "Marked Zone " in text:
        return True
    if re.search(r"Zone\s+\d+\s+Avg\s+\(", text):
        return True
    if re.search(r"^Draw Name,", text, flags=re.M):
        return True
    return False


def build_param_index(rows: list[tuple[str, str, str]]) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str, str]]]:
    latest: dict[str, tuple[str, str]] = {}
    ordered: list[tuple[str, str, str]] = []
    for name, value, units in rows:
        key = safe_str(name)
        ordered.append((key, value, units))
        if key:
            latest[key] = (value, units)
    return latest, ordered


def get_param(latest: dict[str, tuple[str, str]], *names: str) -> str:
    for name in names:
        if name in latest:
            return safe_str(latest[name][0])
    return ""


def get_unit(latest: dict[str, tuple[str, str]], *names: str) -> str:
    for name in names:
        if name in latest:
            return safe_str(latest[name][1])
    return ""


def derive_preform_number(draw_name: str, explicit: str) -> str:
    if explicit:
        return explicit
    m = re.match(r"(.+?)F(?:_\d+)?$", draw_name)
    return m.group(1) if m else draw_name


def normalize_new_value(name: str, value: str) -> str:
    if "Fiber Project" in name:
        return normalize_project(value)
    if "Fiber Geometry Type" in name:
        return normalize_geometry(value)
    if "Main Coating" in name or "Secondary Coating" in name or "Primary Coating" in name:
        return normalize_coating(value)
    return value


def parse_legacy_zones(rows: list[tuple[str, str, str]]) -> dict[int, dict[str, Any]]:
    zones: dict[int, dict[str, Any]] = defaultdict(dict)
    for name, value, units in rows:
        key = safe_str(name)
        if not key:
            continue
        m = re.match(r"Marked Zone (\d+) (Avg|Min|Max) - (.+)$", key)
        if m:
            idx = int(m.group(1))
            stat = m.group(2)
            metric = m.group(3)
            zones[idx].setdefault("stats", {})[(metric, stat)] = (value, units)
            continue
        m = re.match(r"Zone (\d+) (Avg|Min|Max) \((.+)\)$", key)
        if m:
            idx = int(m.group(1))
            stat = m.group(2)
            metric = m.group(3)
            zones[idx].setdefault("stats", {})[(metric, stat)] = (value, units)
            continue
        m = re.match(r"Zone (\d+) Start$", key)
        if m:
            zones[int(m.group(1))]["start"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) End$", key)
        if m:
            zones[int(m.group(1))]["end"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) \|\s*Start$", key)
        if m:
            zones[int(m.group(1))]["start"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) \|\s*End$", key)
        if m:
            zones[int(m.group(1))]["end"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) Fibre Length at Start$", key)
        if m:
            zones[int(m.group(1))]["fiber_start"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) Fibre Length at End$", key)
        if m:
            zones[int(m.group(1))]["fiber_end"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) Pf Process Position at Start$", key)
        if m:
            zones[int(m.group(1))]["pf_start"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) Pf Process Position at End$", key)
        if m:
            zones[int(m.group(1))]["pf_end"] = (value, units)
            continue
        m = re.match(r"Zone (\d+) \|\s*Fiber Length \|\s*(Min|Max)$", key)
        if m:
            zones[int(m.group(1))][f"fiber_{m.group(2).lower()}"] = (value, units)
            continue
        m = re.match(r"Good Zone (\d+) \(km from end\) (Start|End)$", key)
        if m:
            zones[int(m.group(1))][f"kfe_{m.group(2).lower()}"] = (value, units)
            continue
        m = re.match(r"Good Zone (\d+) (Fibre Length Min|Fibre Length Max|Length)$", key)
        if m:
            suffix = {
                "Fibre Length Min": "fiber_min",
                "Fibre Length Max": "fiber_max",
                "Length": "length",
            }[m.group(2)]
            zones[int(m.group(1))][suffix] = (value, units)
    return dict(zones)


def parse_tnm_steps(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    explicit: list[tuple[str, str, str]] = []
    latest, ordered = build_param_index(rows)
    has_explicit = any(name.startswith("T&M Step ") for name, _, _ in ordered)
    if has_explicit:
        return [(n, v, u) for n, v, u in ordered if n.startswith("T&M Step ") or n in {"Total Saved Length", "Total Cut Length"}]
    seq: list[tuple[str, str]] = []
    for name, value, _ in ordered:
        n = safe_str(name).strip()
        if n in {"Cut", "Keep"}:
            seq.append((n.upper(), safe_str(value)))
    step_no = 1
    keep_idx = 1
    total_save = 0.0
    total_cut = 0.0
    for action, value in seq:
        explicit.append((f"T&M Step {step_no} Action", "SAVE" if action == "KEEP" else "CUT", ""))
        explicit.append((f"T&M Step {step_no} Length", value, "km"))
        fv = to_float(value) or 0.0
        if action == "KEEP":
            total_save += fv
            explicit.append((f"T&M Step {step_no} Zone", f"Good Zone {keep_idx}", ""))
            keep_idx += 1
        else:
            total_cut += fv
        step_no += 1
    if seq:
        explicit.append(("Total Saved Length", fmt_value(total_save), "km"))
        explicit.append(("Total Cut Length", fmt_value(total_cut), "km"))
    return explicit


def add_row(out: list[list[str]], name: str, value: Any = "", units: str = "") -> None:
    out.append([name, fmt_value(value), units])


def add_blank(out: list[list[str]]) -> None:
    out.append(["", "", ""])


def build_normalized_rows(rows: list[tuple[str, str, str]], filename: str) -> list[list[str]]:
    latest, ordered = build_param_index(rows)
    zones = parse_legacy_zones(ordered)

    draw_name = get_param(latest, "Order__Draw Name", "Draw Name") or Path(filename).stem
    draw_date = get_param(latest, "Order__Draw Date", "Draw Date")
    preform_number = derive_preform_number(draw_name, get_param(latest, "Order__Preform Number", "Preform Number"))
    project = normalize_project(get_param(latest, "Order__Fiber Project", "Fiber Project"))
    tpl = template_for_project(project)

    geometry = normalize_geometry(get_param(latest, "Order__Fiber Geometry Type", "Fiber Geometry Type"))
    tiger_cut = get_param(latest, "Order__Tiger Cut (%)", "Tiger Cut (%)", "Tiger Cut") or "0.0"
    oct_f2f = get_param(latest, "Order__Octagonal F2F (mm)", "Octagonal F2F (mm)", "Octagonal F2F")
    if not geometry:
        if to_float(tiger_cut) and to_float(tiger_cut) > 0:
            geometry = "TIGER - PM"
        elif oct_f2f and (to_float(oct_f2f) or 0) > 0:
            geometry = "Octagonal"
        else:
            geometry = "ROUND"

    good_zones_count = get_param(latest, "Order__Good Zones Count (required length zones)", "Good Zones Count (required length zones)", "Good Zones Count")
    if not good_zones_count:
        good_zones_count = str(len(zones))

    def pick_value(*names: str, tpl_col: str | None = None, normalizer=None) -> str:
        v = get_param(latest, *names)
        if not v and tpl is not None and tpl_col and tpl_col in tpl.index:
            v = safe_str(tpl.get(tpl_col))
        if normalizer:
            v = normalizer(v)
        return v

    order_fiber_diam = pick_value("Order__Fiber Diameter (µm)", "Fiber Diameter (µm)", "Entry Fiber Diameter", tpl_col="Fiber Diameter (µm)")
    order_fiber_tol = pick_value("Order__Fiber Diameter Tol (± µm)", tpl_col="Fiber Diameter Tol (± µm)")
    main_diam = pick_value("Order__Main Coating Diameter (µm)", "Main Coating Diameter (µm)", "Target First Coating Diameter", tpl_col="Main Coating Diameter (µm)")
    main_tol = pick_value("Order__Main Coating Diameter Tol (± µm)", tpl_col="Main Coating Diameter Tol (± µm)")
    sec_diam = pick_value("Order__Secondary Coating Diameter (µm)", "Secondary Coating Diameter (µm)", "Target Second Coating Diameter", tpl_col="Secondary Coating Diameter (µm)")
    sec_tol = pick_value("Order__Secondary Coating Diameter Tol (± µm)", tpl_col="Secondary Coating Diameter Tol (± µm)")
    tension = pick_value("Order__Tension (g)", "Tension (g)", tpl_col="Tension (g)")
    draw_speed = pick_value("Order__Draw Speed (m/min)", "Draw Speed (m/min)", "Draw Speed", tpl_col="Draw Speed (m/min)")
    main_coating = pick_value("Order__Main Coating", "Main Coating", "Primary Coating", tpl_col="Main Coating", normalizer=normalize_coating)
    sec_coating = pick_value("Order__Secondary Coating", "Secondary Coating", tpl_col="Secondary Coating", normalizer=normalize_coating)
    main_temp = pick_value("Order__Main Coating Temperature (°C)", "Main Coating Temperature (°C)", "Primary Coating Temperature", tpl_col="Main Coating Temperature (°C)")
    sec_temp = pick_value("Order__Secondary Coating Temperature (°C)", "Secondary Coating Temperature (°C)", "Secondary Coating Temperature", "Secondary Coating Temperature (°C)", tpl_col="Secondary Coating Temperature (°C)")

    process_ts = get_param(latest, "Process__Process Setup Timestamp", "Process Setup Timestamp") or draw_date
    preform_diam = get_param(latest, "Process__Preform Diameter", "Preform Diameter")
    preform_shape = get_param(latest, "Process__Preform Shape", "Preform Shape")
    if not preform_shape:
        if "TIGER" in geometry.upper():
            preform_shape = "Tiger Cut"
        elif "OCTA" in geometry.upper():
            preform_shape = "Octagonal"
        else:
            preform_shape = "Circular"
    oct_preform = get_param(latest, "Process__Octagonal Preform", "Octagonal Preform") or ("1" if "OCTA" in geometry.upper() else "0")
    tiger_preform = get_param(latest, "Process__Tiger Preform", "Tiger Preform") or ("1" if "TIGER" in geometry.upper() else "0")
    pm_iris = get_param(latest, "Process__PM Iris System", "PM Iris System") or ("1" if "PM" in geometry.upper() else "0")
    iris_mode = get_param(latest, "Process__Iris Mode", "Iris Mode") or ("PM Auto" if pm_iris == "1" else "Manual")
    base_area = get_param(latest, "Process__Base Area", "Base Area")
    adjusted_area = get_param(latest, "Process__Adjusted Area", "Adjusted Area")
    eff_diam = get_param(latest, "Process__Effective Preform Diameter", "Effective Preform Diameter")
    iris_diam = get_param(latest, "Process__Selected Iris Diameter", "Selected Iris Diameter")
    iris_gap = get_param(latest, "Process__Iris Gap Area", "Iris Gap Area")
    entry_fiber = pick_value("Process__Entry Fiber Diameter", "Entry Fiber Diameter", tpl_col="Fiber Diameter (µm)")
    tgt_first = pick_value("Process__Target First Coating Diameter", "Target First Coating Diameter", "Main Coating Diameter (µm)", tpl_col="Main Coating Diameter (µm)")
    tgt_second = pick_value("Process__Target Second Coating Diameter", "Target Second Coating Diameter", "Secondary Coating Diameter (µm)", tpl_col="Secondary Coating Diameter (µm)")
    first_theory = get_param(latest, "Process__First Coating Diameter (Theoretical)", "First Coating Diameter (Theoretical)")
    second_theory = get_param(latest, "Process__Second Coating Diameter (Theoretical)", "Second Coating Diameter (Theoretical)")
    proc_main = pick_value("Process__Primary Coating", "Primary Coating", "Main Coating", tpl_col="Main Coating", normalizer=normalize_coating)
    proc_sec = pick_value("Process__Secondary Coating", "Secondary Coating", tpl_col="Secondary Coating", normalizer=normalize_coating)
    proc_main_temp = pick_value("Process__Primary Coating Temperature", "Primary Coating Temperature", "Main Coating Temperature (°C)", tpl_col="Main Coating Temperature (°C)")
    proc_sec_temp = pick_value("Process__Secondary Coating Temperature", "Secondary Coating Temperature", "Secondary Coating Temperature (°C)", tpl_col="Secondary Coating Temperature (°C)")
    primary_die_diam = get_param(latest, "Process__Primary Die Diameter", "Primary Die Diameter")
    secondary_die_diam = get_param(latest, "Process__Secondary Die Diameter", "Secondary Die Diameter")
    primary_die_name = get_param(latest, "Process__Primary Die Name", "Primary Die Name") or (f"Die_{int(to_float(primary_die_diam))}" if to_float(primary_die_diam) else "")
    secondary_die_name = get_param(latest, "Process__Secondary Die Name", "Secondary Die Name") or (f"Die_{int(to_float(secondary_die_diam))}" if to_float(secondary_die_diam) else "")
    die_sel_mode = get_param(latest, "Process__Coating Die Selection Mode", "Coating Die Selection Mode") or "Auto"
    proc_draw_speed = pick_value("Process__Draw Speed", "Draw Speed (m/min)", "Draw Speed", tpl_col="Draw Speed (m/min)")
    p_gain = get_param(latest, "Process__P Gain (Diameter Control)", "P Gain (Diameter Control)")
    i_gain = get_param(latest, "Process__I Gain (Diameter Control)", "I Gain (Diameter Control)")
    tf_mode = get_param(latest, "Process__TF Mode", "TF Mode")
    inc_tf = get_param(latest, "Process__Increment TF Value", "Increment TF Value")
    drum = get_param(latest, "Process__Selected Drum", "Selected Drum", "Drum | Selected")

    dashboard_ts = get_param(latest, "Zones Saved Timestamp") or draw_date
    log_file = get_param(latest, "Dashboard Log File", "Log File Name")
    x_col = get_param(latest, "Good Zones X Column") or ("Date/Time" if any("Marked Zone " in n for n, _, _ in ordered) else "Fibre Length")
    fiber_end = get_param(latest, "Fiber Length | End (log end)", "Fiber Length End (log end)", "Last Fiber Position")
    fiber_min_log = get_param(latest, "Fiber Length Min (log)")
    fiber_max_log = get_param(latest, "Fiber Length Max (log)", "Last Fiber Position")
    done_desc = get_param(latest, "Done Description")
    done_ts = get_param(latest, "Done Timestamp")
    preform_after = get_param(latest, "Preform Length After Draw")

    out: list[list[str]] = [["Parameter Name", "Value", "Units"]]
    add_row(out, "=== ORDER PARAMETERS ===", "", "")
    add_row(out, "Order__Draw Name", draw_name)
    add_row(out, "Order__Draw Date", draw_date)
    add_row(out, "Order__Order Index", get_param(latest, "Order__Order Index", "Order Index"))
    add_row(out, "Order__Preform Number", preform_number)
    add_row(out, "Order__Fiber Project", project)
    add_row(out, "Order__Priority", get_param(latest, "Order__Priority", "Priority") or "Normal")
    add_row(out, "Order__Order Opener", get_param(latest, "Order__Order Opener", "Order Opener"))
    add_row(out, "Order__Fiber Geometry Type", geometry)
    add_row(out, "Order__Tiger Cut (%)", tiger_cut, "%")
    add_row(out, "Order__Octagonal F2F (mm)", oct_f2f, "mm")
    add_row(out, "Order__Required Length (m) (for T&M+costumer)", get_param(latest, "Order__Required Length (m) (for T&M+costumer)", "Required Length (m) (for T&M+costumer)"), "m")
    add_row(out, "Order__Good Zones Count (required length zones)", good_zones_count, "count")
    add_row(out, "Order__Fiber Diameter (µm)", order_fiber_diam, "µm")
    if order_fiber_tol:
        add_row(out, "Order__Fiber Diameter Tol (± µm)", order_fiber_tol, "µm")
    add_row(out, "Order__Main Coating Diameter (µm)", main_diam, "µm")
    if main_tol:
        add_row(out, "Order__Main Coating Diameter Tol (± µm)", main_tol, "µm")
    add_row(out, "Order__Secondary Coating Diameter (µm)", sec_diam, "µm")
    if sec_tol:
        add_row(out, "Order__Secondary Coating Diameter Tol (± µm)", sec_tol, "µm")
    add_row(out, "Order__Tension (g)", tension, "g")
    add_row(out, "Order__Draw Speed (m/min)", draw_speed, "m/min")
    add_row(out, "Order__Main Coating", main_coating)
    add_row(out, "Order__Secondary Coating", sec_coating)
    add_row(out, "Order__Main Coating Temperature (°C)", main_temp, "°C")
    add_row(out, "Order__Secondary Coating Temperature (°C)", sec_temp, "°C")
    add_row(out, "Order__Order Notes", get_param(latest, "Order__Order Notes", "Order Notes"))
    add_blank(out)

    add_row(out, "=== PROCESS SETUP ===", "", "")
    add_row(out, "Process__Process Setup Timestamp", process_ts)
    add_row(out, "Process__Preform Diameter", preform_diam, "mm")
    add_row(out, "Process__Preform Shape", preform_shape)
    add_row(out, "Process__Octagonal Preform", oct_preform, "bool")
    add_row(out, "Process__Octagonal F2F", oct_f2f, "mm")
    add_row(out, "Process__Tiger Preform", tiger_preform, "bool")
    add_row(out, "Process__Tiger Cut", tiger_cut, "%")
    add_row(out, "Process__PM Iris System", pm_iris, "bool")
    add_row(out, "Process__Iris Mode", iris_mode)
    add_row(out, "Process__Base Area", base_area, "mm^2")
    add_row(out, "Process__Adjusted Area", adjusted_area, "mm^2")
    add_row(out, "Process__Effective Preform Diameter", eff_diam, "mm")
    add_row(out, "Process__Selected Iris Diameter", iris_diam, "mm")
    add_row(out, "Process__Iris Gap Area", iris_gap, "mm^2")
    add_row(out, "Process__Entry Fiber Diameter", entry_fiber, "µm")
    add_row(out, "Process__Target First Coating Diameter", tgt_first, "µm")
    add_row(out, "Process__Target Second Coating Diameter", tgt_second, "µm")
    add_row(out, "Process__First Coating Diameter (Theoretical)", first_theory, "µm")
    add_row(out, "Process__Second Coating Diameter (Theoretical)", second_theory, "µm")
    add_row(out, "Process__Primary Coating", proc_main)
    add_row(out, "Process__Secondary Coating", proc_sec)
    add_row(out, "Process__Primary Coating Temperature", proc_main_temp, "°C")
    add_row(out, "Process__Secondary Coating Temperature", proc_sec_temp, "°C")
    add_row(out, "Process__Primary Die Diameter", primary_die_diam, "µm")
    add_row(out, "Process__Secondary Die Diameter", secondary_die_diam, "µm")
    add_row(out, "Process__Primary Die Name", primary_die_name)
    add_row(out, "Process__Secondary Die Name", secondary_die_name)
    add_row(out, "Process__Coating Die Selection Mode", die_sel_mode)
    add_row(out, "Process__Draw Speed", proc_draw_speed, "m/min")
    add_row(out, "Process__P Gain (Diameter Control)", p_gain)
    add_row(out, "Process__I Gain (Diameter Control)", i_gain)
    add_row(out, "Process__TF Mode", tf_mode)
    add_row(out, "Process__Increment TF Value", inc_tf, "mm")
    add_row(out, "Process__Selected Drum", drum)
    add_blank(out)

    add_row(out, "=== LOGS DATA SECTION ===", "", "")
    add_blank(out)
    add_row(out, "### DASHBOARD ZONES", "", "")
    add_blank(out)
    add_row(out, "Zones Saved Timestamp", dashboard_ts)
    add_row(out, "Dashboard Log File", log_file)
    add_row(out, "Good Zones Count", good_zones_count, "count")
    add_row(out, "Good Zones X Column", x_col)

    for idx in sorted(zones):
        z = zones[idx]
        if "start" in z:
            add_row(out, f"Zone {idx} | Start", z["start"][0], z["start"][1] or "index/label")
        if "end" in z:
            add_row(out, f"Zone {idx} | End", z["end"][0], z["end"][1] or "index/label")
        for (metric, stat), (value, units) in sorted(z.get("stats", {}).items(), key=lambda kv: (kv[0][0], kv[0][1])):
            add_row(out, f"Zone {idx} | {metric} | {stat}", value, units)

    add_blank(out)
    add_blank(out)
    add_row(out, "### WINDER & LENGTH", "", "")
    add_blank(out)
    add_row(out, "Drum | Selected", drum)
    add_row(out, "Fiber Length | End (log end)", fiber_end, get_unit(latest, "Fiber Length End (log end)", "Last Fiber Position") or "km")
    for idx in sorted(zones):
        z = zones[idx]
        fstart = to_float(z.get("fiber_start", ("", ""))[0])
        fend = to_float(z.get("fiber_end", ("", ""))[0])
        fmin = z.get("fiber_min", ("", ""))[0]
        fmax = z.get("fiber_max", ("", ""))[0]
        if (not fmin and fstart is not None and fend is not None):
            fmin = fmt_value(min(fstart, fend))
        if (not fmax and fstart is not None and fend is not None):
            fmax = fmt_value(max(fstart, fend))
        if fmin:
            add_row(out, f"Zone {idx} | Fiber Length | Min", fmin, "km")
        if fmax:
            add_row(out, f"Zone {idx} | Fiber Length | Max", fmax, "km")

    add_blank(out)
    add_row(out, "—", "—", "")
    add_row(out, "CUT/SAVE Plan Source", get_param(latest, "CUT/SAVE Plan Source") or f"{draw_name}.csv")
    add_row(out, "AUTO Plan Mode", get_param(latest, "AUTO Plan Mode"))
    add_row(out, "Zone Length Column (log)", get_param(latest, "Zone Length Column (log)") or "Fibre Length")
    add_row(out, "Fiber Length End (log end)", fiber_end, get_unit(latest, "Fiber Length End (log end)", "Last Fiber Position") or "km")
    add_row(out, "Fiber Length Min (log)", fiber_min_log, get_unit(latest, "Fiber Length Min (log)") or "km")
    add_row(out, "Fiber Length Max (log)", fiber_max_log, get_unit(latest, "Fiber Length Max (log)", "Last Fiber Position") or "km")
    add_row(out, "T&M Coordinate System", get_param(latest, "T&M Coordinate System"))
    add_row(out, "Good Zones Order", get_param(latest, "Good Zones Order"))

    end_km = to_float(fiber_end)
    for idx in sorted(zones):
        z = zones[idx]
        kstart = z.get("kfe_start", ("", ""))[0]
        kend = z.get("kfe_end", ("", ""))[0]
        zlen = z.get("length", ("", ""))[0]
        fstart = to_float(z.get("fiber_start", ("", ""))[0])
        fend = to_float(z.get("fiber_end", ("", ""))[0])
        fmin = z.get("fiber_min", ("", ""))[0]
        fmax = z.get("fiber_max", ("", ""))[0]
        if (not fmin and fstart is not None and fend is not None):
            fmin = fmt_value(min(fstart, fend))
        if (not fmax and fstart is not None and fend is not None):
            fmax = fmt_value(max(fstart, fend))
        if end_km is not None and not kstart and fmax:
            kstart = fmt_value(max(0.0, end_km - float(fmax)))
        if end_km is not None and not kend and fmin:
            kend = fmt_value(max(0.0, end_km - float(fmin)))
        if not zlen and kstart and kend:
            zlen = fmt_value(float(kend) - float(kstart))
        if kstart:
            add_row(out, f"Good Zone {idx} (km from end) Start", kstart, "km")
        if kend:
            add_row(out, f"Good Zone {idx} (km from end) End", kend, "km")
        if zlen:
            add_row(out, f"Good Zone {idx} Length", zlen, "km")
        if fmin:
            add_row(out, f"Good Zone {idx} Fibre Length Min", fmin, "km")
        if fmax:
            add_row(out, f"Good Zone {idx} Fibre Length Max", fmax, "km")

    for name, value, units in parse_tnm_steps(ordered):
        add_row(out, name, value, units)

    add_row(out, "Preform Length After Draw", preform_after, get_unit(latest, "Preform Length After Draw") or "cm")
    add_row(out, "Done Description", done_desc)
    add_row(out, "Done Timestamp", done_ts)
    return out


def normalize_existing_new(path: Path) -> bool:
    rows = read_rows(path)
    out = [["Parameter Name", "Value", "Units"]]
    changed = False
    for name, value, units in rows:
        new_name = name
        if name.startswith("Marked Zone "):
            m = re.match(r"Marked Zone (\d+) (Avg|Min|Max) - (.+)$", name)
            if m:
                new_name = f"Zone {m.group(1)} | {m.group(3)} | {m.group(2)}"
                changed = True
        elif re.match(r"Zone \d+ Start$", name):
            new_name = name.replace(" Start", " | Start")
            changed = True
        elif re.match(r"Zone \d+ End$", name):
            new_name = name.replace(" End", " | End")
            changed = True
        new_value = normalize_new_value(new_name, value)
        if new_value != value:
            changed = True
        out.append([new_name, new_value, units])
    if changed:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(out)
    return changed


def backup_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "backups" / f"dataset_csv_legacy_backup_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    backup = backup_dir()
    converted = 0
    normalized = 0
    untouched = 0
    changed_files: list[str] = []
    for path in sorted(DATASET_DIR.glob("*.csv")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if is_already_new(text):
            if normalize_existing_new(path):
                normalized += 1
                changed_files.append(path.name)
            else:
                untouched += 1
            continue
        if not needs_conversion(text):
            untouched += 1
            continue
        shutil.copy2(path, backup / path.name)
        rows = read_rows(path)
        new_rows = build_normalized_rows(rows, path.name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(new_rows)
        converted += 1
        changed_files.append(path.name)
    print(f"backup={backup}")
    print(f"converted={converted}")
    print(f"normalized={normalized}")
    print(f"untouched={untouched}")
    for name in changed_files:
        print(name)


if __name__ == "__main__":
    main()
