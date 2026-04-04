#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCEL = Path("/Users/ohadformanair/Downloads/OldtowerDB-3.xlsx")
OUT_DIR = ROOT / "legacy_converted_out_v2"
SHEET = "Drawing_data_base"
PROJECT_TEMPLATES_CSV = ROOT / "data" / "projects_fiber_templates.csv"

PROJECT_ALIAS_MAP = {
    "ELOP - LARA": "LARA - ELOP",
    "MKS": "ELOP - 130/10",
}

GEOMETRY_ALIAS_MAP = {
    "ROUND": "ROUND",
    "OCTAGONAL": "Octagonal",
    "PANDA - PM": "PANDA - PM",
    "STEP INDEX": "STEP INDEX",
    "TIGER - PM": "TIGER - PM",
}

COATING_ALIAS_MAP = {
    "OF - 136": "Coating_OF_136",
    "OF-136": "Coating_OF_136",
    "COATING_OF_136": "Coating_OF_136",
    "DP-1032": "DP-1032",
    "DS-2015": "DS-2015",
    "DS-2042": "DS-2042",
}


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _is_missing(value: Any) -> bool:
    return _clean_str(value) in {"", "-"}


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = _clean_str(value)
    if text in {"", "-"}:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else None


def _fmt_dt(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _add(rows: list[dict[str, Any]], name: str, value: Any = "", units: str = "") -> None:
    if value is None:
        value = ""
    rows.append({"Parameter Name": name, "Value": value, "Units": units})


def _blank(rows: list[dict[str, Any]]) -> None:
    rows.append({"Parameter Name": "", "Value": "", "Units": ""})


def _section(rows: list[dict[str, Any]], name: str) -> None:
    _add(rows, name, "", "")


def _slug_filename(name: str) -> str:
    text = _clean_str(name) or "legacy_draw"
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_geometry(raw: str) -> str:
    text = _clean_str(raw).upper()
    if text in GEOMETRY_ALIAS_MAP:
        return GEOMETRY_ALIAS_MAP[text]
    if "TIGER" in text:
        return "TIGER - PM" if "PM" in text else "TIGER"
    if "PANDA" in text:
        return "PANDA - PM"
    if "PM" in text:
        return "PM"
    if "OCTA" in text:
        return "Octagonal"
    if "ROUND" in text:
        return "Round"
    return _clean_str(raw)


def _normalize_project(raw: Any) -> str:
    text = _clean_str(raw)
    if not text:
        return ""
    return PROJECT_ALIAS_MAP.get(text, text)


def _normalize_coating(raw: Any) -> str:
    text = _clean_str(raw).upper()
    if not text:
        return ""
    return COATING_ALIAS_MAP.get(text, _clean_str(raw))


def _process_shape(row: pd.Series) -> str:
    geom = _normalize_geometry(row.get("Fiber Geometry Type", ""))
    if "TIGER" in geom:
        return "Tiger Cut"
    if "OCTA" in geom.upper():
        return "Octagonal"
    if geom:
        return "Circular"
    return ""


def _bool01(flag: bool) -> int:
    return 1 if flag else 0


def _coating_map(row: pd.Series) -> tuple[str, str]:
    main = _normalize_coating(row.get("Main Coating"))
    sec = _normalize_coating(row.get("Secondary Coating"))
    coating_type = _clean_str(row.get("Coating type")).upper()
    if main or sec:
        return main, sec
    mapped = {
        "DCOF": ("Coating_OF_136", "DS-2015"),
        "SCOF": ("", ""),
    }
    return mapped.get(coating_type, ("", ""))


def _circle_area_mm2(diameter_mm: float | None) -> float | None:
    if diameter_mm is None:
        return None
    return math.pi * (diameter_mm ** 2) / 4.0


def _zone_rows(row: pd.Series, rows: list[dict[str, Any]], draw_name: str) -> None:
    total_fiber_m = _to_float(row.get("Total Fiber"))
    end_km = (total_fiber_m / 1000.0) if total_fiber_m is not None else None

    zones: list[dict[str, Any]] = []
    zone_cols = {
        1: ("Start of good zone 1", "End good Zone 1", "Unnamed: 58", "Unnamed: 60"),
        2: ("Start of good zone 2", "End good Zone 2", "Unnamed: 62", "Unnamed: 64"),
        3: ("Start of good zone 3", "End good Zone 3", "Unnamed: 66", "Unnamed: 68"),
    }

    for idx, (start_col, end_col, pre_start_col, pre_end_col) in zone_cols.items():
        sf = _to_float(row.get(start_col))
        ef = _to_float(row.get(end_col))
        if sf is None or ef is None:
            continue
        zones.append(
            {
                "idx": idx,
                "sf": sf,
                "ef": ef,
                "fl_min": min(sf, ef),
                "fl_max": max(sf, ef),
                "sp_cm": _to_float(row.get(pre_start_col)),
                "ep_cm": _to_float(row.get(pre_end_col)),
            }
        )

    _section(rows, "### DASHBOARD ZONES")
    _blank(rows)
    _add(rows, "Zones Saved Timestamp", _fmt_dt(row.get("Draw Date")))
    _add(rows, "Dashboard Log File", "")
    _add(rows, "Good Zones Count", len(zones), "count")
    _add(rows, "Good Zones X Column", "Fibre Length")

    bare_um = _to_float(row.get("Entry Fiber Diameter (µm)"))
    capstan = _to_float(row.get("Vn (m/min) - Actual")) or _to_float(row.get("Draw Speed (m/min)"))
    tension = _to_float(row.get("T (g) - Actual")) or _to_float(row.get("Tension (g)"))
    furnace = _to_float(row.get("Final Process temp"))
    coat1 = _to_float(row.get("Target First Coating Diameter (µm)"))
    coat2 = _to_float(row.get("Target Second Coating Diameter (µm)"))

    for z in zones:
        idx = z["idx"]
        avg_len = (z["fl_min"] + z["fl_max"]) / 2.0
        _add(rows, f"Zone {idx} | Start", z["sf"], "index/label")
        _add(rows, f"Zone {idx} | End", z["ef"], "index/label")

        metric_values = [
            ("Bare Fibre Diameter", bare_um, ""),
            ("Capstan Speed", capstan, ""),
            ("Tension N", tension, "g"),
            ("Furnace DegC Actual", furnace, ""),
            ("Furnace DegC Set", furnace, ""),
            ("Fibre Length", avg_len, ""),
            ("Coated Inner Diameter", coat1, ""),
            ("Coated Outer Diameter", coat2, ""),
        ]
        for metric, base_val, unit in metric_values:
            if base_val is None:
                continue
            min_val = z["fl_min"] if metric == "Fibre Length" else base_val
            max_val = z["fl_max"] if metric == "Fibre Length" else base_val
            avg_val = avg_len if metric == "Fibre Length" else base_val
            _add(rows, f"Zone {idx} | {metric} | Avg", avg_val, unit)
            _add(rows, f"Zone {idx} | {metric} | Min", min_val, unit)
            _add(rows, f"Zone {idx} | {metric} | Max", max_val, unit)

        if z["sp_cm"] is not None and z["ep_cm"] is not None:
            pf_min = min(z["sp_cm"], z["ep_cm"])
            pf_max = max(z["sp_cm"], z["ep_cm"])
            pf_avg = (pf_min + pf_max) / 2.0
            _add(rows, f"Zone {idx} | Pf Process Position | Avg", pf_avg, "")
            _add(rows, f"Zone {idx} | Pf Process Position | Min", pf_min, "")
            _add(rows, f"Zone {idx} | Pf Process Position | Max", pf_max, "")

        if end_km is not None:
            kz_start = max(0.0, end_km - z["fl_max"])
            kz_end = max(0.0, end_km - z["fl_min"])
            _add(rows, f"Good Zone {idx} (km from end) Start", kz_start, "km")
            _add(rows, f"Good Zone {idx} (km from end) End", kz_end, "km")
            _add(rows, f"Good Zone {idx} Length", kz_end - kz_start, "km")
        _add(rows, f"Good Zone {idx} Fibre Length Min", z["fl_min"], "km")
        _add(rows, f"Good Zone {idx} Fibre Length Max", z["fl_max"], "km")

    _blank(rows)
    _blank(rows)
    _section(rows, "### WINDER & LENGTH")
    _blank(rows)
    _add(rows, "Drum | Selected", _clean_str(row.get("Winding")))
    if end_km is not None:
        _add(rows, "Fiber Length | End (log end)", end_km, "km")
    for z in zones:
        idx = z["idx"]
        _add(rows, f"Zone {idx} | Fiber Length | Min", z["fl_min"], "km")
        _add(rows, f"Zone {idx} | Fiber Length | Max", z["fl_max"], "km")

    _blank(rows)
    _add(rows, "—", "—", "")
    _add(rows, "CUT/SAVE Plan Source", f"{draw_name}.csv")
    _add(rows, "AUTO Plan Mode", "Legacy import from cuts/good fiber")
    _add(rows, "Zone Length Column (log)", "Fibre Length")
    if end_km is not None:
        _add(rows, "Fiber Length End (log end)", end_km, "km")
        _add(rows, "Fiber Length Min (log)", 0.0, "km")
        _add(rows, "Fiber Length Max (log)", end_km, "km")
    _add(rows, "T&M Coordinate System", "km_from_end = (L_end - FibreLength)")
    _add(rows, "Good Zones Order", "Ordered from spool end → toward start")

    total_saved_km = 0.0
    total_cut_km = 0.0
    step_no = 1
    for idx in (1, 2, 3):
        cut_val = _to_float(row.get(f"Cut {idx}"))
        save_val = _to_float(row.get(f"Good fiber {idx}"))
        if cut_val is not None and cut_val > 0:
            cut_km = cut_val / 1000.0
            total_cut_km += cut_km
            _add(rows, f"T&M Step {step_no} Action", "CUT")
            _add(rows, f"T&M Step {step_no} Length", cut_km, "km")
            step_no += 1
        if save_val is not None and save_val > 0:
            save_km = save_val / 1000.0
            total_saved_km += save_km
            _add(rows, f"T&M Step {step_no} Action", "SAVE")
            _add(rows, f"T&M Step {step_no} Length", save_km, "km")
            _add(rows, f"T&M Step {step_no} Zone", f"Good Zone {idx}")
            if idx <= len(zones):
                z = zones[idx - 1]
                _add(rows, f"T&M Step {step_no} Fibre Length Min", z["fl_min"], "km")
                _add(rows, f"T&M Step {step_no} Fibre Length Max", z["fl_max"], "km")
            step_no += 1

    _add(rows, "Total Saved Length", total_saved_km, "km")
    _add(rows, "Total Cut Length", total_cut_km, "km")
    preform_after = _to_float(row.get("Unnamed: 70"))
    _add(rows, "Preform Length After Draw", preform_after if preform_after is not None else "", "cm")
    _add(rows, "Done Description", _clean_str(row.get("Done description")))
    _add(rows, "Done Timestamp", _fmt_dt(row.get("Draw Date")))


def _load_project_templates() -> pd.DataFrame:
    if not PROJECT_TEMPLATES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROJECT_TEMPLATES_CSV, keep_default_na=False)
    df["Fiber Project"] = df["Fiber Project"].map(_normalize_project)
    df["Fiber Geometry Type"] = df["Fiber Geometry Type"].map(_normalize_geometry)
    return df


PROJECT_TEMPLATES = _load_project_templates()


def _pick_template(project: str, geometry: str) -> pd.Series | None:
    if PROJECT_TEMPLATES.empty:
        return None
    hit = PROJECT_TEMPLATES[
        (PROJECT_TEMPLATES["Fiber Project"].astype(str).str.strip() == project.strip())
    ]
    if geometry and not hit.empty:
        exact = hit[hit["Fiber Geometry Type"].astype(str).str.strip() == geometry.strip()]
        if not exact.empty:
            return exact.iloc[0]
    if not hit.empty:
        return hit.iloc[0]
    if geometry:
        exact_geom = PROJECT_TEMPLATES[PROJECT_TEMPLATES["Fiber Geometry Type"].astype(str).str.strip() == geometry.strip()]
        if not exact_geom.empty:
            return exact_geom.iloc[0]
    return None


def _value_or_template(raw: Any, tpl: pd.Series | None, tpl_col: str) -> Any:
    text = _clean_str(raw)
    if text not in {"", "-"}:
        return raw
    if tpl is None:
        return ""
    return tpl.get(tpl_col, "")


def _build_dataset_rows(row: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    draw_name = _clean_str(row.get("Draw Name"))
    preform = _clean_str(row.get("Preform Number"))
    project = _normalize_project(row.get("Project"))
    order_index = _to_float(row.get("Order Index"))
    geom = _normalize_geometry(row.get("Fiber Geometry Type"))
    tpl = _pick_template(project, geom)
    main_coating, sec_coating = _coating_map(row)
    if not main_coating and tpl is not None:
        main_coating = _normalize_coating(tpl.get("Main Coating", ""))
    if not sec_coating and tpl is not None:
        sec_coating = _normalize_coating(tpl.get("Secondary Coating", ""))
    geom = _normalize_geometry(row.get("Fiber Geometry Type"))
    octa_f2f = _to_float(row.get("Octagonal F2F (mm)"))
    tiger_cut = (_to_float(_value_or_template("", tpl, "Tiger Cut (%)")) or 0.0) if "TIGER" in geom.upper() else 0.0
    tiger = "TIGER" in geom.upper()
    octa = (octa_f2f or 0) > 0 or "OCTA" in geom.upper()
    pm = "PM" in geom.upper()
    preform_diam = _to_float(row.get("Preform Diameter"))
    iris_diam = _to_float(row.get("Selected Iris Diameter"))
    base_area = _circle_area_mm2(preform_diam)
    iris_area = _circle_area_mm2(iris_diam)
    iris_gap = (iris_area - base_area) if iris_area is not None and base_area is not None else None
    fiber_diam = _to_float(_value_or_template(row.get("Entry Fiber Diameter (µm)"), tpl, "Fiber Diameter (µm)"))
    coat1_diam = _to_float(_value_or_template(row.get("Target First Coating Diameter (µm)"), tpl, "Main Coating Diameter (µm)"))
    coat2_diam = _to_float(_value_or_template(row.get("Target Second Coating Diameter (µm)"), tpl, "Secondary Coating Diameter (µm)"))
    tension = _to_float(_value_or_template(row.get("Tension (g)"), tpl, "Tension (g)"))
    draw_speed = _to_float(_value_or_template(row.get("Draw Speed (m/min)"), tpl, "Draw Speed (m/min)"))
    main_temp = _to_float(_value_or_template(row.get("Main Coating Temperature (°C)"), tpl, "Main Coating Temperature (°C)"))
    sec_temp = _to_float(_value_or_template(row.get("Secondary Coating Temperature (°C)"), tpl, "Secondary Coating Temperature (°C)"))
    fiber_tol = _to_float(tpl.get("Fiber Diameter Tol (± µm)", "")) if tpl is not None else None
    coat1_tol = _to_float(tpl.get("Main Coating Diameter Tol (± µm)", "")) if tpl is not None else None
    coat2_tol = _to_float(tpl.get("Secondary Coating Diameter Tol (± µm)", "")) if tpl is not None else None
    notes_default = _clean_str(tpl.get("Notes Default", "")) if tpl is not None else ""

    _add(rows, "Parameter Name", "Value", "Units")  # temporary marker removed later

    _section(rows, "=== ORDER PARAMETERS ===")
    _add(rows, "Order__Draw Name", draw_name)
    _add(rows, "Order__Draw Date", _fmt_dt(row.get("Draw Date")))
    _add(rows, "Order__Order Index", "" if order_index in {None, -1.0} else int(order_index) if float(order_index).is_integer() else order_index)
    _add(rows, "Order__Preform Number", preform)
    _add(rows, "Order__Fiber Project", project)
    _add(rows, "Order__Priority", "Normal")
    _add(rows, "Order__Order Opener", "")
    _add(rows, "Order__Fiber Geometry Type", geom)
    _add(rows, "Order__Tiger Cut (%)", tiger_cut, "%")
    _add(rows, "Order__Octagonal F2F (mm)", octa_f2f if octa_f2f is not None else "", "mm")
    _add(rows, "Order__Required Length (m) (for T&M+costumer)", _clean_str(row.get("Required Length (m) (for T&M+costumer)")), "m")
    good_zones_count = _to_float(row.get("Good Zones Count (required length zones)"))
    _add(rows, "Order__Good Zones Count (required length zones)", int(good_zones_count) if good_zones_count is not None else "", "count")
    _add(rows, "Order__Fiber Diameter (µm)", fiber_diam or "", "µm")
    if fiber_tol is not None:
        _add(rows, "Order__Fiber Diameter Tol (± µm)", fiber_tol, "µm")
    _add(rows, "Order__Main Coating Diameter (µm)", coat1_diam or "", "µm")
    if coat1_tol is not None:
        _add(rows, "Order__Main Coating Diameter Tol (± µm)", coat1_tol, "µm")
    _add(rows, "Order__Secondary Coating Diameter (µm)", coat2_diam or "", "µm")
    if coat2_tol is not None:
        _add(rows, "Order__Secondary Coating Diameter Tol (± µm)", coat2_tol, "µm")
    _add(rows, "Order__Tension (g)", tension or "", "g")
    _add(rows, "Order__Draw Speed (m/min)", draw_speed or "", "m/min")
    _add(rows, "Order__Main Coating", main_coating)
    _add(rows, "Order__Secondary Coating", sec_coating)
    _add(rows, "Order__Main Coating Temperature (°C)", main_temp or "", "°C")
    _add(rows, "Order__Secondary Coating Temperature (°C)", sec_temp or "", "°C")
    notes = _clean_str(row.get("Order Notes")) or notes_default
    _add(rows, "Order__Order Notes", notes)
    _blank(rows)

    _section(rows, "=== PROCESS SETUP ===")
    _add(rows, "Process__Process Setup Timestamp", _fmt_dt(row.get("Process Setup Timestamp")))
    _add(rows, "Process__Preform Diameter", preform_diam or "", "mm")
    _add(rows, "Process__Preform Shape", _process_shape(row))
    _add(rows, "Process__Octagonal Preform", _bool01(octa), "bool")
    _add(rows, "Process__Octagonal F2F", octa_f2f if octa_f2f is not None else "", "mm")
    _add(rows, "Process__Tiger Preform", _bool01(tiger), "bool")
    _add(rows, "Process__Tiger Cut", tiger_cut, "%")
    _add(rows, "Process__PM Iris System", _bool01(pm), "bool")
    _add(rows, "Process__Iris Mode", "PM Auto" if pm else "Manual")
    _add(rows, "Process__Base Area", base_area if base_area is not None else "", "mm^2")
    _add(rows, "Process__Adjusted Area", base_area if base_area is not None else "", "mm^2")
    _add(rows, "Process__Effective Preform Diameter", preform_diam if preform_diam is not None else "", "mm")
    _add(rows, "Process__Selected Iris Diameter", iris_diam if iris_diam is not None else "", "mm")
    _add(rows, "Process__Iris Gap Area", iris_gap if iris_gap is not None else "", "mm^2")
    _add(rows, "Process__Entry Fiber Diameter", fiber_diam or "", "µm")
    _add(rows, "Process__Target First Coating Diameter", coat1_diam or "", "µm")
    _add(rows, "Process__Target Second Coating Diameter", coat2_diam or "", "µm")
    _add(rows, "Process__First Coating Diameter (Theoretical)", coat1_diam or "", "µm")
    _add(rows, "Process__Second Coating Diameter (Theoretical)", coat2_diam or "", "µm")
    _add(rows, "Process__Primary Coating", main_coating)
    _add(rows, "Process__Secondary Coating", sec_coating)
    _add(rows, "Process__Primary Coating Temperature", main_temp or "", "°C")
    _add(rows, "Process__Secondary Coating Temperature", sec_temp or "", "°C")
    _add(rows, "Process__Primary Die Diameter", "", "µm")
    _add(rows, "Process__Secondary Die Diameter", "", "µm")
    _add(rows, "Process__Primary Die Name", "")
    _add(rows, "Process__Secondary Die Name", "")
    _add(rows, "Process__Coating Die Selection Mode", "Legacy Import")
    _add(rows, "Process__Draw Speed", draw_speed or "", "m/min")
    _add(rows, "Process__P Gain (Diameter Control)", "")
    _add(rows, "Process__I Gain (Diameter Control)", "")
    _add(rows, "Process__TF Mode", "Winder")
    _add(rows, "Process__Increment TF Value", "")
    _add(rows, "Process__Selected Drum", _clean_str(row.get("Winding")))
    _blank(rows)

    _section(rows, "=== LOGS DATA SECTION ===")
    _blank(rows)
    _zone_rows(row, rows, draw_name)

    rows = [r for r in rows if r["Parameter Name"] != "Parameter Name"]
    audit = {
        "draw_name": draw_name,
        "project": project,
        "geometry": geom,
        "template_used": _clean_str(tpl.get("Fiber Project")) if tpl is not None else "",
        "secondary_coating_filled": bool(not _clean_str(row.get("Secondary Coating")) and sec_coating),
        "notes_default_filled": bool(not _clean_str(row.get("Order Notes")) and notes_default),
    }
    return rows, audit


def _load_legacy(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET, header=1)
    df = df.iloc[1:].copy().reset_index(drop=True)
    df = df[df["Preform Number"].notna()].copy().reset_index(drop=True)
    return df


def convert_workbook(excel_path: Path = DEFAULT_EXCEL, out_dir: Path = OUT_DIR) -> tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_legacy(excel_path)
    written = 0
    used_names: dict[str, int] = {}
    audit_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        draw_name = _slug_filename(_clean_str(row.get("Draw Name")))
        if not draw_name:
            preform = _slug_filename(_clean_str(row.get("Preform Number")))
            draw_name = preform or "legacy_draw"
        count = used_names.get(draw_name, 0) + 1
        used_names[draw_name] = count
        final_name = draw_name if count == 1 else f"{draw_name}_{count}"
        out_csv = out_dir / f"{final_name}.csv"

        rows, audit = _build_dataset_rows(row)
        pd.DataFrame(rows, columns=["Parameter Name", "Value", "Units"]).to_csv(out_csv, index=False)
        written += 1
        audit["output_csv"] = out_csv.name
        audit_rows.append(audit)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.glob("*.csv")):
            zf.write(p, arcname=p.name)
    if audit_rows:
        audit_path = out_dir / "_conversion_audit.csv"
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(audit_path, arcname=audit_path.name)
    return written, zip_path


def main() -> None:
    written, zip_path = convert_workbook()
    print(f"Converted {written} legacy rows into {OUT_DIR}")
    print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
