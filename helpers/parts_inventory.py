import os
from datetime import datetime

import pandas as pd


INVENTORY_COLUMNS = [
    "Part Name",
    "Component",
    "Serial Number",
    "Location",
    "Location Serial",
    "Quantity",
    "Min Level",
    "Notes",
    "Last Updated",
]

LOCATION_COLUMNS = [
    "Location Name",
    "Location Serial",
    "Description",
    "Active",
    "Last Updated",
]


def _norm(v: str) -> str:
    return str(v or "").strip().lower()


def ensure_inventory_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=INVENTORY_COLUMNS).to_csv(path, index=False)


def load_inventory(path: str) -> pd.DataFrame:
    ensure_inventory_file(path)
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        df = pd.DataFrame(columns=INVENTORY_COLUMNS)
    for c in INVENTORY_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    # Keep text-like columns explicitly as string for Streamlit data_editor compatibility.
    for c in ["Part Name", "Component", "Serial Number", "Location", "Location Serial", "Notes", "Last Updated"]:
        df[c] = df[c].fillna("").astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0).astype(float)
    df["Min Level"] = pd.to_numeric(df["Min Level"], errors="coerce").fillna(0.0).astype(float)
    return df[INVENTORY_COLUMNS].copy()


def save_inventory(path: str, df: pd.DataFrame) -> None:
    out = df.copy()
    for c in INVENTORY_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    for c in ["Part Name", "Component", "Serial Number", "Location", "Location Serial", "Notes", "Last Updated"]:
        out[c] = out[c].fillna("").astype(str)
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0.0).astype(float)
    out["Min Level"] = pd.to_numeric(out["Min Level"], errors="coerce").fillna(0.0).astype(float)
    out.to_csv(path, index=False)


def _find_row(df: pd.DataFrame, part_name: str, serial_number: str = "") -> int:
    pn = _norm(part_name)
    sn = _norm(serial_number)
    for i, r in df.iterrows():
        if _norm(r.get("Part Name", "")) == pn and _norm(r.get("Serial Number", "")) == sn:
            return int(i)
    if sn:
        for i, r in df.iterrows():
            if _norm(r.get("Part Name", "")) == pn:
                return int(i)
    return -1


def increment_part(
    path: str,
    part_name: str,
    *,
    qty: float = 1.0,
    component: str = "",
    serial_number: str = "",
    location: str = "",
    location_serial: str = "",
    notes: str = "",
) -> None:
    if not str(part_name).strip() or float(qty) <= 0:
        return
    df = load_inventory(path)
    idx = _find_row(df, part_name, serial_number)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if idx >= 0:
        df.at[idx, "Quantity"] = float(df.at[idx, "Quantity"]) + float(qty)
        if component and not str(df.at[idx, "Component"]).strip():
            df.at[idx, "Component"] = str(component).strip()
        if location and not str(df.at[idx, "Location"]).strip():
            df.at[idx, "Location"] = str(location).strip()
        if location_serial and not str(df.at[idx, "Location Serial"]).strip():
            df.at[idx, "Location Serial"] = str(location_serial).strip()
        if notes and not str(df.at[idx, "Notes"]).strip():
            df.at[idx, "Notes"] = str(notes).strip()
        df.at[idx, "Last Updated"] = now
    else:
        row = {
            "Part Name": str(part_name).strip(),
            "Component": str(component or "").strip(),
            "Serial Number": str(serial_number or "").strip(),
            "Location": str(location or "").strip(),
            "Location Serial": str(location_serial or "").strip(),
            "Quantity": float(qty),
            "Min Level": 0.0,
            "Notes": str(notes or "").strip(),
            "Last Updated": now,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_inventory(path, df)


def decrement_part(path: str, part_name: str, *, qty: float = 1.0, serial_number: str = "") -> bool:
    if not str(part_name).strip() or float(qty) <= 0:
        return False
    df = load_inventory(path)
    idx = _find_row(df, part_name, serial_number)
    if idx < 0:
        return False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_qty = max(0.0, float(df.at[idx, "Quantity"]) - float(qty))
    df.at[idx, "Quantity"] = new_qty
    df.at[idx, "Last Updated"] = now
    save_inventory(path, df)
    return True


def set_part_quantity(
    path: str,
    part_name: str,
    *,
    qty: float,
    component: str = "",
    serial_number: str = "",
    location: str = "",
    location_serial: str = "",
    notes: str = "",
) -> None:
    if not str(part_name).strip():
        return
    qty_i = max(0.0, float(qty))
    df = load_inventory(path)
    idx = _find_row(df, part_name, serial_number)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if idx >= 0:
        df.at[idx, "Quantity"] = qty_i
        if component and not str(df.at[idx, "Component"]).strip():
            df.at[idx, "Component"] = str(component).strip()
        if location and not str(df.at[idx, "Location"]).strip():
            df.at[idx, "Location"] = str(location).strip()
        if location_serial and not str(df.at[idx, "Location Serial"]).strip():
            df.at[idx, "Location Serial"] = str(location_serial).strip()
        if notes:
            df.at[idx, "Notes"] = str(notes).strip()
        df.at[idx, "Last Updated"] = now
    else:
        row = {
            "Part Name": str(part_name).strip(),
            "Component": str(component or "").strip(),
            "Serial Number": str(serial_number or "").strip(),
            "Location": str(location or "").strip(),
            "Location Serial": str(location_serial or "").strip(),
            "Quantity": qty_i,
            "Min Level": 0.0,
            "Notes": str(notes or "").strip(),
            "Last Updated": now,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_inventory(path, df)


def ensure_locations_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=LOCATION_COLUMNS).to_csv(path, index=False)


def load_locations(path: str) -> pd.DataFrame:
    ensure_locations_file(path)
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        df = pd.DataFrame(columns=LOCATION_COLUMNS)
    for c in LOCATION_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in ["Location Name", "Location Serial", "Description", "Last Updated"]:
        df[c] = df[c].fillna("").astype(str)
    if "Active" in df.columns:
        df["Active"] = df["Active"].astype(str).str.strip().replace({"": "Yes"})
    return df[LOCATION_COLUMNS].copy()


def save_locations(path: str, df: pd.DataFrame) -> None:
    out = df.copy()
    for c in LOCATION_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    for c in ["Location Name", "Location Serial", "Description", "Last Updated"]:
        out[c] = out[c].fillna("").astype(str)
    out["Active"] = out["Active"].fillna("").astype(str).replace({"": "Yes"})
    out.to_csv(path, index=False)
