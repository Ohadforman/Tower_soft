import os
from datetime import datetime

import pandas as pd


INVENTORY_COLUMNS = [
    "Part Name",
    "Item Type",
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

DEFAULT_GENERAL_TOOLS = [
    "Adjustable wrench set",
    "Allen / hex key set",
    "Screwdriver set (flat/phillips)",
    "Torque wrench",
    "Spanner set",
    "Pliers set",
    "Side cutter",
    "Cleaning brush set",
]


def _norm(v: str) -> str:
    return str(v or "").strip().lower()


def _looks_like_tool_name(part_name: str) -> bool:
    p = _norm(part_name)
    if not p:
        return False
    tool_tokens = [
        "cleaning kit",
        "cleaning cloth",
        "tool",
        "wrench",
        "screwdriver",
        "hex key",
        "allen key",
        "spanner",
    ]
    return any(tok in p for tok in tool_tokens)


def _item_type_default(component: str, part_name: str = "") -> str:
    c = _norm(component)
    if _looks_like_tool_name(part_name):
        return "Tool"
    if c == "consumables":
        return "Consumable"
    return "Part"


def _is_non_consumable_item_type(item_type: str) -> bool:
    return _norm(item_type) == "tool"


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
    # Backward compatibility for older files: infer missing item type.
    df["Item Type"] = (
        df["Item Type"]
        .fillna("")
        .astype(str)
        .apply(lambda x: x.strip())
    )
    missing_type = df["Item Type"].eq("")
    if missing_type.any():
        df.loc[missing_type, "Item Type"] = (
            df.loc[missing_type]
            .apply(lambda r: _item_type_default(r.get("Component", ""), r.get("Part Name", "")), axis=1)
        )
    # Keep text-like columns explicitly as string for Streamlit data_editor compatibility.
    for c in ["Part Name", "Item Type", "Component", "Serial Number", "Location", "Location Serial", "Notes", "Last Updated"]:
        df[c] = df[c].fillna("").astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0).astype(float)
    df["Min Level"] = pd.to_numeric(df["Min Level"], errors="coerce").fillna(0.0).astype(float)
    return df[INVENTORY_COLUMNS].copy()


def save_inventory(path: str, df: pd.DataFrame) -> None:
    out = df.copy()
    for c in INVENTORY_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    for c in ["Part Name", "Item Type", "Component", "Serial Number", "Location", "Location Serial", "Notes", "Last Updated"]:
        out[c] = out[c].fillna("").astype(str)
    out["Item Type"] = out["Item Type"].astype(str).str.strip()
    out.loc[out["Item Type"].eq(""), "Item Type"] = (
        out[out["Item Type"].eq("")]
        .apply(lambda r: _item_type_default(r.get("Component", ""), r.get("Part Name", "")), axis=1)
    )
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0.0).astype(float)
    out["Min Level"] = pd.to_numeric(out["Min Level"], errors="coerce").fillna(0.0).astype(float)
    out.to_csv(path, index=False)


def ensure_general_tools_seed(path: str) -> int:
    """Ensure common general-tools rows exist in inventory.

    Returns the number of newly inserted rows.
    """
    df = load_inventory(path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0
    existing = {
        _norm(r.get("Part Name", ""))
        for _, r in df.iterrows()
        if _norm(r.get("Part Name", ""))
    }
    rows = []
    for tool_name in DEFAULT_GENERAL_TOOLS:
        if _norm(tool_name) in existing:
            continue
        rows.append(
            {
                "Part Name": str(tool_name).strip(),
                "Item Type": "Tool",
                "Component": "General Tools",
                "Serial Number": "",
                "Location": "",
                "Location Serial": "",
                "Quantity": 0.0,
                "Min Level": 0.0,
                "Notes": "Seeded tool template: set qty/location as available.",
                "Last Updated": now,
            }
        )
        added += 1
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_inventory(path, df)
    return added


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
    item_type: str = "",
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
        if item_type and not str(df.at[idx, "Item Type"]).strip():
            df.at[idx, "Item Type"] = str(item_type).strip()
        df.at[idx, "Last Updated"] = now
    else:
        row = {
            "Part Name": str(part_name).strip(),
            "Item Type": str(item_type or _item_type_default(component, part_name)).strip(),
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
    pn = _norm(part_name)
    sn = _norm(serial_number)
    candidates = df[df["Part Name"].astype(str).str.strip().str.lower().eq(pn)].copy()
    if sn:
        cand_sn = candidates[candidates["Serial Number"].astype(str).str.strip().str.lower().eq(sn)]
        if not cand_sn.empty:
            candidates = cand_sn
    if candidates.empty:
        return False
    # Prefer consumable rows when both tool/part variants exist for same name.
    non_tool = candidates[~candidates["Item Type"].astype(str).str.strip().str.lower().eq("tool")]
    if not non_tool.empty:
        idx = int(non_tool.index[0])
    else:
        idx = int(candidates.index[0])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Tools are managed in inventory but never consumed.
    if _is_non_consumable_item_type(df.at[idx, "Item Type"]):
        df.at[idx, "Last Updated"] = now
        save_inventory(path, df)
        return True
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
    item_type: str = "",
) -> None:
    if not str(part_name).strip():
        return
    qty_i = max(0.0, float(qty))
    df = load_inventory(path)
    idx = _find_row(df, part_name, serial_number)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changed = False
    if idx >= 0:
        if float(df.at[idx, "Quantity"]) != qty_i:
            df.at[idx, "Quantity"] = qty_i
            changed = True
        if component and not str(df.at[idx, "Component"]).strip():
            df.at[idx, "Component"] = str(component).strip()
            changed = True
        if location and not str(df.at[idx, "Location"]).strip():
            df.at[idx, "Location"] = str(location).strip()
            changed = True
        if location_serial and not str(df.at[idx, "Location Serial"]).strip():
            df.at[idx, "Location Serial"] = str(location_serial).strip()
            changed = True
        if notes:
            note_v = str(notes).strip()
            if str(df.at[idx, "Notes"]).strip() != note_v:
                df.at[idx, "Notes"] = note_v
                changed = True
        if item_type:
            type_v = str(item_type).strip()
            if str(df.at[idx, "Item Type"]).strip() != type_v:
                df.at[idx, "Item Type"] = type_v
                changed = True
        if changed:
            df.at[idx, "Last Updated"] = now
    else:
        row = {
            "Part Name": str(part_name).strip(),
            "Item Type": str(item_type or _item_type_default(component, part_name)).strip(),
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
        changed = True
    if changed:
        save_inventory(path, df)


def is_non_consumable_part(path: str, part_name: str, serial_number: str = "") -> bool:
    df = load_inventory(path)
    pn = _norm(part_name)
    if not pn:
        return False
    m = df[df["Part Name"].astype(str).str.strip().str.lower().eq(pn)].copy()
    sn = _norm(serial_number)
    if sn:
        m_sn = m[m["Serial Number"].astype(str).str.strip().str.lower().eq(sn)]
        if not m_sn.empty:
            m = m_sn
    if m.empty:
        return False
    return m["Item Type"].astype(str).str.strip().str.lower().eq("tool").any()


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
