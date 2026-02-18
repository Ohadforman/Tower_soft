# helpers/inventory_io.py
from __future__ import annotations

import os
import pandas as pd
from typing import Tuple

from app_io.paths import P
from helpers.text_utils import safe_str, safe_int, now_str

# ---------------------------
# File paths (single source)
# ---------------------------
SAP_FILE = P.sap_rods_inventory_csv
PREFORMS_FILE = P.preform_inventory_csv


# ---------------------------
# Defaults / schemas
# ---------------------------
SAP_REQUIRED_COLS = [
    "Rod ID",              # or SAP code / internal code
    "Description",
    "Qty",                 # integer
    "Min Qty",             # integer
    "Location",
    "Updated At",
    "Notes",
]

PREFORMS_REQUIRED_COLS = [
    "Preform Number",      # primary key
    "Description",
    "Length (cm)",         # float-ish, stored as string ok
    "Geometry",            # Round / TIGER / Octagonal etc
    "Project",
    "Status",              # Available / In Use / Finished etc
    "Updated At",
    "Notes",
]


def _ensure_file(path: str, required_cols: list[str]) -> None:
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)

    if not os.path.exists(path):
        pd.DataFrame(columns=required_cols).to_csv(path, index=False)
        return

    # If exists, ensure required columns
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        # if corrupt/unreadable, recreate with required cols (safe fallback)
        pd.DataFrame(columns=required_cols).to_csv(path, index=False)
        return

    changed = False
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""
            changed = True

    if changed:
        df.to_csv(path, index=False)


# ---------------------------
# Public API
# ---------------------------
def ensure_sap_inventory_file() -> str:
    _ensure_file(SAP_FILE, SAP_REQUIRED_COLS)
    return SAP_FILE


def ensure_preforms_inventory_file() -> str:
    _ensure_file(PREFORMS_FILE, PREFORMS_REQUIRED_COLS)
    return PREFORMS_FILE


def load_sap_inventory() -> pd.DataFrame:
    ensure_sap_inventory_file()
    return pd.read_csv(SAP_FILE, keep_default_na=False)


def save_sap_inventory(df: pd.DataFrame) -> None:
    ensure_sap_inventory_file()
    df.to_csv(SAP_FILE, index=False)


def load_preforms_inventory() -> pd.DataFrame:
    ensure_preforms_inventory_file()
    return pd.read_csv(PREFORMS_FILE, keep_default_na=False)


def save_preforms_inventory(df: pd.DataFrame) -> None:
    ensure_preforms_inventory_file()
    df.to_csv(PREFORMS_FILE, index=False)


def decrement_sap_rod_qty(rod_id: str, amount: int = 1) -> Tuple[bool, str]:
    """
    Decrease Qty for a rod row. Creates the file if missing.
    Returns (ok, message).
    """
    rod_id = safe_str(rod_id)
    amount = safe_int(amount, 1)
    if not rod_id:
        return False, "No Rod ID provided."
    if amount <= 0:
        return False, "Amount must be > 0."

    df = load_sap_inventory()
    if df.empty:
        return False, "SAP inventory is empty."

    # match by normalized string
    m = df["Rod ID"].astype(str).str.strip().eq(rod_id)
    if not m.any():
        return False, f"Rod ID not found: {rod_id}"

    i = df.index[m][0]
    cur = safe_int(df.at[i, "Qty"], 0)
    new_qty = max(0, cur - amount)

    df.at[i, "Qty"] = int(new_qty)
    df.at[i, "Updated At"] = now_str()
    save_sap_inventory(df)

    return True, f"Rod {rod_id}: Qty {cur} → {new_qty} (−{amount})"


def add_preform_length(preform_number: str, length_cm: float, source_draw: str = "") -> Tuple[bool, str]:
    """
    Adds/updates length for a preform.
    If preform doesn't exist → creates it.
    """
    preform_number = safe_str(preform_number)
    if not preform_number:
        return False, "No Preform Number provided."

    try:
        length_cm = float(length_cm)
    except Exception:
        return False, "Invalid length (cm)."

    df = load_preforms_inventory()

    if df.empty:
        df = pd.DataFrame(columns=PREFORMS_REQUIRED_COLS)

    m = df["Preform Number"].astype(str).str.strip().eq(preform_number)
    if m.any():
        i = df.index[m][0]
        df.at[i, "Length (cm)"] = str(length_cm)
        df.at[i, "Updated At"] = now_str()
        if source_draw:
            notes = safe_str(df.at[i, "Notes"])
            add = f"[{now_str()}] Length update from {source_draw}: {length_cm} cm"
            df.at[i, "Notes"] = (notes + "\n" + add).strip() if notes else add
    else:
        row = {c: "" for c in PREFORMS_REQUIRED_COLS}
        row["Preform Number"] = preform_number
        row["Length (cm)"] = str(length_cm)
        row["Updated At"] = now_str()
        row["Notes"] = f"[{now_str()}] Created from {source_draw}" if source_draw else f"[{now_str()}] Created"
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    save_preforms_inventory(df)
    return True, f"Preform {preform_number} updated: Length (cm) = {length_cm}"