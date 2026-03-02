import os
import re
from datetime import datetime

import pandas as pd

from helpers.dates import compute_next_planned_draw_date
from helpers.match_utils import norm_str, alt_names


def get_most_recent_dataset_csv(dataset_dir: str):
    if not os.path.exists(dataset_dir):
        return None
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    if not files:
        return None
    return os.path.basename(max(files, key=os.path.getmtime))


def append_preform_length(preforms_file: str, preform_name: str, length_cm: float, source_draw: str):
    row = {
        "Preform Name": str(preform_name).strip(),
        "Length": float(length_cm),
        "Unit": "cm",
        "Updated Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source Draw": str(source_draw).strip(),
    }

    if os.path.exists(preforms_file):
        df = pd.read_csv(preforms_file)
    else:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(preforms_file, index=False)


def is_pm_draw_from_dataset_csv(df_params: pd.DataFrame) -> bool:
    """
    Detect PM draw from dataset CSV:
    Parameter Name == 'Is PM Draw' and Value in {1,true,yes,y}
    """
    if df_params is None or df_params.empty:
        return False
    if "Parameter Name" not in df_params.columns or "Value" not in df_params.columns:
        return False
    try:
        pn = df_params["Parameter Name"].astype(str).str.strip().str.lower()
        m = pn.isin(["is pm draw", "pm iris system"])
        if not m.any():
            return False
        val = df_params.loc[m, "Value"].iloc[0]
        if isinstance(val, (int, float)):
            return int(val) == 1
        val_str = str(val).strip().lower()
        return val_str in {"1", "true", "yes", "y"}
    except Exception:
        return False


def ensure_sap_inventory_file(sap_inventory_file: str) -> str:
    if not os.path.exists(sap_inventory_file):
        df = pd.DataFrame([{
            "Item Name": "SAP Rods Set",
            "Count": 0,
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Notes": "Auto-created",
        }])
        df.to_csv(sap_inventory_file, index=False)
    return sap_inventory_file


def decrement_sap_rods_set_by_one(sap_inventory_file: str, source_draw: str, when_str: str = None):
    ensure_sap_inventory_file(sap_inventory_file)
    when_str = when_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_draw = str(source_draw or "").strip()

    try:
        inv = pd.read_csv(sap_inventory_file, keep_default_na=False)
    except Exception as e:
        return False, f"Failed reading {sap_inventory_file}: {e}"

    inv.columns = [str(c).replace("\ufeff", "").strip() for c in inv.columns]
    for col in ["Item Name", "Count", "Last Updated", "Notes"]:
        if col not in inv.columns:
            inv[col] = ""

    name_norm = inv["Item Name"].astype(str).str.strip().str.lower()
    m = name_norm.eq("sap rods set")

    if not m.any():
        new_row = pd.DataFrame([{
            "Item Name": "SAP Rods Set",
            "Count": 0,
            "Last Updated": when_str,
            "Notes": "Auto-added row",
        }])
        inv = pd.concat([inv, new_row], ignore_index=True)
        m = inv["Item Name"].astype(str).str.strip().str.lower().eq("sap rods set")

    idx = inv[m].index[0]
    try:
        current = int(float(inv.loc[idx, "Count"]))
    except Exception:
        current = 0

    if current <= 0:
        inv.loc[idx, "Last Updated"] = when_str
        prev_notes = str(inv.loc[idx, "Notes"]).strip()
        add = f"[{when_str}] Attempted decrement for PM draw {source_draw} but Count was {current}."
        inv.loc[idx, "Notes"] = (prev_notes + "\n" + add).strip() if prev_notes else add
        inv.to_csv(sap_inventory_file, index=False)
        return False, f"SAP inventory NOT decremented (Count={current}). Please refill/update inventory."

    inv.loc[idx, "Count"] = current - 1
    inv.loc[idx, "Last Updated"] = when_str
    prev_notes = str(inv.loc[idx, "Notes"]).strip()
    add = f"[{when_str}] -1 set (PM draw {source_draw}). New Count={current-1}."
    inv.loc[idx, "Notes"] = (prev_notes + "\n" + add).strip() if prev_notes else add
    inv.to_csv(sap_inventory_file, index=False)
    return True, f"SAP Rods Set inventory updated: {current} -> {current-1}"


def mark_draw_order_failed_by_dataset_csv(
    orders_file: str,
    dataset_csv_filename: str,
    failed_desc: str,
    preform_len_after_cm: float,
):
    if not os.path.exists(orders_file):
        return False, f"{orders_file} not found (couldn't mark order failed)."

    try:
        orders = pd.read_csv(orders_file, keep_default_na=False)
    except Exception as e:
        return False, f"Failed reading {orders_file}: {e}"

    orders.columns = [str(c).replace("\ufeff", "").strip() for c in orders.columns]

    for col, default in {
        "Status": "Pending",
        "Active CSV": "",
        "Done CSV": "",
        "Done Description": "",
        "Done Timestamp": "",
        "Failed CSV": "",
        "Failed Description": "",
        "Failed Timestamp": "",
        "Preform Length After Draw (cm)": "",
        "Next Planned Draw Date": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }.items():
        if col not in orders.columns:
            orders[col] = default

    def _norm_col(series):
        return (
            series.astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
        )

    target = norm_str(dataset_csv_filename)
    target_alts = alt_names(target)

    active_norm = _norm_col(orders["Active CSV"])
    done_norm = _norm_col(orders["Done CSV"])

    match = pd.Series([False] * len(orders))
    for t in target_alts:
        match = match | (active_norm == t) | (done_norm == t)

    if not match.any():
        for t in target_alts:
            match = match | active_norm.str.endswith(t, na=False) | done_norm.str.endswith(t, na=False)

    if not match.any():
        for t in target_alts:
            contains = active_norm.str.contains(re.escape(t), na=False) | done_norm.str.contains(re.escape(t), na=False)
            if contains.sum() == 1:
                match = contains
                break

    if not match.any():
        sample_active = active_norm.dropna().unique()[:12].tolist()
        return False, (
            f"No matching row found in draw_orders.csv for '{dataset_csv_filename}' "
            f"(matched against Active CSV / Done CSV).\n"
            f"Sample Active CSV values: {sample_active}"
        )

    if match.sum() > 1:
        return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    orders.loc[match, "Status"] = "Failed"
    orders.loc[match, "Failed CSV"] = os.path.basename(dataset_csv_filename)
    orders.loc[match, "Failed Description"] = str(failed_desc).strip()
    orders.loc[match, "Failed Timestamp"] = now_str
    orders.loc[match, "Preform Length After Draw (cm)"] = float(preform_len_after_cm)
    orders.loc[match, "Status Updated At"] = now_str
    if "Assigned Dataset CSV" in orders.columns:
        cur = orders.loc[match, "Assigned Dataset CSV"].astype(str).iloc[0].strip()
        if cur == "" or cur.lower() == "nan":
            orders.loc[match, "Assigned Dataset CSV"] = os.path.basename(dataset_csv_filename)
    orders.to_csv(orders_file, index=False)
    return True, "Order marked as FAILED."


def reset_failed_order_to_beginning_and_schedule(
    orders_file: str,
    dataset_csv_filename: str,
    schedule_date: str = None,
    scheduled_status: str = "Scheduled",
):
    if not os.path.exists(orders_file):
        return False, f"{orders_file} not found."

    try:
        orders = pd.read_csv(orders_file, keep_default_na=False)
    except Exception as e:
        return False, f"Failed reading {orders_file}: {e}"

    orders.columns = [str(c).replace("\ufeff", "").strip() for c in orders.columns]

    for col, default in {
        "Status": "Pending",
        "Active CSV": "",
        "Done CSV": "",
        "Done Description": "",
        "Done Timestamp": "",
        "Failed CSV": "",
        "Failed Description": "",
        "Failed Timestamp": "",
        "Preform Length After Draw (cm)": "",
        "Next Planned Draw Date": "",
        "T&M Moved": False,
        "T&M Moved Timestamp": "",
    }.items():
        if col not in orders.columns:
            orders[col] = default

    def _norm_col(series):
        return (
            series.astype(str).fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.lower()
        )

    target = norm_str(dataset_csv_filename)
    target_alts = alt_names(target)

    active_norm = _norm_col(orders["Active CSV"])
    done_norm = _norm_col(orders["Done CSV"])
    fail_norm = _norm_col(orders["Failed CSV"]) if "Failed CSV" in orders.columns else pd.Series([""] * len(orders))

    match = pd.Series([False] * len(orders))
    for t in target_alts:
        match = match | (active_norm == t) | (done_norm == t) | (fail_norm == t)

    if not match.any():
        for t in target_alts:
            match = match | active_norm.str.endswith(t, na=False) | done_norm.str.endswith(t, na=False) | fail_norm.str.endswith(t, na=False)

    if not match.any():
        return False, f"No matching row found for '{dataset_csv_filename}'."
    if match.sum() > 1:
        return False, f"Multiple matching rows found for '{dataset_csv_filename}'. Please fix duplicates in draw_orders.csv."

    if schedule_date is None:
        schedule_date = compute_next_planned_draw_date(datetime.now())
    schedule_date = "" if schedule_date is None else str(schedule_date).strip()

    if schedule_date:
        orders.loc[match, "Status"] = scheduled_status
        orders.loc[match, "Next Planned Draw Date"] = schedule_date
    else:
        orders.loc[match, "Status"] = "Pending"
        orders.loc[match, "Next Planned Draw Date"] = ""

    orders.loc[match, "Active CSV"] = ""
    orders.loc[match, "Done CSV"] = ""
    orders.loc[match, "Done Description"] = ""
    orders.loc[match, "Done Timestamp"] = ""
    orders.loc[match, "Failed CSV"] = ""
    orders.loc[match, "Failed Description"] = ""
    orders.loc[match, "Failed Timestamp"] = ""
    orders.loc[match, "T&M Moved"] = False
    orders.loc[match, "T&M Moved Timestamp"] = ""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "Status Updated At" in orders.columns:
        orders.loc[match, "Status Updated At"] = now_str
    if "Last Reset Timestamp" not in orders.columns:
        orders["Last Reset Timestamp"] = ""
    orders.loc[match, "Last Reset Timestamp"] = now_str
    orders.to_csv(orders_file, index=False)

    if schedule_date:
        return True, f"Reset to {scheduled_status}. Next Planned Draw Date = {schedule_date}."
    return True, "Reset to Pending (no schedule)."
