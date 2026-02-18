# helpers/dataset_param_parsers.py
import re
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd


# ==========================================================
# Basic dataset-csv helpers
# ==========================================================
def param_map(df_params: pd.DataFrame) -> dict:
    """
    Returns dict {Parameter Name -> Value} using LAST occurrence (newest wins).
    """
    if df_params is None or df_params.empty:
        return {}
    if "Parameter Name" not in df_params.columns or "Value" not in df_params.columns:
        return {}
    d = {}
    for _, r in df_params.iterrows():
        k = str(r.get("Parameter Name", "")).strip()
        if not k:
            continue
        d[k] = r.get("Value", "")
    return d


def _parse_steps(df_params: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Robust STEP parser from dataset csv.
    Accepts:
      - STEP 1 Action / STEP 1 Length
      - STEP 1 Length (km) / [km] / etc
      - T&M STEP 1 Action, TM STEP 1 Action
      - weird spacing / case
    Returns list[(action, length_km)] for SAVE/CUT
    """
    if df_params is None or df_params.empty:
        return []
    if "Parameter Name" not in df_params.columns or "Value" not in df_params.columns:
        return []

    names = df_params["Parameter Name"].astype(str).fillna("")
    vals = df_params["Value"]

    norm = (
        names.str.replace("\ufeff", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    steps: List[Tuple[str, float]] = []
    i = 1
    while True:
        pat_action = rf"(^|.*\b)(t&m\s+|tm\s+)?step\s*{i}\s*action(\b|$)"
        pat_len = rf"(^|.*\b)(t&m\s+|tm\s+)?step\s*{i}\s*length(\b|$)"

        mA = norm.str.contains(pat_action, regex=True, na=False)
        mL = norm.str.contains(pat_len, regex=True, na=False)

        if not mA.any() or not mL.any():
            break

        action = str(vals.loc[mA].iloc[-1]).strip().upper()
        length = pd.to_numeric(pd.Series([vals.loc[mL].iloc[-1]]), errors="coerce").iloc[0]

        if pd.notna(length) and float(length) > 0 and action in ("SAVE", "CUT"):
            steps.append((action, float(length)))
        i += 1

    return steps


def parse_step_plan_anyway(df_params: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Returns list[(action, length_km)].
    Tries multiple ways so we don't miss the STEP plan due to naming variants.
    """
    steps = _parse_steps(df_params)
    if steps:
        return steps

    d = param_map(df_params)
    if not d:
        return []

    steps2: List[Tuple[str, float]] = []
    i = 1
    while True:
        action_keys = [f"STEP {i} Action", f"T&M STEP {i} Action", f"TM STEP {i} Action"]
        length_keys = [
            f"STEP {i} Length", f"STEP {i} Length (km)", f"STEP {i} Length [km]",
            f"T&M STEP {i} Length", f"T&M STEP {i} Length (km)",
            f"TM STEP {i} Length", f"TM STEP {i} Length (km)"
        ]

        ak = next((k for k in action_keys if k in d), None)
        lk = next((k for k in length_keys if k in d), None)

        if not ak or not lk:
            ak = next((k for k in d.keys() if re.search(rf"\bstep\s*{i}\s*action\b", k, re.I)), None)
            lk = next((k for k in d.keys() if re.search(rf"\bstep\s*{i}\s*length\b", k, re.I)), None)
            if not ak or not lk:
                break

        action = str(d.get(ak, "")).strip().upper()
        length = pd.to_numeric(pd.Series([d.get(lk, "")]), errors="coerce").iloc[0]

        if pd.notna(length) and float(length) > 0 and action in ("SAVE", "CUT"):
            steps2.append((action, float(length)))

        i += 1

    return steps2


def _parse_zones_from_end(df_params: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Parses:
      Zone i Start (from end)
      Zone i End (from end)
    Returns list of dicts sorted by 'a' (from end coord)
    """
    d = param_map(df_params)
    zones: List[Dict[str, Any]] = []
    i = 1
    while True:
        ks = f"Zone {i} Start (from end)"
        ke = f"Zone {i} End (from end)"
        if ks not in d or ke not in d:
            break
        try:
            a = float(d[ks])
            b = float(d[ke])
        except Exception:
            a = None
            b = None
        if a is not None and b is not None:
            if b < a:
                a, b = b, a
            zones.append({"i": i, "a": a, "b": b, "len": (b - a)})
        i += 1
    zones.sort(key=lambda z: z["a"])
    return zones


def _parse_marked_zone_lengths(df_params: pd.DataFrame) -> List[float]:
    d = param_map(df_params)
    out: List[float] = []
    i = 1
    while True:
        k = f"Marked Zone {i} Length"
        if k not in d:
            break
        try:
            L = float(d[k])
        except Exception:
            L = None
        if L is not None and L > 0:
            out.append(L)
        i += 1
    return out


def _find_zone_avg_values(df_params: pd.DataFrame, wanted_cols: list) -> dict:
    """
    Looks for: 'Good Zone {i} Avg - {col}'
    Returns dict {col: avg across zones}
    """
    out: Dict[str, float] = {}
    if df_params is None or df_params.empty:
        return out
    if "Parameter Name" not in df_params.columns or "Value" not in df_params.columns:
        return out

    pnames = df_params["Parameter Name"].astype(str)
    for col in wanted_cols:
        mask = (
            pnames.str.contains(r"^Good Zone \d+ Avg - ", regex=True, na=False)
            & pnames.str.contains(re.escape(str(col)), na=False)
        )
        vals = pd.to_numeric(df_params.loc[mask, "Value"], errors="coerce").dropna()
        if not vals.empty:
            out[col] = float(vals.mean())
    return out


# ==========================================================
# Log helpers (length column + zone lengths)
# ==========================================================
def _find_length_col_for_km(df: pd.DataFrame) -> Optional[str]:
    """
    Find Fibre/Fiber Length column (km-ish).
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    cmap = {str(c).strip().lower(): c for c in cols}

    preferred = [
        "fibre length (km)", "fiber length (km)",
        "fibre length", "fiber length",
        "fibre_length", "fiber_length",
        "fibre length km", "fiber length km",
    ]
    for k in preferred:
        if k in cmap:
            return cmap[k]

    for c in cols:
        cl = str(c).strip().lower()
        if "length" in cl and ("fiber" in cl or "fibre" in cl):
            return c

    for c in cols:
        if "length" in str(c).strip().lower():
            return c

    return None


def zone_lengths_from_log_km(df_log: pd.DataFrame, x_axis: str, zones: list):
    """
    zones = [(start,end), ...] in x-axis domain
    Returns:
      zones_info = [{"i":1,"start":..,"end":..,"len_km":..}, ...]
      length_col = chosen length col or ""
    """
    Lcol = _find_length_col_for_km(df_log)
    if not Lcol:
        return [], ""

    zones_info = []
    for i, (start, end) in enumerate(zones, start=1):
        try:
            zdf = df_log[(df_log[x_axis] >= start) & (df_log[x_axis] <= end)]
        except Exception:
            zdf = pd.DataFrame()

        if zdf.empty:
            zones_info.append({"i": i, "start": start, "end": end, "len_km": None})
            continue

        L = pd.to_numeric(zdf[Lcol], errors="coerce").dropna()
        if L.empty:
            zones_info.append({"i": i, "start": start, "end": end, "len_km": None})
            continue

        dk = float(L.iloc[-1] - L.iloc[0])
        if dk < 0:
            dk = abs(dk)

        zones_info.append({"i": i, "start": start, "end": end, "len_km": dk})

    return zones_info, Lcol


# ==========================================================
# AUTO T&M plan from spool end (CORRECT)
# ==========================================================
def build_tm_instruction_rows_auto_from_good_zones(
    filtered_df: pd.DataFrame,
    x_axis: str,
    good_zones: list,
    length_col_name: Optional[str] = None,
    dataset_csv_name: str = "",
):
    """
    AUTO T&M PLAN computed FROM THE END OF THE SPOOL (0 = spool end):

    - Uses Fibre Length last value as "end of spool".
    - Converts each selected good zone to "km_from_end" coordinates.
    - Orders zones by spool-end order (closest to end first).
    - Generates steps: CUT gap -> SAVE zone -> CUT gap -> SAVE zone ... -> CUT remainder to start.

    CSV is written so an operator can understand EACH zone as a REAL fiber region:
      - Fibre Length Min/Max (km)
      - km_from_end Start/End (km)
      - Zone Length (km)

    Steps contain only what matters operationally:
      - SAVE steps reference "Good Zone i" + its Fibre Length Min/Max
      - CUT steps have no zone reference
    """
    rows = []

    # header (ONLY ONCE)
    rows.append({"Parameter Name": "—", "Value": "—", "Units": ""})
    rows.append({"Parameter Name": "CUT/SAVE Plan Source", "Value": str(dataset_csv_name), "Units": ""})
    rows.append({"Parameter Name": "AUTO Plan Mode", "Value": "CUT/SAVE from spool end (0 at log end)", "Units": ""})

    if filtered_df is None or filtered_df.empty:
        rows.append({"Parameter Name": "T&M Instructions", "Value": "Log is empty / not loaded.", "Units": ""})
        return rows

    if not good_zones:
        rows.append({"Parameter Name": "T&M Instructions", "Value": "No good zones selected.", "Units": ""})
        return rows

    if x_axis not in filtered_df.columns:
        rows.append({"Parameter Name": "T&M Instructions", "Value": f"X-axis '{x_axis}' missing in log.", "Units": ""})
        return rows

    # sort by x so "last length" is end-of-spool
    dfw = filtered_df.sort_values(by=x_axis).copy()

    length_col = length_col_name or _find_length_col_for_km(dfw)
    if not length_col or length_col not in dfw.columns:
        rows.append({"Parameter Name": "T&M Instructions", "Value": "Could not find Fibre Length column in log.", "Units": ""})
        rows.append({"Parameter Name": "T&M Fix", "Value": "Ensure log CSV has a numeric Fibre/Fiber Length column (km).", "Units": ""})
        return rows

    L_all = pd.to_numeric(dfw[length_col], errors="coerce").dropna()
    if L_all.empty:
        rows.append({"Parameter Name": "T&M Instructions", "Value": f"Length column '{length_col}' is not numeric.", "Units": ""})
        return rows

    L_end = float(L_all.iloc[-1])
    L_min_log = float(L_all.min())
    L_max_log = float(L_all.max())

    rows.append({"Parameter Name": "Zone Length Column (log)", "Value": str(length_col), "Units": ""})
    rows.append({"Parameter Name": "Fiber Length End (log end)", "Value": float(L_end), "Units": "km"})
    rows.append({"Parameter Name": "Fiber Length Min (log)", "Value": float(L_min_log), "Units": "km"})
    rows.append({"Parameter Name": "Fiber Length Max (log)", "Value": float(L_max_log), "Units": "km"})
    rows.append({"Parameter Name": "T&M Coordinate System", "Value": "km_from_end = (L_end - FibreLength)", "Units": ""})

    # ----------------------------------------------------------
    # Build zone blocks (each is a REAL fibre length region)
    # ----------------------------------------------------------
    zone_blocks = []

    for (zs, ze) in good_zones:
        try:
            zdf = dfw[(dfw[x_axis] >= zs) & (dfw[x_axis] <= ze)]
        except Exception:
            zdf = pd.DataFrame()

        if zdf.empty:
            continue

        Lz = pd.to_numeric(zdf[length_col], errors="coerce").dropna()
        if Lz.empty:
            continue

        # REAL fibre region:
        L_min = float(Lz.min())
        L_max = float(Lz.max())
        z_len = abs(L_max - L_min)

        # from-end region:
        # near end -> larger fibre length -> smaller km_from_end
        km0 = max(0.0, L_end - L_max)  # zone start from end (closest edge)
        km1 = max(0.0, L_end - L_min)  # zone end from end (far edge)

        zone_blocks.append({
            "L_min": L_min,
            "L_max": L_max,
            "len_km": z_len,
            "km0": min(km0, km1),
            "km1": max(km0, km1),
        })

    if not zone_blocks:
        rows.append({"Parameter Name": "T&M Instructions", "Value": "Could not compute zones from log.", "Units": ""})
        return rows

    # Order zones the way the operator sees them on the spool:
    # closest to end first => smallest km0 first
    zone_blocks.sort(key=lambda z: z["km0"])

    # ----------------------------------------------------------
    # Write a clean "Good Zones" block (no orig/marked confusion)
    # ----------------------------------------------------------
    rows.append({"Parameter Name": "Good Zones Order", "Value": "Ordered from spool end → toward start", "Units": ""})

    for i, z in enumerate(zone_blocks, start=1):
        rows.append({"Parameter Name": f"Good Zone {i} (km from end) Start", "Value": float(z["km0"]), "Units": "km"})
        rows.append({"Parameter Name": f"Good Zone {i} (km from end) End", "Value": float(z["km1"]), "Units": "km"})
        rows.append({"Parameter Name": f"Good Zone {i} Length", "Value": float(z["len_km"]), "Units": "km"})
        rows.append({"Parameter Name": f"Good Zone {i} Fibre Length Min", "Value": float(z["L_min"]), "Units": "km"})
        rows.append({"Parameter Name": f"Good Zone {i} Fibre Length Max", "Value": float(z["L_max"]), "Units": "km"})

    # ----------------------------------------------------------
    # Build CUT/SAVE steps from spool end
    # ----------------------------------------------------------
    tm_i = 1
    cur = 0.0  # km from end
    total_save = 0.0
    total_cut = 0.0

    def add_step(action: str, length_km: float, zone_idx: Optional[int] = None):
        nonlocal tm_i, total_save, total_cut

        length_km = float(max(0.0, length_km))
        if length_km <= 1e-9:
            return

        rows.append({"Parameter Name": f"T&M Step {tm_i} Action", "Value": action, "Units": ""})
        rows.append({"Parameter Name": f"T&M Step {tm_i} Length", "Value": float(length_km), "Units": "km"})

        # SAVE steps point to the Good Zone number AND repeat its fibre length range
        if action == "SAVE" and zone_idx is not None:
            z = zone_blocks[zone_idx - 1]
            rows.append({"Parameter Name": f"T&M Step {tm_i} Zone", "Value": f"Good Zone {zone_idx}", "Units": ""})
            rows.append({"Parameter Name": f"T&M Step {tm_i} Fibre Length Min", "Value": float(z["L_min"]), "Units": "km"})
            rows.append({"Parameter Name": f"T&M Step {tm_i} Fibre Length Max", "Value": float(z["L_max"]), "Units": "km"})

        if action == "SAVE":
            total_save += length_km
        else:
            total_cut += length_km

        tm_i += 1

    # Walk zones from end -> start
    for zone_idx, z in enumerate(zone_blocks, start=1):
        gap = float(z["km0"] - cur)
        if gap > 1e-9:
            add_step("CUT", gap)     # no zone
            cur += gap

        add_step("SAVE", float(z["len_km"]), zone_idx=zone_idx)
        cur = max(cur, float(z["km1"]))

    # Final CUT remainder down to fibre start (km_from_end = L_end)
    if cur < L_end - 1e-9:
        add_step("CUT", float(L_end - cur))

    rows.append({"Parameter Name": "Total Saved Length", "Value": float(total_save), "Units": "km"})
    rows.append({"Parameter Name": "Total Cut Length", "Value": float(total_cut), "Units": "km"})
    return rows


__all__ = [
    "param_map",
    "_parse_steps",
    "parse_step_plan_anyway",
    "_parse_zones_from_end",
    "_parse_marked_zone_lengths",
    "_find_zone_avg_values",
    "zone_lengths_from_log_km",
    "build_tm_instruction_rows_auto_from_good_zones",
]