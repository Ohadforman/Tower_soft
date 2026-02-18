import re
import numpy as np
import pandas as pd


def _get_float_param(df_params: pd.DataFrame, name: str, default=np.nan):
    m = df_params["Parameter Name"].astype(str).str.strip().eq(name)
    if not m.any():
        return default
    v = df_params.loc[m, "Value"].iloc[0]
    try:
        return float(str(v).strip())
    except Exception:
        return default


def parse_good_zones_from_dataset_csv(df_params: pd.DataFrame):
    """
    Returns zones as list of dicts:
      [{"i":1, "start_m":..., "end_m":...}, ...]
    Supports:
      - "Zone i Start (from end)" + "Zone i End (from end)"  (values in meters)
      - fallback: "Marked Zone i Length" sequential from end (values in meters)
    """
    if df_params is None or df_params.empty:
        return []

    # normalize
    p = df_params.copy()
    p["Parameter Name"] = p["Parameter Name"].astype(str).str.strip()
    p["Value"] = p["Value"].astype(str).str.strip()

    # total length (meters) — use what you have in your dataset CSV
    total_m = _get_float_param(p, "Fiber Total Length (Log End)", np.nan)
    if np.isnan(total_m):
        total_m = _get_float_param(p, "Total Saved Length", np.nan)

    zones = []

    # ---- 1) explicit start/end from end ----
    start_pat = re.compile(r"^Zone\s*(\d+)\s*Start", re.IGNORECASE)
    end_pat = re.compile(r"^Zone\s*(\d+)\s*End", re.IGNORECASE)

    starts = {}
    ends = {}

    for _, row in p.iterrows():
        nm = row["Parameter Name"]
        val = row["Value"]
        try:
            fv = float(val)
        except Exception:
            continue

        ms = start_pat.match(nm)
        me = end_pat.match(nm)
        if ms:
            i = int(ms.group(1))
            starts[i] = fv
        if me:
            i = int(me.group(1))
            ends[i] = fv

    if starts and ends and not np.isnan(total_m):
        for i in sorted(set(starts) & set(ends)):
            # values are "from end" -> convert to absolute from start
            s_from_end = starts[i]
            e_from_end = ends[i]
            s_abs = float(total_m - max(s_from_end, e_from_end))
            e_abs = float(total_m - min(s_from_end, e_from_end))
            zones.append({"i": i, "start_m": max(0.0, s_abs), "end_m": max(0.0, e_abs)})

        zones = [z for z in zones if z["end_m"] > z["start_m"]]
        return zones

    # ---- 2) fallback: sequential "Marked Zone i Length" from end ----
    len_pat = re.compile(r"^Marked\s*Zone\s*(\d+)\s*Length", re.IGNORECASE)
    lengths = []
    for _, row in p.iterrows():
        m = len_pat.match(row["Parameter Name"])
        if not m:
            continue
        i = int(m.group(1))
        try:
            L = float(row["Value"])
        except Exception:
            continue
        if L > 0:
            lengths.append((i, L))

    if lengths and not np.isnan(total_m):
        lengths = sorted(lengths, key=lambda x: x[0])
        cursor_from_end = 0.0
        for i, L in lengths:
            s_from_end = cursor_from_end + L
            e_from_end = cursor_from_end
            s_abs = float(total_m - s_from_end)
            e_abs = float(total_m - e_from_end)
            zones.append({"i": i, "start_m": max(0.0, s_abs), "end_m": max(0.0, e_abs)})
            cursor_from_end += L

        zones = [z for z in zones if z["end_m"] > z["start_m"]]
        return zones

    return []