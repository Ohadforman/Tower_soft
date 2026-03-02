def render_consumables_tab(P):
    # ==========================================================
    # Imports (local to tab)
    # ==========================================================
    import os, json, math
    from datetime import datetime, timedelta

    import pandas as pd
    import streamlit as st

    # ✅ Your project imports
    from app_io.paths import ensure_logs_dir, ensure_gas_reports_dir, ensure_dir

    ensure_logs_dir()
    ensure_gas_reports_dir()

    # ==========================================================
    # UI polish
    # ==========================================================
    st.markdown("""
    <style>
      .block-container { padding-top: 1.55rem; }

      .section-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        padding: 14px 14px;
        border-radius: 14px;
        margin-bottom: 14px;
      }

      .vessel {
        height: 120px; width: 34px; border: 1px solid rgba(255,255,255,0.22);
        margin: auto; position: relative; border-radius: 10px;
        background: rgba(255,255,255,0.04);
        overflow: hidden;
      }
      .vessel-fill { position: absolute; bottom: 0; width: 100%; background: rgba(76, 175, 80, 0.85); }

      .muted { opacity: 0.75; }
      .low-card { border: 1px solid rgba(255, 77, 77, 0.60) !important; background: rgba(255, 77, 77, 0.05) !important; }
      .low-num { color: rgba(255, 170, 170, 1.0); font-weight: 800; }
      code { font-size: 0.86rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🍃 Tower state — Consumables & Dies")

    # ==========================================================
    # Constants / Files
    # ==========================================================
    container_labels = ["A", "B", "C", "D"]

    LOW_STOCK_KG = 1.0
    WAREHOUSE_STOCK_FILE = P.coating_stock_json
    TOWER_TEMPS_CSV = P.tower_temps_csv
    TOWER_CONTAINERS_CSV = P.tower_containers_csv
    CONTAINER_SNAPSHOT_FILE = P.container_levels_prev_json
    CONTAINER_CFG_PATH = P.container_config_json

    # Ensure parent dirs exist for moved files.
    for pth in [WAREHOUSE_STOCK_FILE, TOWER_TEMPS_CSV, TOWER_CONTAINERS_CSV, CONTAINER_SNAPSHOT_FILE]:
        ensure_dir(os.path.dirname(pth) or ".")

    # ==========================================================
    # Helpers
    # ==========================================================
    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return float(default)
            if isinstance(x, str) and x.strip() == "":
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _read_json(path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default

    def _write_json(path, obj):
        ensure_dir(os.path.dirname(path) or ".")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=4)
        os.replace(tmp, path)

    def _read_one_row_csv(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_csv(path)
            if df.empty:
                return {}
            return df.iloc[-1].to_dict()
        except Exception:
            return {}

    def _write_one_row_csv(path: str, cols: list, data: dict):
        ensure_dir(os.path.dirname(path) or ".")
        row = {c: "" for c in cols}
        row.update(data or {})
        row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([row], columns=cols).to_csv(path, index=False)

    def _list_files_recursive(root: str, exts=(".csv",)):
        out = []
        try:
            for base, _, files in os.walk(root):
                for fn in files:
                    if fn.lower().endswith(exts):
                        out.append(os.path.join(base, fn))
        except Exception:
            pass
        out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return out

    # ==========================================================
    # Load coating types
    # ==========================================================
    try:
        with open(P.coating_config_json, "r") as config_file:
            config = json.load(config_file)
    except Exception:
        st.error(f"Missing coating config file: {P.coating_config_json}")
        st.stop()
    coatings = config.get("coatings", {})
    coating_types = list(coatings.keys())

    # ==========================================================
    # 1) TEMPS CSV (wide)
    # ==========================================================
    TEMP_COLS = [
        "updated_at",
        "die_holder_primary_c",
        "die_holder_secondary_c",
        "A_container_c", "A_pipe_c",
        "B_container_c", "B_pipe_c",
        "C_container_c", "C_pipe_c",
        "D_container_c", "D_pipe_c",
    ]
    TEMP_STATE_KEYS = {
        "die_holder_primary_c": "die_holder_primary_temp_state",
        "die_holder_secondary_c": "die_holder_secondary_temp_state",
        "A_container_c": "temp_A_container_state",
        "A_pipe_c": "temp_A_pipe_state",
        "B_container_c": "temp_B_container_state",
        "B_pipe_c": "temp_B_pipe_state",
        "C_container_c": "temp_C_container_state",
        "C_pipe_c": "temp_C_pipe_state",
        "D_container_c": "temp_D_container_state",
        "D_pipe_c": "temp_D_pipe_state",
    }

    wide_temps = _read_one_row_csv(TOWER_TEMPS_CSV)
    for col, skey in TEMP_STATE_KEYS.items():
        if skey not in st.session_state:
            if col in wide_temps and str(wide_temps.get(col, "")).strip() != "":
                st.session_state[skey] = float(_safe_float(wide_temps[col], 25.0))
            else:
                st.session_state[skey] = 25.0

    # ==========================================================
    # 2) CONTAINERS CSV (wide)
    # ==========================================================
    CONTAINER_COLS = [
        "updated_at",
        "A_level_kg", "A_type",
        "B_level_kg", "B_type",
        "C_level_kg", "C_type",
        "D_level_kg", "D_type",
    ]
    def _lvl_key(lab): return f"cont_level_{lab}"
    def _type_key(lab): return f"cont_type_{lab}"

    wide_cont = _read_one_row_csv(TOWER_CONTAINERS_CSV)

    legacy_cfg = _read_json(CONTAINER_CFG_PATH, {})
    if not isinstance(legacy_cfg, dict):
        legacy_cfg = {}

    for lab in container_labels:
        default_level = 0.0
        default_type = coating_types[0] if coating_types else ""

        lvl_col = f"{lab}_level_kg"
        typ_col = f"{lab}_type"

        if lvl_col in wide_cont and str(wide_cont.get(lvl_col, "")).strip() != "":
            default_level = _safe_float(wide_cont.get(lvl_col), default_level)
        else:
            if isinstance(legacy_cfg.get(lab, {}), dict):
                default_level = _safe_float(legacy_cfg.get(lab, {}).get("level", default_level), default_level)

        if typ_col in wide_cont and str(wide_cont.get(typ_col, "")).strip() != "":
            default_type = str(wide_cont.get(typ_col))
        else:
            if isinstance(legacy_cfg.get(lab, {}), dict):
                default_type = str(legacy_cfg.get(lab, {}).get("type", default_type))

        if coating_types and default_type not in coating_types:
            default_type = coating_types[0]

        st.session_state.setdefault(_lvl_key(lab), float(default_level))
        st.session_state.setdefault(_type_key(lab), default_type)

    # ==========================================================
    # Top toolbar: Refresh buttons
    # ==========================================================
    tL, tM, tR = st.columns([1.3, 1.3, 1])
    with tL:
        st.caption(f"Temps CSV: `{TOWER_TEMPS_CSV}`")
        refresh_temps = st.button("🔄 Refresh temps", use_container_width=True, key="refresh_temps_btn")
    with tM:
        st.caption(f"Containers CSV: `{TOWER_CONTAINERS_CSV}`")
        refresh_containers = st.button("🔄 Refresh containers", use_container_width=True, key="refresh_containers_btn")
    with tR:
        st.caption(" ")
        st.caption(" ")

    if refresh_temps:
        wide_temps = _read_one_row_csv(TOWER_TEMPS_CSV)
        for col, skey in TEMP_STATE_KEYS.items():
            if col in wide_temps and str(wide_temps.get(col, "")).strip() != "":
                st.session_state[skey] = float(_safe_float(wide_temps[col], st.session_state.get(skey, 25.0)))
        st.success("Temps reloaded from CSV.")
        st.rerun()

    if refresh_containers:
        wide_cont = _read_one_row_csv(TOWER_CONTAINERS_CSV)
        for lab in container_labels:
            lvl_col = f"{lab}_level_kg"
            typ_col = f"{lab}_type"
            if lvl_col in wide_cont and str(wide_cont.get(lvl_col, "")).strip() != "":
                st.session_state[_lvl_key(lab)] = float(_safe_float(wide_cont[lvl_col], st.session_state[_lvl_key(lab)]))
            if typ_col in wide_cont and str(wide_cont.get(typ_col, "")).strip() != "":
                t = str(wide_cont[typ_col])
                if coating_types and t not in coating_types:
                    t = coating_types[0]
                st.session_state[_type_key(lab)] = t
        st.success("Containers reloaded from CSV.")
        st.rerun()

    # ==========================================================
    # Warehouse stock (kg) - JSON
    # ==========================================================
    warehouse_stock = _read_json(WAREHOUSE_STOCK_FILE, {})
    if not isinstance(warehouse_stock, dict):
        warehouse_stock = {}
    for t in coating_types:
        warehouse_stock[t] = _safe_float(warehouse_stock.get(t, 0.0), 0.0)

    prev_snapshot = _read_json(CONTAINER_SNAPSHOT_FILE, {})
    if not isinstance(prev_snapshot, dict):
        prev_snapshot = {}
    for lab in container_labels:
        prev_snapshot.setdefault(
            lab,
            {"level": float(st.session_state[_lvl_key(lab)]), "type": str(st.session_state[_type_key(lab)])}
        )

    # ==========================================================
    # UI: Containers
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🧪 Coating Containers (A–D)")
    st.caption("Rule: level ↑ = refill (auto subtract from warehouse). level ↓ = consumption.")

    cols = st.columns(4)
    current_container_state = {}

    for col, lab in zip(cols, container_labels):
        with col:
            st.markdown(f"**Container {lab}**")

            lvl = st.slider(
                f"Fill Level {lab} (kg)",
                0.0, 4.0,
                float(st.session_state[_lvl_key(lab)]),
                0.1,
                key=_lvl_key(lab)
            )

            if coating_types:
                cur_t = st.session_state[_type_key(lab)]
                if cur_t not in coating_types:
                    cur_t = coating_types[0]
                idx_t = coating_types.index(cur_t)
                ctype = st.selectbox(
                    f"Coating Type {lab}",
                    options=coating_types,
                    index=idx_t,
                    key=_type_key(lab)
                )
            else:
                ctype = ""
                st.info("No coating types configured.")

            fill_height = int((float(lvl) / 4.0) * 100.0)
            st.markdown(
                f"""
                <div class="vessel"><div class="vessel-fill" style="height:{fill_height}%;"></div></div>
                <div style="text-align:center; margin-top:6px;"><b>{float(lvl):.2f} kg</b></div>
                """,
                unsafe_allow_html=True
            )

            current_container_state[lab] = {"level": float(lvl), "type": str(ctype)}

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # Apply refill delta to warehouse (snapshot-based)
    # ==========================================================
    refill_events = []
    for lab in container_labels:
        prev_level = _safe_float(prev_snapshot.get(lab, {}).get("level", 0.0), 0.0)
        cur_level = _safe_float(current_container_state[lab]["level"], 0.0)
        cur_type = str(current_container_state[lab]["type"])
        delta = cur_level - prev_level

        if delta > 1e-9 and cur_type in warehouse_stock:
            before = _safe_float(warehouse_stock.get(cur_type, 0.0), 0.0)
            after = max(0.0, before - float(delta))
            warehouse_stock[cur_type] = after
            refill_events.append((lab, cur_type, float(delta), before, after))

        prev_snapshot[lab] = {"level": float(cur_level), "type": cur_type}

    try:
        _write_json(WAREHOUSE_STOCK_FILE, warehouse_stock)
        _write_json(CONTAINER_SNAPSHOT_FILE, prev_snapshot)
    except Exception as e:
        st.error(f"Auto-save failed: {e}")

    if refill_events:
        with st.expander("🧾 Detected refills (auto)", expanded=False):
            for lab, ctype, delta, before, after in refill_events:
                st.write(
                    f"Container **{lab}** refilled **+{delta:.2f} kg** of **{ctype}** → "
                    f"Warehouse: {before:.2f} → {after:.2f} kg"
                )

    # ==========================================================
    # Auto-save containers to CSV
    # ==========================================================
    last_cont_saved = st.session_state.get("containers_last_saved_snapshot", {})
    cur_cont_snapshot = {lab: (current_container_state[lab]["level"], current_container_state[lab]["type"]) for lab in container_labels}

    if not last_cont_saved:
        st.session_state["containers_last_saved_snapshot"] = cur_cont_snapshot
        last_cont_saved = cur_cont_snapshot

    if cur_cont_snapshot != last_cont_saved:
        out = {}
        for lab in container_labels:
            out[f"{lab}_level_kg"] = float(current_container_state[lab]["level"])
            out[f"{lab}_type"] = str(current_container_state[lab]["type"])

        try:
            _write_one_row_csv(TOWER_CONTAINERS_CSV, CONTAINER_COLS, out)
            st.session_state["containers_last_saved_snapshot"] = cur_cont_snapshot

            legacy_out = {lab: {"level": float(current_container_state[lab]["level"]), "type": str(current_container_state[lab]["type"])} for lab in container_labels}
            _write_json(CONTAINER_CFG_PATH, legacy_out)
        except Exception as e:
            st.error(f"Failed to write containers CSV: {e}")

    # ==========================================================
    # Stock by type (warehouse + containers)
    # ==========================================================
    def _sum_containers_by_type(container_state: dict):
        sums = {t: 0.0 for t in coating_types}
        for lab in container_labels:
            t = container_state.get(lab, {}).get("type", "")
            lvl = _safe_float(container_state.get(lab, {}).get("level", 0.0), 0.0)
            if t in sums:
                sums[t] += float(lvl)
        return sums

    container_sums = _sum_containers_by_type(current_container_state)
    total_by_type = {
        t: _safe_float(warehouse_stock.get(t, 0.0), 0.0) + _safe_float(container_sums.get(t, 0.0), 0.0)
        for t in coating_types
    }

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🏷️ Coating Stock by Type (Auto)")
    st.caption("Computed from warehouse + container contents. Red when total < 1 kg.")

    if coating_types:
        max_total = max(total_by_type.values()) if total_by_type else 0.0
        display_max = max(40.0, math.ceil(max_total / 5.0) * 5.0) if max_total > 0 else 40.0

        rows = [coating_types[i:i + 4] for i in range(0, len(coating_types), 4)]
        for row in rows:
            cols = st.columns(len(row))
            for col, ctype in zip(cols, row):
                with col:
                    total_kg = float(total_by_type.get(ctype, 0.0))
                    ware_kg = float(warehouse_stock.get(ctype, 0.0))
                    cont_kg = float(container_sums.get(ctype, 0.0))
                    fill_height = int(min(100.0, (total_kg / display_max) * 100.0)) if display_max > 0 else 0
                    is_low = total_kg < LOW_STOCK_KG

                    st.markdown(f"**{ctype}**")
                    st.markdown(
                        f"""
                        <div class="{'vessel low-card' if is_low else 'vessel'}">
                          <div class="vessel-fill" style="height:{fill_height}%; {'background: rgba(255, 77, 77, 0.85);' if is_low else ''}"></div>
                        </div>
                        <div style="text-align:center; margin-top:6px;">
                          <div class="{ 'low-num' if is_low else '' }"><b>{total_kg:.2f} kg</b></div>
                          <div class="muted" style="font-size:0.85rem;">Warehouse {ware_kg:.2f} + Containers {cont_kg:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info("No coating types found in coating config.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # Warehouse editor
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📦 Warehouse Stock (Edit when new material arrives)")
    st.caption("Bulk stock not inside containers A–D.")

    if coating_types:
        edited = False
        rows = [coating_types[i:i + 3] for i in range(0, len(coating_types), 3)]
        for row in rows:
            cols = st.columns(len(row))
            for col, ctype in zip(cols, row):
                with col:
                    k = f"wh_{ctype}"
                    st.session_state.setdefault(k, float(warehouse_stock.get(ctype, 0.0)))
                    val = st.number_input(
                        f"{ctype} (kg)",
                        min_value=0.0,
                        step=0.1,
                        value=float(st.session_state[k]),
                        key=k
                    )
                    if abs(val - float(warehouse_stock.get(ctype, 0.0))) > 1e-9:
                        warehouse_stock[ctype] = float(val)
                        edited = True

        if edited:
            try:
                _write_json(WAREHOUSE_STOCK_FILE, warehouse_stock)
                st.success("Warehouse stock updated.")
            except Exception as e:
                st.error(f"Failed to save warehouse stock: {e}")
    else:
        st.info("No coating types configured.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # Temps UI
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🌡️ Temperatures (CSV-based)")

    for lab in container_labels:
        st.markdown(f"**Container {lab} temps**")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                f"Container temp {lab} (°C)",
                min_value=0.0,
                step=0.1,
                value=float(st.session_state[TEMP_STATE_KEYS[f"{lab}_container_c"]]),
                key=TEMP_STATE_KEYS[f"{lab}_container_c"]
            )
        with c2:
            st.number_input(
                f"Pipe temp {lab} (°C)",
                min_value=0.0,
                step=0.1,
                value=float(st.session_state[TEMP_STATE_KEYS[f"{lab}_pipe_c"]]),
                key=TEMP_STATE_KEYS[f"{lab}_pipe_c"]
            )

    st.markdown("---")
    st.subheader("🔥 Die Holder Heater (Global)")
    cH1, cH2 = st.columns(2)
    with cH1:
        st.number_input(
            "Primary die holder heater temp (°C)",
            min_value=0.0,
            step=0.1,
            value=float(st.session_state[TEMP_STATE_KEYS["die_holder_primary_c"]]),
            key=TEMP_STATE_KEYS["die_holder_primary_c"]
        )
    with cH2:
        st.number_input(
            "Secondary die holder heater temp (°C)",
            min_value=0.0,
            step=0.1,
            value=float(st.session_state[TEMP_STATE_KEYS["die_holder_secondary_c"]]),
            key=TEMP_STATE_KEYS["die_holder_secondary_c"]
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-save temps to CSV if changed
    last_temp_saved = st.session_state.get("temps_last_saved_snapshot", {})
    cur_temp_snapshot = {skey: float(st.session_state.get(skey, 0.0)) for skey in TEMP_STATE_KEYS.values()}

    if not last_temp_saved:
        st.session_state["temps_last_saved_snapshot"] = cur_temp_snapshot
        last_temp_saved = cur_temp_snapshot

    if cur_temp_snapshot != last_temp_saved:
        out = {
            "die_holder_primary_c": float(st.session_state[TEMP_STATE_KEYS["die_holder_primary_c"]]),
            "die_holder_secondary_c": float(st.session_state[TEMP_STATE_KEYS["die_holder_secondary_c"]]),
        }
        for lab in container_labels:
            out[f"{lab}_container_c"] = float(st.session_state[TEMP_STATE_KEYS[f"{lab}_container_c"]])
            out[f"{lab}_pipe_c"] = float(st.session_state[TEMP_STATE_KEYS[f"{lab}_pipe_c"]])

        try:
            _write_one_row_csv(TOWER_TEMPS_CSV, TEMP_COLS, out)
            st.session_state["temps_last_saved_snapshot"] = cur_temp_snapshot
        except Exception as e:
            st.error(f"Failed to write temps CSV: {e}")

    # ==========================================================
    # Dies system
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🔩 Dies System")

    DIES_CONFIG_PATH = P.dies_config_json

    default_cfg = {
        f"Station {i}": {
            "entry_die_um": 0.0,
            "primary_die_um": 0.0,
            "primary_on_tower": False,
            "secondary_on_tower": False
        } for i in range(1, 7)
    }

    if os.path.exists(DIES_CONFIG_PATH):
        with open(DIES_CONFIG_PATH, "r") as f:
            try:
                dies_cfg = json.load(f)
                if not isinstance(dies_cfg, dict) or len(dies_cfg) == 0:
                    dies_cfg = default_cfg
            except Exception:
                dies_cfg = default_cfg
    else:
        dies_cfg = default_cfg
        with open(DIES_CONFIG_PATH, "w") as f:
            json.dump(dies_cfg, f, indent=4)

    station_names = list(dies_cfg.keys())

    for name in station_names:
        safe_key = name.replace(" ", "_").replace("/", "_")
        st.session_state.setdefault(f"dies_entry_{safe_key}", float(dies_cfg.get(name, {}).get("entry_die_um", 0.0)))
        st.session_state.setdefault(f"dies_primary_{safe_key}", float(dies_cfg.get(name, {}).get("primary_die_um", 0.0)))
        st.session_state.setdefault(f"dies_primary_on_{safe_key}", bool(dies_cfg.get(name, {}).get("primary_on_tower", False)))
        st.session_state.setdefault(f"dies_secondary_on_{safe_key}", bool(dies_cfg.get(name, {}).get("secondary_on_tower", False)))

    rows = [station_names[i:i + 3] for i in range(0, len(station_names), 3)]
    updated_dies_cfg = {}

    for row in rows:
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            safe_key = name.replace(" ", "_").replace("/", "_")
            with col:
                st.markdown(f"### {name}")

                entry_um = st.number_input("Entry die (µm)", min_value=0.0, step=1.0, format="%.1f", key=f"dies_entry_{safe_key}")
                primary_um = st.number_input("Primary die (µm)", min_value=0.0, step=1.0, format="%.1f", key=f"dies_primary_{safe_key}")
                primary_on = st.checkbox("Primary on tower", key=f"dies_primary_on_{safe_key}")
                secondary_on = st.checkbox("Secondary on tower", key=f"dies_secondary_on_{safe_key}")

                updated_dies_cfg[name] = {
                    "entry_die_um": float(entry_um),
                    "primary_die_um": float(primary_um),
                    "primary_on_tower": bool(primary_on),
                    "secondary_on_tower": bool(secondary_on),
                }

                st.caption(f"Entry: **{entry_um:.1f} µm** | Primary: **{primary_um:.1f} µm**")

    try:
        with open(DIES_CONFIG_PATH, "w") as f:
            json.dump(updated_dies_cfg, f, indent=4)
        st.caption(f"Auto-saved to `{DIES_CONFIG_PATH}`")
    except Exception as e:
        st.error(f"Failed to save dies config: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # 🧯 GAS REPORTS (AUTO MONTHLY CSV) ✅ ALL MFCs = ARGON
    #   - NO JSON
    #   - NO daily/weekly
    #   - outputs one CSV: argon_monthly_report.csv
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🧯 Argon Report — Monthly (AUTO from logs)")
    st.caption("Auto-builds a single monthly CSV report from logs. All Furnace MFC1–4 Actual are summed as Argon.")

    GAS_DIR = P.gas_reports_dir
    LOGS_DIR = P.logs_dir
    ensure_dir(GAS_DIR)
    ensure_dir(LOGS_DIR)

    REPORT_CSV = os.path.join(GAS_DIR, "argon_monthly_report.csv")
    STATE_JSON = os.path.join(GAS_DIR, "_argon_monthly_state.json")

    TIME_COL = "Date/Time"
    MFC_ACTUAL_COLS = [
        "Furnace MFC1 Actual",
        "Furnace MFC2 Actual",
        "Furnace MFC3 Actual",
        "Furnace MFC4 Actual",
    ]


    def _read_json(path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default


    def _write_json(path, obj):
        ensure_dir(os.path.dirname(path) or ".")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=4)
        os.replace(tmp, path)


    def _parse_dt_series_date_time(col: pd.Series) -> pd.Series:
        # Handles your format: '19/11/2024 12:44:33772'
        s = col.astype(str)

        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt.notna().sum() >= max(10, int(0.6 * len(dt))):
            return dt

        out = []
        for v in s.tolist():
            try:
                parts = v.split()
                if len(parts) < 2:
                    out.append(pd.NaT);
                    continue
                dpart = parts[0]
                tpart = parts[1]
                tt = tpart.split(":")
                if len(tt) != 3:
                    out.append(pd.NaT);
                    continue
                hh = int(tt[0]);
                mm = int(tt[1])
                secms = tt[2].strip()  # e.g. "33772"
                ss = int(secms[:2]) if len(secms) >= 2 else 0
                ms_str = secms[2:] if len(secms) > 2 else ""
                ms = int(ms_str) if (ms_str.isdigit() and ms_str != "") else 0
                if ms >= 1000:
                    ms = int(ms_str[:3]) if len(ms_str) >= 3 else ms % 1000

                dd, mon, yy = dpart.split("/")
                out.append(datetime(int(yy), int(mon), int(dd), hh, mm, ss, int(ms) * 1000))
            except Exception:
                out.append(pd.NaT)

        return pd.to_datetime(pd.Series(out), errors="coerce")


    def _list_logs():
        out = []
        try:
            for base, _, files in os.walk(LOGS_DIR):
                for fn in files:
                    if fn.lower().endswith(".csv"):
                        out.append(os.path.join(base, fn))
        except Exception:
            pass
        return out


    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)


    # UI controls
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        dt_cap_s = st.number_input("dt cap (sec)", min_value=0.1, step=0.1, value=2.0, key="argon_monthly_dt_cap")
    with g2:
        last_days = st.number_input("Scan last N days (0=all)", min_value=0, step=1, value=365,
                                    key="argon_monthly_scan_days")
    with g3:
        force = st.button("♻️ Force rebuild", use_container_width=True, key="argon_monthly_force")

    state = _read_json(STATE_JSON, {"last_scan_mtime": 0.0, "last_run": ""})


    def _build_monthly_csv(force_rebuild: bool = False):
        logs = _list_logs()
        if not logs:
            return {"info": f"No logs found in `{LOGS_DIR}`"}

        if int(last_days) > 0:
            cutoff = datetime.now() - timedelta(days=int(last_days))
            keep = []
            for p in logs:
                try:
                    if datetime.fromtimestamp(os.path.getmtime(p)) >= cutoff:
                        keep.append(p)
                except Exception:
                    pass
            logs = keep

        newest_mtime = 0.0
        for p in logs:
            try:
                newest_mtime = max(newest_mtime, os.path.getmtime(p))
            except Exception:
                pass

        if (not force_rebuild) and newest_mtime <= float(state.get("last_scan_mtime", 0.0)):
            return {"info": "Up-to-date (no new logs detected)."}

        # accum by YYYY-MM
        accum = {}  # month -> dict

        for lp in logs:
            try:
                df = pd.read_csv(lp)
            except Exception:
                continue

            if TIME_COL not in df.columns:
                continue

            if not any(c in df.columns for c in MFC_ACTUAL_COLS):
                continue

            df["_t"] = _parse_dt_series_date_time(df[TIME_COL])
            df = df.dropna(subset=["_t"]).sort_values("_t")
            if len(df) < 3:
                continue

            # Argon flow = sum of MFCs (SLPM)
            flow = None
            for c in MFC_ACTUAL_COLS:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                    flow = s if flow is None else (flow + s)
            if flow is None:
                continue

            dt_s = df["_t"].diff().dt.total_seconds().fillna(0.0)
            dt_s = dt_s.clip(lower=0.0, upper=float(dt_cap_s))
            dt_min = dt_s / 60.0

            month_key = df["_t"].dt.strftime("%Y-%m")
            work = pd.DataFrame({"month": month_key, "flow": flow, "dt_min": dt_min})
            work["SL"] = work["flow"] * work["dt_min"]

            for m, g in work.groupby("month", dropna=True):
                if m not in accum:
                    accum[m] = {
                        "total_SL": 0.0,
                        "total_minutes": 0.0,
                        "min_slpm": None,
                        "max_slpm": None,
                        "sum_flow_weighted": 0.0,  # flow * minutes
                        "logs": set(),
                        "rows": 0,
                    }

                total_sl = float(g["SL"].sum())
                total_min = float(g["dt_min"].sum())

                accum[m]["total_SL"] += total_sl
                accum[m]["total_minutes"] += total_min
                accum[m]["sum_flow_weighted"] += float((g["flow"] * g["dt_min"]).sum())
                mn = float(g["flow"].min())
                mx = float(g["flow"].max())
                accum[m]["min_slpm"] = mn if accum[m]["min_slpm"] is None else min(accum[m]["min_slpm"], mn)
                accum[m]["max_slpm"] = mx if accum[m]["max_slpm"] is None else max(accum[m]["max_slpm"], mx)
                accum[m]["logs"].add(lp)
                accum[m]["rows"] += int(len(g))

        if not accum:
            return {"info": "No valid data found in scanned logs."}

        # Build CSV
        rows = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for m, a in accum.items():
            avg_slpm = (a["sum_flow_weighted"] / a["total_minutes"]) if a["total_minutes"] > 0 else 0.0
            rows.append({
                "month": m,
                "gas": "Argon",
                "total_standard_liters": float(a["total_SL"]),
                "total_minutes": float(a["total_minutes"]),
                "avg_slpm": float(avg_slpm),
                "min_slpm": float(a["min_slpm"] if a["min_slpm"] is not None else 0.0),
                "max_slpm": float(a["max_slpm"] if a["max_slpm"] is not None else 0.0),
                "logs_count": int(len(a["logs"])),
                "rows_used": int(a["rows"]),
                "updated_at": now_str
            })

        out_df = pd.DataFrame(rows).sort_values("month", ascending=False)
        out_df.to_csv(REPORT_CSV, index=False)

        state["last_scan_mtime"] = float(newest_mtime)
        state["last_run"] = now_str
        _write_json(STATE_JSON, state)

        return {"updated": True, "rows": len(out_df), "logs_scanned": len(logs)}


    # ✅ AUTO build on load
    info = _build_monthly_csv(force_rebuild=bool(force))

    if "info" in info:
        st.caption(info["info"])
    else:
        st.caption(f"AUTO updated ✅ | logs scanned: {info.get('logs_scanned', 0)} | months: {info.get('rows', 0)}")

    # View report
    if os.path.exists(REPORT_CSV):
        try:
            rep = pd.read_csv(REPORT_CSV)
            st.dataframe(rep, use_container_width=True, hide_index=True)
            with open(REPORT_CSV, "r") as f:
                st.download_button(
                    "⬇️ Download monthly CSV",
                    data=f.read(),
                    file_name=os.path.basename(REPORT_CSV),
                    mime="text/csv",
                    use_container_width=True,
                    key="argon_monthly_download"
                )
        except Exception as e:
            st.error(f"Failed to load report CSV: {e}")
    else:
        st.info("Monthly report CSV not created yet.")

    st.markdown("</div>", unsafe_allow_html=True)
