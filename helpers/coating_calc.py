# helpers/coating_calc.py
from __future__ import annotations

import math
from typing import List, Optional, Tuple

# ---------------------------------------------------------
# Viscosity imports (make these robust)
# ---------------------------------------------------------
try:
    from vis_ds2015 import get_viscosityDS2015, get_viscosityDP1032
except Exception:
    from helpers.vis_ds2015 import get_viscosityDS2015, get_viscosityDP1032


# ---------------------------------------------------------
# Basics
# ---------------------------------------------------------
def to_float(v, default=0.0) -> float:
    try:
        if v is None:
            return float(default)
        s = str(v).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _append_dict_rows(rows, data: dict, units_map: dict = None):
    if not data:
        return
    units_map = units_map or {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        rows.append({"Parameter Name": str(k), "Value": v, "Units": units_map.get(k, "")})


# ---------------------------------------------------------
# Core coating model (returns coated DIAMETER in µm)
# ---------------------------------------------------------
def calculate_coating_thickness_diameter_um(
    entry_fiber_diameter_um: float,
    die_diameter_um: float,
    mu_kg_m_s: float,
    rho_kg_m3: float,
    neck_length_m: float,
    pulling_speed_m_s: float,
    g_m_s2: float = 9.80665,
) -> float:
    """
    Your exact model, returns COATED DIAMETER in µm.
    """
    entry_fiber_diameter_um = to_float(entry_fiber_diameter_um, 0.0)
    die_diameter_um = to_float(die_diameter_um, 0.0)
    mu = to_float(mu_kg_m_s, 1.0)
    rho = to_float(rho_kg_m3, 1000.0)
    L = to_float(neck_length_m, 0.01)
    V = to_float(pulling_speed_m_s, 0.917)
    g = to_float(g_m_s2, 9.80665)

    if entry_fiber_diameter_um <= 0 or die_diameter_um <= 0 or mu <= 0 or rho <= 0 or L <= 0 or V <= 0:
        return float(entry_fiber_diameter_um)

    R = (die_diameter_um / 2.0) * 1e-6  # die radius (m)
    r = (entry_fiber_diameter_um / 2.0) * 1e-6  # fiber radius (m)

    if r <= 0 or R <= 0 or r >= R:
        return float(entry_fiber_diameter_um)

    k = r / R
    ln_k = math.log(k)  # k<1 => ln(k)<0 ok

    delta_P = L * rho * g
    Phi = (delta_P * (R**2)) / (8.0 * mu * L * V)

    term1 = Phi * (1.0 - k**4 + ((1.0 - k**2) ** 2) / ln_k)
    term2 = -(k**2 + (1.0 - k**2) / (2.0 * ln_k))

    inside = term1 + term2 + k**2
    if inside <= 0:
        return float(entry_fiber_diameter_um)

    t = R * (math.sqrt(inside) - k)  # meters
    coated_um = entry_fiber_diameter_um + (t * 2.0 * 1e6)
    return float(coated_um)


def _match_key_case_insensitive(name: str, keys: List[str]) -> Optional[str]:
    n = str(name or "").strip()
    if not n:
        return None
    low = n.lower()

    for k in keys:
        if str(k).strip().lower() == low:
            return k

    for k in keys:
        kl = str(k).strip().lower()
        if low in kl or kl in low:
            return k

    return None


def _get_viscosity_for_coating(coating_name: str, temp_c: float) -> float:
    name = str(coating_name or "").strip().lower()
    if "dp1032" in name or "dp-1032" in name:
        return float(get_viscosityDP1032(float(temp_c)))
    # default DS family
    return float(get_viscosityDS2015(float(temp_c)))


# ---------------------------------------------------------
# FC/SC prediction for chosen stock dies
# ---------------------------------------------------------
def coating_predict_fc_sc_um(
    entry_um: float,
    primary_die_name: str,
    secondary_die_name: str,
    primary_coating_name: str,
    secondary_coating_name: str,
    t1_c: float,
    t2_c: float,
    pulling_speed_m_s: float,
    g_m_s2: float,
    coating_cfg: dict,
) -> Tuple[float, float, dict]:
    dies = (coating_cfg or {}).get("dies", {})
    coats = (coating_cfg or {}).get("coatings", {})

    if primary_die_name not in dies or secondary_die_name not in dies:
        raise ValueError(f"Die not found in config_coating.json: {primary_die_name} / {secondary_die_name}")

    if primary_coating_name not in coats or secondary_coating_name not in coats:
        raise ValueError(f"Coating not found in config_coating.json: {primary_coating_name} / {secondary_coating_name}")

    pd = dies[primary_die_name]
    sd = dies[secondary_die_name]

    p_die_um = to_float(pd.get("Die_Diameter", 0.0), 0.0)
    s_die_um = to_float(sd.get("Die_Diameter", 0.0), 0.0)
    p_L_m = to_float(pd.get("Neck_Length", 0.01), 0.01)
    s_L_m = to_float(sd.get("Neck_Length", 0.01), 0.01)

    p_rho = to_float(coats[primary_coating_name].get("Density", 1000.0), 1000.0)
    s_rho = to_float(coats[secondary_coating_name].get("Density", 1000.0), 1000.0)

    mu1 = _get_viscosity_for_coating(primary_coating_name, t1_c)
    mu2 = _get_viscosity_for_coating(secondary_coating_name, t2_c)

    fc_um = calculate_coating_thickness_diameter_um(
        entry_fiber_diameter_um=entry_um,
        die_diameter_um=p_die_um,
        mu_kg_m_s=mu1,
        rho_kg_m3=p_rho,
        neck_length_m=p_L_m,
        pulling_speed_m_s=pulling_speed_m_s,
        g_m_s2=g_m_s2,
    )

    sc_um = calculate_coating_thickness_diameter_um(
        entry_fiber_diameter_um=fc_um,
        die_diameter_um=s_die_um,
        mu_kg_m_s=mu2,
        rho_kg_m3=s_rho,
        neck_length_m=s_L_m,
        pulling_speed_m_s=pulling_speed_m_s,
        g_m_s2=g_m_s2,
    )

    aux = {
        "mu1": mu1,
        "mu2": mu2,
        "rho1": p_rho,
        "rho2": s_rho,
        "p_die_um": p_die_um,
        "s_die_um": s_die_um,
        "p_L_m": p_L_m,
        "s_L_m": s_L_m,
        "V_m_s": pulling_speed_m_s,
        "g": g_m_s2,
    }
    return float(fc_um), float(sc_um), aux


# ---------------------------------------------------------
# Auto die selection (brute force over stock dies)
# ---------------------------------------------------------
def auto_select_dies_from_coating_calc(
    entry_um,
    tgt1_um,
    tgt2_um,
    coat1,
    coat2,
    t1_c,
    t2_c,
    config,
    draw_speed_m_min=None,
    pulling_speed_m_s=None,
    g_m_s2=9.80665,
    return_top_n=0,
):
    def _norm_key(s):
        s = str(s or "").strip().upper()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        return s

    def _match_key(name, keys):
        n = _norm_key(name)
        if not n:
            return None
        for k in keys:
            if _norm_key(k) == n:
                return k
        for k in keys:
            kk = _norm_key(k)
            if n in kk or kk in n:
                return k
        return None

    entry_um = to_float(entry_um, 0.0)
    tgt1_um = to_float(tgt1_um, 0.0)
    tgt2_um = to_float(tgt2_um, 0.0)
    t1_c = to_float(t1_c, 25.0)
    t2_c = to_float(t2_c, 25.0)

    # speed
    V = None
    if draw_speed_m_min is not None:
        vmin = to_float(draw_speed_m_min, 0.0)
        if vmin > 0:
            V = vmin / 60.0
    if V is None and pulling_speed_m_s is not None:
        vs = to_float(pulling_speed_m_s, 0.0)
        if vs > 0:
            V = vs
    if V is None:
        V = 0.917

    g = to_float(g_m_s2, 9.80665)

    dies = (config or {}).get("dies", {}) or {}
    coats = (config or {}).get("coatings", {}) or {}
    die_names = list(dies.keys())
    if not die_names:
        raise ValueError("config_coating.json has no dies.")

    c1 = _match_key(coat1, list(coats.keys()))
    c2 = _match_key(coat2, list(coats.keys()))
    if not c1:
        raise ValueError(f"Primary coating '{coat1}' not found in config_coating.json coatings.")
    if not c2:
        raise ValueError(f"Secondary coating '{coat2}' not found in config_coating.json coatings.")

    def _viscosity(coat_key, temp_c):
        nk = _norm_key(coat_key)
        if "DP1032" in nk:
            return float(get_viscosityDP1032(float(temp_c)))
        return float(get_viscosityDS2015(float(temp_c)))

    mu1 = _viscosity(c1, t1_c)
    mu2 = _viscosity(c2, t2_c)

    rho1 = to_float(coats[c1].get("Density", 1000.0), 1000.0)
    rho2 = to_float(coats[c2].get("Density", 1000.0), 1000.0)

    w1 = 1.0 if tgt1_um > 0 else 0.0
    w2 = 1.0 if tgt2_um > 0 else 0.0

    if w1 == 0.0 and w2 == 0.0:
        pdie = die_names[0]
        sdie = die_names[0]
        p = dies[pdie]
        s = dies[sdie]

        fc = calculate_coating_thickness_diameter_um(
            entry_fiber_diameter_um=entry_um,
            die_diameter_um=to_float(p.get("Die_Diameter", 0.0), 0.0),
            mu_kg_m_s=mu1,
            rho_kg_m3=rho1,
            neck_length_m=to_float(p.get("Neck_Length", 0.01), 0.01),
            pulling_speed_m_s=V,
            g_m_s2=g,
        )
        sc = calculate_coating_thickness_diameter_um(
            entry_fiber_diameter_um=fc,
            die_diameter_um=to_float(s.get("Die_Diameter", 0.0), 0.0),
            mu_kg_m_s=mu2,
            rho_kg_m3=rho2,
            neck_length_m=to_float(s.get("Neck_Length", 0.01), 0.01),
            pulling_speed_m_s=V,
            g_m_s2=g,
        )

        details = (
            "🧮 **Auto coating calculator (viscosity model)**\n"
            "- No targets provided → using first die pair.\n"
            f"- Speed: V={V:.3f} m/s\n"
            f"- Predicted FC={fc:.2f} µm, SC={sc:.2f} µm\n"
        )
        if return_top_n and return_top_n > 0:
            return pdie, sdie, float(fc), float(sc), details, []
        return pdie, sdie, float(fc), float(sc), details

    best = None
    best_score = 1e99
    best_fc = None
    best_sc = None
    best_aux = None
    ranked = []

    for pdie in die_names:
        pd = dies.get(pdie, {}) or {}
        p_die_um = to_float(pd.get("Die_Diameter", 0.0), 0.0)
        p_L = to_float(pd.get("Neck_Length", 0.01), 0.01)
        if p_die_um <= 0 or p_L <= 0:
            continue

        fc = calculate_coating_thickness_diameter_um(
            entry_fiber_diameter_um=entry_um,
            die_diameter_um=p_die_um,
            mu_kg_m_s=mu1,
            rho_kg_m3=rho1,
            neck_length_m=p_L,
            pulling_speed_m_s=V,
            g_m_s2=g,
        )

        for sdie in die_names:
            sd = dies.get(sdie, {}) or {}
            s_die_um = to_float(sd.get("Die_Diameter", 0.0), 0.0)
            s_L = to_float(sd.get("Neck_Length", 0.01), 0.01)
            if s_die_um <= 0 or s_L <= 0:
                continue

            sc = calculate_coating_thickness_diameter_um(
                entry_fiber_diameter_um=fc,
                die_diameter_um=s_die_um,
                mu_kg_m_s=mu2,
                rho_kg_m3=rho2,
                neck_length_m=s_L,
                pulling_speed_m_s=V,
                g_m_s2=g,
            )

            err_fc = abs(float(fc) - float(tgt1_um)) if w1 else 0.0
            err_sc = abs(float(sc) - float(tgt2_um)) if w2 else 0.0

            score = 0.0
            if w1:
                score += (float(fc) - float(tgt1_um)) ** 2
            if w2:
                score += (float(sc) - float(tgt2_um)) ** 2

            if return_top_n and return_top_n > 0:
                ranked.append(
                    {
                        "Primary Die": pdie,
                        "Secondary Die": sdie,
                        "Pred FC (µm)": float(fc),
                        "Pred SC (µm)": float(sc),
                        "Err FC (µm)": float(err_fc),
                        "Err SC (µm)": float(err_sc),
                        "Score": float(score),
                        "P Die D (µm)": float(p_die_um),
                        "S Die D (µm)": float(s_die_um),
                        "P L (m)": float(p_L),
                        "S L (m)": float(s_L),
                    }
                )

            if score < best_score:
                best_score = score
                best = (pdie, sdie)
                best_fc = float(fc)
                best_sc = float(sc)
                best_aux = (p_die_um, p_L, s_die_um, s_L)

    if not best:
        raise RuntimeError("Auto die search failed: no valid die pairs could be evaluated.")

    pdie, sdie = best
    p_die_um, p_L, s_die_um, s_L = best_aux

    thick1 = (best_fc - entry_um) / 2.0 if entry_um > 0 else None
    thick2 = (best_sc - best_fc) / 2.0 if best_fc is not None else None

    details = []
    details.append("🧮 **Auto coating calculator (viscosity model)**")
    details.append(f"- Entry fiber: **{entry_um:.2f} µm** | Targets: FC={tgt1_um:.2f} µm, SC={tgt2_um:.2f} µm")
    details.append(f"- Coatings: **{c1} @ {t1_c:.1f}°C**, **{c2} @ {t2_c:.1f}°C**")
    details.append(f"- Viscosities: μ1={mu1:.4f} kg/(m·s), μ2={mu2:.4f} kg/(m·s)")
    details.append(
        f"- Best dies: **{pdie}** (D={p_die_um:.1f}µm, L={p_L:.4f}m), **{sdie}** (D={s_die_um:.1f}µm, L={s_L:.4f}m)"
    )
    details.append(f"- Speed: V={V:.3f} m/s (from Draw Speed)")
    details.append(f"- ✅ Predicted FC: **{best_fc:.2f} µm**")
    details.append(f"- ✅ Predicted SC: **{best_sc:.2f} µm**")
    if thick1 is not None:
        details.append(f"- Primary thickness ≈ **{thick1:.2f} µm**")
    if thick2 is not None:
        details.append(f"- Secondary thickness ≈ **{thick2:.2f} µm**")

    details_str = "\n".join(details)

    if return_top_n and return_top_n > 0:
        ranked.sort(key=lambda r: r["Score"])
        ranked = ranked[: int(return_top_n)]
        return pdie, sdie, best_fc, best_sc, details_str, ranked

    return pdie, sdie, best_fc, best_sc, details_str


# ---------------------------------------------------------
# Ranking helpers (FIXED: tuple bug + wrong function call)
# ---------------------------------------------------------
def rank_dies_for_target(
    target_diameter_um,
    entry_diameter_um,
    dies_dict,  # config["dies"]
    mu,
    rho,
    L,
    V,
    g=9.80665,
    top_n=10,
):
    rows = []
    for die_name, dcfg in (dies_dict or {}).items():
        die_d = to_float(dcfg.get("Die_Diameter", 0.0), 0.0)
        Lm = to_float(dcfg.get("Neck_Length", L), L)

        pred = calculate_coated_diameter_um(
            entry_fiber_diameter_um=entry_diameter_um,
            die_diameter_um=die_d,
            mu=mu,
            rho=rho,
            L=Lm,
            V=V,
            g=g,
        )

        if not math.isfinite(pred):
            continue

        err = pred - target_diameter_um
        rows.append(
            {
                "die_name": die_name,
                "die_diameter_um": die_d,
                "neck_length_m": Lm,
                "predicted_um": pred,
                "error_um": err,
                "abs_error_um": abs(err),
            }
        )

    rows.sort(key=lambda r: r["abs_error_um"])
    return rows[:top_n], (rows[0] if rows else None)


def calculate_coated_diameter_um(entry_fiber_diameter_um, die_diameter_um, mu, rho, L, V, g=9.80665):
    """
    Wrapper around your model, returns predicted coated diameter [um] or NaN if invalid.
    """
    if die_diameter_um is None or entry_fiber_diameter_um is None:
        return float("nan")

    die_diameter_um = to_float(die_diameter_um, 0.0)
    entry_fiber_diameter_um = to_float(entry_fiber_diameter_um, 0.0)

    if die_diameter_um <= 0 or entry_fiber_diameter_um <= 0:
        return float("nan")

    if entry_fiber_diameter_um >= die_diameter_um:
        return float("nan")

    try:
        # ✅ FIX: call the correct function
        return calculate_coating_thickness_diameter_um(
            entry_fiber_diameter_um,
            die_diameter_um,
            mu,
            rho,
            L,
            V,
            g,
        )
    except Exception:
        return float("nan")