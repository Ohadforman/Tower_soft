"""
Coating-die physics toolbox (Panoliaskos–Hallett–Garis, Applied Optics 1985)
============================================================================

Tower units (inputs/outputs):
  - Pressure: bar
  - Speed: m/min
  - Viscosity: mPa·s
  - Lengths/diameters: µm

Internal calculations use SI (Pa, m, m/s, Pa·s).

Key equations from the paper (ao-24-15-2309.pdf):
  - Eq.(14): Φ = ΔP R^2 / (8 μ L V)          (dimensionless)
  - Eq.(10): t/R = f(k, Φ, q)               (cured thickness ratio)

CRITICAL PRACTICAL NOTE (why predictions look too thin in real towers):
  - ΔP in Eq.(14) is the PRESSURE DROP ACROSS THE DIE LAND.
    Your regulator gauge may not equal that.
  - UV coatings can shear-thin strongly in small dies -> μ_eff < μ(T)
  - Applicator entrance / metering / clamp makes an "effective land" different from your L.

So we add:
  - Inverse solver: given desired OD, compute required Φ and required ΔP_bar
  - Calibration factor alpha_phi:
        Φ_effective = alpha_phi * Φ_model
    Fit alpha_phi from ONE measured stable point, then predict/control.

REGIMES:
  We classify regimes using normalized Λ* (operator/control concept):
    Λ  = ΔP/(μV) computed in SI (units 1/m)
    Λ* = Λ / Λ_ref  (dimensionless)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# 0) Unit conversions (tower units <-> SI)
# ============================================================

def bar_to_pa(p_bar: float) -> float:
    return p_bar * 1e5

def pa_to_bar(p_pa: float) -> float:
    return p_pa / 1e5

def mmin_to_ms(v_m_min: float) -> float:
    return v_m_min / 60.0

def mPas_to_Pas(mu_mPas: float) -> float:
    return mu_mPas * 1e-3

def um_to_m(x_um: float) -> float:
    return x_um * 1e-6

def m_to_um(x_m: float) -> float:
    return x_m * 1e6


# ============================================================
# 1) Viscosity models (temperature-only; exponential fit)
# ============================================================

@dataclass
class ViscosityModel:
    """
    Exponential fit: mu(T) = A * exp(B*T)
      - T in °C
      - mu in mPa·s
    With 1 point: constant viscosity.
    """
    points_TC_mu_mPas: List[Tuple[float, float]]

    def mu_mPas(self, T_C: float) -> float:
        pts = self.points_TC_mu_mPas
        if len(pts) == 0:
            raise ValueError("No viscosity points provided.")
        if len(pts) == 1:
            return float(pts[0][1])

        Ts = [p[0] for p in pts]
        mus = [p[1] for p in pts]
        ys = [math.log(m) for m in mus]

        n = len(Ts)
        xbar = sum(Ts) / n
        ybar = sum(ys) / n
        num = sum((x - xbar) * (y - ybar) for x, y in zip(Ts, ys))
        den = sum((x - xbar) ** 2 for x in Ts)
        if den == 0:
            return float(mus[0])

        B = num / den
        lnA = ybar - B * xbar
        A = math.exp(lnA)
        return A * math.exp(B * T_C)


# Put your real points here (add from your graph!)
DP1032_mu = ViscosityModel(points_TC_mu_mPas=[
    (25.0, 4465.0),
    # (35.0, 2100.0),
    # (45.0, 1100.0),
    # (55.0, 600.0),
])

DS2042_mu = ViscosityModel(points_TC_mu_mPas=[
    (25.0, 6002.0),
    (35.0, 2104.0),
])


# ============================================================
# 2) Geometry container
# ============================================================

@dataclass
class Die:
    """
    R_um   : die land radius [µm]  (NOT diameter)
    L_um   : die land length [µm]
    r_in_um: inner radius entering die [µm]
    """
    R_um: float
    L_um: float
    r_in_um: float

    @property
    def R_m(self) -> float:
        return um_to_m(self.R_um)

    @property
    def L_m(self) -> float:
        return um_to_m(self.L_um)

    @property
    def k(self) -> float:
        if self.R_um <= 0:
            raise ValueError("R_um must be > 0.")
        return self.r_in_um / self.R_um

    @property
    def gap_um(self) -> float:
        return self.R_um - self.r_in_um


# ============================================================
# 3) Core groups: Λ and Φ (paper Eq.14 defines Φ)
# ============================================================

def Lambda_pressure_drag(deltaP_bar: float, mu_mPas: float, V_m_min: float) -> float:
    """
    Λ = ΔP/(μV) computed in SI -> units 1/m (NOT dimensionless).
    """
    dP = bar_to_pa(deltaP_bar)
    mu = mPas_to_Pas(mu_mPas)
    V = mmin_to_ms(V_m_min)
    if mu <= 0 or V <= 0:
        raise ValueError("mu and V must be > 0.")
    return dP / (mu * V)

def Phi_eq14(deltaP_bar: float, die: Die, mu_mPas: float, V_m_min: float) -> float:
    """
    Eq.(14):
      Φ = ΔP R^2 / (8 μ L V)
    """
    dP = bar_to_pa(deltaP_bar)
    R = die.R_m
    L = die.L_m
    mu = mPas_to_Pas(mu_mPas)
    V = mmin_to_ms(V_m_min)
    if mu <= 0 or V <= 0 or L <= 0:
        raise ValueError("mu, V, and L must be > 0.")
    return (dP * R * R) / (8.0 * mu * L * V)

def deltaP_from_Phi_eq14(Phi: float, die: Die, mu_mPas: float, V_m_min: float) -> float:
    """
    Invert Eq.(14) to get ΔP (bar) from Φ.
      ΔP = Φ * 8 μ L V / R^2
    """
    R = die.R_m
    L = die.L_m
    mu = mPas_to_Pas(mu_mPas)
    V = mmin_to_ms(V_m_min)
    dP_pa = Phi * 8.0 * mu * L * V / (R * R)
    return pa_to_bar(dP_pa)


# ============================================================
# 4) Thickness prediction: Eq.(10)
# ============================================================

def thickness_ratio_eq10(k: float, Phi: float, q: float = 1.0) -> float:
    """
    Eq.(10) as in your screenshot.

    t/R = q Φ [ 1 - k^4 + (1 - k^2)^2 / ln(k) ]
          - q [ k^2 + (1 - k^2)/(2 ln(k)) ]
          + (k^2 Φ)^(1/2)
          - k

    Notes:
      - ln(k) is negative (k<1)
      - (k^2 Φ)^(1/2) = k*sqrt(Φ)
      - q is loading factor (cured solid / liquid volume), often ~1 as first pass.
    """
    if not (0.0 < k < 1.0):
        raise ValueError("k must be in (0,1).")
    if Phi < 0:
        raise ValueError("Phi must be >= 0.")

    ln_k = math.log(k)  # negative
    A = (1.0 - k**4 + ((1.0 - k**2)**2) / ln_k)
    B = (k**2 + (1.0 - k**2) / (2.0 * ln_k))

    return (q * Phi * A) - (q * B) + (k * math.sqrt(Phi)) - k

def thickness_um_eq10(deltaP_bar: float, die: Die, mu_mPas: float, V_m_min: float, q: float = 1.0,
                      alpha_phi: float = 1.0) -> float:
    """
    alpha_phi allows calibration:
      Φ_effective = alpha_phi * Φ_model
    """
    Phi_model = Phi_eq14(deltaP_bar, die, mu_mPas, V_m_min)
    Phi_eff = alpha_phi * Phi_model
    t_over_R = thickness_ratio_eq10(die.k, Phi_eff, q=q)
    t_m = t_over_R * die.R_m
    return m_to_um(t_m)

def predicted_OD_um_eq10(deltaP_bar: float, die: Die, mu_mPas: float, V_m_min: float, q: float = 1.0,
                         alpha_phi: float = 1.0) -> float:
    t_um = thickness_um_eq10(deltaP_bar, die, mu_mPas, V_m_min, q=q, alpha_phi=alpha_phi)
    return 2.0 * (die.r_in_um + t_um)


# ============================================================
# 5) Inverse solver: given desired OD -> required Φ and ΔP
# ============================================================

def solve_Phi_for_target_t_over_R(k: float, q: float, target_t_over_R: float,
                                  Phi_lo: float = 0.0, Phi_hi: float = 50.0,
                                  iters: int = 80) -> float:
    """
    Solve Eq.(10) for Φ given k, q, and target t/R.
    Uses bisection; assumes monotonic in practical region.

    If it can't bracket, it will expand Phi_hi.
    """
    def f(Phi: float) -> float:
        return thickness_ratio_eq10(k, Phi, q=q) - target_t_over_R

    lo = Phi_lo
    hi = Phi_hi
    flo = f(lo)
    fhi = f(hi)

    # Expand until bracket found or too big
    expand = 0
    while flo * fhi > 0 and expand < 20:
        hi *= 2.0
        fhi = f(hi)
        expand += 1

    if flo * fhi > 0:
        raise RuntimeError(
            "Could not bracket solution for Phi. "
            "Check k/q/target or increase Phi_hi."
        )

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)

def required_deltaP_bar_for_target_OD(
    target_OD_um: float,
    inner_OD_um: float,
    die_R_um: float,
    die_L_um: float,
    mu_mPas: float,
    V_m_min: float,
    q: float = 1.0
) -> tuple[float, float]:
    """
    Returns:
      (Phi_required, deltaP_required_bar)
    """
    if target_OD_um <= inner_OD_um:
        raise ValueError("target_OD_um must be > inner_OD_um.")
    t_um = 0.5 * (target_OD_um - inner_OD_um)  # radial thickness
    R_um = die_R_um
    target_t_over_R = t_um / R_um

    die = Die(R_um=die_R_um, L_um=die_L_um, r_in_um=inner_OD_um / 2.0)
    Phi_req = solve_Phi_for_target_t_over_R(die.k, q=q, target_t_over_R=target_t_over_R)
    dP_req_bar = deltaP_from_Phi_eq14(Phi_req, die, mu_mPas, V_m_min)
    return Phi_req, dP_req_bar

def fit_alpha_phi_from_one_measured_point(
    measured_OD_um: float,
    inner_OD_um: float,
    deltaP_bar: float,
    die_R_um: float,
    die_L_um: float,
    mu_mPas: float,
    V_m_min: float,
    q: float = 1.0
) -> float:
    """
    Fit alpha_phi so that Eq.(10) matches ONE measured OD at given ΔP.

    Procedure:
      - Convert measured OD to target t/R
      - Solve Eq.(10) for Φ_required (what physics needs)
      - Compute Φ_model from Eq.(14) using your assumed ΔP
      - alpha_phi = Φ_required / Φ_model

    Interpretation:
      alpha_phi > 1 means: "effective Φ is larger than model predicts"
      (could be higher true ΔP at land, lower μ_eff, or shorter effective L).
    """
    die = Die(R_um=die_R_um, L_um=die_L_um, r_in_um=inner_OD_um / 2.0)

    t_um = 0.5 * (measured_OD_um - inner_OD_um)
    target_t_over_R = t_um / die_R_um

    Phi_req = solve_Phi_for_target_t_over_R(die.k, q=q, target_t_over_R=target_t_over_R)
    Phi_model = Phi_eq14(deltaP_bar, die, mu_mPas, V_m_min)
    if Phi_model <= 0:
        raise ValueError("Phi_model <= 0; check inputs.")
    return Phi_req / Phi_model


# ============================================================
# 6) Regimes (normalized Λ*)
# ============================================================

@dataclass
class RegimeThresholds:
    drag_to_mixed: float = 0.6
    mixed_to_pressure: float = 1.4

@dataclass
class RegimeRefPoint:
    deltaP_bar: float
    mu_mPas: float
    V_m_min: float

def Lambda_star(deltaP_bar: float, mu_mPas: float, V_m_min: float, ref: RegimeRefPoint) -> float:
    lam = Lambda_pressure_drag(deltaP_bar, mu_mPas, V_m_min)
    lam_ref = Lambda_pressure_drag(ref.deltaP_bar, ref.mu_mPas, ref.V_m_min)
    if lam_ref == 0:
        raise ValueError("Invalid reference point (Λ_ref = 0).")
    return lam / lam_ref

def classify_regime(lam_star: float, thr: RegimeThresholds = RegimeThresholds()) -> str:
    if lam_star < thr.drag_to_mixed:
        return "Drag-dominated (stable / thin)"
    if lam_star < thr.mixed_to_pressure:
        return "Mixed regime (normal operation)"
    return "Pressure-dominated (accumulation / flooding risk)"


# ============================================================
# 7) Stage runner
# ============================================================

@dataclass
class StageResult:
    name: str
    mu_mPas: float
    Phi_model: float
    Phi_effective: float
    Lambda_SI: float
    Lambda_star: float
    regime: str
    k: float
    gap_um: float
    thickness_um: float
    OD_out_um: float


def run_stage(
    name: str,
    deltaP_bar: float,
    V_m_min: float,
    T_C: float,
    die_R_um: float,
    die_L_um: float,
    inner_OD_um: float,
    mu_model: ViscosityModel,
    q: float = 1.0,
    alpha_phi: float = 1.0,
    ref: Optional[RegimeRefPoint] = None,
    thr: RegimeThresholds = RegimeThresholds(),
) -> StageResult:

    die = Die(R_um=die_R_um, L_um=die_L_um, r_in_um=inner_OD_um / 2.0)

    mu = mu_model.mu_mPas(T_C)
    Phi_model = Phi_eq14(deltaP_bar, die, mu, V_m_min)
    Phi_eff = alpha_phi * Phi_model

    t_um = thickness_um_eq10(deltaP_bar, die, mu, V_m_min, q=q, alpha_phi=alpha_phi)
    OD_out = inner_OD_um + 2.0 * t_um

    lam = Lambda_pressure_drag(deltaP_bar, mu, V_m_min)
    if ref is None:
        lam_star = float("nan")
        reg = "Regime not classified (no reference provided)"
    else:
        lam_star = Lambda_star(deltaP_bar, mu, V_m_min, ref)
        reg = classify_regime(lam_star, thr)

    return StageResult(
        name=name,
        mu_mPas=mu,
        Phi_model=Phi_model,
        Phi_effective=Phi_eff,
        Lambda_SI=lam,
        Lambda_star=lam_star,
        regime=reg,
        k=die.k,
        gap_um=die.gap_um,
        thickness_um=t_um,
        OD_out_um=OD_out,
    )


def print_stage(s: StageResult) -> None:
    print(f"\n=== {s.name} ===")
    print(f"mu [mPa·s]            : {s.mu_mPas:.1f}")
    print(f"k = r_in/R [-]        : {s.k:.4f}")
    print(f"gap h [µm]            : {s.gap_um:.2f}")
    print(f"Φ_model (Eq.14) [-]   : {s.Phi_model:.3e}")
    print(f"Φ_effective [-]       : {s.Phi_effective:.3e}")
    print(f"Λ (SI) [1/m]          : {s.Lambda_SI:.3e}")
    if math.isnan(s.Lambda_star):
        print(f"Λ* (norm)             : n/a")
        print(f"Regime                : {s.regime}")
    else:
        print(f"Λ* (norm) [-]         : {s.Lambda_star:.3f}")
        print(f"Regime                : {s.regime}")
    print(f"Thickness t [µm]      : {s.thickness_um:.3f}")
    print(f"OD out [µm]           : {s.OD_out_um:.3f}")


# ============================================================
# 8) Example / HOW TO USE
# ============================================================

if __name__ == "__main__":
    # -------------------------
    # Your nominal inputs
    # -------------------------
    V = 20.0          # m/min
    glass_OD = 40.0   # µm

    # These must be LAND RADIUS and LAND LENGTH
    die1_R = 65.0     # µm (radius)
    die1_L = 200.0    # µm (land length)

    P1 = 0.30         # bar (your gauge setpoint)
    T1 = 28.0         # °C

    # -------------------------
    # 1) Raw prediction (usually too thin vs tower)
    # -------------------------
    s_raw = run_stage(
        name="Primary (RAW Eq.10)",
        deltaP_bar=P1,
        V_m_min=V,
        T_C=T1,
        die_R_um=die1_R,
        die_L_um=die1_L,
        inner_OD_um=glass_OD,
        mu_model=DP1032_mu,
        q=1.0,
        alpha_phi=1.0,
        ref=RegimeRefPoint(deltaP_bar=P1, mu_mPas=DP1032_mu.mu_mPas(T1), V_m_min=V),
    )
    print_stage(s_raw)

    # -------------------------
    # 2) Invert: what ΔP across land would be needed for a target OD?
    #    Example target primary OD = 80 µm (your tower expectation)
    # -------------------------
    mu_now = DP1032_mu.mu_mPas(T1)
    Phi_req, dP_req = required_deltaP_bar_for_target_OD(
        target_OD_um=110,
        inner_OD_um=glass_OD,
        die_R_um=die1_R,
        die_L_um=die1_L,
        mu_mPas=mu_now,
        V_m_min=V,
        q=1.0
    )
    print("\n--- Inverse design (Primary) ---")
    print(f"Target OD = 80 µm -> Φ_required ≈ {Phi_req:.3f}")
    print(f"That corresponds to ΔP_required across land ≈ {dP_req:.2f} bar")

    # -------------------------
    # 3) Calibrate alpha_phi from ONE measured stable point
    #    Put here your REAL measured OD for this same condition.
    # -------------------------
    measured_OD = 110  # <-- CHANGE THIS to your measured OD at P1,V,T,die
    alpha = fit_alpha_phi_from_one_measured_point(
        measured_OD_um=measured_OD,
        inner_OD_um=glass_OD,
        deltaP_bar=P1,
        die_R_um=die1_R,
        die_L_um=die1_L,
        mu_mPas=mu_now,
        V_m_min=V,
        q=1.0
    )
    print("\n--- Calibration ---")
    print(f"Measured OD {measured_OD:.1f} µm -> alpha_phi ≈ {alpha:.2f}")
    print("Interpretation: effective Φ is alpha_phi times the simple Eq.14 estimate.")

    # -------------------------
    # 4) Re-run prediction using alpha_phi (this should match your tower)
    # -------------------------
    s_cal = run_stage(
        name="Primary (CALIBRATED Eq.10)",
        deltaP_bar=P1,
        V_m_min=V,
        T_C=T1,
        die_R_um=die1_R,
        die_L_um=die1_L,
        inner_OD_um=glass_OD,
        mu_model=DP1032_mu,
        q=1.0,
        alpha_phi=alpha,
        ref=RegimeRefPoint(deltaP_bar=P1, mu_mPas=mu_now, V_m_min=V),
    )
    print_stage(s_cal)