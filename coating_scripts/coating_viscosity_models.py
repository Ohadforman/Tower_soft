"""
coating_viscosity_models.py

Viscosity(T) helper models for tower coating calculations + basic material properties.

We model viscosity in **mPa·s** as a function of coating temperature in **°C**.

Why VFT?
--------
UV-curable coatings often deviate from a simple Arrhenius law. A VFT (Vogel–Fulcher–Tammann)
model typically fits viscosity-vs-temperature curves better over the 20–60°C range.

VFT model (log space):
    ln(mu) = A + B / (T_K - T0)
where:
    mu  = viscosity [mPa·s]
    T_K = temperature [K] = T_C + 273.15
    A, B, T0 = fitted parameters (T0 in Kelvin)

Material properties:
-------------------
We also store density (kg/m^3) from datasheets (typically specified at 23°C).

How to add a new coating
------------------------
1) Collect viscosity datapoints from the datasheet (T_C, mu_mPa_s).
2) Fit VFT parameters (A, B, T0) using SciPy curve_fit (recommended).
3) Add a new entry to COATING_MODELS below with the fitted params + density.
4) (Optional) Add sanity points in __main__ to plot & print fit errors.

Tip: Fit in log-space (ln(mu)) to capture multiplicative errors properly.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) Central parameter store (YOUR fitted params + datasheet ρ)
# ============================================================
# DP-1032:
#   VFT: A=-1.74074, B=1215.42, T0=178.901 K
#   Density @23°C: 1.055 g/cm^3 = 1055 kg/m^3
#
# DS-2042:
#   VFT: A=-5.73793, B=1845.08, T0=170.898 K
#   Density @23°C: 1130 kg/m^3
COATING_MODELS = {
    "DP-1032": {"type": "VFT", "A": -1.74074, "B": 1215.42, "T0": 178.901, "rho_kg_m3": 1055.0},
    "DS-2042": {"type": "VFT", "A": -5.73793, "B": 1845.08, "T0": 170.898, "rho_kg_m3": 1130.0},
}


# ============================================================
# 2) Core model implementation
# ============================================================
def mu_vft_mpas(T_C: float | np.ndarray, A: float, B: float, T0: float) -> np.ndarray:
    """
    VFT viscosity model.
    Inputs:
        T_C : temperature in °C (scalar or array)
        A, B, T0 : VFT parameters (T0 in Kelvin)
    Output:
        viscosity in mPa·s (numpy array)
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15

    # Safety clamp: avoids division blow-up if someone calls it near/below T0.
    T_K = np.maximum(T_K, T0 + 1e-6)

    return np.exp(A + B / (T_K - T0))


def get_viscosity_mpas(coating_name: str, T_C: float | np.ndarray) -> np.ndarray:
    """
    Unified viscosity getter.
    Returns viscosity in mPa·s.
    """
    if coating_name not in COATING_MODELS:
        known = ", ".join(sorted(COATING_MODELS.keys()))
        raise KeyError(f"Unknown coating '{coating_name}'. Known: {known}")

    model = COATING_MODELS[coating_name]
    if model["type"] == "VFT":
        return mu_vft_mpas(T_C, model["A"], model["B"], model["T0"])

    raise ValueError(f"Unsupported model type '{model['type']}' for coating '{coating_name}'")


def get_viscosity_pa_s(coating_name: str, T_C: float | np.ndarray) -> np.ndarray:
    """
    Viscosity in Pa·s (SI).
    """
    return get_viscosity_mpas(coating_name, T_C) * 1e-3


def get_density_kg_m3(coating_name: str) -> float:
    """
    Density in kg/m^3 (typically datasheet value at 23°C).
    """
    if coating_name not in COATING_MODELS:
        known = ", ".join(sorted(COATING_MODELS.keys()))
        raise KeyError(f"Unknown coating '{coating_name}'. Known: {known}")

    rho = COATING_MODELS[coating_name].get("rho_kg_m3", None)
    if rho is None:
        raise ValueError(f"No density set for coating '{coating_name}'. Add rho_kg_m3 in COATING_MODELS.")
    return float(rho)


# ============================================================
# 3) Optional: quick visual validation (run file directly)
# ============================================================
if __name__ == "__main__":
    # datapoints used in fits (edit if you have exact datasheet tables)
    data_points = {
        "DP-1032": {
            "T_C": np.array([25, 35, 45, 55], dtype=float),
            "mu":  np.array([4700, 2100, 1100,  600], dtype=float),
        },
        "DS-2042": {
            "T_C": np.array([25, 35, 45, 55], dtype=float),
            "mu":  np.array([6400, 2200,  900,  400], dtype=float),
        },
    }

    T_plot = np.linspace(20, 60, 500)

    for name, dp in data_points.items():
        mu_plot = get_viscosity_mpas(name, T_plot)

        plt.figure()
        plt.plot(T_plot, mu_plot, label="VFT fit")
        plt.scatter(dp["T_C"], dp["mu"], label="Data")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Viscosity (mPa·s)")
        plt.title(f"{name} Viscosity vs Temperature (VFT)")
        plt.grid(True)
        plt.legend()
        plt.show()

        print(f"\n{name} — density = {get_density_kg_m3(name):.1f} kg/m³")
        print("T (°C) | Data (mPa·s) | Fit (mPa·s) | Error (mPa·s) | Error (%)")
        print("-" * 66)
        for T, mu_obs in zip(dp["T_C"], dp["mu"]):
            mu_fit = float(get_viscosity_mpas(name, T))
            err = mu_fit - mu_obs
            err_pct = 100.0 * err / mu_obs
            print(f"{T:5.0f} | {mu_obs:12.1f} | {mu_fit:11.1f} | {err:11.1f} | {err_pct:8.2f}")