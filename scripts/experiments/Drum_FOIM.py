import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Inputs
# ----------------------------
d_um = 900          # fiber diameter in µm
L_m  = 1000         # target length in meters (1 km)
ID_cm = 2.0         # inner drum diameter in cm
OD_max_cm = 8.0     # your constraint: can't exceed 8 cm OD

d = d_um * 1e-6           # m
ID = ID_cm / 100          # m
r_in = ID / 2             # m

# ----------------------------
# Core formulas
# ----------------------------
def required_width_cm(OD_cm, L=L_m, d=d, r_in=r_in):
    """W(OD) in cm."""
    OD = OD_cm / 100
    r_out = OD / 2
    # avoid division by zero / invalid region
    denom = np.pi * (r_out**2 - r_in**2)
    W = (L * d**2) / denom
    return W * 100  # to cm

def required_OD_cm(W_cm, L=L_m, d=d, r_in=r_in):
    """OD(W) in cm."""
    W = W_cm / 100
    r_out = np.sqrt(r_in**2 + (L * d**2) / (np.pi * W))
    OD = 2 * r_out
    return OD * 100  # to cm

# ----------------------------
# Plot: width vs outer diameter
# ----------------------------
OD_cm = np.linspace(ID_cm * 3.5, 8, 600)  # start slightly above ID
W_cm = required_width_cm(OD_cm)

plt.figure(figsize=(8,5))
plt.plot(OD_cm, W_cm)
#plt.yscale("log")  # huge dynamic range near ID
#plt.axvline(OD_max_cm, linestyle="--")
plt.xlabel("Outer drum diameter OD (cm)")
plt.ylabel("Required drum width W for 1 km (cm)")
plt.title(f"{d_um} µm fiber, ID={ID_cm} cm: width needed vs OD")
plt.tight_layout()
plt.show()

# ----------------------------
# "Optimization" under OD constraint:
# minimize width subject to OD <= OD_max
# -> best is simply OD = OD_max (use the largest allowed OD)
# ----------------------------
W_needed_at_ODmax = required_width_cm(OD_max_cm)
print(f"Required width at OD_max={OD_max_cm} cm: {W_needed_at_ODmax:.2f} cm")

# ----------------------------
# Optional: plot OD vs width (inverse view)
# ----------------------------
W_cm_grid = np.linspace(0.3, 30, 500)  # cm
OD_cm_grid = required_OD_cm(W_cm_grid)

plt.figure(figsize=(8,5))
plt.plot(W_cm_grid, OD_cm_grid)
plt.axhline(OD_max_cm, linestyle="--")
plt.xlabel("Drum width W (cm)")
plt.ylabel("Required outer diameter OD (cm)")
plt.title(f"{d_um} µm fiber, ID={ID_cm} cm: OD needed vs width")
plt.tight_layout()
plt.show()