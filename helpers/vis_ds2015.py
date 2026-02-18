import numpy as np
import scipy.optimize as opt

# Given data points (temperature in °C, viscosity in mPa·s)
temperature_extended = np.array([25, 30, 35, 40, 45, 50, 55])  # Temperature in °C
viscosity_mPa_s_extended = np.array([5900, 3500, 2040, 1200, 700, 400, 250])  # Viscosity in mPa·s

# Convert viscosity to kg/(m·s)
viscosity_kg_m_s_extended = viscosity_mPa_s_extended * 0.001

# Define an exponential function for fitting
def exp_function(T, a, b, c):
    return a * np.exp(b * T) + c

# Fit the curve using an exponential function
params_exp, _ = opt.curve_fit(exp_function, temperature_extended, viscosity_kg_m_s_extended, p0=[1, -0.1, 0])

# Create a function using the fitted parameters
def get_viscosityDS2015(temp_C):
    """Returns the viscosity in kg/(m·s) for a given temperature in °C using an exponential fit."""
    a, b, c = params_exp
    return exp_function(temp_C, a, b, c)

def get_viscosityDP1032(temp_C):
    """Returns the viscosity in kg/(m·s) for a given temperature in °C using an exponential fit."""
    a = 34.884
    b = -0.0806
    c = -0.162
    return a * np.exp(b * temp_C) + c

def get_viscosityOF136(temp_C):
    """Returns the viscosity in kg/(m·s) for a given temperature in °C using an exponential fit."""
    return 2.2

