import math
import json
from vis_ds2015 import get_viscosityDS2015, get_viscosityDP1032



def load_config():
    """Loads the configuration from the JSON file."""
    with open('config_coating.json') as config_file:
        return json.load(config_file)

def evaluate_viscosity(viscosity_formula, T):
    """Evaluates the viscosity formula safely."""
    allowed_names = {"math": math, "T": T}
    return eval(viscosity_formula, {"__builtins__": {}}, allowed_names)

def calculate_coating_thickness(entry_fiber_diameter, die_diameter, mu, rho, L, V, g):
    """Calculates coating thickness and coated fiber diameter."""
    R = (die_diameter / 2) * 10**-6  # Die Radius (m)
    r = (entry_fiber_diameter / 2) * 10**-6  # Fiber Radius (m)
    k = r / R
    ln_k = math.log(k)

    # Pressure drop calculation
    delta_P = L * rho * g

    # Φ calculation
    Phi = (delta_P * R**2) / (8 * mu * L * V)

    # Calculate the coating thickness (t)
    term1 = Phi * (1 - k**4 + ((1 - k**2)**2) / ln_k)
    term2 = - (k**2 + (1 - k**2) / (2 * ln_k))  # Ensure valid sqrt input
    t = R * ((term1 + term2 + k**2)**0.5 - k)

    coated_fiber_diameter = entry_fiber_diameter + (t * 2 * 1e6)  # Add thickness and convert to microns
    return coated_fiber_diameter  # Return in microns

# Load configuration
config = load_config()



# User inputs fiber diameter
Entry_Fiber_Diameter = float(input("Enter the fiber diameter in microns: "))

# User inputs temperature
Temperature_first = float(input("Enter first coat temperature in °C: "))
Temperature_second = float(input("Enter second coat temperature in °C: "))
# User selects die and coating for Primary and Secondary coating
print("Available Dies:", ", ".join(config["dies"].keys()))
Primary_Die = input("Select Primary Die: ")
Secondary_Die = input("Select Secondary Die: ")

print("Available Coatings:", ", ".join(config["coatings"].keys()))
Primary_Coating = input("Select Primary Coating: ")
Secondary_Coating = input("Select Secondary Coating: ")

# Load selected die and coating values
Primary_Die_Diameter = config["dies"][Primary_Die]["Die_Diameter"]
Primary_Die_Neck_Length = config["dies"][Primary_Die]["Neck_Length"]
Secondary_Die_Diameter = config["dies"][Secondary_Die]["Die_Diameter"]
Secondary_Die_Neck_Length = config["dies"][Secondary_Die]["Neck_Length"]

# Retrieve viscosities based on selected coating and user-provided temperature
Primary_Coating_Viscosity = get_viscosityDP1032(Temperature_first)
Secondary_Coating_Viscosity = get_viscosityDS2015(Temperature_second)

print(f"Primary Coating Viscosity at {Temperature_first}°C: {Primary_Coating_Viscosity:.4f} kg/(m·s)")
print(f"Secondary Coating Viscosity at {Temperature_second}°C: {Secondary_Coating_Viscosity:.4f} kg/(m·s)")

Primary_Coating_Density = config["coatings"][Primary_Coating]["Density"]
Secondary_Coating_Density = config["coatings"][Secondary_Coating]["Density"]

# Constants
V = 0.917  # Pulling speed (m/s)
g = 9.8  # Gravity (m/s²)

# Compute coating thickness for Primary and Secondary coatings
FC_diameter = calculate_coating_thickness(Entry_Fiber_Diameter, Primary_Die_Diameter,
                                          Primary_Coating_Viscosity, Primary_Coating_Density,
                                          Primary_Die_Neck_Length, V, g)

SC_diameter = calculate_coating_thickness(FC_diameter, Secondary_Die_Diameter,
                                          Secondary_Coating_Viscosity, Secondary_Coating_Density,
                                          Secondary_Die_Neck_Length, V, g)

# Display results
print(f"\nEntry fiber diameter: {round(Entry_Fiber_Diameter,2)}μm")
print(f"\nFirst coating diameter: {round(FC_diameter,2)}μm, using Die: {Primary_Die} and Coating: {Primary_Coating}")
print(f"\nSecondary coating diameter: {round(SC_diameter,2)}μm, using Die: {Secondary_Die} and Coating: {Secondary_Coating}")