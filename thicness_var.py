import numpy as np
import matplotlib.pyplot as plt


def compute_final_amplitude(A0, di, x):
    """Computes final coating thickness variation with a realistic growth rate."""
    return A0 * np.exp(di * x)


def empirical_initial_amplitude(Re, S, pin_ratio, R_ratio, L_ratio, h0):
    """Empirical correlation for initial amplitude based on paper."""
    n1, n2, n3, n4, n5, n6, C = 1.2, -0.5, -1.1, -0.7, -0.3, 0.8, 0.08  # Constants from paper
    return C * (Re ** n1) * (S ** n2) * (pin_ratio ** n3) * (R_ratio ** n4) * (L_ratio ** n5) * (h0 ** n6)


def compute_final_thickness(h0, Af):
    """Computes final coating thickness based on initial thickness and variation."""
    return max(h0 + Af, 0)  # Ensures no negative thickness values


def calculate_h0(P_in, P_out, V_f, mu, R_f, R_d, L):
    """Computes the initial coating thickness h_0 using the corrected Equation (1) from the paper."""
    k = R_f / R_d
    delta_p = P_in - P_out
    term1 = (1 - k ** 4) + ((1 - k ** 2) ** 2 / np.log(k))
    term2 = (1 - k ** 2) / (2 * np.log(k))
    h0_over_R2 = (delta_p * R_d ** 2) / (8 * mu * L * V_f) * (term1 ** 0.5 - term2) - k
    h0 = h0_over_R2 * R_d
    return max(0, min(h0, 20e-6))  # Clamped between 0 and 20 microns


def calculate_results(die_diameter, fiber_diameter, velocity_mpm, viscosity, density, surface_tension, pressure_bar,
                      die_length, tension_grams):
    """Calculates coating thickness based on user inputs with improved h0 calculation."""
    velocity = velocity_mpm / 60  # Convert m/min to m/s
    pressure = pressure_bar * 1e5  # Convert bar to Pa
    R_f = fiber_diameter / 2 / 1e6  # Convert microns to meters
    R_d = die_diameter / 2 / 1e6  # Convert microns to meters
    L = die_length / 1e6  # Convert microns to meters
    h0 = calculate_h0(pressure, 101325, velocity, viscosity, R_f, R_d, L)  # Compute h_0 using corrected Equation (1)

    Re = (density * velocity * R_f) / viscosity
    S = (density * surface_tension) / (viscosity ** 2)
    pin_ratio = pressure / 101325  # Normalize by atmospheric pressure
    R_ratio = die_diameter / fiber_diameter
    L_ratio = die_length / die_diameter
    di = 0.5  # Reduced to a more realistic value to prevent extreme growth
    x = 0.1  # Distance from die exit to cure oven (m)

    A0 = empirical_initial_amplitude(Re, S, pin_ratio, R_ratio, L_ratio, h0)
    Af = compute_final_amplitude(A0, di, x)
    h_final = compute_final_thickness(h0, Af)
    final_diameter = fiber_diameter + (2 * h_final * 1e6)  # Convert meters to microns

    return Re, A0, Af, h_final, final_diameter


def plot_final_diameter_vs_velocity(die_diameter, fiber_diameter, velocity_range_mpm, viscosity, density,
                                    surface_tension, pressure_bar, die_length, tension_grams, label):
    """Plots final fiber + coating diameter vs drawing velocity with realistic values."""
    final_diameter_values = []
    for velocity_mpm in velocity_range_mpm:
        _, _, _, _, final_diameter = calculate_results(die_diameter, fiber_diameter, velocity_mpm, viscosity, density,
                                                       surface_tension, pressure_bar, die_length, tension_grams)
        final_diameter_values.append(final_diameter)

    plt.plot(velocity_range_mpm, final_diameter_values, marker='o', label=label)


def main():
    """Example case with predefined parameters and fixed calculations."""
    example_die_diameter = 136  # Microns
    example_fiber_diameter = 80  # Microns
    example_velocity_range_mpm = np.linspace(10, 300, 10)
    example_viscosity = 0.1  # Pa.s
    example_density = 1100  # kg/m^3
    example_surface_tension = 0.028  # N/m
    example_pressure_bar = 1.5  # bar
    example_die_length = 2000  # Microns
    example_tension_grams = 50  # Grams

    plt.figure(figsize=(8, 5))
    plot_final_diameter_vs_velocity(example_die_diameter, example_fiber_diameter, example_velocity_range_mpm,
                                    example_viscosity, example_density, example_surface_tension, example_pressure_bar,
                                    example_die_length, example_tension_grams, label='Example 136μm Die')

    plt.xlabel('Drawing Velocity (m/min)')
    plt.ylabel('Final Fiber + Coating Diameter (microns)')
    plt.title('Final Diameter vs Drawing Velocity (Fixed)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
