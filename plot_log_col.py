import pandas as pd
import matplotlib.pyplot as plt

def plot_selected_sensors(csv_file):
    """
    Plots data for selected sensors using user-specified X and Y axes.

    :param csv_file: Path to the CSV file containing numeric data.
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Display available columns for X-axis selection
    print("Available columns for selection:", df.columns.tolist())
    x_axis_column = input("Enter the column name for the X-axis: ").strip()
    if x_axis_column not in df.columns:
        raise ValueError(f"Invalid X-axis column: {x_axis_column}")

    # Display available sensor names
    available_sensors = df.columns.tolist()  # All columns are potential Y-axes
    print("Available sensors:", available_sensors)

    # User selects Y-axis sensors
    selected_sensors = input("Enter the sensor names for Y-axis separated by commas: ").split(',')
    selected_sensors = [sensor.strip() for sensor in selected_sensors if sensor.strip() in available_sensors]

    if not selected_sensors:
        raise ValueError("No valid Y-axis sensors selected!")

    # Create subplots
    num_sensors = len(selected_sensors)
    fig, axes = plt.subplots(num_sensors, 1, figsize=(12, 3 * num_sensors), sharex=True)

    if num_sensors == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for ax, sensor in zip(axes, selected_sensors):
        ax.plot(df[x_axis_column], df[sensor], marker='o', linestyle='-', markersize=4)
        ax.set_title(f"{sensor} vs {x_axis_column}", fontsize=12, fontweight='bold')
        ax.set_ylabel(sensor, fontsize=10)  # **Explicitly setting Y-axis label**
        ax.grid(True)

    plt.xlabel(x_axis_column, fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.subplots_adjust(left=0.15, hspace=0.4)  # Ensure space for y-labels
    plt.show()

# Example usage
csv_file = "./logs/modified_Training2_19112024_144443.csv"
plot_selected_sensors(csv_file)