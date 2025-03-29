import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from recent_csv import get_most_recent_csv  # Import the function
import json

# Load directory configuration from JSON
CONFIG_FILE = "directory_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")


def save_correlation_plots_to_pdf(csv_file, pdf_output_directory, logs_directory, output_csv):
    """
    Reads a CSV file, computes the correlation matrix, keeps only the upper diagonal values,
    appends it to an existing CSV file in logs/, and saves heatmap & trend plots to a multi-page PDF.
    """
    os.makedirs(pdf_output_directory, exist_ok=True)
    os.makedirs(logs_directory, exist_ok=True)

    pdf_filename = os.path.join(pdf_output_directory, "correlation_tower_data.pdf")
    output_csv_path = os.path.join(logs_directory, output_csv)
    log_name = os.path.basename(csv_file).replace(".csv", "")

    file_mod_time = os.path.getmtime(csv_file)
    file_mod_date = datetime.fromtimestamp(file_mod_time).strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(csv_file)
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_numeric.fillna(0, inplace=True)
    df_numeric = df_numeric.loc[:, df_numeric.var() != 0]
    correlation_matrix = df_numeric.corr()
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    upper_corr_values = correlation_matrix.where(mask).stack()

    row_df = pd.DataFrame([upper_corr_values.values],
                          columns=upper_corr_values.index.map(lambda x: f"{x[0]}-{x[1]}"))
    row_df.insert(0, "Log Name", log_name)
    row_df.insert(1, "Date", file_mod_date)
    row_df.fillna(0, inplace=True)

    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        updated_df = pd.concat([existing_df, row_df], ignore_index=True)
        updated_df.to_csv(output_csv_path, index=False)
    else:
        row_df.to_csv(output_csv_path, index=False)

    print(f"✅ Correlation matrix saved to: {output_csv_path}")
    df_corr = pd.read_csv(output_csv_path)
    if "Log Name" in df_corr.columns and "Date" in df_corr.columns:
        df_corr = df_corr.drop(columns=["Log Name", "Date"])

    with PdfPages(pdf_filename) as pdf:
        # First plot - Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.1,
            ax=ax,
            mask=~mask,
            annot_kws={"size": 4}
        )
        ax.set_title(f"Upper Diagonal Correlation Heatmap for {log_name}")
        fig.suptitle("Correlation Tower Data", fontsize=10, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Second plot - Line plots for correlation pairs
        correlation_pairs = list(df_corr.columns)
        num_pairs = len(correlation_pairs)
        plots_per_page = 35  # Maximum number of plots per page

        for i in range(0, num_pairs, plots_per_page):
            num_subplots = min(plots_per_page, num_pairs - i)
            fig, axes = plt.subplots(5, 7, figsize=(10, 6))
            axes = axes.flatten()

            for j in range(num_subplots):
                index = i + j
                col = correlation_pairs[index]
                ax = axes[j]
                ax.plot(df_corr.index, df_corr[col], marker='o', linestyle='-', label=col, markersize=1, linewidth=1)
                ax.set_title(f"{col}", fontsize=4)
                ax.tick_params(axis='both', labelsize=4)
                ax.grid(True, linewidth=0.3)

            # Hide any extra empty subplots
            for j in range(num_subplots, len(axes)):
                axes[j].axis("off")

            plt.subplots_adjust(wspace=0.4, hspace=0.5)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"📄 Correlation plots saved to: {pdf_filename}")


if __name__ == "__main__":
    folder_path = config["logs_directory"]
    recent_csv = get_most_recent_csv()
    folder_out_name = os.path.splitext(recent_csv)[0]
    logs_directory = config["correlation_logs_directory"]
    output_directory = os.path.join(config["output_folders"], folder_out_name)
    csv_file = os.path.join(folder_path, recent_csv)
    save_correlation_plots_to_pdf(csv_file, output_directory, logs_directory, config["correlation_csv"])
