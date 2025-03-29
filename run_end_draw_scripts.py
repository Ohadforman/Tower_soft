import subprocess

# List of Python scripts to run
scripts_to_run = [
    "csv_col_nam_map.py",    # makes a columns names change
    "create_folder.py",      # create a folder for the output pdfs
    "corr_map.py",           # create correlation calculations
    "good_fiber.py",         # takes the values from good zones and represent
    "coating_function.py",   # coating report maker
    "report_pdf_maker.py"    # merge for full report
]

# Loop through the list and execute each script
for script in scripts_to_run:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script}: {e}\n")
