from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# List of pipeline scripts (in order)
SCRIPTS = [
    BASE_DIR / "csv_col_nam_map.py",   # column name normalization
    BASE_DIR / "create_folder.py",     # create output folder from latest CSV
    BASE_DIR / "corr_map.py",          # correlation calculations
    BASE_DIR / "good_fiber.py",        # good zones representation
    BASE_DIR / "coating_function.py",  # coating report
    BASE_DIR / "report_pdf_maker.py",  # merged report
]


def main() -> int:
    for script in SCRIPTS:
        try:
            print(f"Running {script.name}...")
            subprocess.run([sys.executable, str(script)], check=True, cwd=str(PROJECT_ROOT))
            print(f"{script.name} completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error while running {script.name}: {e}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
