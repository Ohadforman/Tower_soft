import os

DATASET_DIR = "data_set_csv"


def ensure_dataset_dir() -> str:
    os.makedirs(DATASET_DIR, exist_ok=True)
    return DATASET_DIR


def safe_dataset_filename(name: str) -> str:
    """
    Prevent subfolders and weird paths.
    Converts 'ELOP - 130/test.csv' -> 'test.csv'
    """
    s = str(name or "").replace("\\", "/").strip()
    s = os.path.basename(s)
    s = s.replace("/", "_").strip()
    return s


def dataset_csv_path(filename: str) -> str:
    ensure_dataset_dir()
    return os.path.join(DATASET_DIR, safe_dataset_filename(filename))