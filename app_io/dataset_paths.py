"""
Backward-compatible dataset path helpers.
Prefer importing from app_io.paths directly in new code.
"""

from app_io.paths import ensure_dataset_dir, dataset_csv_path

__all__ = ["ensure_dataset_dir", "dataset_csv_path"]

