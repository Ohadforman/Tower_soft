from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


REQUIRED_CSV_COLUMNS: Dict[str, List[str]] = {
    "orders": ["Status", "Preform Number", "Fiber Project", "Timestamp"],
    "parts_orders": ["Status", "Part Name", "Details"],
    "schedule": ["Event Type", "Start DateTime", "End DateTime", "Description", "Recurrence"],
    "tower_temps": [
        "die_holder_primary_c",
        "die_holder_secondary_c",
        "A_container_c",
        "B_container_c",
        "C_container_c",
        "D_container_c",
    ],
    "tower_containers": ["A_level_kg", "A_type", "B_level_kg", "B_type", "C_level_kg", "C_type", "D_level_kg", "D_type"],
}


@dataclass
class CsvValidationResult:
    ok: bool
    missing_columns: List[str]
    row_count: int
    columns: List[str]


def validate_csv_schema(path: str, required_columns: List[str]) -> CsvValidationResult:
    df = pd.read_csv(path, keep_default_na=False)
    cols = list(df.columns)
    missing = [c for c in required_columns if c not in cols]
    return CsvValidationResult(
        ok=(len(missing) == 0),
        missing_columns=missing,
        row_count=len(df),
        columns=cols,
    )
