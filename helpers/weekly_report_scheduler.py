from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from app_io.paths import P, ensure_weekly_reports_dir, weekly_report_path
from helpers.app_logger import log_event
from helpers.json_io import load_json
from scripts.cli.run_weekly_report import generate_weekly_report


@dataclass
class WeeklyReportAutoResult:
    ran: bool
    reason: str
    due_iso: str
    pdf_path: str = ""
    error: str = ""


def _this_week_due(now: datetime) -> datetime:
    # Monday=0 ... Sunday=6; Wednesday is 2.
    monday = now - timedelta(days=now.weekday())
    due = monday.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=2)
    return due


def _load_state() -> dict:
    return load_json(P.weekly_reports_state_json, default={})


def _save_state(payload: dict) -> None:
    ensure_weekly_reports_dir()
    os.makedirs(os.path.dirname(P.weekly_reports_state_json), exist_ok=True)
    with open(P.weekly_reports_state_json, "w", encoding="utf-8") as f:
        import json

        json.dump(payload, f, ensure_ascii=True, indent=2)


def maybe_run_weekly_report_auto(now: datetime | None = None) -> WeeklyReportAutoResult:
    now = now or datetime.now()
    due = _this_week_due(now)

    if now < due:
        return WeeklyReportAutoResult(
            ran=False,
            reason="not_due_yet",
            due_iso=due.isoformat(timespec="seconds"),
        )

    due_iso = due.isoformat(timespec="seconds")
    state = _load_state()
    last_due_iso = str(state.get("last_due_iso", "")).strip()
    if last_due_iso == due_iso:
        return WeeklyReportAutoResult(
            ran=False,
            reason="already_ran_for_due_slot",
            due_iso=due_iso,
            pdf_path=str(state.get("last_pdf", "") or ""),
        )

    start_dt = pd.Timestamp(due - timedelta(days=7))
    end_dt = pd.Timestamp(due)
    file_name = f"tower_weekly_report_{start_dt.strftime('%Y%m%d_%H%M')}_{end_dt.strftime('%Y%m%d_%H%M')}.pdf"
    out_pdf = weekly_report_path(file_name)

    try:
        result = generate_weekly_report(start_dt=start_dt, end_dt=end_dt, dt_cap=2.0, output_pdf=out_pdf)
        pdf_path = str(result.get("pdf_path", out_pdf))
        new_state = {
            "last_due_iso": due_iso,
            "last_run_at": datetime.now().isoformat(timespec="seconds"),
            "last_pdf": pdf_path,
            "last_period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            },
        }
        _save_state(new_state)
        log_event(
            "weekly_report_auto_generated",
            due_iso=due_iso,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            pdf=pdf_path,
        )
        return WeeklyReportAutoResult(
            ran=True,
            reason="generated",
            due_iso=due_iso,
            pdf_path=pdf_path,
        )
    except Exception as e:
        msg = str(e)
        log_event("weekly_report_auto_failed", due_iso=due_iso, error=msg)
        return WeeklyReportAutoResult(
            ran=False,
            reason="generation_failed",
            due_iso=due_iso,
            error=msg,
        )
