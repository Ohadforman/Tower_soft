from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import streamlit as st

from app_io.config import coating_options_from_cfg
from app_io.paths import ensure_backups_dir
from helpers.constants import FAILED_DESC_COL
from helpers.json_io import load_json
from renders.support.assets import get_base64_image


@dataclass
class AppRuntime:
    failed_reason_col: str
    orders_file: str
    image_base64: str


def configure_page() -> None:
    st.set_page_config(
        page_title="Tower",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


@st.cache_data(show_spinner=False)
def _load_coating_config(path: str):
    return load_json(path)


def ensure_coating_config(P) -> None:
    try:
        config = _load_coating_config(P.coating_config_json)
    except FileNotFoundError:
        st.error(f"Missing coating config file: {P.coating_config_json}")
        st.stop()
    coating_options = coating_options_from_cfg(config)
    coatings = config.get("coatings", {})
    dies = config.get("dies", {})
    if not coating_options or not coatings or not dies:
        st.error("Coatings and/or Dies not configured in config_coating.json")
        st.stop()


def build_runtime(P) -> AppRuntime:
    # Keep startup light; tabs open DB connections only when needed.
    ensure_backups_dir()
    return AppRuntime(
        failed_reason_col=FAILED_DESC_COL,
        orders_file=P.orders_csv,
        image_base64=get_base64_image(P.home_bg_image),
    )


def render_startup_banner(P, startup_fail_count: int = 0, safe_mode: bool = False) -> None:
    root_fallback = os.path.abspath(os.path.join(P.root_dir, "data"))
    db_path = os.path.abspath(P.duckdb_path)
    db_mode = "fallback" if db_path.startswith(root_fallback + os.sep) else "local"
    latest_audit = "not found"
    try:
        patt = os.path.join(P.reports_dir, "path_audit", "path_permissions_*.json")
        files = sorted(glob.glob(patt))
        if files:
            with open(files[-1], "r", encoding="utf-8") as f:
                rep = json.load(f)
            summ = rep.get("summary", {})
            latest_audit = f"issues={summ.get('issues', '?')}, critical={summ.get('critical_issues', '?')}"
    except Exception:
        pass

    mode_txt = "SAFE MODE" if safe_mode else "NORMAL"
    level = "⚠️" if startup_fail_count else "✅"
    st.info(
        f"{level} Startup: `{mode_txt}` | root: `{P.root_dir}` | "
        f"duckdb: `{db_mode}` (`{db_path}`) | backups: `{P.backups_dir}` | "
        f"last audit: `{latest_audit}`"
    )
