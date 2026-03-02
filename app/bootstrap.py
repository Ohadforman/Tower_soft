from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app_io.config import coating_options_from_cfg
from app_io.paths import ensure_backups_dir
from helpers.constants import FAILED_DESC_COL
from helpers.duckdb_io import get_duckdb_conn
from helpers.json_io import load_json
from renders.support.assets import get_base64_image


@dataclass
class AppRuntime:
    failed_reason_col: str
    orders_file: str
    image_base64: str
    _con: object


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
    # Keep connection initialization at startup to preserve current behavior.
    ensure_backups_dir()
    con = get_duckdb_conn(P.duckdb_path)
    return AppRuntime(
        failed_reason_col=FAILED_DESC_COL,
        orders_file=P.orders_csv,
        image_base64=get_base64_image(P.home_bg_image),
        _con=con,
    )
