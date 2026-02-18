# helpers/db.py
from __future__ import annotations
import duckdb
import streamlit as st
from app_io.paths import P

def get_duckdb_conn():
    """
    One DuckDB connection per Streamlit session.
    """
    if "tower_con" not in st.session_state:
        st.session_state.tower_con = duckdb.connect(P.duckdb_path)
    return st.session_state.tower_con