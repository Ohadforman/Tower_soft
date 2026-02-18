# helpers/duckdb_io.py
import duckdb
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_duckdb_conn(db_path: str):
    # read_only=False because you write sometimes
    return duckdb.connect(db_path)