import streamlit as st

def safe_str_from_state(key: str, default="") -> str:
    """
    Safely read a string value from st.session_state.
    Always returns a stripped string.
    """
    return str(st.session_state.get(key, default) or "").strip()