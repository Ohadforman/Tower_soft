def render_process_setup_tab_main(P, ORDERS_FILE):
    import os
    import pandas as pd
    import streamlit as st

    from renders.components.process_setup import render_process_setup_tab

    def _mtime(path: str) -> float:
        try:
            return float(os.path.getmtime(path))
        except Exception:
            return 0.0

    @st.cache_data(show_spinner=False)
    def _read_orders_cached(path: str, file_mtime: float):
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=False)

    try:
        orders_path = ORDERS_FILE if ORDERS_FILE else P.orders_csv
        df_orders = _read_orders_cached(orders_path, _mtime(orders_path))
    except Exception:
        df_orders = pd.DataFrame()
    
    render_process_setup_tab(
        orders_df=df_orders,
        orders_file=ORDERS_FILE,
    )
