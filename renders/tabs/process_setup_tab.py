def render_process_setup_tab_main(P, ORDERS_FILE):
    import os
    import pandas as pd

    from renders.components.process_setup import render_process_setup_tab

    df_orders = pd.read_csv(P.orders_csv, keep_default_na=False) if os.path.exists(P.orders_csv) else pd.DataFrame()
    try:
        df_orders = pd.read_csv(ORDERS_FILE, keep_default_na=False) if os.path.exists(ORDERS_FILE) else pd.DataFrame()
    except Exception:
        df_orders = pd.DataFrame()
    
    render_process_setup_tab(
        orders_df=df_orders,
        orders_file=ORDERS_FILE,
    )
