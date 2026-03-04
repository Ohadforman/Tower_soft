def render_process_setup_tab_main(P, ORDERS_FILE):
    import os
    import pandas as pd

    from renders.components.process_setup import render_process_setup_tab

    try:
        orders_path = ORDERS_FILE if ORDERS_FILE else P.orders_csv
        df_orders = pd.read_csv(orders_path, keep_default_na=False) if os.path.exists(orders_path) else pd.DataFrame()
    except Exception:
        df_orders = pd.DataFrame()
    
    render_process_setup_tab(
        orders_df=df_orders,
        orders_file=ORDERS_FILE,
    )
