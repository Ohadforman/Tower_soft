from app_io.legacy_path_compat import install_legacy_path_compat
from app_io.paths import P

# Install compatibility before importing/using modules that may still touch legacy root names.
install_legacy_path_compat(P)

from app.bootstrap import build_runtime, configure_page, ensure_coating_config
from app.navigation import init_session_state, render_sidebar_navigation
from app.router import render_selected_tab

configure_page()
ensure_coating_config(P)
runtime = build_runtime(P)
init_session_state()
selected_tab = render_sidebar_navigation()

render_selected_tab(
    tab_selection=selected_tab,
    P=P,
    image_base64=runtime.image_base64,
    failed_reason_col=runtime.failed_reason_col,
    orders_file=runtime.orders_file,
)
