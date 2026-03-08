import os
import html

import streamlit as st

from app_io.legacy_path_compat import install_legacy_path_compat
from app_io.paths import P

# Install compatibility before importing/using modules that may still touch legacy root names.
install_legacy_path_compat(P)

from app.bootstrap import build_runtime, configure_page, ensure_coating_config
from app.navigation import init_session_state, render_sidebar_navigation
from app.router import render_selected_tab
from helpers.app_logger import log_event
from helpers.error_registry import get_error_help
from helpers.weekly_report_scheduler import maybe_run_weekly_report_auto
from renders.support.style_utils import apply_blue_clean_base_theme
from tests.runners.preflight import run_checks

configure_page()
apply_blue_clean_base_theme()
safe_mode = os.environ.get("TOWER_SAFE_MODE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
if "startup_results_cache" not in st.session_state:
    st.session_state["startup_results_cache"] = run_checks()
startup_results = st.session_state["startup_results_cache"]
startup_failures = [r for r in startup_results if not r.ok]

if startup_failures and not safe_mode:
    log_event("startup_checks_failed", safe_mode=False, failures=len(startup_failures))
    st.error("Startup checks failed. Fix the following issues before running the app:")
    lines = [f"- [{r.code}] {r.name}: {r.details}" for r in startup_failures]
    st.code("\n".join(lines), language="text")
    st.caption("Quick fixes:")
    for code in {r.code for r in startup_failures}:
        h = get_error_help(code)
        st.write(f"- `{h['code']}` {h['title']}: {h['fix']}")
    st.info("Tip: set `TOWER_SAFE_MODE=1` to open only safe tabs for debugging.")
    st.stop()

if startup_failures and safe_mode:
    log_event("startup_checks_failed", safe_mode=True, failures=len(startup_failures))
    st.warning("Startup checks failed. Running in SAFE MODE with limited tabs.")
    lines = [f"- [{r.code}] {r.name}: {r.details}" for r in startup_failures]
    st.code("\n".join(lines), language="text")
    with st.expander("Quick fix hints"):
        for code in {r.code for r in startup_failures}:
            h = get_error_help(code)
            st.write(f"- `{h['code']}` {h['title']}: {h['fix']}")

ensure_coating_config(P)
runtime = build_runtime(P)
init_session_state()

# Weekly report auto-run:
# Run at most once per Streamlit session; helper itself enforces once per due slot.
if "weekly_report_auto_checked" not in st.session_state:
    auto_res = maybe_run_weekly_report_auto()
    st.session_state["weekly_report_auto_checked"] = True
    st.session_state["weekly_report_auto_result"] = {
        "ran": auto_res.ran,
        "reason": auto_res.reason,
        "due_iso": auto_res.due_iso,
        "pdf_path": auto_res.pdf_path,
        "error": auto_res.error,
    }
    if auto_res.ran:
        st.toast("Weekly report generated automatically.", icon="✅")
    elif auto_res.reason == "generation_failed":
        st.toast("Weekly report auto-generation failed. Check logs.", icon="⚠️")

selected_tab = render_sidebar_navigation(safe_mode=safe_mode and bool(startup_failures))
if st.session_state.get("_theme_last_tab") != selected_tab:
    apply_blue_clean_base_theme(force=True)
    st.session_state["_theme_last_tab"] = selected_tab

# One-time startup popup on first page only (Home).
if (
    selected_tab == "🏠 Home"
    and not startup_failures
    and not st.session_state.get("startup_popup_shown", False)
):
    checks_html = "".join(
        f"<li>{html.escape(r.name)}: <b>PASS</b></li>" for r in startup_results
    )
    st.markdown(
        f"""
        <style>
          .startup-popup {{
            position: fixed;
            top: 78px;
            right: 18px;
            z-index: 9999;
            width: min(520px, 92vw);
            background: rgba(12, 22, 36, 0.95);
            border: 1px solid rgba(120, 200, 255, 0.45);
            border-radius: 14px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
            color: #e8f4ff;
            padding: 14px 16px;
            animation: fadeOutStartup 8s forwards;
          }}
          .startup-popup h4 {{
            margin: 0 0 8px 0;
            font-size: 16px;
          }}
          .startup-popup ul {{
            margin: 0;
            padding-left: 20px;
            max-height: 180px;
            overflow: auto;
            font-size: 13px;
          }}
          @keyframes fadeOutStartup {{
            0%, 80% {{ opacity: 1; transform: translateY(0); }}
            100% {{ opacity: 0; transform: translateY(-8px); visibility: hidden; }}
          }}
        </style>
        <div class="startup-popup">
          <h4>✅ Startup Checks Passed</h4>
          <ul>{checks_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["startup_popup_shown"] = True

render_selected_tab(
    tab_selection=selected_tab,
    P=P,
    image_base64=runtime.image_base64,
    failed_reason_col=runtime.failed_reason_col,
    orders_file=runtime.orders_file,
)
