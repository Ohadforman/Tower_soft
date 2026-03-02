"""
Compatibility wrapper for legacy imports.
Old code may import: from renders.process_setup_tab import render_process_setup_tab_main
"""


def render_process_setup_tab_main(*args, **kwargs):
    from renders.tabs.process_setup_tab import render_process_setup_tab_main as _impl
    return _impl(*args, **kwargs)

