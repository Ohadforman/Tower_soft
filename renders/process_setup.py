"""
Compatibility wrapper for legacy imports.
Old code may import: from renders.process_setup import render_process_setup_tab
"""


def render_process_setup_tab(*args, **kwargs):
    from renders.components.process_setup import render_process_setup_tab as _impl
    return _impl(*args, **kwargs)

