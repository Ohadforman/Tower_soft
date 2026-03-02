from __future__ import annotations

import builtins
import os
from typing import Dict


def _build_legacy_map(P) -> Dict[str, str]:
    base = P.root_dir
    return {
        "draw_orders.csv": P.orders_csv,
        os.path.join(base, "draw_orders.csv"): P.orders_csv,
        "part_orders.csv": P.parts_orders_csv,
        os.path.join(base, "part_orders.csv"): P.parts_orders_csv,
        "tower_schedule.csv": P.schedule_csv,
        os.path.join(base, "tower_schedule.csv"): P.schedule_csv,
        "tower_temps.csv": P.tower_temps_csv,
        os.path.join(base, "tower_temps.csv"): P.tower_temps_csv,
        "tower_containers.csv": P.tower_containers_csv,
        os.path.join(base, "tower_containers.csv"): P.tower_containers_csv,
        "config_coating.json": P.coating_config_json,
        os.path.join(base, "config_coating.json"): P.coating_config_json,
        "protocols.json": P.protocols_json,
        os.path.join(base, "protocols.json"): P.protocols_json,
        "development_projects.csv": os.path.join(P.data_dir, "development_projects.csv"),
        os.path.join(base, "development_projects.csv"): os.path.join(P.data_dir, "development_projects.csv"),
        "development_experiments.csv": os.path.join(P.data_dir, "development_experiments.csv"),
        os.path.join(base, "development_experiments.csv"): os.path.join(P.data_dir, "development_experiments.csv"),
        "experiment_updates.csv": P.experiment_updates_csv,
        os.path.join(base, "experiment_updates.csv"): P.experiment_updates_csv,
    }


def install_legacy_path_compat(P) -> None:
    """
    Runtime compatibility shim:
    transparently redirects legacy root file paths to canonical P.* paths.
    """
    if getattr(install_legacy_path_compat, "_installed", False):
        return

    legacy_map = _build_legacy_map(P)

    def _resolve(path):
        try:
            p = os.fspath(path)
        except Exception:
            return path
        if p in legacy_map:
            return legacy_map[p]
        bn = os.path.basename(p)
        if bn in legacy_map and (p == bn or os.path.dirname(p) == P.root_dir):
            return legacy_map[bn]
        return path

    _orig_open = builtins.open
    _orig_exists = os.path.exists
    _orig_isfile = os.path.isfile
    _orig_getmtime = os.path.getmtime
    _orig_stat = os.stat

    def _open_compat(file, *args, **kwargs):
        return _orig_open(_resolve(file), *args, **kwargs)

    def _exists_compat(path):
        if _orig_exists(path):
            return True
        return _orig_exists(_resolve(path))

    def _isfile_compat(path):
        if _orig_isfile(path):
            return True
        return _orig_isfile(_resolve(path))

    def _getmtime_compat(path):
        try:
            return _orig_getmtime(path)
        except Exception:
            return _orig_getmtime(_resolve(path))

    def _stat_compat(path, *args, **kwargs):
        try:
            return _orig_stat(path, *args, **kwargs)
        except Exception:
            return _orig_stat(_resolve(path), *args, **kwargs)

    builtins.open = _open_compat
    os.path.exists = _exists_compat
    os.path.isfile = _isfile_compat
    os.path.getmtime = _getmtime_compat
    os.stat = _stat_compat
    install_legacy_path_compat._installed = True
