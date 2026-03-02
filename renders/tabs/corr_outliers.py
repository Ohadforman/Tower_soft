import itertools
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_corr_outliers_tab(draw_folder: str, maint_folder: str):
    os.makedirs(maint_folder, exist_ok=True)

    def _safe_key(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))[:90]

    def _to_float(v):
        try:
            if v is None:
                return np.nan
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return np.nan
            return float(s)
        except Exception:
            return np.nan

    def _rolling_corr(df: pd.DataFrame, col_a: str, col_b: str, win: int):
        s = df[["window_time", "file", col_a, col_b]].copy()
        s = s.dropna(subset=[col_a, col_b])
        if len(s) < win:
            return None

        corrs, times, files = [], [], []
        for i in range(win - 1, len(s)):
            w = s.iloc[i - win + 1 : i + 1]
            c = w[col_a].corr(w[col_b])
            if pd.notna(c):
                corrs.append(float(c))
                times.append(s.iloc[i]["window_time"])
                files.append(s.iloc[i]["file"])

        if not corrs:
            return None

        return pd.DataFrame({"window_time": times, "file": files, "corr": corrs})

    st.markdown("---")
    st.subheader("1) Scan logs → one numeric snapshot per file (time = file mtime)")

    if not os.path.exists(draw_folder):
        st.warning(f"Logs folder not found: {draw_folder}")
        return

    log_files = sorted(
        [os.path.join(draw_folder, f) for f in os.listdir(draw_folder) if f.lower().endswith(".csv")],
        key=lambda p: os.path.getmtime(p),
    )

    if not log_files:
        st.info("No log CSVs found.")
        return

    st.caption(f"Found {len(log_files)} log CSV files.")

    n_files = len(log_files)
    max_cap = min(2000, n_files)
    if max_cap <= 1:
        st.info(f"Only {n_files} log file(s) found - need at least 2.")
        return

    max_files = st.slider(
        "Max files to process",
        min_value=2,
        max_value=max_cap,
        value=min(300, max_cap),
        step=1 if max_cap < 25 else 10,
        key="corr_many_max_files",
    )
    log_files = log_files[-max_files:]

    rows = []
    fail = 0
    for path in log_files:
        try:
            df = pd.read_csv(path)
            if df is None or df.empty:
                continue

            t = datetime.fromtimestamp(os.path.getmtime(path))
            last = df.iloc[-1].copy()
            rec = {"window_time": pd.to_datetime(t), "file": os.path.basename(path)}

            for col in df.columns:
                num = _to_float(last[col])
                if np.isfinite(num):
                    rec[col] = num

            rows.append(rec)
        except Exception:
            fail += 1

    if not rows:
        st.warning("No usable numeric data found in logs.")
        return

    base = pd.DataFrame(rows)
    base["window_time"] = pd.to_datetime(base["window_time"], errors="coerce")
    base = base.sort_values("window_time").reset_index(drop=True)

    st.caption(f"Usable rows: {len(base)} | Failed files: {fail}")

    with st.expander("Preview numeric table", expanded=False):
        st.dataframe(base.tail(50), use_container_width=True)

    numeric_cols = [c for c in base.columns if c not in {"window_time", "file"}]
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns across logs to compute correlations.")
        return

    st.markdown("---")
    st.subheader("2) Pair settings")

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.0])
    with c1:
        win_max = max(5, min(200, len(base)))
        win = st.slider(
            "Rolling window (points)",
            min_value=3,
            max_value=win_max,
            value=min(20, win_max),
            step=1 if win_max < 25 else 5,
            key="corr_many_win",
        )
    with c2:
        min_points = st.number_input(
            "Min points for a pair (after NaN drop)",
            min_value=3,
            value=max(10, int(win)),
            step=1,
            key="corr_many_min_points",
        )
    with c3:
        max_pairs = st.number_input(
            "Max pairs to plot",
            min_value=1,
            value=20,
            step=1,
            key="corr_many_max_pairs",
        )
    with c4:
        pair_sort = st.selectbox(
            "Sort pairs by",
            ["|median corr| (strongest)", "corr variability (std)", "alphabetical"],
            index=0,
            key="corr_many_sort",
        )

    col_filter = st.text_input(
        "Column filter (optional): only include columns containing this text (case-insensitive)",
        value="",
        key="corr_many_filter",
    ).strip().lower()

    cols_use = numeric_cols
    if col_filter:
        cols_use = [c for c in numeric_cols if col_filter in str(c).lower()]

    if len(cols_use) < 2:
        st.warning("Filter left fewer than 2 numeric columns.")
        return

    st.markdown("---")
    st.subheader("3) Compute + plot correlations for MANY pairs")

    pairs = list(itertools.combinations(cols_use, 2))
    if not pairs:
        st.info("No pairs available.")
        return

    st.caption(f"Candidate pairs: {len(pairs)} (from {len(cols_use)} numeric columns)")

    scored = []
    for a, b in pairs:
        s = base[[a, b]].dropna()
        n = len(s)
        if n < int(min_points):
            continue

        c = s[a].corr(s[b])
        if pd.isna(c):
            continue

        var_est = np.nan
        if n >= max(int(win), 8):
            idxs = np.linspace(max(int(win) - 1, 0), n - 1, num=min(30, n - int(win) + 1), dtype=int)
            cc = []
            for ii in idxs:
                w = s.iloc[ii - int(win) + 1 : ii + 1]
                if len(w) == int(win):
                    vv = w[a].corr(w[b])
                    if pd.notna(vv):
                        cc.append(float(vv))
            if cc:
                var_est = float(np.nanstd(cc))

        scored.append(
            {"a": a, "b": b, "n": int(n), "corr": float(c), "abs_corr": float(abs(c)), "var_est": var_est}
        )

    if not scored:
        st.warning("No pairs passed the Min points requirement.")
        return

    df_pairs = pd.DataFrame(scored)
    if pair_sort == "|median corr| (strongest)":
        df_pairs = df_pairs.sort_values(["abs_corr", "n"], ascending=[False, False])
    elif pair_sort == "corr variability (std)":
        df_pairs = df_pairs.sort_values(["var_est", "n"], ascending=[False, False])
    else:
        df_pairs = df_pairs.sort_values(["a", "b"], ascending=[True, True])

    df_pairs = df_pairs.head(int(max_pairs)).reset_index(drop=True)

    with st.expander("Selected pairs (preview)", expanded=False):
        st.dataframe(df_pairs, use_container_width=True)

    for i, row in df_pairs.iterrows():
        a = row["a"]
        b = row["b"]

        g = _rolling_corr(base, a, b, int(win))
        if g is None or g.empty:
            continue

        title = f"{a}  vs  {b}  | rolling corr (win={int(win)})"
        fig = px.line(g, x="window_time", y="corr", markers=True, title=title)

        key = f"corr_pair_{i}_{_safe_key(a)}__{_safe_key(b)}__w{int(win)}"
        with st.expander(f"Pair {i + 1}: {a} <-> {b}", expanded=(i == 0)):
            st.caption(f"Points used (after NaN drop): {len(base[[a, b]].dropna())} | Rolling points: {len(g)}")
            st.plotly_chart(fig, use_container_width=True, key=key)
