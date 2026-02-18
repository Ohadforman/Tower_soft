import os
import json
import pandas as pd

THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(THIS_DIR, "ml_models")

TRUST_JSON = os.path.join(MODEL_DIR, "trust_report.json")
WF_JSON = os.path.join(MODEL_DIR, "walk_forward_report.json")
META_JSON = os.path.join(MODEL_DIR, "metadata.json")  # step2p5 output
COMPARE_JSON = os.path.join(MODEL_DIR, "metadata_compare_ridge_huber.json")  # step2p6 output
WORSE_CSV = os.path.join(MODEL_DIR, "worse_cases_report.csv")

OUT_MD = os.path.join(MODEL_DIR, "ml_wrapup_summary.md")
OUT_TXT = os.path.join(MODEL_DIR, "ml_wrapup_summary.txt")


def _read_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def fmt(x, nd=3):
    try:
        if x is None:
            return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main():
    trust = _read_json(TRUST_JSON)
    wf = _read_json(WF_JSON)
    meta = _read_json(COMPARE_JSON) or _read_json(META_JSON)

    worse_df = pd.read_csv(WORSE_CSV) if os.path.exists(WORSE_CSV) else None

    lines = []
    lines.append("# ML Coating Calibration — Wrap-up Summary\n")

    # -----------------------------
    # Models used
    # -----------------------------
    if meta:
        model_type = meta.get("model_type", "ridge/unknown")
        ridge_alpha = meta.get("ridge_alpha", {})
        cv = meta.get("cv", {})
        lines.append("## Model configuration")
        lines.append(f"- Model family: **{model_type}** (small-data linear baseline)")
        if ridge_alpha:
            lines.append(f"- Ridge alpha (main): **{ridge_alpha.get('main')}**")
            lines.append(f"- Ridge alpha (secondary): **{ridge_alpha.get('secondary')}**")
        if cv:
            dm = cv.get("diameter_mae_um", {})
            imp = cv.get("diameter_improvement_pct", {})
            lines.append("- CV (LOO) diameter MAE:")
            lines.append(f"  - Main: theory {fmt(dm.get('main_baseline_theory'))} → corrected {fmt(dm.get('main_corrected'))} µm ({fmt(imp.get('main'),2)}%)")
            lines.append(f"  - Secondary: theory {fmt(dm.get('secondary_baseline_theory'))} → corrected {fmt(dm.get('secondary_corrected'))} µm ({fmt(imp.get('secondary'),2)}%)")
        lines.append("")

    # -----------------------------
    # Walk-forward
    # -----------------------------
    if wf:
        main_over = wf.get("main", {}).get("overall", {})
        sec_over = wf.get("secondary", {}).get("overall", {})
        lines.append("## Walk-forward validation (more realistic than LOO)")
        lines.append(f"- Main: baseline MAE {fmt(main_over.get('baseline_mae_um_mean'))} → corrected MAE {fmt(main_over.get('corrected_mae_um_mean'))} µm "
                     f"({fmt(main_over.get('improvement_pct_mean'),2)}%), blocks={main_over.get('blocks')}")
        lines.append(f"- Secondary: baseline MAE {fmt(sec_over.get('baseline_mae_um_mean'))} → corrected MAE {fmt(sec_over.get('corrected_mae_um_mean'))} µm "
                     f"({fmt(sec_over.get('improvement_pct_mean'),2)}%), blocks={sec_over.get('blocks')}")
        lines.append("")

    # -----------------------------
    # Trust tracker
    # -----------------------------
    if trust:
        lines.append("## Rolling trust metrics (shadow-mode)")
        for w in trust.get("windows", []):
            win = w.get("window")
            n = w.get("n")
            m = w.get("main", {})
            s = w.get("secondary", {})
            lines.append(f"### Last {win} draws (n={n})")
            lines.append(f"- Main: Improved {fmt(m.get('pct_improved'),1)}% | Worse {fmt(m.get('pct_worse'),1)}% | "
                         f"mean Δerr {fmt(m.get('mean_improvement_um'))} µm | worst {fmt(m.get('worst_improvement_um'))} µm")
            lines.append(f"- Secondary: Improved {fmt(s.get('pct_improved'),1)}% | Worse {fmt(s.get('pct_worse'),1)}% | "
                         f"mean Δerr {fmt(s.get('mean_improvement_um'))} µm | worst {fmt(s.get('worst_improvement_um'))} µm")
        lines.append("")

    # -----------------------------
    # Worst-cases patterns
    # -----------------------------
    if worse_df is not None and not worse_df.empty:
        lines.append("## Worst-case analysis (where ML got worse)")
        main_worst = worse_df.dropna(subset=["main_improvement_um"]).sort_values("main_improvement_um").head(5)
        sec_worst = worse_df.dropna(subset=["sec_improvement_um"]).sort_values("sec_improvement_um").head(5)

        lines.append("### Top 5 worst MAIN cases")
        for _, r in main_worst.iterrows():
            lines.append(f"- {r['dataset_csv']}: Δerr={fmt(r['main_improvement_um'])} µm, "
                         f"main_type={r.get('main_coating_type')}, sec_type={r.get('sec_coating_type')}, "
                         f"speed={fmt(r.get('speed_m_min'))}, tension={fmt(r.get('tension_g'))}, "
                         f"Tmain={fmt(r.get('main_coat_temp_c'))}, Tsec={fmt(r.get('sec_coat_temp_c'))}, Tfurn={fmt(r.get('furnace_temp_c'))}")

        lines.append("\n### Top 5 worst SECONDARY cases")
        for _, r in sec_worst.iterrows():
            lines.append(f"- {r['dataset_csv']}: Δerr={fmt(r['sec_improvement_um'])} µm, "
                         f"main_type={r.get('main_coating_type')}, sec_type={r.get('sec_coating_type')}, "
                         f"speed={fmt(r.get('speed_m_min'))}, tension={fmt(r.get('tension_g'))}, "
                         f"Tmain={fmt(r.get('main_coat_temp_c'))}, Tsec={fmt(r.get('sec_coat_temp_c'))}, Tfurn={fmt(r.get('furnace_temp_c'))}")

        lines.append("\n### Observed pattern (interpretation)")
        lines.append("- ML improves error in most stable operating regions, but degrades on **edge regimes** (nonlinear zones) such as low-speed or extreme temperature conditions and certain coating-type combinations.")
        lines.append("- Current deployment mode is **shadow-mode**: theory remains authoritative; ML outputs are logged for continuous evaluation.")
        lines.append("")

    # -----------------------------
    # Next steps
    # -----------------------------
    lines.append("## Recommended next steps (still shadow-mode)")
    lines.append("1. Keep accumulating real draws (target: 200–500 rows).")
    lines.append("2. Add an 'envelope/risk' label (inside/edge/outside) to identify out-of-distribution conditions.")
    lines.append("3. Consider segmented models by coating chemistry once enough data per type exists.")
    lines.append("")

    md = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(md)

    print("✅ Wrote:")
    print("-", OUT_MD)
    print("-", OUT_TXT)


if __name__ == "__main__":
    main()