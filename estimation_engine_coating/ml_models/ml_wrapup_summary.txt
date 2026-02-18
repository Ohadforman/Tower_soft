# ML Coating Calibration — Wrap-up Summary

## Model configuration
- Model family: **ridge/unknown** (small-data linear baseline)

## Walk-forward validation (more realistic than LOO)
- Main: baseline MAE 1.412 → corrected MAE 0.824 µm (30.20%), blocks=8
- Secondary: baseline MAE 1.482 → corrected MAE 1.086 µm (19.20%), blocks=8

## Rolling trust metrics (shadow-mode)
### Last 10 draws (n=10)
- Main: Improved 80.0% | Worse 20.0% | mean Δerr 0.758 µm | worst -0.307 µm
- Secondary: Improved 100.0% | Worse 0.0% | mean Δerr 1.243 µm | worst 0.036 µm
### Last 20 draws (n=20)
- Main: Improved 85.0% | Worse 15.0% | mean Δerr 0.785 µm | worst -0.307 µm
- Secondary: Improved 80.0% | Worse 20.0% | mean Δerr 0.781 µm | worst -0.970 µm
### Last 50 draws (n=50)
- Main: Improved 80.0% | Worse 20.0% | mean Δerr 0.791 µm | worst -1.160 µm
- Secondary: Improved 80.0% | Worse 20.0% | mean Δerr 0.655 µm | worst -1.649 µm

## Worst-case analysis (where ML got worse)
### Top 5 worst MAIN cases
- FAKE_DRAW_0020.csv: Δerr=-1.160 µm, main_type=DSM-950, sec_type=OF-136, speed=18.357, tension=47.755, Tmain=50.969, Tsec=50.233, Tfurn=1922.092
- FAKE_DRAW_0022.csv: Δerr=-0.958 µm, main_type=OF-136, sec_type=OF-136, speed=15.201, tension=39.099, Tmain=38.343, Tsec=51.561, Tfurn=1864.175
- FAKE_DRAW_0017.csv: Δerr=-0.673 µm, main_type=DSM-950, sec_type=OF-136, speed=6.653, tension=32.726, Tmain=41.319, Tsec=41.214, Tfurn=1756.065
- FAKE_DRAW_0016.csv: Δerr=-0.432 µm, main_type=DP1032, sec_type=DSM-950, speed=6.866, tension=67.037, Tmain=55.380, Tsec=53.250, Tfurn=1979.209
- FAKE_DRAW_0002.csv: Δerr=-0.374 µm, main_type=DP1032, sec_type=DSM-950, speed=9.568, tension=70.057, Tmain=48.837, Tsec=59.888, Tfurn=1901.364

### Top 5 worst SECONDARY cases
- FAKE_DRAW_0022.csv: Δerr=-1.649 µm, main_type=OF-136, sec_type=OF-136, speed=15.201, tension=39.099, Tmain=38.343, Tsec=51.561, Tfurn=1864.175
- FAKE_DRAW_0047.csv: Δerr=-0.970 µm, main_type=DP1032, sec_type=OF-136, speed=8.738, tension=42.324, Tmain=54.206, Tsec=47.202, Tfurn=1846.428
- FAKE_DRAW_0002.csv: Δerr=-0.944 µm, main_type=DP1032, sec_type=DSM-950, speed=9.568, tension=70.057, Tmain=48.837, Tsec=59.888, Tfurn=1901.364
- FAKE_DRAW_0007.csv: Δerr=-0.450 µm, main_type=OF-136, sec_type=DSM-950, speed=15.405, tension=57.038, Tmain=51.555, Tsec=38.290, Tfurn=2012.223
- FAKE_DRAW_0024.csv: Δerr=-0.443 µm, main_type=DSM-950, sec_type=DP1032, speed=9.769, tension=57.577, Tmain=50.498, Tsec=39.679, Tfurn=1999.838

### Observed pattern (interpretation)
- ML improves error in most stable operating regions, but degrades on **edge regimes** (nonlinear zones) such as low-speed or extreme temperature conditions and certain coating-type combinations.
- Current deployment mode is **shadow-mode**: theory remains authoritative; ML outputs are logged for continuous evaluation.

## Recommended next steps (still shadow-mode)
1. Keep accumulating real draws (target: 200–500 rows).
2. Add an 'envelope/risk' label (inside/edge/outside) to identify out-of-distribution conditions.
3. Consider segmented models by coating chemistry once enough data per type exists.
