# Performance Notes

This document summarizes the current performance design of the Tower app after the V2 optimization pass.

## What We Optimized

The main theme of the optimization work was:

- reduce repeated file reads
- cache read-only data by file modification time
- avoid heavy hidden UI work by making sections truly lazy
- keep behavior unchanged while improving rerun cost

## Tabs Improved

The following areas were optimized:

- `Maintenance`
- `Consumables`
- `Tower Parts`
- `Process Setup`
- `Order Draw`
- `Home`
- `Report Center`
- `Schedule`
- `Development Process`
- `SQL Lab`
- `Draw Finalize`
- `Dashboard`

## Main Optimization Patterns

### 1. Cached CSV / JSON reads

Where a tab was repeatedly reading the same file during reruns, the load path was moved behind `st.cache_data(...)` and keyed by file mtime.

This is used for:

- order tables
- inventory files
- locations files
- schedule CSVs
- recent report/export scans
- project / experiment CSVs
- manual and dataset support files

### 2. Cached directory scans

Repeated filesystem discovery was reduced with cached helpers for:

- dataset CSV lists
- recent report file lists
- manual PDF signatures
- recursive log scans

### 3. Truly lazy heavy sections

In Streamlit, `st.expander(...)` hides UI but does not automatically skip all heavy compute work.

So heavy sections were moved behind toggles where appropriate. This means the heavy code only runs when the section is actually opened.

This was applied especially in:

- Maintenance
- Consumables
- Dashboard

### 4. Maintenance-specific optimization

`Maintenance` was the biggest rerun hotspot. Improvements included:

- cached readers for maintenance CSV/state files
- cached inventory wrappers
- cached manual/BOM context builders
- cached preset library loads
- precomputed task group helpers
- faster group filtering
- true lazy rendering for hidden maintenance workflow sections

## Current Remaining Hotspots

The app is in a much better place now, but these are still the heaviest areas:

### Maintenance

Still the largest file and the largest single render scope in the app.

Remaining future gains would come from:

- splitting it into smaller modules
- caching more derived readiness maps
- reducing repeated heavy dataframe shaping in active flows

### Dashboard

The basic file I/O side is already in good shape.

The main remaining cost is:

- plot creation
- numeric transformations
- large dataframe prep for advanced analysis

### SQL Lab

Still naturally heavy because of:

- large dataset handling
- query/result shaping
- plotting

### Consumables

The biggest historical scan cost is in argon/log analytics, which is now lazy, but still heavy when opened by design.

## Design Rules For Future Changes

When adding new functionality, follow these rules:

1. If a section reads files repeatedly, cache it by file mtime.
2. If a section is optional and heavy, put it behind a toggle, not only an expander.
3. Avoid `df.apply(..., axis=1)` on large dataframes when a vectorized or precomputed helper is possible.
4. Reuse shared loaders instead of reading the same CSV in multiple places.
5. Prefer one cached discovery helper per folder instead of repeated `os.listdir(...)` / `glob(...)` calls.

## Safe Regression Check

After performance changes, run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/cli/run_app_tests.py
```

Expected current result:

- `24 PASS`
- `0 WARN`
- `0 FAIL`

## Recommended Next Step

If another performance pass is needed later, the best order is:

1. split `maintenance_tab.py`
2. profile `dashboard_tab.py`
3. profile `sql_lab.py`

That is the point where further speed work becomes structural engineering, not quick caching.
