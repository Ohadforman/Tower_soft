# Tower Rebuild Migration Blueprint

This rebuild is the safe mirror of the production Streamlit Tower app.

Goal:
- keep the existing app functional as the source of truth
- rebuild page-by-page here with a stronger UI and no Streamlit dependency
- verify behavior here first
- only then port the proven UX decisions back into the main product if needed

## Source Of Truth

Real app navigation and routing:
- `/Users/ohadformanair/PycharmProjects/Tower_work/app/navigation.py`
- `/Users/ohadformanair/PycharmProjects/Tower_work/app/router.py`

Real app tab implementations:
- `/Users/ohadformanair/PycharmProjects/Tower_work/renders/tabs/`

Rebuild app:
- `/Users/ohadformanair/PycharmProjects/Tower_work/tower_rebuild/`

## Real App Surface Map

### Home & Project Management
- `Home`
- `Schedule`
- `Order Draw`
- `Tower Parts`

### Operations
- `Tower State - Consumables and Dies`
- `Process Setup`
- `Maintenance`
- `Dashboard`
- `Draw Finalize`
- `Tower Parts`
- `Data Diagnostics`
- `Report Center`

### Monitoring & Research
- `SQL Lab`
- `Development Process`

## Complexity Snapshot

Largest pages in the current app by file size:
- `maintenance_tab.py`: 8982 lines
- `tower_parts_tab.py`: 3123 lines
- `sql_lab.py`: 2421 lines
- `report_center_tab.py`: 1674 lines
- `dashboard_tab.py`: 1253 lines
- `home_tab.py`: 1240 lines
- `consumables_tab.py`: 1213 lines
- `order_draw_tab.py`: 1160 lines
- `development_process_tab.py`: 1145 lines

Interpretation:
- `Maintenance` is its own migration program, not a single quick page port
- `Tower Parts`, `Order Draw`, and `Dashboard` are the best next functional migrations after the current base
- `SQL Lab` should be treated as a specialized tool migration, not a normal page redesign

## Rebuild Status

### Already rebuilt with real UI work
- `Home`
- `Schedule`
- `Tower Parts`
- `Maintenance`
- `Order Draw`
- `Dashboard`
- `Draw Finalize`
- `Data Diagnostics`
- `Report Center`
- `SQL Lab`
- `Development Process`
- `Process Setup`
- `Tower State - Consumables and Dies`

### Still needing deeper parity work
- `Maintenance`
- `Dashboard`
- `Draw Finalize`
- `SQL Lab`
- `Development Process`
- `Process Setup`
- `Home`

## Current Highest-Gap Areas

1. `Maintenance`
- builder/manual flow still needs deeper parity
- faults and correlation lanes need stronger real actions

2. `Dashboard`
- math lab still needs more of the real advanced behavior
- save workflow and zone workflow can still get closer to production

3. `Draw Finalize`
- core flow is there now
- still needs more production-grade guidance and edge-case handling

4. `SQL Lab`
- strong already, but still needs more exact event/draw investigation depth

## Rules For Safe Migration

1. Do not change the Streamlit page first.
2. Read the real page behavior and data dependencies.
3. Rebuild the workflow here with the new UI.
4. Test it against real data here.
5. Only after this version feels correct do we consider changing the original app.

## Recommended Migration Order

### Phase 1: Core operational flow
1. `Order Draw`
2. `Tower Parts`
3. `Dashboard`
4. `Data Diagnostics`

### Phase 2: Operational depth
1. `Consumables and Dies`
2. `Draw Finalize`
3. `Report Center`

### Phase 3: Heavy systems
1. `Maintenance`
2. `SQL Lab`
3. `Development Process`

## Why This Order

- `Order Draw` is central to the tower workflow and should become the next serious page.
- `Tower Parts` and `Dashboard` connect directly to daily operations and are highly visible.
- `Data Diagnostics` is safer to rebuild early because it helps us trust incoming data.
- `Maintenance` is the biggest and riskiest page, so it should come after the shared UI system is stronger.

## Immediate Next Build

### Next page to rebuild here
- `Order Draw`

### What to extract from the real app first
- filters and workflow stages
- order queue structure
- actions and status transitions
- data files / helpers it depends on
- summary widgets that operators actually use

### Target outcome in rebuild
- no Streamlit
- clearer layout
- calmer default state
- foldable heavy lists
- strong operational scanning
- same behavior as the current app
