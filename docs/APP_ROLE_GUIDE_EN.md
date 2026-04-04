# Tower App Role Guide (English)

## Introduction

This guide explains how to work with the Tower Management Software by coworker role.

The app is organized around three practical user groups:

1. `Home & Project Management`
2. `Operations`
3. `Monitoring & Research`

These groups are not only navigation labels. They describe how work should move through the app:

- plan the work
- prepare and execute the work
- analyze and improve the work

The goal is that each coworker can stay focused on the tabs that matter for their job, while the app keeps one connected workflow across orders, process setup, maintenance, parts, reports, and diagnostics.

Important working rule:

- keep role ownership aligned to the app groups
- do not mix the groups casually
- the main intentional overlap is that the project manager also handles parts orders and parts management
- diagnostics and reporting responsibility belongs to the operator side of the workflow

## App Groups Logic

### 1. Home & Project Management

Main purpose:

- daily overview
- scheduling
- draw order intake
- tower parts management

Tabs in this group:

- `Home`
- `Schedule`
- `Order Draw`
- `Tower Parts`

### 2. Operations

Main purpose:

- run the tower
- maintain the tower
- monitor live production state
- finalize real work
- diagnose the health of the app/data environment
- produce operational reports

Tabs in this group:

- `Tower state - Consumables and dies`
- `Process Setup`
- `Maintenance`
- `Dashboard`
- `Draw Finalize`
- `Data Diagnostics`
- `Report Center`

### 3. Monitoring & Research

Main purpose:

- investigate signals and datasets
- compare behavior across draws
- document development and experiments

Tabs in this group:

- `SQL Lab`
- `Development Process`

## Role 1: Operator

### Main objective

Run the tower safely and smoothly, with correct setup, correct stock awareness, and clean execution records.

### Main tabs to use

- `Tower state - Consumables and dies`
- `Process Setup`
- `Maintenance`
- `Dashboard`
- `Draw Finalize`
- `Data Diagnostics`
- `Report Center`

### What each operator tab gives you

#### `Tower state - Consumables and dies`

What it gives:

- coating levels
- die state
- temperatures
- containers
- argon/consumables visibility

Why use it:

- this is where the operator wins or loses the shift before the shift even starts

#### `Process Setup`

What it gives:

- order-based setup
- manual quick-start when there is no order
- dataset CSV creation/selection
- draw identity control before starting work

Why use it:

- this is the gate that keeps the run tied to the correct dataset and process identity

#### `Maintenance`

What it gives:

- maintenance dashboard
- day-pack prep
- schedule/forecast for maintenance
- execution records
- tests, thresholds, work package, parts/tools readiness

Why use it:

- this tab is not just maintenance logging, it is the operating system for doing maintenance correctly

#### `Dashboard`

What it gives:

- live dataset view
- signal visualization
- zone work
- process visibility during and after production

Why use it:

- this is the operator's visual truth layer for what actually happened in the run

#### `Draw Finalize`

What it gives:

- closeout of the draw
- done/fail notes
- final record quality
- downstream reporting accuracy

Why use it:

- if this tab is wrong, every report after it is weaker

#### `Data Diagnostics`

What it gives:

- app health
- path and environment checks
- startup/release/test control
- backup and support actions

Why use it:

- the operator owns the working condition of the app environment, not only the machine state

#### `Report Center`

What it gives:

- structured period reports
- production, maintenance, tests, and resource summaries
- operational output for team communication

Why use it:

- the operator closes the real work, so the operator should also own the operational reporting output

### Typical operator workflow

1. Open `Tower state - Consumables and dies`
   - verify coatings, dies, temperatures, containers, argon state
   - confirm low-stock warnings
   - check whether parts/consumables need ordering

2. Open `Process Setup`
   - prepare the run from an order, or use manual quick-start if needed
   - create/select the correct dataset CSV
   - confirm project/preform/settings before run

3. If maintenance is required, open `Maintenance`
   - check dashboard and readiness state
   - use `Prepare Day Pack`
   - use `Execute + Records`
   - record tests/measurements if the task requires them

4. Use `Dashboard`
   - inspect the dataset and visual state of the run
   - verify signals and zone-related work

5. At the end of work, use `Draw Finalize`
   - close the draw correctly
   - save notes, zones, done/fail context, and output state

6. Run `Data Diagnostics` when needed
   - verify app health
   - verify environment readiness
   - support deployment/debug workflow

7. Use `Report Center`
   - produce operational reports after the work is real and closed

### What the operator should care about most

- what is running now
- what must happen before the next run
- what is blocked by missing parts
- whether setup and consumables state are correct
- whether maintenance is ready or active
- whether the app itself is healthy enough to trust

### Operator automation already in the app

- startup checks run automatically before full app use
- path/config validation protects against wrong file locations
- local DuckDB policy avoids network locking issues
- maintenance tests can trigger follow-up actions automatically
- low-stock status can drive ordering workflow
- work-package and test data are saved automatically when executed
- diagnostics can run full readiness checks from one place
- report generation can package the operational story after execution

### Operator best practices

- do not skip `Prepare Day Pack` for maintenance that needs parts
- keep manual-start process setup only for real no-order cases
- record measured values when the task asks for them
- use `Draw Finalize` as the closing source of truth
- use `Data Diagnostics` before blaming the app for bad behavior
- use `Report Center` after work is completed, not before

## Role 2: Project Manager

### Main objective

Control project flow, order flow, planning, readiness, and reporting.

### Main tabs to use

- `Home`
- `Schedule`
- `Order Draw`
- `Tower Parts`

### What each project-manager tab gives you

#### `Home`

What it gives:

- top-level status of the tower
- what is running now
- what finished well
- what failed
- quick schedule and execution context

Why use it:

- one screen, immediate control, no guessing

#### `Schedule`

What it gives:

- the production and maintenance calendar
- past-week and next-week view
- planning collision visibility
- where capacity is going

Why use it:

- this is where planning stops being hope and becomes operational reality

#### `Order Draw`

What it gives:

- draw order creation
- order editing
- project/preform planning
- order notes and execution intent

Why use it:

- because a messy order becomes a messy process setup and a messy report

#### `Tower Parts`

What it gives:

- parts orders
- inventory
- mounted vs stock visibility
- low-stock ordering workflow
- locations and serial tracking

Why use it:

- this is a project-management tool as much as a parts tool, because timing without material readiness is fake planning

### Typical project manager workflow

1. Open `Home`
   - review live draw monitor
   - review done/failed sections
   - review high-level order and schedule indicators

2. Open `Order Draw`
   - create new draw orders
   - edit existing orders
   - track order status and notes
   - manage order builder inputs

3. Open `Schedule`
   - place draw events
   - place maintenance windows
   - check past and next week planning

4. Open `Tower Parts`
   - track parts orders
   - verify inventory readiness for upcoming work
   - monitor low stock and mounted-vs-stock visibility

### What the project manager should care about most

- order readiness
- schedule realism
- parts availability
- maintenance readiness before planned work

### Project manager automation already in the app

- maintenance preparation state can show whether jobs are ready or blocked

### Project manager best practices

- use `Order Draw` and `Schedule` together, not separately
- review `Tower Parts` before committing maintenance or production windows

## Role 3: Researcher / Process Developer

### Main objective

Understand the data, compare runs, study behavior, and document experiments in a way that stays connected to real tower work.

### Main tabs to use

- `SQL Lab`
- `Development Process`

### What each researcher tab gives you

#### `SQL Lab`

What it gives:

- deep filtering on draw datasets
- grouped parameter logic
- event-aware analysis
- overlay of maintenance/fault context when needed
- result tables for real investigation

Why use it:

- this is the strongest analysis tool in the app and the right place to ask hard questions about the process

#### `Development Process`

What it gives:

- experiment/project structure
- updates and conclusions
- file attachments
- notes with traceability

Why use it:

- because analysis without documented conclusions is just a good memory waiting to be lost

### Typical researcher workflow

1. Open `SQL Lab`
   - filter by signal groups
   - compare maintenance/fault context against draw behavior
   - use grouped parameters for zone-based analysis
   - inspect results and event time context

2. Open `Development Process`
   - register project
   - add experiments
   - attach files/notes
   - log updates and conclusions

### What the researcher should care about most

- dataset correctness
- parameter grouping accuracy
- experiment traceability
- reproducible analysis
- clear linkage between observed behavior and operational events

### Research automation already in the app

- startup checks protect against broken paths and missing assets
- SQL Lab indexing helps repeated analysis
- Development Process keeps projects and experiments linked

### Researcher best practices

- use `Development Process` to document conclusions, not personal memory
- treat SQL Lab filters as a saved analytical workflow
- keep draw interpretation linked to maintenance and fault context when relevant

## Cross-Functional Flows

## A. Draw production flow

1. `Order Draw`
2. `Schedule`
3. `Process Setup`
4. live draw + dataset capture
5. `Dashboard` / `SQL Lab` / monitoring
6. `Draw Finalize`
7. `Report Center`

## B. Maintenance flow

1. `Maintenance` Builder
2. `Prepare Day Pack`
3. `Schedule + Forecast`
4. `Execute + Records`
5. parts/tools usage + measurements
6. `Report Center`

## C. Parts flow

1. low stock / missing part detected
2. create or review order in `Tower Parts`
3. receive item
4. place item in storage location
5. move to mounted or stock state as needed
6. use in maintenance or production

## Automation and System Behavior

## Startup automation

At app startup:

- preflight checks run
- path health is validated
- config is validated
- safe-mode logic is available if needed

This prevents many silent path/config failures.

## Data and file automation

- canonical file paths are managed from `P` in `app_io.paths`
- legacy file names are redirected through compatibility logic
- diagnostics can verify whether files are readable/writable

## Maintenance automation

- preset-based tests can trigger follow-up maintenance
- threshold hits can create prep-needed or schedule events
- work packages and execution logs are saved automatically

## Report automation

- weekly report flow can generate PDFs
- Report Center can generate custom period reports
- release/deploy checks can generate support artifacts

## Diagnostics automation

From `Data Diagnostics`, the app can run:

- app tests
- path audit
- environment pretest
- release check
- full health
- backup/export actions

## Who Should Use Which Group First

- Operator -> start in `Operations`
- Project Manager -> start in `Home & Project Management`
- Researcher -> start in `Monitoring & Research`

This is the cleanest mental model for the app.

## Troubleshooting by Role

### Operator

If something feels wrong:

- check consumables state
- check schedule
- check maintenance readiness
- check whether inventory is blocking work

### Project Manager

If planning looks wrong:

- check order data
- check part status
- check maintenance blockers
- check diagnostics before assuming logic is broken

### Researcher

If results look wrong:

- verify dataset selection
- verify SQL filter group definition
- verify event time context
- verify whether maintenance/fault overlays are relevant

## Final recommendation

The app works best when each coworker uses the right group first, but still respects the full workflow:

- operators execute
- project managers coordinate
- researchers explain and improve

That is the intended logic of the software.

And yes, it is a strong system now. The app is not a pile of tabs anymore. It has clear ownership, clear workflow, and clear responsibility lines. That is exactly why it works.
