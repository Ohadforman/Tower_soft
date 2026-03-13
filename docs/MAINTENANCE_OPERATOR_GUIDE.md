# Maintenance Operator Guide

## Purpose

This guide explains how to work in the Maintenance tab as an operator or maintenance lead.

The target workflow is simple:

1. Build the maintenance task correctly once.
2. Prepare the job with the right parts/tools and manual pages.
3. Schedule it at the right time.
4. Execute it with safety, measurements, and records.
5. Let the app update history, inventory, and follow-up actions.

## Main Flow

The active maintenance flow in the app is:

1. `0) Builder (Tasks + BOM)`
2. `1) Prepare Day Pack`
3. `2) Schedule + Forecast`
4. `3) Execute + Records`

Use the steps in that order unless you are only reviewing status.

## What the App Is Managing for You

The Maintenance tab is now designed to manage one continuous workflow:

1. Define the task correctly.
2. Define the parts and tools needed.
3. Define the work package and test logic.
4. Prepare the job before the work window.
5. Execute it with live records.
6. Save the result back into task history, parts flow, and reports.

The goal is to avoid separate spreadsheets, memory-based tracking, and last-minute missing-part surprises.

## Dashboard

At the top of the Maintenance tab:

- `OVERDUE`, `DUE SOON`, `ROUTINE`, `OK`
  Shows current task status distribution.
- `Open Faults`, `Critical Open`
  Shows live fault pressure.
- `In Progress`, `Blocked (Parts)`, `Prep Ready`
  Shows current execution readiness.
- `Test Monitor`
  Shows recent saved maintenance tests, threshold hits, and condition-met results.

Use the dashboard first to understand what needs attention now.

## Step 0: Builder (Tasks + BOM)

Use Builder to define the task correctly once.

### What to set in Builder

- Task groups
  Example: `Weekly`, `3-Month`, `Hours`, `Draw-Count`, `Test`
- Timing and triggers
  Example: hours, draws, calendar, or two triggers together
- Required parts
  Consumables or replaceable items needed for the task
- Required tools
  Non-consumable tools needed to perform the task
- Manual page pinning
  Exact manual page for BOM items or task context
- Work package
  Preparation, safety, procedure, stop plan, completion criteria
- Test + condition config
  Measured values, thresholds, and follow-up action for test-based tasks

### Builder checklist

Before a task is considered ready, confirm:

- the task belongs to the right group
- the trigger basis is correct
- the parts list is realistic
- the tools list is realistic
- the work package explains how to do the job
- the manual context is pinned correctly
- the test preset is attached if the task requires measured values

If one of these is missing, the task may still appear in schedule, but it will not be operationally ready.

### Test + Condition Capture

Use this when the maintenance task requires measured values or pass/fail logic.

Examples:

- `Voltage (V)`
- `Airflow (m/s)`
- `X offset (mm)`
- `Tare error`
- `Pressure Rise (bar)`

Each task can define:

- `Test_Preset`
- `Test_Fields`
- `Test_Thresholds`
- `Test_Condition`
- `Test_Action`

If a task is only a standard maintenance action and does not need measured data, leave these blank.

### Central Preset Library

Inside Builder there is a central preset editor:

- `⚙️ Preset Library (central)`

Use it to manage reusable test templates in one place.

Saved file:

- [maintenance_test_presets.json](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_test_presets.json)

Current tower-oriented presets include:

- Furnace Heating Element Voltage Trend
- Pyrometer Alignment + Window Cleanliness
- Storage Vacuum Rise
- Clean-Air Velocity
- Interlocks Test
- X-Y Alignment
- Fibre Position Centering
- Tension Gauge Calibration
- Bearing Play + Rotation
- Top Cap Erosion
- Bottom Door Distortion

Use the preset library when you want one rule to stay consistent across many tasks.

Examples:

- one voltage tolerance used everywhere
- one vacuum-rise threshold used everywhere
- one airflow range used everywhere

If you update the central preset, use it as the source of truth and then re-check the tasks that use it.

## Step 1: Prepare Day Pack

Use this at the start of a maintenance shift or before a scheduled maintenance window.

### What to check

- Which tasks are due today or in the chosen package
- Parts readiness
- Work package exists and is usable
- Missing parts need orders
- Intake flow to Tower Parts is ready

### Expected result

By the end of this step, each task should be one of:

- `PREP_READY`
- `BLOCKED_PARTS`
- still pending if not yet prepared

### What good preparation looks like

A well-prepared task should answer these questions before anyone starts work:

- Do we have the needed parts in stock?
- Do we have the needed tools?
- Do we know the manual page and procedure page?
- Do we know whether the draw must stop?
- Do we know the expected safety restrictions?
- Do we know whether the task is inspection-only or likely to consume parts?

## Step 2: Schedule + Forecast

Use this when you want to place maintenance into real available windows.

### What this step is for

- plan by urgency
- plan by task group
- align with workdays and preferred maintenance days
- see future due pressure by hours, draws, and calendar

### Good practice

- Use `Prepare Day Pack` first for jobs that need parts.
- Use `Schedule + Forecast` after BOM and work package are already correct.

### Scheduling intent

Use this step for two different kinds of work:

- recurring planned maintenance
- follow-up work created by inspection/test results

This is important because some jobs should only be scheduled after a condition or test confirms they are truly needed.

## Step 3: Execute + Records

This is the live execution workspace.

### Inside the task workspace

You can:

- see required parts
- see required tools
- open manual context
- reserve/release parts
- view work package
- start the task
- capture measurements/tests
- mark inspection-only or replacement-needed for conditional tasks
- apply done updates

### Execution statuses you will see

- `IN_PROGRESS`
  The task has started and is currently being worked.
- `PREP_READY`
  The task is prepared and waiting for execution.
- `BLOCKED_PARTS`
  The task is known but cannot continue until required parts are available.
- `PREP_NEEDED`
  The task still needs preparation or a follow-up condition triggered it.
- `DONE`
  Work was completed and recorded.

### Test + Condition Capture during execution

If the task has test fields defined, the operator will see:

- the preset name
- the fields to capture
- the threshold table
- the condition text
- the action text

Result choices:

- `Auto from thresholds`
- `Monitor only`
- `Condition met`
- `Condition not met`

If thresholds are hit, the app can:

- save the measurement result
- set `PREP_NEEDED`
- create a maintenance schedule event immediately

### Saved test log

All execution-time test results are written to:

- [maintenance_test_records.csv](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_test_records.csv)

Each row stores:

- timestamp
- task ID
- component
- task
- preset used
- result mode
- whether condition was met
- whether thresholds were auto-hit
- threshold hit labels
- measured values
- notes
- actor

Use this log as the measured-data history for maintenance.

Typical examples:

- voltage trend records over time
- airflow checks after service
- tension calibration checks
- vacuum rise observations
- X/Y offset measurements after alignment work

## Task Types

There are two practical types of maintenance tasks:

### 1. Standard maintenance tasks

Examples:

- replace a filter
- grease a bearing
- clean a surface
- inspect a pulley

These may only need:

- groups
- timing
- BOM
- work package

### 2. Test / monitoring tasks

Examples:

- voltage trend check
- vacuum rise check
- interlocks test
- clean-air velocity check
- tension gauge calibration

These should also define:

- test fields
- thresholds
- condition text
- action text

If the test is important enough to exist on its own, create it as a real maintenance task and give it a group such as:

- `Test`
- `Weekly`
- `3-Month`
- `Hours`

This is the recommended rule:

- if the measurement is only a check inside another task, keep it inside that task
- if the measurement is a repeating inspection with its own cadence, make it its own task

## Conditional Tasks

Some tasks are inspection-first, then replacement only if needed.

Examples:

- inspect cap erosion
- inspect bottom door distortion
- inspect pyrometer window

For these:

1. Use the task to inspect and measure.
2. Record the measured values.
3. If condition is met, move to:
   - `PREP_NEEDED`
   - `BLOCKED_PARTS`
   - or schedule the follow-up maintenance

Do not force every inspection to consume parts. Only consume parts when replacement was actually done.

## Parts and Tools

The app separates:

- `Required_Parts`
  Consumables or replaceable items
- `Required_Tools`
  Non-consumable tools needed for the job

During execution:

- parts can be reserved/released/consumed
- tools are checked for availability but are not consumed

### Practical rule

- `Required_Parts` means the job may consume or replace something
- `Required_Tools` means the job needs access to something reusable

Examples:

- bearing set -> part
- seal kit -> part
- IPA wipes / cleaning cloth -> tool/support consumable according to your tower workflow
- gauge / wrench / alignment tool -> tool

## Safety

The work package includes:

- preparation checklist
- safety protocol
- procedure
- draw stop plan
- completion criteria

High fall-risk tasks can automatically force:

- `NO ENTRY (high fall risk)`

This safety state must be respected in the real maintenance workflow.

### Safety protocol guidance

Use the Safety section to describe:

- PPE
- fall-risk restrictions
- guarding/rail rules
- chemical handling
- lockout / isolation requirements
- access restrictions

Do not use Safety to store the procedure itself. Keep procedure steps in the Procedure section.

## Recommended Operator Workflow

### For a recurring maintenance job

1. Open Builder.
2. Confirm groups, timing, BOM, tools, work package.
3. Add test preset if the task needs measured values.
4. Save.
5. Go to Prepare Day Pack and verify parts/tools readiness.
6. Go to Schedule + Forecast and place the task.
7. On the work day, open Execute + Records.
8. Reserve parts if needed.
9. Start task.
10. Record measured values and threshold result if relevant.
11. Mark done.

### For a condition-driven job

1. Open the task in Execute + Records.
2. Record measured values or inspection result.
3. If threshold/condition is met:
   - set `PREP_NEEDED`
   - schedule the follow-up
   - order missing parts if needed
4. Complete when real maintenance is finished.

### For a test-only check

1. Open the task.
2. Enter measured values.
3. Use `Auto from thresholds` if the task has a defined limit.
4. Review whether the app triggered a condition.
5. If no action is needed, save as monitoring history only.

## Data Files Used by This Flow

- [maintenance_work_packages.csv](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_work_packages.csv)
- [maintenance_task_state.csv](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_task_state.csv)
- [maintenance_actions_log.csv](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_actions_log.csv)
- [maintenance_parts_reservations.csv](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_parts_reservations.csv)
- [maintenance_test_records.csv](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_test_records.csv)
- [maintenance_test_presets.json](/Users/ohadformanair/PycharmProjects/Tower_work/maintenance/maintenance_test_presets.json)

These files are operational data. They should stay consistent with the app workflow and should not be edited casually outside the app unless you are doing controlled maintenance administration.

## Good Operating Rules

- Build the task once, execute many times.
- Keep threshold logic in the preset library, not scattered manually.
- Use tests only where measured data is actually useful.
- Use task groups to control when work appears.
- Use `Execute + Records` as the single source for real task execution and measurement logging.
- Keep parts and tools accurate, because readiness depends on them.

## If Something Looks Wrong

Check these first:

1. Is the task BOM correct?
2. Is the preset selected correctly?
3. Are thresholds realistic?
4. Did the task get the right group/timing?
5. Did the measurement values save in `maintenance_test_records.csv`?
6. Is the maintenance schedule event created when condition is met?

If not, fix it from Builder first, then re-run the task flow.

## Related Documents

- [Operations Runbook](/Users/ohadformanair/PycharmProjects/Tower_work/docs/OPERATIONS.md)
- [Architecture](/Users/ohadformanair/PycharmProjects/Tower_work/docs/ARCHITECTURE.md)
- [V2 Deploy Protocol](/Users/ohadformanair/PycharmProjects/Tower_work/docs/V2_DEPLOY_PROTOCOL.md)
