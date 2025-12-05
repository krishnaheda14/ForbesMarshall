# Strategic Machine Scheduling and Resource Optimization

A full-stack decision-support system for CNC job-shop scheduling that integrates rule-based heuristics, genetic algorithms, cost and outsourcing analysis, and interactive visualization. The system helps production planners minimize tardiness and cost, simulate breakdowns, evaluate make-or-buy decisions, and experiment with capacity changes.

---

## 1. Problem Statement and Objectives

Modern CNC job-shops face:

- Highly variable demand and routing (multiple operations per job).
- Conflicting objectives: due-date adherence, machine utilization, and outsourcing cost.
- Limited visibility into the impact of breakdowns, new machines, and job priorities.

**Objectives:**

1. Build a scheduling engine that generates feasible schedules for multi-operation CNC jobs.
2. Support multiple heuristics (SPT, EDD, CR, Priority) and a Genetic Algorithm (GA) optimizer.
3. Quantify performance via KPIs: tardiness, makespan, utilization, on-time percentage, etc.
4. Analyze outsourcing vs in-house decisions, machine ROI, and capacity expansion (new machines).
5. Provide an interactive web UI (Gantt chart, tables, dashboards) for planners.
6. Allow “what-if” experiments (due date shifts, breakdowns, machine additions/removals).

---

## 2. High-Level Architecture

The system follows a **client–server** architecture:

- **Frontend:** React + Vite + Material-UI, Zustand store, Plotly for Gantt charts.
- **Backend:** FastAPI (Python), Pandas-based data layer, custom scheduling core.
- **Data:** CSV files for jobs, machines, vendors, and material changeovers.

### 2.1 Component Overview

- **Backend (`backend/`)**
  - `main.py`: FastAPI application defining REST endpoints for:
    - Data loading/unloading.
    - Schedule computation and application.
    - Genetic Algorithm optimization.
    - Breakdown simulation.
    - Machine add/remove (currently UI for add removed).
    - Outsourcing analysis and machine ROI.
    - Due date adjustment and metrics computation.
  - `cnc_scheduler_core.py`: Core rule-based scheduling logic and metrics:
    - Builds operation–machine time matrix (`df_effective`).
    - Chooses eligible machines for each operation.
    - Assigns operations to time intervals (Start_Time, End_Time).
    - Computes metrics such as makespan and tardiness.
  - Excel ingestion (for advanced flows):
    - `excel_ingestion.py`, `schema_mapping.py`, `data_transformer.py`, `models.py`.

- **Frontend (`frontend/`)**
  - `src/pages/Dashboard.jsx`: Central landing page (load data, compute/apply heuristics, see KPIs).
  - `src/pages/GanttView.jsx`: Interactive Gantt chart of scheduled operations.
  - `src/pages/OperationStatus.jsx`: Detailed table with CR and Tardiness breakdown per operation (with calculation dialogs).
  - `src/pages/Comparison.jsx`: Heuristic comparison, metrics vs heuristic.
  - `src/pages/GAOptimizer.jsx`: GA-based optimization UI with evolution plots.
  - `src/pages/CostAnalysis.jsx`: Hourly rate vs cost, outsourcing analysis.
  - `src/pages/OutsourcingAnalysis.jsx`: Make-or-buy analysis for operations.
  - `src/pages/CapexAnalysis.jsx`: CapEx recommendations and ROI (Buy action currently hidden in UI).
  - `src/pages/Settings.jsx`: System information, activity log, reset & remove-machine.
  - `src/store/useSchedulerStore.js`: Central client-side state (current heuristic, schedule, metrics, GA results).
  - `src/services/api.js`: Axios API client for backend endpoints.

- **Data (`data/`)**
  - `jobs_dataset.csv`: Jobs with operations, quantities, routing sequence, release and due days, flags for outsourcing.
  - `machine_data.csv`: Machine IDs, types, eligible operation types, speed factors, and cost parameters.
  - `vendor_data.csv`: Vendor rates for outsourced operations.
  - `previous_next_material.csv`: Material changeover / setup penalties.

---

## 3. Data Model and Assumptions

### 3.1 Jobs and Operations (`jobs_dataset.csv`)

Key columns:

- `Job_ID`: Unique job identifier (e.g., `J101`).
- `Operation_ID`: Operation identifier (e.g., `J101_Op1`).
- `Op_Seq`: Sequence number of the operation within the job.
- `Op_Type`: Type of operation (MILLING, TURNING, GRINDING, DRILLING, etc.).
- `Quantity`: Number of units to process.
- `Proc_Time_per_Unit`: Time per unit (in minutes, after conversion).
- `Setup_Time`: Setup time for the operation (minutes).
- `Transfer_Min`: Transfer / handling time between operations (minutes).
- `Release_Day`, `Due_Day`: Release and due dates in days.
- `Priority`: Job priority (1 = highest, 3 = lowest).
- `Outsource_Flag`: Indicates if the operation is typically outsourced.
- `Vendor_Ref`: Vendor reference for outsourced work.

Derived columns in the backend:

- `Total_Proc_Min` = `Quantity * Proc_Time_per_Unit`.
- `Release_Time_Min` = `Release_Day * 1440`.
- `Due_Time_Min` = `Due_Day * 1440`.

### 3.2 Machines (`machine_data.csv`)

Key columns:

- `Machine_ID`: e.g., `M1`, `M2`.
- `Machine_Type`: e.g., MILLING, TURNING, GRINDING, DRILLING, combinations.
- `Op_Types`: Comma-separated operation types that machine can process.
- `Speed_Factor`: Relative processing speed (1.0 = baseline).
- Cost parameters: Hourly rate, maintenance, energy cost, purchase price.

### 3.3 Outsourcing and Vendors (`vendor_data.csv`)

- Vendor-specific rates for outsourced operations (per hour/per part).
- Used for make-or-buy comparisons and ROI analysis.

---

## 4. Scheduling Logic and Algorithms

### 4.1 Rule-Based Heuristics

The core scheduler in `cnc_scheduler_core.py` supports:

- **SPT (Shortest Processing Time)**:
  - Orders operations by ascending `Total_Proc_Min`.
- **EDD (Earliest Due Date)**:
  - Orders operations by ascending `Due_Time_Min`.
- **CR (Critical Ratio)**:
  - $CR = \frac{\text{Due\_Time\_Min} - \text{Release\_Time\_Min}}{\text{Total\_Proc\_Min}}$  
  - Lower CR indicates more urgent.
- **PRIORITY**:
  - Respects job-level priority; within same priority, other secondary rules apply.

Scheduling steps:

1. Build a list of available operations considering precedence constraints (`Op_Seq`).
2. For each available operation:
   - Find eligible machines via `get_eligible_machines(op_type)` using `machine_data.csv`.
   - Compute effective processing times (considering `Speed_Factor` and setup/transfer).
3. Schedule the operation on the selected machine at the earliest feasible time respecting:
   - Machine availability.
   - Operation precedence within a job.
4. Record:
   - `Start_Time`, `End_Time`, `Machine_ID`, `Proc_Time`, `Assignment_Type` (IN-HOUSE / OUTSOURCE).

### 4.2 Genetic Algorithm Optimizer

`GAOptimizer.jsx` + `run_ga_optimization` (backend):

- Encodes a schedule as a chromosome (operation sequences / machine assignments).
- Uses:
  - Selection, crossover, mutation operators.
  - Fitness function combining objectives (e.g., tardiness and makespan).
- Produces:
  - Best-found schedule (operations with Start/End/Machine).
  - Evolution history (fitness vs generation).
  - Final KPIs (tardiness, makespan, utilization).
- Frontend:
  - Displays performance evolution and final Gantt chart.
  - Persists GA results in the store so they survive tab changes.

### 4.3 Metrics and Objectives

Backed by `calculate_metrics` and related functions:

- **Makespan (days)**: $(\max(\text{End\_Time}) - \min(\text{Start\_Time}))/1440$.
- **Total tardiness (days)**: 
  - $T = \sum \max(0, \text{End\_Time} - \text{Due\_Time\_Min}) / 1440$.
- **On-time percentage**:
  - $\% = 100 \cdot \frac{\#\{\text{operations with } \text{End\_Time} \le \text{Due\_Time\_Min}\}}{\text{total operations}}$.
- **Machine utilization**:
  - Ratio of busy time to available horizon per machine.

The backend selects the “best” heuristic based primarily on **tardiness minimization**, with makespan as a tiebreaker.

---

## 5. System Workflow and Sequence

This describes a typical end-to-end usage scenario.

### 5.1 High-Level Sequence (Textual)

1. **User Action**: From `Dashboard`, clicks “Load Data”.
2. **Frontend → Backend**: `POST /api/data/load`.
3. **Backend**:
   - Loads all CSVs (`jobs_dataset.csv`, `machine_data.csv`, etc.).
   - Computes derived columns (`Total_Proc_Min`, `Release_Time_Min`, `Due_Time_Min`).
   - Builds in-memory dataframes for subsequent scheduling.
4. **User Action**: Clicks “Compute All Heuristics”.
5. **Frontend → Backend**: `POST /api/schedule/compute-all`.
6. **Backend**:
   - For each heuristic (SPT, EDD, CR, PRIORITY), runs the scheduler.
   - Computes KPIs and stores results in memory.
   - Selects best heuristic based on total tardiness and makespan.
7. **User Action**: Applies chosen heuristic.
   - Frontend calls `POST /api/schedule/apply`.
8. **Backend**: Sets active schedule, returns operations with Start/End/Machine.
9. **Frontend**:
   - Updates store state with `currentHeuristic`, `currentSchedule`.
   - `Dashboard`, `GanttView`, and `OperationStatus` pages render KPIs and charts.
10. **User Action**: Navigates to `OperationStatus` and `GanttView`:
    - Can click an operation to inspect CR or tardiness calculations.
    - Can adjust due dates for a job via the tardiness dialog (which calls `/api/data/adjust-due-date` and recomputes schedules).
11. **User Action**: Navigates to `CostAnalysis` / `OutsourcingAnalysis` / `MachineROI`:
    - Backend recomputes cost and ROI metrics based on the current schedule.
12. **User Action** (optional): Uses `GAOptimizer` to run a genetic algorithm optimization, compares the GA schedule to heuristic schedules.

### 5.2 Simplified Sequence Diagram (Text Form)

Actors: **User**, **React Frontend**, **FastAPI Backend**, **Scheduler Core**

1. User → Frontend: Click “Load Data”.
2. Frontend → Backend: `POST /api/data/load`.
3. Backend → Scheduler Core: Initialize dataframes.
4. Scheduler Core → Backend: Return prepared data.
5. User → Frontend: Click “Compute All Heuristics”.
6. Frontend → Backend: `POST /api/schedule/compute-all`.
7. Backend → Scheduler Core: Run heuristic scheduling loops.
8. Scheduler Core → Backend: Schedules + metrics per heuristic.
9. Backend → Frontend: Summary metrics; best heuristic suggestion.
10. User → Frontend: Click “Apply Heuristic”.
11. Frontend → Backend: `POST /api/schedule/apply`.
12. Backend → Frontend: Active schedule (operations).
13. Frontend: Updates `Dashboard`, `GanttView`, `OperationStatus`.
14. User: Runs GA optimizer (optional), cost/outsourcing analysis (optional).

---

## 6. Frontend Implementation Details

Key frontend features:

- **State Management (`useSchedulerStore.js`)**:
  - Stores:
    - `currentHeuristic`, `currentSchedule`.
    - Metrics and heuristic comparison data.
    - GA results and evolution history (persisted).
  - Actions:
    - `setCurrentSchedule`, `setMetrics`, `setCurrentHeuristic`, GA setters/clearers.

- **Gantt Chart (`GanttView.jsx`)**:
  - Uses Plotly to render machine-wise bars:
    - X-axis: time (minutes or converted to hours/days).
    - Y-axis: machine lanes.
    - Bar segments colored by operation or assignment type.
  - Shows breakdowns (maintenance/outsourcing) as separate traces.

- **Operation Status (`OperationStatus.jsx`)**:
  - MUI `Table` with:
    - Columns: Job_ID, Operation_ID, Machine, Priority, Proc_Time, CR, Release, Start, End, Due_Time, Tardiness, Status.
    - Sorting, paging, search.
  - CR Dialog:
    - Shows CR formula and computed values from schedule data.
  - Tardiness Dialog:
    - Shows tardiness formula: `max(0, End_Time - Due_Time_Min)`.
    - Allows adjusting due date by a delta in days (backend recomputes schedule).

- **GA Optimizer (`GAOptimizer.jsx`)**:
  - Form inputs for GA parameters: population size, generations, crossover/mutation rates, objective mode.
  - Calls backend `POST /api/schedule/cpsat` or GA endpoint (depending on actual implementation file).
  - Renders:
    - Fitness evolution chart.
    - Final schedule Gantt.
    - Detailed GA metrics.

- **Cost & Outsourcing (`CostAnalysis.jsx`, `OutsourcingAnalysis.jsx`, `MachineROI.jsx`)**:
  - Fetch cost metrics from backend.
  - Visualize:
    - Cost vs hourly rate.
    - In-house vs outsource breakdown.
    - Machine ROI metrics.

---

## 7. Backend Implementation Details

### 7.1 API Endpoints (selected)

- `POST /api/data/load`:
  - Loads CSVs into memory and prepares derived columns.
- `POST /api/data/unload`:
  - Clears in-memory dataframes.
- `GET /api/data/info`:
  - Returns summary: number of jobs, operations, machines.

- `POST /api/schedule/compute`:
  - Body: `{ "heuristic": "SPT" | "EDD" | "CR" | "PRIORITY" }`.
  - Returns schedule + metrics for that heuristic.

- `POST /api/schedule/compute-all`:
  - Returns metrics for all supported heuristics; selects a best heuristic.

- `POST /api/schedule/apply`:
  - Sets active schedule to chosen heuristic’s result.

- `GET /api/schedule/current`:
  - Returns currently active schedule.

- `POST /api/machine/breakdown`:
  - Simulates a breakdown for a machine across a time range.

- `POST /api/analysis/machine-roi`:
  - Analyzes machine-level ROI using in-house vs outsourcing cost.

- `POST /api/data/adjust-due-date`:
  - Body: `{ "job_id": "J101", "delta_days": -2 }`.
  - Adjusts `Due_Day` and `Due_Time_Min` for that job, recomputes schedules and metrics, and persists back to `jobs_dataset.csv` when possible.

- `POST /api/machines/add` / `POST /api/machines/remove`:
  - Adjust machine set and recompute schedules (Add Machine UI currently hidden).

### 7.2 Error Handling and Robustness

- Defensive handling of missing columns (e.g., `Priority`) in merges.
- Consistent time units:
  - All internal scheduling uses minutes.
  - One day = 1440 minutes.
- Dynamic machine eligibility:
  - Uses `machine_data.csv` to determine which machines can process which operations.

---

## 8. Setup and Run Instructions

### 8.1 Backend (FastAPI)

From project root:

```powershell
# 1. Create and activate virtual environment (once)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Run backend (FastAPI with Uvicorn)
uvicorn main:app --reload --port 8001
```
