"""CP-SAT based advanced scheduler.

This module provides an exact/near-exact optimization approach using
Google OR-Tools CP-SAT solver. It supports:
  - Machine assignment (optional intervals per machine)
  - Non-overlap of operations on same machine
  - Release times
  - Precedence constraints (Op_Seq within Job_ID)
  - Tardiness variables
  - Makespan (max end time) variable
  - Multiple objective modes

Objective modes:
  * 'min_tardiness'            -> Minimize total tardiness
  * 'min_weighted'             -> Minimize total tardiness + alpha * makespan
  * 'min_makespan'             -> Minimize makespan
  * 'lex_tardiness_makespan'   -> Lexicographic (total tardiness, makespan)

Return format aligns with CNCScheduler: list of dicts with timing and machine.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from ortools.sat.python import cp_model
import pandas as pd

DEFAULT_ALPHA = 0.1  # weight for makespan in 'min_weighted'

class CPSATSchedulerResult:
    def __init__(self, schedule: List[Dict[str, Any]], objective_value: int, status: str, solver_stats: Dict[str, Any]):
        self.schedule = schedule
        self.objective_value = objective_value
        self.status = status
        self.solver_stats = solver_stats


def solve_with_cpsat(
    df_ops: pd.DataFrame,
    df_machines: pd.DataFrame,
    df_effective: pd.DataFrame,
    df_penalties: pd.DataFrame = None,
    objective_mode: str = 'min_weighted',
    alpha: float = DEFAULT_ALPHA,
    time_limit_seconds: int = 30,
    log: bool = False
) -> CPSATSchedulerResult:
    """Solve scheduling problem with CP-SAT.

    Args:
        df_ops: Operations dataframe (Job_ID, Operation_ID, Op_Seq, Release_Time_Min, Due_Time_Min, Setup_Time, Transfer_Min, Mat_Type, Priority, Assignment_Type)
        df_machines: Machines dataframe (Machine_ID, optional Maintenance_Window)
        df_effective: Mapping of Operation_ID -> Machine_ID with Effective_Proc_Time
        df_penalties: Material change penalties table (used for sequence-dependent setup)
        objective_mode: One of supported objective modes
        alpha: Makespan weight when using 'min_weighted'
        time_limit_seconds: solver wall-clock limit
        log: if True, prints solver log

    Returns:
        CPSATSchedulerResult
    """
    model = cp_model.CpModel()

    # Filter in-house operations (OUTSOURCE already fixed) and sort by sequence
    ops = df_ops[df_ops.get('Assignment_Type', 'IN_HOUSE') != 'OUTSOURCE'].copy()
    ops.sort_values(['Job_ID', 'Op_Seq'], inplace=True)

    # Build eligibility map
    eligible_map: Dict[str, List[Tuple[str, int]]] = {}
    for _, row in df_effective.iterrows():
        op_id = row['Operation_ID']
        machine_id = row['Machine_ID']
        eff_time = int(row['Effective_Proc_Time'])  # assume already minutes
        eligible_map.setdefault(op_id, []).append((machine_id, eff_time))

    # Variables containers
    start_vars: Dict[str, cp_model.IntVar] = {}
    end_vars: Dict[str, cp_model.IntVar] = {}
    tardiness_vars: Dict[str, cp_model.IntVar] = {}
    # Optional intervals per (op,machine)
    interval_vars: Dict[Tuple[str, str], cp_model.IntervalVar] = {}
    presence_vars: Dict[Tuple[str, str], cp_model.BoolVar] = {}

    HORIZON = int(1.2 * (ops['Total_Proc_Min'].sum() + 1)) if 'Total_Proc_Min' in ops.columns else 100000
    # Fallback horizon if missing totals
    if HORIZON <= 0:
        HORIZON = 100000

    # Create per-operation aggregated start/end tied through selected machine assignment
    for _, op in ops.iterrows():
        op_id = op['Operation_ID']
        release = int(op.get('Release_Time_Min', 0))
        due = int(op.get('Due_Time_Min', HORIZON))
        # Setup + transfer will be added per machine interval
        start_var = model.NewIntVar(release, HORIZON, f'start_{op_id}')
        end_var = model.NewIntVar(release, HORIZON, f'end_{op_id}')
        tardiness = model.NewIntVar(0, HORIZON, f'tardiness_{op_id}')
        start_vars[op_id] = start_var
        end_vars[op_id] = end_var
        tardiness_vars[op_id] = tardiness

        # Eligible machines
        candidates = eligible_map.get(op_id, [])
        if not candidates:
            # If no machine can run it, force tardiness large and skip intervals
            model.Add(end_var == start_var)
            model.Add(tardiness == due)  # penalize
            continue

        chosen_end_exprs = []
        presence_list = []

        for machine_id, eff_time in candidates:
            setup_time = int(op.get('Setup_Time', 0))
            transfer_time = int(op.get('Transfer_Min', 0))
            prev_mat = op.get('Mat_Type', None)
            # Optional: sequence-dependent penalty (simplified = 0; integrate if needed)
            penalty = 0
            duration = setup_time + eff_time + transfer_time + penalty
            presence = model.NewBoolVar(f'pres_{op_id}_{machine_id}')
            interval = model.NewOptionalIntervalVar(start_var, duration, end_var, presence, f'int_{op_id}_{machine_id}')
            interval_vars[(op_id, machine_id)] = interval
            presence_vars[(op_id, machine_id)] = presence
            chosen_end_exprs.append(presence)
            presence_list.append(presence)

        # Exactly one machine assignment
        model.Add(sum(presence_list) == 1)

        # Tardiness definition
        model.Add(tardiness >= end_var - due)
        model.Add(tardiness >= 0)

    # Precedence constraints: end(prev) <= start(next)
    for (job_id, job_group) in ops.groupby('Job_ID'):
        job_sorted = job_group.sort_values('Op_Seq')
        prev_op_id = None
        for _, op in job_sorted.iterrows():
            if prev_op_id is not None:
                model.Add(end_vars[prev_op_id] <= start_vars[op['Operation_ID']])
            prev_op_id = op['Operation_ID']

    # Machine no-overlap: gather intervals per machine
    for machine_id in df_machines['Machine_ID']:
        machine_ints = [interval_vars[(op_id, m_id)]
                        for (op_id, m_id), interval in interval_vars.items()
                        if m_id == machine_id]
        if machine_ints:
            model.AddNoOverlap(machine_ints)

    # Makespan var
    makespan = model.NewIntVar(0, HORIZON, 'makespan')
    for op_id, end_v in end_vars.items():
        model.Add(end_v <= makespan)
    model.AddMaxEquality(makespan, list(end_vars.values()))

    total_tardiness = model.NewIntVar(0, HORIZON * max(1, len(end_vars)), 'total_tardiness')
    model.Add(total_tardiness == sum(tardiness_vars.values()))

    # Objective handling
    if objective_mode == 'min_tardiness':
        model.Minimize(total_tardiness)
    elif objective_mode == 'min_makespan':
        model.Minimize(makespan)
    elif objective_mode == 'lex_tardiness_makespan':
        # Lexicographic: implement via weighted large coefficient
        BIG = HORIZON * 100
        model.Minimize(total_tardiness * BIG + makespan)
    else:  # 'min_weighted'
        # Scale alpha to integer weight by multiplying
        weight = int(alpha * 1000)
        model.Minimize(total_tardiness * 1000 + makespan * weight)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8
    if not log:
        solver.parameters.log_search_progress = False

    status_code = solver.Solve(model)
    status_map = {
        cp_model.OPTIMAL: 'OPTIMAL',
        cp_model.FEASIBLE: 'FEASIBLE',
        cp_model.INFEASIBLE: 'INFEASIBLE',
        cp_model.MODEL_INVALID: 'MODEL_INVALID',
        cp_model.UNKNOWN: 'UNKNOWN'
    }
    status = status_map.get(status_code, 'UNKNOWN')

    schedule_rows: List[Dict[str, Any]] = []
    if status in ('OPTIMAL', 'FEASIBLE'):
        for _, op in ops.iterrows():
            op_id = op['Operation_ID']
            # Identify chosen machine (presence var true)
            chosen_machine = None
            for (oid, mid), presence in presence_vars.items():
                if oid == op_id and solver.BooleanValue(presence):
                    chosen_machine = mid
                    break
            schedule_rows.append({
                'Operation_ID': op_id,
                'Job_ID': op['Job_ID'],
                'Machine_ID': chosen_machine or 'UNASSIGNED',
                'Start_Time': solver.Value(start_vars[op_id]),
                'End_Time': solver.Value(end_vars[op_id]),
                'Due_Time': int(op.get('Due_Time_Min', 0)),
                'Tardiness': solver.Value(tardiness_vars[op_id]),
                'Priority': op.get('Priority', 3),
                'Assignment_Type': op.get('Assignment_Type', 'IN_HOUSE'),
                'Objective_Mode': objective_mode
            })

    result = CPSATSchedulerResult(
        schedule=schedule_rows,
        objective_value=solver.ObjectiveValue() if status in ('OPTIMAL', 'FEASIBLE') else None,
        status=status,
        solver_stats={
            'conflicts': solver.NumConflicts(),
            'branches': solver.NumBranches(),
            'wall_time': solver.WallTime(),
        }
    )
    return result

if __name__ == '__main__':
    # Minimal dry-run example placeholder (expects preloaded DataFrames)
    print("CP-SAT scheduler module loaded. Integrate into backend to execute.")
