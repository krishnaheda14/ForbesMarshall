# utils/metrics.py
"""
Metrics calculation and KPI functions
"""
import pandas as pd


def calculate_metrics(schedule_df, df_ops, heuristic_name):
    """Calculate comprehensive metrics for a schedule"""
    if schedule_df is None or schedule_df.empty:
        return {
            'Heuristic': heuristic_name,
            'Makespan_Days': 0,
            'Total_Tardiness_Days': 0,
            'Total_Cost_$': 0,
            'On_Time_%': 0,
            'Machine_Utilization_%': 0
        }

    makespan_min = schedule_df['End_Time'].max() if not schedule_df.empty else 0
    makespan_days = makespan_min / 480

    total_tardiness_min = schedule_df['Tardiness'].sum() if 'Tardiness' in schedule_df.columns else 0
    total_tardiness_days = total_tardiness_min / 480

    late_ops = len(schedule_df[schedule_df['Tardiness'] > 0]['Operation_ID'].unique()) if not schedule_df.empty else 0
    total_ops = df_ops['Operation_ID'].nunique()

    if late_ops > total_ops:
        late_ops = total_ops

    avg_tardiness = schedule_df.groupby('Operation_ID')['Tardiness'].sum().mean() if not schedule_df.empty else 0

    # Get machine count from session state or default
    try:
        import streamlit as st
        if 'base_df_machines' in st.session_state:
            machine_count = len(st.session_state.base_df_machines)
        else:
            machine_count = 5
    except:
        machine_count = 5

    total_setup_time = schedule_df['Setup_Time'].sum() if 'Setup_Time' in schedule_df.columns else 0
    total_proc_time = schedule_df['Proc_Time'].sum() if 'Proc_Time' in schedule_df.columns else 0
    total_transfer_time = schedule_df['Transfer_Time'].sum() if 'Transfer_Time' in schedule_df.columns else 0
    total_productive_time = total_setup_time + total_proc_time + total_transfer_time

    total_available_time = machine_count * makespan_min if makespan_min > 0 else machine_count * 1
    utilization = (total_productive_time / total_available_time) * 100 if total_available_time > 0 else 0

    machine_utilization = {}
    if not schedule_df.empty:
        for machine_id in schedule_df['Machine_ID'].unique():
            machine_ops = schedule_df[schedule_df['Machine_ID'] == machine_id]
            machine_productive = (machine_ops['Setup_Time'].sum() +
                                machine_ops['Proc_Time'].sum() +
                                machine_ops['Transfer_Time'].sum())
            machine_util = (machine_productive / makespan_min) * 100 if makespan_min > 0 else 0
            machine_utilization[machine_id] = round(machine_util, 1)

    inhouse_cost = schedule_df['Proc_Time'].sum() / 60 * 50 if 'Proc_Time' in schedule_df.columns else 0
    outsource_cost = df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE']['Outsource_Cost'].sum() if 'Outsource_Cost' in df_ops.columns else 0
    total_cost = inhouse_cost + outsource_cost

    ontime_pct = ((total_ops - late_ops) / total_ops) * 100 if total_ops > 0 else 100
    ontime_pct = max(0, min(100, ontime_pct))

    return {
        'Heuristic': heuristic_name,
        'Makespan_Days': round(makespan_days, 2),
        'Total_Tardiness_Days': round(total_tardiness_days, 2),
        'Late_Operations': int(late_ops),
        'Total_Operations': int(total_ops),
        'On_Time_%': round(ontime_pct, 1),
        'Avg_Tardiness_Min': round(avg_tardiness, 1),
        'Machine_Utilization_%': round(utilization, 1),
        'Total_Cost_$': round(total_cost, 2),
        '_Machine_Details': machine_utilization,
        '_Scheduled_Ops': len(schedule_df),
        '_Total_Ops': len(df_ops),
        '_Productive_Time_Days': round(total_productive_time / 480, 2)
    }


def check_breakdown_conflicts(schedule_df, machines_df):
    """
    Check if any scheduled operations overlap with maintenance/breakdown windows.
    Returns list of conflicts with details.
    """
    conflicts = []
    
    if schedule_df.empty:
        return conflicts
    
    for _, job in schedule_df.iterrows():
        machine_id = job['Machine_ID']
        job_start = job['Start_Time']
        job_end = job['End_Time']
        
        # Get maintenance windows for this machine
        machine_row = machines_df[machines_df['Machine_ID'] == machine_id]
        if machine_row.empty:
            continue
        
        maintenance = machine_row.iloc[0].get('Maintenance_Window', None)
        if not maintenance:
            continue
        
        # Normalize to list
        windows = []
        if isinstance(maintenance, dict) and maintenance:
            windows = [maintenance]
        elif isinstance(maintenance, list):
            windows = [w for w in maintenance if isinstance(w, dict) and w]
        
        # Check for overlap
        for window in windows:
            mw_start = window.get('start', 0)
            mw_end = window.get('end', 0)
            
            # Overlap check: job overlaps if it starts before window ends AND ends after window starts
            if job_start < mw_end and job_end > mw_start:
                conflicts.append({
                    'Operation_ID': job['Operation_ID'],
                    'Job_ID': job.get('Job_ID', 'N/A'),
                    'Machine_ID': machine_id,
                    'Job_Start': job_start,
                    'Job_End': job_end,
                    'Breakdown_Start': mw_start,
                    'Breakdown_End': mw_end,
                    'Overlap_Minutes': min(job_end, mw_end) - max(job_start, mw_start)
                })
    
    return conflicts
