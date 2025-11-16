# utils/helpers.py
"""
Helper functions for data processing and calculations
"""
import pandas as pd
import numpy as np
import streamlit as st


def dbg(msg):
    """Simple debug helper that writes to Streamlit if available"""
    try:
        st.write(msg)
    except Exception:
        print(msg)


def safe_toast(message, icon=None):
    """Guarded toast wrapper: some Streamlit installs may not support st.toast"""
    try:
        if icon is not None:
            st.toast(message, icon=icon)
        else:
            st.toast(message)
    except Exception:
        try:
            st.success(message)
        except Exception:
            print(message)


def parse_maintenance(maintenance_str):
    """Parse maintenance window string into time dictionary"""
    if pd.isna(maintenance_str) or maintenance_str == 'None':
        return None
    try:
        parts = maintenance_str.replace("Day", "").replace(",", "").strip().split()
        day = int(parts[0])
        times = parts[1].split('-')
        start_hour, start_min = map(int, times[0].split(':'))
        end_hour, end_min = map(int, times[1].split(':'))
        MINUTES_PER_WORKDAY = 8 * 60
        WORK_START_HOUR = 8
        WORK_END_HOUR = 16

        def clock_to_work_minutes(hour, minute):
            if WORK_START_HOUR <= hour < WORK_END_HOUR:
                work_hour = hour - WORK_START_HOUR
                return work_hour * 60 + minute
            else:
                return None

        start_work_min = clock_to_work_minutes(start_hour, start_min)
        end_work_min = clock_to_work_minutes(end_hour, end_min)

        if start_work_min is None or end_work_min is None:
            return None

        start_time = (day - 1) * MINUTES_PER_WORKDAY + start_work_min
        end_time = (day - 1) * MINUTES_PER_WORKDAY + end_work_min

        return {'start': start_time, 'end': end_time, 'duration': end_time - start_time}
    except Exception:
        return None


def get_eligible_machines(op_type):
    """Get list of eligible machines for operation type"""
    if op_type == 'MILLING':
        return ['M1', 'M3', 'M4']
    elif op_type == 'TURNING':
        return ['M6', 'M9']
    elif op_type == 'GRINDING':
        return ['M6', 'M9']
    elif op_type == 'DRILLING':
        return ['M1', 'M3', 'M4']
    else:
        return []


def get_setup_penalty(prev_material, next_material, df_penalties):
    """Calculate setup penalty for material changeover"""
    if not prev_material or not next_material:
        return 0
    penalty = df_penalties[
        (df_penalties['Previous Material'] == prev_material) &
        (df_penalties['Next Material'] == next_material)
    ]
    return penalty.iloc[0]['Penalty Time (min)'] if len(penalty) > 0 else 15


def calculate_inhouse_cost(operation, df_effective, hourly_rate=30):
    """Calculate in-house cost for an operation"""
    op_id = operation['Operation_ID']
    eligible = df_effective[df_effective['Operation_ID'] == op_id]

    if len(eligible) == 0:
        return float('inf'), None

    best_option = eligible.loc[eligible['Total_Time'].idxmin()]
    labor_cost = (best_option['Total_Time'] / 60) * hourly_rate
    material_cost = operation['Quantity'] * 0.5
    total_cost = labor_cost + material_cost

    return total_cost, best_option['Machine_ID']


def make_or_buy_decision(operation, df_effective, cost_threshold=0.9, hourly_rate=30):
    """Make outsourcing decision based on cost threshold"""
    inhouse_cost, best_machine = calculate_inhouse_cost(operation, df_effective, hourly_rate)
    inhouse_time = operation.get('Total_Proc_Min', operation.get('Proc_Time_per_Unit', 0) * operation.get('Quantity', 1)) + operation.get('Setup_Time', 0)
    outsource_cost = operation.get('Outsource_Cost', np.inf)
    outsource_time = operation.get('Outsource_Time_Min', np.inf)

    earliest_start = operation.get('Release_Time_Min', 0)
    earliest_finish = earliest_start + inhouse_time
    can_meet_deadline = earliest_finish <= operation.get('Due_Time_Min', np.inf)

    if not can_meet_deadline and outsource_time < inhouse_time:
        return 'OUTSOURCE', outsource_cost, 'Deadline constraint'
    if outsource_cost < (inhouse_cost * cost_threshold):
        return 'OUTSOURCE', outsource_cost, 'Cost advantage'
    return 'IN_HOUSE', inhouse_cost, 'Best in-house'
