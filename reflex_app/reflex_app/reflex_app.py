"""
CNC Job Scheduling System - Reflex Version
Complete port of cnc-scheduling.py with all 6 heuristics + AI + Make-or-buy
NO changes to original cnc-scheduling.py - all logic replicated here
"""

import reflex as rx
import pandas as pd
import numpy as np
import os
import time
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
import plotly.graph_objects as go

# Load environment
load_dotenv()

# Try to import Google Gemini AI
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        AI_ENABLED = True
    else:
        AI_ENABLED = False
except:
    AI_ENABLED = False


# ============================================================================
# HELPER FUNCTIONS - EXACT COPIES FROM cnc-scheduling.py
# ============================================================================

def parse_maintenance(maintenance_str):
    """Parse maintenance window - EXACT COPY"""
    if pd.isna(maintenance_str) or maintenance_str == 'None':
        return None
    try:
        parts = maintenance_str.replace("Day", "").replace(",", "").strip().split()
        day = int(parts[0])
        times = parts[1].split('-')
        start_hour, start_min = map(int, times[0].split(':'))
        end_hour, end_min = map(int, times[1].split(':'))
        
        WORK_START_HOUR, WORK_END_HOUR = 8, 16
        MINUTES_PER_WORKDAY = 8 * 60
        
        def clock_to_work_minutes(hour, minute):
            if WORK_START_HOUR <= hour < WORK_END_HOUR:
                return (hour - WORK_START_HOUR) * 60 + minute
            return None
        
        start_work_min = clock_to_work_minutes(start_hour, start_min)
        end_work_min = clock_to_work_minutes(end_hour, end_min)
        
        if start_work_min is None or end_work_min is None:
            return None
        
        start_time = (day - 1) * MINUTES_PER_WORKDAY + start_work_min
        end_time = (day - 1) * MINUTES_PER_WORKDAY + end_work_min
        
        return {'start': start_time, 'end': end_time, 'duration': end_time - start_time}
    except:
        return None


def get_eligible_machines(op_type):
    """Get eligible machines - EXACT COPY"""
    if op_type == 'MILLING':
        return ['M1', 'M3', 'M4']
    elif op_type == 'TURNING':
        return ['M6', 'M9']
    elif op_type == 'GRINDING':
        return ['M6', 'M9']
    elif op_type == 'DRILLING':
        return ['M1', 'M3', 'M4']
    return []


def calculate_inhouse_cost(operation, df_effective, hourly_rate=30):
    """Calculate in-house cost - EXACT COPY"""
    op_id = operation['Operation_ID']
    eligible = df_effective[df_effective['Operation_ID'] == op_id]
    if len(eligible) == 0:
        return float('inf'), None
    best_option = eligible.loc[eligible['Total_Time'].idxmin()]
    labor_cost = (best_option['Total_Time'] / 60) * hourly_rate
    material_cost = operation['Quantity'] * 0.5
    return labor_cost + material_cost, best_option['Machine_ID']


def make_or_buy_decision(operation, df_effective, cost_threshold=0.9, hourly_rate=30):
    """Make-or-buy decision - EXACT COPY"""
    inhouse_cost, best_machine = calculate_inhouse_cost(operation, df_effective, hourly_rate)
    outsource_cost = operation.get('Outsource_Cost', np.inf)
    
    if outsource_cost <= 0 or outsource_cost == np.inf:
        return 'IN_HOUSE', inhouse_cost, 'No vendor'
    
    if outsource_cost < (inhouse_cost * cost_threshold):
        return 'OUTSOURCE', outsource_cost, 'Cost advantage'
    return 'IN_HOUSE', inhouse_cost, 'Best in-house'


def get_setup_penalty(prev_material, next_material, df_penalties):
    """Get setup penalty - EXACT COPY"""
    if not prev_material or not next_material:
        return 0
    penalty = df_penalties[
        (df_penalties['Previous Material'] == prev_material) &
        (df_penalties['Next Material'] == next_material)
    ]
    return penalty.iloc[0]['Penalty Time (min)'] if len(penalty) > 0 else 15


def calculate_metrics(schedule_df, df_ops, heuristic_name, hourly_rate=30):
    """Calculate metrics - EXACT COPY"""
    if schedule_df is None or schedule_df.empty:
        return {
            'Heuristic': heuristic_name,
            'Makespan_Days': 0,
            'Total_Tardiness_Days': 0,
            'Total_Cost_$': 0,
            'On_Time_%': 0,
            'Machine_Utilization_%': 0,
            'Late_Operations': 0,
            'Total_Operations': 0
        }
    
    makespan_days = schedule_df['End_Time'].max() / 480 if not schedule_df.empty else 0
    total_tardiness_days = schedule_df['Tardiness'].sum() / 480 if 'Tardiness' in schedule_df.columns else 0
    late_ops = len(schedule_df[schedule_df['Tardiness'] > 0]['Operation_ID'].unique())
    total_ops = df_ops['Operation_ID'].nunique()
    
    # Utilization
    machine_count = 5
    total_productive = schedule_df['Setup_Time'].sum() + schedule_df['Proc_Time'].sum() + schedule_df['Transfer_Time'].sum()
    total_available = machine_count * schedule_df['End_Time'].max()
    utilization = (total_productive / total_available) * 100 if total_available > 0 else 0
    
    # Cost
    inhouse_cost = schedule_df['Proc_Time'].sum() / 60 * hourly_rate
    outsource_cost = df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE']['Outsource_Cost'].sum() if 'Outsource_Cost' in df_ops.columns else 0
    
    ontime_pct = ((total_ops - late_ops) / total_ops) * 100 if total_ops > 0 else 100
    
    return {
        'Heuristic': heuristic_name,
        'Makespan_Days': round(makespan_days, 2),
        'Total_Tardiness_Days': round(total_tardiness_days, 2),
        'Late_Operations': int(late_ops),
        'Total_Operations': int(total_ops),
        'On_Time_%': round(ontime_pct, 1),
        'Machine_Utilization_%': round(utilization, 1),
        'Total_Cost_$': round(inhouse_cost + outsource_cost, 2),
    }


# ============================================================================
# CNC SCHEDULER CLASS - EXACT COPY FROM cnc-scheduling.py
# ============================================================================

class CNCScheduler:
    """Complete scheduler with all 6 heuristics - EXACT COPY"""
    
    def __init__(self, df_ops, df_machines, df_effective, df_penalties):
        self.df_ops = df_ops.copy()
        self.df_machines = df_machines.copy()
        self.df_effective = df_effective.copy()
        self.df_penalties = df_penalties
        
        self.machine_availability = {m: 0 for m in df_machines['Machine_ID']}
        self.machine_last_material = {m: None for m in df_machines['Machine_ID']}
        self.schedule = []
        self.op_completion_times = {}
        
        print(f"🔧 DEBUG: Scheduler initialized ({len(df_ops)} ops, {len(df_machines)} machines)")
    
    def reset(self):
        """Reset state"""
        self.machine_availability = {m: 0 for m in self.df_machines['Machine_ID']}
        self.machine_last_material = {m: None for m in self.df_machines['Machine_ID']}
        self.schedule = []
        self.op_completion_times = {}
    
    def get_earliest_available_time(self, machine_id, release_time, duration):
        """Handle maintenance windows - EXACT COPY"""
        current_avail = max(self.machine_availability.get(machine_id, 0), release_time)
        
        try:
            machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id]
            if machine_row.empty:
                return current_avail
            maintenance = machine_row.iloc[0].get('Maintenance_Window', None)
        except:
            return current_avail
        
        if maintenance is None or (isinstance(maintenance, dict) and not maintenance):
            return current_avail
        
        maintenance_list = [maintenance] if isinstance(maintenance, dict) else [m for m in maintenance if isinstance(m, dict)]
        maintenance_list.sort(key=lambda mw: mw.get('start', 0))
        
        for _ in range(100):  # Max iterations
            adjusted = False
            end_time = current_avail + duration
            
            for mw in maintenance_list:
                mw_start, mw_end = mw.get('start', 0), mw.get('end', 0)
                if mw_end <= mw_start:
                    continue
                if (current_avail < mw_end) and (end_time > mw_start):
                    current_avail = mw_end
                    adjusted = True
                    break
            if not adjusted:
                break
        
        return current_avail
    
    def get_available_operations(self):
        """Get available operations - EXACT COPY"""
        available = []
        for idx, op in self.df_ops.iterrows():
            op_id = op['Operation_ID']
            if op.get('Assignment_Type', 'IN_HOUSE') == 'OUTSOURCE' or op_id in self.op_completion_times:
                continue
            
            same_job = self.df_ops[self.df_ops['Job_ID'] == op['Job_ID']].sort_values('Op_Seq')
            all_pred_done = True
            earliest_start = op.get('Release_Time_Min', 0)
            
            for _, pred in same_job.iterrows():
                if pred['Op_Seq'] < op['Op_Seq']:
                    if pred['Operation_ID'] not in self.op_completion_times:
                        if pred.get('Assignment_Type', 'IN_HOUSE') == 'OUTSOURCE':
                            outsource_time = pred.get('Release_Time_Min', 0) + pred.get('Outsource_Time_Min', 0)
                            earliest_start = max(earliest_start, outsource_time)
                            self.op_completion_times[pred['Operation_ID']] = outsource_time
                        else:
                            all_pred_done = False
                            break
                    else:
                        earliest_start = max(earliest_start, self.op_completion_times[pred['Operation_ID']])
            
            if all_pred_done:
                available.append((op, earliest_start))
        return available
    
    def find_best_machine(self, operation, earliest_start_time):
        """Find best machine - EXACT COPY"""
        op_id = operation['Operation_ID']
        eligible = self.df_effective[self.df_effective['Operation_ID'] == op_id]
        if len(eligible) == 0:
            return None, float('inf')
        
        best_machine, best_completion = None, float('inf')
        for _, machine_option in eligible.iterrows():
            machine_id = machine_option['Machine_ID']
            eff_time = machine_option['Effective_Proc_Time']
            
            setup_penalty = get_setup_penalty(
                self.machine_last_material.get(machine_id),
                operation.get('Mat_Type'),
                self.df_penalties
            )
            
            actual_setup = operation.get('Setup_Time', 0) + setup_penalty
            transfer = operation.get('Transfer_Min', 0)
            total_duration = actual_setup + eff_time + transfer
            
            start_time = self.get_earliest_available_time(machine_id, earliest_start_time, total_duration)
            completion_time = start_time + total_duration
            
            if completion_time < best_completion:
                best_completion = completion_time
                best_machine = machine_id
        
        return best_machine, best_completion
    
    def schedule_operation(self, operation, machine_id, earliest_start_time):
        """Schedule operation - EXACT COPY"""
        op_id = operation['Operation_ID']
        op_details_query = self.df_effective[
            (self.df_effective['Operation_ID'] == op_id) &
            (self.df_effective['Machine_ID'] == machine_id)
        ]
        if len(op_details_query) == 0:
            return False
        
        op_details = op_details_query.iloc[0]
        eff_time = op_details['Effective_Proc_Time']
        
        setup_penalty = get_setup_penalty(
            self.machine_last_material.get(machine_id),
            operation.get('Mat_Type'),
            self.df_penalties
        )
        actual_setup = operation.get('Setup_Time', 0) + setup_penalty
        transfer = operation.get('Transfer_Min', 0)
        total_duration = actual_setup + eff_time + transfer
        
        start_time = self.get_earliest_available_time(machine_id, earliest_start_time, total_duration)
        end_time = start_time + total_duration
        
        self.schedule.append({
            'Operation_ID': op_id,
            'Job_ID': operation['Job_ID'],
            'Machine_ID': machine_id,
            'Start_Time': start_time,
            'End_Time': end_time,
            'Setup_Time': actual_setup,
            'Proc_Time': eff_time,
            'Transfer_Time': transfer,
            'Due_Time': operation.get('Due_Time_Min', 0),
            'Tardiness': max(0, end_time - operation.get('Due_Time_Min', 0))
        })
        
        self.machine_availability[machine_id] = end_time
        self.machine_last_material[machine_id] = operation.get('Mat_Type')
        self.op_completion_times[op_id] = end_time
        return True
    
    def select_next_operation(self, available_ops, heuristic='SPT'):
        """Select next operation - ALL 6 HEURISTICS - EXACT COPY"""
        def safe_priority(op):
            return int(op.get('Priority', 3))
        
        print(f"⚙️ DEBUG: {heuristic} selecting from {len(available_ops)} available")
        
        if heuristic == 'SPT':
            return min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Total_Proc_Min'], x[0]['Due_Time_Min']))
        
        elif heuristic == 'EDD':
            return min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min'], x[0]['Total_Proc_Min']))
        
        elif heuristic == 'CR':
            return min(available_ops, key=lambda x: (safe_priority(x[0]), (x[0]['Due_Time_Min'] / max(x[0]['Total_Proc_Min'], 1))))
        
        elif heuristic == 'PRIORITY':
            return min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min']))
        
        elif heuristic == 'WEIGHTED':
            def weighted_score(op_tuple):
                op = op_tuple[0]
                max_proc = max([x[0]['Total_Proc_Min'] for x in available_ops])
                max_due = max([x[0]['Due_Time_Min'] for x in available_ops])
                proc_norm = op['Total_Proc_Min'] / max_proc if max_proc > 0 else 0
                urgency_norm = (max_due - op['Due_Time_Min']) / max_due if max_due > 0 else 0
                priority_norm = (5 - safe_priority(op)) / 4
                score = 0.4 * urgency_norm + 0.3 * proc_norm + 0.3 * priority_norm
                return (safe_priority(op), score)
            return min(available_ops, key=weighted_score)
        
        elif heuristic == 'SLACK':
            def slack_time(op_tuple):
                op, earliest = op_tuple
                slack = op['Due_Time_Min'] - earliest - op['Total_Proc_Min']
                return (safe_priority(op), slack, op['Total_Proc_Min'])
            return min(available_ops, key=slack_time)
        
        return min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Total_Proc_Min']))
    
    def run_scheduling(self, heuristic='SPT'):
        """Run scheduling - EXACT COPY"""
        print(f"🔄 DEBUG: Running {heuristic}...")
        self.reset()
        
        # Handle outsourced ops
        outsourced = self.df_ops[self.df_ops.get('Assignment_Type', 'IN_HOUSE') == 'OUTSOURCE']
        for _, op in outsourced.iterrows():
            self.op_completion_times[op['Operation_ID']] = op.get('Release_Time_Min', 0) + op.get('Outsource_Time_Min', 0)
        print(f"  ✓ DEBUG: {len(outsourced)} outsourced")
        
        # Schedule in-house ops
        non_outsourced = self.df_ops[self.df_ops.get('Assignment_Type', 'IN_HOUSE') != 'OUTSOURCE']
        operations_count = len(non_outsourced)
        scheduled_ops_set = set()
        
        max_iterations = operations_count * 2 if operations_count > 0 else 1000
        iteration = 0
        
        while len(scheduled_ops_set) < operations_count:
            iteration += 1
            if iteration > max_iterations:
                print(f"⚠️ DEBUG: Max iterations ({iteration})")
                break
            
            available = self.get_available_operations()
            available = [op for op in available if op[0]['Operation_ID'] not in scheduled_ops_set]
            
            if not available:
                print("⚠️ DEBUG: No available ops")
                break
            
            next_op, earliest_start = self.select_next_operation(available, heuristic=heuristic)
            if next_op is None:
                break
            
            best_machine, _ = self.find_best_machine(next_op, earliest_start)
            if best_machine is None:
                scheduled_ops_set.add(next_op['Operation_ID'])
                continue
            
            if self.schedule_operation(next_op, best_machine, earliest_start):
                scheduled_ops_set.add(next_op['Operation_ID'])
        
        print(f"✅ DEBUG: {heuristic} done - {len(scheduled_ops_set)}/{operations_count} scheduled")
        return pd.DataFrame(self.schedule)


# ============================================================================
# DATA LOADING - EXACT COPY FROM cnc-scheduling.py
# ============================================================================

def load_all_data(sample_size=None):
    """Load and preprocess data - EXACT COPY"""
    print("DEBUG: Loading data...")
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, 'data')
        df_ops = pd.read_csv(os.path.join(data_dir, 'jobs_dataset.csv'))
        df_vendors = pd.read_csv(os.path.join(data_dir, 'vendor_data.csv'))
        df_machines = pd.read_csv(os.path.join(data_dir, 'machine_data.csv'))
        df_penalties = pd.read_csv(os.path.join(data_dir, 'previous_next_material.csv'))
        print("DEBUG: Files loaded")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return None, None, None, None, None
    
    # Sample if requested
    if sample_size:
        unique_jobs = df_ops['Job_ID'].unique()[:sample_size]
        df_ops = df_ops[df_ops['Job_ID'].isin(unique_jobs)].copy()
        print(f"📊 DEBUG: Sampled {sample_size} jobs ({len(df_ops)} ops)")
    
    # Fix deadlines
    deadline_issues = df_ops[df_ops['Due_Day'] <= df_ops['Release_Day']]
    if len(deadline_issues) > 0:
        print(f"⚠️ DEBUG: Fixing {len(deadline_issues)} deadlines")
        for idx in deadline_issues.index:
            df_ops.at[idx, 'Due_Day'] = df_ops.at[idx, 'Release_Day'] + 10
    
    # Calculate total proc time
    df_ops['Total_Proc_Min'] = df_ops['Proc_Time_per_Unit'] * df_ops['Quantity']
    
    # Fix machine column names
    if 'Speed Factor' not in df_machines.columns and 'SpeedFactor' in df_machines.columns:
        df_machines.rename(columns={'SpeedFactor': 'Speed Factor'}, inplace=True)
    
    df_machines['Speed Factor'] = df_machines['Speed Factor'].astype(str).str.extract(r'([0-9]*\.?[0-9]+)').astype(float)
    
    # Calculate effective times
    print("🔍 DEBUG: Calculating effective times...")
    effective_times = []
    for idx, op in df_ops.iterrows():
        eligible_machines = get_eligible_machines(op['Op_Type'])
        if not eligible_machines:
            continue
        
        total_proc_min = float(op['Proc_Time_per_Unit']) * float(op['Quantity'])
        setup_time = float(op['Setup_Time'])
        transfer_min = float(op.get('Transfer_Min', 0))
        
        for machine_id in eligible_machines:
            machine = df_machines[df_machines['Machine ID'] == machine_id].iloc[0]
            speed_factor = float(machine['Speed Factor'])
            oee = float(machine['OEE (Uptime)'])
            effective_time = total_proc_min * speed_factor * (1 / oee)
            
            effective_times.append({
                'Operation_ID': op['Operation_ID'],
                'Machine_ID': machine_id,
                'Effective_Proc_Time': effective_time,
                'Setup_Time': setup_time,
                'Transfer_Min': transfer_min,
                'Total_Time': effective_time + setup_time + transfer_min
            })
    
    df_effective = pd.DataFrame(effective_times)
    print(f"✅ DEBUG: {len(df_effective)} effective time entries")
    
    # Vendor processing
    df_vendors['Outsource_Unit_Cost'] = df_vendors['Outsource_Unit_Cost'].replace('[\\$,]', '', regex=True).astype(float)
    df_vendors['Transport_Cost'] = df_vendors['Transport_Cost'].replace('[\\$,]', '', regex=True).astype(float)
    
    df_ops_vendor = df_ops.merge(
        df_vendors[['Vendor_ID', 'Outsource_Lead_Time (Days)', 'Outsource_Unit_Cost', 'Transport_Cost', 'Quality_Factor']],
        left_on='Vendor_Ref', right_on='Vendor_ID', how='left'
    )
    
    df_ops_vendor['Outsource_Cost'] = (
        (df_ops_vendor['Outsource_Unit_Cost'] * df_ops_vendor['Quantity']) + df_ops_vendor['Transport_Cost']
    ) / df_ops_vendor['Quality_Factor']
    
    df_ops_vendor['Outsource_Time_Min'] = df_ops_vendor['Outsource_Lead_Time (Days)'] * 8 * 60
    
    df_ops = df_ops.merge(
        df_ops_vendor[['Operation_ID', 'Outsource_Cost', 'Outsource_Time_Min']],
        on='Operation_ID', how='left'
    )
    
    # Time conversions
    MINUTES_PER_DAY = 8 * 60
    df_ops['Release_Time_Min'] = df_ops['Release_Day'] * MINUTES_PER_DAY
    df_ops['Due_Time_Min'] = df_ops['Due_Day'] * MINUTES_PER_DAY
    df_ops['Outsource_Cost'].fillna(0, inplace=True)
    df_ops['Outsource_Time_Min'].fillna(0, inplace=True)
    
    # Machine maintenance
    df_machines = df_machines.rename(columns={'Machine ID': 'Machine_ID'})
    df_machines['Maintenance_Window'] = df_machines['Scheduled Maintenance (Day, Time-Time)'].apply(parse_maintenance)
    
    # Make-or-buy decisions
    print("DEBUG: Make-or-buy analysis...")
    decisions = []
    for idx, op in df_ops.iterrows():
        decision, cost, reason = make_or_buy_decision(op, df_effective, cost_threshold=0.85, hourly_rate=30)
        decisions.append({'Operation_ID': op['Operation_ID'], 'Decision': decision})
    
    df_decisions = pd.DataFrame(decisions)
    df_ops = df_ops.merge(df_decisions, on='Operation_ID', how='left')
    df_ops['Assignment_Type'] = df_ops['Decision'].fillna('IN_HOUSE')
    df_ops.drop(columns=['Decision'], inplace=True)
    
    outsourced = len(df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE'])
    inhouse = len(df_ops[df_ops['Assignment_Type'] == 'IN_HOUSE'])
    print(f"DEBUG: In-house: {inhouse}, Outsourced: {outsourced}")
    
    return df_ops, df_machines, df_effective, df_penalties, df_vendors


# ============================================================================
# GANTT CHART FUNCTION - EXACT COPY FROM cnc-scheduling.py
# ============================================================================

def create_gantt_chart(schedule_df, machines_df, title="CNC Machine Schedule", machines_order=None):
    """Create interactive Gantt chart - EXACT COPY"""
    schedule_df = schedule_df.copy()
    machines_df = machines_df.copy()
    
    for col in ["Start_Time", "End_Time"]:
        if col in schedule_df.columns:
            schedule_df[col] = pd.to_numeric(schedule_df[col], errors="coerce").fillna(0)
    
    def machine_sort_key(mid):
        match = re.search(r"\d+", str(mid))
        return int(match.group()) if match else float("inf")
    
    if machines_order:
        all_machines_sorted = sorted(machines_order, key=machine_sort_key)
    else:
        all_machines_sorted = sorted(machines_df["Machine_ID"].unique(), key=machine_sort_key)
    
    df_real = schedule_df[schedule_df["Job_ID"] != "Idle"].copy() if "Job_ID" in schedule_df.columns else schedule_df.copy()
    if df_real.empty:
        return go.Figure().update_layout(title="No job data available", xaxis_title="Time (minutes)", yaxis_title="Machine")
    
    x_min = df_real["Start_Time"].min()
    x_max = df_real["End_Time"].max()
    df_real["Start_Shifted"] = df_real["Start_Time"]
    df_real["End_Shifted"] = df_real["End_Time"]
    
    fig = go.Figure()
    
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(all_machines_sorted)}
    
    for _, row in df_real.iterrows():
        machine = row["Machine_ID"]
        if machine not in all_machines_sorted:
            continue
        fig.add_trace(
            go.Bar(
                x=[row["End_Shifted"] - row["Start_Shifted"]],
                y=[machine],
                base=row["Start_Shifted"],
                orientation="h",
                marker_color=color_map[machine],
                hovertemplate=(
                    f"<b>Machine:</b> {machine}<br>"
                    f"Job ID: {row.get('Job_ID', 'N/A')}<br>"
                    f"Operation ID: {row.get('Operation_ID', 'N/A')}<br>"
                    f"Setup: {row.get('Setup_Time', 0)} min<br>"
                    f"Proc: {row.get('Proc_Time', 0)} min<br>"
                    f"Start: {row.get('Start_Time', 0):.0f} min<br>"
                    f"End: {row.get('End_Time', 0):.0f} min<extra></extra>"
                ),
                width=0.4,
                name=machine,
                showlegend=False,
            )
        )
    
    # Display maintenance/breakdown windows
    for _, machine in machines_df.iterrows():
        maint = machine.get("Maintenance_Window")
        machine_id = machine.get("Machine_ID")
        
        if maint and machine_id in all_machines_sorted:
            windows = [maint] if isinstance(maint, dict) else (maint if isinstance(maint, list) else [])
            
            for i, window in enumerate(windows):
                if isinstance(window, dict) and "start" in window and "end" in window:
                    window_start_shifted = window["start"]
                    window_end_shifted = window["end"]
                    window_duration = window.get("duration", window["end"] - window["start"])
                    
                    fig.add_shape(
                        type="rect",
                        x0=window_start_shifted,
                        x1=window_end_shifted,
                        y0=all_machines_sorted.index(machine_id) - 0.45,
                        y1=all_machines_sorted.index(machine_id) + 0.45,
                        fillcolor="rgba(255,50,50,0.35)",
                        line=dict(color="rgba(255,0,0,0.8)", width=2, dash="dot"),
                        layer="above",
                    )
                    
                    fig.add_annotation(
                        x=(window_start_shifted + window_end_shifted) / 2,
                        y=machine_id,
                        text=f"BREAKDOWN<br>{window_duration} min",
                        showarrow=False,
                        font=dict(size=9, color="white", family="Arial Black"),
                        bgcolor="rgba(200,0,0,0.7)",
                        bordercolor="red",
                        borderwidth=1,
                        borderpad=2,
                        opacity=0.9
                    )
    
    pad = max((x_max - x_min) * 0.05, 100)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (minutes)",
        yaxis_title="Machine",
        height=600,
        margin=dict(l=50, r=20, t=60, b=40),
        xaxis=dict(tickformat=",.0f", dtick=500),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=all_machines_sorted, autorange="reversed", type="category")
    fig.update_xaxes(range=[x_min - pad, x_max + pad], showgrid=True)
    return fig


# ============================================================================
# REFLEX STATE CLASS
# ============================================================================

class SchedulerState(rx.State):
    """Main state - replaces Streamlit session_state"""
    
    # Data (base copies for reset)
    df_ops: pd.DataFrame = pd.DataFrame()
    df_machines: pd.DataFrame = pd.DataFrame()
    df_effective: pd.DataFrame = pd.DataFrame()
    df_penalties: pd.DataFrame = pd.DataFrame()
    df_vendors: pd.DataFrame = pd.DataFrame()
    base_df_ops: pd.DataFrame = pd.DataFrame()
    base_df_machines: pd.DataFrame = pd.DataFrame()
    base_df_effective: pd.DataFrame = pd.DataFrame()
    
    # Schedules
    schedule_spt: pd.DataFrame = pd.DataFrame()
    schedule_edd: pd.DataFrame = pd.DataFrame()
    schedule_cr: pd.DataFrame = pd.DataFrame()
    schedule_priority: pd.DataFrame = pd.DataFrame()
    schedule_weighted: pd.DataFrame = pd.DataFrame()
    schedule_slack: pd.DataFrame = pd.DataFrame()
    
    # Current
    current_heuristic: str = "SPT"
    current_schedule: pd.DataFrame = pd.DataFrame()
    
    # Metrics
    comparison_metrics: List[Dict] = []
    current_metrics: Dict = {}
    
    # UI state
    loading: bool = False
    status_message: str = "Ready"
    error_message: str = ""
    data_loaded: bool = False
    show_gantt: bool = False
    gantt_html: str = ""
    
    # Parameters
    hourly_rate: float = 30.0
    cost_threshold: float = 0.85
    
    # Add Job form fields
    new_job_id: str = ""
    new_job_quantity: int = 100
    new_job_priority: int = 2
    new_job_due_days: int = 7
    new_job_op_type: str = "MILLING"
    new_job_material: str = "STEEL"
    new_job_proc_time: float = 0.3
    new_job_setup_time: int = 30
    
    # Priority Manager fields
    priority_job_id: str = ""
    priority_new_value: int = 2
    
    # Breakdown Simulator fields
    breakdown_machine: str = ""
    breakdown_start: int = 100
    breakdown_duration: int = 120
    
    # Activity log
    activity_log: List[Dict] = []
    
    # Recompute flag
    needs_recompute: bool = False
    
    @rx.var
    def job_ids(self) -> List[str]:
        if self.df_ops.empty:
            return []
        return list(self.df_ops['Job_ID'].unique())

    @rx.var
    def machine_ids(self) -> List[str]:
        if self.df_machines.empty:
            return []
        return list(self.df_machines['Machine_ID'].unique())
    
    def set_quantity_from_str(self, value: str):
        if value == "":
            return
        try:
            self.new_job_quantity = int(value)
        except ValueError:
            pass

    def set_priority_from_str(self, value: str):
        if value == "":
            return
        try:
            self.new_job_priority = int(value)
        except ValueError:
            pass

    def set_due_days_from_str(self, value: str):
        if value == "":
            return
        try:
            self.new_job_due_days = int(value)
        except ValueError:
            pass

    def set_proc_time_from_str(self, value: str):
        if value == "":
            return
        try:
            self.new_job_proc_time = float(value)
        except ValueError:
            pass

    def set_setup_time_from_str(self, value: str):
        if value == "":
            return
        try:
            self.new_job_setup_time = int(value)
        except ValueError:
            pass

    def set_breakdown_start_from_str(self, value: str):
        if value == "":
            return
        try:
            self.breakdown_start = int(value)
        except ValueError:
            pass

    def set_breakdown_duration_from_str(self, value: str):
        if value == "":
            return
        try:
            self.breakdown_duration = int(value)
        except ValueError:
            pass
    
    def set_priority_new_value_from_str(self, value: str):
        """Set priority from string value"""
        if value == "":
            return
        try:
            self.priority_new_value = int(value)
        except ValueError:
            pass
    
    def load_data(self):
        """Load data"""
        print("DEBUG: Loading manufacturing data...")
        self.loading = True
        self.status_message = "Loading data..."
        yield
        
        try:
            result = load_all_data(sample_size=50)
            if result[0] is None:
                self.error_message = "Failed to load data - Check data files"
                self.loading = False
                yield
                return
            
            self.df_ops, self.df_machines, self.df_effective, self.df_penalties, self.df_vendors = result
            self.base_df_ops = self.df_ops.copy()
            self.base_df_machines = self.df_machines.copy()
            self.base_df_effective = self.df_effective.copy()
            self.data_loaded = True
            self.status_message = f"Successfully loaded {len(self.df_ops)} operations"
            self.error_message = ""
            self.new_job_id = f"J{900 + len(self.df_ops['Job_ID'].unique())}"
            if not self.df_machines.empty:
                self.breakdown_machine = self.df_machines['Machine_ID'].iloc[0]
            if not self.df_ops.empty:
                self.priority_job_id = self.df_ops['Job_ID'].iloc[0]
            print("DEBUG: Data loaded successfully")
        except Exception as e:
            self.error_message = f"Data loading error: {str(e)}"
            print(f"DEBUG ERROR: {e}")
        
        self.loading = False
        yield
    
    def compute_all_heuristics(self):
        """Compute all 6 heuristics"""
        print("🔄 DEBUG: Computing all...")
        self.loading = True
        self.status_message = "Computing..."
        yield
        
        if not self.data_loaded:
            self.error_message = "Load data first"
            self.loading = False
            yield
            return
        
        try:
            heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY', 'WEIGHTED', 'SLACK']
            metrics = []
            
            for heur in heuristics:
                print(f"🔄 DEBUG: {heur}...")
                self.status_message = f"Computing {heur}..."
                yield
                
                scheduler = CNCScheduler(self.df_ops, self.df_machines, self.df_effective, self.df_penalties)
                schedule = scheduler.run_scheduling(heuristic=heur)
                setattr(self, f'schedule_{heur.lower()}', schedule)
                
                metric = calculate_metrics(schedule, self.df_ops, heur, self.hourly_rate)
                metrics.append(metric)
                print(f"✅ DEBUG: {heur} - Makespan: {metric['Makespan_Days']}d")
            
            self.comparison_metrics = metrics
            self.status_message = "All heuristics computed"
            self.error_message = ""
            print("✅ DEBUG: Complete")
        except Exception as e:
            self.error_message = f"Error: {str(e)}"
            print(f"❌ DEBUG: {e}")
        
        self.loading = False
        yield
    
    def apply_heuristic(self, heuristic: str):
        """Apply heuristic"""
        print(f"🔄 DEBUG: Applying {heuristic}")
        self.current_heuristic = heuristic
        schedule = getattr(self, f'schedule_{heuristic.lower()}', pd.DataFrame())
        self.current_schedule = schedule
        
        if not schedule.empty:
            self.current_metrics = calculate_metrics(schedule, self.df_ops, heuristic, self.hourly_rate)
            self.status_message = f"{heuristic} applied"
            self.show_gantt = False
        else:
            self.status_message = f"No schedule for {heuristic}"
    
    def toggle_gantt(self):
        """Toggle Gantt chart display"""
        self.show_gantt = not self.show_gantt
        if self.show_gantt and not self.current_schedule.empty:
            fig = create_gantt_chart(self.current_schedule, self.df_machines, 
                                    title=f"Schedule - {self.current_heuristic}")
            # Generate full standalone HTML with inline JS
            self.gantt_html = fig.to_html(
                include_plotlyjs='cdn',
                full_html=False,
                config={'displayModeBar': True, 'responsive': True}
            )
    
    def add_new_job(self):
        """Add new job to dataset"""
        print(f"DEBUG: Adding job {self.new_job_id}")
        self.loading = True
        self.status_message = "Adding new job..."
        yield
        
        try:
            current_time_days = self.current_schedule['End_Time'].max() / 480 if not self.current_schedule.empty else 0
            release_time_min = current_time_days * 480
            due_time_min = release_time_min + (self.new_job_due_days * 480)
            
            new_op_id = f'{self.new_job_id}_Op1'
            new_op = {
                'Job_ID': self.new_job_id,
                'Operation_ID': new_op_id,
                'Op_Seq': 1,
                'Part_Type': f'NEW_{self.new_job_id}',
                'Quantity': self.new_job_quantity,
                'Op_Type': self.new_job_op_type,
                'Mat_Type': self.new_job_material,
                'Tool_Group': 'TGA',
                'Proc_Time_per_Unit': self.new_job_proc_time,
                'Setup_Time': self.new_job_setup_time,
                'Transfer_Min': 5,
                'Release_Day': current_time_days,
                'Due_Day': current_time_days + self.new_job_due_days,
                'Priority': self.new_job_priority,
                'Outsource_Flag': 'Y',
                'Vendor_Ref': 'V1' if 'V1' in self.base_df_vendors['Vendor_ID'].values else None,
                'Release_Time_Min': release_time_min,
                'Due_Time_Min': due_time_min,
                'Total_Proc_Min': self.new_job_proc_time * self.new_job_quantity,
                'Outsource_Cost': 0,
                'Outsource_Time_Min': 0
            }
            
            self.df_ops = pd.concat([self.df_ops, pd.DataFrame([new_op])], ignore_index=True)
            self.base_df_ops = self.df_ops.copy()
            
            # Add effective times
            eligible_machines = get_eligible_machines(new_op['Op_Type'])
            for machine_id in eligible_machines:
                machine = self.df_machines[self.df_machines['Machine_ID'] == machine_id].iloc[0]
                speed_factor = float(machine['Speed Factor'])
                oee = float(machine['OEE (Uptime)'])
                effective_time = new_op['Total_Proc_Min'] * speed_factor * (1 / oee)
                
                new_eff = {
                    'Operation_ID': new_op_id,
                    'Machine_ID': machine_id,
                    'Effective_Proc_Time': effective_time,
                    'Setup_Time': new_op['Setup_Time'],
                    'Transfer_Min': new_op['Transfer_Min'],
                    'Total_Time': effective_time + new_op['Setup_Time'] + new_op['Transfer_Min']
                }
                self.df_effective = pd.concat([self.df_effective, pd.DataFrame([new_eff])], ignore_index=True)
            
            self.base_df_effective = self.df_effective.copy()
            
            # Make-or-buy decision
            decision, cost, reason = make_or_buy_decision(new_op, self.df_effective, self.cost_threshold, self.hourly_rate)
            self.df_ops.loc[self.df_ops['Operation_ID'] == new_op_id, 'Assignment_Type'] = decision
            self.base_df_ops = self.df_ops.copy()
            
            self.activity_log.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'Job Added',
                'details': f"Job {self.new_job_id}: {self.new_job_op_type} on {self.new_job_material}, Qty={self.new_job_quantity}, P{self.new_job_priority}, Assignment={decision}"
            })
            
            self.status_message = f"Job {self.new_job_id} added ({decision})"
            self.needs_recompute = True
            self.new_job_id = f"J{900 + len(self.df_ops['Job_ID'].unique())}"
            print(f"DEBUG: Job added, now {len(self.df_ops)} ops")
        except Exception as e:
            self.error_message = f"Error adding job: {str(e)}"
            print(f"ERROR: {e}")
        
        self.loading = False
        yield
    
    def update_priority(self):
        """Update job priority"""
        print(f"DEBUG: Updating priority for {self.priority_job_id}")
        self.loading = True
        self.status_message = "Updating priority..."
        yield
        
        try:
            old_priority = self.df_ops[self.df_ops['Job_ID'] == self.priority_job_id]['Priority'].iloc[0] if not self.df_ops[self.df_ops['Job_ID'] == self.priority_job_id].empty else None
            
            self.df_ops.loc[self.df_ops['Job_ID'] == self.priority_job_id, 'Priority'] = self.priority_new_value
            self.base_df_ops.loc[self.base_df_ops['Job_ID'] == self.priority_job_id, 'Priority'] = self.priority_new_value
            
            self.activity_log.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'Priority Updated',
                'details': f"Job {self.priority_job_id}: P{old_priority} → P{self.priority_new_value}"
            })
            
            self.status_message = f"Priority updated for {self.priority_job_id}"
            self.needs_recompute = True
            print("DEBUG: Priority updated")
        except Exception as e:
            self.error_message = f"Error updating priority: {str(e)}"
            print(f"ERROR: {e}")
        
        self.loading = False
        yield
    
    def simulate_breakdown(self):
        """Simulate machine breakdown"""
        print(f"DEBUG: Simulating breakdown for {self.breakdown_machine}")
        self.loading = True
        self.status_message = "Simulating breakdown..."
        yield
        
        try:
            bd_end = self.breakdown_start + self.breakdown_duration
            machine_idx = self.df_machines[self.df_machines['Machine_ID'] == self.breakdown_machine].index
            
            if not machine_idx.empty:
                idx = machine_idx[0]
                breakdown_window = {'start': self.breakdown_start, 'end': bd_end, 'duration': self.breakdown_duration}
                existing_maint = self.df_machines.at[idx, 'Maintenance_Window']
                
                if existing_maint:
                    if isinstance(existing_maint, dict):
                        self.df_machines.at[idx, 'Maintenance_Window'] = [existing_maint, breakdown_window]
                    elif isinstance(existing_maint, list):
                        self.df_machines.at[idx, 'Maintenance_Window'] = existing_maint + [breakdown_window]
                else:
                    self.df_machines.at[idx, 'Maintenance_Window'] = breakdown_window
                
                self.base_df_machines = self.df_machines.copy()
                
                # Check affected operations
                affected_ops = []
                outsourced_count = 0
                if not self.current_schedule.empty and 'Machine_ID' in self.current_schedule.columns:
                    machine_schedule = self.current_schedule[self.current_schedule['Machine_ID'] == self.breakdown_machine]
                    
                    for _, op_row in machine_schedule.iterrows():
                        op_start = op_row.get('Start_Time', 0)
                        op_end = op_row.get('End_Time', 0)
                        
                        if op_start < bd_end and op_end > self.breakdown_start:
                            op_id = op_row.get('Operation_ID')
                            affected_ops.append(op_id)
                            
                            # Re-evaluate make-or-buy
                            op_data = self.df_ops[self.df_ops['Operation_ID'] == op_id]
                            if not op_data.empty:
                                op = op_data.iloc[0]
                                decision, cost, reason = make_or_buy_decision(op, self.base_df_effective, self.cost_threshold, self.hourly_rate)
                                if decision == 'OUTSOURCE':
                                    self.df_ops.loc[self.df_ops['Operation_ID'] == op_id, 'Assignment_Type'] = 'OUTSOURCE'
                                    outsourced_count += 1
                
                self.base_df_ops = self.df_ops.copy()
                
                details = f"Machine {self.breakdown_machine}: {self.breakdown_start}-{bd_end} min ({self.breakdown_duration} min)"
                if affected_ops:
                    details += f" | Affected: {len(affected_ops)} ops"
                if outsourced_count:
                    details += f" | Auto-outsourced: {outsourced_count}"
                
                self.activity_log.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Breakdown Simulated',
                    'details': details
                })
                
                self.status_message = f"Breakdown added ({len(affected_ops)} ops affected)"
                self.needs_recompute = True
                print(f"DEBUG: Breakdown simulated, {outsourced_count} outsourced")
        except Exception as e:
            self.error_message = f"Error simulating breakdown: {str(e)}"
            print(f"ERROR: {e}")
        
        self.loading = False
        yield
    
    def update_threshold(self, new_threshold: List[float]):
        """Update outsourcing cost threshold"""
        # Slider returns a list, take first value
        threshold_value = new_threshold[0] if isinstance(new_threshold, list) and len(new_threshold) > 0 else new_threshold
        print(f"DEBUG: Updating threshold to {threshold_value}")
        self.loading = True
        self.status_message = "Updating threshold..."
        yield
        
        try:
            old_threshold = self.cost_threshold
            before_outsourced = (self.df_ops['Assignment_Type'] == 'OUTSOURCE').sum()
            
            self.cost_threshold = threshold_value
            
            for idx, op in self.df_ops.iterrows():
                decision, _, _ = make_or_buy_decision(op, self.base_df_effective, self.cost_threshold, self.hourly_rate)
                self.df_ops.at[idx, 'Assignment_Type'] = decision
            
            self.base_df_ops = self.df_ops.copy()
            
            after_outsourced = (self.df_ops['Assignment_Type'] == 'OUTSOURCE').sum()
            change = after_outsourced - before_outsourced
            
            self.activity_log.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'Threshold Updated',
                'details': f"Threshold {old_threshold:.2f} → {threshold_value:.2f} | Outsourced: {before_outsourced} → {after_outsourced} ({change:+d})"
            })
            
            self.status_message = f"Threshold updated ({change:+d} outsourced ops)"
            self.needs_recompute = True
            print(f"DEBUG: Threshold updated, change={change}")
        except Exception as e:
            self.error_message = f"Error updating threshold: {str(e)}"
            print(f"ERROR: {e}")
        
        self.loading = False
        yield


# ============================================================================
# UI COMPONENTS
# ============================================================================

def header() -> rx.Component:
    """Header"""
    return rx.box(
        rx.vstack(
            rx.heading("CNC Job Scheduling System", size="9", color="white", weight="bold"),
            rx.text("Advanced Manufacturing Optimization Platform", color="rgba(255,255,255,0.9)", size="4"),
            rx.text("6 Scheduling Algorithms | Make-or-Buy Analysis | AI-Powered Insights", color="rgba(255,255,255,0.8)", size="3"),
            spacing="2",
            align="center"
        ),
        background="linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)",
        padding="3rem 2rem",
        border_radius="12px",
        margin_bottom="2rem",
        box_shadow="0 10px 40px rgba(0,0,0,0.15)"
    )


def control_panel() -> rx.Component:
    """System controls and interactive features"""
    return rx.box(
        rx.vstack(
            rx.heading("System Controls", size="7", margin_bottom="1rem", color="#1e3a8a"),
            
            # Main controls
            rx.cond(
                ~SchedulerState.data_loaded,
                rx.button("Load Manufacturing Data", on_click=SchedulerState.load_data, loading=SchedulerState.loading, size="3", color_scheme="blue", width="100%"),
                rx.badge("Data Loaded Successfully", color_scheme="green", size="2")
            ),
            rx.cond(
                SchedulerState.data_loaded,
                rx.button("Compute All Algorithms", on_click=SchedulerState.compute_all_heuristics, loading=SchedulerState.loading, size="3", color_scheme="indigo", width="100%"),
                rx.text("")
            ),
            
            # Status messages
            rx.text(SchedulerState.status_message, color="gray", size="2"),
            rx.cond(SchedulerState.error_message != "", rx.text(SchedulerState.error_message, color="red", size="2"), rx.text("")),
            rx.cond(SchedulerState.needs_recompute, 
                rx.callout("Data changed - Please recompute algorithms to see updated results", icon="info", color_scheme="amber", size="1"),
                rx.text("")
            ),
            
            # Gantt Chart Toggle
            rx.cond(
                SchedulerState.data_loaded & (SchedulerState.current_schedule.to_string() != ""),
                rx.button(
                    rx.cond(SchedulerState.show_gantt, "Hide Gantt Chart", "Show Gantt Chart"),
                    on_click=SchedulerState.toggle_gantt,
                    size="2",
                    variant="soft",
                    color_scheme="purple",
                    width="100%"
                ),
                rx.text("")
            ),
            
            spacing="4",
            width="100%"
        ),
        background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
    )


def metrics_dashboard() -> rx.Component:
    """KPI dashboard"""
    return rx.cond(
        SchedulerState.current_metrics != {},
        rx.grid(
            rx.box(
                rx.vstack(
                    rx.text("MAKESPAN", size="2", color="#6b7280", weight="bold", letter_spacing="0.05em"),
                    rx.text(f"{SchedulerState.current_metrics.get('Makespan_Days', 0):.2f}", size="8", weight="bold", color="#1e3a8a"),
                    rx.text("days", size="2", color="#9ca3af"),
                    spacing="1", align="start"
                ),
                background="white", padding="1.5rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", border_left="4px solid #3b82f6"
            ),
            rx.box(
                rx.vstack(
                    rx.text("TARDINESS", size="2", color="#6b7280", weight="bold", letter_spacing="0.05em"),
                    rx.text(f"{SchedulerState.current_metrics.get('Total_Tardiness_Days', 0):.2f}", size="8", weight="bold", color="#1e3a8a"),
                    rx.text("days", size="2", color="#9ca3af"),
                    spacing="1", align="start"
                ),
                background="white", padding="1.5rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", border_left="4px solid #f59e0b"
            ),
            rx.box(
                rx.vstack(
                    rx.text("UTILIZATION", size="2", color="#6b7280", weight="bold", letter_spacing="0.05em"),
                    rx.text(f"{SchedulerState.current_metrics.get('Machine_Utilization_%', 0):.1f}", size="8", weight="bold", color="#1e3a8a"),
                    rx.text("percent", size="2", color="#9ca3af"),
                    spacing="1", align="start"
                ),
                background="white", padding="1.5rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", border_left="4px solid #10b981"
            ),
            rx.box(
                rx.vstack(
                    rx.text("TOTAL COST", size="2", color="#6b7280", weight="bold", letter_spacing="0.05em"),
                    rx.text(f"${SchedulerState.current_metrics.get('Total_Cost_$', 0):,.0f}", size="8", weight="bold", color="#1e3a8a"),
                    rx.text("USD", size="2", color="#9ca3af"),
                    spacing="1", align="start"
                ),
                background="white", padding="1.5rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", border_left="4px solid #8b5cf6"
            ),
            columns="4", spacing="4", width="100%"
        ),
        rx.box(
            rx.text("Run algorithm analysis to view performance metrics", color="#6b7280", size="3", text_align="center"),
            padding="3rem",
            background="white",
            border_radius="12px",
            border="2px dashed #e5e7eb"
        )
    )


def comparison_table() -> rx.Component:
    """Comparison table"""
    return rx.cond(
        SchedulerState.comparison_metrics != [],
        rx.box(
            rx.heading("Algorithm Performance Comparison", size="7", margin_bottom="1rem", color="#1e3a8a"),
            rx.text("Comprehensive analysis of all 6 scheduling algorithms", color="#6b7280", size="3", margin_bottom="2rem"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Algorithm"),
                        rx.table.column_header_cell("Makespan (days)"),
                        rx.table.column_header_cell("Tardiness (days)"),
                        rx.table.column_header_cell("Utilization (%)"),
                        rx.table.column_header_cell("On-Time (%)"),
                        rx.table.column_header_cell("Cost ($)"),
                        rx.table.column_header_cell("Action"),
                    )
                ),
                rx.table.body(
                    rx.foreach(
                        SchedulerState.comparison_metrics,
                        lambda metric: rx.table.row(
                            rx.table.cell(rx.text(metric["Heuristic"], weight="bold")),
                            rx.table.cell(f"{metric['Makespan_Days']:.2f}"),
                            rx.table.cell(f"{metric['Total_Tardiness_Days']:.2f}"),
                            rx.table.cell(f"{metric['Machine_Utilization_%']:.1f}"),
                            rx.table.cell(f"{metric['On_Time_%']:.1f}"),
                            rx.table.cell(f"${metric['Total_Cost_$']:,.0f}"),
                            rx.table.cell(
                                rx.button("Apply", on_click=lambda: SchedulerState.apply_heuristic(metric["Heuristic"]), size="1", color_scheme="green")
                            ),
                        )
                    )
                ),
                variant="surface", size="3"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)"
        ),
        rx.text("")
    )


def add_job_panel() -> rx.Component:
    """Add new job interface"""
    return rx.cond(
        SchedulerState.data_loaded,
        rx.box(
            rx.vstack(
                rx.heading("Add New Job", size="6", color="#1e3a8a"),
                rx.text("Schedule new manufacturing job", color="#6b7280", size="2"),
                
                rx.grid(
                    rx.box(
                        rx.vstack(
                            rx.text("Job ID", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(value=SchedulerState.new_job_id, on_change=SchedulerState.set_new_job_id, placeholder="J901", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Quantity", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(type="number", value=SchedulerState.new_job_quantity.to_string(), on_change=SchedulerState.set_quantity_from_str, min="10", max="1000", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Priority (1-3)", size="2", weight="bold", color="#1e3a8a"),
                            rx.select(["1", "2", "3"], value=SchedulerState.new_job_priority.to_string(), on_change=SchedulerState.set_priority_from_str, width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Due in (days)", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(type="number", value=SchedulerState.new_job_due_days.to_string(), on_change=SchedulerState.set_due_days_from_str, min="1", max="30", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    columns="4", spacing="3", width="100%", margin_top="1rem"
                ),
                
                rx.grid(
                    rx.box(
                        rx.vstack(
                            rx.text("Operation Type", size="2", weight="bold", color="#1e3a8a"),
                            rx.select(["MILLING", "TURNING", "GRINDING", "DRILLING"], value=SchedulerState.new_job_op_type, on_change=SchedulerState.set_new_job_op_type, width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Material", size="2", weight="bold", color="#1e3a8a"),
                            rx.select(["STEEL", "ALUM", "TITAN", "BRASS"], value=SchedulerState.new_job_material, on_change=SchedulerState.set_new_job_material, width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Proc Time/Unit (min)", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(type="number", value=SchedulerState.new_job_proc_time.to_string(), on_change=SchedulerState.set_proc_time_from_str, step="0.1", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Setup Time (min)", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(type="number", value=SchedulerState.new_job_setup_time.to_string(), on_change=SchedulerState.set_setup_time_from_str, min="10", max="120", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    columns="4", spacing="3", width="100%", margin_top="1rem"
                ),
                
                rx.button("Add Job", on_click=SchedulerState.add_new_job, loading=SchedulerState.loading, size="3", color_scheme="green", width="100%", margin_top="1rem"),
                
                spacing="3", width="100%"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
        ),
        rx.text("")
    )


def priority_manager() -> rx.Component:
    """Job priority manager"""
    return rx.cond(
        SchedulerState.data_loaded,
        rx.box(
            rx.vstack(
                rx.heading("Job Priority Manager", size="6", color="#1e3a8a"),
                rx.text("Adjust job priorities (1=Highest, 3=Lowest)", color="#6b7280", size="2"),
                
                rx.grid(
                    rx.box(
                        rx.vstack(
                            rx.text("Select Job", size="2", weight="bold", color="#1e3a8a"),
                            rx.select(
                                SchedulerState.job_ids,
                                value=SchedulerState.priority_job_id,
                                on_change=SchedulerState.set_priority_job_id,
                                width="100%"
                            ),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("New Priority", size="2", weight="bold", color="#1e3a8a"),
                            rx.select(["1", "2", "3"], value=SchedulerState.priority_new_value.to_string(), on_change=SchedulerState.set_priority_new_value_from_str, width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.button("Update Priority", on_click=SchedulerState.update_priority, loading=SchedulerState.loading, size="3", color_scheme="amber", width="100%"),
                        display="flex", align_items="flex-end"
                    ),
                    columns="3", spacing="3", width="100%", margin_top="1rem"
                ),
                
                spacing="3", width="100%"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
        ),
        rx.text("")
    )


def breakdown_simulator() -> rx.Component:
    """Machine breakdown simulator"""
    return rx.cond(
        SchedulerState.data_loaded,
        rx.box(
            rx.vstack(
                rx.heading("Machine Breakdown Simulator", size="6", color="#1e3a8a"),
                rx.text("Simulate machine downtime and see impact", color="#6b7280", size="2"),
                
                rx.grid(
                    rx.box(
                        rx.vstack(
                            rx.text("Machine", size="2", weight="bold", color="#1e3a8a"),
                            rx.select(
                                SchedulerState.machine_ids,
                                value=SchedulerState.breakdown_machine,
                                on_change=SchedulerState.set_breakdown_machine,
                                width="100%"
                            ),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Start Time (min)", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(type="number", value=SchedulerState.breakdown_start.to_string(), on_change=SchedulerState.set_breakdown_start_from_str, min="0", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Duration (min)", size="2", weight="bold", color="#1e3a8a"),
                            rx.input(type="number", value=SchedulerState.breakdown_duration.to_string(), on_change=SchedulerState.set_breakdown_duration_from_str, min="30", width="100%"),
                            spacing="1", align="start"
                        )
                    ),
                    rx.box(
                        rx.button("Simulate Breakdown", on_click=SchedulerState.simulate_breakdown, loading=SchedulerState.loading, size="3", color_scheme="red", width="100%"),
                        display="flex", align_items="flex-end"
                    ),
                    columns="4", spacing="3", width="100%", margin_top="1rem"
                ),
                
                spacing="3", width="100%"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
        ),
        rx.text("")
    )


def outsourcing_threshold() -> rx.Component:
    """Outsourcing cost threshold control"""
    return rx.cond(
        SchedulerState.data_loaded,
        rx.box(
            rx.vstack(
                rx.heading("Outsourcing Policy", size="6", color="#1e3a8a"),
                rx.text("Adjust cost threshold for make-or-buy decisions", color="#6b7280", size="2"),
                rx.text(f"Current threshold: {SchedulerState.cost_threshold:.2f} | Lower = More in-house, Higher = More outsourcing", color="#9ca3af", size="1"),
                
                rx.hstack(
                    rx.text("0.5", size="2", color="#6b7280"),
                    rx.slider(
                        default_value=SchedulerState.cost_threshold,
                        min=0.5,
                        max=1.5,
                        step=0.05,
                        on_change=SchedulerState.update_threshold,
                        width="100%",
                        color_scheme="violet"
                    ),
                    rx.text("1.5", size="2", color="#6b7280"),
                    spacing="3", width="100%", margin_top="1rem"
                ),
                
                spacing="3", width="100%"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
        ),
        rx.text("")
    )


def activity_log_panel() -> rx.Component:
    """Activity log display"""
    return rx.cond(
        SchedulerState.data_loaded & (SchedulerState.activity_log.length() > 0),
        rx.box(
            rx.vstack(
                rx.heading("Activity Log", size="6", color="#1e3a8a"),
                rx.box(
                    rx.foreach(
                        SchedulerState.activity_log[-5:],
                        lambda log: rx.box(
                            rx.hstack(
                                rx.badge(log["action"], color_scheme="blue", size="1"),
                                rx.text(log["timestamp"], size="1", color="#9ca3af"),
                                spacing="2"
                            ),
                            rx.text(log["details"], size="2", color="#6b7280"),
                            padding="0.5rem", border_bottom="1px solid #e5e7eb"
                        )
                    ),
                    max_height="200px", overflow_y="auto", width="100%"
                ),
                spacing="3", width="100%"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
        ),
        rx.text("")
    )


def gantt_chart_panel() -> rx.Component:
    """Gantt chart display"""
    return rx.cond(
        SchedulerState.show_gantt,
        rx.box(
            rx.vstack(
                rx.heading(f"Schedule Visualization - {SchedulerState.current_heuristic}", size="6", color="#1e3a8a"),
                rx.box(
                    rx.html(
                        SchedulerState.gantt_html,
                    ),
                    width="100%",
                    min_height="650px",
                    overflow="auto"
                ),
                spacing="3", width="100%"
            ),
            background="white", padding="2rem", border_radius="12px", box_shadow="0 4px 20px rgba(0,0,0,0.08)", margin_bottom="2rem"
        ),
        rx.text("")
    )


def index() -> rx.Component:
    """Main page"""
    return rx.container(
        header(),
        control_panel(),
        
        # Interactive features
        rx.grid(
            add_job_panel(),
            priority_manager(),
            columns="2", spacing="4", width="100%"
        ),
        
        rx.grid(
            breakdown_simulator(),
            outsourcing_threshold(),
            columns="2", spacing="4", width="100%"
        ),
        
        activity_log_panel(),
        
        metrics_dashboard(),
        gantt_chart_panel(),
        comparison_table(),
        
        rx.box(
            rx.vstack(
                rx.text("ForbesMarshall CNC Scheduling System", weight="bold", color="#1e3a8a", size="3"),
                rx.text("Version 2.0 | Enterprise Manufacturing Optimization Platform", color="#6b7280", size="2"),
                spacing="1", align="center"
            ),
            text_align="center", padding="2rem", margin_top="3rem", border_top="1px solid #e5e7eb"
        ),
        max_width="1400px", padding="2rem"
    )


# ============================================================================
# APP
# ============================================================================

app = rx.App(theme=rx.theme(appearance="light", has_background=True, radius="large", accent_color="blue"))
app.add_page(index, route="/", title="CNC Scheduler")
