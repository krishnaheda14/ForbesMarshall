# backend/cnc_scheduler_core.py
"""
Core CNC Scheduling Logic - Extracted and Cleaned for Backend API
Contains scheduling algorithms and helper functions without UI dependencies.
"""

import pandas as pd
import numpy as np
import time

# ---------------------------
# Helper Functions
# ---------------------------

def parse_maintenance(maintenance_str):
    """Parse maintenance window string into structured data"""
    if pd.isna(maintenance_str) or str(maintenance_str).lower() == 'none':
        return []
    try:
        if isinstance(maintenance_str, list):
            return maintenance_str
            
        parts = str(maintenance_str).replace("Day", "").replace(",", "").strip().split()
        if len(parts) < 2: return []
        
        day_str = parts[0]
        times = parts[1].split('-')
        start_hour, start_min = map(int, times[0].split(':'))
        end_hour, end_min = map(int, times[1].split(':'))
        
        days_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        day_offset = days_map.get(day_str, 0) * 24 * 60
        
        start_time = day_offset + (start_hour * 60) + start_min
        end_time = day_offset + (end_hour * 60) + end_min
        
        return [{'start': start_time, 'end': end_time, 'duration': end_time - start_time}]
    except Exception:
        return []

def get_eligible_machines(op_type):
    """Get list of machines eligible for an operation type"""
    mapping = {
        'MILLING': ['M1', 'M3', 'M4'],
        'TURNING': ['M6', 'M9'],
        'GRINDING': ['M6', 'M9', 'M5'],
        'DRILLING': ['M1', 'M3', 'M4', 'M2']
    }
    return mapping.get(str(op_type).upper(), [])

def calculate_inhouse_cost(operation, df_effective, hourly_rate=30):
    """Calculate in-house production cost for an operation"""
    op_id = operation['Operation_ID']
    eligible = df_effective[df_effective['Operation_ID'] == op_id]

    if len(eligible) == 0:
        return None, None

    best_option = eligible.loc[eligible['Total_Time'].idxmin()]
    labor_cost = (best_option['Total_Time'] / 60) * hourly_rate
    material_cost = operation.get('Quantity', 1) * 0.5 
    total_cost = labor_cost + material_cost

    return total_cost, best_option['Machine_ID']

def make_or_buy_decision(operation, df_effective, threshold=0.9):
    """
    Make-or-buy decision.
    Outsources ONLY if:
    1. It meets the cost threshold
    2. AND the vendor can deliver before the Due Date
    """
    if operation.get('Outsource_Flag') != 'Y':
        return None
        
    result = calculate_inhouse_cost(operation, df_effective)
    # If we physically can't do it in-house (no machine), we MUST outsource regardless of time
    if not result or result[0] is None:
        return ('OUTSOURCE', operation.get('Outsource_Cost', 0))
    
    inhouse_cost, _ = result
    outsource_cost = operation.get('Outsource_Cost', float('inf'))
    
    # Data Fix
    if outsource_cost <= 0:
        outsource_cost = inhouse_cost * 1.2 
    
    # 1. Check Cost
    is_cost_effective = outsource_cost < (inhouse_cost * threshold)
    
    # 2. Check Time (The Fix)
    # Vendor time includes shipping/lead time (often days)
    # In-house time is just processing + setup (often hours)
    outsource_time = operation.get('Outsource_Time_Min', float('inf'))
    release_time = operation.get('Release_Time_Min', 0)
    due_time = operation.get('Due_Time_Min', float('inf'))
    
    # Will outsourcing miss the deadline?
    will_be_late = (release_time + outsource_time) > due_time
    
    # DECISION LOGIC:
    # If it's cost effective AND won't be late -> Outsource
    if is_cost_effective and not will_be_late:
        return ('OUTSOURCE', outsource_cost)
        
    return None
def get_setup_penalty(prev_material, next_material, df_penalties):
    """Get setup penalty for material changeover"""
    if not prev_material or not next_material:
        return 0
    try:
        penalty = df_penalties[
            (df_penalties['Previous_Material'] == prev_material) &
            (df_penalties['Next_Material'] == next_material)
        ]
        return penalty.iloc[0]['Penalty_Min'] if not penalty.empty else 0
    except Exception:
        return 0

def calculate_metrics(schedule_df, df_ops, heuristic_name='SPT'):
    """Calculate performance metrics for a schedule"""
    if schedule_df.empty:
        return {}

    makespan = schedule_df['End_Time'].max() / (8 * 60)
    tardiness = schedule_df['Tardiness'].sum() / (8 * 60)
    late_ops = (schedule_df['Tardiness'] > 0).sum()
    total_ops = len(schedule_df)
    on_time_pct = ((total_ops - late_ops) / total_ops * 100) if total_ops > 0 else 0
    
    total_proc = schedule_df['Proc_Time'].sum()
    num_machines = schedule_df['Machine_ID'].nunique()
    total_avail = schedule_df['End_Time'].max() * num_machines
    utilization = (total_proc / total_avail * 100) if total_avail > 0 else 0
    
    total_cost = (schedule_df['Proc_Time'].sum() / 60 * 30)
    
    return {
        'Heuristic': heuristic_name,
        'Makespan_Days': round(makespan, 2),
        'Total_Tardiness_Days': round(tardiness, 2),
        'Late_Operations': int(late_ops),
        'Total_Operations': int(total_ops),
        'On_Time_%': round(on_time_pct, 1),
        'Machine_Utilization_%': round(utilization, 1),
        'Total_Cost_$': round(total_cost, 2)
    }

def analyze_capacity_for_new_job(new_job, current_schedule):
    return {"status": "Not implemented in basic core"}


# ---------------------------
# CNC Scheduler Class
# ---------------------------

class CNCScheduler:
    """Main scheduling engine for CNC operations"""
    
    def __init__(self, df_ops, df_machines, df_effective, df_penalties):
        self.df_ops = df_ops.copy()
        self.df_machines = df_machines.copy()
        self.df_effective = df_effective.copy()
        self.df_penalties = df_penalties

        self.machine_availability = {m: 0 for m in df_machines['Machine_ID']}
        self.machine_last_material = {m: None for m in df_machines['Machine_ID']}

        self.schedule = []
        self.op_completion_times = {}

    def reset(self):
        self.machine_availability = {m: 0 for m in self.df_machines['Machine_ID']}
        self.machine_last_material = {m: None for m in self.df_machines['Machine_ID']}
        self.schedule = []
        self.op_completion_times = {}

    def get_earliest_available_time(self, machine_id, release_time, duration):
        candidate_start = max(self.machine_availability.get(machine_id, 0), release_time)
        try:
            machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id]
            if machine_row.empty: return candidate_start
            maintenance = machine_row.iloc[0].get('Maintenance_Window', None)
        except Exception:
            return candidate_start
        
        if not maintenance: return candidate_start
            
        windows = maintenance if isinstance(maintenance, list) else [maintenance]
        windows.sort(key=lambda w: w.get('start', 0))
        
        for attempt in range(100):
            candidate_end = candidate_start + duration
            conflicts = False
            for window in windows:
                w_start = window.get('start', 0)
                w_end = window.get('end', 0)
                if w_end <= w_start: continue
                if candidate_start < w_end and candidate_end > w_start:
                    candidate_start = w_end
                    conflicts = True
                    break
            if not conflicts: return candidate_start
        
        return candidate_start + duration

    def get_available_operations(self):
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
                            out_time = pred.get('Release_Time_Min', 0) + pred.get('Outsource_Time_Min', 0)
                            earliest_start = max(earliest_start, out_time)
                            self.op_completion_times[pred['Operation_ID']] = out_time
                        else:
                            all_pred_done = False
                            break
                    else:
                        earliest_start = max(earliest_start, self.op_completion_times[pred['Operation_ID']])
            if all_pred_done:
                available.append((op, earliest_start))
        return available

    def find_best_machine(self, operation, earliest_start_time):
        op_id = operation['Operation_ID']
        eligible = self.df_effective[self.df_effective['Operation_ID'] == op_id]
        if len(eligible) == 0: return None, float('inf')

        best_machine = None
        best_completion = float('inf')
        
        for _, machine_option in eligible.iterrows():
            machine_id = machine_option['Machine_ID']
            eff_time = machine_option['Effective_Proc_Time']
            prev_material = self.machine_last_material.get(machine_id)
            setup_penalty = get_setup_penalty(prev_material, operation.get('Mat_Type', None), self.df_penalties)
            actual_setup_time = operation.get('Setup_Time', 0) + setup_penalty
            transfer_time = operation.get('Transfer_Min', 0)
            total_duration = actual_setup_time + eff_time + transfer_time
            start_time = self.get_earliest_available_time(machine_id, earliest_start_time, total_duration)
            
            if (start_time + total_duration) < best_completion:
                best_completion = start_time + total_duration
                best_machine = machine_id
                
        return best_machine, best_completion

    def schedule_operation(self, operation, machine_id, earliest_start_time):
        op_id = operation['Operation_ID']
        op_details_query = self.df_effective[
            (self.df_effective['Operation_ID'] == op_id) &
            (self.df_effective['Machine_ID'] == machine_id)
        ]
        if len(op_details_query) == 0: return False

        op_details = op_details_query.iloc[0]
        eff_time = op_details['Effective_Proc_Time']
        prev_material = self.machine_last_material.get(machine_id)
        setup_penalty = get_setup_penalty(prev_material, operation.get('Mat_Type', None), self.df_penalties)
        actual_setup_time = operation.get('Setup_Time', 0) + setup_penalty
        transfer_time = operation.get('Transfer_Min', 0)
        total_duration = actual_setup_time + eff_time + transfer_time
        start_time = self.get_earliest_available_time(machine_id, earliest_start_time, total_duration)
        end_time = start_time + total_duration

        self.schedule.append({
            'Operation_ID': op_id,
            'Job_ID': operation['Job_ID'],
            'Machine_ID': machine_id,
            'Start_Time': start_time,
            'End_Time': end_time,
            'Setup_Time': actual_setup_time,
            'Proc_Time': eff_time,
            'Transfer_Time': transfer_time,
            'Due_Time': operation.get('Due_Time_Min', 0),
            'Tardiness': max(0, end_time - operation.get('Due_Time_Min', 0)),
            'Priority': int(operation.get('Priority', 3)),
            'Assignment_Type': 'IN_HOUSE',
            'Outsource_Cost': 0  # In-house cost handled separately, placeholder
        })

        self.machine_availability[machine_id] = end_time
        self.machine_last_material[machine_id] = operation.get('Mat_Type', None)
        self.op_completion_times[op_id] = end_time
        return True

    def select_next_operation(self, available_ops, heuristic='SPT'):
        def safe_priority(op): return int(op.get('Priority', 3))
        
        if heuristic == 'SPT':
            op, earliest_start = min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Total_Proc_Min']))
        elif heuristic == 'EDD':
            op, earliest_start = min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min']))
        elif heuristic == 'CR':
            op, earliest_start = min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min'] / max(x[0]['Total_Proc_Min'], 1)))
        elif heuristic == 'PRIORITY':
            op, earliest_start = min(available_ops, key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min']))
        else:
            op, earliest_start = min(available_ops, key=lambda x: (x[0]['Total_Proc_Min'], x[0]['Due_Time_Min']))

        return op, earliest_start

    def run_scheduling(self, heuristic='SPT', verbose=False):
        self.reset()
        
        # 1. Handle Outsourced Operations First
        outsourced_ops = self.df_ops[self.df_ops.get('Assignment_Type', 'IN_HOUSE') == 'OUTSOURCE']
        for _, op in outsourced_ops.iterrows():
            outsource_time = op.get('Outsource_Time_Min', op.get('Total_Proc_Min', 0))
            release_time = op.get('Release_Time_Min', 0)
            completion = release_time + outsource_time
            
            self.op_completion_times[op['Operation_ID']] = completion
            
            # Added Outsource_Cost here
            self.schedule.append({
                'Operation_ID': op['Operation_ID'],
                'Job_ID': op['Job_ID'],
                'Machine_ID': 'OUTSOURCE',
                'Start_Time': release_time,
                'End_Time': completion,
                'Setup_Time': 0,
                'Proc_Time': 0,
                'Transfer_Time': 0,
                'Due_Time': op.get('Due_Time_Min', 0),
                'Tardiness': max(0, completion - op.get('Due_Time_Min', 0)),
                'Priority': int(op.get('Priority', 3)),
                'Assignment_Type': 'OUTSOURCE',
                'Outsource_Cost': float(op.get('Outsource_Cost', 0)) # <--- CRITICAL FIX
            })

        # 2. Schedule In-House Operations
        non_outsourced = self.df_ops[self.df_ops.get('Assignment_Type', 'IN_HOUSE') != 'OUTSOURCE']
        operations_count = len(non_outsourced)
        scheduled_ops_set = set()

        max_iterations = operations_count * 2 if operations_count > 0 else 1000
        iteration = 0

        while len(scheduled_ops_set) < operations_count:
            iteration += 1
            if iteration > max_iterations: break

            available = self.get_available_operations()
            available = [op for op in available if op[0]['Operation_ID'] not in scheduled_ops_set]

            if not available: break

            next_op, earliest_start_time = self.select_next_operation(available, heuristic=heuristic)
            if next_op is None: break

            best_machine, best_completion = self.find_best_machine(next_op, earliest_start_time)
            
            if best_machine is None:
                # Force outsource if no machine found
                outsource_time = next_op.get('Total_Proc_Min', 0)
                release_time = next_op.get('Release_Time_Min', 0)
                
                # Added Outsource_Cost here too
                self.schedule.append({
                    'Operation_ID': next_op['Operation_ID'],
                    'Job_ID': next_op['Job_ID'],
                    'Machine_ID': 'OUTSOURCE',
                    'Start_Time': release_time,
                    'End_Time': release_time + outsource_time,
                    'Setup_Time': 0,
                    'Proc_Time': 0,
                    'Transfer_Time': 0,
                    'Due_Time': next_op.get('Due_Time_Min', 0),
                    'Tardiness': 0,
                    'Priority': int(next_op.get('Priority', 3)),
                    'Assignment_Type': 'OUTSOURCE',
                    'Outsource_Cost': float(next_op.get('Outsource_Cost', 0)) # <--- CRITICAL FIX
                })
                self.op_completion_times[next_op['Operation_ID']] = release_time + outsource_time
                scheduled_ops_set.add(next_op['Operation_ID'])
                continue

            success = self.schedule_operation(next_op, best_machine, earliest_start_time)
            if success:
                scheduled_ops_set.add(next_op['Operation_ID'])

        return pd.DataFrame(self.schedule)