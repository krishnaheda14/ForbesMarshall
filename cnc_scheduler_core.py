# cnc_scheduler_core.py
"""
Core CNC Scheduling Logic - Extracted from cnc-scheduling.py
This module contains all the scheduling algorithms and helper functions
without any UI dependencies (Streamlit removed).
"""

import pandas as pd
import numpy as np
import time

# ---------------------------
# Helper Functions
# ---------------------------

def parse_maintenance(maintenance_str):
    """Parse maintenance window string into structured data"""
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
            if hour < WORK_START_HOUR or hour > WORK_END_HOUR:
                return None
            return (hour - WORK_START_HOUR) * 60 + minute

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
    """Get list of machines eligible for an operation type"""
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

def calculate_inhouse_cost(operation, df_effective, hourly_rate=30):
    """Calculate in-house production cost for an operation"""
    op_id = operation['Operation_ID']
    eligible = df_effective[df_effective['Operation_ID'] == op_id]

    if len(eligible) == 0:
        return None, None

    best_option = eligible.loc[eligible['Total_Time'].idxmin()]
    labor_cost = (best_option['Total_Time'] / 60) * hourly_rate
    material_cost = operation['Quantity'] * 0.5
    total_cost = labor_cost + material_cost

    return total_cost, best_option['Machine_ID']

def make_or_buy_decision(operation, df_effective, cost_threshold=0.9, hourly_rate=30):
    """Make-or-buy decision for an operation"""
    result = calculate_inhouse_cost(operation, df_effective, hourly_rate)
    if result[0] is None:
        return 'OUTSOURCE', 0, 'No eligible machines'
    
    inhouse_cost, best_machine = result
    inhouse_time = operation.get('Total_Proc_Min', operation.get('Proc_Time_per_Unit', 0) * operation.get('Quantity', 1)) + operation.get('Setup_Time', 0)
    outsource_cost = operation.get('Outsource_Cost', np.inf)
    outsource_time = operation.get('Outsource_Time_Min', np.inf)

    if outsource_cost <= 0 or outsource_cost == np.inf:
        return 'IN_HOUSE', inhouse_cost, 'Best in-house'
    
    earliest_start = operation.get('Release_Time_Min', 0)
    earliest_finish = earliest_start + inhouse_time
    can_meet_deadline = earliest_finish <= operation.get('Due_Time_Min', np.inf)

    if not can_meet_deadline and outsource_time < inhouse_time:
        return 'OUTSOURCE', outsource_cost, 'Cannot meet deadline in-house'
    if outsource_cost < (inhouse_cost * cost_threshold):
        return 'OUTSOURCE', outsource_cost, 'More cost-effective'
    return 'IN_HOUSE', inhouse_cost, 'Best in-house'

def get_setup_penalty(prev_material, next_material, df_penalties):
    """Get setup penalty for material changeover"""
    if not prev_material or not next_material:
        return 0
    penalty = df_penalties[
        (df_penalties['Previous_Material'] == prev_material) &
        (df_penalties['Next_Material'] == next_material)
    ]
    return penalty.iloc[0]['Penalty_Time_(min)'] if len(penalty) > 0 else 15

def calculate_metrics(schedule_df, df_ops, heuristic_name, hourly_rate=30):
    """Calculate performance metrics for a schedule"""
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

    machine_count = schedule_df['Machine_ID'].nunique() if not schedule_df.empty else 5

    total_setup_time = schedule_df['Setup_Time'].sum() if 'Setup_Time' in schedule_df.columns else 0
    total_proc_time = schedule_df['Proc_Time'].sum() if 'Proc_Time' in schedule_df.columns else 0
    total_transfer_time = schedule_df['Transfer_Time'].sum() if 'Transfer_Time' in schedule_df.columns else 0
    total_productive_time = total_setup_time + total_proc_time + total_transfer_time

    total_available_time = machine_count * makespan_min if makespan_min > 0 else machine_count * 1
    utilization = (total_productive_time / total_available_time) * 100 if total_available_time > 0 else 0

    inhouse_cost = schedule_df['Proc_Time'].sum() / 60 * hourly_rate if 'Proc_Time' in schedule_df.columns else 0
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
        'Total_Cost_$': round(total_cost, 2)
    }

def analyze_capacity_for_new_job(new_job_ops, current_schedule, df_machines, df_effective, due_time_min):
    """Analyze capacity for scheduling a new job"""
    analysis = {
        'feasible': False,
        'recommendation': 'OUTSOURCE',
        'reasons': [],
        'metrics': {}
    }

    if not current_schedule.empty:
        current_makespan = current_schedule['End_Time'].max()
        total_machines = len(df_machines)
        total_productive = (current_schedule['Setup_Time'].sum() +
                           current_schedule['Proc_Time'].sum() +
                           current_schedule['Transfer_Time'].sum())
        avg_utilization = (total_productive / (current_makespan * total_machines)) * 100 if current_makespan > 0 else 0
        analysis['metrics']['current_makespan_days'] = current_makespan / 480
        analysis['metrics']['current_utilization'] = round(avg_utilization, 1)
    else:
        current_makespan = 0
        analysis['metrics']['current_makespan_days'] = 0
        analysis['metrics']['current_utilization'] = 0

    total_new_time = 0
    operations_schedulable = 0

    for op in new_job_ops:
        op_id = op['Operation_ID']
        eligible_times = df_effective[df_effective['Operation_ID'] == op_id]
        if not eligible_times.empty:
            min_time = eligible_times['Total_Time'].min()
            total_new_time += min_time
            operations_schedulable += 1

    if operations_schedulable == 0:
        analysis['feasible'] = False
        analysis['recommendation'] = 'OUTSOURCE'
        analysis['reasons'].append("No eligible machines found for operations")
        return analysis

    if operations_schedulable < len(new_job_ops):
        analysis['reasons'].append(f"Only {operations_schedulable}/{len(new_job_ops)} operations can be scheduled in-house")

    estimated_completion = current_makespan + total_new_time
    analysis['metrics']['estimated_completion_days'] = estimated_completion / 480
    analysis['metrics']['due_date_days'] = due_time_min / 480
    analysis['metrics']['new_job_time_days'] = total_new_time / 480

    deadline_buffer = due_time_min - estimated_completion
    analysis['metrics']['deadline_buffer_days'] = deadline_buffer / 480

    if deadline_buffer < 0:
        analysis['feasible'] = False
        analysis['recommendation'] = 'OUTSOURCE'
        analysis['reasons'].append(f"Cannot meet deadline - Need {abs(deadline_buffer)/480:.1f} more days")
    else:
        analysis['feasible'] = True
        analysis['recommendation'] = 'SCHEDULE'
        analysis['reasons'].append(f"Can meet deadline with {deadline_buffer/480:.1f} days buffer")

    return analysis

# ---------------------------
# CNC Scheduler Class
# ---------------------------

class CNCScheduler:
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
        """Find earliest available time slot considering maintenance windows"""
        current_avail = max(self.machine_availability.get(machine_id, 0), release_time)

        try:
            machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id].iloc[0]
            maintenance = machine_row.get('Maintenance_Window')
        except Exception:
            return current_avail

        if maintenance is None or (isinstance(maintenance, dict) and not maintenance):
            return current_avail

        maintenance_list = (
            [maintenance] if isinstance(maintenance, dict)
            else [m for m in maintenance if isinstance(m, dict)]
        )

        maintenance_list.sort(key=lambda mw: mw.get('start', 0))

        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            overlaps = False
            for mw in maintenance_list:
                mw_start = mw.get('start', 0)
                mw_end = mw.get('end', 0)
                
                job_end = current_avail + duration
                
                if not (job_end <= mw_start or current_avail >= mw_end):
                    current_avail = mw_end
                    overlaps = True
                    break
            
            if not overlaps:
                break
            
            iteration += 1

        return current_avail

    def get_available_operations(self):
        """Get list of operations ready to be scheduled"""
        available = []
        for idx, op in self.df_ops.iterrows():
            if op.get('Assignment_Type', 'IN_HOUSE') == 'OUTSOURCE':
                continue
            
            op_id = op['Operation_ID']
            if op_id in self.op_completion_times:
                continue
            
            op_seq = op['Op_Seq']
            if op_seq == 1:
                available.append(op)
            else:
                job_id = op['Job_ID']
                prev_seq = op_seq - 1
                prev_ops = self.df_ops[
                    (self.df_ops['Job_ID'] == job_id) &
                    (self.df_ops['Op_Seq'] == prev_seq)
                ]
                if len(prev_ops) > 0:
                    prev_op_id = prev_ops.iloc[0]['Operation_ID']
                    if prev_op_id in self.op_completion_times:
                        available.append(op)
        return available

    def find_best_machine(self, operation, earliest_start_time):
        """Find best machine for an operation"""
        op_id = operation['Operation_ID']
        eligible = self.df_effective[self.df_effective['Operation_ID'] == op_id]
        if len(eligible) == 0:
            return None, None

        best_machine = None
        best_completion = float('inf')
        for _, machine_option in eligible.iterrows():
            machine_id = machine_option['Machine_ID']
            duration = machine_option['Total_Time']
            start = self.get_earliest_available_time(machine_id, earliest_start_time, duration)
            completion = start + duration
            if completion < best_completion:
                best_completion = completion
                best_machine = machine_id
        return best_machine, best_completion

    def schedule_operation(self, operation, machine_id, earliest_start_time):
        """Schedule an operation on a machine"""
        op_id = operation['Operation_ID']
        op_details_query = self.df_effective[
            (self.df_effective['Operation_ID'] == op_id) &
            (self.df_effective['Machine_ID'] == machine_id)
        ]
        if len(op_details_query) == 0:
            return False

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
            'Tardiness': max(0, end_time - operation.get('Due_Time_Min', 0))
        })

        self.machine_availability[machine_id] = end_time
        self.machine_last_material[machine_id] = operation.get('Mat_Type', None)
        self.op_completion_times[op_id] = end_time
        return True

    def select_next_operation(self, available_ops, heuristic='SPT'):
        """Select next operation based on heuristic"""
        def safe_priority(op):
            return int(op.get('Priority', 3))

        if heuristic == 'SPT':
            # SPT: choose by shortest processing time only (no priority preference)
            return min(available_ops, key=lambda op: (op.get('Total_Proc_Min', 0), op.get('Due_Time_Min', 0), op.get('Operation_ID')))
        elif heuristic == 'EDD':
            # EDD: choose by earliest due date only (no priority preference)
            return min(available_ops, key=lambda op: (op.get('Due_Time_Min', 0), op.get('Total_Proc_Min', 0), op.get('Operation_ID')))
        elif heuristic == 'CR':
            # Critical Ratio: (Due_Time - Release_Time) / Total_Proc_Min
            # Lower CR means more urgent (less slack per unit proc time)
            def cr_value(op):
                proc = op.get('Total_Proc_Min', 1) or 1
                window = op.get('Due_Time_Min', 0) - op.get('Release_Time_Min', 0)
                try:
                    return window / proc
                except Exception:
                    return float('inf')

            return min(available_ops, key=lambda op: (cr_value(op), op.get('Due_Time_Min', 0), op.get('Total_Proc_Min', 0), op.get('Operation_ID')))
        elif heuristic == 'PRIORITY':
            return min(available_ops, key=lambda op: safe_priority(op))
        elif heuristic == 'WEIGHTED':
            scored = []
            for op in available_ops:
                urgency_score = op.get('Due_Time_Min', 0) - op.get('Release_Time_Min', 0)
                efficiency_score = op.get('Total_Proc_Min', 0)
                priority_score = safe_priority(op)
                weighted_score = (0.4 * urgency_score) + (0.3 * efficiency_score) + (0.3 * priority_score * 100)
                scored.append((weighted_score, op))
            return min(scored, key=lambda x: x[0])[1]
        elif heuristic == 'SLACK':
            current_time = max(self.machine_availability.values()) if self.machine_availability else 0
            scored = []
            for op in available_ops:
                slack = op.get('Due_Time_Min', 0) - current_time - op.get('Total_Proc_Min', 0)
                scored.append((slack, op))
            return min(scored, key=lambda x: x[0])[1]
        else:
            return available_ops[0]

    def run_scheduling(self, heuristic='SPT', verbose=False):
        """Run scheduling algorithm"""
        self.reset()
        
        # Handle Assignment_Type column safely
        if 'Assignment_Type' in self.df_ops.columns:
            outsourced_ops = self.df_ops[self.df_ops['Assignment_Type'] == 'OUTSOURCE']
            non_outsourced = self.df_ops[self.df_ops['Assignment_Type'] != 'OUTSOURCE']
        else:
            outsourced_ops = self.df_ops[self.df_ops.index < 0]  # Empty DataFrame
            non_outsourced = self.df_ops.copy()
        
        for _, op in outsourced_ops.iterrows():
            self.op_completion_times[op['Operation_ID']] = 0

        operations_count = len(non_outsourced)
        scheduled_ops_set = set()

        max_iterations = operations_count * 2 if operations_count > 0 else 1000
        iteration = 0

        while len(scheduled_ops_set) < operations_count:
            if iteration >= max_iterations:
                break

            available_ops = self.get_available_operations()
            if not available_ops:
                iteration += 1
                continue

            selected_op = self.select_next_operation(available_ops, heuristic)
            op_id = selected_op['Operation_ID']

            if op_id in scheduled_ops_set:
                iteration += 1
                continue

            release_time = selected_op.get('Release_Time_Min', 0)
            job_id = selected_op['Job_ID']
            op_seq = selected_op['Op_Seq']

            if op_seq > 1:
                prev_ops = self.df_ops[
                    (self.df_ops['Job_ID'] == job_id) &
                    (self.df_ops['Op_Seq'] == op_seq - 1)
                ]
                if len(prev_ops) > 0:
                    prev_op_id = prev_ops.iloc[0]['Operation_ID']
                    if prev_op_id in self.op_completion_times:
                        release_time = max(release_time, self.op_completion_times[prev_op_id])

            best_machine, _ = self.find_best_machine(selected_op, release_time)
            if best_machine is None:
                iteration += 1
                continue

            success = self.schedule_operation(selected_op, best_machine, release_time)
            if success:
                scheduled_ops_set.add(op_id)

            iteration += 1

        return pd.DataFrame(self.schedule)
