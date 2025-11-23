# core/scheduler.py
"""
CNC Scheduler - Core scheduling algorithms
"""
import pandas as pd
import streamlit as st
from utils.helpers import get_setup_penalty


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
        """Reset scheduler state"""
        self.machine_availability = {m: 0 for m in self.df_machines['Machine_ID']}
        self.machine_last_material = {m: None for m in self.df_machines['Machine_ID']}
        self.schedule = []
        self.op_completion_times = {}

    def get_earliest_available_time(self, machine_id, release_time, duration):
        """
        Find the earliest time slot that can accommodate a job of given duration,
        ensuring NO overlap with any maintenance/breakdown windows.
        
        Returns the start time that guarantees the job won't interfere with breakdowns.
        """
        # Start from the maximum of machine availability and job release time
        candidate_start = max(self.machine_availability.get(machine_id, 0), release_time)
        
        # Get maintenance/breakdown windows for this machine
        try:
            machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id]
            if machine_row.empty:
                return candidate_start
            
            maintenance = machine_row.iloc[0].get('Maintenance_Window', None)
        except Exception:
            return candidate_start
        
        # No maintenance windows - return immediately
        if not maintenance or (isinstance(maintenance, dict) and not maintenance):
            return candidate_start
        
        # Normalize to list of windows
        if isinstance(maintenance, dict):
            windows = [maintenance]
        elif isinstance(maintenance, list):
            windows = [w for w in maintenance if isinstance(w, dict) and w]
        else:
            return candidate_start
        
        if not windows:
            return candidate_start
        
        # Sort windows by start time
        windows.sort(key=lambda w: w.get('start', 0))
        
        # Find a valid time slot that doesn't overlap with ANY window
        for attempt in range(100):  # Safety limit
            candidate_end = candidate_start + duration
            conflicts = False
            
            for window in windows:
                w_start = window.get('start', 0)
                w_end = window.get('end', 0)
                
                # Skip invalid windows
                if w_end <= w_start:
                    continue
                
                # Check for ANY overlap
                if candidate_start < w_end and candidate_end > w_start:
                    # Conflict found - move candidate_start to after this window
                    candidate_start = w_end
                    conflicts = True
                    break  # Restart check with new candidate_start
            
            # If no conflicts with any window, we found a valid slot
            if not conflicts:
                return candidate_start
        
        # Safety fallback: schedule after all windows
        latest_end = max(w.get('end', 0) for w in windows)
        return latest_end + 10

    def get_available_operations(self):
        """Get list of operations ready to be scheduled"""
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
                            outsource_complete_time = pred.get('Release_Time_Min', 0) + pred.get('Outsource_Time_Min', 0)
                            earliest_start = max(earliest_start, outsource_complete_time)
                            self.op_completion_times[pred['Operation_ID']] = outsource_complete_time
                        else:
                            all_pred_done = False
                            break
                    else:
                        earliest_start = max(earliest_start, self.op_completion_times[pred['Operation_ID']])
            if all_pred_done:
                available.append((op, earliest_start))
        return available

    def find_best_machine(self, operation, earliest_start_time):
        """Find the best machine for an operation"""
        op_id = operation['Operation_ID']
        eligible = self.df_effective[self.df_effective['Operation_ID'] == op_id]
        if len(eligible) == 0:
            return None, float('inf')

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
                # Check if this slot overlaps any breakdown
                machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id]
                maintenance = machine_row.iloc[0].get('Maintenance_Window', None) if not machine_row.empty else None
                overlaps_breakdown = False
                if maintenance:
                    windows = [maintenance] if isinstance(maintenance, dict) else [w for w in maintenance if isinstance(w, dict) and w] if isinstance(maintenance, list) else []
                    for window in windows:
                        w_start = window.get('start', 0)
                        w_end = window.get('end', 0)
                        if start_time < w_end and (start_time + total_duration) > w_start:
                            overlaps_breakdown = True
                            break
                if not overlaps_breakdown and (start_time + total_duration) < best_completion:
                    best_completion = start_time + total_duration
                    best_machine = machine_id
            return best_machine, best_completion

    def schedule_operation(self, operation, machine_id, earliest_start_time):
        """
        Schedule an operation on a specific machine, ensuring NO overlap with breakdowns.
        Updates machine availability to the NEXT valid time (accounting for breakdowns).
        """
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

        # Get the safe start time (avoids all breakdowns)
        start_time = self.get_earliest_available_time(machine_id, earliest_start_time, total_duration)
        end_time = start_time + total_duration

        # Add to schedule
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
            'Priority': operation.get('Priority', 3),
            'Assignment_Type': operation.get('Assignment_Type', 'IN_HOUSE')
        })

        # Set machine availability to the NEXT safe time after this job
        next_availability = end_time
        
        try:
            machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id]
            if not machine_row.empty:
                maintenance = machine_row.iloc[0].get('Maintenance_Window', None)
                
                if maintenance:
                    # Normalize to list
                    if isinstance(maintenance, dict):
                        windows = [maintenance]
                    elif isinstance(maintenance, list):
                        windows = [w for w in maintenance if isinstance(w, dict) and w]
                    else:
                        windows = []
                    
                    # Check if end_time falls inside ANY breakdown window
                    for window in windows:
                        w_start = window.get('start', 0)
                        w_end = window.get('end', 0)
                        
                        # If job ends during a breakdown window, push availability to after breakdown
                        if w_start <= end_time < w_end:
                            next_availability = w_end
                            break
        except Exception:
            pass
        
        # Update machine availability for next job
        self.machine_availability[machine_id] = next_availability
        self.machine_last_material[machine_id] = operation.get('Mat_Type', None)
        self.op_completion_times[op_id] = end_time
        return True

    def select_next_operation(self, available_ops, heuristic='SPT'):
        """Select next operation based on heuristic"""
        def safe_priority(op):
            return int(op.get('Priority', 3))
        
        # Selection logic based on heuristic
        if heuristic == 'SPT':
            # Shortest Processing Time - schedule shortest jobs first
            op, earliest_start = min(
                available_ops,
                key=lambda x: (x[0]['Total_Proc_Min'], x[0]['Due_Time_Min'])
            )
        elif heuristic == 'EDD':
            # Earliest Due Date - schedule jobs with nearest deadlines first
            op, earliest_start = min(
                available_ops,
                key=lambda x: (x[0]['Due_Time_Min'], x[0]['Total_Proc_Min'])
            )
        elif heuristic == 'CR':
            # Critical Ratio - schedule based on due_date/processing_time ratio
            op, earliest_start = min(
                available_ops,
                key=lambda x: (x[0]['Due_Time_Min'] / max(x[0]['Total_Proc_Min'], 1), x[0]['Total_Proc_Min'])
            )
        elif heuristic == 'PRIORITY':
            # Priority-based - ONLY this heuristic uses job priority as primary criterion
            op, earliest_start = min(
                available_ops,
                key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min'], x[0]['Total_Proc_Min'])
            )
        # ...existing code...
        else:
            # Default fallback to SPT
            op, earliest_start = min(
                available_ops,
                key=lambda x: (x[0]['Total_Proc_Min'], x[0]['Due_Time_Min'])
            )

        return op, earliest_start

    def run_scheduling(self, heuristic='SPT', verbose=True):
        """Run the complete scheduling algorithm"""
        if verbose:
            st.write(f"🔄 Starting {heuristic} scheduling...")

        self.reset()
        outsourced_ops = self.df_ops[self.df_ops.get('Assignment_Type', 'IN_HOUSE') == 'OUTSOURCE']
        if verbose:
            st.write(f"  ✓ Outsourced: {len(outsourced_ops)} operations (auto-complete)")

        for _, op in outsourced_ops.iterrows():
            self.op_completion_times[op['Operation_ID']] = op.get('Release_Time_Min', 0) + op.get('Outsource_Time_Min', 0)

        non_outsourced = self.df_ops[self.df_ops.get('Assignment_Type', 'IN_HOUSE') != 'OUTSOURCE']
        operations_count = len(non_outsourced)
        scheduled_ops_set = set()

        if verbose:
            st.write(f"  ✓ Scheduling in-house: {operations_count} operations")

        max_iterations = operations_count * 2 if operations_count > 0 else 1000
        iteration = 0

        while len(scheduled_ops_set) < operations_count:
            iteration += 1
            if iteration > max_iterations:
                st.warning(f"⚠️ Max iterations reached ({iteration})")
                break

            available = self.get_available_operations()
            available = [op for op in available if op[0]['Operation_ID'] not in scheduled_ops_set]

            if not available:
                st.write("⚠️ No available operations remaining.")
                break

            next_op, earliest_start_time = self.select_next_operation(available, heuristic=heuristic)
            if next_op is None:
                st.write("⚠️ No operation selected — stopping.")
                break

            best_machine, best_completion = self.find_best_machine(next_op, earliest_start_time)
            if best_machine is None:
                # No available machine without breakdown conflict, outsource if allowed
                next_op['Assignment_Type'] = 'OUTSOURCE'
                outsource_time = next_op.get('Outsource_Time_Min', next_op.get('Total_Proc_Min', 0))
                release_time = next_op.get('Release_Time_Min', 0)
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
                    'Tardiness': max(0, (release_time + outsource_time) - next_op.get('Due_Time_Min', 0)),
                    'Priority': next_op.get('Priority', 3),
                    'Assignment_Type': 'OUTSOURCE'
                })
                self.op_completion_times[next_op['Operation_ID']] = release_time + outsource_time
                scheduled_ops_set.add(next_op['Operation_ID'])
                continue

            success = self.schedule_operation(next_op, best_machine, earliest_start_time)
            if success:
                scheduled_ops_set.add(next_op['Operation_ID'])

            if verbose and iteration % 100 == 0:
                st.write(f"⏳ Progress: {len(scheduled_ops_set)}/{operations_count} scheduled")

        if verbose:
            st.success(f"✅ {heuristic} scheduling complete: {len(scheduled_ops_set)}/{operations_count} operations scheduled")

        st.session_state.triggered_by_priority_manager = False
        return pd.DataFrame(self.schedule)
