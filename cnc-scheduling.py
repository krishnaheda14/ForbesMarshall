# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini AI
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        AI_ENABLED = True
    else:
        AI_ENABLED = False
        st.warning("⚠️ Gemini API key not found in .env file. AI insights disabled.")
except Exception as e:
    AI_ENABLED = False
    print(f"AI initialization failed: {e}")

def trigger_recompute_prompt(ss, label: str):
    """
    Unified helper to handle post-update behavior for:
    - Machine Breakdown
    - Priority Update
    - Outsourcing Policy Update
    """
    # Display user-facing success + guidance
    st.success(f"✅ {label} completed successfully.")
    st.info("💡 Please click **'🧪 Compute All Heuristics'** in the sidebar "
            "to recompute schedules and view updated recommendations.")

    # Set session state to prepare heuristic recomputation
    ss.recalculate_all_heuristics = True
    ss.breakdown_message_visible = True
    ss.current_page = "comparison"

    # Clear Streamlit caches to avoid stale data
    st.cache_data.clear()
    st.cache_resource.clear()

    # Small visual feedback toast
    st.toast("⚙ Update registered — ready for heuristic recomputation.", icon="⚙️")

# ---------------------------
# AI Insights Helper
# ---------------------------
def get_ai_insights(prompt, context_data=None):
    """
    Generate AI-powered insights using Gemini.
    
    Args:
        prompt: The question or topic to get insights about
        context_data: Optional dict with relevant data to provide context
    
    Returns:
        str: AI-generated insights or error message
    """
    if not AI_ENABLED:
        return "❌ AI insights are disabled. Please add GEMINI_API_KEY to your .env file."
    
    try:
        # Build enhanced prompt with context
        full_prompt = f"""
You are an expert in manufacturing scheduling, operations research, and production planning.
You are analyzing a CNC job scheduling system.

{prompt}
"""
        
        if context_data:
            full_prompt += "\n\nContext Data:\n"
            for key, value in context_data.items():
                full_prompt += f"- {key}: {value}\n"
        
        full_prompt += "\n\nProvide clear, actionable insights in 3-5 concise bullet points. Be specific and practical."
        
        # Generate response
        response = gemini_model.generate_content(full_prompt)
        return response.text
    
    except Exception as e:
        return f"❌ Error generating AI insights: {str(e)}"

# ---------------------------
# Helper: make debug visible
# ---------------------------
def dbg(msg):
    # simple debug helper that writes to Streamlit if available
    try:
        st.write(msg)
    except Exception:
        print(msg)

# ---------------------------
# Old helpers (from your code)
# ---------------------------
def parse_maintenance(maintenance_str):
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

def calculate_inhouse_cost(operation, df_effective, hourly_rate=50):
    op_id = operation['Operation_ID']
    eligible = df_effective[df_effective['Operation_ID'] == op_id]

    if len(eligible) == 0:
        return float('inf'), None

    best_option = eligible.loc[eligible['Total_Time'].idxmin()]
    labor_cost = (best_option['Total_Time'] / 60) * hourly_rate
    material_cost = operation['Quantity'] * 0.5
    total_cost = labor_cost + material_cost

    return total_cost, best_option['Machine_ID']

def make_or_buy_decision(operation, df_effective, cost_threshold=0.9, hourly_rate=50):
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

def get_setup_penalty(prev_material, next_material, df_penalties):
    if not prev_material or not next_material:
        return 0
    penalty = df_penalties[
        (df_penalties['Previous Material'] == prev_material) &
        (df_penalties['Next Material'] == next_material)
    ]
    return penalty.iloc[0]['Penalty Time (min)'] if len(penalty) > 0 else 15

# ---------------------------
# Metrics (unchanged logic)
# ---------------------------

def calculate_metrics(schedule_df, df_ops, heuristic_name):
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

    if 'base_df_machines' in st.session_state:
        machine_count = len(st.session_state.base_df_machines)
    else:
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

def refresh_all_heuristics_metrics(ss):
    heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
    metrics = []

    for heur in heuristics:
        schedule_key = f'schedule_{heur.lower()}'
        schedule = getattr(ss, schedule_key, None)

        if schedule is None or ss.get('force_metric_refresh', False):
            scheduler = CNCScheduler(
                ss.base_df_ops,
                ss.base_df_machines,
                ss.base_df_effective,
                ss.base_df_penalties
            )
            schedule = scheduler.run_scheduling(heuristic=heur)
            setattr(ss, schedule_key, schedule)

        metrics.append(calculate_metrics(schedule, ss.base_df_ops, heur))

    ss.df_metrics = pd.DataFrame(metrics)
    ss.force_metric_refresh = False

# ---------------------------
# CNCScheduler (unchanged)
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
        """
        Determine the earliest available start time for a machine, considering
        current availability, job release time, and any maintenance/breakdown windows.
        """
        # Start from max of current availability and job release
        current_avail = max(self.machine_availability.get(machine_id, 0), release_time)

        # Try fetching machine info safely
        try:
            machine_row = self.df_machines[self.df_machines['Machine_ID'] == machine_id]
            if machine_row.empty:
                return current_avail
            maintenance = machine_row.iloc[0].get('Maintenance_Window', None)
        except Exception:
            return current_avail

        # ✅ Handle no maintenance/breakdown case
        if maintenance is None or (isinstance(maintenance, dict) and not maintenance):
            return current_avail

        # ✅ Normalize to a list for multiple windows
        maintenance_list = (
            [maintenance] if isinstance(maintenance, dict)
            else [m for m in maintenance if isinstance(m, dict)]
        )

        # Sort maintenance windows by start time for safety
        maintenance_list.sort(key=lambda mw: mw.get('start', 0))

        # ✅ Adjust for any overlap between current availability and maintenance windows
        # Keep iterating until we find a time slot that doesn't conflict with any breakdown/maintenance
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            adjusted = False
            end_time = current_avail + duration
            
            for mw in maintenance_list:
                mw_start = mw.get('start', 0)
                mw_end = mw.get('end', 0)

                # Skip invalid windows
                if mw_end <= mw_start:
                    continue

                # ✅ CRITICAL: Check if operation [current_avail, end_time] overlaps breakdown [mw_start, mw_end]
                # Overlap occurs when: operation_start < breakdown_end AND operation_end > breakdown_start
                overlap = (current_avail < mw_end) and (end_time > mw_start)
                
                if overlap:
                    # ✅ Move operation to start AFTER the breakdown/maintenance window
                    current_avail = mw_end
                    adjusted = True
                    break  # Recheck all windows with new start time
                    
            if not adjusted:
                break  # No conflicts found, we have a valid time slot
            iteration += 1

        # ✅ Return the earliest available start time (do NOT update machine_availability here)
        return current_avail


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
            completion_time = start_time + total_duration

            if completion_time < best_completion:
                best_completion = completion_time
                best_machine = machine_id
        return best_machine, best_completion

    def schedule_operation(self, operation, machine_id, earliest_start_time):
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

        # ✅ Update machine availability to the end of this operation
        self.machine_availability[machine_id] = end_time
        self.machine_last_material[machine_id] = operation.get('Mat_Type', None)
        self.op_completion_times[op_id] = end_time
        return True

    def select_next_operation(self, available_ops, heuristic='SPT'):
        def safe_priority(op):
            return int(op.get('Priority', 3))

        if heuristic == 'SPT':
            rule = "SPT"
        elif heuristic == 'EDD':
            rule = "EDD"
        elif heuristic == 'CR':
            rule = "CR"
        elif heuristic == 'PRIORITY':
            rule = "PRIORITY"
        else:
            rule = "SPT (Default)"

        st.caption(f"⚙️ Active Selection Rule: {rule}")

        if heuristic == 'SPT':
            op, earliest_start = min(
                available_ops,
                key=lambda x: (safe_priority(x[0]), x[0]['Total_Proc_Min'], x[0]['Due_Time_Min'])
            )
        elif heuristic == 'EDD':
            op, earliest_start = min(
                available_ops,
                key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min'], x[0]['Total_Proc_Min'])
            )
        elif heuristic == 'CR':
            op, earliest_start = min(
                available_ops,
                key=lambda x: (safe_priority(x[0]), (x[0]['Due_Time_Min'] / max(x[0]['Total_Proc_Min'], 1)))
            )
        elif heuristic == 'PRIORITY':
            op, earliest_start = min(
                available_ops,
                key=lambda x: (safe_priority(x[0]), x[0]['Due_Time_Min'])
            )
        else:
            op, earliest_start = min(
                available_ops,
                key=lambda x: (safe_priority(x[0]), x[0]['Total_Proc_Min'])
            )

        return op, earliest_start

    def run_scheduling(self, heuristic='SPT', verbose=True):
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

        chain = {
            'SPT': "SPT",
            'EDD': "EDD",
            'CR': "CR",
            'PRIORITY': "PRIORITY"
        }.get(heuristic, "SPT")
        st.info(f"🧩 Using Scheduling Rule: **{chain}**")

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
                self.op_completion_times[next_op['Operation_ID']] = float('inf')
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

# ---------------------------
# Live capacity & data loading
# ---------------------------
def analyze_capacity_for_new_job(new_job_ops, current_schedule, df_machines, df_effective, due_time_min):
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
        analysis['reasons'].append("❌ No eligible machines found for operations")
        return analysis

    if operations_schedulable < len(new_job_ops):
        analysis['reasons'].append(f"⚠️ Only {operations_schedulable}/{len(new_job_ops)} operations can be scheduled in-house")

    estimated_completion = current_makespan + total_new_time
    analysis['metrics']['estimated_completion_days'] = estimated_completion / 480
    analysis['metrics']['due_date_days'] = due_time_min / 480
    analysis['metrics']['new_job_time_days'] = total_new_time / 480

    deadline_buffer = due_time_min - estimated_completion
    analysis['metrics']['deadline_buffer_days'] = deadline_buffer / 480

    if deadline_buffer < 0:
        analysis['feasible'] = False
        analysis['recommendation'] = 'OUTSOURCE'
        analysis['reasons'].append(f"❌ Cannot meet deadline - Need {abs(deadline_buffer)/480:.1f} more days")
        analysis['reasons'].append(f"   Estimated completion: Day {estimated_completion/480:.1f}")
        analysis['reasons'].append(f"   Due date: Day {due_time_min/480:.1f}")
    else:
        analysis['feasible'] = True
        analysis['recommendation'] = 'SCHEDULE'
        analysis['reasons'].append(f"✅ Can meet deadline with {deadline_buffer/480:.1f} days buffer")

    if current_makespan > 0:
        projected_utilization = ((total_productive + total_new_time) / (estimated_completion * len(df_machines))) * 100 if estimated_completion > 0 else 0
        analysis['metrics']['projected_utilization'] = round(projected_utilization, 1)

        if projected_utilization > 90:
            analysis['reasons'].append(f"⚠️ High utilization: {projected_utilization:.1f}% (machines heavily loaded)")
        elif projected_utilization > 75:
            analysis['reasons'].append(f"✅ Good utilization: {projected_utilization:.1f}% (balanced load)")
        else:
            analysis['reasons'].append(f"✅ Low utilization: {projected_utilization:.1f}% (capacity available)")

    return analysis

# ---------------------------
# Data loading & preprocessing
# ---------------------------
@st.cache_data
def load_all_data(sample_size=None, _cache_version=2):
    try:
        df_ops = pd.read_csv("data/jobs_dataset.csv")
        df_vendors = pd.read_csv("data/vendor_data.csv")
        df_machines = pd.read_csv("data/machine_data.csv")
        df_penalties = pd.read_csv("data/previous_next_material.csv")
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure the 'data' folder is in the same directory as app.py")
        st.stop()

    st.write("🔍 DEBUG: Validating data quality...")
    validation_issues = []

    required_ops_cols = ['Job_ID', 'Operation_ID', 'Op_Seq', 'Quantity', 'Op_Type',
                         'Mat_Type', 'Proc_Time_per_Unit', 'Setup_Time', 'Release_Day', 'Due_Day']
    missing_cols = [col for col in required_ops_cols if col not in df_ops.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns in jobs_dataset.csv: {missing_cols}")
        st.stop()

    if (df_ops['Quantity'] <= 0).any():
        validation_issues.append(f"Found {(df_ops['Quantity'] <= 0).sum()} operations with Quantity <= 0")
    if (df_ops['Proc_Time_per_Unit'] < 0).any():
        validation_issues.append(f"Found {(df_ops['Proc_Time_per_Unit'] < 0).sum()} operations with negative processing time")
    if (df_ops['Setup_Time'] < 0).any():
        validation_issues.append(f"Found {(df_ops['Setup_Time'] < 0).sum()} operations with negative setup time")

    if validation_issues:
        st.warning("⚠️ Data Quality Issues Found:")
        for issue in validation_issues:
            st.write(f"   - {issue}")
    else:
        st.success("✅ Data validation passed")

    if sample_size:
        unique_jobs = df_ops['Job_ID'].unique()[:sample_size]
        df_ops = df_ops[df_ops['Job_ID'].isin(unique_jobs)].copy()
        st.info(f"📊 **TEST MODE**: Using {sample_size} jobs ({len(df_ops)} operations) for faster performance")

    st.write("🔍 DEBUG: Checking for deadline anomalies...")
    deadline_issues = df_ops[df_ops['Due_Day'] <= df_ops['Release_Day']]
    if len(deadline_issues) > 0:
        st.warning(f"⚠️ Found {len(deadline_issues)} operations with impossible deadlines!")
        for idx in deadline_issues.index:
            release_day = df_ops.at[idx, 'Release_Day']
            proc_time_days = df_ops.at[idx, 'Proc_Time_per_Unit'] * df_ops.at[idx, 'Quantity'] / 480
            setup_time_days = df_ops.at[idx, 'Setup_Time'] / 480
            lead_time = max(7, int(proc_time_days + setup_time_days + 3))
            df_ops.at[idx, 'Due_Day'] = release_day + lead_time
        st.success(f"✅ Fixed {len(deadline_issues)} deadline issues - added realistic lead times")
    else:
        st.success("✅ No deadline anomalies found")

    df_ops['Total_Proc_Min'] = df_ops['Proc_Time_per_Unit'] * df_ops['Quantity']

    # Normalize machines df columns
    if 'Speed Factor' not in df_machines.columns and 'SpeedFactor' in df_machines.columns:
        df_machines.rename(columns={'SpeedFactor': 'Speed Factor'}, inplace=True)

    df_machines['Speed Factor'] = (
        df_machines['Speed Factor']
        .astype(str)
        .str.extract(r'([0-9]*\.?[0-9]+)')
        .astype(float)
    )

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
            total_time = effective_time + setup_time + transfer_min

            effective_times.append({
                'Operation_ID': op['Operation_ID'],
                'Machine_ID': machine_id,
                'Effective_Proc_Time': effective_time,
                'Setup_Time': setup_time,
                'Transfer_Min': transfer_min,
                'Total_Time': total_time
            })
    df_effective = pd.DataFrame(effective_times)

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

    MINUTES_PER_DAY = 8 * 60
    df_ops['Release_Time_Min'] = df_ops['Release_Day'] * MINUTES_PER_DAY
    df_ops['Due_Time_Min'] = df_ops['Due_Day'] * MINUTES_PER_DAY
    df_ops['Outsource_Cost'].fillna(0, inplace=True)
    df_ops['Outsource_Time_Min'].fillna(0, inplace=True)
    df_ops['Completion_Day'] = 0

    df_machines = df_machines.rename(columns={'Machine ID': 'Machine_ID'})
    df_machines['Maintenance_Window'] = df_machines['Scheduled Maintenance (Day, Time-Time)'].apply(parse_maintenance)

    decisions = []
    for idx, op in df_ops.iterrows():
        decision, cost, reason = make_or_buy_decision(op, df_effective, cost_threshold=0.9)
        decisions.append({'Operation_ID': op['Operation_ID'], 'Decision': decision})

    df_decisions = pd.DataFrame(decisions)
    df_ops = df_ops.merge(df_decisions, on='Operation_ID', how='left')
    df_ops['Assignment_Type'] = df_ops['Decision'].fillna('IN_HOUSE')
    df_ops.drop(columns=['Decision'], inplace=True)

    total_ops = len(df_ops)
    outsourced_ops = len(df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE'])
    inhouse_ops = len(df_ops[df_ops['Assignment_Type'] == 'IN_HOUSE'])
    outsource_pct = (outsourced_ops / total_ops) * 100 if total_ops > 0 else 0

    st.write(f"📊 **Make-or-Buy Analysis:**")
    st.write(f"   - Total Operations: {total_ops}")
    st.write(f"   - In-House: {inhouse_ops} ({100-outsource_pct:.1f}%)")
    st.write(f"   - Outsourced: {outsourced_ops} ({outsource_pct:.1f}%)")

    if outsource_pct > 50:
        st.warning(f"⚠️ **HIGH OUTSOURCING**: {outsource_pct:.0f}% of operations outsourced! This will reduce machine utilization.")
        st.write("   💡 Consider: Lowering cost threshold or adding machine capacity")

    return df_ops, df_machines, df_effective, df_penalties, df_vendors


def run_single_heuristic(_df_ops, _df_machines, _df_effective, _df_penalties, heuristic='SPT'):
    st.write(f"🔍 **EXPLAINER**: Running {heuristic} scheduling algorithm")
    st.write(f"   📊 Processing {len(_df_ops)} operations across {len(_df_machines)} machines")

    import time
    start_time = time.time()

    scheduler = CNCScheduler(_df_ops, _df_machines, _df_effective, _df_penalties)
    init_time = time.time() - start_time
    st.write(f"   ✅ Scheduler initialized in {init_time:.2f}s")

    st.write(f"   🔄 Executing {heuristic} heuristic (this may take some time)...")
    heuristic_start = time.time()
    schedule = scheduler.run_scheduling(heuristic=heuristic)
    exec_time = time.time() - heuristic_start

    st.write(f"   ✅ {heuristic} completed in {exec_time:.2f}s")
    st.write(f"   📈 Successfully scheduled {len(schedule)} operations")

    return schedule

# ---------------------------
# Visualization helpers (unchanged but included)
# ---------------------------

def create_gantt_chart(
    _schedule_df,
    _machines_df,
    title="CNC Machine Schedule",
    _cache_key="",
    machines_order=None
):
    import re
    import pandas as pd
    import plotly.graph_objects as go

    schedule_df = _schedule_df.copy()
    machines_df = _machines_df.copy()

    for col in ["Start_Time", "End_Time"]:
        if col in schedule_df.columns:
            schedule_df[col] = pd.to_numeric(schedule_df[col], errors="coerce").fillna(0)

    def machine_sort_key(mid):
        match = re.search(r"\d+", str(mid))
        return int(match.group()) if match else float("inf")

    if machines_order:
        all_machines_sorted = sorted(machines_order, key=machine_sort_key)
    else:
        all_machines_sorted = sorted(
            machines_df["Machine_ID"].unique(), key=machine_sort_key
        )

    df_real = schedule_df[schedule_df["Job_ID"] != "Idle"].copy() if "Job_ID" in schedule_df.columns else schedule_df.copy()
    if df_real.empty:
        return go.Figure().update_layout(
            title="No job data available",
            xaxis_title="Time (minutes)",
            yaxis_title="Machine",
        )

    x_min = df_real["Start_Time"].min()
    x_max = df_real["End_Time"].max()
    df_real["Start_Shifted"] = df_real["Start_Time"] - x_min
    df_real["End_Shifted"] = df_real["End_Time"] - x_min

    fig = go.Figure()

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]
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

    # ✅ Display all maintenance/breakdown windows
    for _, machine in machines_df.iterrows():
        maint = machine.get("Maintenance_Window")
        machine_id = machine.get("Machine_ID")
        
        if maint and machine_id in all_machines_sorted:
            # Handle both single window (dict) and multiple windows (list)
            windows = [maint] if isinstance(maint, dict) else (maint if isinstance(maint, list) else [])
            
            for i, window in enumerate(windows):
                if isinstance(window, dict) and "start" in window and "end" in window:
                    # Calculate shifted positions
                    window_start_shifted = window["start"] - x_min
                    window_end_shifted = window["end"] - x_min
                    window_duration = window.get("duration", window["end"] - window["start"])
                    
                    # Add semi-transparent red rectangle for breakdown/maintenance
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
                    
                    # Add annotation label for breakdown
                    fig.add_annotation(
                        x=(window_start_shifted + window_end_shifted) / 2,
                        y=machine_id,
                        text=f"⚠️ BREAKDOWN<br>{window_duration} min",
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
        title=dict(text=title, font=dict(size=16, color="#E0E0E0")),
        xaxis_title="Time (minutes, shifted)",
        yaxis_title="Machine",
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=60, b=40),
        font=dict(size=12, color="#E0E0E0"),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.3)",
            color="#E0E0E0"
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            color="#E0E0E0"
        ),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=all_machines_sorted,
        autorange="reversed",
        type="category",
    )
    fig.update_xaxes(range=[0 - pad, (x_max - x_min) + pad], showgrid=True)
    return fig


def create_kpi_dashboard(_schedule_df, _df_ops_current, _machines_df, _heuristic_name="", _cache_key=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if _schedule_df is None or _schedule_df.empty:
        st.warning(f"No schedule data available to compute KPIs for {_heuristic_name}.")
        return go.Figure(), {
            'Makespan_Days': 0,
            'Total_Tardiness_Days': 0,
            'Total_Cost_$': 0,
            'On_Time_%': 0,
            'Machine_Utilization_%': 0
        }

    metrics = calculate_metrics(_schedule_df, _df_ops_current, _heuristic_name)

    makespan_days = metrics['Makespan_Days']
    tardiness_days = metrics['Total_Tardiness_Days']
    total_cost = metrics['Total_Cost_$']
    outsource_pct = (
        100 * _df_ops_current[_df_ops_current['Assignment_Type'] == 'OUTSOURCE']['Operation_ID'].nunique()
        / _df_ops_current['Operation_ID'].nunique()
        if len(_df_ops_current) > 0 else 0
    )
    ontime_pct = metrics['On_Time_%']
    utilization = metrics['Machine_Utilization_%']

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Makespan (Days)',
            'Total Tardiness (Days)',
            'Total Cost ($)',
            'Outsourced %',
            'On-Time Delivery % (Operations)',
            'Machine Utilization %'
        ),
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
        ]
    )

    fig.add_trace(go.Indicator(mode="number", value=round(makespan_days, 2)), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=round(tardiness_days, 2)), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number", value=round(total_cost, 2), number={'prefix': "$"}), row=1, col=3)

    fig.add_trace(
        go.Indicator(mode="gauge+number", value=round(outsource_pct, 1), gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "purple"}}),
        row=2, col=1
    )
    fig.add_trace(
        go.Indicator(mode="gauge+number", value=round(ontime_pct, 1), gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}),
        row=2, col=2
    )
    fig.add_trace(
        go.Indicator(mode="gauge+number", value=round(utilization, 1), gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "cyan"}}),
        row=2, col=3
    )

    fig.update_layout(height=400, margin=dict(t=50, b=10, l=10, r=10))
    return fig, metrics

# ---------------------------
# Operation status table
# ---------------------------

# ---------------------------
# Operation status table (UPDATED)
# ---------------------------

def create_operation_status_table(schedule_df, df_ops, _cache_key=None):
    """
    Create operation-level status table with safety checks.
    Handles cases where no schedule exists yet (before heuristic applied).
    """
    import pandas as pd, time

    # 🧩 SAFETY CHECK 1 — schedule not available yet
    if schedule_df is None or schedule_df.empty:
        st.warning("⚠️ No schedule available yet. Please compute and apply a heuristic first to view operation status.")
        placeholder = pd.DataFrame({
            "Message": ["No scheduled operations found."],
            "Suggestion": ["Click 'Compute All Heuristics' in sidebar, then 'Apply Selected Heuristic'."]
        })
        return placeholder

    # 🧩 SAFETY CHECK 2 — missing required column
    if 'Operation_ID' not in schedule_df.columns:
        st.warning("⚠️ Schedule is missing 'Operation_ID'. Please ensure a valid schedule was applied.")
        placeholder = pd.DataFrame({
            "Message": ["Invalid schedule data."],
            "Suggestion": ["Try reapplying a heuristic or recomputing all heuristics."]
        })
        return placeholder

    schedule_df = schedule_df.copy()
    df_ops = df_ops.copy()

    # Derive due times if not already present
    if 'Due_Time_Min' not in df_ops.columns:
        df_ops['Due_Time_Min'] = df_ops['Due_Day'] * 480

    op_status = []
    current_time = schedule_df['End_Time'].max() if not schedule_df.empty else 0

    for op_id in df_ops['Operation_ID'].unique():
        op_row_def = df_ops[df_ops['Operation_ID'] == op_id].iloc[0]
        op_row_sched = schedule_df[schedule_df['Operation_ID'] == op_id] \
            if 'Operation_ID' in schedule_df.columns else pd.DataFrame()

        due_time_min = op_row_def.get('Due_Time_Min', 0)
        total_proc_min = op_row_def.get('Total_Proc_Min', op_row_def.get('Proc_Time_per_Unit', 0) * op_row_def.get('Quantity', 1))
        assignment = op_row_def.get('Assignment_Type', 'IN_HOUSE')
        priority = int(op_row_def.get('Priority', 3))
        job_id = op_row_def.get('Job_ID', None)

        # <<< NEW: Variable to hold machine assignment >>>
        machine_assigned = 'N/A'

        # CASE 1: Scheduled
        if not op_row_sched.empty:
            finish_time = op_row_sched['End_Time'].max()
            finish_day = finish_time / 480
            tardiness_days = max(0, (finish_time - due_time_min) / 480)
            status = "On-Time" if tardiness_days == 0 else "Late"

            # <<< NEW: Get the assigned machine from the schedule >>>
            try:
                machine_assigned = op_row_sched['Machine_ID'].iloc[0]
            except IndexError:
                machine_assigned = 'Error'

        # CASE 2: Outsourced
        elif assignment == "OUTSOURCE":
            est_finish_time = op_row_def.get('Release_Time_Min', 0) + op_row_def.get('Outsource_Time_Min', 0)
            finish_time = est_finish_time
            finish_day = finish_time / 480
            tardiness_days = max(0, (finish_time - due_time_min) / 480)
            status = "Outsourced" if tardiness_days == 0 else "Outsource Delay"
            
            # <<< NEW: Set machine to 'OUTSOURCE' >>>
            machine_assigned = 'OUTSOURCE'

        # CASE 3: Pending
        else:
            finish_time = current_time
            finish_day = finish_time / 480

            if current_time > due_time_min:
                tardiness_days = round((current_time - due_time_min) / 480, 2)
                status = "Overdue (Pending)"
            else:
                tardiness_days = 0
                status = "Pending"

            # <<< NEW: Set machine to 'PENDING' >>>
            machine_assigned = 'PENDING'

        # Critical Ratio
        time_remaining = max(0, due_time_min - finish_time)
        cr_value = round(time_remaining / max(total_proc_min, 1), 2)

        op_status.append({
            'Job_ID': job_id,
            'Operation_ID': op_id,
            'Machine_ID': machine_assigned,  # <<< NEW: Added to dictionary >>>
            'Priority': priority,
            'Assignment': assignment,
            'Total_Proc_Min': round(total_proc_min, 2),
            'CR_Value': cr_value,
            'Finish_Day': round(finish_day, 2),
            'Due_Day': round(due_time_min / 480, 2),
            'Tardiness_Days': round(tardiness_days, 2),
            'Status': status,
            'Updated': time.strftime("%H:%M:%S")
        })

    op_status_df = pd.DataFrame(op_status)

    # <<< NEW: Re-order columns so Machine_ID is easy to see >>>
    column_order = [
        'Job_ID', 'Operation_ID', 'Machine_ID', 'Priority', 'Assignment',
        'Status', 'Total_Proc_Min', 'CR_Value', 'Finish_Day', 'Due_Day',
        'Tardiness_Days', 'Updated'
    ]
    # Keep only columns that actually exist
    final_columns = [col for col in column_order if col in op_status_df.columns]
    op_status_df = op_status_df[final_columns]


    # Sorting based on active heuristic
    heuristic = st.session_state.current_heuristic if 'current_heuristic' in st.session_state and st.session_state.current_heuristic else 'SPT'
    if heuristic == 'SPT':
        op_status_df = op_status_df.sort_values(['Priority', 'Total_Proc_Min']).reset_index(drop=True)
        sort_label = "Priority → SPT (Total Processing Time)"
    elif heuristic == 'EDD':
        op_status_df = op_status_df.sort_values(['Priority', 'Due_Day']).reset_index(drop=True)
        sort_label = "Priority → EDD (Due Day)"
    elif heuristic == 'CR':
        op_status_df = op_status_df.sort_values(['Priority', 'CR_Value']).reset_index(drop=True)
        sort_label = "Priority → CR (Critical Ratio)"
    elif heuristic == 'PRIORITY':
        op_status_df = op_status_df.sort_values(['Priority']).reset_index(drop=True)
        sort_label = "Priority Only"
    else:
        op_status_df = op_status_df.sort_values(['Priority', 'Due_Day']).reset_index(drop=True)
        sort_label = "Priority → Due Date (Fallback)"

    st.caption(f"📋 Active Sorting Rule: {sort_label}")
    return op_status_df

# ---------------------------
# EXPORT
# ---------------------------
def export_schedule(schedule_df):
    if schedule_df is None or schedule_df.empty:
        st.warning("⚠️ No schedule data available to export. Please compute and apply a heuristic first.")
        return b""

    export_df = schedule_df.copy()

    # Safety: ensure essential columns exist
    required_cols = ["Start_Time", "End_Time", "Tardiness"]
    for col in required_cols:
        if col not in export_df.columns:
            st.error(f"❌ Cannot export: missing column '{col}' in schedule.")
            return b""

    export_df['Start_Day'] = export_df['Start_Time'] / 480
    export_df['End_Day'] = export_df['End_Time'] / 480
    export_df['Tardiness_Days'] = export_df['Tardiness'] / 480 if 'Tardiness' in export_df.columns else 0

    export_df = export_df[[
        'Job_ID', 'Operation_ID', 'Machine_ID',
        'Start_Time', 'End_Time', 'Start_Day', 'End_Day',
        'Setup_Time', 'Proc_Time', 'Transfer_Time',
        'Due_Time', 'Tardiness', 'Tardiness_Days'
    ]]

    return export_df.to_csv(index=False).encode('utf-8')


# ---------------------------
# NEW: compute_all_heuristics_and_metrics + apply_heuristic_to_dataset
# ---------------------------
def compute_all_heuristics_and_metrics(ss, show_progress=True):
    heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
    metrics = []
    schedules = {}

    if show_progress:
        st.info("🔄 Computing schedules for all heuristics using the CURRENT dataset "
                "(including any breakdowns, priority, or outsourcing updates)...")
        progress = st.progress(0)

    # ✅ Always use latest state
    ss.base_df_ops = ss.df_ops.copy()
    ss.base_df_machines = ss.df_machines.copy()

    # ✅ Clear resource cache to force re-run
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    # ✅ Compute all heuristics freshly
    for i, heur in enumerate(heuristics):
        schedule_key = f"schedule_{heur.lower()}"
        try:
            schedule = run_single_heuristic(
                ss.base_df_ops.copy(),
                ss.base_df_machines.copy(),
                ss.base_df_effective.copy(),
                ss.base_df_penalties.copy(),
                heuristic=heur,
            )
            schedules[heur] = schedule.copy()
            setattr(ss, schedule_key, schedule.copy())
            metrics.append(calculate_metrics(schedule.copy(), ss.base_df_ops.copy(), heur))
        except Exception as e:
            dbg(f"⚠️ compute_all failed for {heur}: {e}")
            existing = getattr(ss, schedule_key, pd.DataFrame())
            metrics.append(calculate_metrics(existing, ss.base_df_ops.copy(), heur))

        if show_progress:
            progress.progress((i + 1) / len(heuristics))

    # ✅ Store updated metrics
    ss.df_metrics = pd.DataFrame(metrics)
    ss.schedule_update_key = f"compute_all_{int(time.time())}"
    ss.recalculate_all_heuristics = False
    ss.force_metric_refresh = False

    # ✅ LOG ACTIVITY
    if "activity_log" not in ss:
        ss.activity_log = []
    ss.activity_log.append({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'action': 'All Heuristics Computed',
        'details': f"Computed: {', '.join(heuristics)} | Dataset: {len(ss.base_df_ops)} ops, {len(ss.base_df_machines)} machines",
        'affected_items': 'All schedules'
    })

    # ✅ Notify success
    st.success("✅ All heuristics recomputed successfully.")
    st.toast("📊 Updated heuristic metrics ready for review!", icon="📈")

    # ✅ Clear any pending prompts (from breakdown / priority / outsourcing)
    ss.breakdown_pending = False
    ss.breakdown_message_visible = False
    if hasattr(ss, "priority_update_message_visible"):
        ss.priority_update_message_visible = False
    if hasattr(ss, "outsourcing_update_message_visible"):
        ss.outsourcing_update_message_visible = False

    # ✅ Always redirect to comparison page after recomputation
    ss.current_page = "comparison"

    # ✅ Preserve user’s current heuristic (no default to SPT)
    current_h = ss.get("current_heuristic", None)
    if not current_h or current_h not in heuristics:
        st.warning("⚠️ No heuristic selected. Please select one from the comparison table.")
        ss.current_heuristic = None
        ss.last_applied_heuristic = None
        ss.current_schedule = pd.DataFrame()
        return schedules, ss.df_metrics

    # ✅ Save back for persistence
    ss.current_heuristic = current_h
    ss.last_applied_heuristic = current_h

    # ✅ Sync current heuristic’s schedule
    schedule_key = f"schedule_{current_h.lower()}"
    if hasattr(ss, schedule_key):
        ss.current_schedule = getattr(ss, schedule_key).copy()
        st.toast(f"🔁 Loaded updated schedule for {current_h}", icon="📈")
    else:
        st.warning(f"⚠️ No schedule found for {current_h}. Please select a heuristic.")
        ss.current_schedule = pd.DataFrame()

    # ✅ Guarantee essential columns exist
    required_cols = ['Job_ID', 'Machine_ID', 'Start_Time', 'End_Time']
    for col in required_cols:
        if col not in ss.current_schedule.columns:
            ss.current_schedule[col] = []

    # ✅ Force KPI refresh
    ss.schedule_update_key = str(time.time())
    st.session_state.schedule_update_key = ss.schedule_update_key

    # ✅ Re-render app with latest state
    st.rerun()

    return schedules, ss.df_metrics



def apply_heuristic_to_dataset(ss, heuristic):
    schedule_key = f"schedule_{heuristic.lower()}"
    schedule_df = getattr(ss, schedule_key, None)

    if schedule_df is None or schedule_df.empty:
        st.error(f"No computed schedule found for {heuristic}. Click 'Compute Heuristics' first.")
        return False

    sched = schedule_df.copy()
    if 'Operation_ID' not in sched.columns:
        st.error("Schedule missing Operation_ID — cannot apply.")
        return False

    sched = sched[['Operation_ID', 'Job_ID', 'Machine_ID', 'Start_Time', 'End_Time', 'Tardiness']].copy()
    ops = ss.base_df_ops.copy()

    ops = ops.merge(sched, on='Operation_ID', how='left', suffixes=('', '_sched'))

    for col in ['Machine_ID', 'Start_Time', 'End_Time', 'Tardiness']:
        if f"{col}_sched" in ops.columns:
            ops[col] = ops[f"{col}_sched"].combine_first(ops.get(col))

    ops = ops.drop(columns=[c for c in ops.columns if c.endswith('_sched')], errors='ignore')

    ops['Completion_Day'] = ops.apply(
        lambda r: (r['End_Time'] / 480) if pd.notna(r.get('End_Time')) else r.get('Completion_Day', 0),
        axis=1
    )
    ops['Assigned_By_Heuristic'] = heuristic

    ss.base_df_ops = ops.copy()
    ss.df_ops = ops.copy()

    # ✅ LOG ACTIVITY
    if "activity_log" not in ss:
        ss.activity_log = []
    ss.activity_log.append({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'action': f'Heuristic Applied: {heuristic}',
        'details': f"Schedule size: {len(schedule_df)} operations | Updated operation assignments",
        'affected_items': f"{len(schedule_df)} scheduled operations"
    })

    # ✅ Persist current heuristic choice
    ss.current_heuristic = heuristic
    ss.last_applied_heuristic = heuristic
    ss.current_schedule = schedule_df.copy()

    # 🧩 Immediately sync this heuristic to current view
    ss.df_ops = ss.base_df_ops.copy()
    ss.df_machines = ss.base_df_machines.copy()
    ss.schedule_update_key = str(time.time())

    ss.force_metric_refresh = False
    ss.recalculate_all_heuristics = False

    # 🧩 Recompute KPIs for the applied heuristic
    try:
        st.info(f"📊 Recomputing KPI metrics for {heuristic} ...")
        metrics_df = calculate_metrics(
            ss.current_schedule.copy(),
            ss.df_ops.copy(),
            heuristic
        )

        # ✅ Update or create metrics table
        if hasattr(ss, "df_metrics") and ss.df_metrics is not None:
            ss.df_metrics = pd.concat([
                ss.df_metrics[ss.df_metrics["Heuristic"] != heuristic],
                pd.DataFrame([metrics_df])
            ], ignore_index=True)
        else:
            ss.df_metrics = pd.DataFrame([metrics_df])

        st.success(f"📈 KPI metrics updated for {heuristic}.")
    except Exception as e:
        st.error(f"❌ Failed to update KPI metrics for {heuristic}: {e}")

    # ✅ Sync selected heuristic’s schedule across all
    for h in ['SPT', 'EDD', 'CR', 'PRIORITY']:
        setattr(ss, f'schedule_{h.lower()}', ss.current_schedule.copy())

    st.success(f"✅ {heuristic} schedule copied to all heuristics for next operation updates.")

    # ✅ Switch to detailed heuristic view
    ss.current_page = "heuristic_view"
    ss.force_metric_refresh = True
    ss.schedule_update_key = str(time.time())

    st.rerun()

# ---------------------------
# Sidebar widgets & controls
# ---------------------------
def draw_compute_apply_controls(ss):
    st.sidebar.markdown("### 🧮 Compute & Apply Heuristics")
    if st.sidebar.button("🧪 Compute All Heuristics (current dataset)", key="compute_all_heurs"):
        compute_all_heuristics_and_metrics(ss, show_progress=True)
        st.rerun()

    heuristic_options = ('SPT', 'EDD', 'CR', 'PRIORITY')
    
    # --- START FIX ---
    # This logic IDENTICALLY matches the logic in your draw_heuristic_selector
    # It finds the correct index if a heuristic is selected
    # or defaults the *display* to index 0 if ss.current_heuristic is None.
    # It does NOT change ss.current_heuristic.
    
    current_h = ss.get("current_heuristic") # Get value (could be None)
    
    if current_h in heuristic_options:
        current_index = heuristic_options.index(current_h)
    else:
        current_index = 0 # Default display to 'SPT' if state is None
    # --- END FIX ---

    apply_choice = st.sidebar.selectbox(
        "Choose heuristic to APPLY (persist):", 
        heuristic_options,
        index=current_index, # Use the synced index
        key="apply_choice"
    )
    
    if st.sidebar.button("✅ Apply Selected Heuristic", key="apply_heur"):
        ok = apply_heuristic_to_dataset(ss, apply_choice)
        if ok:
            st.rerun()


# ---------------------------
# Existing sidebar widgets (modified to persist base data changes)
# ---------------------------
def draw_heuristic_selector(ss):
    st.sidebar.markdown("### 🎯 Scheduling Algorithm")
    st.sidebar.info("**EXPLAINER**: Different algorithms prioritize different factors:\n"
                    "- **SPT**: Shortest jobs first (fast completion)\n"
                    "- **EDD**: Due dates first (minimize lateness)\n"
                    "- **CR**: Critical ratio (balance time & work)\n"
                    "- **PRIORITY**: High priority jobs first")

    heuristic_options = ('SPT', 'EDD', 'CR', 'PRIORITY')
    
    # 1. Ensure current_heuristic is valid, default to SPT if None
    if ss.current_heuristic not in heuristic_options:
        ss.current_heuristic = 'SPT'

    # 2. CRITICAL FIX: Sync the widget's internal state to match ss.current_heuristic
    # This prevents the widget from reverting to 'SPT' (or a previous value) 
    # when interactions happen elsewhere (like the Breakdown dropdown).
    if 'heuristic_selector' not in st.session_state:
        st.session_state.heuristic_selector = ss.current_heuristic
    elif st.session_state.heuristic_selector != ss.current_heuristic:
        st.session_state.heuristic_selector = ss.current_heuristic

    # 3. Draw the selectbox
    # Note: We don't need 'index' here because setting st.session_state['heuristic_selector'] 
    # sets the selected value automatically.
    selected_heuristic = st.sidebar.selectbox(
        "Select Scheduling Algorithm (view only)",
        heuristic_options,
        key='heuristic_selector'
    )

    # 4. Handle User Interaction
    # Only update if the returned value differs from the state 
    # (The sync in step 2 ensures this only happens if the USER actually changed this specific dropdown)
    if selected_heuristic != ss.current_heuristic:
        st.sidebar.write(f"🔄 Switching view to {selected_heuristic} (no apply).")
        schedule_key = f'schedule_{selected_heuristic.lower()}'
        
        if getattr(ss, schedule_key, None) is None:
            st.sidebar.warning(f"No computed schedule for {selected_heuristic}. Use 'Compute All Heuristics' first.")
            # Revert the dropdown visual to the valid heuristic to avoid confusion
            st.session_state.heuristic_selector = ss.current_heuristic
        else:
            ss.current_heuristic = selected_heuristic
            ss.current_schedule = getattr(ss, schedule_key).copy()
            # set working copies to base data (dataset remains canonical until user applies)
            ss.df_ops = ss.base_df_ops.copy()
            ss.df_machines = ss.base_df_machines.copy()
            st.toast(f"📊 Viewing {ss.current_heuristic} schedule (not applied).", icon="ℹ️")
            st.rerun()

def draw_live_job_scheduler(ss):
    st.sidebar.markdown("### 🎯 Add Job")
    st.sidebar.info("**EXPLAINER**: Add a new job and the system will analyze if it can be scheduled in-house or needs outsourcing.")

    with st.sidebar.expander("📋 Schedule New Job", expanded=False):
        st.write("**Enter Job Details:**")
        col1, col2 = st.columns(2)
        with col1:
            live_job_id = st.text_input("Job ID:", f"J{900 + len(ss.df_ops['Job_ID'].unique())}", key='live_job_id')
            live_quantity = st.number_input("Quantity:", min_value=10, max_value=1000, value=100, step=10, key='live_qty')
        with col2:
            live_priority = st.selectbox("Priority:", [1, 2, 3], index=0, key='live_priority')
            live_due_days = st.number_input("Due in (days):", min_value=1, max_value=30, value=7, step=1, key='live_due')

        st.write("**Operations Required:**")
        live_op_count = st.slider("Number of Operations:", 1, 5, 2, key='live_op_count')

        operations_config = []
        for i in range(live_op_count):
            st.write(f"**Operation {i+1}:**")
            col_a, col_b = st.columns(2)
            with col_a:
                op_type = st.selectbox(
                    f"Op{i+1} Type:",
                    ['MILLING', 'TURNING', 'GRINDING', 'DRILLING'],
                    key=f'live_op{i}_type'
                )
            with col_b:
                material = st.selectbox(
                    f"Op{i+1} Material:",
                    ['STEEL', 'ALUM', 'TITAN', 'BRASS'],
                    key=f'live_op{i}_mat'
                )
            col_c, col_d = st.columns(2)
            with col_c:
                proc_time = st.number_input(
                    f"Op{i+1} Time/Unit (min):",
                    min_value=0.1, max_value=5.0, value=0.3, step=0.1,
                    key=f'live_op{i}_time'
                )
            with col_d:
                setup_time = st.number_input(
                    f"Op{i+1} Setup (min):",
                    min_value=10, max_value=120, value=30, step=5,
                    key=f'live_op{i}_setup'
                )

            operations_config.append({
                'op_type': op_type,
                'material': material,
                'proc_time': proc_time,
                'setup_time': setup_time
            })

            if i < live_op_count - 1:
                st.divider()

        st.divider()

        col_analyze, col_add = st.columns(2)
        with col_analyze:
            if st.button("🔍 Analyze Capacity", use_container_width=True, type="primary"):
                current_time_days = ss.current_schedule['End_Time'].max() / 480 if not ss.current_schedule.empty else 0
                release_time_min = current_time_days * 480
                due_time_min = release_time_min + (live_due_days * 480)

                new_job_ops = []
                for i, op_config in enumerate(operations_config):
                    new_op_id = f'{live_job_id}_Op{i+1}'
                    new_op = {
                        'Job_ID': live_job_id,
                        'Operation_ID': new_op_id,
                        'Op_Seq': i + 1,
                        'Part_Type': f'NEW_{live_job_id}',
                        'Quantity': live_quantity,
                        'Op_Type': op_config['op_type'],
                        'Mat_Type': op_config['material'],
                        'Tool_Group': 'TGA',
                        'Proc_Time_per_Unit': op_config['proc_time'],
                        'Setup_Time': op_config['setup_time'],
                        'Transfer_Min': 5,
                        'Release_Day': current_time_days,
                        'Due_Day': current_time_days + live_due_days,
                        'Priority': live_priority,
                        'Outsource_Flag': 'Y',
                        'Vendor_Ref': 'V1' if 'V1' in ss.base_df_vendors['Vendor_ID'].values else None,
                        'Release_Time_Min': release_time_min,
                        'Due_Time_Min': due_time_min,
                        'Total_Proc_Min': op_config['proc_time'] * live_quantity
                    }
                    new_job_ops.append(new_op)

                new_eff_times = []
                for op in new_job_ops:
                    eligible_machines = get_eligible_machines(op['Op_Type'])
                    if not eligible_machines:
                        continue

                    for machine_id in eligible_machines:
                        machine = ss.base_df_machines[ss.base_df_machines['Machine_ID'] == machine_id].iloc[0]
                        speed_factor = float(machine['Speed Factor'])
                        oee = float(machine['OEE (Uptime)'])
                        effective_time = op['Total_Proc_Min'] * speed_factor * (1 / oee)
                        total_time = effective_time + op['Setup_Time'] + op['Transfer_Min']

                        new_eff_times.append({
                            'Operation_ID': op['Operation_ID'],
                            'Machine_ID': machine_id,
                            'Effective_Proc_Time': effective_time,
                            'Setup_Time': op['Setup_Time'],
                            'Transfer_Min': op['Transfer_Min'],
                            'Total_Time': total_time
                        })

                df_new_effective = pd.DataFrame(new_eff_times)

                analysis = analyze_capacity_for_new_job(
                    new_job_ops,
                    ss.current_schedule,
                    ss.df_machines,
                    df_new_effective,
                    due_time_min
                )

                ss.live_job_analysis = analysis
                ss.live_job_ops_pending = new_job_ops
                ss.live_job_effective_pending = df_new_effective
                st.rerun()

        # ----------------------------
        # 🧩 DYNAMIC ANALYSIS DISPLAY
        # ----------------------------
        if hasattr(ss, 'live_job_analysis') and ss.live_job_analysis:
            st.divider()
            st.write("### 📊 Capacity Analysis Results")

            analysis = ss.live_job_analysis

            live_current_makespan_min = ss.current_schedule['End_Time'].max() if not ss.current_schedule.empty else 0
            live_current_makespan_days = live_current_makespan_min / 480

            live_utilization = 0.0
            if not ss.current_schedule.empty and live_current_makespan_min > 0:
                total_productive = (ss.current_schedule['Setup_Time'].sum() +
                                   ss.current_schedule['Proc_Time'].sum() +
                                   ss.current_schedule['Transfer_Time'].sum())
                total_machines = len(ss.df_machines)
                live_utilization = (total_productive / (live_current_makespan_min * total_machines)) * 100

            live_due_days = ss.get('live_due', 7)
            new_job_time_days = analysis['metrics'].get('new_job_time_days', 0)
            live_estimated_completion_days = live_current_makespan_days + new_job_time_days
            live_due_date_days = live_current_makespan_days + live_due_days
            live_buffer_days = live_due_date_days - live_estimated_completion_days
            projected_utilization = analysis['metrics'].get('projected_utilization', 0)

            if analysis['recommendation'] == 'SCHEDULE':
                st.success("✅ **RECOMMENDATION: SCHEDULE IN-HOUSE**")
            else:
                st.error("❌ **RECOMMENDATION: OUTSOURCE**")

            st.write("**Current State (Live):**")
            st.write(f"- Current makespan: Day {live_current_makespan_days:.1f}")
            st.write(f"- Current utilization: {live_utilization:.1f}%")

            st.write("**With New Job (Projected):**")
            st.write(f"- Estimated completion: Day {live_estimated_completion_days:.1f}")
            st.write(f"- Due date: Day {live_due_date_days:.1f}")
            st.write(f"- Projected utilization: {projected_utilization:.1f}%")

            st.write("**Analysis:**")
            if live_buffer_days < 0:
                st.write(f"❌ Cannot meet deadline - Need {abs(live_buffer_days):.1f} more days")
            else:
                st.write(f"✅ Can meet deadline with {live_buffer_days:.1f} days buffer")

            for reason in analysis['reasons']:
                if "deadline" not in reason.lower() and "utilization" not in reason.lower():
                    st.write(reason)
            util_reason = next((r for r in analysis['reasons'] if "utilization" in r.lower()), None)
            if util_reason:
                st.write(util_reason)

            with col_add:
                if st.button("➕ Add Job", use_container_width=True, disabled=False):
                    
                    # --- START FIX: VALIDATION CHECK ---
                    
                    # Get the Job ID the user typed in
                    job_id_to_add = ss.get('live_job_id', '') 
                    
                    # Check if this Job_ID already exists in the base dataframe
                    if job_id_to_add in ss.base_df_ops['Job_ID'].values:
                        st.error(f"❌ Job ID '{job_id_to_add}' already exists. Please enter a unique Job ID.")
                        st.stop() # Stop execution for this button press
                        
                    # --- END FIX ---

                    # If the check passed, continue with the rest of the logic
                    with st.spinner(f"Adding {job_id_to_add} to schedule..."):
                        
                        current_time_days_add = ss.current_schedule['End_Time'].max() / 480 if not ss.current_schedule.empty else 0
                        live_due_days_add = ss.get('live_due', 7) 
                        new_release_time_min = current_time_days_add * 480
                        new_due_time_min = new_release_time_min + (live_due_days_add * 480)

                        df_new_ops = pd.DataFrame(ss.live_job_ops_pending)
                        
                        # Set the Job_ID from the (now validated) widget
                        df_new_ops['Job_ID'] = job_id_to_add 

                        df_new_ops['Release_Day'] = current_time_days_add
                        df_new_ops['Due_Day'] = current_time_days_add + live_due_days_add
                        df_new_ops['Release_Time_Min'] = new_release_time_min
                        df_new_ops['Due_Time_Min'] = new_due_time_min
                        
                        if analysis['recommendation'] == 'OUTSOURCE':
                            df_new_ops['Assignment_Type'] = 'OUTSOURCE'
                            df_new_ops['Outsource_Cost'] = df_new_ops['Quantity'] * 5.0
                            df_new_ops['Outsource_Time_Min'] = live_due_days_add * 480 * 0.8
                        else:
                            df_new_ops['Assignment_Type'] = 'IN_HOUSE'
                            df_new_ops['Outsource_Cost'] = 0
                            df_new_ops['Outsource_Time_Min'] = 0

                        ss.df_ops = pd.concat([ss.df_ops, df_new_ops], ignore_index=True)
                        ss.base_df_ops = pd.concat([ss.base_df_ops, df_new_ops], ignore_index=True)
                        ss.base_df_effective = pd.concat([
                            ss.base_df_effective,
                            ss.live_job_effective_pending
                        ], ignore_index=True)

                        scheduler_new = CNCScheduler(
                            ss.df_ops, ss.df_machines,
                            ss.base_df_effective, ss.base_df_penalties
                        )
                        ss.current_schedule = scheduler_new.run_scheduling(heuristic=ss.current_heuristic)

                        del ss.live_job_analysis
                        del ss.live_job_ops_pending
                        del ss.live_job_effective_pending

                        st.cache_data.clear()
                        st.toast(f"✅ Job {job_id_to_add} added successfully!", icon="🎯")
                    st.rerun()

def draw_system_reset(ss):
    if st.sidebar.button("🔄 Reset System to Original State", use_container_width=True):
        with st.spinner("Resetting system..."):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(ss.keys()):
                del ss[key]
            st.toast("System reset!", icon="♻")
        st.rerun()

def draw_data_export(ss):
    csv_data = export_schedule(ss.current_schedule)
    safe_name = (ss.current_heuristic or "none").lower()
    st.sidebar.download_button(
        label="💾 Export Current Schedule (CSV)",
        data=csv_data,
        file_name=f"cnc_schedule_{safe_name}_current.csv",
        mime='text/csv',
        use_container_width=True
    )

def draw_breakdown_simulator(ss):
    with st.sidebar.expander("🔧 Machine Breakdown Simulator"):
        machine_list = ss.df_machines['Machine_ID'].unique()
        bd_machine = st.selectbox("Machine:", machine_list, key='bd_machine')

        st.sidebar.write("🧩 Min Start Time:", ss.current_schedule["Start_Time"].min() if not ss.current_schedule.empty else 0)
        st.sidebar.write("🧩 Max End Time:", ss.current_schedule["End_Time"].max() if not ss.current_schedule.empty else 0)

        # Dynamic range based on actual schedule
        min_slider_val = 0
        max_slider_val = 60000
        default_val = 100
        
        if not ss.current_schedule.empty:
            min_slider_val = int(ss.current_schedule['Start_Time'].min())
            max_slider_val = int(ss.current_schedule['End_Time'].max())
            default_val = min(int((min_slider_val + max_slider_val) / 2), max_slider_val)

        bd_start = st.slider("Breakdown Start (min):", min_slider_val, max_slider_val, default_val, key='bd_start')
        bd_duration = st.slider("Breakdown Duration (min):", 30, 1000, 120, key='bd_duration')

        if st.button("Simulate Breakdown", key='bd_button'):
            with st.spinner(f"Simulating breakdown for {bd_machine}..."):
                df_machines_temp = ss.df_machines.copy()
                df_ops_temp = ss.df_ops.copy()
                bd_end = bd_start + bd_duration
                machine_idx = df_machines_temp[df_machines_temp['Machine_ID'] == bd_machine].index

                if not machine_idx.empty:
                    idx = machine_idx[0]
                    breakdown_window = {'start': bd_start, 'end': bd_end, 'duration': bd_duration}
                    existing_maint = df_machines_temp.at[idx, 'Maintenance_Window']

                    if existing_maint:
                        st.warning(f"{bd_machine} already has maintenance. Adding breakdown window.")
                        # Merge with existing maintenance (support multiple windows)
                        if isinstance(existing_maint, dict):
                            df_machines_temp.at[idx, 'Maintenance_Window'] = [existing_maint, breakdown_window]
                        elif isinstance(existing_maint, list):
                            df_machines_temp.at[idx, 'Maintenance_Window'] = existing_maint + [breakdown_window]
                    else:
                        df_machines_temp.at[idx, 'Maintenance_Window'] = breakdown_window

                    # ✅ CRITICAL: Identify operations currently scheduled during breakdown
                    affected_ops = []
                    if not ss.current_schedule.empty and 'Machine_ID' in ss.current_schedule.columns:
                        machine_schedule = ss.current_schedule[
                            ss.current_schedule['Machine_ID'] == bd_machine
                        ]
                        
                        for _, op_row in machine_schedule.iterrows():
                            op_start = op_row.get('Start_Time', 0)
                            op_end = op_row.get('End_Time', 0)
                            
                            # Check overlap: operation overlaps breakdown if start < bd_end AND end > bd_start
                            if op_start < bd_end and op_end > bd_start:
                                affected_ops.append({
                                    'Operation_ID': op_row.get('Operation_ID'),
                                    'Job_ID': op_row.get('Job_ID'),
                                    'Original_Start': op_start,
                                    'Original_End': op_end
                                })
                    
                    # ✅ Re-evaluate make-or-buy for affected operations
                    outsourced_count = 0
                    for affected in affected_ops:
                        op_id = affected['Operation_ID']
                        op_data = df_ops_temp[df_ops_temp['Operation_ID'] == op_id]
                        
                        if not op_data.empty:
                            op = op_data.iloc[0]
                            # Re-run make-or-buy decision with current threshold
                            decision, cost, reason = make_or_buy_decision(
                                op, 
                                ss.base_df_effective, 
                                cost_threshold=ss.cost_threshold
                            )
                            
                            # Update assignment if outsourcing is better due to breakdown
                            if decision == 'OUTSOURCE':
                                df_ops_temp.loc[df_ops_temp['Operation_ID'] == op_id, 'Assignment_Type'] = 'OUTSOURCE'
                                outsourced_count += 1
                    
                    # ✅ Persist updated data
                    ss.df_machines = df_machines_temp.copy()
                    ss.base_df_machines = df_machines_temp.copy()
                    ss.df_ops = df_ops_temp.copy()
                    ss.base_df_ops = df_ops_temp.copy()

                    # ✅ LOG ACTIVITY
                    if "activity_log" not in ss:
                        ss.activity_log = []
                    
                    log_details = f"Machine: {bd_machine}, Start: {bd_start} min, Duration: {bd_duration} min, End: {bd_end} min"
                    if len(affected_ops) > 0:
                        affected_op_ids = [op['Operation_ID'] for op in affected_ops]
                        log_details += f" | Affected Operations: {', '.join(affected_op_ids)}"
                    if outsourced_count > 0:
                        log_details += f" | Auto-outsourced: {outsourced_count} ops"
                    
                    ss.activity_log.append({
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'Machine Breakdown Added',
                        'details': log_details,
                        'affected_items': f"{bd_machine} ({len(affected_ops)} ops affected)"
                    })

                    # ✅ Mark recomputation required
                    ss.recalculate_all_heuristics = True
                    ss.breakdown_pending = True
                    ss.breakdown_message_visible = True
                    ss.current_page = "comparison"  # Redirect user to heuristic comparison page

                    # ✅ Clear caches to force fresh recalculation
                    st.cache_data.clear()
                    st.cache_resource.clear()

                    # ✅ User-facing messages
                    st.success(f"🚨 Breakdown added for {bd_machine} "
                               f"(Start={bd_start} min, Duration={bd_duration} min, End={bd_end} min)")
                    
                    if len(affected_ops) > 0:
                        st.warning(f"⚠️ {len(affected_ops)} operation(s) were scheduled during breakdown time")
                        if outsourced_count > 0:
                            st.info(f"📦 {outsourced_count} operation(s) reassigned to OUTSOURCE due to breakdown conflict")
                        st.info(f"🔄 Remaining operations will be rescheduled after breakdown window (after {bd_end} min)")
                        
                        # 🤖 AI Breakdown Impact Analysis
                        if AI_ENABLED:
                            context = {
                                "Machine": bd_machine,
                                "Breakdown Start (min)": bd_start,
                                "Breakdown Duration (min)": bd_duration,
                                "Affected Operations": len(affected_ops),
                                "Auto-Outsourced": outsourced_count,
                                "Affected Operation IDs": [op['Operation_ID'] for op in affected_ops][:5]  # First 5
                            }
                            
                            prompt = f"""
A machine breakdown has occurred on {bd_machine}.

Analyze:
1. Immediate impact on production schedule and delivery commitments
2. Cost implications (outsourcing vs delays)
3. Risk mitigation strategies for affected operations
4. Recommendations for preventing similar disruptions
"""
                            
                            with st.expander("🤖 AI Breakdown Impact Analysis", expanded=False):
                                with st.spinner("🤖 Analyzing breakdown impact..."):
                                    insights = get_ai_insights(prompt, context)
                                    st.markdown(insights)
                    
                    st.info("💡 Click **'🧪 Compute All Heuristics'** to recompute schedules with breakdown enforced.")

                    # ✅ Toast notification for subtle visual feedback
                    st.toast("⚙ Machine breakdown applied — ready for heuristic recomputation.", icon="⚙️")

                    st.rerun()

        # ✅ Persistent message display (after rerun)
        if hasattr(ss, "breakdown_message_visible") and ss.breakdown_message_visible:
            st.info("💡 Please click **'🧪 Compute All Heuristics'** in the sidebar "
                    "to recompute schedules and view updated recommendations.")





def draw_priority_manager(ss):
    with st.sidebar.expander("⚡ Job Priority Manager"):
        job_list = ss.df_ops['Job_ID'].unique()
        priority_job = st.selectbox("Job ID:", job_list, key='priority_job')
        new_priority = st.radio("New Priority:", [1, 2, 3], index=1, horizontal=True, key='priority_val')

        if st.button("Update Priority", key='priority_button'):
            with st.spinner(f"Updating {priority_job} to P{new_priority}..."):
                # Get old priority for logging
                old_priority = ss.df_ops[ss.df_ops['Job_ID'] == priority_job]['Priority'].iloc[0] if not ss.df_ops[ss.df_ops['Job_ID'] == priority_job].empty else None
                
                # ✅ Update dataset priority
                ss.df_ops.loc[ss.df_ops['Job_ID'] == priority_job, 'Priority'] = new_priority
                ss.base_df_ops.loc[ss.base_df_ops['Job_ID'] == priority_job, 'Priority'] = new_priority

                # ✅ LOG ACTIVITY
                if "activity_log" not in ss:
                    ss.activity_log = []
                ss.activity_log.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Priority Updated',
                    'details': f"Job: {priority_job}, Changed from P{old_priority} to P{new_priority}",
                    'affected_items': priority_job
                })

                # ✅ Mark recomputation required
                ss.triggered_by_priority_manager = True
                ss.recalculate_all_heuristics = True
                ss.breakdown_pending = True
                ss.breakdown_message_visible = True   # <-- same variable as breakdown
                ss.current_page = "comparison"

                # ✅ Clean up caches
                st.cache_data.clear()
                st.cache_resource.clear()

                st.success(f"✅ Priority for {priority_job} updated to P{new_priority}.")
                st.toast("⚙ Priority changed — please recompute heuristics.", icon="⚙️")
                trigger_recompute_prompt(ss, f"Priority for {priority_job} updated to P{new_priority}")
        

                st.rerun()

    # ✅ Persistent message (same as breakdown)
    if hasattr(ss, "breakdown_message_visible") and ss.breakdown_message_visible:
        st.info("💡 Please click **'🧪 Compute All Heuristics'** in the sidebar "
                "to recompute schedules and view updated recommendations.")


def draw_outsourcing_policy(ss):
    with st.sidebar.expander("💰 Outsourcing Policy"):
        st.info("**EXPLAINER:** Adjust the cost threshold to control how aggressively operations are outsourced.\n\n"
                "- Lower threshold (0.5–0.8): Mostly in-house\n"
                "- Medium (0.9–1.1): Balanced\n"
                "- High (1.2–1.5): More outsourcing")

        new_threshold = st.slider(
            "Cost Threshold:",
            0.5, 1.5, ss.cost_threshold, 0.05, key='thresh_slider'
        )

        before_outsourced = (ss.df_ops["Assignment_Type"] == "OUTSOURCE").sum()
        total_ops = len(ss.df_ops)
        before_pct = (before_outsourced / total_ops * 100) if total_ops > 0 else 0
        st.write(f"🔹 Current Outsourced: **{before_outsourced}/{total_ops} ({before_pct:.1f}%)**")

        if st.button("Update Policy", key="thresh_button"):
            with st.spinner("Updating Make-or-Buy decisions..."):
                ss.cost_threshold = new_threshold

                for idx, op in ss.df_ops.iterrows():
                    decision, _, _ = make_or_buy_decision(op, ss.base_df_effective, cost_threshold=ss.cost_threshold)
                    ss.df_ops.at[idx, "Assignment_Type"] = decision

                ss.base_df_ops = ss.df_ops.copy()

                after_outsourced = (ss.df_ops["Assignment_Type"] == "OUTSOURCE").sum()
                after_pct = (after_outsourced / total_ops * 100) if total_ops > 0 else 0
                change = after_outsourced - before_outsourced
                direction = "increased" if change > 0 else "decreased"

                st.success(
                    f"💰 Outsourcing threshold set to **{ss.cost_threshold}**.\n"
                    f"📊 Outsourced ops {direction} by **{abs(change)}** "
                    f"({before_pct:.1f}% → {after_pct:.1f}%)."
                )

                # ✅ LOG ACTIVITY
                if "activity_log" not in ss:
                    ss.activity_log = []
                ss.activity_log.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Outsourcing Policy Updated',
                    'details': f"Threshold changed to {new_threshold:.2f} | Outsourced: {before_outsourced} → {after_outsourced} ({after_pct:.1f}%) | {direction.title()} by {abs(change)} ops",
                    'affected_items': f"{abs(change)} operations"
                })

                # ✅ Persistent reminder + navigation
                ss.triggered_by_outsourcing_policy = True
                ss.recalculate_all_heuristics = True
                ss.force_metric_refresh = True
                ss.current_page = "comparison"
                ss.outsourcing_update_message_visible = True
                ss.outsourcing_update_time = time.time()

                st.cache_data.clear()
                st.cache_resource.clear()

                st.toast("💼 Policy updated — please recompute heuristics.", icon="⚙️")
                trigger_recompute_prompt(ss, f"Outsourcing Policy updated (threshold = {ss.cost_threshold})")

                st.rerun()

    # ✅ Persistent message shown until user recomputes or 2 minutes pass
        if hasattr(ss, "outsourcing_update_message_visible") and ss.outsourcing_update_message_visible:
            st.info(
                "💡 Outsourcing policy updated. Click **'🧪 Compute All Heuristics'** in the sidebar "
                "to recompute schedules and view updated heuristic comparison."
            )
            if time.time() - ss.get("outsourcing_update_time", 0) > 120:
                ss.outsourcing_update_message_visible = False



def draw_job_deleter(ss):
    """
    Adds a sidebar widget to delete an entire job (and all its operations)
    from the dataset. Automatically updates the current schedule.
    """
    with st.sidebar.expander("Delete Job"):
        
        # 1. Get a list of all current jobs
        all_jobs = sorted(ss.df_ops['Job_ID'].unique())
        if len(all_jobs) == 0:
            st.write("No jobs to delete.")
            return

        # 2. Selectbox to choose job
        job_to_delete = st.selectbox("Select Job to Delete:", all_jobs, key='job_delete_select')
        
        # 3. Warning
        st.warning(f"⚠️ Deleting **{job_to_delete}** is permanent and will remove ALL its operations.")
        
        # 4. Button
        if st.button("Delete This Entire Job", key='job_delete_button', type="primary"):
            with st.spinner(f"Deleting {job_to_delete} and all its operations..."):
                
                # 5. Find all operations to delete (BEFORE deleting from df_ops)
                ops_to_delete = ss.df_ops[ss.df_ops['Job_ID'] == job_to_delete]['Operation_ID'].unique()
                
                # 6. Remove from the working dataframe
                ss.df_ops = ss.df_ops[ss.df_ops['Job_ID'] != job_to_delete].copy()
                
                # 7. Remove from the base dataframe for persistence
                ss.base_df_ops = ss.base_df_ops[ss.base_df_ops['Job_ID'] != job_to_delete].copy()
                
                # 8. CRITICAL: Remove from the effective times dataframe
                ss.base_df_effective = ss.base_df_effective[~ss.base_df_effective['Operation_ID'].isin(ops_to_delete)].copy()

                # ✅ LOG ACTIVITY
                if "activity_log" not in ss:
                    ss.activity_log = []
                ss.activity_log.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Job Deleted',
                    'details': f"Deleted Job: {job_to_delete} | Operations removed: {len(ops_to_delete)} ({', '.join(ops_to_delete)})",
                    'affected_items': f"{job_to_delete} ({len(ops_to_delete)} ops)"
                })

                st.success(f"✅ Deleted {job_to_delete} ({len(ops_to_delete)} operations removed).")

                # 9. Immediately recompute the *current* schedule
                if not ss.current_heuristic:
                    st.warning("No heuristic selected. Deleting data only.")
                    ss.current_schedule = pd.DataFrame() # Clear schedule
                    ss.recalculate_all_heurISTICS = True # Force user to compute
                    ss.current_page = "comparison"
                    st.rerun()
                    return

                st.write(f"🔄 Re-running {ss.current_heuristic} schedule...")

                scheduler_new = CNCScheduler(
                    ss.df_ops, # Use the just-updated df_ops
                    ss.df_machines,
                    ss.base_df_effective, # Use the just-updated df_effective
                    ss.base_df_penalties
                )
                ss.current_schedule = scheduler_new.run_scheduling(heuristic=ss.current_heuristic)

                st.cache_data.clear()
                st.toast(f"✅ Job {job_to_delete} deleted. Schedule updated.", icon="🗑️")
                
                # 10. Rerun the app
                st.rerun()
# ---------------------------
# Main page content functions
# ---------------------------
def draw_kpi_dashboard(ss):
    if ss.get("current_heuristic"):
        st.header(f"📊 KPI Dashboard ({ss.current_heuristic})")
    else:
        st.warning("⚠️ No heuristic selected. Please compute and select one from the comparison page.")

    st.caption(f"🧠 DEBUG: KPI source rows = {len(ss.current_schedule) if ss.current_schedule is not None else 0}")

    
    # 🚨 Force unique cache key each render to bypass stale cache
    _cache_key = str(time.time())

    kpi_fig, metrics = create_kpi_dashboard(
        ss.current_schedule if ss.current_schedule is not None else pd.DataFrame(),
        ss.df_ops,
        ss.df_machines,
        ss.current_heuristic or "",
        _cache_key=_cache_key
    )

    st.plotly_chart(kpi_fig, use_container_width=True)
    st.caption(f"Last Updated: {time.strftime('%H:%M:%S')} ({ss.current_heuristic})")

    col1, col2, col3 = st.columns(3)
    col1.metric("Makespan (Days)", metrics['Makespan_Days'])
    col2.metric("Total Tardiness (Days)", metrics['Total_Tardiness_Days'])
    col3.metric("On-Time %", metrics['On_Time_%'])
    
    # 🤖 AI Insights Button
    if AI_ENABLED and st.button("🤖 Get AI Insights on Performance", key="ai_kpi_insights"):
        with st.spinner("🤖 AI analyzing performance metrics..."):
            context = {
                "Heuristic": ss.current_heuristic,
                "Makespan (Days)": metrics['Makespan_Days'],
                "Total Tardiness (Days)": metrics['Total_Tardiness_Days'],
                "On-Time Delivery %": metrics['On_Time_%'],
                "Machine Utilization %": metrics['Machine_Utilization_%'],
                "Total Cost ($)": metrics['Total_Cost_$'],
                "Late Operations": metrics.get('Late_Operations', 0),
                "Total Operations": metrics.get('Total_Operations', 0)
            }
            
            prompt = f"""
Analyze this CNC scheduling performance using the {ss.current_heuristic} heuristic.

Identify:
1. Key strengths of the current schedule
2. Performance bottlenecks or concerns
3. Specific actionable recommendations to improve metrics
4. Whether a different heuristic might perform better and why
"""
            
            insights = get_ai_insights(prompt, context)
            st.info("🤖 **AI-Powered Performance Analysis:**")
            st.markdown(insights)


def draw_gantt_tab(ss):
    st.header(f"📈 Gantt Chart ({ss.current_heuristic or 'N/A'})")
    with st.spinner("Generating Gantt chart..."):
        _cache_key = str(time.time())

        gantt_fig = create_gantt_chart(
            ss.current_schedule if not ss.current_schedule is None else pd.DataFrame(),
            ss.df_machines,
            f"{ss.current_heuristic or 'N/A'} Schedule",
            _cache_key=_cache_key,
            machines_order=ss.machine_order
        )
        st.plotly_chart(gantt_fig, use_container_width=True)
        
        # 🤖 AI Gantt Chart Insights
        if AI_ENABLED and not ss.current_schedule.empty:
            if st.button("🤖 Get AI Insights on Schedule Visualization", key="ai_gantt_insights"):
                with st.spinner("🤖 AI analyzing schedule timeline..."):
                    machine_utilization = {}
                    makespan = ss.current_schedule['End_Time'].max()
                    
                    for machine in ss.df_machines['Machine_ID'].unique():
                        machine_ops = ss.current_schedule[ss.current_schedule['Machine_ID'] == machine]
                        if not machine_ops.empty:
                            productive_time = (machine_ops['Setup_Time'].sum() + 
                                             machine_ops['Proc_Time'].sum() + 
                                             machine_ops['Transfer_Time'].sum())
                            util = (productive_time / makespan * 100) if makespan > 0 else 0
                            machine_utilization[machine] = round(util, 1)
                    
                    # Check for breakdowns in schedule
                    breakdown_count = 0
                    for _, machine in ss.df_machines.iterrows():
                        maint = machine.get('Maintenance_Window')
                        if maint:
                            if isinstance(maint, list):
                                breakdown_count += len(maint)
                            elif isinstance(maint, dict):
                                breakdown_count += 1
                    
                    context = {
                        "Heuristic": ss.current_heuristic,
                        "Makespan (min)": makespan,
                        "Total Operations Scheduled": len(ss.current_schedule),
                        "Machine Utilization": machine_utilization,
                        "Active Breakdowns": breakdown_count,
                        "Machines": len(ss.df_machines)
                    }
                    
                    prompt = f"""
Analyze this Gantt chart visualization of the CNC scheduling timeline.

Provide insights on:
1. Load balancing across machines (identify overloaded or underutilized machines)
2. Potential scheduling bottlenecks or idle time gaps
3. Impact of any breakdowns/maintenance windows on the schedule
4. Recommendations for better resource allocation
"""
                    
                    insights = get_ai_insights(prompt, context)
                    st.info("🤖 **AI Schedule Timeline Analysis:**")
                    st.markdown(insights)

def draw_operation_status_tab(ss):
    st.header(f"📋 Operation Status ({ss.current_heuristic or 'N/A'})")
    with st.spinner("Generating operation status table..."):
        _cache_key = getattr(ss, "schedule_update_key", str(time.time()))
        status_table = create_operation_status_table(ss.current_schedule.copy() if not ss.current_schedule is None else pd.DataFrame(), ss.df_ops.copy(), _cache_key=_cache_key)

        st.info(f"🔄 Cache Key: {getattr(ss, 'schedule_update_key', 'N/A')}")
        st.dataframe(status_table, use_container_width=True, height=500)
        
        # 🤖 AI Operation Status Insights
        if AI_ENABLED and not status_table.empty and 'Status' in status_table.columns:
            if st.button("🤖 Get AI Insights on Operation Status", key="ai_operation_insights"):
                with st.spinner("🤖 AI analyzing operation status..."):
                    late_ops = len(status_table[status_table['Status'] == '⏰ Late']) if 'Status' in status_table.columns else 0
                    pending_ops = len(status_table[status_table['Status'] == '⏳ Pending']) if 'Status' in status_table.columns else 0
                    completed_ops = len(status_table[status_table['Status'] == '✅ Completed']) if 'Status' in status_table.columns else 0
                    outsourced_ops = len(status_table[status_table['Assignment'] == 'OUTSOURCE']) if 'Assignment' in status_table.columns else 0
                    
                    context = {
                        "Total Operations": len(status_table),
                        "Late Operations": late_ops,
                        "Pending Operations": pending_ops,
                        "Completed Operations": completed_ops,
                        "Outsourced Operations": outsourced_ops,
                        "Current Heuristic": ss.current_heuristic or 'None'
                    }
                    
                    prompt = f"""
Analyze the current operation status distribution in the CNC scheduling system.

Provide:
1. Assessment of schedule health based on late/pending/completed ratios
2. Potential bottlenecks or scheduling inefficiencies
3. Recommendations for operations that might benefit from priority adjustments
4. Suggestions for optimizing the mix of in-house vs outsourced work
"""
                    
                    insights = get_ai_insights(prompt, context)
                    st.info("🤖 **AI Operation Status Analysis:**")
                    st.markdown(insights)

def draw_comparison_tab(ss):
    st.header("⚖ Heuristic Comparison")

    with st.expander("🔧 DEBUG INFO (Click to see state)"):
        st.write(f"recalculate_all_heuristics flag: {getattr(ss, 'recalculate_all_heuristics', False)}")
        st.write(f"force_metric_refresh flag: {getattr(ss, 'force_metric_refresh', False)}")
        st.write(f"schedule_update_key: {ss.get('schedule_update_key', 'NOT SET')}")
        st.write(f"Has df_metrics: {hasattr(ss, 'df_metrics')}")
        if hasattr(ss, 'df_metrics'):
            st.write(f"df_metrics shape: {ss.df_metrics.shape}")

    st.info("**EXPLAINER**: This tab compares all 4 scheduling algorithms on the CURRENT dataset.")

    # 🤖 AI DATASET INSIGHTS SECTION
    if AI_ENABLED:
        with st.expander("🤖 AI-Generated Dataset Insights", expanded=False):
            st.caption("📊 **Comprehensive Dataset Analysis**")
            
            if st.button("🚀 Generate Dataset Insights", key="ai_dataset_insights", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analyzing entire dataset and scheduling environment..."):
                    # Gather comprehensive dataset statistics
                    total_jobs = ss.base_df_ops['Job_ID'].nunique() if hasattr(ss, 'base_df_ops') else 0
                    total_operations = len(ss.base_df_ops) if hasattr(ss, 'base_df_ops') else 0
                    total_machines = len(ss.base_df_machines) if hasattr(ss, 'base_df_machines') else 0
                    
                    # Operation type distribution
                    op_type_dist = ss.base_df_ops['Op_Type'].value_counts().to_dict() if hasattr(ss, 'base_df_ops') and 'Op_Type' in ss.base_df_ops.columns else {}
                    
                    # Assignment type distribution
                    assignment_dist = ss.base_df_ops['Assignment_Type'].value_counts().to_dict() if hasattr(ss, 'base_df_ops') and 'Assignment_Type' in ss.base_df_ops.columns else {}
                    
                    # Priority distribution
                    priority_dist = ss.base_df_ops['Priority'].value_counts().to_dict() if hasattr(ss, 'base_df_ops') and 'Priority' in ss.base_df_ops.columns else {}
                    
                    # Material type distribution
                    material_dist = ss.base_df_ops['Mat_Type'].value_counts().to_dict() if hasattr(ss, 'base_df_ops') and 'Mat_Type' in ss.base_df_ops.columns else {}
                    
                    # Time-based metrics
                    avg_proc_time = ss.base_df_ops['Total_Proc_Min'].mean() if hasattr(ss, 'base_df_ops') and 'Total_Proc_Min' in ss.base_df_ops.columns else 0
                    avg_setup_time = ss.base_df_ops['Setup_Time'].mean() if hasattr(ss, 'base_df_ops') and 'Setup_Time' in ss.base_df_ops.columns else 0
                    
                    # Deadline analysis
                    avg_release_day = ss.base_df_ops['Release_Day'].mean() if hasattr(ss, 'base_df_ops') and 'Release_Day' in ss.base_df_ops.columns else 0
                    avg_due_day = ss.base_df_ops['Due_Day'].mean() if hasattr(ss, 'base_df_ops') and 'Due_Day' in ss.base_df_ops.columns else 0
                    avg_lead_time = avg_due_day - avg_release_day
                    
                    # Machine availability
                    machines_with_maintenance = 0
                    if hasattr(ss, 'base_df_machines'):
                        for _, machine in ss.base_df_machines.iterrows():
                            if machine.get('Maintenance_Window'):
                                machines_with_maintenance += 1
                    
                    # Cost analysis
                    avg_outsource_cost = 0
                    if hasattr(ss, 'base_df_ops') and 'Outsource_Cost' in ss.base_df_ops.columns:
                        outsourced = ss.base_df_ops[ss.base_df_ops['Assignment_Type'] == 'OUTSOURCE']
                        if not outsourced.empty:
                            avg_outsource_cost = outsourced['Outsource_Cost'].mean()
                    
                    context = {
                        "Total Jobs": total_jobs,
                        "Total Operations": total_operations,
                        "Total Machines": total_machines,
                        "Operation Types": op_type_dist,
                        "Assignment Distribution": assignment_dist,
                        "Priority Distribution": priority_dist,
                        "Material Types": material_dist,
                        "Average Processing Time (min)": round(avg_proc_time, 2),
                        "Average Setup Time (min)": round(avg_setup_time, 2),
                        "Average Lead Time (days)": round(avg_lead_time, 2),
                        "Machines with Maintenance/Breakdown": machines_with_maintenance,
                        "Average Outsourcing Cost ($)": round(avg_outsource_cost, 2),
                        "Current Outsourcing Threshold": ss.cost_threshold if hasattr(ss, 'cost_threshold') else 'N/A'
                    }
                    
                    prompt = f"""
You are analyzing a CNC manufacturing dataset with {total_jobs} jobs and {total_operations} operations across {total_machines} machines.

Provide a comprehensive analysis covering:

1. **Dataset Characteristics & Complexity**
   - Overall workload assessment
   - Operation mix diversity
   - Material variety implications

2. **Capacity & Resource Analysis**
   - Machine utilization potential
   - Bottleneck risks based on operation types
   - Maintenance impact on capacity

3. **Scheduling Challenges**
   - Time constraint analysis (lead times vs processing times)
   - Priority distribution implications
   - Setup time vs processing time ratio concerns

4. **Make-or-Buy Strategy**
   - Current outsourcing rate assessment
   - Cost-effectiveness of current threshold
   - Strategic recommendations for in-house vs outsourcing

5. **Risk Factors & Opportunities**
   - Identify potential scheduling risks
   - Optimization opportunities
   - Data-driven recommendations for improved performance

6. **Operational Insights**
   - Workload distribution patterns
   - Material setup implications
   - Suggested operational improvements

Provide specific, actionable insights based on the data. Be concise but comprehensive.
"""
                    
                    insights = get_ai_insights(prompt, context)
                    
                    # Display insights in a structured format
                    st.markdown("---")
                    st.markdown("### 📊 **AI Dataset Analysis Report**")
                    st.markdown(insights)
                    st.markdown("---")
                    
                    # Show key statistics
                    st.markdown("#### 📈 **Key Dataset Statistics**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Jobs", total_jobs)
                        st.metric("Total Operations", total_operations)
                    with col2:
                        st.metric("Total Machines", total_machines)
                        st.metric("Machines w/ Maintenance", machines_with_maintenance)
                    with col3:
                        st.metric("Avg Processing (min)", f"{avg_proc_time:.1f}")
                        st.metric("Avg Setup (min)", f"{avg_setup_time:.1f}")
                    with col4:
                        st.metric("Avg Lead Time (days)", f"{avg_lead_time:.1f}")
                        outsource_pct = (assignment_dist.get('OUTSOURCE', 0) / total_operations * 100) if total_operations > 0 else 0
                        st.metric("Outsourced %", f"{outsource_pct:.1f}%")
                    
                    # Distribution charts
                    if op_type_dist:
                        st.markdown("#### 📊 **Operation Type Distribution**")
                        st.bar_chart(pd.Series(op_type_dist))
                    
                    if priority_dist:
                        st.markdown("#### ⚡ **Priority Distribution**")
                        st.bar_chart(pd.Series(priority_dist))

    # ✅ ACTIVITY LOG DISPLAY
    if hasattr(ss, 'activity_log') and len(ss.activity_log) > 0:
        activity_count = len(ss.activity_log)
        with st.expander(f"📋 Activity Log ({activity_count} activities recorded)", expanded=True):
            log_df = pd.DataFrame(ss.activity_log)
            # Reverse to show most recent first
            log_df = log_df.iloc[::-1].reset_index(drop=True)
            
            st.caption("**Most Recent Activities First** ⬇️")
            st.dataframe(
                log_df[['timestamp', 'action', 'details', 'affected_items']], 
                use_container_width=True,
                height=300
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📊 Total: {activity_count} logged activities")
            with col2:
                # Download activity log
                log_csv = log_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download CSV",
                    data=log_csv,
                    file_name=f"activity_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
            
            # 🤖 AI Activity Log Analysis
            if AI_ENABLED:
                if st.button("🤖 Analyze Activity Patterns", key="ai_activity_analysis"):
                    with st.spinner("🤖 AI analyzing activity patterns..."):
                        recent_activities = log_df.head(10).to_dict('records')
                        action_counts = log_df['action'].value_counts().to_dict()
                        
                        context = {
                            "Total Activities": activity_count,
                            "Recent Activities (last 10)": [f"{a['timestamp']}: {a['action']}" for a in recent_activities],
                            "Action Distribution": action_counts,
                            "Most Common Action": log_df['action'].mode()[0] if not log_df['action'].mode().empty else "N/A"
                        }
                        
                        prompt = f"""
Analyze the activity log patterns from this CNC scheduling system.

Provide insights on:
1. Overall system usage patterns and trends
2. Frequency and timing of critical actions (breakdowns, priority changes, etc.)
3. Potential operational inefficiencies or concerns based on activity patterns
4. Recommendations for improving workflow based on observed activities
"""
                        
                        insights = get_ai_insights(prompt, context)
                        st.info("🤖 **AI Activity Pattern Analysis:**")
                        st.markdown(insights)
    else:
        st.info("📋 **Activity Log**: No activities recorded yet. Actions will appear here once you start using the system.")

    # ✅ Automatically recompute metrics when dataset changes
    recalc_flag = st.session_state.get('recalculate_all_heuristics', False)
    if recalc_flag:
        st.warning("⚠️ DETECTED: Dataset changed — recalculating comparison metrics...")
        with st.spinner("🔄 Recalculating all 4 heuristics..."):
            try:
                metrics = []
                heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
                for i, heur in enumerate(heuristics):
                    st.write(f"  {i+1}/4: Computing {heur} metrics...")
                    schedule_key = f'schedule_{heur.lower()}'
                    schedule = getattr(ss, schedule_key, None)
                    if schedule is not None:
                        metric = calculate_metrics(schedule.copy(), ss.base_df_ops.copy(), heur)
                        metrics.append(metric)
                        st.write(f"    ✅ {heur}: Makespan={metric.get('Makespan_Days', 'N/A'):.2f}")
                    else:
                        st.warning(f"    ❌ {heur}: Schedule not found!")
                ss.df_metrics = pd.DataFrame(metrics)
                st.session_state.recalculate_all_heuristics = False
                st.success("✅ Comparison table UPDATED!")
            except Exception as e:
                st.error(f"❌ Error recalculating metrics: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
                ss.recalculate_all_heuristics = False

    # ✅ Display comparison table and recommendation
    if hasattr(ss, 'df_metrics') and ss.df_metrics is not None and len(ss.df_metrics) > 0:
        st.write("**Performance Comparison:**")
        display_cols = [c for c in ss.df_metrics.columns if not c.startswith('_')]
        try:
            st.dataframe(ss.df_metrics[display_cols], use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying table: {str(e)}")

        # ✅ Recommend best heuristic (lowest Total_Tardiness_Days)
        try:
            best_row = ss.df_metrics.sort_values('Total_Tardiness_Days').iloc[0]
            st.success(f"🏆 Recommended heuristic (by lowest Total Tardiness): **{best_row['Heuristic']}**")
            st.write("Recommendation details:")
            st.write(best_row.to_dict())
            
            # 🤖 AI Insights Button for Heuristic Comparison
            if AI_ENABLED:
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🤖 AI Analysis", key="ai_heuristic_comparison", use_container_width=True):
                        with st.spinner("🤖 AI comparing all heuristics..."):
                            comparison_data = ss.df_metrics.to_dict('records')
                            context = {
                                "Number of Heuristics": len(comparison_data),
                                "Comparison Metrics": comparison_data,
                                "Recommended": best_row['Heuristic'],
                                "Total Operations": len(ss.base_df_ops),
                                "Total Machines": len(ss.base_df_machines)
                            }
                            
                            prompt = f"""
Compare these {len(comparison_data)} scheduling heuristics: {', '.join([h['Heuristic'] for h in comparison_data])}.

Provide:
1. Why {best_row['Heuristic']} is recommended based on the metrics
2. Trade-offs between different heuristics
3. When you might choose a different heuristic despite the recommendation
4. Specific business scenarios where each heuristic excels
"""
                            
                            insights = get_ai_insights(prompt, context)
                            st.info("🤖 **AI Heuristic Comparison Analysis:**")
                            st.markdown(insights)
        except Exception:
            pass

        with st.expander("📖 Metrics Explained"):
            st.markdown("""
            - **Makespan_Days**: Total time to complete all operations (lower is better)
            - **Total_Tardiness_Days**: Sum of all delays (lower is better)
            - **Late_Operations**: Count of late deliveries (lower is better)
            - **On_Time_%**: On-time delivery rate (higher is better)
            - **Machine_Utilization_%**: Machine busy rate (higher is better)
            - **Total_Cost_$**: Total cost (lower is better)
            """)
    else:
        st.info("📊 No comparison data yet. Click '🧪 Compute All Heuristics' in the sidebar to generate comparison results.")

def handle_error(e):
    st.error(f"❌ ERROR: An error occurred during execution")
    st.error(f"Error Type: {type(e).__name__}")
    st.error(f"Error Message: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# ---------------------------
# MAIN
# ---------------------------
if 'recalculate_all_heuristics' not in st.session_state:
    st.session_state.recalculate_all_heuristics = False
if 'force_metric_refresh' not in st.session_state:
    st.session_state.force_metric_refresh = False

def initialize_app(ss):
    dbg("🔧 DEBUG: System not initialized, starting initialization...")
    dbg("🔄 DEBUG: Clearing old cached data...")
    st.cache_data.clear()

    with st.spinner("Loading and preprocessing data..."):
        SAMPLE_SIZE = 50  # Change to None to use full dataset
        df_ops, df_machines, df_effective, df_penalties, df_vendors = load_all_data(sample_size=SAMPLE_SIZE)
        dbg("✅ DEBUG: Data loaded successfully")
        dbg(f"  - Jobs: {df_ops['Job_ID'].nunique()}, Operations: {len(df_ops)}")
        dbg(f"  - Machines: {len(df_machines)}")
        dbg(f"  - Effective times: {len(df_effective)}")

        ss.base_df_ops = df_ops
        ss.base_df_machines = df_machines
        ss.base_df_effective = df_effective
        ss.base_df_penalties = df_penalties
        ss.base_df_vendors = df_vendors
        ss.machine_order = sorted(ss.base_df_machines['Machine_ID'].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

        dbg("✅ DEBUG: Base data stored in session state")

    # DO NOT run SPT automatically — user will compute heuristics explicitly
    ss.schedule_spt = None
    ss.schedule_edd = None
    ss.schedule_cr = None
    ss.schedule_priority = None

    ss.df_ops = ss.base_df_ops.copy()
    ss.df_machines = ss.base_df_machines.copy()
    ss.current_schedule = pd.DataFrame()
    ss.cost_threshold = 0.9
    ss.initialized = True
    if "current_heuristic" not in ss:
        ss.current_heuristic = None
    if "last_applied_heuristic" not in ss:
        ss.last_applied_heuristic = None
    
    # ✅ Initialize Activity Log
    if "activity_log" not in ss:
        ss.activity_log = []
        ss.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'System Initialized',
            'details': f'Loaded {len(df_ops)} operations, {len(df_machines)} machines',
            'affected_items': 'All'
        })



    st.info("🔎 Raw dataset loaded. Click 'Compute All Heuristics' in the sidebar to run scheduling on current data.")
    st.toast("System initialized (raw data loaded).", icon="✅")

def main():
    st.title("🏭 Advanced CNC Job Scheduling System (Operation-Specific)")
    st.success("✅ **VERSION 2.0 (Operation-Level)** - Job → Operation granularity applied")

    if 'cache_version' not in st.session_state or st.session_state.cache_version < 2:
        st.error("⚠️ **ACTION REQUIRED**: Old cached data detected! Click '🔄 Reset System' in the sidebar to apply fixes.")
        st.session_state.cache_version = 2

    ss = st.session_state
    if "current_page" not in ss:
        ss.current_page = "comparison"  # default landing page


    try:
        if 'initialized' not in ss:
            initialize_app(ss)
        if 'schedule_update_key' not in ss:
            ss.schedule_update_key = str(time.time())

        st.sidebar.title("Controls")
        page = st.sidebar.radio(
            "📋 Select Page",
            ["Heuristic Comparison", "Selected Heuristic View"],
            index=0 if ss.current_page == "comparison" else 1
        )
        ss.current_page = "comparison" if page == "Heuristic Comparison" else "heuristic_view"
        st.sidebar.divider()

        # New compute/apply controls
        draw_compute_apply_controls(ss)
        st.sidebar.divider()

        # draw_heuristic_selector(ss)
        # st.sidebar.divider()
        draw_live_job_scheduler(ss)
        st.sidebar.divider()
        draw_system_reset(ss)
        draw_data_export(ss)
        st.sidebar.divider()
        draw_breakdown_simulator(ss)
        draw_priority_manager(ss)
        draw_job_deleter(ss)
        draw_outsourcing_policy(ss)

        # Main view
        # ---------------- MAIN PAGE ROUTING ----------------
        if ss.current_page == "comparison":
            st.title("⚖ Heuristic Comparison & Recommendation")
            st.info("Compare all scheduling heuristics on the current dataset and choose the best one to apply.")
            draw_comparison_tab(ss)
            st.divider()
            st.markdown("### 💡 Once you choose a heuristic, go to 'Selected Heuristic View' from the sidebar to see details.")
        else:
            st.title(f"📊 {ss.current_heuristic or 'No heuristic selected yet'} — Detailed View")
            if ss.current_heuristic:
                draw_kpi_dashboard(ss)
                tab1, tab2 = st.tabs(["📈 Gantt Chart", "📋 Operation Status"])
                with tab1:
                    draw_gantt_tab(ss)
                with tab2:
                    draw_operation_status_tab(ss)
            else:
                st.warning("⚠️ No heuristic has been applied yet. Go to 'Heuristic Comparison' to choose one.")



    except Exception as e:
        handle_error(e)

if __name__ == "__main__":
    main()
