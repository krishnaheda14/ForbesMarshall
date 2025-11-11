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

# Load environment variables (API keys)
load_dotenv()

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

    # Small visual feedback (guarded)
    safe_toast("⚙ Update registered — ready for heuristic recomputation.", icon="⚙️")

# ---------------------------
# Helper: make debug visible
# ---------------------------
def dbg(msg):
    # simple debug helper that writes to Streamlit if available
    try:
        st.write(msg)
    except Exception:
        print(msg)


# Guarded toast wrapper: some Streamlit installs may not support st.toast
def safe_toast(message, icon=None):
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

# ---------------------------
# AI Helper Functions (Gemini)
# ---------------------------
def get_gemini_model():
    """Initialize and return Gemini AI model if API key is available."""
    try:
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-pro')
        return None
    except ImportError:
        return None
    except Exception as e:
        st.warning(f"Gemini AI initialization failed: {str(e)}")
        return None

def generate_ai_insights(prompt, context_data=None):
    """Generate AI insights using Gemini."""
    model = get_gemini_model()
    if not model:
        return None
    
    try:
        full_prompt = prompt
        if context_data:
            full_prompt = f"{context_data}\n\n{prompt}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"AI generation failed: {str(e)}")
        return None

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
        while True:
            adjusted = False
            for mw in maintenance_list:
                mw_start = mw.get('start', 0)
                mw_end = mw.get('end', 0)

                # Skip invalid windows
                if mw_end <= mw_start:
                    continue

                end_time = current_avail + duration

                # Check if this job would overlap the maintenance window
                overlap = (current_avail < mw_end) and (end_time > mw_start)
                if overlap:
                    # Move start to after maintenance end
                    current_avail = mw_end
                    adjusted = True
                    break  # Recheck all windows again
            if not adjusted:
                break

        # ✅ Update machine availability to reflect the new time slot
        self.machine_availability[machine_id] = current_avail + duration

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

    for _, machine in machines_df.iterrows():
        maint = machine.get("Maintenance_Window")
        machine_id = machine.get("Machine_ID")
        if maint and "start" in maint and "end" in maint and machine_id in all_machines_sorted:
            fig.add_shape(
                type="rect",
                x0=maint["start"] - x_min,
                x1=maint["end"] - x_min,
                y0=all_machines_sorted.index(machine_id) - 0.4,
                y1=all_machines_sorted.index(machine_id) + 0.4,
                fillcolor="rgba(255,0,0,0.25)",
                line_width=0,
                layer="below",
            )

    pad = max((x_max - x_min) * 0.05, 100)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='black')),
        xaxis_title=dict(text="Time (minutes, shifted)", font=dict(size=14, color='black')),
        yaxis_title=dict(text="Machine", font=dict(size=14, color='black')),
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100, r=50, t=80, b=60),
        font=dict(size=12, color='black'),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=all_machines_sorted,
        autorange="reversed",
        type="category",
        tickfont=dict(size=12, color='black'),
        titlefont=dict(size=14, color='black'),
    )
    fig.update_xaxes(
        range=[0 - pad, (x_max - x_min) + pad], 
        showgrid=True,
        tickfont=dict(size=12, color='black'),
        titlefont=dict(size=14, color='black'),
    )
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

    # ✅ Notify success
    st.success("✅ All heuristics recomputed successfully.")
    safe_toast("📊 Updated heuristic metrics ready for review!", icon="📈")

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
        safe_toast(f"🔁 Loaded updated schedule for {current_h}", icon="📈")
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
            safe_toast(f"📊 Viewing {ss.current_heuristic} schedule (not applied).", icon="ℹ️")
            st.rerun()

def draw_live_job_scheduler(ss):
    st.write("**➕ Add New Job**")
    
    col1, col2 = st.columns(2)
    with col1:
        live_job_id = st.text_input("Job ID:", f"J{900 + len(ss.df_ops['Job_ID'].unique())}", key='live_job_id')
        live_quantity = st.number_input("Quantity:", min_value=10, max_value=1000, value=100, step=10, key='live_qty')
    with col2:
        live_priority = st.selectbox("Priority:", [1, 2, 3], index=0, key='live_priority')
        live_due_days = st.number_input("Due (days):", min_value=1, max_value=30, value=7, step=1, key='live_due')

    live_op_count = st.slider("Operations:", 1, 5, 2, key='live_op_count')

    operations_config = []
    for i in range(live_op_count):
        with st.container():
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                op_type = st.selectbox(f"Op{i+1}:", ['MILLING', 'TURNING', 'GRINDING', 'DRILLING'], key=f'live_op{i}_type')
            with col_b:
                material = st.selectbox(f"Mat:", ['STEEL', 'ALUM', 'TITAN', 'BRASS'], key=f'live_op{i}_mat')
            with col_c:
                proc_time = st.number_input(f"Time:", min_value=0.1, max_value=5.0, value=0.3, step=0.1, key=f'live_op{i}_time')
            with col_d:
                setup_time = st.number_input(f"Setup:", min_value=10, max_value=120, value=30, step=5, key=f'live_op{i}_setup')

            operations_config.append({
                'op_type': op_type,
                'material': material,
                'proc_time': proc_time,
                'setup_time': setup_time
            })

    col_analyze, col_add = st.columns(2)
    with col_analyze:
        if st.button("🔍 Analyze", key='analyze_capacity_button', use_container_width=True):
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

        # Analysis results display
        if hasattr(ss, 'live_job_analysis') and ss.live_job_analysis:
            st.divider()
            analysis = ss.live_job_analysis
            
            if analysis['recommendation'] == 'SCHEDULE':
                st.success("✅ Can schedule in-house")
            else:
                st.error("❌ Recommend outsourcing")
            
            st.caption(f"Completion: Day {analysis['metrics'].get('new_job_time_days', 0):.1f} • "
                      f"Utilization: {analysis['metrics'].get('projected_utilization', 0):.1f}%")

        with col_add:
            if st.button("➕ Add Job", key='add_job_button', use_container_width=True, disabled=not hasattr(ss, 'live_job_analysis')):
                # Validation check
                job_id_to_add = ss.get('live_job_id', '') 
                if job_id_to_add in ss.base_df_ops['Job_ID'].values:
                    st.error(f"Job '{job_id_to_add}' exists!")
                    st.stop()

                with st.spinner(f"Adding job..."):
                    current_time_days_add = ss.current_schedule['End_Time'].max() / 480 if not ss.current_schedule.empty else 0
                    live_due_days_add = ss.get('live_due', 7) 
                    new_release_time_min = current_time_days_add * 480
                    new_due_time_min = new_release_time_min + (live_due_days_add * 480)

                    df_new_ops = pd.DataFrame(ss.live_job_ops_pending)
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
                    ss.base_df_effective = pd.concat([ss.base_df_effective, ss.live_job_effective_pending], ignore_index=True)

                    scheduler_new = CNCScheduler(ss.df_ops, ss.df_machines, ss.base_df_effective, ss.base_df_penalties)
                    ss.current_schedule = scheduler_new.run_scheduling(heuristic=ss.current_heuristic)

                    del ss.live_job_analysis
                    del ss.live_job_ops_pending
                    del ss.live_job_effective_pending

                    st.cache_data.clear()
                    safe_toast(f"✅ Job {job_id_to_add} added!", icon="🎯")
                st.rerun()

def draw_job_deleter(ss):
    st.write("**🗑️ Delete Job**")
    all_jobs = sorted(ss.df_ops['Job_ID'].unique())
    
    if len(all_jobs) == 0:
        st.write("No jobs to delete.")
        return
    
    job_to_delete = st.selectbox("Select job:", all_jobs, key='delete_job_select')
    
    if st.button("Delete", key='delete_job_button', type="secondary", use_container_width=True):
        with st.spinner(f"Deleting {job_to_delete}..."):
            # Find all operations to delete
            ops_to_delete = ss.df_ops[ss.df_ops['Job_ID'] == job_to_delete]['Operation_ID'].unique()
            
            # Remove from dataframes
            ss.df_ops = ss.df_ops[ss.df_ops['Job_ID'] != job_to_delete].copy()
            ss.base_df_ops = ss.base_df_ops[ss.base_df_ops['Job_ID'] != job_to_delete].copy()
            ss.base_df_effective = ss.base_df_effective[~ss.base_df_effective['Operation_ID'].isin(ops_to_delete)].copy()
            
            # Recompute schedule if heuristic is active
            if ss.current_heuristic:
                scheduler = CNCScheduler(ss.df_ops, ss.df_machines, ss.base_df_effective, ss.base_df_penalties)
                ss.current_schedule = scheduler.run_scheduling(heuristic=ss.current_heuristic)
            else:
                ss.current_schedule = pd.DataFrame()
                ss.recalculate_all_heuristics = True
            
            st.cache_data.clear()
            safe_toast(f"✅ Deleted {job_to_delete}", icon="🗑️")
        st.rerun()

def draw_breakdown_simulator(ss):
    st.write("**⚙️ Machine Breakdown**")
    st.info("Simulate machine breakdowns (feature placeholder)")

def draw_priority_manager(ss):
    st.write("**📊 Job Priority**")
    all_jobs = sorted(ss.df_ops['Job_ID'].unique())
    if len(all_jobs) == 0:
        st.write("No jobs available")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        job_sel = st.selectbox("Job:", all_jobs, key='priority_job')
    with col2:
        new_priority = st.radio("Priority:", [1, 2, 3, 4], horizontal=True, key='priority_val')
    
    if st.button("Update Priority", key='priority_btn', use_container_width=True):
        ss.df_ops.loc[ss.df_ops['Job_ID'] == job_sel, 'Priority'] = new_priority
        ss.base_df_ops.loc[ss.base_df_ops['Job_ID'] == job_sel, 'Priority'] = new_priority
        safe_toast(f"✅ {job_sel} priority set to {new_priority}", icon="📊")
        st.rerun()

def draw_outsourcing_policy(ss):
    st.write("**🏭 Outsourcing Threshold**")
    threshold = st.slider(
        "Complexity threshold:",
        min_value=0.0,
        max_value=2.0,
        value=0.9,
        step=0.1,
        help="Ops with score > threshold get outsourced",
        key='outsource_threshold'
    )
    st.caption(f"Current: {threshold} — Higher = more outsourcing")

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
    
    # --- AI-POWERED PERFORMANCE INSIGHTS ---
    with st.expander("🤖 AI Performance Analysis & Recommendations", expanded=False):
        if st.checkbox("Generate AI Insights", key='kpi_ai_insights'):
            with st.spinner("🧠 Analyzing performance..."):
                context = f"""
                CURRENT SCHEDULE PERFORMANCE ({ss.current_heuristic}):
                - Makespan: {metrics['Makespan_Days']:.2f} days
                - Total Tardiness: {metrics['Total_Tardiness_Days']:.2f} days
                - On-Time Delivery: {metrics['On_Time_%']:.1f}%
                - Machine Utilization: {metrics['Machine_Utilization_%']:.1f}%
                - Total Cost: ${metrics['Total_Cost_$']:.2f}
                - Total Operations: {len(ss.df_ops)}
                - Machines: {len(ss.df_machines)}
                
                Analyze this CNC manufacturing schedule and provide:
                1. PERFORMANCE ASSESSMENT: Is this schedule optimal? (2-3 sentences)
                2. TOP 3 BOTTLENECKS: What's limiting performance?
                3. QUICK WINS: 2-3 actionable improvements
                4. RISK ALERT: Any concerning metrics that need immediate attention?
                
                Be specific, actionable, and focused on manufacturing operations.
                """
                
                insights = generate_ai_insights(
                    "Provide detailed performance analysis and recommendations.",
                    context_data=context
                )
                
                if insights:
                    st.markdown(insights)
                    st.caption("💡 Analysis powered by Google Gemini AI")
                else:
                    st.info("💡 AI insights require `google-generativeai` package and valid API key in .env file")


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

def draw_operation_status_tab(ss):
    st.header(f"📋 Operation Status ({ss.current_heuristic or 'N/A'})")
    with st.spinner("Generating operation status table..."):
        _cache_key = getattr(ss, "schedule_update_key", str(time.time()))
        status_table = create_operation_status_table(ss.current_schedule.copy() if not ss.current_schedule is None else pd.DataFrame(), ss.df_ops.copy(), _cache_key=_cache_key)

    st.info(f"🔄 Cache Key: {getattr(ss, 'schedule_update_key', 'N/A')}")
    st.dataframe(status_table, use_container_width=True, height=500)

def draw_comparison_tab(ss):
    
    # === AI & SETTINGS SECTION ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("🤖 AI-Powered Analysis", expanded=False):
            tab1, tab2 = st.tabs(["Dataset Quality", "Algorithm Insights"])
            
            with tab1:
                if st.checkbox("Analyze Dataset Quality", key='ai_data_quality'):
                    with st.spinner("🧠 Analyzing..."):
                        df_ops = ss.base_df_ops
                        context = f"""
                        DATASET: {len(df_ops)} ops, {df_ops['Job_ID'].nunique()} jobs, {len(ss.df_machines)} machines
                        Release: {df_ops['Release_Day'].min()}-{df_ops['Release_Day'].max()} days
                        Due: {df_ops['Due_Day'].min()}-{df_ops['Due_Day'].max()} days
                        Outsourced: {len(df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE'])}/{len(df_ops)} ({len(df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE'])/len(df_ops)*100:.1f}%)
                        Lead time avg: {(df_ops['Due_Day'] - df_ops['Release_Day']).mean():.1f} days
                        
                        Provide: 1) Quality score 1-10, 2) Critical issues, 3) Optimization tips
                        """
                        insights = generate_ai_insights("Analyze CNC dataset quality", context_data=context)
                        if insights:
                            st.markdown(insights)
                            st.caption("💡 Powered by Gemini AI")
                        else:
                            st.info("💡 Install `google-generativeai` and add API key to .env")
            
            with tab2:
                st.info("Run 'Compute All Algorithms' first, then AI analysis will appear in results below")
    
    with col2:
        with st.expander("⚖️ Priority Weights", expanded=False):
            w_makespan = st.slider("Makespan", 0.0, 5.0, 1.0, 0.1, key='w_makespan')
            w_tardiness = st.slider("Tardiness", 0.0, 5.0, 1.0, 0.1, key='w_tardiness')
            w_ontime = st.slider("On-Time %", 0.0, 5.0, 1.0, 0.1, key='w_ontime')
            w_util = st.slider("Utilization", 0.0, 5.0, 1.0, 0.1, key='w_util')
            w_cost = st.slider("Cost", 0.0, 5.0, 1.0, 0.1, key='w_cost')
            weights_sum = w_makespan + w_tardiness + w_ontime + w_util + w_cost

    st.divider()
    
    # === COMPARISON RESULTS ===

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

    # ✅ Display comparison results
    if hasattr(ss, 'df_metrics') and ss.df_metrics is not None and len(ss.df_metrics) > 0:
        try:
            # Calculate composite scores
            dfm = ss.df_metrics.copy()
            for col in ['Makespan_Days', 'Total_Tardiness_Days', 'On_Time_%', 'Machine_Utilization_%', 'Total_Cost_$']:
                if col not in dfm.columns:
                    dfm[col] = 0.0

            # Normalize metrics (0-1 scale, higher is better)
            def norm(s):
                mn, mx = s.min(), s.max()
                return pd.Series([0.5] * len(s), index=s.index) if mx == mn else (s - mn) / (mx - mn)

            n_makespan = 1.0 - norm(dfm['Makespan_Days'])
            n_tardiness = 1.0 - norm(dfm['Total_Tardiness_Days'])
            n_ontime = norm(dfm['On_Time_%'])
            n_util = norm(dfm['Machine_Utilization_%'])
            n_cost = 1.0 - norm(dfm['Total_Cost_$'])

            # Weighted composite score
            total_weight = max(1e-9, (w_makespan + w_tardiness + w_ontime + w_util + w_cost))
            dfm['Score'] = (w_makespan * n_makespan + w_tardiness * n_tardiness + 
                           w_ontime * n_ontime + w_util * n_util + w_cost * n_cost) / total_weight

            # Display ranked table
            dfm_display = dfm.sort_values('Score', ascending=False).reset_index(drop=True)
            st.dataframe(dfm_display.style.highlight_max(subset=['Score'], color='lightgreen'), use_container_width=True)

            # Recommendation
            best = dfm_display.iloc[0]
            st.success(f"🏆 **Recommended: {best['Heuristic']}** (Score: {best['Score']:.3f}) — Makespan: {best['Makespan_Days']:.1f}d, Tardiness: {best['Total_Tardiness_Days']:.1f}d, On-Time: {best['On_Time_%']:.1f}%")

            # AI Analysis (optional)
            with st.expander("🤖 Why This Algorithm?", expanded=False):
                if os.getenv('GEMINI_API_KEY') and st.checkbox("Enable AI Analysis", value=False, key='enable_gemini'):
                    with st.spinner("Generating insights..."):
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                            model = genai.GenerativeModel('gemini-pro')
                            prompt = f"""Explain why {best['Heuristic']} (score {best['Score']:.3f}) is best for this CNC scheduling scenario:
                            - Makespan: {best['Makespan_Days']:.1f}d, Tardiness: {best['Total_Tardiness_Days']:.1f}d, On-Time: {best['On_Time_%']:.1f}%
                            - User priorities: Makespan={w_makespan}, Tardiness={w_tardiness}, On-Time={w_ontime}, Util={w_util}, Cost={w_cost}
                            
                            Provide: 1) Why it's optimal, 2) Business impact, 3) Trade-offs vs 2nd best. Be concise (3-4 sentences)."""
                            st.markdown(model.generate_content(prompt).text)
                        except Exception as e:
                            st.warning(f"AI unavailable: {e}")
                else:
                    st.info("💡 **Quick Guide:**\n- **SPT**: Fast completion, good for throughput\n- **EDD**: Minimizes tardiness, best for deadlines\n- **CR**: Balances urgency and workload\n- **PRIORITY**: Follows business priorities")

        except Exception as e:
            st.error(f"Error computing recommendation: {e}")
            
    else:
        st.info("📊 No comparison data yet. Click '🧪 Compute All Heuristics' in the sidebar to generate comparison results.")

def handle_error(e):
    st.error(f"❌ ERROR: An error occurred during execution")
    st.error(f"Error Type: {type(e).__name__}")
    st.error(f"Error Message: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# ---------------------------
# Diagnostic utility: compute realistic utilization metrics
# ---------------------------
def display_utilization_diagnostics(ss):
    """
    Diagnostic panel showing why utilization is low and what realistic targets are.
    """
    try:
        st.markdown("### 🔍 Utilization Diagnostic Report")
        
        if not hasattr(ss, 'base_df_ops') or not hasattr(ss, 'base_df_machines'):
            st.warning("Data not loaded yet. Initialize the app first.")
            return
        
        df_ops = ss.base_df_ops.copy()
        df_machines = ss.base_df_machines.copy()
        
        # Calculate total work content (productive time)
        total_setup = df_ops['Setup_Time'].sum()
        total_proc = df_ops['Total_Proc_Min'].sum()
        total_transfer = df_ops.get('Transfer_Min', pd.Series([0]*len(df_ops))).sum()
        total_productive_min = total_setup + total_proc + total_transfer
        total_productive_days = total_productive_min / 480
        
        # Count operations by assignment
        inhouse_ops = len(df_ops[df_ops['Assignment_Type'] == 'IN_HOUSE'])
        outsourced_ops = len(df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE'])
        total_ops = len(df_ops)
        
        # Machine count
        num_machines = len(df_machines)
        
        # Calculate planning horizon
        min_release = df_ops['Release_Day'].min()
        max_due = df_ops['Due_Day'].max()
        horizon_days = max_due - min_release
        
        # Theoretical utilization (if work was packed perfectly)
        if horizon_days > 0 and num_machines > 0:
            theoretical_util = (total_productive_days / (num_machines * horizon_days)) * 100
        else:
            theoretical_util = 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Operations", total_ops)
        col1.metric("In-House Ops", inhouse_ops)
        col1.metric("Outsourced Ops", outsourced_ops)
        
        col2.metric("Total Work Content", f"{total_productive_days:.1f} days")
        col2.metric("Machine Count", num_machines)
        col2.metric("Planning Horizon", f"{horizon_days:.1f} days")
        
        col3.metric("Theoretical Max Utilization", f"{theoretical_util:.1f}%")
        col3.metric("Realistic Target", f"{theoretical_util * 0.7:.1f}%")
        
        st.divider()
        
        st.markdown("### 📋 Interpretation")
        
        if theoretical_util < 10:
            st.error(f"""
            ⚠️ **Very Low Utilization ({theoretical_util:.1f}%)**
            
            **Why this is happening:**
            - You have {num_machines} machines available for {horizon_days:.1f} days = {num_machines * horizon_days:.1f} machine-days of capacity.
            - But only {total_productive_days:.1f} days of actual work content.
            - Even if perfectly packed, machines would be idle {100-theoretical_util:.1f}% of the time.
            
            **Recommended fixes:**
            1. **Reduce machine count** to 2-3 machines (or use full dataset with more jobs).
            2. **Reduce outsourcing** (adjust threshold in sidebar to keep more work in-house).
            3. **Fix Release/Due dates** in your CSV (currently all Release_Day=33 but Due_Day=5-32).
            4. **Load full dataset** (change SAMPLE_SIZE to None in code).
            """)
        elif theoretical_util < 30:
            st.warning(f"""
            ⚠️ **Low Utilization ({theoretical_util:.1f}%)**
            
            Your current workload can't fully utilize {num_machines} machines over {horizon_days:.1f} days.
            
            **Options:**
            - Reduce machines to {int(num_machines * theoretical_util / 60) + 1}-{int(num_machines * theoretical_util / 50) + 1} to reach 50-60% utilization.
            - Add more jobs or reduce outsourcing to increase in-house work content.
            """)
        elif theoretical_util < 70:
            st.success(f"""
            ✅ **Reasonable Utilization Range ({theoretical_util:.1f}%)**
            
            Your workload is balanced for the number of machines. Target realistic utilization: {theoretical_util * 0.7:.1f}%.
            """)
        else:
            st.info(f"""
            📊 **High Capacity Utilization ({theoretical_util:.1f}%)**
            
            Machines will be busy most of the time. Monitor for:
            - Potential bottlenecks or delays.
            - Need for overtime or additional capacity.
            """)
        
        # Show breakdown by operation type
        st.markdown("### 📊 Work Breakdown by Operation Type")
        op_breakdown = df_ops.groupby('Op_Type').agg({
            'Operation_ID': 'count',
            'Total_Proc_Min': 'sum',
            'Setup_Time': 'sum'
        }).rename(columns={'Operation_ID': 'Count', 'Total_Proc_Min': 'Total_Proc_Min', 'Setup_Time': 'Total_Setup_Min'})
        op_breakdown['Total_Time_Days'] = (op_breakdown['Total_Proc_Min'] + op_breakdown['Total_Setup_Min']) / 480
        st.dataframe(op_breakdown, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error computing diagnostics: {e}")
        import traceback
        st.code(traceback.format_exc())

# ---------------------------
# MAIN
# ---------------------------
# Configuration: Set to None to load full dataset (200 jobs), or specify number of jobs to sample
SAMPLE_SIZE = None  # Load ALL 200 jobs for realistic utilization

if 'recalculate_all_heuristics' not in st.session_state:
    st.session_state.recalculate_all_heuristics = False
if 'force_metric_refresh' not in st.session_state:
    st.session_state.force_metric_refresh = False

def initialize_app(ss):
    dbg("🔧 DEBUG: System not initialized, starting initialization...")
    dbg("🔄 DEBUG: Clearing old cached data...")
    st.cache_data.clear()

    with st.spinner("Loading and preprocessing data..."):
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



    st.info("🔎 Raw dataset loaded. Click 'Compute All Heuristics' in the sidebar to run scheduling on current data.")
    safe_toast("System initialized (raw data loaded).", icon="✅")

def main():
    st.set_page_config(page_title="CNC Scheduling", page_icon="🏭", layout="wide")
    st.title("🏭 CNC Job Scheduling System")
    
    if 'cache_version' not in st.session_state or st.session_state.cache_version < 2:
        st.session_state.cache_version = 2

    ss = st.session_state
    if "current_page" not in ss:
        ss.current_page = "comparison"

    try:
        if 'initialized' not in ss:
            initialize_app(ss)
        if 'schedule_update_key' not in ss:
            ss.schedule_update_key = str(time.time())

        # ============ SIMPLIFIED SIDEBAR ============
        with st.sidebar:
            st.title("⚙️ Controls")
            
            # Page Navigation
            page = st.radio(
                "� Navigation",
                ["📊 Compare Algorithms", "🔍 View Schedule"],
                index=0 if ss.current_page == "comparison" else 1
            )
            ss.current_page = "comparison" if "Compare" in page else "heuristic_view"
            
            st.divider()
            
            # Section 1: Compute & Apply
            with st.expander("🎯 **1. Compute & Apply**", expanded=True):
                if st.button("🧪 Compute All Algorithms", use_container_width=True):
                    compute_all_heuristics_and_metrics(ss, show_progress=True)
                    st.rerun()
                
                heuristic_options = ('SPT', 'EDD', 'CR', 'PRIORITY')
                current_h = ss.get("current_heuristic")
                current_index = heuristic_options.index(current_h) if current_h in heuristic_options else 0
                
                apply_choice = st.selectbox(
                    "Select algorithm:", 
                    heuristic_options,
                    index=current_index,
                    help="Choose which algorithm to apply to your schedule"
                )
                
                if st.button("✅ Apply Algorithm", key='apply_heur', use_container_width=True):
                    ok = apply_heuristic_to_dataset(ss, apply_choice)
                    if ok:
                        st.rerun()
            
            # Section 2: Manage Jobs
            with st.expander("📋 **2. Manage Jobs**", expanded=False):
                draw_live_job_scheduler(ss)
                st.divider()
                draw_job_deleter(ss)
            
            # Section 3: Advanced Settings
            with st.expander("⚙️ **3. Advanced Settings**", expanded=False):
                draw_breakdown_simulator(ss)
                st.divider()
                draw_priority_manager(ss)
                st.divider()
                draw_outsourcing_policy(ss)
            
            # Section 4: System
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset", use_container_width=True):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    for key in list(ss.keys()):
                        del ss[key]
                    st.rerun()
            
            with col2:
                csv_data = export_schedule(ss.current_schedule)
                safe_name = (ss.current_heuristic or "none").lower()
                st.download_button(
                    label="💾 Export",
                    data=csv_data,
                    file_name=f"schedule_{safe_name}.csv",
                    mime='text/csv',
                    use_container_width=True
                )

        # ============ MAIN CONTENT ============
        if ss.current_page == "comparison":
            st.header("📊 Algorithm Comparison")
            draw_comparison_tab(ss)
        else:
            if ss.current_heuristic:
                st.header(f"📈 {ss.current_heuristic} Schedule")
                draw_kpi_dashboard(ss)
                
                tab1, tab2 = st.tabs(["📈 Gantt Chart", "📋 Operations"])
                with tab1:
                    draw_gantt_tab(ss)
                with tab2:
                    draw_operation_status_tab(ss)
            else:
                st.warning("⚠️ No algorithm applied. Go to 'Compare Algorithms' to select one.")

    except Exception as e:
        handle_error(e)

if __name__ == "__main__":
    main()
