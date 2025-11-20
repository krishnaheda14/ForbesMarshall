# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Import scheduling logic from existing file
import sys
sys.path.append('..')
from cnc_scheduler_core import (
    CNCScheduler,
    parse_maintenance,
    get_eligible_machines,
    calculate_inhouse_cost,
    make_or_buy_decision,
    get_setup_penalty,
    calculate_metrics,
    analyze_capacity_for_new_job
)

# Import new Excel ingestion modules
from excel_ingestion import ExcelIngestor, normalize_column_names
from schema_mapping import SchemaMapper
from data_transformer import DataTransformer
from models import CanonicalJob, ColumnMapping, MappingTemplate

# Load environment variables
load_dotenv()

app = FastAPI(title="CNC Scheduling API", version="2.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Configure Gemini AI
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        AI_ENABLED = True
    else:
        AI_ENABLED = False
except Exception as e:
    AI_ENABLED = False
    print(f"AI initialization failed: {e}")

# Global state management (in production, use Redis or database)
class AppState:
    def __init__(self):
        self.df_ops = None
        self.df_machines = None
        self.df_effective = None
        self.df_penalties = None
        self.df_vendors = None
        self.schedules = {}
        self.metrics = {}
        self.current_heuristic = None
        self.cost_threshold = 0.9
        self.activity_log = []

state = AppState()

# Pydantic models
class LoadDataRequest(BaseModel):
    sample_size: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "sample_size": None
            }
        }

class ComputeHeuristicRequest(BaseModel):
    heuristic: str

class ApplyHeuristicRequest(BaseModel):
    heuristic: str

class MachineBreakdownRequest(BaseModel):
    machine_id: str
    start_time: float
    duration: float

class PriorityUpdateRequest(BaseModel):
    job_id: str
    priority: int

class OutsourcingPolicyRequest(BaseModel):
    cost_threshold: float

class NewJobRequest(BaseModel):
    job_id: str
    quantity: int
    due_day: int
    operations: List[Dict[str, Any]]

class AIInsightRequest(BaseModel):
    prompt: str
    context_data: Optional[Dict[str, Any]] = None

# Helper functions
def load_all_data(sample_size=None):
    """Load and preprocess all data files"""
    # Get the parent directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    try:
        # Use jobs_dataset.csv for normal operations
        df_ops = pd.read_csv(os.path.join(data_dir, 'jobs_dataset.csv'))
        df_machines = pd.read_csv(os.path.join(data_dir, 'machine_data.csv'))
        df_vendors = pd.read_csv(os.path.join(data_dir, 'vendor_data.csv'))
        df_penalties = pd.read_csv(os.path.join(data_dir, 'previous_next_material.csv'))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {e}")

    # Data preprocessing (same as original)
    
    # Normalize column names - replace spaces with underscores
    df_machines.columns = df_machines.columns.str.replace(' ', '_')
    df_ops.columns = df_ops.columns.str.replace(' ', '_')
    df_vendors.columns = df_vendors.columns.str.replace(' ', '_')
    df_penalties.columns = df_penalties.columns.str.replace(' ', '_')
    
    if sample_size:
        unique_jobs = df_ops['Job_ID'].unique()[:sample_size]
        df_ops = df_ops[df_ops['Job_ID'].isin(unique_jobs)].copy()

    df_ops['Total_Proc_Min'] = df_ops['Proc_Time_per_Unit'] * df_ops['Quantity']

    # Normalize machines df
    if 'Speed_Factor' not in df_machines.columns and 'SpeedFactor' in df_machines.columns:
        df_machines.rename(columns={'SpeedFactor': 'Speed_Factor'}, inplace=True)

    df_machines['Speed_Factor'] = (
        df_machines['Speed_Factor']
        .astype(str)
        .str.extract(r'([0-9]*\.?[0-9]+)')
        .astype(float)
    )

    # Calculate effective times
    effective_times = []
    for idx, op in df_ops.iterrows():
        op_type = op['Op_Type']
        eligible_machines = get_eligible_machines(op_type)
        for machine_id in eligible_machines:
            machine_row = df_machines[df_machines['Machine_ID'] == machine_id]
            if len(machine_row) > 0:
                speed_factor = machine_row.iloc[0]['Speed_Factor']
                effective_proc_time = op['Total_Proc_Min'] / speed_factor
                total_time = op['Setup_Time'] + effective_proc_time + op.get('Transfer_Min', 0)
                effective_times.append({
                    'Operation_ID': op['Operation_ID'],
                    'Machine_ID': machine_id,
                    'Effective_Proc_Time': effective_proc_time,
                    'Total_Time': total_time
                })
    df_effective = pd.DataFrame(effective_times)

    # Process vendor data
    df_vendors['Outsource_Unit_Cost'] = df_vendors['Outsource_Unit_Cost'].replace('[\\$,]', '', regex=True).astype(float)
    df_vendors['Transport_Cost'] = df_vendors['Transport_Cost'].replace('[\\$,]', '', regex=True).astype(float)

    df_ops_vendor = df_ops.merge(
        df_vendors[['Vendor_ID', 'Outsource_Lead_Time_(Days)', 'Outsource_Unit_Cost', 'Transport_Cost', 'Quality_Factor']],
        left_on='Vendor_Ref', right_on='Vendor_ID', how='left'
    )

    df_ops_vendor['Outsource_Cost'] = (
        (df_ops_vendor['Outsource_Unit_Cost'] * df_ops_vendor['Quantity']) + df_ops_vendor['Transport_Cost']
    ) / df_ops_vendor['Quality_Factor']

    df_ops_vendor['Outsource_Time_Min'] = df_ops_vendor['Outsource_Lead_Time_(Days)'] * 8 * 60

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

    # Parse maintenance windows - column name already normalized to underscores
    maintenance_col = 'Scheduled_Maintenance_(Day,_Time-Time)'
    if maintenance_col in df_machines.columns:
        df_machines['Maintenance_Window'] = df_machines[maintenance_col].apply(parse_maintenance)
    else:
        df_machines['Maintenance_Window'] = None

    # Make-or-buy decisions
    decisions = []
    for idx, op in df_ops.iterrows():
        result = make_or_buy_decision(op, df_effective, state.cost_threshold)
        decisions.append({'Operation_ID': op['Operation_ID'], 'Decision': result[0] if result else 'IN_HOUSE'})

    df_decisions = pd.DataFrame(decisions)
    df_ops = df_ops.merge(df_decisions, on='Operation_ID', how='left')
    df_ops['Assignment_Type'] = df_ops['Decision'].fillna('IN_HOUSE')
    df_ops.drop(columns=['Decision'], inplace=True)

    return df_ops, df_machines, df_effective, df_penalties, df_vendors

def get_ai_insights(prompt: str, context_data: Optional[Dict] = None):
    """Generate AI insights using Gemini"""
    if not AI_ENABLED:
        return "AI insights are disabled. Please add GEMINI_API_KEY to your .env file."
    
    try:
        full_prompt = f"""
You are an expert in manufacturing scheduling, operations research, and production planning.

**CONTEXT:**
This is a CNC job scheduling application with 6 heuristics:
- SPT (Shortest Processing Time)
- EDD (Earliest Due Date)
- CR (Critical Ratio)
- PRIORITY (Job priority-based)
- WEIGHTED (Multi-objective)
- SLACK (Minimum slack time)

{prompt}
"""
        
        if context_data:
            full_prompt += f"\n\n**DATA CONTEXT:**\n{context_data}\n"
        
        full_prompt += "\n\nProvide clear, actionable insights in 3-5 concise bullet points."
        
        response = gemini_model.generate_content(full_prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

# API Endpoints

@app.get("/")
def read_root():
    return {"message": "CNC Scheduling API v2.0", "status": "running"}

@app.post("/api/data/load")
async def load_data(request: LoadDataRequest = Body(default=LoadDataRequest(sample_size=None))):
    """Load and initialize dataset"""
    try:
        sample_size = request.sample_size
        df_ops, df_machines, df_effective, df_penalties, df_vendors = load_all_data(sample_size)
        
        state.df_ops = df_ops
        state.df_machines = df_machines
        state.df_effective = df_effective
        state.df_penalties = df_penalties
        state.df_vendors = df_vendors
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Data Loaded',
            'details': f"Loaded {len(df_ops)} operations, {len(df_machines)} machines"
        })
        
        return {
            "status": "success",
            "message": "Data loaded successfully",
            "stats": {
                "total_operations": len(df_ops),
                "total_machines": len(df_machines),
                "total_jobs": df_ops['Job_ID'].nunique(),
                "outsourced_ops": len(df_ops[df_ops['Assignment_Type'] == 'OUTSOURCE']),
                "inhouse_ops": len(df_ops[df_ops['Assignment_Type'] == 'IN_HOUSE'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/info")
def get_data_info():
    """Get current dataset information"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /api/data/load first.")
    
    return {
        "operations": len(state.df_ops),
        "machines": len(state.df_machines),
        "jobs": state.df_ops['Job_ID'].nunique(),
        "current_heuristic": state.current_heuristic,
        "cost_threshold": state.cost_threshold
    }

@app.get("/api/data/machines")
def get_machine_data():
    """Get machine data including maintenance windows"""
    if state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Please load data first from the Dashboard.")
    
    try:
        machines = state.df_machines.to_dict('records')
        return {"machines": machines, "count": len(machines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching machine data: {str(e)}")

@app.post("/api/schedule/compute")
def compute_heuristic(request: ComputeHeuristicRequest):
    """Compute schedule for a specific heuristic"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    heuristic = request.heuristic.upper()
    if heuristic not in ['SPT', 'EDD', 'CR', 'PRIORITY', 'WEIGHTED', 'SLACK']:
        raise HTTPException(status_code=400, detail="Invalid heuristic")
    
    try:
        scheduler = CNCScheduler(
            state.df_ops.copy(),
            state.df_machines.copy(),
            state.df_effective.copy(),
            state.df_penalties.copy()
        )
        
        schedule = scheduler.run_scheduling(heuristic=heuristic, verbose=False)
        metrics = calculate_metrics(schedule, state.df_ops, heuristic)
        
        state.schedules[heuristic] = schedule
        state.metrics[heuristic] = metrics
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': f'Computed {heuristic}',
            'details': f"Scheduled {len(schedule)} operations"
        })
        
        return {
            "status": "success",
            "heuristic": heuristic,
            "schedule": schedule.to_dict('records'),
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/schedule/compute-all")
def compute_all_heuristics():
    """Compute schedules for all heuristics"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
    results = {}
    
    try:
        for heur in heuristics:
            scheduler = CNCScheduler(
                state.df_ops.copy(),
                state.df_machines.copy(),
                state.df_effective.copy(),
                state.df_penalties.copy()
            )
            
            schedule = scheduler.run_scheduling(heuristic=heur, verbose=False)
            metrics = calculate_metrics(schedule, state.df_ops, heur)
            
            state.schedules[heur] = schedule
            state.metrics[heur] = metrics
            
            results[heur] = {
                "schedule_size": len(schedule),
                "metrics": metrics
            }
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'All Heuristics Computed',
            'details': f"Computed: {', '.join(heuristics)}"
        })
        
        return {
            "status": "success",
            "results": results,
            "comparison": list(state.metrics.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/schedule/apply")
def apply_heuristic(request: ApplyHeuristicRequest):
    """Apply a computed heuristic to the dataset"""
    heuristic = request.heuristic.upper()
    
    if heuristic not in state.schedules:
        raise HTTPException(status_code=400, detail=f"Heuristic {heuristic} not computed yet")
    
    try:
        schedule_df = state.schedules[heuristic].copy()
        state.current_heuristic = heuristic
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': f'Applied {heuristic}',
            'details': f"Set as current heuristic"
        })
        
        return {
            "status": "success",
            "message": f"{heuristic} applied successfully",
            "schedule": schedule_df.to_dict('records'),
            "metrics": state.metrics.get(heuristic, {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/schedule/current")
def get_current_schedule():
    """Get the currently applied schedule"""
    if state.current_heuristic is None:
        raise HTTPException(status_code=400, detail="No heuristic applied yet")
    
    schedule = state.schedules.get(state.current_heuristic)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    return {
        "heuristic": state.current_heuristic,
        "schedule": schedule.to_dict('records'),
        "metrics": state.metrics.get(state.current_heuristic, {})
    }

@app.post("/api/machine/breakdown")
def simulate_breakdown(request: MachineBreakdownRequest):
    """Simulate machine breakdown"""
    if state.df_machines is None:
        raise HTTPException(
            status_code=400, 
            detail="Data not loaded. Please load dataset first by clicking 'Load Dataset' button on the Dashboard."
        )
    
    try:
        machine_idx = state.df_machines[state.df_machines['Machine_ID'] == request.machine_id].index
        if len(machine_idx) == 0:
            available_machines = state.df_machines['Machine_ID'].tolist()
            raise HTTPException(
                status_code=404, 
                detail=f"Machine '{request.machine_id}' not found. Available machines: {', '.join(available_machines[:10])}"
            )
        
        idx = machine_idx[0]
        existing_maint = state.df_machines.at[idx, 'Maintenance_Window']
        
        new_breakdown = {
            'start': request.start_time,
            'end': request.start_time + request.duration,
            'duration': request.duration
        }
        
        if existing_maint is None or (isinstance(existing_maint, dict) and not existing_maint):
            state.df_machines.at[idx, 'Maintenance_Window'] = new_breakdown
        else:
            if isinstance(existing_maint, dict):
                existing_maint = [existing_maint]
            existing_maint.append(new_breakdown)
            state.df_machines.at[idx, 'Maintenance_Window'] = existing_maint
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Machine Breakdown Simulated',
            'details': f"Machine: {request.machine_id}, Duration: {request.duration} min"
        })
        
        return {
            "status": "success",
            "message": "Breakdown simulated. Recompute heuristics to see impact.",
            "breakdown": new_breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/job/priority")
def update_priority(request: PriorityUpdateRequest):
    """Update job priority"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        job_mask = state.df_ops['Job_ID'] == request.job_id
        if not job_mask.any():
            raise HTTPException(status_code=404, detail="Job not found")
        
        state.df_ops.loc[job_mask, 'Priority'] = request.priority
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Priority Updated',
            'details': f"Job: {request.job_id}, New Priority: {request.priority}"
        })
        
        return {
            "status": "success",
            "message": f"Priority updated for {request.job_id}. Recompute heuristics to see impact."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/outsourcing/policy")
def update_outsourcing_policy(request: OutsourcingPolicyRequest):
    """Update outsourcing cost threshold and recompute metrics"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        old_threshold = state.cost_threshold
        state.cost_threshold = request.cost_threshold
        
        # Recalculate make-or-buy decisions
        decisions = []
        for idx, op in state.df_ops.iterrows():
            result = make_or_buy_decision(op, state.df_effective, state.cost_threshold)
            decisions.append({'Operation_ID': op['Operation_ID'], 'Decision': result[0] if result else 'IN_HOUSE'})
        
        df_decisions = pd.DataFrame(decisions)
        state.df_ops = state.df_ops.merge(df_decisions, on='Operation_ID', how='left', suffixes=('', '_new'))
        state.df_ops['Assignment_Type'] = state.df_ops['Decision_new'].fillna('IN_HOUSE')
        state.df_ops.drop(columns=['Decision_new'], inplace=True, errors='ignore')
        
        new_outsourced = len(state.df_ops[state.df_ops['Assignment_Type'] == 'OUTSOURCE'])
        
        # Recompute metrics for current heuristic if available
        if state.current_heuristic and state.schedules.get(state.current_heuristic):
            scheduler = CNCScheduler(
                state.df_jobs, state.df_ops, state.df_machines,
                state.df_prev_next, state.cost_threshold
            )
            schedule = state.schedules[state.current_heuristic]
            metrics = scheduler.compute_metrics(schedule)
            state.metrics[state.current_heuristic] = metrics
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Outsourcing Policy Updated',
            'details': f"Threshold: {old_threshold} → {request.cost_threshold}, Outsourced: {new_outsourced}"
        })
        
        return {
            "status": "success",
            "message": "Outsourcing policy updated and metrics recomputed",
            "new_outsourced_count": new_outsourced,
            "total_operations": len(state.df_ops),
            "metrics": state.metrics.get(state.current_heuristic, {}) if state.current_heuristic else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/insights")
def get_insights(request: AIInsightRequest):
    """Get AI-powered insights"""
    try:
        insights = get_ai_insights(request.prompt, request.context_data)
        return {
            "status": "success",
            "insights": insights,
            "ai_enabled": AI_ENABLED
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/activity-log")
def get_activity_log():
    """Get activity log"""
    return {
        "status": "success",
        "log": state.activity_log
    }

@app.get("/api/metrics/comparison")
def get_metrics_comparison():
    """Get comparison of all computed heuristics"""
    if not state.metrics:
        raise HTTPException(status_code=400, detail="No metrics available. Compute heuristics first.")
    
    return {
        "status": "success",
        "metrics": list(state.metrics.values())
    }

@app.post("/api/analysis/hourly-cost")
def analyze_hourly_cost(request: dict):
    """Analyze cost impact of different hourly rates with scheduling trade-offs"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        heuristic = request.get('heuristic', 'SPT')
        hourly_rates = request.get('hourly_rates', [20, 25, 30, 35, 40, 45, 50, 60, 70, 80])
        
        results = []
        
        for rate in hourly_rates:
            # Determine which operations to outsource
            ops_to_schedule = []
            outsourced_ops_list = []
            outsource_cost = 0
            
            for idx, op in state.df_ops.iterrows():
                op_inhouse_cost = (op['Total_Proc_Min'] / 60) * rate
                op_outsource_cost = op.get('Outsource_Cost', 0)
                
                # Outsource if vendor cost < 85% of in-house cost
                if op_outsource_cost > 0 and op_outsource_cost < (op_inhouse_cost * 0.85):
                    outsourced_ops_list.append(op)
                    outsource_cost += op_outsource_cost
                else:
                    ops_to_schedule.append(op)
            
            # Create temp dataframe for in-house operations
            temp_ops = pd.DataFrame(ops_to_schedule) if ops_to_schedule else state.df_ops.copy()
            
            # Run scheduling on in-house operations only
            if len(temp_ops) > 0:
                scheduler = CNCScheduler(
                    temp_ops,
                    state.df_machines.copy(),
                    state.df_effective.copy(),
                    state.df_penalties.copy()
                )
                schedule = scheduler.run_scheduling(heuristic=heuristic, verbose=False)
                
                # Calculate in-house cost from actual scheduled time
                inhouse_cost = (schedule['Proc_Time'].sum() / 60) * rate if not schedule.empty else 0
                
                # Calculate metrics
                metrics = calculate_metrics(schedule, temp_ops, heuristic)
                
                # Extract key scheduling metrics
                makespan_days = metrics.get('Makespan_Days', 0)
                tardiness_days = metrics.get('Total_Tardiness_Days', 0)
                utilization = metrics.get('Machine_Utilization_%', 0)
                on_time_pct = metrics.get('On_Time_%', 0)
                late_ops = metrics.get('Late_Operations', 0)
            else:
                inhouse_cost = 0
                makespan_days = 0
                tardiness_days = 0
                utilization = 0
                on_time_pct = 100
                late_ops = 0
            
            inhouse_ops = len(ops_to_schedule)
            outsourced_ops = len(outsourced_ops_list)
            outsourcing_pct = (outsourced_ops / len(state.df_ops)) * 100
            total_cost = inhouse_cost + outsource_cost
            
            results.append({
                'hourly_rate': rate,
                'inhouse_cost': round(inhouse_cost, 2),
                'outsource_cost': round(outsource_cost, 2),
                'total_cost': round(total_cost, 2),
                'outsourcing_pct': round(outsourcing_pct, 2),
                'inhouse_ops': inhouse_ops,
                'outsourced_ops': outsourced_ops,
                # Scheduling metrics showing trade-offs
                'makespan_days': round(makespan_days, 2),
                'tardiness_days': round(tardiness_days, 2),
                'utilization_pct': round(utilization, 2),
                'on_time_pct': round(on_time_pct, 2),
                'late_operations': late_ops
            })
        
        # Find key metrics
        lowest_cost = min(results, key=lambda x: x['total_cost'])
        max_outsource = max(results, key=lambda x: x['outsourcing_pct'])
        current_rate = next((r for r in results if r['hourly_rate'] == 30), results[0])
        best_on_time = max(results, key=lambda x: x['on_time_pct'])
        lowest_tardiness = min(results, key=lambda x: x['tardiness_days'])
        
        return {
            "status": "success",
            "heuristic": heuristic,
            "results": results,
            "lowest_cost_rate": lowest_cost['hourly_rate'],
            "lowest_cost": lowest_cost['total_cost'],
            "max_outsource_rate": max_outsource['hourly_rate'],
            "max_outsourcing": max_outsource['outsourcing_pct'],
            "current_cost": current_rate['total_cost'],
            "current_outsourcing": current_rate['outsourcing_pct'],
            "best_on_time_rate": best_on_time['hourly_rate'],
            "best_on_time_pct": best_on_time['on_time_pct'],
            "lowest_tardiness_rate": lowest_tardiness['hourly_rate'],
            "lowest_tardiness": lowest_tardiness['tardiness_days'],
            "total_operations": len(state.df_ops),
            "trade_off_insight": f"At ${lowest_cost['hourly_rate']}/hr (lowest cost), you'll have {lowest_cost['late_operations']} late operations with {lowest_cost['tardiness_days']:.1f} days total tardiness. At ${max_outsource['hourly_rate']}/hr, outsourcing reaches {max_outsource['outsourcing_pct']:.1f}% but may reduce tardiness to {max_outsource['tardiness_days']:.1f} days."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/job/{job_id}")
def delete_job(job_id: str):
    """Delete a job and all its operations"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        job_mask = state.df_ops['Job_ID'] == job_id
        if not job_mask.any():
            raise HTTPException(status_code=404, detail="Job not found")
        
        ops_count = job_mask.sum()
        state.df_ops = state.df_ops[~job_mask].reset_index(drop=True)
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Job Deleted',
            'details': f"Job: {job_id}, Removed {ops_count} operations"
        })
        
        return {
            "status": "success",
            "message": f"Deleted job {job_id} and {ops_count} operations"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXCEL INGESTION AND AUTOMATIC SCHEMA MAPPING ENDPOINTS
# ============================================================================

# Initialize Excel ingestor and schema mapper
excel_ingestor = ExcelIngestor()
schema_mapper = SchemaMapper(gemini_model if AI_ENABLED else None)

@app.post("/api/excel/upload")
async def upload_excel(file: UploadFile = File(...)):
    """
    Upload Excel file and get sheet information
    """
    try:
        result = await excel_ingestor.load_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/excel/parse")
async def parse_excel_sheet(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = None,
    sample_rows: int = 10
):
    """
    Parse specific sheet and extract column information
    """
    try:
        result = await excel_ingestor.parse_sheet(file, sheet_name, sample_rows)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/excel/auto-map")
async def auto_map_columns(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = None,
    use_llm: bool = True
):
    """
    Automatically map Excel columns to canonical schema
    Uses both heuristics and LLM reasoning
    """
    try:
        # Parse the sheet first
        parse_result = await excel_ingestor.parse_sheet(file, sheet_name, sample_rows=10)
        
        columns_info = parse_result['columns']
        sample_data = parse_result['sample_data']
        
        # Get automatic mappings
        mappings = schema_mapper.auto_map(columns_info, sample_data, use_llm=use_llm)
        
        # Format for frontend
        mapping_list = []
        for col_name, mapping_data in mappings.items():
            mapping_list.append({
                'excel_column': col_name,
                'canonical_field': mapping_data['canonical_field'],
                'confidence': mapping_data['confidence'],
                'source': mapping_data['source'],
                'reasoning': mapping_data.get('reasoning', ''),
                'available_fields': list(schema_mapper.CANONICAL_FIELDS.keys())
            })
        
        return {
            "status": "success",
            "mappings": mapping_list,
            "sheet_info": {
                "sheet_name": parse_result['sheet_name'],
                "row_count": parse_result['row_count'],
                "column_count": parse_result['column_count']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ConfirmMappingRequest(BaseModel):
    """Request body for confirming column mappings"""
    mappings: Dict[str, str]  # {excel_column: canonical_field}
    save_as_template: bool = False
    template_name: Optional[str] = None


@app.post("/api/excel/transform")
async def transform_excel_data(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    mappings: str = Form(...),
    save_as_template: bool = Form(False),
    template_name: Optional[str] = Form(None)
):
    """
    Transform Excel data to canonical jobs using confirmed mappings
    """
    try:
        # Parse mappings JSON string
        import json
        mappings_dict = json.loads(mappings)
        
        # Re-parse the file
        await excel_ingestor.parse_sheet(file, sheet_name)
        df = excel_ingestor.get_dataframe()
        
        # Transform data
        transformer = DataTransformer()
        jobs, errors, warnings = transformer.transform(df, mappings_dict)
        
        if len(errors) > 0 and len(jobs) == 0:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Data transformation failed",
                    "errors": errors,
                    "warnings": warnings
                }
            )
        
        # Convert to dict for JSON response
        jobs_data = [job.dict() for job in jobs]
        
        # TODO: Save template if requested
        
        return {
            "status": "success" if len(errors) == 0 else "partial_success",
            "jobs": jobs_data,
            "job_count": len(jobs),
            "errors": errors,
            "warnings": warnings,
            "message": f"Transformed {len(jobs)} jobs successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/excel/load-and-schedule")
async def load_excel_and_schedule(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    mappings: str = Form(...),
    heuristic: str = Form("EDD")
):
    """
    Complete workflow: Upload Excel -> Map columns -> Transform -> Schedule
    """
    try:
        import json
        mappings_dict = json.loads(mappings)
        
        # Parse sheet
        await excel_ingestor.parse_sheet(file, sheet_name)
        df = excel_ingestor.get_dataframe()
        
        # Transform to canonical jobs
        transformer = DataTransformer()
        jobs, errors, warnings = transformer.transform(df, mappings_dict)
        
        if len(jobs) == 0:
            raise HTTPException(
                status_code=400,
                detail={"message": "No valid jobs found", "errors": errors}
            )
        
        # Convert jobs to DataFrame format for scheduler
        jobs_data = []
        for job in jobs:
            # Handle due_date - convert datetime to days or use numeric value
            due_day = 999
            if job.due_date:
                if isinstance(job.due_date, (int, float)):
                    due_day = job.due_date
                elif hasattr(job.due_date, 'day'):
                    due_day = job.due_date.day
            
            # Handle priority - convert to numeric
            priority = 3  # default medium priority
            if job.priority_numeric:
                priority = job.priority_numeric
            elif job.priority:
                priority_map = {'HIGH': 1, 'A': 1, 'MEDIUM': 3, 'B': 3, 'LOW': 5, 'C': 5}
                priority = priority_map.get(str(job.priority).upper(), 3)
            
            jobs_data.append({
                'Job_ID': job.job_id,
                'Operation_ID': job.operation_id or f"{job.job_id}_Op1",
                'Op_Seq': 1,  # Single operation per job from Excel
                'Quantity': job.quantity or 1,
                'Proc_Time_per_Unit': job.processing_time,
                'Setup_Time': job.setup_time or 0,
                'Due_Day': due_day,
                'Priority': priority,
                'Mat_Type': job.material_type or 'STEEL',
                'Tool_Group': job.tool_group or 'TGA',
                'Op_Type': job.metadata.get('op_type', 'MILLING') if job.metadata else 'MILLING',
                'Part_Type': job.part_type or 'A',
                'Transfer_Min': 5,
                'Release_Day': 0,
                'Outsource_Flag': 'Y' if job.can_outsource else 'N',
                'Vendor_Ref': job.vendor_id or '',
            })
        
        # Load into scheduler state
        state.df_ops = pd.DataFrame(jobs_data)
        state.df_ops['Total_Proc_Min'] = state.df_ops['Proc_Time_per_Unit'] * state.df_ops['Quantity']
        
        # Load other required data if not already loaded
        if state.df_machines is None:
            # Load all supporting data files
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            
            state.df_machines = pd.read_csv(os.path.join(data_dir, 'machine_data.csv'))
            state.df_vendors = pd.read_csv(os.path.join(data_dir, 'vendor_data.csv'))
            state.df_penalties = pd.read_csv(os.path.join(data_dir, 'previous_next_material.csv'))
            
            # Normalize column names
            state.df_machines.columns = state.df_machines.columns.str.replace(' ', '_')
            state.df_vendors.columns = state.df_vendors.columns.str.replace(' ', '_')
            state.df_penalties.columns = state.df_penalties.columns.str.replace(' ', '_')
            
            # Process Speed_Factor
            if 'Speed_Factor' not in state.df_machines.columns and 'SpeedFactor' in state.df_machines.columns:
                state.df_machines.rename(columns={'SpeedFactor': 'Speed_Factor'}, inplace=True)
            
            state.df_machines['Speed_Factor'] = (
                state.df_machines['Speed_Factor']
                .astype(str)
                .str.extract(r'([0-9]*\.?[0-9]+)')
                .astype(float)
            )
            
            # Parse maintenance windows
            maintenance_col = 'Scheduled_Maintenance_(Day,_Time-Time)'
            if maintenance_col in state.df_machines.columns:
                state.df_machines['Maintenance_Window'] = state.df_machines[maintenance_col].apply(parse_maintenance)
        
        # Calculate effective times for Excel jobs
        effective_times = []
        for idx, op in state.df_ops.iterrows():
            op_type = op.get('Op_Type', 'MILLING')
            eligible_machines = get_eligible_machines(op_type)
            for machine_id in eligible_machines:
                machine_row = state.df_machines[state.df_machines['Machine_ID'] == machine_id]
                if len(machine_row) > 0:
                    speed_factor = machine_row.iloc[0]['Speed_Factor']
                    effective_proc_time = op['Total_Proc_Min'] / speed_factor
                    total_time = op['Setup_Time'] + effective_proc_time + op.get('Transfer_Min', 0)
                    effective_times.append({
                        'Operation_ID': op['Operation_ID'],
                        'Machine_ID': machine_id,
                        'Effective_Proc_Time': effective_proc_time,
                        'Total_Time': total_time
                    })
        state.df_effective = pd.DataFrame(effective_times)
        
        # Run scheduling with selected heuristic
        scheduler = CNCScheduler(
            df_ops=state.df_ops,
            df_machines=state.df_machines,
            df_effective=state.df_effective,
            df_penalties=state.df_penalties
        )
        
        schedule = scheduler.run_scheduling(heuristic=heuristic, verbose=False)
        
        # Calculate metrics
        metrics = calculate_metrics(schedule, state.df_machines)
        
        # Store in state
        state.schedules[heuristic] = schedule
        state.metrics[heuristic] = metrics
        state.current_heuristic = heuristic
        
        return {
            "status": "success",
            "message": f"Scheduled {len(jobs)} jobs using {heuristic}",
            "job_count": len(jobs),
            "heuristic": heuristic,
            "schedule": schedule.to_dict('records'),
            "metrics": metrics,
            "errors": errors,
            "warnings": warnings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
