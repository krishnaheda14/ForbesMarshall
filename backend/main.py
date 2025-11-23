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
import requests

# Import scheduling logic from existing file
import sys
# Ensure project root is on sys.path regardless of current working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from core.cp_sat_scheduler import solve_with_cpsat

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

# Configure AI - OpenRouter with fallback to Gemini
try:
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    gemini_model = None
    
    # Configure Gemini regardless (needed for SchemaMapper)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Prefer OpenRouter for AI insights, but use Gemini for schema mapping
    if OPENROUTER_API_KEY:
        AI_ENABLED = True
        AI_PROVIDER = 'openrouter'
        print(f"AI enabled with OpenRouter (insights) and Gemini (schema mapping: {'Yes' if gemini_model else 'No'})")
    elif GEMINI_API_KEY:
        AI_ENABLED = True
        AI_PROVIDER = 'gemini'
        print("AI enabled with Gemini (both insights and schema mapping)")
    else:
        AI_ENABLED = False
        AI_PROVIDER = None
        print("AI disabled - no API keys found")
except Exception as e:
    AI_ENABLED = False
    AI_PROVIDER = None
    gemini_model = None
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
    """Generate AI insights using OpenRouter or Gemini"""
    if not AI_ENABLED:
        return "AI insights are disabled. Please add OPENROUTER_API_KEY or GEMINI_API_KEY to your .env file."
    
    try:
        full_prompt = f"""
You are an expert in manufacturing scheduling, operations research, and production planning.

**CONTEXT:**
This is a CNC job scheduling application with 6 heuristics:
- SPT (Shortest Processing Time): Schedules shortest jobs first
- EDD (Earliest Due Date): Prioritizes jobs with nearest deadlines
- CR (Critical Ratio): Uses due_date/processing_time ratio
- PRIORITY (Job priority-based): Schedules based on job priority values (1=High, 2=Medium, 3=Low)

{prompt}
"""
        
        if context_data:
            full_prompt += f"\n\n**DATA CONTEXT:**\n{context_data}\n"
        
        full_prompt += """\n\n**OUTPUT REQUIREMENTS:**
Provide 3-5 direct, actionable insights as bullet points.
- Start immediately with insights (no preamble like "Based on the data" or "Here are the insights")
- Use proper spacing between bullets for readability
- Use clear punctuation and complete sentences
- Focus on specific metrics and values (e.g., "Total Cost: $X" not "cost parameter")
- Be concise and professional
- Do not include any meta-commentary about the format"""
        
        # Use OpenRouter if available (better models)
        if AI_PROVIDER == 'openrouter':
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "anthropic/claude-3.5-sonnet",  # Claude 3.5 Sonnet (paid but working)
                    "messages": [
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                # Fallback to Gemini if OpenRouter fails
                if GEMINI_API_KEY and gemini_model:
                    try:
                        print("[AI Insights] OpenRouter failed, trying Gemini fallback...")
                        response = gemini_model.generate_content(full_prompt)
                        return response.text
                    except Exception as gemini_error:
                        return f"Both OpenRouter and Gemini failed. OpenRouter: {response.status_code} - {response.text[:100]}. Gemini: {str(gemini_error)[:100]}"
                else:
                    return f"OpenRouter API error: {response.status_code} - {response.text}"
        
        # Use Gemini as primary
        else:
            if not gemini_model:
                return "AI insights unavailable: No valid API keys configured"
            response = gemini_model.generate_content(full_prompt)
            return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            return "⚠️ Gemini API key is invalid or expired. Please update GEMINI_API_KEY in .env file or use OpenRouter instead."
        return f"Error generating AI insights: {error_msg}"

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
        
        # Debug: report loaded dataframe shapes
        print(f"[load_all_data] df_ops shape: {getattr(df_ops, 'shape', None)}")
        print(f"[load_all_data] df_machines shape: {getattr(df_machines, 'shape', None)}")
        print(f"[load_all_data] df_effective shape: {getattr(df_effective, 'shape', None)}")
        print(f"[load_all_data] df_penalties shape: {getattr(df_penalties, 'shape', None)}")
        print(f"[load_all_data] df_vendors shape: {getattr(df_vendors, 'shape', None)}")

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
        # Return empty data instead of error to allow graceful handling
        return {"machines": [], "count": 0, "message": "No data loaded yet"}
    
    try:
        # Convert DataFrame to dict, handling NaN values
        machines = state.df_machines.fillna('').to_dict('records')
        
        # Clean up any remaining NaN or None values
        cleaned_machines = []
        for machine in machines:
            cleaned_machine = {}
            for key, value in machine.items():
                if pd.isna(value) or value == 'nan':
                    cleaned_machine[key] = None
                else:
                    cleaned_machine[key] = value
            cleaned_machines.append(cleaned_machine)
        
        return {"machines": cleaned_machines, "count": len(cleaned_machines)}
    except Exception as e:
        print(f"Error in get_machine_data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching machine data: {str(e)}")

@app.post("/api/schedule/compute")
def compute_heuristic(request: ComputeHeuristicRequest):
    """Compute schedule for a specific heuristic"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    heuristic = request.heuristic.upper()
    if heuristic not in ['SPT', 'EDD', 'CR', 'PRIORITY']:
        raise HTTPException(status_code=400, detail="Invalid heuristic")
    
    try:
        scheduler = CNCScheduler(
            state.df_ops.copy(),
            state.df_machines.copy(),
            state.df_effective.copy(),
            state.df_penalties.copy()
        )
        
        schedule = scheduler.run_scheduling(heuristic=heuristic, verbose=False)
        # Merge in Priority and Assignment_Type from ops so frontend can render them
        try:
            schedule_df = schedule.merge(
                state.df_ops[['Operation_ID', 'Priority', 'Assignment_Type']],
                on='Operation_ID', how='left'
            )
            schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
            schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
        except Exception:
            schedule_df = schedule.copy()

        metrics = calculate_metrics(schedule_df, state.df_ops, heuristic)

        state.schedules[heuristic] = schedule_df
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
            try:
                schedule_df = schedule.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Assignment_Type']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            except Exception:
                schedule_df = schedule.copy()

            metrics = calculate_metrics(schedule_df, state.df_ops, heur)

            state.schedules[heur] = schedule_df
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

class CPSATRequest(BaseModel):
    objective_mode: str = 'min_weighted'
    alpha: float = 0.1
    time_limit_seconds: int = 30
    log: bool = False

@app.post("/api/schedule/cpsat")
def run_cpsat(request: CPSATRequest):
    """Run CP-SAT optimization to obtain an improved/optimal schedule.

    Returns schedule plus metrics comparable to heuristic results.
    """
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    try:
        # Solve with CP-SAT
        result = solve_with_cpsat(
            state.df_ops.copy(),
            state.df_machines.copy(),
            state.df_effective.copy(),
            state.df_penalties.copy(),
            objective_mode=request.objective_mode,
            alpha=request.alpha,
            time_limit_seconds=request.time_limit_seconds,
            log=request.log
        )

        schedule_df = pd.DataFrame(result.schedule)
        # Merge Priority & Assignment_Type into CPSAT schedule if available
        try:
            schedule_df = schedule_df.merge(
                state.df_ops[['Operation_ID', 'Priority', 'Assignment_Type']],
                on='Operation_ID', how='left'
            )
            schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
            schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
        except Exception:
            pass

        # Compute metrics (reuse calculate_metrics but heuristic label = CPSAT)
        metrics = calculate_metrics(schedule_df, state.df_ops, 'CPSAT')

        # Add solver stats & objective
        metrics['objective_mode'] = request.objective_mode
        metrics['solver_status'] = result.status
        metrics['objective_value'] = result.objective_value
        metrics['solver_conflicts'] = result.solver_stats.get('conflicts')
        metrics['solver_branches'] = result.solver_stats.get('branches')
        metrics['solver_wall_time'] = result.solver_stats.get('wall_time')

        # Store schedule similar to heuristics
        state.schedules['CPSAT'] = schedule_df
        state.metrics['CPSAT'] = metrics

        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Computed CPSAT',
            'details': f"Status={result.status} Obj={result.objective_value} Mode={request.objective_mode}"
        })

        return {
            'status': 'success',
            'schedule': schedule_df.to_dict('records'),
            'metrics': metrics
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/schedule/apply")
def apply_heuristic(request: ApplyHeuristicRequest):
    """Apply a computed heuristic to the dataset"""
    heuristic = request.heuristic.upper()
    
    if heuristic not in state.schedules:
        raise HTTPException(status_code=400, detail=f"Heuristic {heuristic} not computed yet")
    
    try:
        schedule_df = state.schedules[heuristic].copy()
        # Ensure schedule contains Priority and Assignment_Type
        if 'Priority' not in schedule_df.columns or 'Assignment_Type' not in schedule_df.columns:
            try:
                schedule_df = schedule_df.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Assignment_Type']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            except Exception:
                pass
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
    
    try:
        schedule_df = schedule.copy()
        if 'Priority' not in schedule_df.columns or 'Assignment_Type' not in schedule_df.columns:
            try:
                schedule_df = schedule_df.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Assignment_Type']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            except Exception:
                pass

        return {
            "heuristic": state.current_heuristic,
            "schedule": schedule_df.to_dict('records'),
            "metrics": state.metrics.get(state.current_heuristic, {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Clear all existing schedules since breakdown invalidates them
        state.schedules = {}
        state.metrics = {}
        state.current_heuristic = None
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Machine Breakdown Simulated',
            'details': f"Machine: {request.machine_id}, Duration: {request.duration} min"
        })
        
        return {
            "status": "success",
            "message": f"Breakdown simulated on {request.machine_id}. All schedules cleared - please recompute heuristics to see rescheduled operations.",
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
        # Normalize incoming job id and perform string-based match to avoid dtype/casing issues
        incoming_job = str(request.job_id).strip()
        df_job_ids = state.df_ops['Job_ID'].astype(str).str.strip()
        job_mask = df_job_ids == incoming_job
        if not job_mask.any():
            # Provide a helpful debug message listing some available Job_IDs
            sample_ids = state.df_ops['Job_ID'].astype(str).unique()[:10].tolist()
            detail = f"Job '{incoming_job}' not found. Sample Job_IDs: {sample_ids}"
            raise HTTPException(status_code=404, detail=detail)

        # Cast priority safely
        try:
            new_prio = int(request.priority)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid priority value: {request.priority}")

        state.df_ops.loc[job_mask, 'Priority'] = new_prio
        # If a baseline copy exists, update it as well so subsequent computations use the new priority
        if hasattr(state, 'base_df_ops') and state.base_df_ops is not None:
            try:
                base_job_ids = state.base_df_ops['Job_ID'].astype(str).str.strip()
                base_mask = base_job_ids == incoming_job
                if base_mask.any():
                    state.base_df_ops.loc[base_mask, 'Priority'] = new_prio
            except Exception:
                # Non-critical: continue even if baseline update fails
                pass
        
        # Clear cached schedules to force recomputation
        state.schedules = {}
        state.metrics = {}
        state.current_heuristic = None
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Priority Updated',
            'details': f"Job: {request.job_id}, New Priority: {request.priority}"
        })
        
        return {
            "status": "success",
            "message": f"Priority updated for {request.job_id}. Schedules cleared - recompute heuristics to see impact."
        }
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes/details
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Log the traceback to the server console for diagnosis
        print(f"[update_priority] Unexpected error updating priority for {request.job_id}: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Internal server error updating priority for {request.job_id}: {str(e)}")

@app.post("/api/outsourcing/policy")
def update_outsourcing_policy(request: OutsourcingPolicyRequest):
    """Update outsourcing cost threshold and recompute all heuristics"""
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
        merged = state.df_ops.merge(df_decisions, on='Operation_ID', how='left', suffixes=('', '_new'))
        # The merge may create either 'Decision' or 'Decision_new' depending on existing columns.
        if 'Decision_new' in merged.columns:
            merged['Assignment_Type'] = merged['Decision_new'].fillna('IN_HOUSE')
            merged.drop(columns=['Decision_new'], inplace=True, errors='ignore')
        elif 'Decision' in merged.columns:
            merged['Assignment_Type'] = merged['Decision'].fillna('IN_HOUSE')
            merged.drop(columns=['Decision'], inplace=True, errors='ignore')
        else:
            # Fallback: assume IN_HOUSE if nothing available
            merged['Assignment_Type'] = 'IN_HOUSE'

        state.df_ops = merged
        
        new_outsourced = len(state.df_ops[state.df_ops['Assignment_Type'] == 'OUTSOURCE'])
        
        # Recompute ALL heuristics that have been computed to update KPIs
        heuristics_to_recompute = list(state.schedules.keys())
        recomputed_metrics = {}
        
        for heur in heuristics_to_recompute:
            scheduler = CNCScheduler(
                state.df_ops.copy(),
                state.df_machines.copy(),
                state.df_effective.copy(),
                state.df_penalties.copy()
            )

            schedule = scheduler.run_scheduling(heuristic=heur, verbose=False)
            try:
                schedule_df = schedule.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Assignment_Type']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            except Exception:
                schedule_df = schedule.copy()

            metrics = calculate_metrics(schedule_df, state.df_ops, heur)

            state.schedules[heur] = schedule_df
            state.metrics[heur] = metrics
            recomputed_metrics[heur] = metrics
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Outsourcing Policy Updated',
            'details': f"Threshold: {old_threshold} → {request.cost_threshold}, Outsourced: {new_outsourced}, Recomputed: {', '.join(heuristics_to_recompute)}"
        })
        
        return {
            "status": "success",
            "message": f"Outsourcing policy updated. {len(heuristics_to_recompute)} heuristic(s) recomputed with new KPIs.",
            "new_outsourced_count": new_outsourced,
            "total_operations": len(state.df_ops),
            "heuristics_recomputed": heuristics_to_recompute,
            "metrics": recomputed_metrics
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
            outsourcing_details = []
            
            for idx, op in state.df_ops.iterrows():
                op_inhouse_cost = (op['Total_Proc_Min'] / 60) * rate
                op_outsource_cost = op.get('Outsource_Cost', 0)
                
                # Outsource if vendor cost < 85% of in-house cost
                if op_outsource_cost > 0 and op_outsource_cost < (op_inhouse_cost * 0.85):
                    outsourced_ops_list.append(op)
                    outsource_cost += op_outsource_cost
                    
                    # Track outsourcing details
                    savings = op_inhouse_cost - op_outsource_cost
                    savings_pct = (savings / op_inhouse_cost) * 100 if op_inhouse_cost > 0 else 0
                    
                    outsourcing_details.append({
                        'job_id': op['Job_ID'],
                        'operation_id': op['Operation_ID'],
                        'operation_type': op['Operation_Type'],
                        'proc_time_min': op['Total_Proc_Min'],
                        'inhouse_cost': round(op_inhouse_cost, 2),
                        'outsource_cost': round(op_outsource_cost, 2),
                        'savings': round(savings, 2),
                        'savings_pct': round(savings_pct, 2),
                        'vendor_ref': op.get('Vendor_Ref', 'N/A'),
                        'quantity': op.get('Quantity', 1)
                    })
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
                'late_operations': late_ops,
                # Detailed outsourcing breakdown
                'outsourcing_details': outsourcing_details
            })
        
        # Find key metrics
        lowest_cost = min(results, key=lambda x: x['total_cost'])
        max_outsource = max(results, key=lambda x: x['outsourcing_pct'])
        current_rate = next((r for r in results if r['hourly_rate'] == 30), results[0])
        best_on_time = max(results, key=lambda x: x['on_time_pct'])
        lowest_tardiness = min(results, key=lambda x: x['tardiness_days'])
        
        # Analyze outsourcing patterns across all rates
        all_outsourced = []
        for result in results:
            all_outsourced.extend(result['outsourcing_details'])
        
        # Find most frequently outsourced jobs and operation types
        from collections import Counter
        if all_outsourced:
            job_outsource_freq = Counter(item['job_id'] for item in all_outsourced)
            op_type_outsource_freq = Counter(item['operation_type'] for item in all_outsourced)
            
            # Get jobs/ops that are outsourced most frequently
            most_outsourced_jobs = [{'job_id': job, 'frequency': count} 
                                   for job, count in job_outsource_freq.most_common(10)]
            most_outsourced_op_types = [{'operation_type': op_type, 'frequency': count} 
                                        for op_type, count in op_type_outsource_freq.most_common()]
            
            # Calculate average savings per operation type
            op_type_savings = {}
            for item in all_outsourced:
                op_type = item['operation_type']
                if op_type not in op_type_savings:
                    op_type_savings[op_type] = {'total_savings': 0, 'count': 0}
                op_type_savings[op_type]['total_savings'] += item['savings']
                op_type_savings[op_type]['count'] += 1
            
            avg_savings_by_type = [
                {
                    'operation_type': op_type,
                    'avg_savings': round(data['total_savings'] / data['count'], 2),
                    'total_savings': round(data['total_savings'], 2),
                    'count': data['count']
                }
                for op_type, data in op_type_savings.items()
            ]
            avg_savings_by_type.sort(key=lambda x: x['avg_savings'], reverse=True)
            
            # Identify root causes
            root_causes = []
            if avg_savings_by_type:
                top_savings_type = avg_savings_by_type[0]
                root_causes.append(f"{top_savings_type['operation_type']} operations show highest vendor cost advantage (avg ${top_savings_type['avg_savings']} savings per operation)")
            
            high_proc_time_ops = [item for item in all_outsourced if item['proc_time_min'] > 120]
            if high_proc_time_ops:
                root_causes.append(f"{len(high_proc_time_ops)} outsourced operations have processing times >2 hours - vendors likely have specialized equipment")
            
            if most_outsourced_jobs:
                root_causes.append(f"Jobs {', '.join(job['job_id'] for job in most_outsourced_jobs[:3])} are outsourced most frequently - may indicate capacity constraints or missing in-house capabilities")
        else:
            most_outsourced_jobs = []
            most_outsourced_op_types = []
            avg_savings_by_type = []
            root_causes = ["No operations outsourced at any hourly rate - vendor costs exceed in-house costs across the board"]
        
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
            "trade_off_insight": f"At ${lowest_cost['hourly_rate']}/hr (lowest cost), you'll have {lowest_cost['late_operations']} late operations with {lowest_cost['tardiness_days']:.1f} days total tardiness. At ${max_outsource['hourly_rate']}/hr, outsourcing reaches {max_outsource['outsourcing_pct']:.1f}% but may reduce tardiness to {max_outsource['tardiness_days']:.1f} days.",
            # Outsourcing analytics
            "outsourcing_analytics": {
                "most_outsourced_jobs": most_outsourced_jobs,
                "most_outsourced_operation_types": most_outsourced_op_types,
                "avg_savings_by_operation_type": avg_savings_by_type,
                "root_causes": root_causes,
                "total_unique_outsourced_operations": len(set(item['operation_id'] for item in all_outsourced)) if all_outsourced else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/unload")
def unload_data():
    """Unload all dataset and reset state"""
    try:
        state.df_ops = None
        state.df_machines = None
        state.df_effective = None
        state.df_penalties = None
        state.df_vendors = None
        state.current_heuristic = None
        state.cost_threshold = 15000
        state.activity_log = []
        
        return {
            "status": "success",
            "message": "Dataset unloaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/job/add")
def add_job(request: dict):
    """
    Add a new job with operations
    
    Important: Operations are executed in sequence order (Op_Seq).
    The scheduler ensures that operation N+1 only starts after operation N completes.
    For example: Turning (Op_Seq=1) -> Milling (Op_Seq=2) -> Drilling (Op_Seq=3)
    """
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        job_id = request.get('job_id')
        operations = request.get('operations', [])
        
        if not job_id:
            raise HTTPException(status_code=400, detail="Job ID is required")
        
        if not operations:
            raise HTTPException(status_code=400, detail="At least one operation is required")
        
        # Check if job already exists
        if job_id in state.df_ops['Job_ID'].values:
            raise HTTPException(status_code=400, detail=f"Job {job_id} already exists")
        
        # Create new operations with proper sequence handling
        new_ops = []
        for i, op in enumerate(operations):
            op_id = f"{job_id}_OP{i+1}"
            # Convert priority string to numeric
            priority_str = op.get('priority', 'Medium')
            # Normalize to 1=High, 2=Medium, 3=Low
            priority_map = {'High': 1, 'Medium': 2, 'Low': 3, 1: 1, 2: 2, 3: 3}
            priority_num = priority_map.get(priority_str, 2)
            
            new_op = {
                'Job_ID': job_id,
                'Operation_ID': op_id,
                'Op_Seq': i + 1,  # Critical: Operations execute in this order
                'Operation_Type': op.get('operation_type', 'Drilling'),
                'Total_Proc_Min': op.get('proc_time', 60),
                'Setup_Time': op.get('setup_time', 10),
                'Transfer_Min': op.get('transfer_time', 5),
                'Quantity': op.get('quantity', 1),
                'Release_Day': op.get('release_day', 0),
                'Due_Day': op.get('due_day', 10),
                'Priority': priority_num,
                'Vendor_Ref': op.get('vendor_ref', 'V1'),
                'Outsource_Cost': op.get('outsource_cost', 0),
                'Outsource_Time_Min': op.get('outsource_time', 0),
                'Release_Time_Min': op.get('release_day', 0) * 480,
                'Due_Time_Min': op.get('due_day', 10) * 480,
                'Completion_Day': 0,
            }
            new_ops.append(new_op)
        
        # Add to dataframe
        new_df = pd.DataFrame(new_ops)
        state.df_ops = pd.concat([state.df_ops, new_df], ignore_index=True)
        
        # Update effective times for new operations
        new_effective = []
        for op in new_ops:
            op_type = op['Operation_Type']
            eligible_machines = get_eligible_machines(op_type)
            for machine_id in eligible_machines:
                machine_row = state.df_machines[state.df_machines['Machine_ID'] == machine_id]
                if len(machine_row) > 0:
                    speed_factor = machine_row.iloc[0]['Speed_Factor']
                    effective_proc_time = op['Total_Proc_Min'] / speed_factor
                    total_time = op['Setup_Time'] + effective_proc_time + op['Transfer_Min']
                    new_effective.append({
                        'Operation_ID': op['Operation_ID'],
                        'Machine_ID': machine_id,
                        'Effective_Proc_Time': effective_proc_time,
                        'Total_Time': total_time
                    })
        
        if new_effective:
            new_eff_df = pd.DataFrame(new_effective)
            state.df_effective = pd.concat([state.df_effective, new_eff_df], ignore_index=True)
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Job Added',
            'details': f"Job: {job_id}, Added {len(new_ops)} operations"
        })
        
        return {
            "status": "success",
            "message": f"Added job {job_id} with {len(new_ops)} operations",
            "operations_added": len(new_ops)
        }
    except HTTPException:
        raise
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
# Use OpenRouter (same as AI insights) if available, otherwise Gemini
schema_mapper = SchemaMapper(
    gemini_model=gemini_model if AI_ENABLED else None,
    openrouter_api_key=OPENROUTER_API_KEY if AI_PROVIDER == 'openrouter' else None,
    use_openrouter=(AI_PROVIDER == 'openrouter')
)

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
    ISOLATED from main dashboard - does NOT modify global state
    """
    try:
        import json
        mappings_dict = json.loads(mappings)
        
        print(f"[Excel Schedule] Starting with heuristic: {heuristic}")
        print(f"[Excel Schedule] Mappings: {mappings_dict}")
        
        # Parse sheet
        await excel_ingestor.parse_sheet(file, sheet_name)
        df = excel_ingestor.get_dataframe()
        print(f"[Excel Schedule] Parsed {len(df)} rows from Excel")
        
        # Transform to canonical jobs
        transformer = DataTransformer()
        jobs, errors, warnings = transformer.transform(df, mappings_dict)
        print(f"[Excel Schedule] Transformed to {len(jobs)} jobs, {len(errors)} errors, {len(warnings)} warnings")
        
        if len(jobs) == 0:
            raise HTTPException(
                status_code=400,
                detail={"message": "No valid jobs found", "errors": errors}
            )
        
        # Convert jobs to DataFrame format for scheduler (TEMPORARY - NOT GLOBAL STATE)
        jobs_data = []
        
        # Find earliest date as reference (time zero)
        all_dates = []
        for job in jobs:
            if job.release_date and hasattr(job.release_date, 'date'):
                all_dates.append(job.release_date)
            if job.due_date and hasattr(job.due_date, 'date'):
                all_dates.append(job.due_date)
        
        reference_date = min(all_dates).date() if all_dates else None
        print(f"[Excel Schedule] Reference date (time zero): {reference_date}")
        
        for job in jobs:
            # Handle due_date - convert datetime to days from reference
            due_day = 30  # default 30 days if no due date
            if job.due_date:
                if isinstance(job.due_date, (int, float)):
                    due_day = job.due_date
                elif hasattr(job.due_date, 'date'):
                    if reference_date:
                        due_day = (job.due_date.date() - reference_date).days
                    else:
                        due_day = 30
            
            # Handle release_date - convert datetime to days from reference
            release_day = 0
            if job.release_date:
                if isinstance(job.release_date, (int, float)):
                    release_day = job.release_date
                elif hasattr(job.release_date, 'date'):
                    if reference_date:
                        release_day = (job.release_date.date() - reference_date).days
                    else:
                        release_day = 0
            
            # Handle priority - convert to numeric (1=High,2=Medium,3=Low)
            priority = 2  # default medium priority
            if job.priority_numeric:
                priority = int(job.priority_numeric)
            elif job.priority:
                priority_map = {'HIGH': 1, 'A': 1, 'MEDIUM': 2, 'B': 2, 'LOW': 3, 'C': 3}
                priority = priority_map.get(str(job.priority).upper(), 2)
            
            jobs_data.append({
                'Job_ID': job.job_id,
                'Operation_ID': job.operation_id or f"{job.job_id}_Op1",
                'Op_Seq': 1,  # Single operation per job from Excel
                'Quantity': job.quantity or 1,
                'Proc_Time_per_Unit': job.processing_time,
                'Setup_Time': job.setup_time or 0,
                'Due_Day': due_day,
                'Due_Time_Min': due_day * 480,  # Convert days to minutes
                'Priority': priority,
                'Mat_Type': job.material_type or 'STEEL',
                'Tool_Group': job.tool_group or 'TGA',
                'Op_Type': job.metadata.get('op_type', 'MILLING') if job.metadata else 'MILLING',
                'Part_Type': job.part_type or 'A',
                'Transfer_Min': 5,
                'Release_Day': release_day,
                'Release_Time_Min': release_day * 480,  # Convert days to minutes (8-hour workday)
                'Outsource_Flag': 'Y' if job.can_outsource else 'N',
                'Vendor_Ref': job.vendor_id or '',
            })
        
        # Load into scheduler state (TEMPORARY COPIES - DO NOT MODIFY GLOBAL STATE)
        excel_df_ops = pd.DataFrame(jobs_data)
        excel_df_ops['Total_Proc_Min'] = excel_df_ops['Proc_Time_per_Unit'] * excel_df_ops['Quantity']
        print(f"[Excel Schedule] Created operations dataframe with {len(excel_df_ops)} rows")
        
        # Load supporting data files (TEMPORARY - DO NOT MODIFY GLOBAL STATE)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        
        print(f"[Excel Schedule] Loading data from {data_dir}")
        
        excel_df_machines = pd.read_csv(os.path.join(data_dir, 'machine_data.csv'))
        excel_df_vendors = pd.read_csv(os.path.join(data_dir, 'vendor_data.csv'))
        excel_df_penalties = pd.read_csv(os.path.join(data_dir, 'previous_next_material.csv'))
        
        # Normalize column names
        excel_df_machines.columns = excel_df_machines.columns.str.replace(' ', '_')
        excel_df_vendors.columns = excel_df_vendors.columns.str.replace(' ', '_')
        excel_df_penalties.columns = excel_df_penalties.columns.str.replace(' ', '_')
        
        # Process Speed_Factor safely
        if 'Speed_Factor' not in excel_df_machines.columns and 'SpeedFactor' in excel_df_machines.columns:
            excel_df_machines.rename(columns={'SpeedFactor': 'Speed_Factor'}, inplace=True)
        
        if 'Speed_Factor' in excel_df_machines.columns:
            excel_df_machines['Speed_Factor'] = (
                excel_df_machines['Speed_Factor']
                .astype(str)
                .str.extract(r'([0-9]*\.?[0-9]+)', expand=False)
                .fillna('1.0')
                .astype(float)
            )
        else:
            # Default speed factor if column doesn't exist
            excel_df_machines['Speed_Factor'] = 1.0
        
        # Parse maintenance windows
        maintenance_col = 'Scheduled_Maintenance_(Day,_Time-Time)'
        if maintenance_col in excel_df_machines.columns:
            excel_df_machines['Maintenance_Window'] = excel_df_machines[maintenance_col].apply(parse_maintenance)
        
        print(f"[Excel Schedule] Loaded {len(excel_df_machines)} machines")
        
        # Calculate effective times for Excel jobs
        effective_times = []
        for idx, op in excel_df_ops.iterrows():
            op_type = op.get('Op_Type', 'MILLING')
            eligible_machines = get_eligible_machines(op_type)
            for machine_id in eligible_machines:
                machine_row = excel_df_machines[excel_df_machines['Machine_ID'] == machine_id]
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
        
        excel_df_effective = pd.DataFrame(effective_times)
        print(f"[Excel Schedule] Calculated {len(excel_df_effective)} effective time entries")
        
        if excel_df_effective.empty:
            raise HTTPException(
                status_code=500,
                detail="No eligible machines found for any operations. Check operation types and machine capabilities."
            )
        
        # Run scheduling with selected heuristic (ISOLATED - NO GLOBAL STATE MODIFICATION)
        print(f"[Excel Schedule] Starting scheduler with {heuristic}")
        scheduler = CNCScheduler(
            df_ops=excel_df_ops,
            df_machines=excel_df_machines,
            df_effective=excel_df_effective,
            df_penalties=excel_df_penalties
        )
        
        schedule = scheduler.run_scheduling(heuristic=heuristic, verbose=False)
        print(f"[Excel Schedule] Scheduling complete, type: {type(schedule)}, shape: {schedule.shape if hasattr(schedule, 'shape') else 'N/A'}")
        
        # Handle case where scheduler returns False or invalid data
        if schedule is False or schedule is None:
            raise HTTPException(
                status_code=500, 
                detail="Scheduling failed: Scheduler returned no valid schedule. Check that operations can be assigned to available machines."
            )
        
        if not isinstance(schedule, pd.DataFrame) or schedule.empty:
            raise HTTPException(
                status_code=500, 
                detail="Scheduling failed: Unable to generate schedule. Please check if operations have valid machine assignments."
            )
        
        # Calculate metrics (ISOLATED - NO GLOBAL STATE MODIFICATION)
        metrics = calculate_metrics(schedule, excel_df_ops, heuristic)
        print(f"[Excel Schedule] Metrics calculated: {metrics}")
        
        # DO NOT store in global state - Excel upload is completely isolated
        # This keeps Dashboard data and Excel data separate
        
        return {
            "status": "success",
            "message": f"Scheduled {len(jobs)} jobs using {heuristic} (Excel data - isolated from Dashboard)",
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
        import traceback
        error_trace = traceback.format_exc()
        print(f"[Excel Schedule ERROR] {error_trace}")
        raise HTTPException(status_code=500, detail=f"Scheduling error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
