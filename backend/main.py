# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import time
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import json

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
# from core.cp_sat_scheduler import solve_with_cpsat

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
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    gemini_model = None
    
    # Configure Gemini regardless (needed for SchemaMapper)
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Create a gemini model handle lazily; if model name is unsupported this will be caught
            try:
                gemini_model = genai.GenerativeModel('gemini-flash-latest')
            except Exception as gm_err:
                print(f"[main] Gemini model init failed: {type(gm_err).__name__}: {str(gm_err)}")
                gemini_model = None
        except Exception as gerr:
            print(f"[main] Gemini configuration failed: {type(gerr).__name__}: {str(gerr)}")
            gemini_model = None

    # Prefer OpenRouter -> Mistral -> Gemini for insights. Gemini still used for schema mapping if available.
    if OPENROUTER_API_KEY:
        AI_ENABLED = True
        AI_PROVIDER = 'openrouter'
        print(f"AI enabled: OpenRouter primary (Mistral/Gemini fallback possible). Gemini schema mapping: {'Yes' if gemini_model else 'No'})")
    elif MISTRAL_API_KEY:
        AI_ENABLED = True
        AI_PROVIDER = 'mistral'
        print(f"AI enabled: Mistral primary (Gemini schema mapping: {'Yes' if gemini_model else 'No'})")
    elif GEMINI_API_KEY:
        AI_ENABLED = True
        AI_PROVIDER = 'gemini'
        print("AI enabled: Gemini primary for insights and schema mapping")
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
        # Default outsourcing threshold: vendor must be below this fraction of in-house cost
        # to be selected. Lower value -> fewer outsourcing decisions (improves in-house utilization).
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

class BuyMachineRequest(BaseModel):
    machine_id: str  # ID of the machine to clone
    hourly_labor_rate: Optional[float] = 30.0  # $/hr for labor cost calculation

class AddMachineRequest(BaseModel):
    machine_id: str
    machine_type: str
    op_types: str  # Comma-separated list of Op_Types
    speed_factor: float = 1.0
    hourly_rate: float = 30.0
    maintenance_cost: float = 100.0
    energy_cost_per_hour: float = 10.0
    purchase_price: Optional[float] = 50000.0

class RemoveMachineRequest(BaseModel):
    machine_id: str

# Helper functions
def load_all_data(sample_size=None):
    """Load and preprocess all data files"""
    # Get the parent directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    attempts = []
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

    # Total processing time in minutes (per unit * quantity). If Proc_Time is in hours, user must convert upstream.
    if 'Proc_Time_per_Unit' in df_ops.columns:
        proc_per_unit = pd.to_numeric(df_ops['Proc_Time_per_Unit'], errors='coerce')
        qty = pd.to_numeric(df_ops['Quantity'], errors='coerce')
        df_ops['Total_Proc_Min'] = proc_per_unit * qty
        # Heuristic: if Proc_Time_per_Unit looks like hours (small decimals <=24) convert to minutes
        try:
            vals = proc_per_unit.dropna()
            if len(vals) > 0 and vals.max() <= 24 and vals.mean() < 10:
                # Likely hours — convert Total_Proc_Min from hours to minutes
                df_ops['Total_Proc_Min'] = df_ops['Total_Proc_Min'] * 60
                print('[data-load] Converted Proc_Time_per_Unit from hours->minutes (multiplied by 60)')
        except Exception:
            pass
        # Ensure numeric and no NaNs for Total_Proc_Min when Proc_Time_per_Unit path used
        df_ops['Total_Proc_Min'] = pd.to_numeric(df_ops['Total_Proc_Min'], errors='coerce').fillna(0)
    elif 'Proc_Time' in df_ops.columns:
        # Older datasets may have a Proc_Time column already representing per-operation time
        df_ops['Total_Proc_Min'] = pd.to_numeric(df_ops['Proc_Time'], errors='coerce') * pd.to_numeric(df_ops['Quantity'], errors='coerce')
        df_ops['Total_Proc_Min'] = pd.to_numeric(df_ops['Total_Proc_Min'], errors='coerce').fillna(0)
    else:
        # No processing time columns found — default to zero to avoid crashes downstream
        df_ops['Total_Proc_Min'] = 0

    # Release/Due time conversion deferred until after vendor merge so a single
    # consistent minutes-per-day conversion (workday-based) is applied.
    # (See later where MINUTES_PER_DAY = 8 * 60 is used.)

    # DO NOT pre-assign Assignment_Type - let scheduler heuristics determine it
    # Assignment_Type will be set based on make_or_buy_decision during scheduling
    if 'Assignment_Type' not in df_ops.columns:
        df_ops['Assignment_Type'] = None  # Will be determined by scheduler

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

    # Process vendor data (original behavior)
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

    # DO NOT call make_or_buy_decision here - it will be called by the scheduler
    # The scheduler will determine Assignment_Type based on the selected heuristic

    return df_ops, df_machines, df_effective, df_penalties, df_vendors

def get_ai_insights(prompt: str, context_data: Optional[Dict] = None):
    """Generate AI insights using OpenRouter, Mistral, or Gemini with safe fallbacks.

    Strategy:
    - Prefer OpenRouter (Claude) when available
    - If OpenRouter fails due to token limits, retry with a shortened context
    - If OpenRouter unavailable or fails, try Mistral (HTTP) if configured
    - Finally fallback to Gemini (google.generativeai) if configured
    """
    if not AI_ENABLED:
        return {'text': "AI insights are disabled. Please add OPENROUTER_API_KEY, MISTRAL_API_KEY, or GEMINI_API_KEY to your environment.", 'provider_used': None, 'attempts': []}

    # Track provider attempts for debug / frontend visibility
    attempts = []

    # Clean AI text output to remove Markdown and make it professional/plain text
    def _clean_ai_text(text: Optional[str]) -> str:
        try:
            if text is None:
                return ''
            s = str(text)
            # Remove fenced code blocks
            s = re.sub(r'```[\s\S]*?```', '', s)
            # Remove ATX headings (e.g., #, ##, ###)
            s = re.sub(r'^#{1,6}\s*', '', s, flags=re.M)
            # Remove bold/italic markers
            s = s.replace('**', '').replace('__', '')
            # Remove inline code ticks
            s = re.sub(r'`([^`]*)`', r'\1', s)
            # Remove markdown table rows
            lines = []
            for ln in s.splitlines():
                if '|' in ln and re.search(r'\|', ln):
                    continue
                # Normalize unordered list markers
                ln = re.sub(r'^\s*[-*+]\s*', '- ', ln)
                lines.append(ln.rstrip())
            s = '\n'.join(lines)
            # Collapse multiple blank lines
            s = re.sub(r'\n\s*\n+', '\n\n', s)
            return s.strip()
        except Exception:
            return str(text or '')

    def _format_response(raw_text: Optional[str], provider: Optional[str]):
        cleaned = _clean_ai_text(raw_text)
        return {'text': cleaned, 'provider_used': provider, 'attempts': attempts}

    def summarize_context(ctx):
        """Create a compact summary of context_data to avoid token limits."""
        try:
            if ctx is None:
                return ''
            if isinstance(ctx, dict):
                lines = []
                for k, v in list(ctx.items())[:4]:
                    m = v.get('metrics') if isinstance(v, dict) and 'metrics' in v else v
                    if isinstance(m, dict):
                        parts = []
                        for f in ('Makespan_Days', 'Total_Tardiness_Days', 'Total_Cost_$', 'On_Time_%', 'Machine_Utilization_%'):
                            if f in m:
                                parts.append(f"{f}={m.get(f)}")
                        lines.append(f"{k}: " + (", ".join(parts) if parts else "summary_present"))
                    else:
                        lines.append(f"{k}: {str(m)[:200]}")
                return "\n".join(lines)
            if isinstance(ctx, list):
                lines = []
                for i, item in enumerate(ctx[:4], 1):
                    h = item.get('Heuristic') if isinstance(item, dict) else f'item{i}'
                    m = item.get('metrics') if isinstance(item, dict) and 'metrics' in item else item
                    if isinstance(m, dict):
                        parts = []
                        for f in ('Makespan_Days', 'Total_Tardiness_Days', 'Total_Cost_$', 'On_Time_%'):
                            if f in m:
                                parts.append(f"{f}={m.get(f)}")
                        lines.append(f"{h}: " + ", ".join(parts))
                    else:
                        lines.append(f"{h}: {str(m)[:200]}")
                return "\n".join(lines)
            return str(ctx)[:1000]
        except Exception:
            return ''

    # Compose prompts
    base_prompt = f"""
You are an expert in manufacturing scheduling, operations research, and production planning.

**CONTEXT:**
This is a CNC job scheduling application with 4 heuristics:
- SPT (Shortest Processing Time): Schedules shortest jobs first
- EDD (Earliest Due Date): Prioritizes jobs with nearest deadlines
- CR (Critical Ratio): Uses due_date/processing_time ratio
- PRIORITY (Job priority-based): Schedules based on job priority values (1=High, 2=Medium, 3=Low)

{prompt}
"""

    full_prompt = base_prompt
    short_prompt = base_prompt
    if context_data:
        full_prompt += f"\n\n**DATA CONTEXT:**\n{context_data}\n"
        short_summary = summarize_context(context_data)
        short_prompt += f"\n\n**DATA SUMMARY:**\n{short_summary}\n"

    full_prompt += """\n\n**OUTPUT REQUIREMENTS:**
Provide 3-5 direct, actionable insights as bullet points.
- Start immediately with insights (no preamble like \"Based on the data\" or \"Here are the insights\")
- Use proper spacing between bullets for readability
- Use clear punctuation and complete sentences
- Focus on specific metrics and values (e.g., \"Total Cost: $X\" not \"cost parameter\")
- Be concise and professional
- Do not include any meta-commentary about the format"""

    # Helper: call Mistral (best-effort)
    def call_mistral(prompt_text: str):
        if not MISTRAL_API_KEY:
            raise RuntimeError("No MISTRAL_API_KEY configured")
        # Mistral expects a `messages` array similar to other chat endpoints
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            # Surface the body for debugging
            raise RuntimeError(f"Mistral API error: {resp.status_code} - {resp.text}")

        data = resp.json()
        # Preferred shape: choices[0].message.content
        try:
            if isinstance(data, dict) and 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
                first = data['choices'][0]
                # Some responses put the text under message.content
                if isinstance(first, dict):
                    msg = first.get('message') or first.get('message')
                    if isinstance(msg, dict) and 'content' in msg:
                        return msg['content']
                    # fallback to choice text fields
                    for k in ('text', 'content', 'output'):
                        if k in first and isinstance(first[k], str):
                            return first[k]
        except Exception:
            pass

        # Legacy/fallback parsing: accept other shapes
        if isinstance(data, dict):
            if 'output' in data and isinstance(data['output'], str):
                return data['output']
            if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
                first = data['results'][0]
                if isinstance(first, dict):
                    for k in ('content', 'text', 'output'):
                        if k in first:
                            return first[k]
                if isinstance(first, str):
                    return first
            if 'text' in data and isinstance(data['text'], str):
                return data['text']
        return json.dumps(data)[:4000]

    try:
        # 1) OpenRouter primary
        if AI_PROVIDER == 'openrouter':
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}", "Content-Type": "application/json"},
                    json={"model": "anthropic/claude-3.5-sonnet", "messages": [{"role": "user", "content": full_prompt}], "max_tokens": 1000, "temperature": 0.7},
                    timeout=30,
                )
                if resp.status_code == 200:
                    text = resp.json()['choices'][0]['message']['content']
                    attempts.append({'provider': 'openrouter', 'status': 'success'})
                    return _format_response(text, 'openrouter')
                # Token limit exceeded -> retry with short prompt
                if resp.status_code == 402 and context_data:
                    try:
                        print('[AI Insights] OpenRouter token limit exceeded — retrying with shortened context')
                        retry = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}", "Content-Type": "application/json"},
                            json={"model": "anthropic/claude-3.5-sonnet", "messages": [{"role": "user", "content": short_prompt}], "max_tokens": 1000, "temperature": 0.7},
                            timeout=30,
                        )
                        if retry.status_code == 200:
                            text = retry.json()['choices'][0]['message']['content']
                            attempts.append({'provider': 'openrouter', 'status': 'retry_success', 'note': 'short_prompt'})
                            return _format_response(text, 'openrouter')
                    except Exception:
                        pass
                # Otherwise try Mistral then Gemini
                last_err = f"OpenRouter API error: {resp.status_code} - {resp.text[:200]}"
                attempts.append({'provider': 'openrouter', 'status': 'error', 'code': resp.status_code, 'message': resp.text[:200]})
                if MISTRAL_API_KEY:
                    try:
                        mtxt = call_mistral(short_prompt if context_data else full_prompt)
                        attempts.append({'provider': 'mistral', 'status': 'success'})
                        return _format_response(mtxt, 'mistral')
                    except Exception as me:
                        attempts.append({'provider': 'mistral', 'status': 'error', 'message': str(me)[:200]})
                        last_err += f"; Mistral failed: {str(me)[:200]}"
                if GEMINI_API_KEY and gemini_model:
                    try:
                        gm = gemini_model.generate_content(short_prompt if context_data else full_prompt)
                        attempts.append({'provider': 'gemini', 'status': 'success'})
                        return _format_response(gm.text, 'gemini')
                    except Exception as ge:
                        attempts.append({'provider': 'gemini', 'status': 'error', 'message': str(ge)[:200]})
                        last_err += f"; Gemini failed: {str(ge)[:200]}"
                return _format_response(last_err, None)

        # 2) Mistral primary
        if AI_PROVIDER == 'mistral':
            try:
                mtxt = call_mistral(full_prompt)
                attempts.append({'provider': 'mistral', 'status': 'success'})
                return _format_response(mtxt, 'mistral')
            except Exception as me:
                attempts.append({'provider': 'mistral', 'status': 'error', 'message': str(me)[:200]})
                if GEMINI_API_KEY and gemini_model:
                    try:
                        gm = gemini_model.generate_content(short_prompt if context_data else full_prompt)
                        attempts.append({'provider': 'gemini', 'status': 'success'})
                        return _format_response(gm.text, 'gemini')
                    except Exception as ge:
                        attempts.append({'provider': 'gemini', 'status': 'error', 'message': str(ge)[:200]})
                        return _format_response(f"Mistral failed: {str(me)[:200]}; Gemini failed: {str(ge)[:200]}", None)
                return _format_response(f"Mistral failed: {str(me)}", None)

        # 3) Gemini primary
        if AI_PROVIDER == 'gemini':
            if not gemini_model:
                attempts.append({'provider': 'gemini', 'status': 'unconfigured'})
                return _format_response("AI insights unavailable: No valid API keys configured", None)
            response = gemini_model.generate_content(short_prompt if context_data else full_prompt)
            attempts.append({'provider': 'gemini', 'status': 'success'})
            return _format_response(response.text, 'gemini')

    except Exception as e:
        em = str(e)
        attempts.append({'provider': 'internal', 'status': 'error', 'message': em[:300]})
        if "API_KEY_INVALID" in em or "API key not valid" in em:
            return _format_response("⚠️ API key invalid or expired. Please update your AI keys in the environment.", None)
        return _format_response(f"Error generating AI insights: {em}", None)

# API Endpoints

@app.get("/")
def read_root():
    return {"message": "CNC Scheduling API v2.0", "status": "running"}

@app.get("/api/health")
def health_check():
    """Health check endpoint with system status"""
    return {
        "status": "healthy",
        "version": "2.0",
        "ai_enabled": AI_ENABLED,
        "ai_provider": AI_PROVIDER,
        "data_loaded": state.df_ops is not None,
        "machines_count": len(state.df_machines) if state.df_machines is not None else 0,
        "operations_count": len(state.df_ops) if state.df_ops is not None else 0,
        "current_heuristic": state.current_heuristic,
        "available_heuristics": ["SPT", "EDD", "CR", "PRIORITY"]
    }

@app.get("/api/endpoints")
def list_endpoints():
    """List all available API endpoints"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    
    # Group by category
    categories = {
        "Data Management": [],
        "Scheduling": [],
        "Analysis": [],
        "CapEx": [],
        "AI & Insights": [],
        "Excel Ingestion": [],
        "Operations": [],
        "Debug": [],
        "Other": []
    }
    
    for route in routes:
        path = route["path"]
        if "/api/data/" in path:
            categories["Data Management"].append(route)
        elif "/api/schedule/" in path:
            categories["Scheduling"].append(route)
        elif "/api/analysis/" in path:
            categories["Analysis"].append(route)
        elif "/api/capex/" in path:
            categories["CapEx"].append(route)
        elif "/api/ai/" in path:
            categories["AI & Insights"].append(route)
        elif "/api/excel/" in path:
            categories["Excel Ingestion"].append(route)
        elif "/api/job/" in path or "/api/machine/" in path or "/api/outsourcing/" in path:
            categories["Operations"].append(route)
        elif "/api/debug/" in path:
            categories["Debug"].append(route)
        else:
            categories["Other"].append(route)
    
    return {
        "total_endpoints": len(routes),
        "categories": categories
    }

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

@app.get("/api/debug/schedule-sample")
def get_schedule_sample():
    """Debug endpoint: Get sample of current schedule to inspect tardiness"""
    if state.current_heuristic is None or state.current_heuristic not in state.schedules:
        raise HTTPException(status_code=400, detail="No schedule available. Compute a heuristic first.")
    
    schedule = state.schedules[state.current_heuristic]
    
    # Get tardiness statistics
    total_tardiness_min = schedule['Tardiness'].sum()
    max_tardiness_min = schedule['Tardiness'].max()
    avg_tardiness_min = schedule['Tardiness'].mean()
    late_ops = (schedule['Tardiness'] > 0).sum()
    
    # Get sample of late operations
    late_samples = schedule[schedule['Tardiness'] > 0].nlargest(5, 'Tardiness')[
        ['Operation_ID', 'Job_ID', 'Machine_ID', 'Start_Time', 'End_Time', 'Due_Time', 'Tardiness']
    ].to_dict('records')
    
    # Get sample of on-time operations
    ontime_samples = schedule[schedule['Tardiness'] == 0].head(5)[
        ['Operation_ID', 'Job_ID', 'Machine_ID', 'Start_Time', 'End_Time', 'Due_Time', 'Tardiness']
    ].to_dict('records')
    
    return {
        "heuristic": state.current_heuristic,
        "total_operations": len(schedule),
        "late_operations": int(late_ops),
        "tardiness_stats_minutes": {
            "total": float(total_tardiness_min),
            "max": float(max_tardiness_min),
            "average": float(avg_tardiness_min)
        },
        "tardiness_stats_days_8hr": {
            "total": float(total_tardiness_min / 480),
            "max": float(max_tardiness_min / 480),
            "average": float(avg_tardiness_min / 480)
        },
        "late_operation_samples": late_samples,
        "ontime_operation_samples": ontime_samples
    }

@app.get("/api/data/machines")
def get_machine_data():
    """Get machine data including maintenance windows"""
    if state.df_machines is None:
        return {"machines": [], "count": 0, "message": "No data loaded yet"}
    
    try:
        # Convert DataFrame to dict
        # We avoid .fillna('') on the whole DF because it can corrupt lists
        machines = state.df_machines.to_dict('records')
        
        cleaned_machines = []
        for machine in machines:
            cleaned_machine = {}
            for key, value in machine.items():
                # 1. If it's a list/dict (like Maintenance Window), keep it as is
                if isinstance(value, (list, dict)):
                    cleaned_machine[key] = value
                    continue
                
                # 2. Handle scalar values safely
                try:
                    if value is None or pd.isna(value) or str(value).lower() == 'nan':
                        cleaned_machine[key] = None
                    else:
                        cleaned_machine[key] = value
                except Exception:
                    # Fallback for any types that confuse pd.isna
                    cleaned_machine[key] = str(value)
            
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
    # REMOVE 'WEIGHTED' AND 'SLACK' FROM THIS LIST
    if heuristic not in ['SPT', 'EDD', 'CR', 'PRIORITY']:
        raise HTTPException(status_code=400, detail="Invalid heuristic. Allowed: SPT, EDD, CR, PRIORITY")
    
    # ... rest of function
    
    try:
        # Make-or-buy decisions are evaluated at compute time (not during data load)
        # so the scheduler receives operation Assignment_Type values derived
        # from the current state, vendor costs and thresholds.
        df_ops_for_sched = state.df_ops.copy()
        try:
            for idx, op in df_ops_for_sched.iterrows():
                try:
                    decision = make_or_buy_decision(op, state.df_effective, cost_threshold=state.cost_threshold)
                except TypeError:
                    # Support alternate signature: make_or_buy_decision(operation, df_effective, threshold)
                    decision = make_or_buy_decision(op, state.df_effective, state.cost_threshold)

                # If decision indicates outsourcing, set Assignment_Type for scheduler input
                if decision is not None:
                    # decision can be ('OUTSOURCE', cost) or ('OUTSOURCE', cost, reason)
                    if isinstance(decision, (list, tuple)) and len(decision) > 0 and str(decision[0]).upper() == 'OUTSOURCE':
                        df_ops_for_sched.at[idx, 'Assignment_Type'] = 'OUTSOURCE'
        except Exception:
            # Fail-safe: leave state.df_ops unchanged if any error occurs while evaluating decisions
            df_ops_for_sched = state.df_ops.copy()

        scheduler = CNCScheduler(
            df_ops_for_sched,
            state.df_machines.copy(),
            state.df_effective.copy(),
            state.df_penalties.copy()
        )

        schedule = scheduler.run_scheduling(heuristic=heuristic, verbose=False)

        # Defensive check: scheduler may return False or None on error
        if schedule is False or schedule is None:
            raise HTTPException(
                status_code=500,
                detail="Scheduling failed: Scheduler returned no valid schedule. Check that operations can be assigned to available machines."
            )

        if not isinstance(schedule, pd.DataFrame) or schedule.empty:
            raise HTTPException(
                status_code=500,
                detail="Scheduling failed: Scheduler returned invalid or empty schedule."
            )

        # Merge in Priority, Assignment_Type and supporting fields for CR debugging
        try:
            # Rename scheduler's Proc_Time to Scheduled_Proc_Time to avoid confusion
            schedule_df = schedule.rename(columns={'Proc_Time': 'Scheduled_Proc_Time'})
            
            # Merge scheduler output with supporting fields from the original ops
            # but do NOT merge `Assignment_Type` from loaded data — keep the
            # scheduler's Assignment_Type authoritative.
            schedule_df = schedule_df.merge(
                state.df_ops[[
                    'Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min'
                ]],
                on='Operation_ID', how='left'
            )
            schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
            schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            
            # Add Proc_Time field: use Total_Proc_Min from original data (shows actual processing time for all ops)
            schedule_df['Proc_Time'] = schedule_df['Total_Proc_Min']
            # Expose a frontend-friendly release field (minutes) mapped from Release_Time_Min
            try:
                if 'Release_Time_Min' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time_Min'], errors='coerce')
                elif 'Release_Time' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time'], errors='coerce')
                elif 'Release' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release'], errors='coerce')
                else:
                    schedule_df['Release'] = None
                schedule_df['Release_Time'] = schedule_df['Release']
            except Exception:
                schedule_df['Release'] = schedule_df.get('Release_Time_Min', schedule_df.get('Release_Time', None))
                schedule_df['Release_Time'] = schedule_df['Release']
            
            try:
                schedule_df['Critical_Ratio'] = schedule_df.apply(
                    lambda r: (((r.get('Due_Time_Min') or 0) - (r.get('Release_Time_Min') or 0)) / (r.get('Total_Proc_Min') or 1)),
                    axis=1
                )
            except Exception:
                schedule_df['Critical_Ratio'] = None
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
            "schedule": schedule_df.to_dict('records'),
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/schedule/compute-all")
def compute_all_heuristics():
    """Compute schedules for all heuristics"""
    if state.df_ops is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    # UPDATE THIS LIST TO ONLY INCLUDE THE 4 ACTIVE HEURISTICS
    heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
    results = {}
    
    try:
        for heur in heuristics:
            # Evaluate make-or-buy at compute time so each heuristic run uses
            # the same decision logic (but does not change scheduler internals)
            df_ops_for_sched = state.df_ops.copy()
            try:
                for idx, op in df_ops_for_sched.iterrows():
                    try:
                        decision = make_or_buy_decision(op, state.df_effective, cost_threshold=state.cost_threshold)
                    except TypeError:
                        decision = make_or_buy_decision(op, state.df_effective, state.cost_threshold)
                    if isinstance(decision, (list, tuple)) and len(decision) > 0 and str(decision[0]).upper() == 'OUTSOURCE':
                        df_ops_for_sched.at[idx, 'Assignment_Type'] = 'OUTSOURCE'
            except Exception:
                df_ops_for_sched = state.df_ops.copy()

            scheduler = CNCScheduler(
                df_ops_for_sched,
                state.df_machines.copy(),
                state.df_effective.copy(),
                state.df_penalties.copy()
            )

            schedule = scheduler.run_scheduling(heuristic=heur, verbose=False)

            # Defensive check for invalid scheduler returns
            if schedule is False or schedule is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Scheduling failed for heuristic {heur}: scheduler returned no valid schedule"
                )
            if not isinstance(schedule, pd.DataFrame) or schedule.empty:
                raise HTTPException(
                    status_code=500,
                    detail=f"Scheduling failed for heuristic {heur}: scheduler returned invalid or empty schedule"
                )
            try:
                # Rename scheduler's Proc_Time to avoid overwriting original data
                schedule_df = schedule.rename(columns={'Proc_Time': 'Scheduled_Proc_Time'})
                
                # Merge in Priority, Assignment_Type, Proc and Release fields so frontend can display them
                # Keep scheduler's Assignment_Type; merge other supporting fields
                schedule_df = schedule_df.merge(
                    state.df_ops[[
                        'Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min'
                    ]],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
                
                # Add Proc_Time field: use Total_Proc_Min from original data
                schedule_df['Proc_Time'] = schedule_df['Total_Proc_Min']

                # Expose `Release` and `Release_Time` using Release_Time_Min for frontend convenience
                try:
                    if 'Release_Time_Min' in schedule_df.columns:
                        schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time_Min'], errors='coerce')
                    elif 'Release_Time' in schedule_df.columns:
                        schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time'], errors='coerce')
                    else:
                        schedule_df['Release'] = None
                    schedule_df['Release_Time'] = schedule_df['Release']
                except Exception:
                    schedule_df['Release'] = schedule_df.get('Release_Time_Min', schedule_df.get('Release_Time', None))
                    schedule_df['Release_Time'] = schedule_df['Release']
                
                try:
                    schedule_df['Critical_Ratio'] = schedule_df.apply(
                        lambda r: (((r.get('Due_Time_Min') or 0) - (r.get('Release_Time_Min') or 0)) / (r.get('Total_Proc_Min') or 1)),
                        axis=1
                    )
                except Exception:
                    schedule_df['Critical_Ratio'] = None
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
        # Merge Priority, Assignment_Type and release/proc fields into CPSAT schedule if available
        try:
            schedule_df = schedule_df.merge(
                state.df_ops[[
                    'Operation_ID', 'Priority', 'Assignment_Type', 'Release_Time_Min', 'Due_Time_Min', 'Total_Proc_Min'
                ]],
                on='Operation_ID', how='left'
            )
            schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
            schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            # Ensure Proc_Time is visible (from Total_Proc_Min) and expose Release fields
            if 'Total_Proc_Min' in schedule_df.columns:
                schedule_df['Proc_Time'] = pd.to_numeric(schedule_df['Total_Proc_Min'], errors='coerce').fillna(0)
            try:
                if 'Release_Time_Min' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time_Min'], errors='coerce')
                else:
                    schedule_df['Release'] = None
                schedule_df['Release_Time'] = schedule_df['Release']
            except Exception:
                schedule_df['Release'] = schedule_df.get('Release_Time_Min', None)
                schedule_df['Release_Time'] = schedule_df['Release']
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
        # Ensure supporting fields (Proc_Time, Release, Transfer, Setup) are present for frontend
        try:
            sup_cols = [c for c in ['Total_Proc_Min', 'Release_Time_Min', 'Release_Time', 'Due_Time_Min', 'Transfer_Min', 'Setup_Time', 'Outsource_Cost'] if c in state.df_ops.columns]
            if sup_cols:
                schedule_df = schedule_df.merge(state.df_ops[['Operation_ID'] + sup_cols], on='Operation_ID', how='left')

            # Proc_Time: prefer Total_Proc_Min, fallback to Scheduled_Proc_Time
            if 'Total_Proc_Min' in schedule_df.columns:
                schedule_df['Proc_Time'] = pd.to_numeric(schedule_df['Total_Proc_Min'], errors='coerce').fillna(0)
            else:
                schedule_df['Proc_Time'] = pd.to_numeric(schedule_df.get('Scheduled_Proc_Time', 0), errors='coerce').fillna(0)

            # Transfer_Time and Setup_Time
            if 'Transfer_Min' in schedule_df.columns:
                schedule_df['Transfer_Time'] = schedule_df.get('Transfer_Time', schedule_df['Transfer_Min'])
            else:
                schedule_df['Transfer_Time'] = schedule_df.get('Transfer_Time', 0)

            schedule_df['Setup_Time'] = schedule_df.get('Setup_Time', 0).fillna(0) if 'Setup_Time' in schedule_df.columns else 0

            # Expose Release fields consistently
            try:
                if 'Release_Time_Min' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time_Min'], errors='coerce')
                elif 'Release_Time' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time'], errors='coerce')
                elif 'Release' in schedule_df.columns:
                    schedule_df['Release'] = pd.to_numeric(schedule_df['Release'], errors='coerce')
                else:
                    schedule_df['Release'] = None
                schedule_df['Release_Time'] = schedule_df['Release']
            except Exception:
                schedule_df['Release'] = schedule_df.get('Release_Time_Min', schedule_df.get('Release_Time', None))
                schedule_df['Release_Time'] = schedule_df['Release']
            # Ensure outsourced operations carry a valid Outsource_Cost by attempting
            # to compute it from vendor table if the merged value is missing or zero.
            try:
                if 'Outsource_Cost' in schedule_df.columns and state.df_vendors is not None:
                    # build a mapping for quick lookup
                    vend = state.df_vendors.copy()
                    # normalize vendor id column name if needed
                    if 'Vendor_ID' in vend.columns:
                        vend_map = vend.set_index('Vendor_ID').to_dict('index')
                    else:
                        vend_map = {}

                    def _compute_vendor_cost(row):
                        try:
                            if str(row.get('Assignment_Type','')).upper() != 'OUTSOURCE':
                                return row.get('Outsource_Cost', 0)
                            existing = row.get('Outsource_Cost', None)
                            if existing and float(existing) > 0.01:
                                return existing
                            # lookup vendor_ref from state.df_ops
                            opid = row.get('Operation_ID')
                            ref = None
                            try:
                                ref = state.df_ops.loc[state.df_ops['Operation_ID'] == opid, 'Vendor_Ref']
                                if ref is not None and len(ref) > 0:
                                    ref = str(ref.values[0])
                                else:
                                    ref = None
                            except Exception:
                                ref = None

                            if not ref or ref not in vend_map:
                                return existing if existing is not None else 0

                            v = vend_map[ref]
                            unit = float(v.get('Outsource_Unit_Cost', 0) or 0)
                            transport = float(v.get('Transport_Cost', 0) or 0)
                            q = 1
                            try:
                                q = float(state.df_ops.loc[state.df_ops['Operation_ID'] == opid, 'Quantity'].values[0])
                            except Exception:
                                q = float(row.get('Quantity', 1) or 1)

                            q = max(q, 1)
                            quality = float(v.get('Quality_Factor', 1) or 1)
                            if quality == 0:
                                quality = 1
                            calc = ((unit * q) + transport) / quality
                            return float(calc)
                        except Exception:
                            return row.get('Outsource_Cost', 0)

                    schedule_df['Outsource_Cost'] = schedule_df.apply(_compute_vendor_cost, axis=1)
            except Exception:
                pass
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

        # Add Release_Day for frontend OperationStatus display (minutes -> 8-hour workdays)
        try:
            MINUTES_PER_DAY = 8 * 60
            if 'Release_Time_Min' in schedule_df.columns:
                schedule_df['Release_Day'] = (pd.to_numeric(schedule_df['Release_Time_Min'], errors='coerce') / MINUTES_PER_DAY).round(3)
            elif 'Release_Time' in schedule_df.columns:
                schedule_df['Release_Day'] = (pd.to_numeric(schedule_df['Release_Time'], errors='coerce') / MINUTES_PER_DAY).round(3)
            elif 'Release' in schedule_df.columns:
                schedule_df['Release_Day'] = (pd.to_numeric(schedule_df['Release'], errors='coerce') / MINUTES_PER_DAY).round(3)
            else:
                # If no release info, keep None to avoid adding incorrect defaults
                schedule_df['Release_Day'] = None
        except Exception:
            schedule_df['Release_Day'] = schedule_df.get('Release_Time_Min', None)

        # Backfill Outsource_Cost if missing for outsourced operations by consulting vendor table
        try:
            if 'Outsource_Cost' in schedule_df.columns and state.df_vendors is not None:
                vend = state.df_vendors.copy()
                vend_map = vend.set_index('Vendor_ID').to_dict('index') if 'Vendor_ID' in vend.columns else {}

                def _ensure_cost(row):
                    try:
                        if str(row.get('Assignment_Type','')).upper() != 'OUTSOURCE':
                            return row.get('Outsource_Cost', 0)
                        existing = row.get('Outsource_Cost', None)
                        if existing and float(existing) > 0.01:
                            return existing
                        opid = row.get('Operation_ID')
                        ref = None
                        try:
                            ref = state.df_ops.loc[state.df_ops['Operation_ID'] == opid, 'Vendor_Ref']
                            if ref is not None and len(ref) > 0:
                                ref = str(ref.values[0])
                            else:
                                ref = None
                        except Exception:
                            ref = None

                        if not ref or ref not in vend_map:
                            return existing if existing is not None else 0

                        v = vend_map[ref]
                        unit = float(v.get('Outsource_Unit_Cost', 0) or 0)
                        transport = float(v.get('Transport_Cost', 0) or 0)
                        q = 1
                        try:
                            q = float(state.df_ops.loc[state.df_ops['Operation_ID'] == opid, 'Quantity'].values[0])
                        except Exception:
                            q = float(row.get('Quantity', 1) or 1)
                        q = max(q, 1)
                        quality = float(v.get('Quality_Factor', 1) or 1)
                        if quality == 0:
                            quality = 1
                        calc = ((unit * q) + transport) / quality
                        return float(calc)
                    except Exception:
                        return row.get('Outsource_Cost', 0)

                schedule_df['Outsource_Cost'] = schedule_df.apply(_ensure_cost, axis=1)
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
        
        decisions = []
        for idx, op in state.df_ops.iterrows():
            # Result returns ('OUTSOURCE', cost) or None
            result = make_or_buy_decision(op, state.df_effective, state.cost_threshold)
            
            decision = 'IN_HOUSE'
            cost = 0
            
            if result:
                decision = result[0]
                cost = result[1] # This is the estimated cost we calculated
            
            decisions.append({
                'Operation_ID': op['Operation_ID'], 
                'New_Decision': decision,
                'Estimated_Cost': cost
            })
        
        df_decisions = pd.DataFrame(decisions)
        
        # Clean Merge
        if 'New_Decision' in state.df_ops.columns:
            state.df_ops.drop(columns=['New_Decision'], inplace=True)
        if 'Estimated_Cost' in state.df_ops.columns:
            state.df_ops.drop(columns=['Estimated_Cost'], inplace=True)
            
        state.df_ops = state.df_ops.merge(df_decisions, on='Operation_ID', how='left')
        
        # 1. Update Assignments
        state.df_ops['Assignment_Type'] = state.df_ops['New_Decision'].fillna('IN_HOUSE')
        
        # 2. CRITICAL FIX: Update Cost if it was 0/Missing
        # If we decided to outsource, we MUST have a cost value
        mask_update_cost = (state.df_ops['Assignment_Type'] == 'OUTSOURCE') & \
                           ((state.df_ops['Outsource_Cost'] <= 0) | (state.df_ops['Outsource_Cost'].isna()))
        
        state.df_ops.loc[mask_update_cost, 'Outsource_Cost'] = state.df_ops.loc[mask_update_cost, 'Estimated_Cost']
        
        # Cleanup
        state.df_ops.drop(columns=['New_Decision', 'Estimated_Cost'], inplace=True, errors='ignore')
        if 'Decision' in state.df_ops.columns:
            state.df_ops.drop(columns=['Decision'], inplace=True, errors='ignore')
        
        new_outsourced = len(state.df_ops[state.df_ops['Assignment_Type'] == 'OUTSOURCE'])
        
        # Recompute ALL heuristics that we currently have cached.
        # If no schedules cached but a heuristic is currently applied, ensure we at least recompute that one
        heuristics_to_recompute = list(state.schedules.keys())
        if not heuristics_to_recompute and state.current_heuristic:
            heuristics_to_recompute = [state.current_heuristic]
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
                # Merge supporting fields so front-end has access to proc/release/transfer/setup
                merge_base = ['Operation_ID', 'Priority', 'Assignment_Type']
                support = [c for c in ['Total_Proc_Min', 'Release_Time_Min', 'Release_Time', 'Due_Time_Min', 'Transfer_Min', 'Setup_Time', 'Outsource_Cost'] if c in state.df_ops.columns]
                merge_cols = merge_base + support
                schedule_df = schedule.merge(
                    state.df_ops[merge_cols],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')

                # Expose Proc_Time from Total_Proc_Min where available
                if 'Total_Proc_Min' in schedule_df.columns:
                    schedule_df['Proc_Time'] = pd.to_numeric(schedule_df['Total_Proc_Min'], errors='coerce').fillna(0)
                else:
                    schedule_df['Proc_Time'] = pd.to_numeric(schedule_df.get('Scheduled_Proc_Time', 0), errors='coerce').fillna(0)

                # Transfer and Setup
                if 'Transfer_Min' in schedule_df.columns:
                    schedule_df['Transfer_Time'] = schedule_df.get('Transfer_Time', schedule_df['Transfer_Min'])
                else:
                    schedule_df['Transfer_Time'] = schedule_df.get('Transfer_Time', 0)
                schedule_df['Setup_Time'] = schedule_df.get('Setup_Time', 0).fillna(0) if 'Setup_Time' in schedule_df.columns else 0

                # Release mapping
                try:
                    if 'Release_Time_Min' in schedule_df.columns:
                        schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time_Min'], errors='coerce')
                    elif 'Release_Time' in schedule_df.columns:
                        schedule_df['Release'] = pd.to_numeric(schedule_df['Release_Time'], errors='coerce')
                    else:
                        schedule_df['Release'] = None
                    schedule_df['Release_Time'] = schedule_df['Release']
                except Exception:
                    schedule_df['Release'] = schedule_df.get('Release_Time_Min', schedule_df.get('Release_Time', None))
                    schedule_df['Release_Time'] = schedule_df['Release']
            except Exception:
                schedule_df = schedule.copy()

            metrics = calculate_metrics(schedule_df, state.df_ops, heur)

            state.schedules[heur] = schedule_df
            state.metrics[heur] = metrics
            recomputed_metrics[heur] = metrics
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Outsourcing Policy Updated',
            'details': f"Threshold: {old_threshold} -> {request.cost_threshold}, Outsourced: {new_outsourced}"
        })
        # Also return the updated schedule for the currently applied heuristic (if any)
        updated_schedule = None
        updated_metrics_for_current = None
        try:
            if state.current_heuristic and state.current_heuristic in state.schedules:
                updated_schedule = state.schedules[state.current_heuristic].to_dict('records')
                updated_metrics_for_current = state.metrics.get(state.current_heuristic)
        except Exception:
            updated_schedule = None

        return {
            "status": "success",
            "message": f"Outsourcing policy updated. {len(heuristics_to_recompute)} heuristic(s) recomputed.",
            "new_outsourced_count": new_outsourced,
            "total_operations": len(state.df_ops),
            "heuristics_recomputed": heuristics_to_recompute,
            "metrics": recomputed_metrics,
            "updated_schedule": updated_schedule,
            "updated_schedule_metrics": updated_metrics_for_current
        }
    except Exception as e:
        print(f"Error in update_outsourcing_policy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Policy update failed: {str(e)}")
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
                
                # Outsource if vendor cost < 85% of in-house cost (original rule)
                if op_outsource_cost > 0 and op_outsource_cost < (op_inhouse_cost * 0.85):
                    outsourced_ops_list.append(op)
                    outsource_cost += op_outsource_cost
                    
                    # Track outsourcing details
                    savings = op_inhouse_cost - op_outsource_cost
                    savings_pct = (savings / op_inhouse_cost) * 100 if op_inhouse_cost > 0 else 0
                    
                    # Normalize operation type field (handle varying column names)
                    op_type = None
                    for k in ('Operation_Type', 'operation_type', 'Op_Type', 'OpType'):
                        if k in op and pd.notna(op.get(k)):
                            op_type = op.get(k)
                            break
                    if op_type is None:
                        op_type = op.get('Mat_Type') or 'UNKNOWN'

                    outsourcing_details.append({
                        'job_id': op.get('Job_ID'),
                        'operation_id': op.get('Operation_ID'),
                        'operation_type': op_type,
                        'proc_time_min': op.get('Total_Proc_Min', op.get('Proc_Time', 0)),
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

@app.get("/api/analysis/machine-roi")
def analyze_machine_roi():
    """
    Comprehensive Machine ROI Analysis for CapEx investment decisions.
    Analyzes each machine's utilization, throughput, revenue, costs, and ROI.
    """
    if state.df_ops is None or state.current_heuristic is None:
        raise HTTPException(status_code=400, detail="Data not loaded or no heuristic applied. Please compute and apply a heuristic first.")
    
    try:
        schedule_df = state.schedules.get(state.current_heuristic)
        if schedule_df is None or schedule_df.empty:
            raise HTTPException(status_code=400, detail="No schedule available")
        
        # Filter in-house operations only (exclude outsourced)
        in_house = schedule_df[schedule_df['Machine_ID'] != 'OUTSOURCE'].copy()
        
        # Analysis parameters
        hourly_labor_rate = 30  # $/hour
        energy_cost_rate = 0.10  # 10% of labor cost
        annual_hours = 2080  # Standard work year (40 hrs/week * 52 weeks)
        makespan_days = schedule_df['End_Time'].max() / 480 if not schedule_df.empty else 1
        
        # Calculate metrics per machine
        machine_metrics = []
        
        for machine_id in state.df_machines['Machine_ID']:
            machine_ops = in_house[in_house['Machine_ID'] == machine_id]
            machine_row = state.df_machines[state.df_machines['Machine_ID'] == machine_id].iloc[0]
            
            # Basic metrics
            jobs_count = machine_ops['Job_ID'].nunique() if not machine_ops.empty else 0
            operations_count = len(machine_ops)
            
            # Time analysis
            total_proc_time_min = machine_ops['Proc_Time'].sum() if not machine_ops.empty else 0
            total_setup_time_min = machine_ops['Setup_Time'].sum() if not machine_ops.empty else 0
            total_active_time_min = total_proc_time_min + total_setup_time_min
            
            # Utilization calculation
            if not machine_ops.empty:
                machine_start = machine_ops['Start_Time'].min()
                machine_end = machine_ops['End_Time'].max()
                machine_span_min = machine_end - machine_start
                utilization_pct = (total_active_time_min / machine_span_min * 100) if machine_span_min > 0 else 0
            else:
                machine_span_min = 0
                utilization_pct = 0
            
            # Revenue calculation (value produced)
            # Estimate: hourly rate * effective hours processed
            revenue = (total_proc_time_min / 60) * hourly_labor_rate * 1.5  # 1.5x multiplier for value-add
            
            # Operating costs
            labor_cost = (total_active_time_min / 60) * hourly_labor_rate
            energy_cost = labor_cost * energy_cost_rate
            
            # Maintenance cost (estimated from maintenance windows)
            maintenance_cost = 0
            maintenance_hours = 0
            try:
                maint_window = machine_row.get('Maintenance_Window')
                if maint_window:
                    if isinstance(maint_window, list):
                        maintenance_hours = sum(w.get('duration', 0) for w in maint_window) / 60
                    elif isinstance(maint_window, dict):
                        maintenance_hours = maint_window.get('duration', 0) / 60
                    maintenance_cost = maintenance_hours * 50  # $50/hr maintenance labor
            except Exception:
                pass
            
            total_operating_cost = labor_cost + energy_cost + maintenance_cost
            
            # Profit and ROI calculation
            profit = revenue - total_operating_cost
            
            # Purchase price (from machine_data.csv)
            purchase_price = 0
            try:
                price = machine_row.get('Purchase_Price_($)', machine_row.get('Purchase_Price($)'))
                if price and price not in [None, '', 'nan']:
                    purchase_price = float(price)
            except Exception:
                pass
            
            # ROI calculation
            if purchase_price > 0 and profit > 0:
                # Annualize profit based on makespan
                annual_profit = profit * (365 / makespan_days) if makespan_days > 0 else profit
                roi_pct = (annual_profit / purchase_price) * 100
                payback_years = purchase_price / annual_profit if annual_profit > 0 else float('inf')
            else:
                annual_profit = 0
                roi_pct = 0
                payback_years = float('inf')
            
            # Throughput metrics
            throughput_ops_per_day = operations_count / makespan_days if makespan_days > 0 else 0
            avg_cycle_time_hours = (total_active_time_min / operations_count / 60) if operations_count > 0 else 0
            
            # Speed factor
            speed_factor = machine_row.get('Speed_Factor', 1.0)
            
            # Idle time
            idle_time_min = machine_span_min - total_active_time_min if machine_span_min > total_active_time_min else 0
            idle_time_pct = (idle_time_min / machine_span_min * 100) if machine_span_min > 0 else 0
            
            # Op types handled
            op_types = []
            try:
                if not machine_ops.empty and 'Op_Type' in machine_ops.columns:
                    op_types = machine_ops['Op_Type'].unique().tolist()
            except Exception:
                pass
            
            machine_metrics.append({
                'machine_id': machine_id,
                'machine_type': machine_row.get('Machine_Type', 'N/A'),
                'speed_factor': float(speed_factor),
                'jobs_count': int(jobs_count),
                'operations_count': int(operations_count),
                'total_proc_hours': round(total_proc_time_min / 60, 2),
                'total_setup_hours': round(total_setup_time_min / 60, 2),
                'total_active_hours': round(total_active_time_min / 60, 2),
                'machine_span_hours': round(machine_span_min / 60, 2),
                'utilization_pct': round(utilization_pct, 1),
                'idle_hours': round(idle_time_min / 60, 2),
                'idle_pct': round(idle_time_pct, 1),
                'maintenance_hours': round(maintenance_hours, 2),
                'revenue': round(revenue, 2),
                'labor_cost': round(labor_cost, 2),
                'energy_cost': round(energy_cost, 2),
                'maintenance_cost': round(maintenance_cost, 2),
                'total_operating_cost': round(total_operating_cost, 2),
                'profit': round(profit, 2),
                'purchase_price': purchase_price,
                'annual_profit': round(annual_profit, 2),
                'roi_pct': round(roi_pct, 1),
                'payback_years': round(payback_years, 2) if payback_years != float('inf') else None,
                'throughput_ops_per_day': round(throughput_ops_per_day, 2),
                'avg_cycle_time_hours': round(avg_cycle_time_hours, 2),
                'op_types_handled': op_types
            })
        
        # Sort by ROI descending
        machine_metrics.sort(key=lambda x: x['roi_pct'], reverse=True)
        
        # Overall summary
        total_revenue = sum(m['revenue'] for m in machine_metrics)
        total_costs = sum(m['total_operating_cost'] for m in machine_metrics)
        total_profit = sum(m['profit'] for m in machine_metrics)
        avg_utilization = sum(m['utilization_pct'] for m in machine_metrics) / len(machine_metrics) if machine_metrics else 0
        
        # Investment recommendations (realistic rules)
        # - Use plant-wide avg utilization as a guardrail: if plant avg utilization is low,
        #   avoid aggressive expansion unless a machine is clearly capacity-constrained.
        recommendations = []
        plant_util = avg_utilization

        for m in machine_metrics:
            util = m.get('utilization_pct', 0)
            roi = m.get('roi_pct', 0)
            payback = m.get('payback_years')
            ops = m.get('operations_count', 0)

            # No work on machine -> review for redeploy/decommission
            if ops == 0:
                recommendations.append({
                    'machine_id': m['machine_id'],
                    'recommendation': 'REVIEW',
                    'priority': 'LOW',
                    'reason': 'No operations scheduled. Consider redeployment, selling or converting to flexible use.',
                    'potential_savings': round(m.get('maintenance_cost', 0) * 12, 2)
                })
                continue

            # Critical: negative ROI or clear loss-making
            if roi < 0:
                recommendations.append({
                    'machine_id': m['machine_id'],
                    'recommendation': 'CRITICAL_REVIEW',
                    'priority': 'HIGH',
                    'reason': f'Negative ROI ({roi}%). Operating costs exceed value produced. Immediate operational review required.',
                    'annual_loss': round(abs(m.get('annual_profit', 0)), 2)
                })
                continue

            # Capacity expansion logic (realistic): require both high utilization AND reasonable payback/ROI
            expand = False
            expand_reason = ''

            # If machine is heavily loaded (>=85%), strong case for expansion regardless of plant average
            if util >= 85:
                expand = True
                expand_reason = f'High utilization ({util}%) indicates capacity constraint.'

            # Otherwise require utilization above threshold AND financial justification
            elif util >= 75:
                # require either strong ROI or acceptable payback
                if (payback and payback <= 5) or roi >= 30:
                    expand = True
                    expand_reason = f'Utilization {util}% with favorable ROI/payback (ROI {roi}%, payback {payback} yrs).'

            # If plant utilization is low (<60%), be conservative: only expand at very high util (>=85)
            if plant_util < 60 and util < 85:
                # downgrade expansion to 'CONSIDER' or 'OPTIMIZE' unless clearly justified
                if expand:
                    recommendations.append({
                        'machine_id': m['machine_id'],
                        'recommendation': 'CONSIDER',
                        'priority': 'MEDIUM',
                        'reason': f"Plant avg utilization is low ({plant_util:.1f}%). {expand_reason} Consider operational fixes before CapEx.",
                        'estimated_additional_revenue': round(m.get('annual_profit', 0) * 0.5, 2)
                    })
                    continue

            if expand:
                recommendations.append({
                    'machine_id': m['machine_id'],
                    'recommendation': 'EXPAND',
                    'priority': 'HIGH' if util >= 85 or (payback and payback <= 5) else 'MEDIUM',
                    'reason': expand_reason or 'Capacity expansion recommended based on utilization and financials.',
                    'estimated_additional_revenue': round(m.get('annual_profit', 0) * 0.8, 2)
                })
                continue

            # Optimization candidates: moderate utilization but low ROI or high idle time
            if util < 50 and ops > 0:
                recommendations.append({
                    'machine_id': m['machine_id'],
                    'recommendation': 'OPTIMIZE',
                    'priority': 'MEDIUM',
                    'reason': f'Low utilization ({util}%). Consider shifting workload or rescheduling to increase throughput.',
                    'potential_savings': round(m.get('total_operating_cost', 0) * 0.25, 2)
                })
                continue

            # Conservative default: suggest monitoring / incremental improvements
            recommendations.append({
                'machine_id': m['machine_id'],
                'recommendation': 'MONITOR',
                'priority': 'LOW',
                'reason': 'Machine shows acceptable performance; prioritize monitoring and incremental optimization.',
            })
        
        return {
            "status": "success",
            "heuristic": state.current_heuristic,
            "analysis_period_days": round(makespan_days, 2),
            "parameters": {
                "hourly_labor_rate": hourly_labor_rate,
                "energy_cost_rate_pct": energy_cost_rate * 100,
                "annual_work_hours": annual_hours
            },
            "summary": {
                "total_machines": len(machine_metrics),
                "active_machines": len([m for m in machine_metrics if m['operations_count'] > 0]),
                "total_revenue": round(total_revenue, 2),
                "total_operating_costs": round(total_costs, 2),
                "total_profit": round(total_profit, 2),
                "avg_utilization_pct": round(avg_utilization, 1),
                "highest_roi_machine": machine_metrics[0]['machine_id'] if machine_metrics and machine_metrics[0]['roi_pct'] > 0 else None,
                "highest_roi_value": machine_metrics[0]['roi_pct'] if machine_metrics else 0
            },
            "machines": machine_metrics,
            "recommendations": recommendations
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Machine ROI analysis failed: {str(e)}")

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
    Add a new job with operations.
    Fills ALL required columns to prevent backend crashes.
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
        
        # Create new operations
        new_ops = []
        for i, op in enumerate(operations):
            op_id = f"{job_id}_OP{i+1}"
            
            # Safe Priority Handling
            raw_priority = op.get('priority', 3)
            try:
                priority_num = int(raw_priority)
            except (ValueError, TypeError):
                priority_map = {'High': 1, 'Medium': 3, 'Low': 5}
                priority_num = priority_map.get(str(raw_priority), 3)
            
            # Safe Numeric Conversions
            qty = int(op.get('quantity', 1))
            proc_time = float(op.get('proc_time', 60))
            
            new_op = {
                'Job_ID': job_id,
                'Operation_ID': op_id,
                'Op_Seq': i + 1,
                'Op_Type': op.get('operation_type', 'MILLING'),
                'Part_Type': 'New_Part',      # <--- Added Default
                'Mat_Type': 'STEEL',          # <--- Added Default (Prevents Scheduler Crash)
                'Tool_Group': 'TGA',          # <--- Added Default
                'Proc_Time_per_Unit': proc_time,
                'Total_Proc_Min': proc_time,  # Assuming input is total per op
                'Setup_Time': float(op.get('setup_time', 10)),
                'Transfer_Min': float(op.get('transfer_time', 5)),
                'Quantity': qty,
                'Release_Day': int(op.get('release_day', 0)),
                'Due_Day': int(op.get('due_day', 10)),
                'Priority': priority_num,
                'Vendor_Ref': op.get('vendor_ref', 'V1'),
                'Outsource_Cost': float(op.get('outsource_cost', 0)),
                'Outsource_Time_Min': float(op.get('outsource_time', 0)),
                'Release_Time_Min': int(op.get('release_day', 0)) * 480,
                'Due_Time_Min': int(op.get('due_day', 10)) * 480,
                'Completion_Day': 0,
                'Assignment_Type': 'IN_HOUSE',
                'Outsource_Flag': 'N'
            }
            new_ops.append(new_op)
        
        # Add to dataframe
        new_df = pd.DataFrame(new_ops)
        # Align columns with existing dataframe to prevent schema mismatch
        for col in state.df_ops.columns:
            if col not in new_df.columns:
                new_df[col] = None # Fill missing columns with None
        
        # Keep only relevant columns
        new_df = new_df[state.df_ops.columns.intersection(new_df.columns)]
        
        state.df_ops = pd.concat([state.df_ops, new_df], ignore_index=True)
        
        # Update effective times for new operations
        new_effective = []
        for op in new_ops:
            op_type = op['Op_Type']
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
        print(f"Error adding job: {e}")
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
# CAPEX ANALYSIS & MACHINE PURCHASE ENDPOINTS
# ============================================================================

@app.post("/api/capex/analyze")
def analyze_capex_opportunity(hourly_labor_rate: float = 30.0):
    """
    Analyze outsourced operations to find the biggest offender (most outsourced op type).
    For each machine that can perform that operation, calculate:
    - Cost to buy (machine purchase price)
    - Cost to run in-house (labor + power/energy for those jobs)
    - Savings (vendor cost - in-house cost)
    - ROI / payback period
    """
    if state.df_ops is None or state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        # Find outsourced operations
        outsourced = state.df_ops[state.df_ops['Assignment_Type'] == 'OUTSOURCE'].copy()
        
        simulated_mode = False
        if len(outsourced) == 0:
            # No operations currently marked as outsourced. Perform a simulation
            # using the provided hourly_labor_rate to see which operations would
            # be outsourced if make-or-buy were re-evaluated. This allows users
            # to test "what-if" scenarios (e.g., very high labor rates).
            simulated_mode = True
            simulated_outsourced_idx = []
            # Ensure df_effective is available for in-house cost calc
            df_effective = state.df_effective if getattr(state, 'df_effective', None) is not None else pd.DataFrame()

            for idx, op in state.df_ops.iterrows():
                # Try to compute in-house cost for this operation
                try:
                    inhouse_cost, _ = calculate_inhouse_cost(op, df_effective, hourly_rate=hourly_labor_rate)
                except Exception:
                    inhouse_cost = None

                # Use provided outsource cost if available, else fallback
                try:
                    outsource_cost = float(op.get('Outsource_Cost', 0) or 0)
                except Exception:
                    outsource_cost = 0

                # Basic fallback when data missing
                if outsource_cost <= 0.1 and inhouse_cost:
                    outsource_cost = inhouse_cost * 1.2

                # If we couldn't compute in-house, prefer outsource if vendor cost looks reasonable
                decision_outsource = False
                if inhouse_cost is None:
                    if outsource_cost > 0:
                        decision_outsource = True
                else:
                    if outsource_cost < (inhouse_cost * state.cost_threshold):
                        decision_outsource = True

                if decision_outsource:
                    simulated_outsourced_idx.append(idx)

            outsourced = state.df_ops.loc[simulated_outsourced_idx].copy()

            if len(outsourced) == 0:
                return {
                    "status": "success",
                    "message": "No outsourced operations found (including simulated scenario).",
                    "biggest_offender": None,
                    "recommendations": [],
                    "simulated": True
                }
        
        # Count by operation type
        op_type_counts = outsourced.groupby('Op_Type').size().sort_values(ascending=False)
        biggest_offender = op_type_counts.index[0]
        offender_count = int(op_type_counts.iloc[0])
        
        # Get all operations of that type that are outsourced
        offender_ops = outsourced[outsourced['Op_Type'] == biggest_offender].copy()
        
        # Total vendor cost for these operations
        total_vendor_cost = offender_ops['Outsource_Cost'].sum()
        
        # Find eligible machines (machines that can perform this operation type)
        eligible_machines = get_eligible_machines(biggest_offender)
        
        if len(eligible_machines) == 0:
            return {
                "status": "success",
                "message": f"Biggest offender is {biggest_offender} ({offender_count} ops), but no machines can handle it.",
                "biggest_offender": biggest_offender,
                "offender_count": offender_count,
                "recommendations": []
            }
        
        # For each eligible machine, compute financial metrics
        recommendations = []
        
        for machine_id in eligible_machines:
            machine_row = state.df_machines[state.df_machines['Machine_ID'] == machine_id]
            if len(machine_row) == 0:
                continue
            machine_row = machine_row.iloc[0]
            
            # Purchase price (default if not in CSV)
            purchase_price = None
            for col in ['Purchase_Price_($)', 'Purchase_Price', 'Purchase_Cost']:
                if col in machine_row.index:
                    purchase_price = machine_row.get(col)
                    break
            
            if purchase_price is None:
                purchase_price = 150000  # default $150k
            
            try:
                purchase_price = float(purchase_price)
            except Exception:
                purchase_price = 150000
            
            # Speed factor for effective time calculation
            speed_factor = machine_row.get('Speed_Factor', 1.0)
            try:
                speed_factor = float(speed_factor)
            except Exception:
                speed_factor = 1.0
            
            # Calculate in-house cost to run these jobs on this machine
            # Cost = (processing_time_hrs * hourly_labor_rate) + energy_cost
            # Simplification: energy cost ~ 10% of labor cost
            total_proc_min = 0
            for idx, op in offender_ops.iterrows():
                proc_min = op.get('Total_Proc_Min', 0)
                effective_min = proc_min / speed_factor if speed_factor > 0 else proc_min
                total_proc_min += effective_min
            
            total_proc_hrs = total_proc_min / 60.0
            labor_cost = total_proc_hrs * hourly_labor_rate
            energy_cost = labor_cost * 0.1  # assume 10% of labor for simplicity
            total_inhouse_cost = labor_cost + energy_cost
            
            # Savings = vendor cost - in-house cost
            savings = total_vendor_cost - total_inhouse_cost
            
            # ROI / Payback period (years)
            # Payback = Purchase Price / Annual Savings
            # Assume these jobs recur annually (or use actual frequency if known)
            # For simplicity: annual_savings = savings (treating current dataset as 1 year)
            if savings > 0:
                payback_years = purchase_price / savings
            else:
                payback_years = None  # not profitable
            
            recommendations.append({
                'machine_id': machine_id,
                'machine_type': machine_row.get('Machine_Type', 'Unknown'),
                'purchase_price': round(purchase_price, 2),
                'labor_cost': round(labor_cost, 2),
                'energy_cost': round(energy_cost, 2),
                'total_inhouse_cost': round(total_inhouse_cost, 2),
                'vendor_cost': round(total_vendor_cost, 2),
                'savings': round(savings, 2),
                'payback_years': round(payback_years, 2) if payback_years else None,
                'jobs_count': len(offender_ops),
                'total_proc_hours': round(total_proc_hrs, 2),
                'calculation_details': {
                    'hourly_labor_rate': hourly_labor_rate,
                    'speed_factor': speed_factor,
                    'total_processing_minutes': round(total_proc_min, 2),
                    'formula': {
                        'labor': f"{round(total_proc_hrs, 2)} hrs × ${hourly_labor_rate}/hr = ${round(labor_cost, 2)}",
                        'energy': f"10% of labor = ${round(energy_cost, 2)}",
                        'total_inhouse': f"${round(labor_cost, 2)} + ${round(energy_cost, 2)} = ${round(total_inhouse_cost, 2)}",
                        'savings': f"${round(total_vendor_cost, 2)} (vendor) - ${round(total_inhouse_cost, 2)} (in-house) = ${round(savings, 2)}",
                        'payback': f"${round(purchase_price, 2)} ÷ ${round(savings, 2)}/year = {round(payback_years, 2) if payback_years else 'N/A'} years" if savings > 0 else "No payback (vendor is cheaper)"
                    }
                }
            })
        
        # Sort by savings (descending)
        recommendations.sort(key=lambda x: x['savings'], reverse=True)
        
        # Generate AI explanation if enabled
        ai_explanation = None
        if AI_ENABLED and len(recommendations) > 0:
            try:
                best_rec = recommendations[0]
                analysis_context = f"""
You are a manufacturing financial analyst. Analyze this capital expenditure opportunity:

**Situation:**
- Operation Type: {biggest_offender}
- Currently Outsourced: {offender_count} operations
- Total Vendor Cost: ${round(total_vendor_cost, 2):,}

**Best Machine Recommendation:**
- Machine: {best_rec['machine_id']} ({best_rec['machine_type']})
- Purchase Price: ${best_rec['purchase_price']:,}
- Annual In-House Cost: ${best_rec['total_inhouse_cost']:,}
  - Labor ({best_rec['total_proc_hours']:.1f} hrs @ ${hourly_labor_rate}/hr): ${best_rec['labor_cost']:,}
  - Energy (10% estimate): ${best_rec['energy_cost']:,}
- Annual Savings: ${best_rec['savings']:,}
- Payback Period: {best_rec['payback_years'] if best_rec['payback_years'] else 'Not profitable'} years

Provide a brief 3-4 sentence analysis covering:
1. Whether this investment makes financial sense and why
2. Key risk factors or considerations
3. Strategic recommendation (buy now, wait, or avoid)

Be concise and actionable. Use simple language suitable for a manufacturing manager.
"""
                
                ai_explanation = get_ai_insights(analysis_context, context_data=None)
                
            except Exception as e:
                print(f"[CapEx] AI explanation failed: {e}")
                ai_explanation = None
        
        return {
            "status": "success",
            "biggest_offender": biggest_offender,
            "offender_count": offender_count,
            "total_vendor_cost": round(total_vendor_cost, 2),
            "recommendations": recommendations,
            "simulated": simulated_mode,
            "ai_explanation": ai_explanation,
            "assumptions": {
                "energy_cost_percent": 10,
                "dataset_represents": "annual volume",
                "hourly_labor_rate_used": hourly_labor_rate
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/capex/buy-machine")
def buy_machine(request: BuyMachineRequest):
    """
    Clone the specified machine and append it to machine_data.csv permanently.
    Assigns a new Machine_ID (e.g., M1 -> M1_NEW1).
    After adding machine, automatically recompute all heuristics and apply the best one.
    """
    if state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        machine_id = request.machine_id
        
        # Find the machine to clone
        machine_row = state.df_machines[state.df_machines['Machine_ID'] == machine_id]
        if len(machine_row) == 0:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
        
        machine_row = machine_row.iloc[0].copy()
        
        # Generate new ID
        # Find existing clones (e.g., M1_NEW1, M1_NEW2)
        base_id = machine_id
        existing_ids = state.df_machines['Machine_ID'].tolist()
        clone_num = 1
        new_id = f"{base_id}_NEW{clone_num}"
        while new_id in existing_ids:
            clone_num += 1
            new_id = f"{base_id}_NEW{clone_num}"
        
        # Create new row
        new_row = machine_row.to_dict()
        new_row['Machine_ID'] = new_id
        
        # Append to in-memory state
        new_row_df = pd.DataFrame([new_row])
        state.df_machines = pd.concat([state.df_machines, new_row_df], ignore_index=True)
        
        # Write to CSV permanently
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'machine_data.csv')
        
        state.df_machines.to_csv(csv_path, index=False)
        
        # Rebuild effective times with new machine
        effective_times = []
        for idx, op in state.df_ops.iterrows():
            op_type = op['Op_Type']
            eligible = get_eligible_machines(op_type)
            for mid in eligible:
                mrow = state.df_machines[state.df_machines['Machine_ID'] == mid]
                if len(mrow) > 0:
                    speed_factor = mrow.iloc[0]['Speed_Factor']
                    effective_proc_time = op['Total_Proc_Min'] / speed_factor
                    total_time = op['Setup_Time'] + effective_proc_time + op.get('Transfer_Min', 0)
                    effective_times.append({
                        'Operation_ID': op['Operation_ID'],
                        'Machine_ID': mid,
                        'Effective_Proc_Time': effective_proc_time,
                        'Total_Time': total_time
                    })
        state.df_effective = pd.DataFrame(effective_times)
        
        # Auto-recompute all heuristics to assign tasks to new machine
        heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
        results = {}
        best_heuristic = None
        best_score = float('inf')
        
        for heur in heuristics:
            df_ops_for_sched = state.df_ops.copy()
            try:
                for idx, op in df_ops_for_sched.iterrows():
                    try:
                        decision = make_or_buy_decision(op, state.df_effective, cost_threshold=state.cost_threshold)
                    except TypeError:
                        decision = make_or_buy_decision(op, state.df_effective, state.cost_threshold)
                    if isinstance(decision, (list, tuple)) and len(decision) > 0 and str(decision[0]).upper() == 'OUTSOURCE':
                        df_ops_for_sched.at[idx, 'Assignment_Type'] = 'OUTSOURCE'
            except Exception:
                pass

            scheduler = CNCScheduler(
                df_ops_for_sched,
                state.df_machines.copy(),
                state.df_effective.copy(),
                state.df_penalties.copy()
            )

            schedule = scheduler.run_scheduling(heuristic=heur, verbose=False)
            
            if schedule is not None and isinstance(schedule, pd.DataFrame) and not schedule.empty:
                schedule_df = schedule.rename(columns={'Proc_Time': 'Scheduled_Proc_Time'})
                schedule_df = schedule_df.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
                schedule_df['Proc_Time'] = schedule_df['Total_Proc_Min']
                
                metrics = calculate_metrics(schedule_df, state.df_machines)
                state.schedules[heur] = schedule_df
                state.metrics[heur] = metrics
                results[heur] = {'schedule': schedule_df.to_dict('records'), 'metrics': metrics}
                
                # Track best heuristic by makespan
                if metrics.get('Makespan_Days', float('inf')) < best_score:
                    best_score = metrics['Makespan_Days']
                    best_heuristic = heur
        
        # Apply the best heuristic automatically
        if best_heuristic and best_heuristic in state.schedules:
            state.current_heuristic = best_heuristic
            state.current_schedule = state.schedules[best_heuristic]
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Machine Purchased',
            'details': f"Cloned {machine_id} as {new_id}, recomputed all heuristics, applied {best_heuristic}"
        })
        
        return {
            "status": "success",
            "message": f"Successfully purchased {new_id} (clone of {machine_id}), recomputed heuristics, and applied {best_heuristic}",
            "new_machine_id": new_id,
            "machines_count": len(state.df_machines),
            "best_heuristic": best_heuristic,
            "results": {k: {'metrics': v['metrics']} for k, v in results.items()}
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/machines/add")
def add_machine(request: AddMachineRequest):
    """
    Add a new machine with specified parameters.
    Automatically rebuilds effective times and recomputes all heuristics.
    """
    if state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        # Check if machine ID already exists
        if request.machine_id in state.df_machines['Machine_ID'].values:
            raise HTTPException(status_code=400, detail=f"Machine {request.machine_id} already exists")
        
        # Create new machine row
        new_machine = {
            'Machine_ID': request.machine_id,
            'Machine_Type': request.machine_type,
            'Op_Types': request.op_types,
            'Speed_Factor': request.speed_factor,
            'Hourly_Rate': request.hourly_rate,
            'Maintenance_Cost': request.maintenance_cost,
            'Energy_Cost_per_Hour': request.energy_cost_per_hour,
            'Purchase_Price': request.purchase_price or 50000.0
        }
        
        # Append to in-memory state
        new_row_df = pd.DataFrame([new_machine])
        state.df_machines = pd.concat([state.df_machines, new_row_df], ignore_index=True)
        
        # Write to CSV permanently
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'machine_data.csv')
        state.df_machines.to_csv(csv_path, index=False)
        
        # Rebuild effective times
        effective_times = []
        for idx, op in state.df_ops.iterrows():
            op_type = op['Op_Type']
            eligible = get_eligible_machines(op_type)
            for mid in eligible:
                mrow = state.df_machines[state.df_machines['Machine_ID'] == mid]
                if len(mrow) > 0:
                    speed_factor = mrow.iloc[0]['Speed_Factor']
                    effective_proc_time = op['Total_Proc_Min'] / speed_factor
                    total_time = op['Setup_Time'] + effective_proc_time + op.get('Transfer_Min', 0)
                    effective_times.append({
                        'Operation_ID': op['Operation_ID'],
                        'Machine_ID': mid,
                        'Effective_Proc_Time': effective_proc_time,
                        'Total_Time': total_time
                    })
        state.df_effective = pd.DataFrame(effective_times)
        
        # Recompute all heuristics
        heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
        results = {}
        best_heuristic = None
        best_score = float('inf')
        
        for heur in heuristics:
            df_ops_for_sched = state.df_ops.copy()
            try:
                for idx, op in df_ops_for_sched.iterrows():
                    try:
                        decision = make_or_buy_decision(op, state.df_effective, cost_threshold=state.cost_threshold)
                    except TypeError:
                        decision = make_or_buy_decision(op, state.df_effective, state.cost_threshold)
                    if isinstance(decision, (list, tuple)) and len(decision) > 0 and str(decision[0]).upper() == 'OUTSOURCE':
                        df_ops_for_sched.at[idx, 'Assignment_Type'] = 'OUTSOURCE'
            except Exception:
                pass

            scheduler = CNCScheduler(
                df_ops_for_sched,
                state.df_machines.copy(),
                state.df_effective.copy(),
                state.df_penalties.copy()
            )

            schedule = scheduler.run_scheduling(heuristic=heur, verbose=False)
            
            if schedule is not None and isinstance(schedule, pd.DataFrame) and not schedule.empty:
                schedule_df = schedule.rename(columns={'Proc_Time': 'Scheduled_Proc_Time'})
                schedule_df = schedule_df.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
                schedule_df['Proc_Time'] = schedule_df['Total_Proc_Min']
                
                metrics = calculate_metrics(schedule_df, state.df_machines)
                state.schedules[heur] = schedule_df
                state.metrics[heur] = metrics
                results[heur] = {'schedule': schedule_df.to_dict('records'), 'metrics': metrics}
                
                if metrics.get('Makespan_Days', float('inf')) < best_score:
                    best_score = metrics['Makespan_Days']
                    best_heuristic = heur
        
        # Apply best heuristic
        if best_heuristic and best_heuristic in state.schedules:
            state.current_heuristic = best_heuristic
            state.current_schedule = state.schedules[best_heuristic]
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Machine Added',
            'details': f"Added {request.machine_id}, recomputed all heuristics, applied {best_heuristic}"
        })
        
        return {
            "status": "success",
            "message": f"Successfully added machine {request.machine_id}, recomputed heuristics, and applied {best_heuristic}",
            "machine_id": request.machine_id,
            "machines_count": len(state.df_machines),
            "best_heuristic": best_heuristic,
            "results": {k: {'metrics': v['metrics']} for k, v in results.items()}
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/machines/remove")
def remove_machine(request: RemoveMachineRequest):
    """
    Remove a machine from the system.
    Automatically rebuilds effective times and recomputes all heuristics.
    """
    if state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        machine_id = request.machine_id
        
        # Check if machine exists
        if machine_id not in state.df_machines['Machine_ID'].values:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
        
        # Remove from in-memory state
        state.df_machines = state.df_machines[state.df_machines['Machine_ID'] != machine_id].copy()
        
        # Write to CSV permanently
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'machine_data.csv')
        state.df_machines.to_csv(csv_path, index=False)
        
        # Rebuild effective times (excluding removed machine)
        effective_times = []
        for idx, op in state.df_ops.iterrows():
            op_type = op['Op_Type']
            eligible = get_eligible_machines(op_type)
            for mid in eligible:
                if mid == machine_id:
                    continue  # Skip removed machine
                mrow = state.df_machines[state.df_machines['Machine_ID'] == mid]
                if len(mrow) > 0:
                    speed_factor = mrow.iloc[0]['Speed_Factor']
                    effective_proc_time = op['Total_Proc_Min'] / speed_factor
                    total_time = op['Setup_Time'] + effective_proc_time + op.get('Transfer_Min', 0)
                    effective_times.append({
                        'Operation_ID': op['Operation_ID'],
                        'Machine_ID': mid,
                        'Effective_Proc_Time': effective_proc_time,
                        'Total_Time': total_time
                    })
        state.df_effective = pd.DataFrame(effective_times)
        
        # Recompute all heuristics
        heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY']
        results = {}
        best_heuristic = None
        best_score = float('inf')
        
        for heur in heuristics:
            df_ops_for_sched = state.df_ops.copy()
            try:
                for idx, op in df_ops_for_sched.iterrows():
                    try:
                        decision = make_or_buy_decision(op, state.df_effective, cost_threshold=state.cost_threshold)
                    except TypeError:
                        decision = make_or_buy_decision(op, state.df_effective, state.cost_threshold)
                    if isinstance(decision, (list, tuple)) and len(decision) > 0 and str(decision[0]).upper() == 'OUTSOURCE':
                        df_ops_for_sched.at[idx, 'Assignment_Type'] = 'OUTSOURCE'
            except Exception:
                pass

            scheduler = CNCScheduler(
                df_ops_for_sched,
                state.df_machines.copy(),
                state.df_effective.copy(),
                state.df_penalties.copy()
            )

            schedule = scheduler.run_scheduling(heuristic=heur, verbose=False)
            
            if schedule is not None and isinstance(schedule, pd.DataFrame) and not schedule.empty:
                schedule_df = schedule.rename(columns={'Proc_Time': 'Scheduled_Proc_Time'})
                schedule_df = schedule_df.merge(
                    state.df_ops[['Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min']],
                    on='Operation_ID', how='left'
                )
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
                schedule_df['Proc_Time'] = schedule_df['Total_Proc_Min']
                
                metrics = calculate_metrics(schedule_df, state.df_machines)
                state.schedules[heur] = schedule_df
                state.metrics[heur] = metrics
                results[heur] = {'schedule': schedule_df.to_dict('records'), 'metrics': metrics}
                
                if metrics.get('Makespan_Days', float('inf')) < best_score:
                    best_score = metrics['Makespan_Days']
                    best_heuristic = heur
        
        # Apply best heuristic
        if best_heuristic and best_heuristic in state.schedules:
            state.current_heuristic = best_heuristic
            state.current_schedule = state.schedules[best_heuristic]
        
        state.activity_log.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Machine Removed',
            'details': f"Removed {machine_id}, recomputed all heuristics, applied {best_heuristic}"
        })
        
        return {
            "status": "success",
            "message": f"Successfully removed machine {machine_id}, recomputed heuristics, and applied {best_heuristic}",
            "machine_id": machine_id,
            "machines_count": len(state.df_machines),
            "best_heuristic": best_heuristic,
            "results": {k: {'metrics': v['metrics']} for k, v in results.items()}
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    use_openrouter=(AI_PROVIDER == 'openrouter'),
    mistral_api_key=os.environ.get('MISTRAL_API_KEY'),
    mistral_model=os.environ.get('MISTRAL_MODEL', 'mistral-small')
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


from fastapi import Form


@app.post("/api/excel/auto-map")
async def auto_map_columns(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    use_llm: bool = Form(True)
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
        excel_df_ops['Total_Proc_Min'] = pd.to_numeric(excel_df_ops['Proc_Time_per_Unit'], errors='coerce') * pd.to_numeric(excel_df_ops['Quantity'], errors='coerce')
        excel_df_ops['Total_Proc_Min'] = pd.to_numeric(excel_df_ops['Total_Proc_Min'], errors='coerce').fillna(0)
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
        
        # If no eligible machines were found based on operation type mapping,
        # fall back to assigning each operation to all machines (using speed factors)
        # so the scheduler can still attempt to schedule uploaded Excel jobs.
        if excel_df_effective.empty:
            fallback_times = []
            machine_ids = excel_df_machines['Machine_ID'].unique().tolist()
            print(f"[Excel Schedule] No eligible machines found via op-type mapping. Falling back to all machines: {machine_ids}")
            for idx, op in excel_df_ops.iterrows():
                for machine_id in machine_ids:
                    machine_row = excel_df_machines[excel_df_machines['Machine_ID'] == machine_id]
                    if len(machine_row) == 0:
                        continue
                    speed_factor = machine_row.iloc[0].get('Speed_Factor', 1.0)
                    try:
                        effective_proc_time = op['Total_Proc_Min'] / float(speed_factor) if float(speed_factor) != 0 else op['Total_Proc_Min']
                    except Exception:
                        effective_proc_time = op['Total_Proc_Min']
                    total_time = op.get('Setup_Time', 0) + effective_proc_time + op.get('Transfer_Min', 0)
                    fallback_times.append({
                        'Operation_ID': op['Operation_ID'],
                        'Machine_ID': machine_id,
                        'Effective_Proc_Time': effective_proc_time,
                        'Total_Time': total_time
                    })
            excel_df_effective = pd.DataFrame(fallback_times)
            print(f"[Excel Schedule] Fallback effective entries created: {len(excel_df_effective)}")
            if excel_df_effective.empty:
                raise HTTPException(
                    status_code=500,
                    detail="No machines available for fallback assignment. Check machine data file."
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
        
        # Merge scheduler output with supporting fields from the Excel ops
        try:
            schedule_df = schedule.rename(columns={'Proc_Time': 'Scheduled_Proc_Time'})
            # Merge supporting fields so Proc_Time, Transfer_Time, Setup_Time reflect original data
            merge_cols = ['Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min', 'Transfer_Min', 'Setup_Time', 'Outsource_Cost']
            available_merge = [c for c in merge_cols if c in excel_df_ops.columns]
            if available_merge:
                schedule_df = schedule_df.merge(excel_df_ops[available_merge], on='Operation_ID', how='left')

            # Ensure Priority and Assignment_Type are present
            if 'Priority' in schedule_df.columns:
                schedule_df['Priority'] = schedule_df['Priority'].fillna(3).astype(int)
            else:
                schedule_df['Priority'] = 3
            if 'Assignment_Type' in schedule_df.columns:
                schedule_df['Assignment_Type'] = schedule_df['Assignment_Type'].fillna('IN_HOUSE')
            else:
                schedule_df['Assignment_Type'] = 'IN_HOUSE'

            # Use Total_Proc_Min as the canonical Proc_Time for display/metrics (shows the real work amount)
            if 'Total_Proc_Min' in schedule_df.columns:
                schedule_df['Proc_Time'] = pd.to_numeric(schedule_df['Total_Proc_Min'], errors='coerce').fillna(0)
            else:
                # Fallback to Scheduled_Proc_Time where available
                schedule_df['Proc_Time'] = pd.to_numeric(schedule_df.get('Scheduled_Proc_Time', 0), errors='coerce').fillna(0)

            # Safety: if any Proc_Time are still zero but Total_Proc_Min exists elsewhere, prefer that
            try:
                if 'Total_Proc_Min' in schedule_df.columns:
                    mask_zero = schedule_df['Proc_Time'] == 0
                    schedule_df.loc[mask_zero, 'Proc_Time'] = pd.to_numeric(schedule_df.loc[mask_zero, 'Total_Proc_Min'], errors='coerce').fillna(0)
            except Exception:
                pass

            # Ensure Transfer_Time shows any provided Transfer_Min when scheduler left it as 0 for outsource
            if 'Transfer_Min' in schedule_df.columns:
                if 'Transfer_Time' in schedule_df.columns:
                    schedule_df['Transfer_Time'] = schedule_df['Transfer_Time'].fillna(schedule_df['Transfer_Min'])
                else:
                    schedule_df['Transfer_Time'] = schedule_df['Transfer_Min']

            # Ensure Setup_Time is present
            if 'Setup_Time' in schedule_df.columns:
                schedule_df['Setup_Time'] = schedule_df['Setup_Time'].fillna(0)
            else:
                schedule_df['Setup_Time'] = 0

        except Exception:
            schedule_df = schedule.copy()

        # Calculate metrics (ISOLATED - NO GLOBAL STATE MODIFICATION) using merged schedule
        metrics = calculate_metrics(schedule_df, excel_df_ops, heuristic)
        print(f"[Excel Schedule] Metrics calculated: {metrics}")
        
        # DO NOT store in global state - Excel upload is completely isolated
        # This keeps Dashboard data and Excel data separate
        
        return {
            "status": "success",
            "message": f"Scheduled {len(jobs)} jobs using {heuristic} (Excel data - isolated from Dashboard)",
            "job_count": len(jobs),
            "heuristic": heuristic,
            "schedule": schedule_df.to_dict('records'),
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
