# app.py - MODULAR VERSION
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

# ============================================
# MODULAR IMPORTS - New Architecture
# ============================================
from core import CNCScheduler
from utils import (
    parse_maintenance,
    get_eligible_machines,
    get_setup_penalty,
    calculate_inhouse_cost,
    make_or_buy_decision,
    calculate_metrics,
    check_breakdown_conflicts,
    dbg as utils_dbg,
    safe_toast as utils_safe_toast
)
from data_loader import load_all_data as modular_load_all_data

def trigger_recompute_prompt(ss, label: str, redirect_to_comparison=True):
    """
    Unified helper to handle post-update behavior for:
    - Machine Breakdown
    - Priority Update
    - Outsourcing Policy Update
    
    Args:
        ss: Session state
        label: Label for success message
        redirect_to_comparison: If True, redirect to comparison page; if False, stay on current page
    """
    # Display user-facing success + guidance
    st.success(f"✅ {label} completed successfully.")
    
    # Check if heuristic is already applied
    if not ss.get('current_heuristic'):
        st.warning("💡 No heuristic applied yet. Please click **'🧪 Compute All Heuristics'** and **'Apply Selected Heuristic'** to see updated schedule.")
    else:
        st.info("💡 Please click **'🧪 Compute All Heuristics'** in the sidebar "
                "to recompute schedules and view updated recommendations.")

    # Set session state to prepare heuristic recomputation
    ss.recalculate_all_heuristics = True
    ss.breakdown_message_visible = True
    
    # Only redirect to comparison if requested
    if redirect_to_comparison:
        ss.current_page = "comparison"

    # Clear Streamlit caches to avoid stale data
    st.cache_data.clear()
    st.cache_resource.clear()

    # Small visual feedback (guarded)
    safe_toast("⚙ Update registered — ready for heuristic recomputation.", icon="⚙️")

# ---------------------------
# Helper functions - using modular versions
# ---------------------------
def dbg(msg):
    utils_dbg(msg)

def safe_toast(message, icon=None):
    utils_safe_toast(message, icon)

# ---------------------------
# Data loading - using modular version
# ---------------------------
@st.cache_data
def load_all_data(sample_size=None, cost_threshold=0.9, hourly_rate=30, _cache_version=3):
    """
    Wrapper for modular load_all_data function.
    Returns: (df_ops, df_machines, df_effective, df_penalties, df_vendors)
    """
    return modular_load_all_data(sample_size, cost_threshold, hourly_rate, _cache_version)

# TODO: Add remaining UI functions and logic here
# This is a starter template - copy over UI functions from original file
# while using the modular imports above

if __name__ == "__main__":
    st.title("🏭 CNC Scheduling - MODULAR VERSION")
    st.info("This is the new modular version. Original file preserved as cnc-scheduling.py")
