# data_loader/loader.py
"""
Data loading and preprocessing
"""
import pandas as pd
import numpy as np
import streamlit as st
from utils.helpers import (
    parse_maintenance,
    get_eligible_machines,
    make_or_buy_decision
)


@st.cache_data
def load_all_data(sample_size=None, cost_threshold=0.9, hourly_rate=30, _cache_version=3):
    """
    Load and preprocess all data with make-or-buy decisions.
    Cache key includes cost_threshold so changing it triggers recalculation.
    """
    try:
        df_ops = pd.read_csv("data/jobs_dataset.csv")
        df_vendors = pd.read_csv("data/vendor_data.csv")
        df_machines = pd.read_csv("data/machine_data.csv")
        df_penalties = pd.read_csv("data/previous_next_material.csv")
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure the 'data' folder exists")
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
        st.success(f"✅ Fixed {len(deadline_issues)} deadline issues")
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

    # Calculate effective processing times
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

    # Process vendor data
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

    # Make-or-buy decisions
    decisions = []
    for idx, op in df_ops.iterrows():
        decision, cost, reason = make_or_buy_decision(op, df_effective, cost_threshold=cost_threshold, hourly_rate=hourly_rate)
        decisions.append({'Operation_ID': op['Operation_ID'], 'Decision': decision, 'Reason': reason, 'Cost': cost})

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
        st.warning(f"⚠️ **HIGH OUTSOURCING**: {outsource_pct:.0f}% outsourced!")
        st.write("   💡 Consider: Lowering cost threshold or adding capacity")

    return df_ops, df_machines, df_effective, df_penalties, df_vendors
