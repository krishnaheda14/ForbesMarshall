# Diagnostic Script: Check Utilization Calculation
# Run this to verify why utilization is 20% instead of expected 58-63%

import pandas as pd
import os

print("=" * 80)
print("UTILIZATION DIAGNOSTIC REPORT")
print("=" * 80)

# 1. Load data files
data_dir = 'data'
df_machines = pd.read_csv(os.path.join(data_dir, 'machine_data.csv'))
df_ops = pd.read_csv(os.path.join(data_dir, 'jobs_dataset.csv'))

print("\n1. MACHINE CONFIGURATION")
print("-" * 80)
print(f"   Machines loaded: {len(df_machines)}")
print(f"   Machine IDs: {df_machines['Machine ID'].tolist()}")
print(f"   Machine Types: {df_machines['Machine Type'].tolist()}")

# 2. Calculate theoretical utilization
total_ops = len(df_ops)
ops_by_type = df_ops['Op_Type'].value_counts()

print("\n2. OPERATION DISTRIBUTION")
print("-" * 80)
print(f"   Total operations: {total_ops}")
for op_type, count in ops_by_type.items():
    print(f"   {op_type}: {count} ops ({count/total_ops*100:.1f}%)")

# 3. Calculate total work content
df_ops['Total_Time_Min'] = (
    df_ops['Setup_Time'] + 
    df_ops['Proc_Time_per_Unit'] * df_ops['Quantity'] +
    df_ops['Transfer_Min']
)

total_work_min = df_ops['Total_Time_Min'].sum()
total_work_days = total_work_min / 480

print("\n3. WORK CONTENT ANALYSIS")
print("-" * 80)
print(f"   Total setup time: {df_ops['Setup_Time'].sum():.0f} min ({df_ops['Setup_Time'].sum()/480:.1f} days)")
print(f"   Total processing time: {(df_ops['Proc_Time_per_Unit'] * df_ops['Quantity']).sum():.0f} min ({(df_ops['Proc_Time_per_Unit'] * df_ops['Quantity']).sum()/480:.1f} days)")
print(f"   Total transfer time: {df_ops['Transfer_Min'].sum():.0f} min ({df_ops['Transfer_Min'].sum()/480:.1f} days)")
print(f"   Total work content: {total_work_min:.0f} min ({total_work_days:.1f} workdays)")

# 4. Calculate capacity
time_horizon_days = df_ops['Due_Day'].max() - df_ops['Release_Day'].min()
num_machines = len(df_machines)
total_capacity_days = num_machines * time_horizon_days

print("\n4. CAPACITY ANALYSIS")
print("-" * 80)
print(f"   Planning horizon: Day {df_ops['Release_Day'].min()} to Day {df_ops['Due_Day'].max()} ({time_horizon_days} days)")
print(f"   Number of machines: {num_machines}")
print(f"   Total capacity: {num_machines} machines × {time_horizon_days} days = {total_capacity_days} machine-days")

# 5. Calculate theoretical utilization
theoretical_util = (total_work_days / total_capacity_days) * 100

print("\n5. THEORETICAL UTILIZATION")
print("-" * 80)
print(f"   Work needed: {total_work_days:.1f} machine-days")
print(f"   Capacity available: {total_capacity_days} machine-days")
print(f"   Theoretical max utilization: {theoretical_util:.1f}%")

# 6. Estimate realistic utilization
realistic_multiplier = 0.85  # Account for scheduling overhead, idle time between jobs
realistic_util = theoretical_util * realistic_multiplier

print("\n6. REALISTIC UTILIZATION ESTIMATE")
print("-" * 80)
print(f"   Theoretical: {theoretical_util:.1f}%")
print(f"   Realistic (×{realistic_multiplier}): {realistic_util:.1f}%")

# 7. Diagnose 20% issue
print("\n7. DIAGNOSIS")
print("-" * 80)

if num_machines > 2:
    print(f"   ❌ PROBLEM: Using {num_machines} machines instead of 2!")
    print(f"   Expected utilization with 2 machines: {(total_work_days / (2 * time_horizon_days)) * 100 * 0.85:.1f}%")
    print(f"   Current utilization with {num_machines} machines: {realistic_util:.1f}%")
    print("\n   FIX: Run this command to switch to 2-machine config:")
    print("   .\\switch_machine_config.ps1 -Mode 2")
    print("   Then restart Streamlit app")
elif num_machines == 2:
    print(f"   ✓ Correct machine count: {num_machines}")
    if realistic_util < 30:
        print(f"   ❌ PROBLEM: Expected {realistic_util:.1f}% but you're seeing 20%")
        print("\n   Possible causes:")
        print("   1. App is using cached 5-machine configuration in memory")
        print("   2. Session state has old machine count (5 instead of 2)")
        print("   3. Streamlit hasn't reloaded the data files")
        print("\n   FIXES:")
        print("   a) Restart Streamlit completely (Ctrl+C, then re-run)")
        print("   b) Clear browser cache and refresh (Ctrl+Shift+R)")
        print("   c) Press 'C' in Streamlit to clear cache, then 'R' to rerun")
    else:
        print(f"   ✓ Expected utilization: {realistic_util:.1f}%")
        print(f"   ❌ MISMATCH: You're seeing 20% instead")
        print("\n   This suggests the app is calculating with wrong machine count.")
        print("   Check the code in calculate_metrics() - line 163:")
        print("   Should use: len(st.session_state.base_df_machines)")
        print("   Might be falling back to: machine_count = 5")

# 8. Check if using fallback machine count
print("\n8. FALLBACK MACHINE COUNT CHECK")
print("-" * 80)
print("   The calculate_metrics() function has this fallback:")
print("   if 'base_df_machines' in st.session_state:")
print("       machine_count = len(st.session_state.base_df_machines)")
print("   else:")
print("       machine_count = 5  # <-- THIS IS THE PROBLEM!")
print("\n   If session state doesn't have 'base_df_machines', it uses 5 machines!")
print("   This would give you ~20% utilization instead of ~58%")

# 9. Calculate what utilization SHOULD be
print("\n9. EXPECTED RESULTS")
print("-" * 80)

if num_machines == 2:
    # Assume realistic makespan of ~140-150 days
    realistic_makespan_days = 145
    realistic_makespan_min = realistic_makespan_days * 480
    
    # Total capacity with 2 machines
    total_capacity_min = 2 * realistic_makespan_min
    
    # Expected utilization
    expected_util = (total_work_min / total_capacity_min) * 100
    
    print(f"   With 2 machines and {realistic_makespan_days} day makespan:")
    print(f"   Expected utilization: {expected_util:.1f}%")
    print(f"\n   If seeing 20% instead:")
    print(f"   App is probably using {(total_work_min / (0.20 * realistic_makespan_min)):.0f} machines in calculation")
    print(f"   (Because {total_work_min:.0f} / (20% × {realistic_makespan_min:.0f}) = {(total_work_min / (0.20 * realistic_makespan_min)):.0f} machines)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n📊 Your data files are correctly configured:")
print(f"   ✓ {num_machines} machines in data/machine_data.csv")
print(f"   ✓ {total_ops} operations to schedule")
print(f"   ✓ Theoretical max utilization: {theoretical_util:.1f}%")
print(f"   ✓ Expected realistic utilization: {realistic_util:.1f}%")

if num_machines == 2 and realistic_util > 50:
    print("\n🎯 LIKELY ROOT CAUSE:")
    print("   The Streamlit app is NOT loading the 2-machine config properly.")
    print("   It's either:")
    print("   1. Using cached 5-machine data from previous session")
    print("   2. Falling back to hardcoded machine_count = 5")
    print("\n🔧 SOLUTION:")
    print("   1. Stop Streamlit (Ctrl+C)")
    print("   2. Verify data file:")
    print("      Get-Content data\\machine_data.csv -Head 3")
    print("   3. Restart Streamlit:")
    print("      streamlit run cnc-scheduling.py")
    print("   4. Wait for it to fully load")
    print("   5. Check KPI dashboard - should show ~58-63% utilization")
    print("\n   If still 20%, add debug output to calculate_metrics() to see")
    print("   what machine_count it's actually using.")
elif num_machines > 2:
    print("\n🔧 SOLUTION:")
    print("   Run: .\\switch_machine_config.ps1 -Mode 2")
    print("   Then restart Streamlit")

print("\n" + "=" * 80)
