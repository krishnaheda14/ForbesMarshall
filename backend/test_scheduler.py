# Direct scheduler test
import sys
sys.path.append('..')

import pandas as pd
from cnc_scheduler_core import CNCScheduler, calculate_metrics, get_eligible_machines

print("🧪 Testing Scheduler Directly...")
print("=" * 60)

# Create simple test data
jobs_data = [
    {
        'Job_ID': 'J001',
        'Operation_ID': 'J001_Op1',
        'Op_Seq': 1,
        'Quantity': 10,
        'Proc_Time_per_Unit': 30,
        'Total_Proc_Min': 300,
        'Setup_Time': 5,
        'Due_Day': 10,
        'Due_Time_Min': 4800,
        'Priority': 1,
        'Mat_Type': 'STEEL',
        'Tool_Group': 'TGA',
        'Op_Type': 'MILLING',
        'Part_Type': 'A',
        'Transfer_Min': 5,
        'Release_Day': 0,
        'Release_Time_Min': 0,
    }
]

df_ops = pd.DataFrame(jobs_data)
print(f"✅ Created {len(df_ops)} operations")
print(df_ops[['Operation_ID', 'Op_Type', 'Total_Proc_Min']])

# Load machine data
df_machines = pd.read_csv('../data/machine_data.csv')
df_machines.columns = df_machines.columns.str.replace(' ', '_')
print(f"✅ Loaded {len(df_machines)} machines")

# Load penalties
df_penalties = pd.read_csv('../data/previous_next_material.csv')
df_penalties.columns = df_penalties.columns.str.replace(' ', '_')
print(f"✅ Loaded {len(df_penalties)} penalty rules")

# Calculate effective times
print(f"\n📊 Calculating effective times...")
eligible_machines = get_eligible_machines('MILLING')
print(f"Eligible machines for MILLING: {eligible_machines}")

effective_times = []
for _, op in df_ops.iterrows():
    op_type = op['Op_Type']
    machines = get_eligible_machines(op_type)
    
    for machine_id in machines:
        machine_row = df_machines[df_machines['Machine_ID'] == machine_id]
        if len(machine_row) > 0:
            # Get speed factor
            if 'Speed_Factor' in df_machines.columns:
                speed_str = str(machine_row.iloc[0]['Speed_Factor'])
                import re
                match = re.search(r'([0-9]*\.?[0-9]+)', speed_str)
                speed_factor = float(match.group(1)) if match else 1.0
            else:
                speed_factor = 1.0
            
            effective_proc_time = op['Total_Proc_Min'] / speed_factor
            total_time = op['Setup_Time'] + effective_proc_time + op['Transfer_Min']
            
            effective_times.append({
                'Operation_ID': op['Operation_ID'],
                'Machine_ID': machine_id,
                'Effective_Proc_Time': effective_proc_time,
                'Total_Time': total_time
            })

df_effective = pd.DataFrame(effective_times)
print(f"✅ Created {len(df_effective)} effective time entries")
print(df_effective)

# Run scheduler
print(f"\n🔄 Running scheduler...")
try:
    scheduler = CNCScheduler(
        df_ops=df_ops,
        df_machines=df_machines,
        df_effective=df_effective,
        df_penalties=df_penalties
    )
    
    schedule = scheduler.run_scheduling(heuristic='SPT', verbose=True)
    
    print(f"\n📋 Schedule type: {type(schedule)}")
    print(f"Schedule shape: {schedule.shape if hasattr(schedule, 'shape') else 'N/A'}")
    
    if isinstance(schedule, pd.DataFrame) and not schedule.empty:
        print(f"✅ Schedule created successfully!")
        print(schedule)
        
        # Calculate metrics
        metrics = calculate_metrics(schedule, df_ops, 'SPT')
        print(f"\n📊 Metrics: {metrics}")
    else:
        print(f"❌ Schedule is invalid: {schedule}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
