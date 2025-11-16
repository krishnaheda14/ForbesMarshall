"""
Integration test - verify modular components work with main app
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("INTEGRATION TEST - Modular Components with Main App")
print("=" * 80)

# Test 1: Import all modular components
print("\n[TEST 1] Importing modular components...")
try:
    from core import CNCScheduler
    from utils import (
        parse_maintenance,
        get_eligible_machines,
        get_setup_penalty,
        calculate_inhouse_cost,
        make_or_buy_decision,
        calculate_metrics,
        check_breakdown_conflicts,
        dbg,
        safe_toast
    )
    from data_loader import load_all_data
    print("✅ All modular imports successful")
except Exception as e:
    print(f"❌ Module import failed: {e}")
    sys.exit(1)

# Test 2: Verify functions are available
print("\n[TEST 2] Verifying function availability...")
functions = [
    ('parse_maintenance', parse_maintenance),
    ('get_eligible_machines', get_eligible_machines),
    ('CNCScheduler', CNCScheduler),
    ('calculate_metrics', calculate_metrics),
    ('load_all_data', load_all_data)
]

for name, func in functions:
    assert callable(func), f"{name} is not callable"
    print(f"✅ {name} - available")

# Test 3: Test basic functionality
print("\n[TEST 3] Testing basic functionality...")

# Test parse_maintenance
maintenance_str = "Day 3, 10:00-12:00"
result = parse_maintenance(maintenance_str)
assert result is not None, "parse_maintenance returned None"
assert 'start' in result, "Missing start key"
print(f"✅ parse_maintenance: {maintenance_str} → start={result['start']}, end={result['end']}")

# Test get_eligible_machines  
milling_machines = get_eligible_machines('MILLING')
assert 'M1' in milling_machines, "M1 should be eligible for MILLING"
print(f"✅ get_eligible_machines('MILLING'): {milling_machines}")

# Test 4: Verify CNCScheduler class structure
print("\n[TEST 4] Verifying CNCScheduler class structure...")
import pandas as pd

# Create minimal test data
test_ops = pd.DataFrame([{
    'Operation_ID': 'TEST1',
    'Job_ID': 'J1',
    'Op_Seq': 1,
    'Assignment_Type': 'IN_HOUSE',
    'Total_Proc_Min': 30
}])

test_machines = pd.DataFrame([{
    'Machine_ID': 'M1',
    'Maintenance_Window': None
}])

test_effective = pd.DataFrame([{
    'Operation_ID': 'TEST1',
    'Machine_ID': 'M1',
    'Effective_Proc_Time': 30,
    'Setup_Time': 10,
    'Transfer_Min': 5,
    'Total_Time': 45
}])

test_penalties = pd.DataFrame([{
    'Previous Material': 'Steel',
    'Next Material': 'Aluminum',
    'Penalty Time (min)': 15
}])

try:
    scheduler = CNCScheduler(test_ops, test_machines, test_effective, test_penalties)
    assert hasattr(scheduler, 'run_scheduling'), "Missing run_scheduling method"
    assert hasattr(scheduler, 'get_earliest_available_time'), "Missing get_earliest_available_time method"
    assert hasattr(scheduler, 'schedule_operation'), "Missing schedule_operation method"
    print("✅ CNCScheduler class structure verified")
    print(f"   - Machine availability: {scheduler.machine_availability}")
    print(f"   - Methods: run_scheduling, get_earliest_available_time, schedule_operation")
except Exception as e:
    print(f"❌ CNCScheduler verification failed: {e}")
    sys.exit(1)

# Test 5: Test metrics calculation
print("\n[TEST 5] Testing metrics calculation...")

test_schedule = pd.DataFrame([{
    'Operation_ID': 'TEST1',
    'Job_ID': 'J1',
    'Machine_ID': 'M1',
    'Start_Time': 0,
    'End_Time': 45,
    'Setup_Time': 10,
    'Proc_Time': 30,
    'Transfer_Time': 5,
    'Due_Time': 100,
    'Tardiness': 0
}])

metrics = calculate_metrics(test_schedule, test_ops, 'TEST')
assert metrics['Heuristic'] == 'TEST', "Wrong heuristic name"
assert metrics['Makespan_Days'] > 0, "Makespan should be > 0"
assert metrics['On_Time_%'] == 100, "Should be 100% on-time"
print("✅ Metrics calculation working")
print(f"   - Makespan: {metrics['Makespan_Days']} days")
print(f"   - On-Time: {metrics['On_Time_%']}%")
print(f"   - Utilization: {metrics['Machine_Utilization_%']}%")

# Test 6: Test conflict detection
print("\n[TEST 6] Testing breakdown conflict detection...")

conflict_schedule = pd.DataFrame([{
    'Operation_ID': 'CONFLICT',
    'Job_ID': 'J2',
    'Machine_ID': 'M1',
    'Start_Time': 100,
    'End_Time': 200
}])

conflict_machines = pd.DataFrame([{
    'Machine_ID': 'M1',
    'Maintenance_Window': {'start': 150, 'end': 250, 'duration': 100}
}])

conflicts = check_breakdown_conflicts(conflict_schedule, conflict_machines)
assert len(conflicts) > 0, "Should detect conflict"
assert conflicts[0]['Overlap_Minutes'] > 0, "Overlap should be > 0"
print(f"✅ Conflict detection working")
print(f"   - Detected {len(conflicts)} conflict(s)")
print(f"   - Overlap: {conflicts[0]['Overlap_Minutes']} minutes")

# Final Summary
print("\n" + "=" * 80)
print("INTEGRATION TEST SUMMARY")
print("=" * 80)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("\nVerified Components:")
print("  ✅ core.scheduler - CNCScheduler class")
print("  ✅ utils.helpers - Helper functions")
print("  ✅ utils.metrics - Metrics & conflict detection")
print("  ✅ data_loader - Data loading pipeline")
print("\n📊 Modular components are fully integrated and ready to use!")
print("=" * 80)
