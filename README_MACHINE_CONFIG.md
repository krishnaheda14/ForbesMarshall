# CNC Scheduler - Machine Configuration Guide

## Current Configuration: 2 Machines (High Utilization)

### Active Machines:
- **M1**: MILLING (handles MILLING + DRILLING operations)
- **M6**: TURNING/GRINDING (handles TURNING + GRINDING operations)

### Expected Performance:
- **Utilization**: ~58-63% (realistic for job shops)
- **Makespan**: Longer timeline (fewer machines = more sequential work)
- **On-Time %**: May be slightly lower due to capacity constraint

---

## How to Switch Configurations

### Option 1: Using PowerShell Script (Recommended)

```powershell
# Switch to 2-machine (high utilization)
.\switch_machine_config.ps1 -Mode 2

# Switch to 5-machine (low utilization)
.\switch_machine_config.ps1 -Mode 5

# Alternative syntax
.\switch_machine_config.ps1 -Mode high
.\switch_machine_config.ps1 -Mode low
```

### Option 2: Manual Copy

```powershell
# For 2 machines (high util):
Copy-Item "data\machine_data_high_util.csv" "data\machine_data.csv" -Force

# For 5 machines (low util):
Copy-Item "data\machine_data_original_5_machines.csv" "data\machine_data.csv" -Force
```

---

## IMPORTANT: Code Changes Required

After switching machine configurations, you **MUST** update the `get_eligible_machines()` function in `cnc-scheduling.py`:

### For 2-Machine Config (Current):
```python
def get_eligible_machines(op_type):
    if op_type == 'MILLING':
        return ['M1']
    elif op_type == 'TURNING':
        return ['M6']
    elif op_type == 'GRINDING':
        return ['M6']
    elif op_type == 'DRILLING':
        return ['M1']
    else:
        return []
```

### For 5-Machine Config:
```python
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
```

---

## Comparison

| Metric | 2 Machines | 5 Machines |
|--------|------------|------------|
| **Machines** | M1, M6 | M1, M3, M4, M6, M9 |
| **Capacity** | 200 machine-days | 500 machine-days |
| **Theoretical Util** | 68.2% | 27.3% |
| **Real Util** | ~58-63% | ~10-15% |
| **Use Case** | Realistic job shop | Excess capacity demo |
| **Makespan** | Longer | Shorter |
| **Bottleneck** | High (M1 or M6) | Low |

---

## Files in data/ folder

- `machine_data.csv` - **Active configuration** (currently 2 machines)
- `machine_data_high_util.csv` - 2-machine configuration
- `machine_data_original_5_machines.csv` - Original 5-machine configuration

---

## Troubleshooting

### Error: "single positional indexer is out-of-bounds"
**Cause**: `get_eligible_machines()` is still configured for 5 machines but only 2 exist in data  
**Fix**: Update `get_eligible_machines()` function to match active configuration

### Low Utilization with 2 Machines
**Check**:
1. Is `machine_data.csv` actually using 2 machines? 
2. Is `get_eligible_machines()` returning correct machines?
3. Clear Streamlit cache and reload

### High Utilization with 5 Machines
**This is impossible** - theoretical max is 27.3%. If you see high util with 5 machines, check:
1. Machine data was actually loaded correctly
2. Operations are being scheduled (not all outsourced)

---

## Recommendations

- **For Demos/Presentations**: Use 2-machine config (realistic ~60% utilization)
- **For Testing Algorithms**: Use 5-machine config (shows scheduling differences better)
- **For Production**: Adjust based on actual shop floor capacity

---

Last Updated: November 6, 2025
