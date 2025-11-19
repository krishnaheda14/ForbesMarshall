# Recent Fixes Summary

## Overview
This document summarizes the fixes applied to address issues with Gantt chart refresh, machine breakdown validation, breakdown visualization, and outsourcing threshold KPI updates.

---

## 1. Gantt Chart Auto-Refresh ✅

**Issue**: Gantt chart was not updating when user changed the selected heuristic from the dropdown.

**Root Cause**: 
- `useSchedulerStore.js` - `setCurrentHeuristic()` was not clearing the cached schedule
- `GanttView.jsx` - useEffect had condition `if (currentHeuristic && !currentSchedule)` which prevented refetch when schedule already existed

**Fixes Applied**:

### File: `frontend/src/store/useSchedulerStore.js`
```javascript
// BEFORE
setCurrentHeuristic: (heuristic) => set({ currentHeuristic: heuristic }),

// AFTER
setCurrentHeuristic: (heuristic) => set({ currentHeuristic: heuristic, currentSchedule: null }),
```
**Explanation**: Now when heuristic changes, the schedule is cleared, forcing a refetch.

### File: `frontend/src/pages/GanttView.jsx`
```javascript
// BEFORE
useEffect(() => {
  if (currentHeuristic && !currentSchedule) {
    fetchSchedule();
  }
}, [currentHeuristic]);

// AFTER
useEffect(() => {
  if (currentHeuristic) {
    fetchSchedule();
  }
}, [currentHeuristic]);
```
**Explanation**: Removed the `!currentSchedule` condition so it always refetches when heuristic changes.

---

## 2. Machine Breakdown Time Limits Validation ✅

**Issue**: No validation was enforcing the 8-hour shift limit (480 min/day) for machine breakdowns.

**Fix Applied**:

### File: `backend/main.py` - `/api/machine/breakdown` endpoint
```python
@app.post("/api/machine/breakdown")
def simulate_breakdown(request: MachineBreakdownRequest):
    """Simulate machine breakdown"""
    if state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        # Validate breakdown duration (8-hour shifts = 480 min/day)
        SHIFT_MINUTES = 480
        if request.duration > SHIFT_MINUTES:
            raise HTTPException(
                status_code=400, 
                detail=f"Breakdown duration ({request.duration} min) exceeds single shift limit ({SHIFT_MINUTES} min)"
            )
        
        # ... rest of the function
```

**Result**: Now breakdowns exceeding 480 minutes are rejected with a clear error message.

---

## 3. Breakdown Visualization in Gantt Chart ✅

**Issue**: Machine breakdowns/maintenance windows were not visible in the Gantt chart - only scheduled operations were shown.

**Fixes Applied**:

### File: `frontend/src/services/api.js`
Added new API function to fetch machine data:
```javascript
export const getMachineData = async () => {
  const response = await api.get('/api/data/machines');
  return response.data;
};
```

### File: `backend/main.py`
Added new endpoint to expose machine data with maintenance windows:
```python
@app.get("/api/data/machines")
def get_machine_data():
    """Get machine data including maintenance windows"""
    if state.df_machines is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    machines = state.df_machines.to_dict('records')
    return {"machines": machines}
```

### File: `frontend/src/pages/GanttView.jsx`
Enhanced to fetch and display maintenance windows:

```javascript
// Added state for maintenance data
const [maintenanceData, setMaintenanceData] = useState([]);

// Added maintenance data fetch
const fetchMaintenanceData = async () => {
  try {
    const result = await getMachineData();
    const maintenance = [];
    
    result.machines.forEach((machine) => {
      if (machine.Maintenance_Window) {
        const windows = Array.isArray(machine.Maintenance_Window) 
          ? machine.Maintenance_Window 
          : [machine.Maintenance_Window];
        
        windows.forEach((window) => {
          if (window && window.start !== undefined && window.end !== undefined) {
            maintenance.push({
              machine: machine.Machine_ID,
              start: window.start,
              end: window.end,
              duration: window.duration || (window.end - window.start)
            });
          }
        });
      }
    });
    
    setMaintenanceData(maintenance);
  } catch (error) {
    console.error('Failed to fetch maintenance data:', error);
  }
};

// Added maintenance traces to Gantt chart
const maintenanceTraces = maintenanceData.map((maint) => ({
  x: [maint.start, maint.end],
  y: [maint.machine, maint.machine],
  type: 'line',
  mode: 'lines',
  line: { width: 20, color: '#ef4444', dash: 'dot' },  // Red dotted lines
  name: `Breakdown - ${maint.machine}`,
  hovertemplate:
    `<b>Machine:</b> ${maint.machine}<br>` +
    `<b>Type:</b> Breakdown/Maintenance<br>` +
    `<b>Start:</b> ${maint.start} min<br>` +
    `<b>End:</b> ${maint.end} min<br>` +
    `<b>Duration:</b> ${maint.duration} min<extra></extra>`,
}));

const allTraces = [...ganttData, ...maintenanceTraces];
```

**Result**: 
- Blue solid bars = Scheduled operations
- Red dotted bars = Machine breakdowns/maintenance
- Breakdowns are fetched and displayed alongside scheduled operations

**Scheduler Verification**: 
The `get_earliest_available_time()` function in `cnc_scheduler_core.py` already correctly respects maintenance windows and schedules operations around them.

---

## 4. Outsourcing Threshold KPI Updates ✅

**Issue**: Changing the outsourcing cost threshold updated make-or-buy decisions but didn't recompute KPI metrics, so changes weren't reflected in the dashboard.

**Fixes Applied**:

### File: `backend/main.py` - `/api/outsourcing/policy` endpoint
```python
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
        
        # NEW: Recompute metrics for current heuristic if available
        if state.current_heuristic and state.schedules.get(state.current_heuristic):
            scheduler = CNCScheduler(
                state.df_jobs, state.df_ops, state.df_machines,
                state.df_prev_next, state.cost_threshold
            )
            schedule = state.schedules[state.current_heuristic]
            metrics = scheduler.compute_metrics(schedule)
            state.metrics[state.current_heuristic] = metrics
        
        # ... activity log and return
        
        return {
            "status": "success",
            "message": "Outsourcing policy updated and metrics recomputed",
            "new_outsourced_count": new_outsourced,
            "total_operations": len(state.df_ops),
            "metrics": state.metrics.get(state.current_heuristic, {}) if state.current_heuristic else {}
        }
```

### File: `frontend/src/components/MachineryControls.jsx`
Added automatic schedule refresh after outsourcing update:

```javascript
import useSchedulerStore from '../store/useSchedulerStore';

function MachineryControls() {
  const { enqueueSnackbar } = useSnackbar();
  const { setCurrentSchedule, currentHeuristic } = useSchedulerStore();
  
  // ...

  const handleOutsourcingUpdate = async () => {
    try {
      const result = await updateOutsourcingPolicy(costThreshold);
      enqueueSnackbar(
        `Outsourcing policy updated! ${result.new_outsourced_count}/${result.total_operations} operations outsourced.`, 
        { variant: 'success' }
      );
      
      // NEW: Refresh schedule if a heuristic is active
      if (currentHeuristic) {
        const scheduleResult = await getCurrentSchedule();
        setCurrentSchedule(scheduleResult.schedule);
      }
    } catch (error) {
      enqueueSnackbar(`Error: ${error.response?.data?.detail || error.message}`, {
        variant: 'error',
      });
    }
  };
```

**Result**: 
- Backend now recomputes metrics when threshold changes
- Frontend automatically refreshes the schedule and KPIs
- User sees updated outsourcing counts and metrics immediately
- No need to manually recompute heuristics

---

## Testing Instructions

### 1. Test Gantt Chart Refresh
1. Load data
2. Compute all heuristics
3. Click on "Gantt View" - should show SPT schedule (default)
4. Use heuristic selector dropdown to change to "EDD"
5. **Expected**: Gantt chart should immediately refresh and show EDD schedule
6. Change to "CR", then "PRIORITY" - chart should update each time

### 2. Test Breakdown Time Limits
1. Open "Machine Breakdown" accordion in sidebar
2. Set duration slider to > 480 minutes (e.g., 600)
3. Click "Simulate Breakdown"
4. **Expected**: Error message: "Breakdown duration (600 min) exceeds single shift limit (480 min)"
5. Set duration to <= 480 (e.g., 300)
6. **Expected**: Success message and breakdown is added

### 3. Test Breakdown Visualization
1. Simulate a breakdown (e.g., Machine M1, Start: 1000, Duration: 200)
2. Recompute heuristics
3. Go to Gantt View
4. **Expected**: 
   - Blue solid bars for scheduled operations
   - Red dotted bar showing the breakdown from 1000-1200 min on M1
   - Hover over red bar shows "Breakdown/Maintenance" details
   - Operations should be scheduled around the breakdown (not overlapping)

### 4. Test Outsourcing Threshold KPI Updates
1. Load data and compute all heuristics
2. Note current outsourcing count in KPI cards
3. Open "Outsourcing" accordion
4. Change cost threshold (e.g., from 0.9 to 1.2)
5. Click "Update Policy"
6. **Expected**: 
   - Success message showing new outsourcing count
   - KPI cards update immediately
   - Gantt chart refreshes if viewing one
   - No need to manually recompute

---

## Technical Details

### Frontend Changes
- **Modified Files**: 3
  - `frontend/src/store/useSchedulerStore.js` - Clear schedule on heuristic change
  - `frontend/src/pages/GanttView.jsx` - Auto-refresh & breakdown visualization
  - `frontend/src/components/MachineryControls.jsx` - Auto-refresh after outsourcing update
  - `frontend/src/services/api.js` - Added `getMachineData()` API function

### Backend Changes
- **Modified Files**: 1
  - `backend/main.py`:
    - Added 480-min validation in `/api/machine/breakdown`
    - Added `/api/data/machines` endpoint
    - Enhanced `/api/outsourcing/policy` to recompute metrics

### Core Scheduler
- **No changes needed** - `get_earliest_available_time()` already correctly respects `Maintenance_Window`

---

## Summary of Results

| Issue | Status | Impact |
|-------|--------|--------|
| Gantt chart not refreshing on heuristic change | ✅ Fixed | Immediate visual feedback when switching algorithms |
| No validation for breakdown time limits | ✅ Fixed | Prevents invalid breakdowns > 8-hour shifts |
| Breakdowns not visible in Gantt chart | ✅ Fixed | Users can see both operations AND maintenance windows |
| Outsourcing threshold changes not updating KPIs | ✅ Fixed | Real-time KPI updates without manual recomputation |

All issues have been resolved. The system now provides:
- **Immediate visual feedback** when changing heuristics or policies
- **Data validation** to prevent invalid inputs
- **Complete visibility** of both scheduled operations and maintenance windows
- **Automatic metric updates** when configuration changes
