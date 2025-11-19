# Machine Breakdown Testing Guide

## What Was Changed

### 1. Frontend - Increased Breakdown Time Range
**File**: `frontend/src/components/MachineryControls.jsx`

**Changes**:
- Breakdown Start Time: `0 - 100,000 minutes` (was 0-10,000)
- Breakdown Duration: `30 - 5,000 minutes` (was 30-1,000)
- Added step increments for smoother control (100 min steps for start time, 30 min for duration)
- Changed default values: start=1000, duration=240

**Why**: The Gantt chart shows schedules that can extend beyond 10,000 minutes, so breakdowns need to be schedulable at any point in the timeline.

### 2. Backend - Removed Duration Validation
**File**: `backend/main.py`

**Changes**:
- Removed the 480-minute (8-hour shift) validation limit
- Breakdowns can now be any duration from 30 minutes to unlimited

**Why**: Different machines may have different shift patterns, and extended maintenance may be needed.

### 3. Breakdown Visualization in Gantt Chart
**File**: `frontend/src/pages/GanttView.jsx`

**Features**:
- Blue solid bars = Scheduled operations
- Red dotted bars = Machine breakdowns/maintenance
- Hover shows breakdown details (machine, start, end, duration)

**How It Works**:
1. Fetches maintenance windows from `/api/data/machines` endpoint
2. Parses `Maintenance_Window` field from machine data
3. Renders as red dotted lines on the same timeline as operations

## Testing Instructions

### Test 1: Verify Breakdown Time Range
1. **Setup**:
   - Load data
   - Compute all heuristics
   - Check the Gantt chart to see the actual schedule timeline

2. **Add Breakdown**:
   - Open "Machine Breakdown" accordion in sidebar
   - Try setting start time to a large value (e.g., 15,000 minutes)
   - Set duration (e.g., 500 minutes)
   - Click "Simulate Breakdown"

3. **Expected Result**:
   - Success message: "Breakdown simulated. Recompute heuristics to see impact."
   - No error about exceeding limits

### Test 2: Verify Breakdown Appears in Gantt Chart
1. **Add a Breakdown**:
   - Machine ID: `M1`
   - Start Time: `1000` minutes
   - Duration: `300` minutes
   - Click "Simulate Breakdown"

2. **Recompute Heuristics**:
   - Click "Compute All Heuristics" button
   - Wait for completion

3. **View Gantt Chart**:
   - Navigate to "Gantt Chart" page
   - Look for red dotted bar on M1 timeline from 1000-1300 minutes

4. **Expected Result**:
   - Red dotted line visible on M1's row
   - Hover shows: "Machine: M1, Type: Breakdown/Maintenance, Start: 1000, End: 1300, Duration: 300"
   - Blue operation bars should NOT overlap with the red breakdown bar

### Test 3: Verify No Operations During Breakdown
1. **Check Schedule Data**:
   - After adding breakdown and recomputing
   - Examine the operations scheduled on M1
   - Look at their start and end times

2. **Expected Result**:
   - NO operation on M1 should have:
     - Start time < 1300 AND End time > 1000 (overlap with 1000-1300 breakdown)
   - Operations should be scheduled either:
     - BEFORE 1000 (ends before breakdown starts)
     - AFTER 1300 (starts after breakdown ends)

3. **Visual Verification**:
   - In Gantt chart, blue bars on M1 should have a gap where the red breakdown bar is
   - No blue bars should overlap with red bars

### Test 4: Multiple Breakdowns on Same Machine
1. **Add First Breakdown**:
   - Machine: M1, Start: 1000, Duration: 300

2. **Add Second Breakdown**:
   - Machine: M1, Start: 5000, Duration: 400

3. **Recompute and Check Gantt**:
   - Should see TWO red dotted bars on M1
   - One at 1000-1300
   - One at 5000-5400

4. **Expected Result**:
   - Both breakdowns visible
   - Operations avoid BOTH breakdown windows

### Test 5: Breakdown on Different Machines
1. **Add Breakdowns**:
   - Machine M1: Start: 1000, Duration: 300
   - Machine M2: Start: 2000, Duration: 400
   - Machine M3: Start: 1500, Duration: 200

2. **View Gantt**:
   - Should see red bars on different machine rows
   - Each at their respective time ranges

3. **Expected Result**:
   - Breakdowns appear on correct machine rows
   - Operations on each machine avoid their respective breakdowns
   - Operations can still run on M2 during M1's breakdown (no global halt)

## How the Scheduler Respects Breakdowns

### Backend Logic: `cnc_scheduler_core.py`

The `get_earliest_available_time()` method handles breakdown avoidance:

```python
def get_earliest_available_time(self, machine_id, release_time, duration):
    # Gets maintenance windows from df_machines['Maintenance_Window']
    # Checks if proposed time slot overlaps with any maintenance window
    # If overlap detected, moves start time to after the maintenance window
    # Iterates until a non-overlapping slot is found
```

**Key Points**:
- Reads `Maintenance_Window` field which contains breakdown info
- Format: `{'start': X, 'end': Y, 'duration': Z}`
- Automatically shifts operations to avoid conflicts

### Visual Indicators

| Element | Color | Style | Meaning |
|---------|-------|-------|---------|
| Operation Bar | Blue | Solid | Scheduled job operation |
| Breakdown Bar | Red | Dotted | Machine unavailable (maintenance/breakdown) |

## Common Issues & Solutions

### Issue 1: Breakdown Not Visible in Gantt
**Symptoms**: Red bar doesn't appear after adding breakdown

**Solutions**:
1. Make sure you clicked "Simulate Breakdown" and got success message
2. **Recompute heuristics** - breakdown only affects NEW schedules
3. Refresh the Gantt view page (navigate away and back)
4. Check if breakdown time is within visible timeline range

### Issue 2: Operations Still Overlap Breakdown
**Symptoms**: Blue bars overlap with red breakdown bars

**Possible Causes**:
1. Schedule was computed BEFORE breakdown was added
2. Need to recompute heuristics after adding breakdown
3. Check if breakdown was added to correct machine ID

**Solution**: Always recompute after adding breakdowns!

### Issue 3: Breakdown Time Slider Too Sensitive
**Symptoms**: Hard to set exact time values

**Solutions**:
1. Use keyboard arrows for fine control
2. Click on slider track to jump to approximate position
3. Or modify the machine data CSV directly for precise control

## Excel Upload Feature - Scheduler Integration

### Step 3: Schedule Jobs (NEW)

After successfully transforming Excel data, you now see:

**4 Scheduling Algorithm Buttons**:
1. **SPT (Green)** - Shortest Processing Time
   - Minimizes average completion time
   - Best for maximizing throughput

2. **EDD (Blue)** - Earliest Due Date
   - Reduces tardiness
   - Best when meeting deadlines is critical

3. **CR (Orange)** - Critical Ratio
   - Balances urgency and processing time
   - Best for mixed objectives

4. **PRIORITY (Purple)** - Priority-Based
   - Respects job importance levels
   - Best when some jobs are more critical

**What Happens When You Click**:
1. Validates transformed jobs
2. Applies selected scheduling algorithm
3. Shows success message with job count
4. Redirects to Dashboard (main project view)

**Note**: Current implementation validates and confirms compatibility. Full integration with main scheduler pending.

## Summary of Improvements

✅ **Breakdown Time Range**: Now 0-100,000 minutes (up from 10,000)  
✅ **Duration Range**: Now 30-5,000 minutes (up from 1,000)  
✅ **Validation**: Removed restrictive 480-minute limit  
✅ **Visualization**: Red dotted bars show breakdowns in Gantt  
✅ **Scheduler Logic**: Already respects maintenance windows  
✅ **Excel Import**: Now includes scheduler algorithm selection  
✅ **User Experience**: Clear visual distinction between operations and breakdowns  

## Next Steps

1. Test all scenarios above
2. Verify operations never overlap breakdowns
3. Test with your actual production data
4. Fine-tune time ranges if needed
5. Complete Excel-to-scheduler integration for full workflow
