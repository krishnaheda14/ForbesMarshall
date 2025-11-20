# Testing Checklist - New Features

## System Status
- ✅ Backend: http://localhost:8001 (Running)
- ✅ Frontend: http://localhost:5173 (Running)

---

## Test 1: Professional UI (No Emojis)

### Steps
1. Navigate to http://localhost:5173
2. Check all pages:
   - Dashboard
   - Gantt Chart
   - Comparison
   - Settings
   - Excel Upload
   - Cost Analysis

### Expected Results
- ✅ NO emojis visible anywhere
- ✅ Clean, professional typography
- ✅ All titles and buttons text-only

---

## Test 2: Improved Gantt Chart Visualization

### Steps
1. Load dataset from Dashboard
2. Click "Compute All Heuristics"
3. Select and apply "SPT"
4. Navigate to Gantt Chart page

### Expected Results
- ✅ Page title: "Gantt Chart" (no emoji)
- ✅ Y-axis label: "Job - Operation"
- ✅ Y-axis values: Show "J001-OP01", "J002-OP02" format
- ✅ X-axis label: "Time (days)"
- ✅ X-axis values: Show decimal days (e.g., 0.5, 1.0, 1.5)
- ✅ Chart height: Adjusts to fit all operations
- ✅ Each job has unique color

### Hover Test
1. Hover over any operation bar

### Expected Hover Info
```
Job: J001
Operation: OP01
Machine: M1
Start: 0.25 days (360 min)
End: 0.50 days (720 min)
Duration: 0.25 days (360 min)
Priority: 1
```

---

## Test 3: Machine Breakdown Visualization

### Steps
1. Load dataset
2. Compute all heuristics
3. Navigate to "Machinery Controls" (sidebar)
4. Fill breakdown form:
   - Machine ID: `M1` (select from dropdown)
   - Start Time: `1440` (1 day)
   - Duration: `240` (4 hours)
5. Click "Simulate Breakdown"
6. Navigate to Gantt Chart
7. Click "Refresh" button (top right)

### Expected Results
- ✅ Success message: "Breakdown simulated. Recompute heuristics to see impact."
- ✅ Gantt chart shows:
  - Red dotted line for breakdown
  - Label: "BREAKDOWN-M1-0"
  - Positioned at day 1.0
  - Duration: 0.167 days (240 min)
- ✅ Breakdown count chip: "1 Breakdown"
- ✅ Warning alert: "1 breakdown period(s) detected. These are shown as red dotted lines."

### Hover on Breakdown
Expected info:
```
Type: Breakdown/Maintenance
Machine: M1
Start: 1.00 days (1440 min)
End: 1.17 days (1680 min)
Duration: 0.17 days (240 min)
```

---

## Test 4: Data Integrity After Breakdown

### Steps
1. Load dataset → Note operation count (e.g., 50 operations)
2. Compute all heuristics
3. Check SPT metrics → Note makespan
4. Add machine breakdown (as above)
5. Go to "Compute Controls"
6. Click "Compute All Heuristics" AGAIN
7. Check SPT metrics again
8. Navigate to "Operation Status" page

### Expected Results
- ✅ New SPT makespan is DIFFERENT (increased due to breakdown)
- ✅ New schedule routes operations around breakdown
- ✅ Operation count UNCHANGED (still 50 operations)
- ✅ No operations lost or duplicated
- ✅ All operations visible in Operation Status table
- ✅ Original `df_ops` data intact

### Verify on Gantt Chart
- ✅ Operations shifted to avoid breakdown window
- ✅ Red dotted breakdown line still visible
- ✅ No operations overlap with breakdown

---

## Test 5: Excel Upload with New Gantt Chart

### Steps
1. Navigate to Settings → Excel Upload
2. Create/use test Excel file with columns:
   - Job_ID, Operation_ID, Part_Type, Operation_Type
   - Processing_Time, Setup_Time, Quantity, Material
   - Priority, Release_Day, Due_Date, Tool_Group
3. Upload file
4. Review auto-mapping
5. Click "Load Dataset"
6. Select heuristic (SPT)
7. Click "Apply & Schedule"
8. Scroll to Gantt Chart Visualization section

### Expected Results
- ✅ Y-axis: Shows Job-Operation format
- ✅ X-axis: Shows days
- ✅ All jobs visible and color-coded
- ✅ Chart height fits all operations
- ✅ No cramped or overlapping bars

---

## Test 6: Multiple Breakdowns

### Steps
1. Load dataset and compute heuristics
2. Add first breakdown:
   - Machine: M1, Start: 1440, Duration: 240
3. Add second breakdown:
   - Machine: M2, Start: 2880, Duration: 360
4. Add third breakdown:
   - Machine: M1, Start: 4320, Duration: 180
5. Navigate to Gantt Chart
6. Click "Refresh"

### Expected Results
- ✅ Breakdown count chip: "3 Breakdowns"
- ✅ Three red dotted lines visible:
  - BREAKDOWN-M1-0 at day 1.0
  - BREAKDOWN-M2-0 at day 2.0
  - BREAKDOWN-M1-1 at day 3.0
- ✅ All breakdowns show in legend
- ✅ Warning alert: "3 breakdown period(s) detected"

---

## Test 7: Breakdown Impact on Different Heuristics

### Steps
1. Load dataset
2. Add breakdown: M1, Start: 1440, Duration: 480
3. Compute all heuristics
4. Compare metrics for SPT, EDD, CR, PRIORITY
5. Navigate to Comparison page

### Expected Results
- ✅ Each heuristic handles breakdown differently
- ✅ Different makespans for each heuristic
- ✅ Some heuristics may have more tardiness
- ✅ Comparison charts show clear differences
- ✅ No "Scheduling failed: False" errors

---

## Test 8: Reset After Breakdown

### Steps
1. Load dataset
2. Add multiple breakdowns
3. Compute heuristics → Note metrics
4. Click "Reload Dataset" on Dashboard
5. Compute heuristics again
6. Navigate to Gantt Chart

### Expected Results
- ✅ Breakdowns cleared (no red dotted lines)
- ✅ Breakdown count chip: Not shown (0 breakdowns)
- ✅ Metrics return to original values (without breakdown impact)
- ✅ Schedule optimized without breakdown constraints

---

## Test 9: Professional UI Consistency

### Check All Pages
Visit each page and verify NO emojis present:

#### Dashboard
- ✅ "CNC Scheduling Dashboard" (no emoji)
- ✅ "Dataset Loaded Successfully" card (no emoji)
- ✅ "AI Insights" section (no emoji)

#### Gantt Chart
- ✅ "Gantt Chart" title (no emoji)
- ✅ Legend text (no emojis)

#### Comparison
- ✅ "Heuristic Comparison" title (no emoji)

#### Excel Upload
- ✅ "Excel Data Import" (no emoji)
- ✅ "Transformation Complete" (no emoji)
- ✅ "Schedule Jobs" button (no emoji)
- ✅ "Tip:" text (no emoji)
- ✅ All result sections (no emojis)

#### Settings
- ✅ "Settings" title (no emoji)

#### Cost Analysis
- ✅ All chart titles (no emojis)
- ✅ All insight boxes (no emojis)

---

## Common Issues & Solutions

### Issue: Breakdown Not Showing on Gantt Chart
**Solution**: Click "Refresh" button in Gantt Chart page after adding breakdown

### Issue: Operations Overlapping with Breakdown
**Solution**: Recompute heuristics after adding breakdown - scheduler will reschedule around it

### Issue: Gantt Chart Still Shows Minutes
**Solution**: Clear browser cache (Ctrl+F5) and reload page

### Issue: Y-Axis Still Shows Machines
**Solution**: Frontend may need hard refresh - close and reopen browser

### Issue: Emojis Still Visible
**Solution**: Hard refresh browser (Ctrl+Shift+R) to reload React components

---

## Performance Expectations

### Gantt Chart Load Time
- Small dataset (< 50 ops): < 1 second
- Medium dataset (50-200 ops): 1-2 seconds
- Large dataset (200+ ops): 2-4 seconds

### Breakdown Simulation
- Add single breakdown: < 500ms
- Fetch and display on Gantt: < 1 second

### Recompute with Breakdown
- All 4 heuristics with 1 breakdown: 2-4 seconds
- Depends on dataset size

---

## Success Criteria

All tests pass when:
1. ✅ No emojis visible anywhere in UI
2. ✅ Gantt chart Y-axis shows Job-Operation format
3. ✅ Gantt chart X-axis shows days
4. ✅ Breakdowns appear as red dotted lines
5. ✅ Breakdown count displayed correctly
6. ✅ Hover info shows both days and minutes
7. ✅ Original data unchanged after recompute
8. ✅ Operations rescheduled around breakdowns
9. ✅ Excel upload works with new Gantt format
10. ✅ Professional, clean UI throughout

**Current Status**: All features implemented and ready for testing
