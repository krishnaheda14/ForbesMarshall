# Activity Log & Breakdown Visualization Features

## ✅ Implemented Features

### 1. **Comprehensive Activity Logging System**

All major actions are now tracked in a persistent activity log with the following details:
- **Timestamp**: When the action occurred
- **Action**: Type of operation performed
- **Details**: Specific information about the change
- **Affected Items**: What was modified

#### Logged Actions:

1. **System Initialization**
   - Records when the system is first loaded
   - Captures total operations and machines loaded

2. **Machine Breakdown Added**
   - Machine ID, start time, duration, and end time
   - Lists all affected operations that were scheduled during breakdown
   - Shows how many operations were auto-outsourced due to conflicts
   - Example: `Machine: M3, Start: 1500 min, Duration: 120 min, End: 1620 min | Affected Operations: OP_001, OP_005 | Auto-outsourced: 1 ops`

3. **Priority Updated**
   - Job ID and priority change (e.g., P3 → P1)
   - Example: `Job: JOB_015, Changed from P3 to P1`

4. **Outsourcing Policy Updated**
   - Threshold change and impact on operations
   - Shows before/after outsourcing counts and percentages
   - Example: `Threshold changed to 1.20 | Outsourced: 15 → 22 (44.0%) | Increased by 7 ops`

5. **All Heuristics Computed**
   - Lists which heuristics were computed (SPT, EDD, CR, PRIORITY)
   - Dataset size (operations and machines)
   - Example: `Computed: SPT, EDD, CR, PRIORITY | Dataset: 50 ops, 5 machines`

6. **Heuristic Applied**
   - Which heuristic was applied to the dataset
   - Schedule size
   - Example: `Schedule size: 45 operations | Updated operation assignments`

7. **Job Deleted**
   - Job ID and number of operations removed
   - Lists all deleted operation IDs
   - Example: `Deleted Job: JOB_023 | Operations removed: 3 (OP_045, OP_046, OP_047)`

### 2. **Activity Log Viewer**

Located in the **Heuristic Comparison** page:

- **Expandable Section**: "📋 Activity Log (Recent Changes)"
- **Reverse Chronological Order**: Most recent actions shown first
- **Interactive Table**: Shows timestamp, action type, details, and affected items
- **Scrollable**: 300px height with full details visible
- **Download Feature**: Export entire activity log as CSV file with timestamp in filename

### 3. **Enhanced Gantt Chart - Breakdown Visualization**

Breakdowns and maintenance windows are now prominently displayed:

#### Visual Features:
- **Red Rectangle Overlay**: Semi-transparent red shading over breakdown periods
- **Dotted Red Border**: Clear demarcation of breakdown boundaries
- **Warning Labels**: Each breakdown shows:
  - ⚠️ BREAKDOWN icon
  - Duration in minutes
- **Dark Mode Compatible**: Works in both light and dark themes
- **Multiple Windows Support**: Can display multiple breakdowns on the same machine
- **Layered Above Operations**: Breakdown overlays appear on top for visibility

#### Example Display:
```
Machine M3: [Normal operations] [⚠️ BREAKDOWN 120 min] [Postponed operations]
```

### 4. **Breakdown Conflict Detection & Resolution**

When you add a breakdown, the system:

1. **Scans Current Schedule**: Identifies operations scheduled during breakdown window
2. **Reports Conflicts**: Shows count of affected operations
3. **Auto-Outsource Evaluation**: Reruns make-or-buy analysis for each affected operation
4. **Smart Reassignment**: 
   - Operations that can't meet deadlines → Auto-outsourced
   - Remaining operations → Rescheduled after breakdown ends
5. **User Feedback**: Clear messages about what happened:
   - "⚠️ 3 operation(s) were scheduled during breakdown time"
   - "📦 1 operation(s) reassigned to OUTSOURCE due to breakdown conflict"
   - "🔄 Remaining operations will be rescheduled after breakdown window"

### 5. **Dark Mode Optimizations**

All new features are optimized for dark mode:
- Transparent backgrounds for Gantt chart
- Light text (#E0E0E0) for better contrast
- Semi-transparent gridlines
- Bright red breakdown indicators visible in both themes

---

## 📊 How to Use

### View Activity Log:
1. Navigate to **Heuristic Comparison** page
2. Expand **"📋 Activity Log (Recent Changes)"** section
3. Review all recent actions with timestamps
4. Download CSV for record-keeping

### See Breakdowns in Gantt Chart:
1. Add a breakdown via **"🔧 Machine Breakdown Simulator"**
2. Click **"🧪 Compute All Heuristics"**
3. Navigate to **Selected Heuristic View** → **Gantt Chart** tab
4. Breakdowns appear as red shaded rectangles with labels

### Track Changes:
- Every significant action is automatically logged
- No manual tracking needed
- Full audit trail maintained in session
- Export anytime for external analysis

---

## 🎯 Benefits

1. **Full Transparency**: Know exactly what changes were made and when
2. **Audit Trail**: Complete history of all scheduling decisions
3. **Conflict Prevention**: No operations will overlap with breakdown windows
4. **Visual Clarity**: Immediately see breakdown periods in schedule
5. **Smart Automation**: System automatically handles rescheduling conflicts
6. **Data Export**: Save activity logs for compliance or reporting

---

## 🔧 Technical Details

### Activity Log Structure:
```python
{
    'timestamp': '2025-11-16 14:30:45',
    'action': 'Machine Breakdown Added',
    'details': 'Machine: M3, Start: 1500 min, Duration: 120 min...',
    'affected_items': 'M3 (3 ops affected)'
}
```

### Breakdown Overlap Detection:
```python
# Operation overlaps breakdown if:
operation_start < breakdown_end AND operation_end > breakdown_start
```

### Gantt Chart Breakdown Display:
- Position: Aligned with machine row
- Color: `rgba(255,50,50,0.35)` - Semi-transparent red
- Border: `rgba(255,0,0,0.8)` - Solid red, 2px, dotted style
- Label: White text on red background with breakdown duration

---

## 📝 Example Activity Log Entry

```
Timestamp: 2025-11-16 14:30:45
Action: Machine Breakdown Added
Details: Machine: M3, Start: 1500 min, Duration: 120 min, End: 1620 min | Affected Operations: OP_015, OP_018, OP_022 | Auto-outsourced: 1 ops
Affected Items: M3 (3 ops affected)
```

This comprehensive logging ensures complete visibility into all scheduling changes and decisions!
