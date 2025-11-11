# Machine Breakdown Simulator - User Guide

## 📍 Where to See Changes After Simulating Breakdown

When you click **"Simulate Breakdown"**, the changes appear in multiple places across the app:

---

### **1. 📈 Gantt Chart Tab (IMMEDIATE VISUAL CHANGE)**

**What Changes:**
- **Timeline extends** - The makespan increases because operations are delayed
- **Gap appears** in the broken machine's timeline during breakdown period
- **Operations shift right** - All operations after the breakdown time are rescheduled
- **Color coding** - You'll see the breakdown period as a gap in activity

**How to Check:**
1. Note the current makespan (rightmost edge of timeline)
2. Click "Simulate Breakdown"
3. Gantt chart reloads automatically
4. Look for the gap in the selected machine's row
5. See how operations are pushed to later times

---

### **2. 🎯 KPI Dashboard (TOP OF MAIN PAGE)**

**What Changes:**
- **Makespan (Days)** ↗️ INCREASES (more time needed to complete all jobs)
- **Total Tardiness (Days)** ↗️ INCREASES (more jobs become late)
- **Total Cost ($)** ↗️ INCREASES (longer production time = higher cost)
- **Machine Utilization %** ↘️ **DECREASES** (breakdown = idle time = lower utilization)
- **On-Time Delivery %** ↘️ DECREASES (more jobs miss deadlines)
- **Outsourced %** → May stay same (unless breakdown forces outsourcing)

**Expected Changes Example:**
```
BEFORE Breakdown:
- Makespan: 142.5 days
- Utilization: 58.3%
- On-Time: 78.5%
- Tardiness: 12.3 days

AFTER Breakdown (120 min on M1):
- Makespan: 142.8 days ↗️ (+0.3 days)
- Utilization: 57.9% ↘️ (-0.4%)
- On-Time: 77.8% ↘️ (-0.7%)
- Tardiness: 13.1 days ↗️ (+0.8 days)
```

---

### **3. 📋 Job Status Tab**

**What Changes:**
- **Finish Time** - Jobs affected by the breakdown will show later finish times
- **Tardiness** - Jobs that were on-time may now show tardiness
- **Status** - Some jobs may change from "On Time" to "Late"

**How to Check:**
1. Before breakdown: Note finish times of jobs assigned to the breakdown machine
2. Simulate breakdown
3. Go to "Job Status" tab
4. Sort by "Finish Time" or "Tardiness"
5. Compare the times - they should be delayed

---

### **4. ⚖ Heuristic Comparison Tab (IF COMPUTED)**

**What Changes (AFTER FIX):**
- **Machine Utilization %** column ↘️ DECREASES for ALL heuristics
- **Makespan Days** ↗️ INCREASES across all algorithms
- **Total Tardiness Days** ↗️ INCREASES
- **Late Jobs** count ↗️ INCREASES
- **Average Utilization** metric at bottom ↘️ DECREASES

**Important:**
- If you've already run "Compute All Heuristics" before the breakdown:
  - The breakdown will **recalculate ALL 4 heuristics** automatically
  - The comparison table will update to show new metrics
  - This may take 10-30 seconds to recalculate

**How to Check:**
1. First, run "Compute All Heuristics" (takes 2-3 min)
2. Note the "Average Utilization" metric
3. Go back to sidebar, simulate breakdown
4. Return to "Heuristic Comparison" tab
5. You should see ALL metrics updated (lower utilization, longer makespan)

---

## 🔍 Why Utilization Shows 20% Instead of Expected 60%

### **Possible Causes:**

#### **1. Still Using 5-Machine Configuration**
- **Check:** Sidebar → "Machine ID" dropdown in breakdown simulator
- **Should See:** Only M1 and M6
- **If You See:** M1, M3, M4, M6, M9 (5 machines)
- **Fix:** App hasn't loaded the 2-machine config yet
  ```powershell
  # Verify the active config
  Get-Content data\machine_data.csv
  
  # Should show only M1 and M6
  # If it shows 5 machines, run:
  .\switch_machine_config.ps1 -Mode 2
  ```

#### **2. Cache Issue**
- **Fix:** Clear Streamlit cache
  - Press `C` key while app is running
  - Or restart the app completely: Ctrl+C and re-run

#### **3. Session State Not Updated**
- **Fix:** Hard refresh
  - Press `R` key in the Streamlit app
  - Or refresh browser (F5)
  - Or restart terminal and re-run app

#### **4. Data Files Not Reloaded**
- **Fix:** Restart the Streamlit app
  ```powershell
  # In terminal where Streamlit is running:
  Ctrl+C  # Stop app
  
  streamlit run cnc-scheduling.py  # Restart
  ```

---

## 🎯 Expected Behavior After Breakdown

### **Breakdown Parameters Example:**
- Machine: M1
- Breakdown Start: 5000 minutes
- Breakdown Duration: 120 minutes (2 hours)

### **Expected Impact:**

1. **Gantt Chart:**
   - M1 row shows gap from minute 5000 to 5120
   - All M1 operations scheduled after minute 5000 shift right by 120 minutes
   - May see cascading delays in dependent operations on other machines

2. **KPI Dashboard:**
   - **Makespan:** Increases by ~120-200 minutes (0.25-0.4 days)
     - Direct: +120 min from breakdown
     - Indirect: Additional delays from operation dependencies
   
   - **Utilization:** Decreases by ~1-2%
     - Formula: Lost time / Total available time
     - 120 min lost on 1 machine / (2 machines × makespan)
     - Example: 120 / (2 × 68,640 min) = -0.09%
     - Additional drop from extended makespan
   
   - **Tardiness:** Increases by 50-200 minutes total
     - Operations near the breakdown time become late
     - Cascading effect on dependent jobs

3. **Job Status:**
   - 3-8 jobs affected (those using M1 after minute 5000)
   - Finish times increase by 120+ minutes
   - 1-3 additional jobs may become late

4. **Comparison Table:**
   - ALL heuristics show similar percentage drops
   - Utilization: -0.5% to -2% across all algorithms
   - Makespan: +0.25 to +0.5 days

---

## 🛠️ Troubleshooting Checklist

- [ ] **Step 1:** Verify 2-machine config is active
  ```powershell
  Get-Content data\machine_data.csv -Head 3
  # Should show: M1, M6
  ```

- [ ] **Step 2:** Restart Streamlit app
  ```powershell
  Ctrl+C
  streamlit run cnc-scheduling.py
  ```

- [ ] **Step 3:** Wait for initial load (shows ~58-63% utilization)

- [ ] **Step 4:** Note the baseline KPI values

- [ ] **Step 5:** Simulate breakdown with these settings:
  - Machine: M1
  - Start: 10000 min
  - Duration: 240 min

- [ ] **Step 6:** Check changes in ALL tabs:
  - Gantt Chart: Gap visible?
  - KPI Dashboard: Utilization decreased?
  - Job Status: Any jobs show increased tardiness?
  - Comparison: (If computed) Metrics updated?

- [ ] **Step 7:** If no changes visible:
  - Press 'C' to clear cache
  - Press 'R' to rerun
  - Check browser console for errors (F12)

---

## 📊 Visual Change Summary

| Location | What to Look For | Expected Change |
|----------|-----------------|-----------------|
| **Gantt Chart** | Timeline length, gaps in machine rows | Gap appears, timeline extends |
| **KPI Dashboard** | Utilization gauge (top right) | Decreases by 1-2% |
| **KPI Dashboard** | Makespan (top left) | Increases by 0.25-0.5 days |
| **KPI Dashboard** | Tardiness (top middle) | Increases by 0.1-0.5 days |
| **Job Status Table** | Finish Time, Tardiness columns | Some values increase |
| **Comparison Table** | Machine_Utilization_% column | Decreases across all heuristics |
| **Comparison Metrics** | Average Utilization metric | Decreases by 1-2% |

---

## 🚀 Quick Test Procedure

1. **Baseline Check:**
   - Open app
   - Note KPI Dashboard utilization: Should be ~58-63%
   - Note makespan: Should be ~140-150 days

2. **Apply Breakdown:**
   - Sidebar → Machine Breakdown Simulator
   - Select: M1
   - Start: 10000 min
   - Duration: 240 min
   - Click "Simulate Breakdown"

3. **Verify Changes:**
   - KPI utilization should drop to ~57-62%
   - Makespan should increase to ~140.5-150.5 days
   - Gantt chart should show gap on M1

4. **Check Persistence:**
   - Switch to different heuristic (e.g., EDD → SPT)
   - Breakdown should persist (still visible in Gantt)
   - Utilization should remain low (~57-62%)

---

## 💡 Pro Tips

1. **Big Breakdown for Obvious Changes:**
   - Use Duration: 480 min (1 full workday)
   - Effect will be more visible (~3-5% utilization drop)

2. **Early Breakdown for Maximum Impact:**
   - Start: 2000-5000 min (early in schedule)
   - Affects more downstream operations

3. **Compare Before/After:**
   - Take screenshot of KPI dashboard BEFORE breakdown
   - Simulate breakdown
   - Compare side-by-side

4. **Reset to Original:**
   - To remove breakdown: Restart the app
   - Or switch heuristic back and forth (resets to base data)

---

**Last Updated:** November 6, 2025  
**Fix Applied:** Breakdown simulator now recalculates ALL metrics automatically
