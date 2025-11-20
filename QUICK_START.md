# Quick Start Guide - CNC Scheduler

## 🚀 Starting the Application

### 1. Start Backend
```powershell
cd backend
C:/Users/Krishna/AppData/Local/Programs/Python/Python310/python.exe -m uvicorn main:app --reload --port 8001
```
**Verify**: http://localhost:8001/docs should show API documentation

### 2. Start Frontend
```powershell
cd frontend
npm run dev
```
**Verify**: http://localhost:5173 should show the dashboard

---

## 📊 Using the Application

### Load Dataset (First Time)
1. Open http://localhost:5173
2. Click **"Load Dataset"** button
3. Wait for green success card: **"Dataset Loaded Successfully"**
4. See operation/machine/job counts

### Apply Scheduling Heuristic
1. Open **Sidebar** (menu icon)
2. Go to **Compute Controls**
3. Click **"Compute All Heuristics"** (computes SPT, EDD, CR, PRIORITY)
4. Select heuristic from dropdown (e.g., **SPT**)
5. Click **"Apply Heuristic"**
6. Navigate to **Dashboard**

### View Results
**Dashboard** shows:
- ✅ Dataset loaded status (persists on refresh)
- ✅ Active heuristic name
- ✅ KPI cards (Makespan, Tardiness, Utilization, On-Time %)
- ✅ AI Insights (auto-generated)

**Gantt View**:
- Visual timeline of scheduled operations
- Color-coded by job
- Machine assignments visible

**Comparison**:
- Bar charts comparing all 4 heuristics
- Metrics side-by-side

---

## 📤 Excel Upload Workflow

### Upload Custom Data
1. Navigate to **Settings** page
2. Click **"Upload Excel File"**
3. Select file with these columns:
   - Job_ID, Operation_ID, Part_Type, Operation_Type
   - Processing_Time, Setup_Time, Quantity, Material
   - Priority, Release_Day, Due_Date, Tool_Group
4. Review auto-mapping preview
5. Click **"Load Dataset"**
6. Select heuristic (SPT/EDD/CR/PRIORITY)
7. Click **"Apply & Schedule"**
8. ✅ Schedule created automatically

---

## 🔄 State Persistence

### What Persists (localStorage)
- ✅ Dataset loaded state
- ✅ Current heuristic selection
- ✅ Computed schedules
- ✅ Metrics for all heuristics

### Refresh Behavior
**Before**: Refresh → Reset to "Load Dataset" ❌
**Now**: Refresh → All state preserved ✅

**Test it**:
1. Load data → Apply heuristic
2. Press **F5** to refresh
3. Dashboard still shows loaded state and heuristic ✅

---

## 🤖 AI Insights

### Auto-Generation (New Feature)
**When heuristic is applied**:
1. AI insights generate automatically ✅
2. Shows loading spinner
3. Displays analysis card with:
   - Performance analysis
   - Bottleneck identification
   - Recommendations

**Refresh Insights**:
Click "Refresh Insights" button to regenerate

---

## 🧪 Testing

### Quick Test
```powershell
python test_excel_workflow.py
```
**Expected Output**:
```
✅ SPT: Scheduled 3 jobs
✅ EDD: Scheduled 3 jobs  
✅ CR: Scheduled 3 jobs
✅ PRIORITY: Scheduled 3 jobs
```

### API Test
```powershell
curl http://localhost:8001/api/data/info
```
**Expected**: JSON with operations count

---

## 🐛 Troubleshooting

### Backend Not Starting
**Error**: `ModuleNotFoundError`
**Solution**:
```powershell
cd backend
pip install -r requirements.txt
```

### Frontend Blank Page
**Error**: Can't connect to backend
**Solution**: Verify backend is running on port 8001
```powershell
curl http://localhost:8001/docs
```

### "Scheduling failed: False"
**Status**: ✅ FIXED (Assignment_Type bug)
**If still occurs**: Restart backend to load fixed code

### State Not Persisting
**Solution**: Clear localStorage
```javascript
// In browser console:
localStorage.clear();
location.reload();
```

---

## 📁 Project Structure

```
Forbesmarshall/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── cnc_scheduler_core.py      # Core scheduler (FIXED)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── Dashboard.jsx      # Enhanced with persistence
│   │   ├── store/
│   │   │   └── useSchedulerStore.js # localStorage persistence
│   │   └── services/
│   │       └── api.js
│   └── package.json
├── data/
│   ├── jobs_dataset.csv
│   ├── machine_data.csv
│   └── vendor_data.csv
├── test_scheduler.py              # Unit test
├── test_excel_workflow.py         # Integration test
├── TESTING_GUIDE.md               # Complete testing instructions
└── SESSION_SUMMARY.md             # What was fixed
```

---

## ✅ Feature Checklist

- [x] Load dataset from CSV
- [x] Load dataset from Excel
- [x] Compute all heuristics (SPT, EDD, CR, PRIORITY)
- [x] Apply heuristic and view schedule
- [x] Gantt chart visualization
- [x] Metrics comparison
- [x] AI-powered insights (auto-generated)
- [x] State persistence (localStorage)
- [x] Machine breakdown simulation
- [x] Job priority updates
- [x] Outsourcing cost analysis
- [x] Activity logging
- [x] API debug console

---

## 🎯 Key Improvements

### Before
- ❌ Excel upload broken (KeyError: False)
- ❌ State resets on refresh
- ❌ Manual AI insights button
- ❌ "Load Dataset" shows even when loaded

### After
- ✅ Excel upload works (all heuristics)
- ✅ State persists across refreshes
- ✅ Auto-generated AI insights
- ✅ Clean persistent UI

---

## 📞 Support

### Documentation
- **TESTING_GUIDE.md** - Detailed testing steps
- **SESSION_SUMMARY.md** - Bug fixes and changes
- **This file** - Quick reference

### API Documentation
http://localhost:8001/docs

### Frontend Dev Tools
- Browser Console: API call logs
- React DevTools: Component inspection
- Network Tab: API request/response

---

## 🚦 System Status Indicators

### ✅ Healthy System
- Backend shows: `Uvicorn running on http://127.0.0.1:8001`
- Frontend shows: `Local: http://localhost:5173/`
- Dashboard displays: "Dataset Loaded Successfully"
- Heuristics compute without errors
- AI insights generate automatically

### ❌ Issues
- Backend error on startup → Check dependencies
- Frontend blank → Check backend connection
- Schedule fails → Check backend logs
- State lost → Check localStorage

---

**Last Updated**: Current session
**Status**: ✅ All systems operational
**Version**: 2.0 (with persistence and auto-insights)
