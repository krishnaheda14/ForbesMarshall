# ✅ Quick Verification Guide - CapEx Feature

## Step 1: Start the Application

### Terminal 1 - Backend
```powershell
cd C:\Users\Krishna\Downloads\Forbesmarshall
python backend\main.py
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
AI enabled: OpenRouter primary (Mistral/Gemini fallback possible)
```

### Terminal 2 - Frontend
```powershell
cd C:\Users\Krishna\Downloads\Forbesmarshall\frontend
npm run dev
```

**Expected Output**:
```
VITE ready in 500ms
Local:   http://localhost:5173/
```

---

## Step 2: Verify Backend Endpoints

### Browser Test:
Open: http://localhost:8001/api/endpoints

**Look for CapEx category with 2 endpoints**:
- `POST /api/capex/analyze`
- `POST /api/capex/buy-machine`

### Or use the test script:
```powershell
cd C:\Users\Krishna\Downloads\Forbesmarshall
python test_capex_api.py
```

**Expected**: 6/6 tests pass ✅

---

## Step 3: Verify Frontend UI

### Open the App:
Navigate to: http://localhost:5173

### Check Sidebar:
Look for "CapEx Analysis" menu item (with TrendingUp icon 📈)
- Should be between "Outsourcing" and "Activity Log"

### Click CapEx Analysis:
- URL should change to: http://localhost:5173/capex
- Page should show:
  - Title: "Capital Expenditure Analysis"
  - Labor rate input (default: $30)
  - "Analyze CapEx Opportunities" button
  - Empty state message

---

## Step 4: Test the Complete Flow

### 4.1 Load Data
1. Click "Dashboard" in sidebar
2. Click "Load Data" button
3. Wait for success message
4. Verify: "Data loaded successfully - X operations, Y machines"

### 4.2 Run a Schedule
1. Stay on Dashboard or go to "Comparison"
2. Select a heuristic (e.g., SPT)
3. Click "Compute Schedule"
4. Wait for schedule to complete
5. Note the "Outsourced" count in the metrics

### 4.3 Analyze CapEx
1. Click "CapEx Analysis" in sidebar
2. (Optional) Change hourly labor rate to $40 or $50
3. Click "Analyze CapEx Opportunities"
4. Watch the API Debug Console (bottom right) - should show:
   - `POST /api/capex/analyze` with 200 status

### 4.4 Review Results
**Summary Card** should show:
- Biggest Offender: (operation type, e.g., "TURNING")
- Outsourced Operations: (number)
- Total Vendor Cost: ($X,XXX.XX)

**Recommendations Table** should show:
- Multiple machines with financial metrics
- Green rows = Profitable
- Red rows = Not profitable
- Payback period in years

### 4.5 Buy a Machine
1. Find a machine with positive savings (green row)
2. Click "Buy" button
3. Wait for success message
4. Verify: "Successfully purchased and added machine MX_NEW1"
5. Check API Debug Console - should show:
   - `POST /api/capex/buy-machine` with 200 status

### 4.6 Verify Purchase
1. Go to "Settings" or "Dashboard"
2. Check machine count increased by 1
3. Or open: `data/machine_data.csv`
4. Should see new row: `M6_NEW1` (or similar)

---

## Step 5: Verify API Debug Console

### Bottom-right corner of UI:
- Should see a bug icon 🐛 with badge showing # of API calls
- Click to expand
- Should show recent API calls including:
  - `/api/capex/analyze`
  - `/api/capex/buy-machine`
- Click on each to see request/response details

---

## ✅ Success Criteria

- [x] Backend starts without errors
- [x] Frontend loads successfully
- [x] CapEx menu item visible in sidebar
- [x] CapEx page loads at /capex
- [x] "Analyze" button triggers API call
- [x] Analysis returns recommendations
- [x] "Buy" button clones machine
- [x] New machine appears in machine_data.csv
- [x] API Debug Console shows all calls
- [x] No errors in browser console
- [x] No errors in backend logs

---

## 🔍 Troubleshooting

### Issue: "No outsourced operations found"
**Solution**: 
- Make sure you ran a schedule first (Step 4.2)
- Check outsourcing policy is not too restrictive
- Try lowering cost threshold in settings

### Issue: "Data not loaded"
**Solution**: 
- Click "Load Data" on Dashboard
- Wait for confirmation message
- Refresh the page if needed

### Issue: "Buy" button disabled
**Solution**: 
- This is normal if savings are negative
- Look for green rows (profitable machines)
- Only profitable machines have enabled Buy buttons

### Issue: API calls failing
**Solution**: 
- Verify backend is running on port 8001
- Check browser console for CORS errors
- Restart both backend and frontend

### Issue: Purchase successful but machine not in schedule
**Solution**: 
- Schedules are invalidated after purchase
- Recompute the schedule to use new machine
- New machine will now be available for assignment

---

## 🎯 Quick Test Commands

### Test backend only:
```powershell
# Terminal 1
python backend\main.py

# Terminal 2
python test_capex_api.py
```

### Check endpoints:
```powershell
curl http://localhost:8001/api/endpoints
curl http://localhost:8001/api/health
```

### Check CSV was updated:
```powershell
Get-Content data\machine_data.csv | Select-Object -Last 5
```

---

## 📸 Screenshots to Verify

1. **Sidebar** - Should show "CapEx Analysis" menu item
2. **CapEx Page** - Empty state with analyze button
3. **Analysis Results** - Summary card + recommendations table
4. **Buy Action** - Success message after purchase
5. **API Debug** - Console showing API calls
6. **CSV File** - New machine row added

---

## ✨ Expected Behavior

### When Analysis is Successful:
- Summary card shows biggest offender operation type
- Table displays 1-5 machine recommendations
- Each row shows purchase price, costs, savings, payback
- Profitable machines have enabled green "Buy" buttons
- Unprofitable machines have disabled red buttons

### When Buy is Successful:
- Success alert appears: "Successfully purchased and added machine..."
- Analysis automatically refreshes
- New machine count increases
- machine_data.csv updated permanently
- Schedules are cleared (need recompute)

### API Debug Console:
- Shows all API calls with timestamps
- Green status = 200-299 (success)
- Red status = 400+ (error)
- Click to expand and see full request/response
- Duration shown in milliseconds

---

*Last Updated: November 30, 2025*
*Status: All features implemented and tested ✅*
