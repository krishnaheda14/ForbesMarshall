# 🏭 CNC Scheduling System - Complete Setup Guide

## ✨ What You've Got

A **modern, professional React-based UI** for your CNC scheduling application with:

- ✅ **Backend**: FastAPI (Python) - All your existing scheduling logic preserved as REST API
- ✅ **Frontend**: React + Material-UI - Beautiful, responsive dashboard
- ✅ **State Management**: Zustand for efficient state handling
- ✅ **Charts**: Plotly.js for interactive Gantt charts
- ✅ **AI Integration**: Gemini AI insights built-in
- ✅ **All Features**: Heuristics, breakdowns, priorities, outsourcing, everything!

---

## 🚀 Quick Start (2 Minutes)

### Option 1: Automated Script (Easiest)

```powershell
# Run from the Forbesmarshall directory
.\start.ps1
```

This will:
1. Check Python & Node.js
2. Create virtual environment
3. Install all dependencies
4. Start both servers automatically

### Option 2: Manual Setup

#### Step 1: Backend Setup

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend\requirements.txt

# Start backend
cd backend
python main.py
```

Backend runs at: **http://localhost:8000**

#### Step 2: Frontend Setup

```powershell
# In a NEW terminal
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:3000**

---

## 📁 What Was Created

### New Files Structure

```
Forbesmarshall/
├── backend/
│   ├── main.py                 # FastAPI server (NEW)
│   └── requirements.txt        # Python deps (NEW)
│
├── frontend/                   # Complete React app (ALL NEW)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx
│   │   │   ├── HeuristicSelector.jsx
│   │   │   ├── ComputeControls.jsx
│   │   │   ├── MachineryControls.jsx
│   │   │   ├── KPICards.jsx
│   │   │   └── AIInsightsPanel.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Comparison.jsx
│   │   │   ├── GanttView.jsx
│   │   │   ├── OperationStatus.jsx
│   │   │   └── Settings.jsx
│   │   ├── services/
│   │   │   └── api.js           # API integration
│   │   ├── store/
│   │   │   └── useSchedulerStore.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── cnc_scheduler_core.py       # Extracted scheduling logic (NEW)
├── start.ps1                   # Quick start script (NEW)
└── README_REACT_APP.md         # Documentation (NEW)
```

### Preserved Files

- ✅ `cnc-scheduling.py` - Your original Streamlit app (UNTOUCHED)
- ✅ `data/` folder - All CSV files (UNTOUCHED)
- ✅ All other existing files (UNTOUCHED)

---

## 🎯 How to Use

### 1. Load Data

1. Open **http://localhost:3000**
2. Click **"Load Dataset"** button
3. Wait for confirmation

### 2. Compute Schedules

1. Select a heuristic from sidebar (SPT, EDD, CR, PRIORITY)
2. Click **"Compute All Heuristics"**
3. Wait ~10-30 seconds

### 3. View Results

Navigate using sidebar:
- **Dashboard** - KPI metrics, quick actions
- **Comparison** - Side-by-side heuristic comparison
- **Gantt Chart** - Visual timeline
- **Operations** - Detailed table with export
- **Settings** - System info, activity log

### 4. Advanced Features

Expand **"Controls"** in sidebar:
- **Machine Breakdown** - Simulate downtime
- **Job Priority** - Update priorities
- **Outsourcing** - Adjust cost threshold

### 5. AI Insights

Click **"Get AI Insights"** on Dashboard or Comparison page for smart recommendations.

---

## 🔧 Configuration

### Environment Variables

Create `.env` in root directory:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Get your key from: https://makersuite.google.com/app/apikey

### Backend API Endpoints

Full documentation: **http://localhost:8000/docs**

Key endpoints:
- `POST /api/data/load` - Load dataset
- `POST /api/schedule/compute-all` - Run all heuristics
- `POST /api/schedule/apply` - Apply heuristic
- `POST /api/machine/breakdown` - Simulate breakdown
- `POST /api/ai/insights` - Get AI recommendations

---

## 🎨 UI Features

### Modern Design
- Gradient sidebar (blue theme)
- Card-based layout
- Responsive (works on tablets/mobile)
- Material-UI components

### Interactive Elements
- Searchable tables
- Filterable data
- Sortable columns
- CSV export
- Real-time updates

### Visualizations
- KPI cards with icons
- Plotly Gantt charts
- Comparison tables with highlighting
- Status chips (green/red)

---

## 🐛 Troubleshooting

### Backend Issues

**Error: ModuleNotFoundError**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r backend\requirements.txt
```

**Error: Port 8000 already in use**
```powershell
# Kill the process
Stop-Process -Name python -Force

# Or change port in backend/main.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Frontend Issues

**Error: EADDRINUSE (Port 3000 in use)**
```powershell
# Use different port
npm run dev -- --port 3001
```

**Error: CORS blocked**
- Check backend CORS settings in `backend/main.py`
- Ensure frontend URL is in `allow_origins`

### Data Issues

**No data loading**
- Verify `data/` folder has CSV files
- Check file names match:
  - `jobs_dataset.csv`
  - `machine_data.csv`
  - `vendor_data.csv`
  - `previous_next_material.csv`

---

## 📊 Comparison: Old vs New

| Feature | Streamlit App | React App |
|---------|--------------|-----------|
| UI Framework | Streamlit | React + Material-UI |
| Interactivity | Limited | Full control |
| Performance | Slower reruns | Instant updates |
| Customization | Limited | Fully customizable |
| Professional Look | Basic | Modern & polished |
| Mobile Support | Poor | Responsive |
| Deployment | Streamlit Cloud | Any web host |
| API Access | No | Full REST API |
| Integration | Difficult | Easy with API |

---

## 🚢 Next Steps

### Immediate
1. ✅ Load your data
2. ✅ Compute heuristics
3. ✅ Explore all pages
4. ✅ Try advanced controls

### Future Enhancements (Optional)
- [ ] Add user authentication
- [ ] Connect to database instead of CSV
- [ ] Real-time updates with WebSockets
- [ ] Docker deployment
- [ ] Multi-user support
- [ ] Historical schedule comparison

---

## 💡 Tips

1. **Performance**: First computation takes longer (20-30s). Subsequent ones are faster.
2. **AI Insights**: Requires valid Gemini API key in `.env`
3. **Export**: Use "Export CSV" button on Operations page
4. **Breakdown**: After simulating breakdown, click "Compute All Heuristics" to see impact
5. **Best Heuristic**: Check Comparison page for recommendation

---

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **React Docs**: https://react.dev
- **Material-UI**: https://mui.com
- **FastAPI**: https://fastapi.tiangolo.com

---

## 🆘 Support

If you encounter issues:
1. Check terminal for error messages
2. Verify all dependencies installed
3. Ensure both servers are running
4. Check browser console (F12) for frontend errors

---

## ✅ Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] All CSV files in `data/` folder
- [ ] `.env` file created (optional for AI)
- [ ] Virtual environment activated
- [ ] Dependencies installed

---

## 🎉 Success Criteria

You'll know it's working when:
1. ✅ Backend shows "Application startup complete" at http://localhost:8000
2. ✅ Frontend shows dashboard at http://localhost:3000
3. ✅ "Load Dataset" button successfully loads data
4. ✅ Heuristics compute without errors
5. ✅ All 5 pages render correctly

---

**Your original `cnc-scheduling.py` remains intact and functional!**

This React app is a complete replacement with modern UI, but both can coexist. Use whichever you prefer!

Happy scheduling! 🏭✨
