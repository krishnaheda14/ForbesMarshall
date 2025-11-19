# CNC Scheduling System - React Frontend with FastAPI Backend

## 🏗️ Project Architecture

This is a modern web application for CNC job scheduling with:
- **Backend**: FastAPI (Python) - Exposes scheduling algorithms as REST API
- **Frontend**: React + Material-UI - Professional dashboard interface
- **State Management**: Zustand
- **Charts**: Recharts & Plotly.js
- **AI Integration**: Google Gemini AI for insights

## 📁 Project Structure

```
Forbesmarshall/
├── backend/
│   └── main.py                 # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Main application pages
│   │   ├── services/          # API integration
│   │   ├── store/             # State management
│   │   ├── App.jsx            # Main app component
│   │   └── main.jsx           # Entry point
│   ├── package.json
│   └── vite.config.js
├── cnc_scheduler_core.py      # Core scheduling logic
├── cnc-scheduling.py          # Original Streamlit app (reference)
└── data/                      # CSV datasets
    ├── jobs_dataset.csv
    ├── machine_data.csv
    ├── vendor_data.csv
    └── previous_next_material.csv
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Create Python virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Python dependencies:**
   ```powershell
   pip install fastapi uvicorn pandas numpy python-dotenv google-generativeai plotly
   ```

3. **Create `.env` file** in the root directory:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the backend server:**
   ```powershell
   cd backend
   python main.py
   ```
   
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node.js dependencies:**
   ```powershell
   cd frontend
   npm install
   ```

2. **Run the development server:**
   ```powershell
   npm run dev
   ```
   
   The React app will open at `http://localhost:3000`

## 🎯 Features

### ✅ Implemented

1. **Data Management**
   - Load datasets from CSV files
   - Real-time data statistics
   - Activity logging

2. **Scheduling Algorithms (Heuristics)**
   - SPT (Shortest Processing Time)
   - EDD (Earliest Due Date)
   - CR (Critical Ratio)
   - PRIORITY (Priority-Based)
   - WEIGHTED (Multi-Objective)
   - SLACK (Minimum Slack)

3. **Dashboard**
   - KPI metrics display
   - Heuristic selection
   - Compute & apply controls
   - AI-powered insights

4. **Advanced Controls**
   - Machine breakdown simulator
   - Job priority manager
   - Outsourcing policy configurator

5. **API Endpoints**
   - `/api/data/load` - Load dataset
   - `/api/data/info` - Get data info
   - `/api/schedule/compute` - Compute single heuristic
   - `/api/schedule/compute-all` - Compute all heuristics
   - `/api/schedule/apply` - Apply heuristic
   - `/api/schedule/current` - Get current schedule
   - `/api/machine/breakdown` - Simulate breakdown
   - `/api/job/priority` - Update priority
   - `/api/outsourcing/policy` - Update outsourcing
   - `/api/ai/insights` - Get AI insights
   - `/api/metrics/comparison` - Compare heuristics
   - `/api/activity-log` - Activity history
   - `/api/job/{job_id}` - Delete job

### 🚧 To Be Completed

The following pages need to be created (component structure is ready):

1. **Comparison Page** (`src/pages/Comparison.jsx`)
   - Side-by-side heuristic comparison table
   - Performance metrics charts
   - Recommendation engine

2. **Gantt Chart Page** (`src/pages/GanttView.jsx`)
   - Interactive Gantt chart using Plotly.js
   - Machine timeline visualization
   - Maintenance window display

3. **Operation Status Page** (`src/pages/OperationStatus.jsx`)
   - Detailed operation table
   - Status tracking (Scheduled, In Progress, Late)
   - Critical ratio calculations
   - Export to CSV

4. **Settings Page** (`src/pages/Settings.jsx`)
   - System configuration
   - Data management
   - Export/import schedules
   - Reset system

## 🎨 UI Components Created

- `Sidebar.jsx` - Navigation with gradient background
- `HeuristicSelector.jsx` - Algorithm selection dropdown
- `ComputeControls.jsx` - Compute & apply buttons
- `MachineryControls.jsx` - Breakdown, priority, outsourcing controls
- `KPICards.jsx` - Metrics display cards
- `AIInsightsPanel.jsx` - AI recommendations panel

## 📊 Data Flow

1. User loads data via Dashboard
2. Data is sent to FastAPI backend
3. Backend processes using `cnc_scheduler_core.py`
4. User selects heuristic from sidebar
5. Clicks "Compute All Heuristics"
6. Backend runs all 6 algorithms
7. Frontend displays comparison
8. User applies best heuristic
9. Dashboard shows KPIs
10. Gantt chart visualizes schedule

## 🔧 API Usage Examples

### Load Data
```javascript
const response = await loadData();
// Returns: { status, message, stats }
```

### Compute All Heuristics
```javascript
const response = await computeAllHeuristics();
// Returns: { status, results, comparison }
```

### Get AI Insights
```javascript
const response = await getAIInsights(
  "Analyze SPT performance",
  { heuristic: "SPT", schedule_size: 150 }
);
// Returns: { status, insights, ai_enabled }
```

## 🎯 Next Steps to Complete

1. **Create Comparison.jsx:**
   - Use Material-UI Table component
   - Add sorting and filtering
   - Integrate Recharts for bar charts
   - Add "Best Heuristic" recommendation banner

2. **Create GanttView.jsx:**
   - Integrate `react-plotly.js`
   - Fetch schedule data from API
   - Display machine timelines
   - Show maintenance windows in red
   - Add zoom/pan controls

3. **Create OperationStatus.jsx:**
   - Create data table with operation details
   - Add status badges (green=On Time, red=Late)
   - Implement search/filter
   - Add CSV export button

4. **Create Settings.jsx:**
   - Add data upload form
   - System reset button
   - Configuration options
   - Activity log display

5. **Add Remaining Features:**
   - Job deletion UI
   - New job scheduler form
   - Real-time updates with WebSockets (optional)
   - Dark mode toggle

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'cnc_scheduler_core'`

**Solution:** Ensure you're in the root directory when running the backend:
```powershell
python backend/main.py
```

### Frontend Issues

**Problem:** CORS errors in browser console

**Solution:** Backend CORS is configured for `localhost:3000`. If using a different port, update `backend/main.py`:
```python
allow_origins=["http://localhost:YOUR_PORT"]
```

## 📝 License

MIT License - Free to use and modify

## 👨‍💻 Development

- Backend API docs: `http://localhost:8000/docs`
- Frontend dev server: `http://localhost:3000`

Happy scheduling! 🏭✨
