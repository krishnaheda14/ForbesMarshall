# CNC Scheduling System v2.0

A comprehensive production scheduling system for CNC machines with AI-powered Excel data import, real-time Gantt visualization, and multiple scheduling algorithms.

## 🌟 Features

- **Multiple Scheduling Algorithms**: SPT, EDD, CR, PRIORITY
- **AI-Powered Excel Import**: Automatically maps any Excel format using Google Gemini
- **Real-time Gantt Visualization**: Interactive timeline with breakdowns and operations
- **Machine Breakdown Simulation**: Test schedule resilience with simulated downtime
- **Outsourcing Optimization**: Dynamic make-or-buy decisions with cost analysis
- **Job Priority Management**: Update job priorities and see instant impact
- **AI Insights**: Get optimization recommendations powered by Gemini

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** and npm ([Download](https://nodejs.org/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Google Gemini API Key** ([Get Free Key](https://makersuite.google.com/app/apikey))

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/krishnaheda14/ForbesMarshall.git
cd ForbesMarshall
```

### Step 2: Set Up Python Backend

1. **Create Virtual Environment** (Windows):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install Python Dependencies**:
```powershell
cd backend
pip install -r requirements.txt
```

3. **Configure Environment Variables**:
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

4. **Verify Data Files**:
Ensure these CSV files exist in the `data/` folder:
- `jobs_dataset.csv`
- `machine_data.csv`
- `vendor_data.csv`
- `previous_next_material.csv`

### Step 3: Set Up React Frontend

1. **Install Node Dependencies**:
```powershell
cd ..\frontend
npm install
```

2. **Configure API URL** (if needed):
Create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8001
```

### Step 4: Start the Application

You need **TWO** terminal windows:

**Terminal 1 - Backend Server**:
```powershell
# From project root
.\venv\Scripts\Activate.ps1
cd backend
python main.py
```
Backend will start on: `http://localhost:8001`

**Terminal 2 - Frontend Server**:
```powershell
# From project root
cd frontend
npm run dev
```
Frontend will start on: `http://localhost:5173`

### Step 5: Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

## 📖 User Guide

### Initial Setup

1. **Load Data**: Click "Load Data" button on Dashboard
2. **Compute Heuristics**: Click "Compute All Heuristics" to generate schedules
3. **Select Algorithm**: Choose from SPT, EDD, CR, or PRIORITY in the sidebar
4. **View Results**: Navigate between Dashboard, Gantt Chart, and Comparison views

### Excel Data Import

1. Click **"Excel Upload"** in the sidebar
2. Upload your Excel file (any format)
3. Review auto-mapped columns (AI-powered)
4. Adjust mappings if needed
5. Click **"Confirm & Transform"**
6. Choose scheduling algorithm to apply

**Required Columns** (minimum):
- Job ID
- Processing Time

**Optional Columns** (auto-detected):
- Due Date, Priority, Machine, Quantity, Outsourcing Cost, etc.

### Machine Breakdown Simulation

1. Open **"Machine Breakdown"** accordion in sidebar
2. Enter Machine ID (e.g., `M1`)
3. Set Start Time (0-100,000 minutes)
4. Set Duration (30-5,000 minutes)
5. Click **"Simulate Breakdown"**
6. **Important**: Click **"Compute All Heuristics"** to see impact
7. View red dotted bars in Gantt Chart showing breakdowns

### Gantt Chart Features

- **Blue Solid Bars**: Scheduled operations
- **Red Dotted Bars**: Machine breakdowns/maintenance
- **Hover**: See operation or breakdown details
- **Zoom**: Scroll to zoom in/out on timeline
- **Auto-Refresh**: Chart updates when you change heuristic

## 🏗️ Project Structure

```
ForbesMarshall/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── models.py                  # Pydantic models for Excel import
│   ├── excel_ingestion.py         # Excel file parser
│   ├── schema_mapping.py          # AI column mapper
│   ├── data_transformer.py        # Data transformer
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/            # React components
│   │   │   ├── ComputeControls.jsx      # Algorithm selector & compute
│   │   │   ├── MachineryControls.jsx    # Breakdown/Priority/Outsource
│   │   │   ├── KPICards.jsx             # Metrics display
│   │   │   └── ...
│   │   ├── pages/                 # Page components
│   │   │   ├── Dashboard.jsx            # Main dashboard
│   │   │   ├── GanttView.jsx            # Gantt timeline
│   │   │   ├── ExcelUpload.jsx          # Excel import wizard
│   │   │   └── ...
│   │   └── services/
│   │       └── api.js             # API client
│   ├── package.json               # Node dependencies
│   └── vite.config.js             # Vite configuration
├── core/
│   └── scheduler.py               # Core scheduling algorithms
├── cnc_scheduler_core.py          # Main scheduler implementation
├── data/
│   ├── jobs_dataset.csv           # Sample job data
│   ├── machine_data.csv           # Machine specifications
│   ├── vendor_data.csv            # Outsourcing vendors
│   └── previous_next_material.csv # Material changeover penalties
└── .gitignore                     # Git ignore rules
```

## 🔧 Configuration

### Backend Configuration

**Port**: Default `8001` (change in `main.py`)
**CORS**: Configured for `http://localhost:5173`
**Gemini Model**: `gemini-1.5-flash` (configurable in `schema_mapping.py`)

### Frontend Configuration

**Port**: Default `5173` (change in `vite.config.js`)
**API Base URL**: `http://localhost:8001` (configurable in `api.js`)

## 📊 Scheduling Algorithms

| Algorithm | Optimization Goal | Best For |
|-----------|------------------|----------|
| **SPT** | Minimize average completion time | High throughput |
| **EDD** | Minimize tardiness | Meeting deadlines |
| **CR** | Balance urgency vs processing time | Mixed objectives |
| **PRIORITY** | Respect job importance | Critical jobs first |

## 🐛 Troubleshooting

### Backend Issues

**Port 8001 already in use**:
```powershell
# Find process
Get-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess
# Kill process
Stop-Process -Id <process_id> -Force
```

**Import errors**:
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Gemini API errors**:
- Verify API key in `.env` file
- Check quota at [Google AI Studio](https://makersuite.google.com/)
- Ensure API key has Gemini API enabled

### Frontend Issues

**Port 5173 already in use**:
```powershell
# Change port in vite.config.js
server: { port: 5174 }
```

**Module not found errors**:
```powershell
# Clean install
rm -r node_modules
rm package-lock.json
npm install
```

**API connection errors**:
- Verify backend is running on port 8001
- Check CORS settings in `backend/main.py`
- Verify `VITE_API_URL` in `frontend/.env`

### Data Issues

**Column name errors**:
- CSV files must have headers with underscores (e.g., `Machine_ID` not `Machine ID`)
- Check column names match exactly in `data/` CSVs

**File not found**:
- Ensure all 4 CSV files exist in `data/` folder
- Check file paths are correct in `main.py`

## 🔒 Security Notes

- **Never commit `.env` file** (already in `.gitignore`)
- **Keep Gemini API key private** (free tier has rate limits)
- **Use environment variables** for all sensitive data
- **CORS is configured for localhost** - update for production

## 📦 Dependencies

### Python (Backend)
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `pandas==2.1.3` - Data manipulation
- `numpy==1.26.2` - Numerical computing
- `google-generativeai==0.3.1` - Gemini AI
- `openpyxl==3.1.5` - Excel file handling
- `python-multipart` - File uploads

### JavaScript (Frontend)
- `react==18.2.0` - UI framework
- `react-router-dom==6.20.0` - Routing
- `@mui/material==5.15.0` - UI components
- `plotly.js-dist==2.27.1` - Gantt charts
- `zustand==4.4.7` - State management
- `axios==1.6.2` - HTTP client
- `notistack==3.0.1` - Notifications

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is developed for Forbes Marshall. All rights reserved.

## 🆘 Support

For issues and questions:
1. Check existing issues on GitHub
2. Review troubleshooting guide above
3. Create a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, Node version)

## 🎯 Roadmap

- [ ] Multi-objective optimization
- [ ] Real-time scheduling updates
- [ ] Export schedules to PDF/Excel
- [ ] Template management for Excel imports
- [ ] User authentication
- [ ] Schedule comparison with diff view
- [ ] Mobile responsive design
- [ ] Docker containerization

## 👥 Authors

- Krishna Heda - [@krishnaheda14](https://github.com/krishnaheda14)

## 🙏 Acknowledgments

- Google Gemini AI for intelligent column mapping
- FastAPI team for excellent documentation
- Material-UI for beautiful components
- Plotly for powerful Gantt visualizations
