# 🏭 CNC Job Scheduling System

An intelligent, AI-powered CNC manufacturing scheduling system built with Streamlit. This application helps optimize production scheduling across multiple CNC machines using various heuristic algorithms and provides AI-driven insights for better decision-making.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### Core Scheduling Capabilities
- **6 Scheduling Algorithms**:
    - **SPT (Shortest Processing Time)**: Optimizes for throughput and fast completion
    - **EDD (Earliest Due Date)**: Minimizes tardiness and late deliveries
    - **CR (Critical Ratio)**: Balances urgency with processing complexity
    - **PRIORITY**: Respects business-defined job priorities
    - **BALANCED (Multi-factor Weighted)**: Blends priority, slack/urgency, and processing time for robust on-time delivery
    - **DEADLINE_FIRST (Urgency-focused)**: Prioritizes operations with least slack to prevent deadline misses

- **Multi-Machine Support**: Schedule operations across multiple CNC machines (MILLING, TURNING, GRINDING, DRILLING)
- **Dynamic Job Management**: Add or delete jobs on-the-fly with real-time capacity analysis
- **Make-or-Buy Decisions**: Intelligent outsourcing recommendations based on lead-time and cost threshold
- **Material Changeover Penalties**: Accounts for setup time when switching between materials
 - **Activity Log (Audit Trail)**: Every change (add/delete job, breakdowns, priority changes, heuristic runs, policy updates) is recorded and filterable

### AI-Powered Intelligence 🤖
- **Algorithm Recommendation**: AI explains why a specific algorithm is best for your priorities
- **Performance Analysis**: Identifies bottlenecks and suggests improvements
- **Dataset Quality Check**: Validates data quality and flags potential issues
- **Context-Aware Insights**: Tailored recommendations based on your weights and constraints

### Visualization & Analytics
- **Interactive Gantt Charts**: Visualize machine schedules with maintenance windows
- **Real-Time KPI Dashboard**: Track makespan, tardiness, utilization, costs
- **Operation Status Table**: Monitor all jobs with completion status and critical ratios
- **Comparison Matrix**: Side-by-side algorithm performance with composite scoring
- **Outsourcing Analysis**: Explains why operations are outsourced and where to pull work back in-house
- **Capacity Planning (What‑If)**: Simulate extra machines, extended shifts, and cost-threshold changes

### Advanced Features
- **Capacity Analysis**: Pre-validate new jobs before adding to schedule
- **Priority Management**: Adjust job priorities to reflect business needs
- **Machine Breakdown Simulator (Simplified)**: Add downtime using minutes-only input; auto prompts to recompute
- **Outsourcing Cost Threshold**: Slider to control make-or-buy; changes propagate to KPIs after recompute
- **Export Functionality**: Download schedules as CSV for further processing

--- 

## 🧭 How It Works (Working)
1. Load data from `data/` CSVs and validate the schema.
2. Run a make‑or‑buy pass using a configurable cost‑threshold to tag each operation as `IN_HOUSE` or `OUTSOURCE` based on in‑house cost/time vs vendor cost/lead‑time.
3. Schedule all in‑house operations with the selected heuristic (or compute all 6), honoring machine calendars, OEE, maintenance/breakdowns, setup penalties, and op precedence.
4. Compute KPIs (makespan, total tardiness, on‑time %, utilization, total cost) and render charts/tables.
5. Compare algorithms, apply the winner, and iterate. Any change (priority, breakdowns, threshold) triggers a recompute prompt and is recorded in the Activity Log.

## 🎯 Purpose

- Reduce outsourcing cost by pulling feasible work in‑house without missing due dates.
- Increase on‑time delivery and shop visibility via transparent KPIs and what‑if tools.
- Provide explainable analysis (outsourcing reasons, capacity levers) for informed decisions.

## 🧰 Tech Stack

- UI & runtime: Streamlit (Python)
- Data: pandas, numpy
- Visualization: plotly
- Config/Secrets: python‑dotenv (optional)
- AI insights: google‑generativeai (optional)
- Export helpers: openpyxl

See `requirements.txt` for exact versions.

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Data Format](#-data-format)
- [Configuration](#%EF%B8%8F-configuration)
- [AI Features](#-ai-features-setup)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/krishnaheda14/ForbesMarshall.git
cd ForbesMarshall
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Set Up AI Features
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_api_key_here
```

Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

---

## 🎯 Quick Start

### Launch the Application
```bash
streamlit run cnc-scheduling.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Workflow
1. **Load Data**: App automatically loads from `data/` folder on startup
2. **Compute Algorithms**: Click "🧪 Compute All Algorithms" in sidebar
3. **Compare Results**: Review the comparison table on the main screen
4. **Apply Best Algorithm**: Select and apply the recommended algorithm
5. **View Schedule**: Navigate to the schedule view to see Gantt chart and KPIs

---

## 📖 Usage Guide

### 1️⃣ Algorithm Comparison
- Navigate to **"📊 Compare Algorithms"** tab
- Click **"Compute All Algorithms"** to run all 6 heuristics (SPT, EDD, CR, PRIORITY, BALANCED, DEADLINE_FIRST)
- Review the **comparison table** with composite scores
- Adjust **priority weights** to match your business goals
- Enable **AI Analysis** for detailed recommendations

### 2️⃣ Viewing Schedules
- Select an algorithm from the comparison table
- Click **"✅ Apply Algorithm"**
- View **Gantt Chart** to visualize machine assignments
- Check **Operations** tab for detailed job status
- Monitor **KPI Dashboard** for performance metrics

### 3️⃣ Managing Jobs

#### Add New Job
1. Expand **"📋 2. Manage Jobs"** in sidebar
2. Enter job details (ID, quantity, priority, due date)
3. Configure operations (type, material, time, setup)
4. Click **"🔍 Analyze"** to check capacity
5. Click **"➕ Add Job"** if feasible

#### Delete Job
1. Select job from dropdown
2. Click **"Delete"** button
3. Schedule automatically recalculates

### 4️⃣ Advanced Settings

#### Machine Breakdown Simulator
1. Expand **"⚙️ 3. Advanced Settings"** in sidebar
2. Select machine from dropdown
3. Enter breakdown window in minutes (start time and duration)
4. Click **"🔧 Add Breakdown"** — you'll be prompted to recompute
5. Click **"Compute All Algorithms"** to see impact
6. View breakdowns in Gantt chart (red dashed rectangles with "🔧 DOWN" label)
7. Check **"Show Current Maintenance/Breakdowns"** to see all windows
8. Click **"Clear All Breakdowns"** to reset to original schedule

#### Priority Manager
- Select job from dropdown
- Choose new priority (1=urgent, 4=low)
- Click **"Update Priority"** — you'll be prompted to recompute
- Click **"Compute All Algorithms"** to update KPIs and schedules

#### Outsourcing Policy (Cost Threshold)
- Use the slider to set the cost threshold for outsourcing decisions
- Click **"Apply Threshold"** — you'll be prompted to recompute
- Click **"Compute All Algorithms"** so KPIs and assignments reflect the new threshold

#### Activity Log
- Navigate to **"📜 Activity Log"** to view a filterable audit trail of jobs, breakdowns, priorities, heuristic computations/applications, and policy changes

#### Capacity Planning
- Navigate to **"📐 Capacity Planning"** to run what‑if scenarios (add machines, extend shifts, adjust threshold) and see projected outsourcing/utilization impacts

---

## 📊 Data Format

### Required CSV Files (in `data/` folder)

#### `jobs_dataset.csv`
```csv
Job_ID,Operation_ID,Op_Seq,Part_Type,Quantity,Op_Type,Mat_Type,Tool_Group,Proc_Time_per_Unit,Setup_Time,Transfer_Min,Release_Day,Due_Day,Priority,Outsource_Flag,Vendor_Ref
J101,J101_Op1,1,A,150,MILLING,ALUM,TGA,0.43,30,5,7,14,3,Y,V_Mill_Std
J101,J101_Op2,2,A,150,TURNING,ALUM,TGB,0.27,20,5,7,14,3,N,
```

**Columns:**
- `Job_ID`: Unique job identifier
- `Operation_ID`: Unique operation identifier
- `Op_Seq`: Operation sequence (1, 2, 3...)
- `Quantity`: Number of parts
- `Op_Type`: MILLING, TURNING, GRINDING, or DRILLING
- `Mat_Type`: STEEL, ALUM, TITAN, or BRASS
- `Proc_Time_per_Unit`: Processing time per unit (minutes)
- `Setup_Time`: Setup time (minutes)
- `Priority`: 1 (urgent) to 4 (low)

#### `machine_data.csv`
```csv
Machine ID,Machine Type,Tool Capacity,Worker Requirement,Scheduled Maintenance (Day, Time-Time),Speed Factor,OEE (Uptime)
M1,MILLING,24,1,None,1,0.9
M6,TURNING/GRINDING,12,1,"Day 7, 09:00-12:00",1,0.85
```

#### `vendor_data.csv`
```csv
Vendor_ID,Op_Type_Specialty,Outsource_Lead_Time (Days),Outsource_Unit_Cost,Transport_Cost,Capacity_Limit,Quality_Factor
V_Mill_Std,MILLING/DRILLING (Alum/Brass),4,$0.75,$100,8000,0.99
```

#### `previous_next_material.csv`
```csv
Previous Material,Next Material,Penalty Time (min)
ALUM,STEEL,30
STEEL,TITAN,25
```

---

## ⚙️ Configuration

### Sample Size
Edit `cnc-scheduling.py` line ~2085:
```python
SAMPLE_SIZE = None  # Load all jobs
# or
SAMPLE_SIZE = 50   # Load first 50 jobs
```

### Machine Configuration
Switch between 2-machine (high utilization) and 5-machine (low utilization):

**Windows PowerShell:**
```powershell
.\switch_machine_config.ps1 -Mode 2  # High utilization (58-63%)
.\switch_machine_config.ps1 -Mode 5  # Low utilization (10-15%)
```

### Eligible Machines
Modify `get_eligible_machines()` function (line ~137) to match your machine setup:
```python
def get_eligible_machines(op_type):
    if op_type == 'MILLING':
        return ['M1', 'M3', 'M4']  # Your mill machines
    elif op_type == 'TURNING':
        return ['M6', 'M9']         # Your turning machines
    # ...
```

---

## 🤖 AI Features Setup

### Google Gemini API (Recommended)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Generate a free API key
3. Create `.env` file:
   ```env
   GEMINI_API_KEY=AIzaSy...your_key_here
   ```
4. Restart Streamlit

### AI Capabilities
- **Algorithm Explanation**: Why a specific heuristic is best
- **Performance Insights**: Bottleneck identification and quick wins
- **Data Quality Analysis**: Dataset health scoring (1-10)
- **Business Impact**: Plain-language recommendations

### Usage
- Enable AI in **"🤖 AI-Powered Analysis"** expander
- Check boxes for specific analysis types
- AI responses appear in ~5 seconds

**Free Tier Limits**: 60 requests/minute (sufficient for typical usage)

---

## 📁 Project Structure

```
ForbesMarshall/
├── cnc-scheduling.py           # Main Streamlit application
├── data/
│   ├── jobs_dataset.csv        # Job and operation data
│   ├── machine_data.csv        # Machine specifications
│   ├── vendor_data.csv         # Outsourcing vendor info
│   └── previous_next_material.csv  # Material changeover penalties
├── .env                        # API keys (create this)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── AI_FEATURES_GUIDE.md       # Detailed AI documentation
├── AI_INTEGRATION_SUMMARY.md  # AI implementation summary
├── diagnose_utilization.py    # Diagnostic utility script
└── switch_machine_config.ps1  # Machine config switcher (Windows)
```

---

## 🏗️ Architecture & Data Flow (Technical Stuff)

- **Data Layer**: CSV inputs → pandas DataFrames → validation and normalization
- **Make‑or‑Buy Module**: Computes in‑house vs outsource cost/time; applies `cost_threshold` to set `Assignment_Type`
- **Scheduling Engine**: Heuristic selector (SPT, EDD, CR, PRIORITY, BALANCED, DEADLINE_FIRST) + machine calendars + setup penalties
- **Metrics/KPIs**: Makespan, total tardiness, on‑time %, utilization, total cost, per‑machine details
- **Visualization**: Plotly Gantt + Streamlit tables
- **What‑If Tools**: Capacity Planning (machines, shifts, thresholds)
- **Audit Trail**: Activity Log with filter/export

Key configurable levers:
- `cost_threshold` (outsourcing aggressiveness)
- Job `Priority` per `Job_ID`
- Machine maintenance/breakdowns (minutes)
- Eligible machine mapping per `Op_Type`

---

## 🎨 Screenshots

### Algorithm Comparison
The main comparison view shows all 6 algorithms with composite scoring:
- Green highlighting for best scores
- Weighted ranking based on your priorities
- One-click algorithm application

### Gantt Chart
Interactive timeline visualization:
- Color-coded by machine
- Shows maintenance windows (striped red)
- Hover for job details (setup, processing, transfer times)

### KPI Dashboard
Real-time metrics display:
- Makespan (total completion time)
- Total tardiness (sum of all delays)
- On-time delivery percentage
- Machine utilization rates
- Total costs (in-house + outsourced)

---

## 🔧 Troubleshooting

### Low Utilization (~20% instead of 58-63%)
**Cause**: App using cached 5-machine config instead of 2-machine config

**Fix**:
```powershell
# 1. Switch config
.\switch_machine_config.ps1 -Mode 2

# 2. Restart Streamlit (Ctrl+C, then rerun)
streamlit run cnc-scheduling.py

# 3. Click "Reset" in sidebar
# 4. Click "Compute All Algorithms"
```

### Schedule Not Updating After Job Add/Delete
**Fix**: Click **"Compute All Algorithms"** to refresh all schedules

### AI Features Not Working
**Checklist**:
- ✅ `.env` file exists in project root
- ✅ `GEMINI_API_KEY` is set correctly
- ✅ `google-generativeai` package installed
- ✅ Internet connection active

### Diagnostic Script
Run diagnostics to check data integrity:
```bash
python diagnose_utilization.py
```

---

## 🧪 Testing

### Verify Installation
```bash
streamlit run cnc-scheduling.py
# Should open browser without errors
```

### Test Workflow
1. ✅ App loads successfully
2. ✅ Click "Compute All Algorithms" (sidebar)
3. ✅ Comparison table displays 6 rows
4. ✅ Apply an algorithm → View schedule
5. ✅ Gantt chart renders
6. ✅ KPI dashboard shows metrics
7. ✅ Add a test job → Analysis runs
8. ✅ Delete a job → Schedule updates

---

## 📈 Performance

### Benchmarks (200 jobs, 2 machines)
- **Data Loading**: ~2-3 seconds
- **Single Algorithm**: ~3-5 seconds
- **All 6 Algorithms**: ~20-28 seconds
- **AI Analysis**: ~5-10 seconds
- **Gantt Chart Render**: ~1-2 seconds

### Optimization Tips
- Use `SAMPLE_SIZE` for faster testing
- Reduce AI analysis frequency
- Clear cache periodically (Reset button)

---

## 🚀 Future Scope & Improvements

- **Exact Optimization**: Add MILP/CP-SAT models (OR‑Tools, Pyomo) for optimal scheduling under constraints; use heuristics as warm starts.
- **Learning‑based Scheduling**: Reinforcement learning or contextual bandits to adapt dispatching rules by shop state.
- **Predictive Maintenance**: Failure‑probability driven preventive maintenance windows; integrate sensor/OEE feeds.
- **Dynamic Vendor Modeling**: Price ladders, lead‑time SLAs, and capacity ceilings per vendor with historical performance.
- **Cost Model Refinements**: Expose in‑house hourly rate, setup penalties, and overhead absorption as configurable admin settings.
- **Scenario Management**: Save/compare what‑if scenarios (machines, shifts, thresholds) with diff reports.
- **Multi‑User & Auth**: Roles (Planner, Supervisor, Viewer), approvals for policy changes, AzureAD/Google SSO.
- **Database & API**: Persist data in Postgres; REST/GraphQL API for MES/ERP integration.
- **Realtime Dispatch**: Event‑driven rescheduling on job arrivals/breakdowns with minimal schedule disruption.
- **Simulation & Robustness**: Monte‑Carlo variability (proc/setup/arrival) with risk‑aware KPIs and buffers.
- **QA & Testing**: Unit tests for decision logic, deterministic seeds, CI pipeline; profiling for hotspots.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Test all changes locally before submitting
- Update README.md if adding new features

### Areas for Improvement
- [ ] Multi-objective optimization (Pareto frontier)
- [ ] Real-time scheduling with job arrivals
- [ ] Machine learning-based scheduling
- [ ] Multi-facility support
- [ ] REST API for integration
- [ ] Mobile-responsive UI

---

## 📝 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Krishna Heda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Krishna Heda**  
GitHub: [@krishnaheda14](https://github.com/krishnaheda14)  
Repository: [ForbesMarshall](https://github.com/krishnaheda14/ForbesMarshall)

---

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **Plotly** - For interactive visualizations
- **Google Gemini** - For AI-powered insights
- **Pandas** - For data manipulation
- **NumPy** - For numerical computations

---

## 📚 Additional Resources

- [AI Features Guide](AI_FEATURES_GUIDE.md) - Detailed AI documentation
- [AI Integration Summary](AI_INTEGRATION_SUMMARY.md) - Implementation details
- [Streamlit Documentation](https://docs.streamlit.io/)
- [CNC Scheduling Theory](https://en.wikipedia.org/wiki/Job_shop_scheduling)

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover this tool.

[![Star on GitHub](https://img.shields.io/github/stars/krishnaheda14/ForbesMarshall?style=social)](https://github.com/krishnaheda14/ForbesMarshall)

---

**Built with ❤️ for Manufacturing Excellence**
