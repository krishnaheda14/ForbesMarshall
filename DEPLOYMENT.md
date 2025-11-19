# Deployment & Setup Guide

## 📋 Prerequisites Checklist

Before starting, verify you have:

- [ ] Python 3.11 or higher installed
- [ ] Node.js 18 or higher installed
- [ ] Git installed
- [ ] Google Gemini API key (free tier available)
- [ ] Terminal/PowerShell access
- [ ] ~500MB free disk space

## 🎯 Complete Setup Steps

### For First-Time Users (Clone from GitHub)

#### 1. Clone Repository

```bash
git clone https://github.com/krishnaheda14/ForbesMarshall.git
cd ForbesMarshall
```

#### 2. Python Backend Setup

**Step 2.1: Create Virtual Environment**

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 2.2: Install Dependencies**

```bash
cd backend
pip install -r requirements.txt
cd ..
```

**Step 2.3: Configure Environment**

Create `.env` file in project root:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key: https://makersuite.google.com/app/apikey

**Step 2.4: Verify Data Files**

Check that these files exist in `data/` folder:
- ✅ `jobs_dataset.csv`
- ✅ `machine_data.csv`
- ✅ `vendor_data.csv`
- ✅ `previous_next_material.csv`

#### 3. React Frontend Setup

**Step 3.1: Install Node Dependencies**

```bash
cd frontend
npm install
```

**Step 3.2: Configure Frontend (Optional)**

Create `frontend/.env` if API is not on localhost:8001:
```env
VITE_API_URL=http://localhost:8001
```

#### 4. Start Application

**You need TWO terminal windows running simultaneously:**

**Terminal 1 - Backend:**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start backend server
cd backend
python main.py
```

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
```

**Terminal 2 - Frontend:**
```powershell
# Start frontend dev server
cd frontend
npm run dev
```

Wait for:
```
➜  Local:   http://localhost:5173/
```

#### 5. Access Application

Open browser: **http://localhost:5173**

First-time steps in UI:
1. Click **"Load Data"** button
2. Click **"Compute All Heuristics"**
3. Explore the features!

---

## 🔄 For Team Members (Pulling Updates)

### Initial Setup (First Time)

```bash
# Clone repository
git clone https://github.com/krishnaheda14/ForbesMarshall.git
cd ForbesMarshall

# Setup backend
python -m venv venv
.\venv\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
cd ..

# Setup frontend
cd frontend
npm install
cd ..

# Create .env file
# Add: GEMINI_API_KEY=your_key_here
```

### Daily Updates (Regular Use)

```bash
# Pull latest changes
git pull origin main

# Update backend dependencies (if requirements.txt changed)
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Update frontend dependencies (if package.json changed)
cd frontend
npm install
cd ..

# Start servers (2 terminals)
# Terminal 1: .\venv\Scripts\Activate.ps1; cd backend; python main.py
# Terminal 2: cd frontend; npm run dev
```

---

## 📤 Pushing to GitHub (For Developers)

### First Time Push

**Step 1: Verify .gitignore**

Ensure `.gitignore` includes:
```gitignore
# Node modules
node_modules/
frontend/node_modules/

# Python virtual environment
venv/
__pycache__/

# Environment variables
.env
.env.local

# Build outputs
frontend/dist/
frontend/build/
```

**Step 2: Check What Will Be Committed**

```bash
git status
```

**Should NOT see**:
- ❌ `node_modules/`
- ❌ `venv/`
- ❌ `.env`
- ❌ `__pycache__/`

**Should see**:
- ✅ `.py` files
- ✅ `.jsx` files
- ✅ `requirements.txt`
- ✅ `package.json`
- ✅ `.csv` files in `data/`

**Step 3: Add and Commit**

```bash
# Add all changes
git add .

# Commit with message
git commit -m "Add Excel import and breakdown features"

# Push to GitHub
git push origin main
```

### Regular Updates

```bash
# Check current status
git status

# Add modified files
git add .

# Commit changes
git commit -m "Your descriptive message here"

# Pull latest changes (avoid conflicts)
git pull origin main

# Push your changes
git push origin main
```

### Handling Conflicts

If you get merge conflicts:

```bash
# Pull with rebase
git pull --rebase origin main

# Fix conflicts in files marked by Git
# Look for <<<<<<, ======, >>>>>> markers

# After fixing
git add .
git rebase --continue

# Push
git push origin main
```

---

## 🏗️ Project File Size Guide

**Will be pushed** (~50MB):
- Source code (.py, .jsx, .js files) - ~5MB
- CSV data files - ~10MB
- Configuration files - ~1MB
- Documentation - ~1MB

**Will NOT be pushed** (~300MB):
- `node_modules/` - ~250MB
- `venv/` - ~50MB
- `__pycache__/` - ~5MB

**Total GitHub repo size**: ~50-60MB

---

## ✅ Verification Checklist

After setup, verify everything works:

### Backend Verification

```bash
# Check backend is running
curl http://localhost:8001

# Expected: {"message":"CNC Scheduling API v2.0","status":"running"}
```

### Frontend Verification

Open browser to http://localhost:5173 and check:

- [ ] Dashboard loads without errors
- [ ] Sidebar menu is visible
- [ ] "Load Data" button works
- [ ] "Compute All Heuristics" shows progress
- [ ] Gantt Chart displays properly
- [ ] Excel Upload page is accessible

### Data Verification

```bash
# Check data files exist
ls data/

# Should show:
# jobs_dataset.csv
# machine_data.csv
# vendor_data.csv
# previous_next_material.csv
```

---

## 🔧 Common Setup Issues

### Issue: Virtual Environment Activation Fails

**Windows PowerShell Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux Permission Error:**
```bash
chmod +x venv/bin/activate
```

### Issue: Port Already in Use

**Backend (8001):**
```powershell
# Windows
Get-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess | Stop-Process

# macOS/Linux
lsof -ti:8001 | xargs kill -9
```

**Frontend (5173):**
```powershell
# Windows
Get-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess | Stop-Process

# macOS/Linux
lsof -ti:5173 | xargs kill -9
```

### Issue: Python Module Not Found

```bash
# Reinstall all dependencies
pip install -r backend/requirements.txt --force-reinstall
```

### Issue: Node Module Not Found

```bash
# Clean install
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Issue: Gemini API Key Invalid

1. Get new key: https://makersuite.google.com/app/apikey
2. Update `.env` file
3. Restart backend server

---

## 🚀 Quick Start Scripts

### Windows PowerShell Scripts

**start-backend.ps1:**
```powershell
.\venv\Scripts\Activate.ps1
cd backend
python main.py
```

**start-frontend.ps1:**
```powershell
cd frontend
npm run dev
```

**Usage:**
```powershell
# Terminal 1
.\start-backend.ps1

# Terminal 2
.\start-frontend.ps1
```

### macOS/Linux Bash Scripts

**start-backend.sh:**
```bash
#!/bin/bash
source venv/bin/activate
cd backend
python main.py
```

**start-frontend.sh:**
```bash
#!/bin/bash
cd frontend
npm run dev
```

**Usage:**
```bash
chmod +x start-backend.sh start-frontend.sh

# Terminal 1
./start-backend.sh

# Terminal 2
./start-frontend.sh
```

---

## 📊 What Gets Committed vs Ignored

### ✅ Committed to GitHub

```
ForbesMarshall/
├── backend/
│   ├── *.py                    ✅ All Python source
│   ├── requirements.txt        ✅ Dependencies list
│   └── models.py, etc.         ✅ All source files
├── frontend/
│   ├── src/                    ✅ All source code
│   ├── package.json            ✅ Dependencies list
│   ├── package-lock.json       ✅ Dependency lock
│   └── vite.config.js          ✅ Configuration
├── data/
│   └── *.csv                   ✅ Sample data
├── .gitignore                  ✅ Git rules
├── README.md                   ✅ Documentation
└── *.md                        ✅ All docs
```

### ❌ Ignored by Git

```
ForbesMarshall/
├── venv/                       ❌ Virtual environment
├── __pycache__/                ❌ Python cache
├── frontend/
│   ├── node_modules/           ❌ Node packages
│   ├── dist/                   ❌ Build output
│   └── .vite/                  ❌ Vite cache
├── .env                        ❌ Secrets/API keys
└── *.pyc                       ❌ Compiled Python
```

---

## 🎓 Best Practices

### For Development

1. **Always pull before you push**
   ```bash
   git pull origin main
   # Make changes
   git push origin main
   ```

2. **Use meaningful commit messages**
   ```bash
   # Good
   git commit -m "Add Excel import with AI mapping"
   
   # Bad
   git commit -m "update"
   ```

3. **Test before committing**
   - Load data successfully
   - Compute heuristics without errors
   - Check Gantt chart displays properly

4. **Never commit sensitive data**
   - API keys → use `.env`
   - Database passwords → use `.env`
   - Personal data → exclude from commits

### For Collaboration

1. **Use branches for features**
   ```bash
   git checkout -b feature/new-algorithm
   # Make changes
   git commit -m "Add new scheduling algorithm"
   git push origin feature/new-algorithm
   # Create Pull Request on GitHub
   ```

2. **Keep dependencies updated**
   ```bash
   # After adding new Python package
   pip freeze > backend/requirements.txt
   
   # After adding new npm package
   cd frontend
   npm install <package> --save
   ```

3. **Document changes**
   - Update README.md for new features
   - Add comments to complex code
   - Create issue for bugs

---

## 📞 Support Resources

- **GitHub Issues**: https://github.com/krishnaheda14/ForbesMarshall/issues
- **Python Docs**: https://docs.python.org/3/
- **React Docs**: https://react.dev/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Gemini API**: https://ai.google.dev/tutorials/python_quickstart

---

## 🎯 Next Steps After Setup

1. **Explore the Dashboard**
   - Load sample data
   - Try different algorithms
   - Compare results

2. **Test Excel Import**
   - Upload your own Excel file
   - Review AI column mapping
   - Schedule imported jobs

3. **Experiment with Features**
   - Simulate machine breakdowns
   - Update job priorities
   - Adjust outsourcing thresholds

4. **Customize for Your Needs**
   - Modify data schemas
   - Add new algorithms
   - Customize UI themes

---

**Remember**: Both backend and frontend must be running simultaneously for the application to work!

Happy Scheduling! 🚀
