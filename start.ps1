# Quick Start Script for CNC Scheduler
# Run this script from PowerShell to start both backend and frontend

Write-Host "🏭 CNC Scheduler - Quick Start" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if Python is installed
Write-Host "`n[1/5] Checking Python..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "✅ $pythonVersion found" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit
}

# Check if Node.js is installed
Write-Host "`n[2/5] Checking Node.js..." -ForegroundColor Yellow
if (Get-Command node -ErrorAction SilentlyContinue) {
    $nodeVersion = node --version
    Write-Host "✅ Node.js $nodeVersion found" -ForegroundColor Green
} else {
    Write-Host "❌ Node.js not found. Please install Node.js 16+" -ForegroundColor Red
    exit
}

# Backend setup
Write-Host "`n[3/5] Setting up Backend..." -ForegroundColor Yellow

if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install -q -r backend\requirements.txt

# Frontend setup
Write-Host "`n[4/5] Setting up Frontend..." -ForegroundColor Yellow
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Cyan
    cd frontend
    npm install --silent
    cd ..
}

# Check for .env file
Write-Host "`n[5/5] Checking configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating template..." -ForegroundColor Yellow
    "GEMINI_API_KEY=your_api_key_here" | Out-File -FilePath .env -Encoding utf8
    Write-Host "📝 Please edit .env and add your Gemini API key" -ForegroundColor Cyan
}

# Start servers
Write-Host "`n✅ Setup complete! Starting servers..." -ForegroundColor Green
Write-Host "`n🔹 Backend will run on: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🔹 Frontend will run on: http://localhost:3000" -ForegroundColor Cyan
Write-Host "🔹 API docs available at: http://localhost:8000/docs`n" -ForegroundColor Cyan

# Start backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; cd backend; python main.py"

# Wait a bit for backend to start
Start-Sleep -Seconds 3

# Start frontend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm run dev"

Write-Host "🚀 Servers starting in separate windows..." -ForegroundColor Green
Write-Host "📖 Check README_REACT_APP.md for usage instructions" -ForegroundColor Cyan
