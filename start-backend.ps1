# Start Backend Server
Write-Host "🚀 Starting CNC Scheduler Backend..." -ForegroundColor Cyan

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Navigate to backend
Set-Location backend

# Start server
Write-Host "Starting FastAPI server on http://localhost:8001" -ForegroundColor Green
python main.py
