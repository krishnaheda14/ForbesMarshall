# Start Frontend Development Server
Write-Host "🎨 Starting CNC Scheduler Frontend..." -ForegroundColor Cyan

# Navigate to frontend
Set-Location frontend

# Start Vite dev server
Write-Host "Starting Vite server on http://localhost:5173" -ForegroundColor Green
npm run dev
