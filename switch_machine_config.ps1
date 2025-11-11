# CNC Scheduler - Machine Configuration Switcher
# This script helps you switch between 2-machine (high utilization) 
# and 5-machine (low utilization) configurations

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("2", "5", "high", "low")]
    [string]$Mode
)

$dataPath = "data"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CNC Scheduler Configuration Switcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Normalize mode
if ($Mode -eq "high") { $Mode = "2" }
if ($Mode -eq "low") { $Mode = "5" }

if ($Mode -eq "2") {
    Write-Host "Switching to HIGH UTILIZATION mode (2 machines)..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if high util file exists
    if (Test-Path "$dataPath\machine_data_high_util.csv") {
        Copy-Item "$dataPath\machine_data_high_util.csv" "$dataPath\machine_data.csv" -Force
        Write-Host "✓ Activated 2-machine configuration:" -ForegroundColor Green
        Write-Host "  - M1 (MILLING)" -ForegroundColor White
        Write-Host "  - M6 (TURNING/GRINDING)" -ForegroundColor White
        Write-Host ""
        Write-Host "Expected Utilization: ~58-63%" -ForegroundColor Green
        Write-Host ""
        Write-Host "✓ IMPORTANT: Update get_eligible_machines() in code to:" -ForegroundColor Yellow
        Write-Host "  MILLING/DRILLING -> ['M1']" -ForegroundColor White
        Write-Host "  TURNING/GRINDING -> ['M6']" -ForegroundColor White
    } else {
        Write-Host "✗ Error: machine_data_high_util.csv not found!" -ForegroundColor Red
        Write-Host "Creating it now..." -ForegroundColor Yellow
        
        $highUtilContent = @"
Machine ID,Machine Type,Tool Capacity,Worker Requirement,"Scheduled Maintenance (Day, Time-Time)",Speed Factor,OEE (Uptime)
M1,MILLING,24,1,None,1,0.9
M6,TURNING/GRINDING,12,1,"Day 7, 09:00-12:00",1,0.85
"@
        $highUtilContent | Out-File "$dataPath\machine_data_high_util.csv" -Encoding UTF8
        Copy-Item "$dataPath\machine_data_high_util.csv" "$dataPath\machine_data.csv" -Force
        Write-Host "✓ Created and activated 2-machine configuration" -ForegroundColor Green
    }
    
} elseif ($Mode -eq "5") {
    Write-Host "Switching to LOW UTILIZATION mode (5 machines)..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if original 5-machine file exists
    if (Test-Path "$dataPath\machine_data_original_5_machines.csv") {
        Copy-Item "$dataPath\machine_data_original_5_machines.csv" "$dataPath\machine_data.csv" -Force
        Write-Host "✓ Activated 5-machine configuration:" -ForegroundColor Green
        Write-Host "  - M1, M3, M4 (MILLING)" -ForegroundColor White
        Write-Host "  - M6, M9 (TURNING/GRINDING)" -ForegroundColor White
        Write-Host ""
        Write-Host "Expected Utilization: ~10-15%" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "✓ IMPORTANT: Update get_eligible_machines() in code to:" -ForegroundColor Yellow
        Write-Host "  MILLING/DRILLING -> ['M1', 'M3', 'M4']" -ForegroundColor White
        Write-Host "  TURNING/GRINDING -> ['M6', 'M9']" -ForegroundColor White
    } else {
        Write-Host "✗ Error: machine_data_original_5_machines.csv not found!" -ForegroundColor Red
        Write-Host "This should have been created automatically." -ForegroundColor Yellow
        Write-Host "Please restore from backup or use 2-machine mode." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Restart your Streamlit app" -ForegroundColor White
Write-Host "2. Clear cache (in sidebar)" -ForegroundColor White
Write-Host "3. Reload data" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
