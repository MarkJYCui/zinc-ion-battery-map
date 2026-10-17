# Autonomous Materials Acceleration Platform (MAP) for Aqueous ZIBs
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Autonomous Materials Acceleration Platform (MAP) for Aqueous ZIBs   " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

$VenvStreamlit = ".\.venv\Scripts\streamlit.exe"

if (Test-Path $VenvStreamlit) {
    Write-Host "Launching with verified project virtual environment (.venv)..." -ForegroundColor Green
    & $VenvStreamlit run app.py
} else {
    Write-Host "Warning: .venv not found. Falling back to system streamlit..." -ForegroundColor Yellow
    streamlit run app.py
}
