# run_figures.ps1
# One-click runner for Windows PowerShell. Run it from anywhere:
#
#   powershell -ExecutionPolicy Bypass -File "$HOME\Documents\G1_hardware_analysis\run_figures.ps1"
#
# First run: creates a private Python environment (.venv) inside this folder and
# installs numpy, pandas and matplotlib into it. Later runs reuse it.
# Any extra arguments are passed straight to make_figs.py, for example:
#
#   ...\run_figures.ps1 --grasp-run 20260918_185109
#   ...\run_figures.ps1 --data "C:\path\to\Real_logs"

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# 1. Find a Python 3 interpreter (the "py" launcher first, then "python").
$python = $null
foreach ($cand in @("py", "python", "python3")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        $ver = & $cand -c "import sys; print(sys.version_info[0])" 2>$null
        if ($ver -eq "3") { $python = $cand; break }
    }
}
if (-not $python) {
    Write-Host "[ERROR] Python 3 not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and run this again." -ForegroundColor Red
    exit 1
}

# 2. Create the private environment once.
# Windows PowerShell 5.1 has no $IsWindows variable (it is always Windows there).
$onWindows = ($null -eq $IsWindows) -or $IsWindows
if ($onWindows) { $venvPy = Join-Path $here ".venv\Scripts\python.exe" }
else            { $venvPy = Join-Path $here ".venv/bin/python" }
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating the Python environment in .venv (first run only)..."
    & $python -m venv (Join-Path $here ".venv")
    & $venvPy -m pip install --upgrade pip --quiet
    & $venvPy -m pip install -r (Join-Path $here "requirements.txt") --quiet
}

# 3. Make the figures.
& $venvPy (Join-Path $here "make_figs.py") @args
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Figures are in: $(Join-Path $here 'figures')" -ForegroundColor Green
    if ($onWindows) { Start-Process (Join-Path $here "figures") }   # opens the folder in Explorer
}
