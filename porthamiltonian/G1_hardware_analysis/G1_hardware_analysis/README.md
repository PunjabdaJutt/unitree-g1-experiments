# G1 hardware logs: analysis figures

Seven figures that explain the right-arm bench test (PH torque law vs position-PD)
and one grasp run of the Unitree G1, recorded on 18 September 2026.
The full explanation of the code and of every figure is in
`report/G1_hardware_analysis_report.pdf`.

## Folder layout

```
Documents\G1_hardware_analysis\
    make_figs.py            the analysis script
    run_figures.ps1         one-click runner for PowerShell
    requirements.txt        numpy, pandas, matplotlib
    data\                   the robot logs (CSV) + Ammar's REAL_LOGS_ARCHITECTURE.md
    figures\                output: fig1 ... fig7 as PNG and PDF, results_summary.csv
    report\                 the PDF report
```

## 1. Put the folder in Documents (once)

Unzip `G1_hardware_analysis.zip` into Documents. In PowerShell:

```powershell
$docs = [Environment]::GetFolderPath("MyDocuments")   # works with OneDrive too
Expand-Archive -Path "$HOME\Downloads\G1_hardware_analysis.zip" -DestinationPath $docs
cd "$docs\G1_hardware_analysis"
```

## 2. Run it

Easiest (creates a private Python environment on the first run, then makes the figures):

```powershell
powershell -ExecutionPolicy Bypass -File .\run_figures.ps1
```

Or by hand, if Python 3 with numpy, pandas and matplotlib is already installed:

```powershell
python make_figs.py
```

## 3. Options

```powershell
python make_figs.py --data "C:\path\to\Real_logs"      # read the logs from another folder
python make_figs.py --grasp-run 20260918_185109          # choose one grasp run by its timestamp
python make_figs.py --bench-ph 20260918_183908 --bench-pd 20260918_184048
python make_figs.py --out "C:\path\to\figures"          # write the figures elsewhere
```

Without options the script uses the newest file of each kind in `data\`.
The same options work after `run_figures.ps1`.
