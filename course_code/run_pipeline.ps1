# Runs the remaining GPU experiments one after another (logs in logs\).
# Usage: powershell -ExecutionPolicy Bypass -File course_code\run_pipeline.ps1 [-WaitPid <pid>]
param([int]$WaitPid = 0)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if ($WaitPid) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 20 } }
Remove-Item results\ppe\*\RUNNING -ErrorAction SilentlyContinue

function Step($name, $argList) {
    "[$(Get-Date -Format HH:mm:ss)] start $name" | Add-Content logs\pipeline.log
    & $py -u @argList *> "logs\$name.log"
    "[$(Get-Date -Format HH:mm:ss)] done  $name (exit $LASTEXITCODE)" | Add-Content logs\pipeline.log
}

Step 'test_ppe' @('course_code\test_ppe.py')
Step 'train_esmoen' @('course_code\train_ppe.py', '--model', 'YOLO-Master-EsMoE-N', '--epochs', '50', '--batch', '8', '--workers', '4')
Step 'test_ppe_all' @('course_code\test_ppe.py')
Step 'ablation_routing' @('course_code\ablation_routing.py')
Step 'alarm_analysis' @('course_code\alarm_analysis.py')
"[$(Get-Date -Format HH:mm:ss)] pipeline finished" | Add-Content logs\pipeline.log
