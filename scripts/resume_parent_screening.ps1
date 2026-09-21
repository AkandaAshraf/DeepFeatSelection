$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $repoRoot 'ExpOutput\parent_screening_laptop'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runnerPath = Join-Path $PSScriptRoot 'parent_screening_laptop.py'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $runnerPath + '"')) -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputRoot "run_$stamp.log") -RedirectStandardError (Join-Path $outputRoot "run_$stamp.err") -PassThru
Write-Output "Started PID $($process.Id). Logs: $outputRoot\run_$stamp.log"
