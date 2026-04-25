$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root ".run_pids.json"

if (-not (Test-Path $pidFile)) {
  Write-Host "No .run_pids.json found. Nothing to stop."
  exit 0
}

$pids = Get-Content $pidFile | ConvertFrom-Json
$targets = @($pids.backend_pid, $pids.frontend_pid) | Where-Object { $_ }

foreach ($targetPid in $targets) {
  try {
    Stop-Process -Id $targetPid -Force -ErrorAction Stop
    Write-Host "Stopped PID $targetPid"
  } catch {
    Write-Host "PID $targetPid already stopped or unavailable"
  }
}

Remove-Item $pidFile -Force
Write-Host "All done."
