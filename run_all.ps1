$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $root "logs"
$pidFile = Join-Path $root ".run_pids.json"

if (Test-Path $pidFile) {
  Remove-Item -LiteralPath $pidFile -Force
}

if (Test-Path $logsDir) {
  Remove-Item -LiteralPath $logsDir -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

Write-Host "Starting FairSight Core backend on http://127.0.0.1:8001 ..."
$backend = Start-Process -FilePath "python" `
  -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001" `
  -WorkingDirectory $root `
  -RedirectStandardOutput (Join-Path $logsDir "backend.out.log") `
  -RedirectStandardError (Join-Path $logsDir "backend.err.log") `
  -PassThru

Write-Host "Starting FairSight Core frontend on http://127.0.0.1:5173 ..."
$frontend = Start-Process -FilePath "npm.cmd" `
  -ArgumentList "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173" `
  -WorkingDirectory (Join-Path $root "frontend") `
  -RedirectStandardOutput (Join-Path $logsDir "frontend.out.log") `
  -RedirectStandardError (Join-Path $logsDir "frontend.err.log") `
  -PassThru

$state = @{
  backend_pid = $backend.Id
  frontend_pid = $frontend.Id
  started_at = (Get-Date).ToString("o")
}

$state | ConvertTo-Json | Set-Content -Path $pidFile

Write-Host "Started."
Write-Host "Backend PID: $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Backend logs: logs\\backend.out.log / logs\\backend.err.log"
Write-Host "Frontend logs: logs\\frontend.out.log / logs\\frontend.err.log"
Write-Host "Use stop_all.ps1 to stop both services."
