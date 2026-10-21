# Easy ML Viewer — Windows launcher (PowerShell with clean SIGINT handling).
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Free-Port([int]$Port) {
  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host "  Killed PID $($c.OwningProcess) on port $Port" } catch {}
  }
}

Write-Host '[1/3] Freeing ports 30211 and 30222...'
Free-Port 30211; Free-Port 30222

Write-Host '[2/3] Starting backend :30211 ...'
$backend = Start-Process -FilePath 'python' -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','30211' -WorkingDirectory "$PWD\backend" -PassThru
Start-Sleep -Seconds 3

Write-Host '[3/3] Starting frontend :30222 ...'
$frontend = Start-Process -FilePath 'npm' -ArgumentList 'run','dev' -WorkingDirectory "$PWD\frontend" -PassThru

Write-Host "Backend PID $($backend.Id) | Frontend PID $($frontend.Id)"
Write-Host 'Opening http://localhost:30222 ...'
Start-Process 'http://localhost:30222'

Register-EngineEvent PowerShell.Exiting -Action {
  Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
  Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
} | Out-Null

Write-Host 'Press Ctrl+C to stop both servers.'
try { while ($true) { Start-Sleep -Seconds 5 } } finally {
  Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
  Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
}
