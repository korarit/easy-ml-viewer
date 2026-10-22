# Easy ML Viewer — Windows launcher (PowerShell, Phase 40 hardened).
# Manages 127.0.0.1:30211 (backend) + 127.0.0.1:30222 (frontend), 100% offline.
# Pre-flight stale-port sweeper + full process-TREE cleanup on Ctrl+C / window close.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Free-Port([int]$Port) {
  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    try {
      Write-Host "  Stale holder on port $Port -> PID $($c.OwningProcess), terminating tree..."
      & taskkill /T /F /PID $c.OwningProcess 2>$null | Out-Null
    } catch {}
  }
}

function Stop-Tree([int]$Pid) {
  try { & taskkill /T /F /PID $Pid 2>$null | Out-Null } catch {}
}

Write-Host '[1/3] Sweeping stale ports 30211 and 30222...'
Free-Port 30211; Free-Port 30222

Write-Host '[2/3] Starting backend on 127.0.0.1:30211 ...'
$backend = Start-Process -FilePath 'python' -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','30211' -WorkingDirectory "$repoRoot\backend" -PassThru
Start-Sleep -Seconds 3
if ($backend.HasExited) { Write-Error 'Backend failed to start (see error above).'; exit 1 }

Write-Host '[3/3] Starting frontend on 127.0.0.1:30222 ...'
$frontend = Start-Process -FilePath 'npm' -ArgumentList 'run','dev','--','--host','127.0.0.1','--port','30222','--strictPort' -WorkingDirectory "$repoRoot\frontend" -PassThru

Write-Host "Backend PID $($backend.Id) | Frontend PID $($frontend.Id)"
Write-Host 'Opening http://127.0.0.1:30222 ...'
Start-Process 'http://127.0.0.1:30222'

$cleanup = {
  Write-Host ''
  Write-Host 'Stopping EasyML servers...'
  Stop-Tree $backend.Id; Stop-Tree $frontend.Id
}
Register-EngineEvent PowerShell.Exiting -Action $cleanup | Out-Null
# Handle window-close (CloseMainWindow) the same way via finally.
Write-Host 'Press Ctrl+C to stop both servers cleanly.'
try { while ($true) { Start-Sleep -Seconds 5 } } finally { Stop-Tree $backend.Id; Stop-Tree $frontend.Id }
