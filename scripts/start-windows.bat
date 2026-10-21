@echo off
REM Easy ML Viewer — Windows launcher (CMD). Kills stale ports, boots backend+frontend.
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo [ERROR] Python 3.11+ not found on PATH. & pause & exit /b 1)
where node >nul 2>nul || (echo [ERROR] Node.js 20+ not found on PATH. & pause & exit /b 1)

echo [1/3] Freeing ports 30211 and 30222 (if stale)...
for %%p in (30211 30222) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p " ^| findstr LISTENING') do (
    echo   Killing PID %%a on port %%p
    taskkill /F /PID %%a >nul 2>nul
  )
)

echo [2/3] Starting backend :30211 ...
start "EasyML-Backend:30211" cmd /k "cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 30211"
timeout /t 3 /nobreak >nul

echo [3/3] Starting frontend :30222 ...
start "EasyML-Frontend:30222" cmd /k "cd frontend && npm run dev"
timeout /t 3 /nobreak >nul
start "" "http://localhost:30222"
echo Done. Close the two terminal windows to stop (or run scripts\stop-windows.bat).
endlocal
