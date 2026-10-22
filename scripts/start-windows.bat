@echo off
REM Easy ML Viewer - Windows launcher (CMD). Phase 40 hardened.
REM Boots backend 127.0.0.1:30211 + frontend 127.0.0.1:30222, 100%% offline.
REM Kills stale EasyML port holders pre-flight; closes both windows on exit.
setlocal
cd /d "%~dp0\.."
where python >nul 2>nul || (echo [ERROR] Python 3.11+ not found on PATH. & pause & exit /b 1)
where node >nul 2>nul || (echo [ERROR] Node.js 20+ not found on PATH. & pause & exit /b 1)

echo [1/3] Sweeping stale ports 30211 and 30222 (EasyML only)...
for %%p in (30211 30222) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p " ^| findstr LISTENING') do (
    echo   Stale holder on port %%p -^> PID %%a, terminating...
    taskkill /F /PID %%a >nul 2>nul
  )
)

echo [2/3] Starting backend on 127.0.0.1:30211 ...
start "EasyML-Backend:30211" cmd /k "cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 30211"
timeout /t 3 /nobreak >nul

echo [3/3] Starting frontend on 127.0.0.1:30222 ...
start "EasyML-Frontend:30222" cmd /k "cd frontend && npm run dev -- --host 127.0.0.1 --port 30222 --strictPort"
timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:30222"
echo.
echo Done. 100%% offline at http://127.0.0.1:30222
echo To stop cleanly: close the two EasyML windows, or run scripts\stop-windows.bat
endlocal
