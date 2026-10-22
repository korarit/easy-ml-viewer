@echo off
REM Easy ML Viewer - stop helper: kills processes holding 30211/30222.
for %%p in (30211 30222) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%p " ^| findstr LISTENING') do (
    echo Stopping PID %%a on port %%p ...
    taskkill /F /PID %%a >nul 2>nul
  )
)
echo All EasyML ports released.
