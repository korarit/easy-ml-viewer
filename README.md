# Easy ML Viewer — Phase 01 Scaffolding

Monorepo · 2 ports · DDD Shared Core · 100% offline.

## Quick start

**Backend** (`127.0.0.1:30211`):
```bash
cd backend
pip install -e ".[dev]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 30211
```

**Frontend** (`127.0.0.1:30222`):
```bash
cd frontend
npm install
npm run dev
```

Or use one-click launchers in `scripts/`:
- Windows CMD → `scripts\start-windows.bat`
- Windows PowerShell → `scripts\start-windows.ps1`
- macOS / Linux → `bash scripts/start-unix.sh`

## Verify (DoD)

```bash
cd backend && python -m pytest -q
curl http://127.0.0.1:30211/api/v1/system/health
```

Open `http://localhost:30222` with Wi-Fi off — page must load < 1s with
zero external requests (check DevTools Network tab).
