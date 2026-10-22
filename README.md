# Easy ML Viewer

100% offline, monorepo ML visualization studio. Inspect any scikit-learn / XGBoost / LightGBM / CatBoost / ONNX model in 3D — upload a dataset, upload a model, map columns, and fly through predictions on an interactive WebGL stage with VCR playback, what-if analysis, and export studio.

- **Backend:** FastAPI + DDD (`127.0.0.1:30211`), 16 routers under `/api/v1`
- **Frontend:** React 19 + Vite + Three.js (`127.0.0.1:30222`), 5-step studio wizard
- **Offline-first:** zero external requests; all caching under `.cache/`

## Architecture

```
easy-ml-viewer/
├── backend/                 # FastAPI, DDD shared core
│   ├── app/
│   │   ├── core/            # config, logging, exceptions, events,
│   │   │                    # memory_guard/monitor, compute_pool,
│   │   │                    # binary_codec (msgpack+zstd), math_3d
│   │   ├── domains/         # compute_engine, dataset, explainability,
│   │   │                    # export_reporting, mapping, model_engine,
│   │   │                    # playback, recording_mlv, simulation,
│   │   │                    # spatial_projection, studio_metadata,
│   │   │                    # system, timeline, whatif
│   │   ├── application/     # orchestrators + use cases
│   │   ├── infrastructure/  # parquet repo, tabular reader,
│   │   │                    # joblib loader, cache manager
│   │   ├── presentation/    # api_router + 16 endpoint modules
│   │   └── main.py          # thread-pinning first, cache sweep, CORS
│   └── tests/               # 85 test files, grouped per phase
├── frontend/                # React 19 + Vite 6 + Tailwind 4 + R3F + Zustand
│   └── src/
│       ├── core/            # api client, stores, theme, components
│       └── domains/         # dataset, model_engine, mapping,
│                            # trajectory_queue, stage3d, evaluation,
│                            # explainability, projection3d, playback,
│                            # timeline, recording, export-studio,
│                            # compute, whatif, studio-ux
├── samples/                 # ~100 bundled CSV / joblib / ONNX / CBM pairs
├── scripts/                 # hardened one-click launchers
├── start-windows.bat / start-windows.ps1 / start-unix.sh
└── .cache/                  # git-ignored runtime Parquet + model cache
```

Backend startup order in `backend/app/main.py` matters: `apply_thread_pinning()` (OMP/MKL/OpenBLAS = 4 threads) runs **before** numpy/sklearn are ever imported, then `ensure_cache_dirs()`, then a stale-cache sweep (files older than 24h / over 1 GB quota) on every boot.

## Prerequisites

- Python >= 3.11
- Node.js 18+ with npm
- No internet required at runtime (npm/pip install needs it once)

## Quick Start

One-click launchers (sweep stale ports, start backend + frontend, open browser):

| OS | Command |
|----|---------|
| Windows CMD | `start-windows.bat` |
| Windows PowerShell | `.\start-windows.ps1` |
| macOS / Linux | `bash start-unix.sh` |
| Stop (Windows) | `stop-windows.bat` or Ctrl+C |

Manual start:

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

Open `http://127.0.0.1:30222`. If the backend is down, the UI shows an offline banner with retry + copy-command actions instead of failing silently.

## Studio Workflow (5 Steps)

The frontend (`frontend/src/App.tsx`) is a gated 5-step wizard — each step unlocks the next. Shortcuts: `Alt+1..5` jump steps, `Ctrl+Shift+D` diagnostics, `?` help.

| Step | View | What it does |
|------|------|--------------|
| 1 · Dataset | `domains/dataset/DatasetStepView` | Drag-drop CSV/TSV/XLSX/Parquet, sheet picker, column profiles, sample preview, memory-savings banner |
| 2 · Model | `domains/model_engine/ModelStepView` | Upload model artifacts, schema + architecture cards, per-algorithm inspector + 2D surface sandbox (full list in Supported Models) |
| 3 · Column Mapping | `domains/mapping/MappingStepView` | Feature/target row mapper with dry-run validation drawer |
| 4 · Trajectory Queue | `domains/trajectory_queue/TrajectoryQueueStepView` | Virtualized (`@tanstack/react-virtual`) row queue builder |
| 5 · 3D Stage & VCR | `domains/stage3d/Stage3DStepView` | WebGL stage (React Three Fiber) + 60 FPS VCR, evaluation lens, explainability overlays, export studio |

Cross-cutting UI: `StudioHeader`, `SystemStatusBar` (RAM + backend health, 10s poll), `SystemDiagnosticsModal` (Force GC), `ChunkedJobProgressBanner`, `GlobalErrorBoundary`, low-memory warning banner.

## API Reference

Base path: `http://127.0.0.1:30211/api/v1`. Interactive docs at `/docs` when the backend runs.

| Prefix | Tag | Purpose |
|--------|-----|---------|
| `/system` | system | `GET /health` (hardware readiness + RAM), `POST /gc` |
| `/datasets` | datasets | `POST /upload` (CSV/TSV/XLSX/Parquet + profiling), `POST /load-sample/{sample_name}` |
| `/datasets` | datasets-query | Filtered queries, residual/outlier slicing |
| `/models` | models | `POST /upload`, `POST /load-sample/{sample_key}`, `GET /supported`, `POST /{id}/trace-preview`, `POST /{id}/surface-preview` |
| `/mappings` | mappings | Column mapping CRUD + dry-run validation |
| `/evaluations` | evaluations | Metrics ribbon, confusion matrix, target-label mapping |
| `/explainability` | explainability | Per-frame feature contributions, waterfall panels |
| `/stage3d` | stage3d | 3D scene payloads per algorithm family |
| `/projection` | projection | Dimensionality-reduction projections |
| `/playback` | playback | VCR transport state |
| `/timeline` | timeline | Trajectory timeline + keyframes |
| `/recordings` | recording-mlv | `.mlv` session recording, dual-compare |
| `/whatif` | whatif | Counterfactual / perturbation runs |
| `/alias` | studio-alias | Human-readable run aliases |
| `/exports` | export-reporting | Report + video export jobs |
| `/compute` | compute-engine | Chunked job orchestration, hardware status, diagnostics |

## Supported Models

The live registry is served at `GET /models/supported` (plugin registry in `backend/app/domains/model_engine/registry.py` — new families hook in via decorator). Accepted artifacts: `.joblib` / `.pkl` for everything, plus native `.ubj` (XGBoost), `.txt` (LightGBM), `.cbm` (CatBoost), and `.onnx`. Unfitted estimators are rejected; `GridSearchCV` / `RandomizedSearchCV` / `Pipeline` / `ColumnTransformer` wrappers are unwrapped automatically.

| Family | Estimators | Notes |
|--------|-----------|-------|
| Linear / OLS | `LinearRegression` | Per-row equation trace + hyperplane mesh |
| Regularized linear | `Ridge`, `RidgeCV`, `RidgeClassifier`, `RidgeClassifierCV`, `Lasso`, `LassoCV`, `LassoLars`, `LassoLarsCV`, `LassoLarsIC` | L1/L2 inspector + surface sandbox |
| ElasticNet | `ElasticNet`, `ElasticNetCV`, `MultiTaskElasticNet`, `MultiTaskElasticNetCV` | Single- and multi-task |
| Logistic regression | `LogisticRegression`, `LogisticRegressionCV` | Binary (sigmoid) + multiclass (softmax) |
| Online linear | `SGDClassifier`, `SGDRegressor`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, `Perceptron` | Margin / tube views |
| Decision trees | `DecisionTreeClassifier/Regressor`, `ExtraTreeClassifier/Regressor` | Voxel-terrace 3D, node trace |
| Forests & bagging | `RandomForestClassifier/Regressor`, `ExtraTreesClassifier/Regressor` | Voting-arena 3D, per-tree agreement, uncertainty halo (ExtraTrees) |
| Boosting (sklearn) | `AdaBoostClassifier/Regressor`, `GradientBoostingClassifier/Regressor`, `HistGradientBoostingClassifier/Regressor` | Sequential cascade + quantized-bin views |
| XGBoost | `XGBClassifier`, `XGBRFClassifier`, `XGBRegressor`, `XGBRFRegressor`, native `xgb.Booster` | Exact TreeSHAP leaf-flow 3D; native `.ubj` supported |
| LightGBM | `LGBMClassifier`, `LGBMRegressor`, `LGBMRanker`, native `lgb.Booster` | Leaf-wise Exact TreeSHAP 3D; native `.txt` supported; ranker task |
| CatBoost | `CatBoostClassifier`, `CatBoostRegressor` | Symmetric-oblivious Exact SHAP; categorical indices honored; native `.cbm` supported |
| Nearest neighbors | `KNeighborsClassifier/Regressor`, `RadiusNeighborsClassifier/Regressor` | Radar-sphere 3D, distance inspector |
| SVM | `SVC`, `NuSVC`, `LinearSVC`, `SVR`, `NuSVR`, `LinearSVR`, `OneClassSVM` | Kernel-saddle / epsilon-tube views; outlier envelope (one-class) |
| Naive Bayes | `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `CategoricalNB` | Bell-hill / probability-tower 3D |
| Discriminant analysis | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` | Class-ellipsoid 3D |
| MLP | `MLPClassifier`, `MLPRegressor` | Floating-synapse 3D, layer inspector |
| sklearn Pipelines | Any `Pipeline` / `ColumnTransformer` ending in a supported estimator | Raw→transformed tunnel view, step breadcrumb, per-stage cards |
| ONNX classical | `ai.onnx.ml` graphs (tree-ensemble, linear, SVM ops) | Graph inspector card + classical stage router |
| ONNX tabular generic | KNN / Naive Bayes / scaler+encoder preprocessing chains | Generic tabular inspector |
| ONNX deep | `Gemm`/`MatMul` graphs | Latent-manifold 3D + intermediate-probe injector |
| Generic fallback | Any other fitted sklearn classifier/regressor exposing `predict` | Predictions + schema only, no family-specific 3D |

Tasks covered: binary classification, multiclass classification, regression, ranking (`LGBMRanker`), and outlier detection (`OneClassSVM`).

## Features

**Dataset ingestion** (`/datasets`)
- Upload CSV / TSV / XLSX / Parquet (auto charset detection, multi-sheet picker with `SheetSelectorBar`)
- Column profiling (`ColumnProfileTable`), sample preview, Parquet conversion cache under `.cache/datasets`
- Filter / sort / page queries and queue-preset selection resolution (`/datasets/{id}/query`, `/resolve-selection`)
- Bundled samples via `POST /datasets/load-sample/{sample_name}`; memory-savings banner in the UI

**Model registry & introspection** (`/models`)
- Upload `.joblib`/`.pkl` (+ native `.ubj`/`.txt`/`.cbm`/`.onnx`) with universal schema detection (embedded names or `feature_{i}` placeholders, class labels, fitted-state check)
- Per-algorithm inspector + 2D surface sandbox components (one pair per family, e.g. `ForestInspectorPanel` + `ForestSurfaceSandbox`)
- Per-row linear equation trace (`/{id}/trace-preview`) and 3D hyperplane mesh (`/{id}/surface-preview`)
- Bundled sample models via `POST /models/load-sample/{sample_key}`; hyperparameter accordion + architecture/feature-schema cards

**Column mapping** (`/mappings`)
- 3-tier fuzzy auto-suggest, positional map-by-order, 100-row float32 dry-run validation, validate-and-save with committed-mapping fetch
- Feature/target row cards, target-column selector, dry-run validation drawer

**Evaluation lens** (`/evaluations`)
- Global metrics compute, metrics ribbon, interactive confusion matrix, target-label mapper modal, residual slicer, outlier slicer card

**3D stage, projection & playback** (`/stage3d`, `/projection`, `/playback`)
- Scene initialization (axis normalization + per-family surface + trajectory points), LOD decision-surface mesh + PCA manifold
- One bespoke WebGL surface per family (sigmoid plane, voxel terrain, voting arena, leaf-flow, radar sphere, synapse brain, latent manifold, …)
- Playback manifest build/fetch for the 60 FPS VCR; virtualized trajectory queue (`@tanstack/react-virtual`)

**Explainability** (`/explainability`)
- Session build + per-frame explanations, feature waterfall panel, frame feature inspector, active-ball HUD overlay
- Exact TreeSHAP for XGBoost / LightGBM / CatBoost; linear-equation traces for linear families

**Timeline** (`/timeline`)
- Multi-track analysis build/fetch plus frame bookmark create/delete

**What-if sandbox** (`/whatif`)
- Session build with feature ranges, per-frame mutation evaluation, minimum-change counterfactual finder (with feature locks)

**Session recording (`.mlv`)** (`/recordings`)
- Bake session to compact binary (msgpack + zstd), instant load, bookmark update + re-bake, size estimate, synchronized dual-`.mlv` comparison, video recording indicator bar

**Aliases** (`/alias`)
- Upsert / fetch class & feature alias config, auto-suggest preview, JSON export/import

**Export studio** (`/exports`)
- Enriched predictions table export + first-rows preview, report/video export modal

**Compute engine** (`/compute`)
- Hardware telemetry, execution-device + chunk-size preference, chunked job register / cancel / pre-flight estimate, inactive-cache purge + GC
- `HardwareStatusPill`, `ChunkedJobProgressBanner`, `ComputeDiagnosticsModal` in the UI

**System & studio UX** (`/system`)
- Health probe (hardware readiness + RAM snapshots), Python GC trigger, `.cache/` purge
- OMP/MKL thread pinning before numpy import, startup stale-cache sweep, memory guard/monitor with low-memory banner
- Global error boundary, offline banner with retry, keyboard shortcuts (`Alt+1..5`, `Ctrl+Shift+D`, `?`), light/dark theme, diagnostics modal with Force GC

## Configuration

Copy and edit the templates — no secrets are committed:

`backend/.env.example` → `backend/.env`:
```ini
APP_NAME=Easy ML Viewer Backend
APP_VERSION=1.0.0
HOST=127.0.0.1
PORT=30211
FRONTEND_ORIGINS=http://localhost:30222,http://127.0.0.1:30222
MAX_PROCESS_RAM_MB=4096
MIN_SYSTEM_FREE_RAM_MB=512
MAX_UPLOAD_SIZE_MB=1024
OMP_NUM_THREADS=4
```

`frontend/.env.example` → `frontend/.env`:
```ini
VITE_API_BASE_URL=http://127.0.0.1:30211
```

Vite proxies `/api` to the backend in dev, so the frontend works with Wi-Fi off. Production build (`npm run build`) emits static `dist/` with `sourcemap: false`.

## Samples

`samples/` ships ~100 ready-to-load dataset+model pairs covering every supported family, e.g. `sample_iris_knn.*`, `sample_binary_xgboost.*`, `sample_reg_lightgbm.*`, `sample_housing_elasticnet.*`, `sample_deep_mlp_onnx.*`, `sample_pipe_mixed.*`. Load them via `POST /datasets/load-sample/{sample_name}`, `POST /models/load-sample/{sample_key}`, or directly from the Step 1 / Step 2 views.

## Verify

```bash
# Backend — full suite (85 test files, phase-grouped under backend/tests/)
cd backend && python -m pytest -q

# Health probe
curl http://127.0.0.1:30211/api/v1/system/health

# Frontend — typecheck + production build
cd frontend && npm run typecheck && npm run build
```

Offline check: turn Wi-Fi off, open `http://127.0.0.1:30222` — the page must load with zero external requests (confirm in DevTools Network tab).
