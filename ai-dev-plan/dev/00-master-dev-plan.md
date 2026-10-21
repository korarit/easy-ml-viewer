# Master Development Plan — Easy ML Viewer (40 Phases Production Blueprint)

เอกสารฉบับนี้คือ **พิมพ์เขียวแม่บท (Master Production Blueprint)** ของโปรเจกต์ **Easy ML Viewer** ซึ่งควบคุมลำดับการพัฒนาทั้งหมด **40 Phases** โดยจัดเรียงตาม **ลำดับการพึ่งพาทางวิศวกรรมจริง (Strict Dependency-First Order)** เพื่อให้ทีมพัฒนาหรือ AI Agent สามารถลงมือสร้างระบบทีละเฟสได้อย่างมั่นคง ทดสอบได้จริงในทุกก้าว และไม่เกิดปัญหาคอขวดทางหน่วยความจำหรือการรื้อโครงสร้างย้อนหลัง

---

## 1. ภาพรวมกลยุทธ์การเชื่อมโยงทั้ง 40 Phases (Dependency-First Integration Strategy)

การพัฒนาแบ่งออกเป็น **7 ช่วงหลัก (7 Strategic Parts)** ที่ต่อยอดกันเป็นชั้นๆ ดังนี้:

```text
+---------------------------------------------------------------------------------------------------+
| PART 1 (Phases 01 - 17): CORE PLATFORM, FIRST MODEL SLICE (LinearRegression) & 3D VCR STUDIO      |
|  [01 Scaffolding/DDD] -> [02 Dataset/float32] -> [03 ModelRegistry] -> [04 LinearRegression Slice]|
|  -> [05 Mapping] -> [06 Trajectory Queue] -> [07 Global Eval] -> [08-09 3D Stage & Surface Mesh]  |
|  -> [10-11 60FPS VCR & Smart Timeline] -> [12 HUD/Waterfall] -> [13 What-If/Counterfactuals]      |
|  -> [14 Command Palette/UX] -> [15 .mlv Format] -> [16 Video/CSV Export] -> [17 CPU/GPU Chunking] |
+---------------------------------------------------------------------------------------------------+
                                                 |
         (Every subsequent Model Adapter automatically inherits 3D VCR, What-If, .mlv & Export!)
                                                 v
+---------------------------------------------------------------------------------------------------+
| PARTS 2 - 6 (Phases 18 - 39): 20+ MODEL FAMILIES PLUGGED INTO BaseModelAdapter & ModelRegistry    |
|  - Part 2 (Phases 18-21): Generalized Linear (Logistic S-Curve, Ridge/Lasso L1/L2, Elastic, SGD)  |
|  - Part 3 (Phases 22-26): Distance, Kernel & Bayes (KNN Radar, SVM Saddle, GNB Hills, LDA/QDA)    |
|  - Part 4 (Phases 27-30): Trees & Bagging (Decision Tree Voxels, Random Forest, ExtraTrees, Ada)  |
|  - Part 5 (Phases 31-35): Gradient Boosting (Sklearn GBM, HistGBM, XGBoost, LightGBM, CatBoost)   |
|  - Part 6 (Phases 36-39): Neural & Universal (MLP Synapse Brain, Pipeline, ONNX ML, ONNX Deep)    |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| PART 7 (Phase 40): CROSS-PLATFORM LAUNCHERS, 20-MODEL DEMO SUITE, DUAL .MLV COMPARE & HARDENING   |
|  [Windows/Mac/Linux Launchers] + [20-Model Instant Gallery] + [Dual 3D Sync Compare] + [100k Test]|
+---------------------------------------------------------------------------------------------------+
```

### โครงสร้าง Monorepo, พอร์ตมาตรฐาน และการรันแบบ 100% Offline
- **`backend/` (Port `30211`)**: Python 3.11+ / FastAPI / DDD Architecture ที่มี `app/core/` เป็น Shared Kernel, ประมวลผลด้วย NumPy (`float32`), PyArrow Parquet, MsgPack, และ Zstandard
- **`frontend/` (Port `30222`)**: React 19 / Vite / TypeScript / Tailwind CSS v4 / Zustand / Three.js (`@react-three/fiber` & `@react-three/drei`)
- **100% Air-Gapped Offline**: บันเดิลฟอนต์ (`Inter`, `JetBrains Mono`) และไอคอน SVG ไว้ในโปรเจกต์ทั้งหมด ห้ามเรียก External CDN ใดๆ นอก `127.0.0.1`
- **UI & Visual Identity**: ควบคุมการออกแบบด้วยเอกสาร [design-system-ci.md](file:///e:/easy-ml-viewer/ai-dev-plan/design-system-ci.md) ในสไตล์ *"Precision Engineering Lab"* (Dark & Light Mode) สุขุม เรียบหรู ไม่ใช้ลูกเล่นไซไฟหลอกตา (Anti-AI-Cringe)

---

## 2. สถาปัตยกรรม DDD 5 Bounded Contexts และสายพานข้อมูล (DDD Architecture & Binary Pipeline)

ระบบแบ่งออกเป็น **1 Shared Kernel (`core`)** และ **5 Bounded Contexts** ที่สื่อสารกันผ่าน Application Orchestrator:

| Bounded Context | หน้าที่รับผิดชอบหลัก (Core Responsibilities) | เชื่อมโยงกับ Phases |
| :--- | :--- | :--- |
| **`core/` (Shared Kernel)** | ตั้งค่า Cross-platform `pathlib.Path`, ระบบ Exception มาตรฐาน, Memory Guard (`psutil`), Binary MsgPack+Zstd Codec, และ Process/Thread Pool | Phase 01, 15, 17, 40 |
| **`dataset/`** | โหลดไฟล์ `.csv`, `.xlsx`, `.parquet`, วิเคราะห์สถิติคอลัมน์ (Profiler), แคสต์ตัวเลขเป็น `float32`, และจัดเก็บลง `.cache/*.parquet` | Phase 02, 06, 17 |
| **`model_engine/`** | นิยาม `BaseModelAdapter`, `ModelRegistry`, ระบบสกัด Schema อัตโนมัติ, และรวม Adapter เฉพาะทางของโมเดลทั้ง 20+ ตระกูล | Phase 03, 04, 18–39 |
| **`mapping/`** | จับคู่คอลัมน์ CSV เข้ากับ Feature ของโมเดลด้วยอัลกอริทึม Fuzzy Matching, ตรวจสอบความเข้ากันได้ของชนิดข้อมูล และจัดการ Categorical Encoding | Phase 05, 35, 37 |
| **`simulation/`** | คำนวณพื้นผิว 3D Mesh ($50 \times 50$), คำนวณ Trajectory Queue, สกัดค่าอิทธิพลรายฟีเจอร์ (SHAP/Contributions),Global Evaluation, และ What-If Sandbox | Phase 06–14, 17 |
| **`recording/`** | บีบอัดผลลัพธ์การจำลองทั้งหมดเป็นไฟล์ `.mlv` (MsgPack + Zstd), โหลดไฟล์ `.mlv` แบบ Instant Playback, ส่งออกคลิป `.webm`/`.mp4`, และเปรียบเทียบ 2 โมเดล | Phase 15, 16, 40 |

---

## 3. ตารางสรุป UI/UX Flow ภาพรวมตั้งแต่ต้นจนจบ (End-to-End Studio UX Flow Matrix)

| ขั้นตอน (Stage) | หน้าจอ / โมดูล UI (Screen / Component) | การกระทำของผู้ใช้ (User Action) | การตอบสนองของระบบและ 3D Stage (System & 3D Feedback) |
| :--- | :--- | :--- | :--- |
| **Step 1: Launch & Entry** | **Welcome Hub & Instant Demo Gallery** | อัปโหลดไฟล์โมเดล + ข้อมูล หรือคลิกปุ่ม `"Try 20-Model Demo Gallery"` / `"Open .mlv"` | ตรวจสอบชนิดโมเดลและไฟล์ข้อมูลทันที หรือเปิดไฟล์ `.mlv` เข้าสู่ฉาก 3D ภายใน `< 0.5s` |
| **Step 2: Schema & Mapping** | **Smart Column Mapper Drawer** | ตรวจสอบเส้นเชื่อมจับคู่ `CSV Column <-> Model Feature` (แก้ไขหากต้องการ) | ระบบทำ Auto-Fuzzy Match ให้อัตโนมัติ แจ้งเตือนคอลัมน์ที่มีค่าว่าง (`NaN`) หรือชนิดข้อมูลไม่ตรง |
| **Step 3: Global Triage** | **Global Evaluation Lens & Error Slicer** | คลิกช่อง `False Positives` ใน Confusion Matrix หรือลากคลุมจุดที่ทายผิดสูงในกราฟ Residual | ระบบคัดกรองเฉพาะแถวที่มีปัญหาเข้าสู่คิวการเล่น (Trajectory Queue) ทันทีโดยไม่ต้องไล่หาเอง |
| **Step 4: 3D Stage & Axes** | **3D WebGL Stage & Axis Selector** | เลือกแกน $X, Y$ (หรือกดปุ่ม `Auto Top-2 Important Features` / `3D PCA View`) | สร้างพื้นผิวการตัดสินใจ 3D ประจำตระกูลโมเดล (ระนาบแบน, คลื่น S-Curve, ภูเขา Bell Hills, หรือขั้นบันได Voxel) |
| **Step 5: VCR Playback** | **60 FPS VCR Transport & Multi-Track Timeline** | กด `Play / Pause`, ลากแถบ Scrubber, หรือคลิกหมุด `Anomaly Pin` บน Timeline | ลูกบอลแสงเคลื่อนที่ผ่านพื้นผิว 3D ที่ 60 FPS พร้อมหางแสง Comet Trail และเส้นประทิ้งดิ่งตั้งฉาก |
| **Step 6: Explainability** | **Floating 3D HUD & Waterfall Panel** | ดูป้ายข้อมูลที่ลอยตามลูกบอล หรือดูกราฟ Waterfall ด้านขวา | แสดงสมการภายในของโมเดล (เช่น เส้นทางใบไม้, แรงดึงเพื่อนบ้าน KNN, หรือค่า Exact TreeSHAP) |
| **Step 7: What-If Lab** | **What-If Sandbox & Counterfactual Ghost** | ลากสไลเดอร์ปรับค่าตัวแปร หรือกดปุ่ม `"Find Minimum Change to Flip"` | ลูกบอลเงา **3D Ghost Sphere** วิ่งแยกออกจากลูกบอลจริงข้ามเส้นแบ่งเขตเพื่อแสดงว่าต้องเปลี่ยนค่าเท่าใดผลลัพธ์จึงพลิก |
| **Step 8: Save & Compare** | **`.mlv` Baker, Video Exporter & Dual Compare** | กดบันทึกเป็นไฟล์ `.mlv`, อัดวิดีโอ `.mp4`, หรือเปิดเทียบ 2 โมเดลซ้าย-ขวา | บันทึกไฟล์ `.mlv` ขนาดกะทัดรัด หรือซิงค์ฉาก 3D ซ้าย-ขวาเพื่อชี้จุดที่สองโมเดลทายขัดแย้งกัน |

---

## 4. ตารางควบคุมงบประมาณหน่วยความจำและทรัพยากรเครื่องมาตรฐาน (Global Memory Budget & Resource Optimization Contract)

ทุกเฟสตั้งแต่ Phase 01 ถึง Phase 40 **ต้องปฏิบัติตามข้อกำหนดด้านหน่วยความจำและประสิทธิภาพตารางนี้อย่างเคร่งครัด**:

| องค์ประกอบ (Subsystem / Resource) | เพดานควบคุมสูงสุด (Strict Limit / Budget) | กลไกบังคับใช้ทางวิศวกรรม (Mandatory Engineering Enforcement) |
| :--- | :--- | :--- |
| **1. Numeric Data Precision** | ห้ามใช้ `float64` ใน Pipeline หลัก (`float32` เท่านั้น) | แคสต์ทุกคอลัมน์ตัวเลขเป็น `np.float32` และดัชนีคลาส/ใบไม้เป็น `np.uint16` / `np.int32` ตั้งแต่วินาทีที่อ่านไฟล์ (ลด RAM 50%) |
| **2. Dataset Storage Cache** | Peak RAM ตอนโหลดไฟล์ CSV/Excel $\le 250\text{ MB}$ | แปลงไฟล์ `.csv`/`.xlsx` เก็บลงดิสก์ใน `.cache/*.parquet` แล้วอ่านเฉพาะคอลัมน์และช่วงแถวที่ใช้งานผ่าน PyArrow Memory-Mapping |
| **3. Inference Chunk Size** | สูงสุด `2,500` แถวต่อ 1 Chunk | ห้ามส่งข้อมูลทั้งไฟล์ (เช่น 100,000 แถว) เข้า `predict()` หรือ `SHAP` รวดเดียว ต้องตัดแบ่งทีละ `2,500` แถวใน `ComputePool` เสมอ |
| **4. 3D Surface Grid Resolution** | มาตรฐาน $50 \times 50$ (`2,500` จุด) / สูงสุด $80 \times 80$ | คำนวณพื้นผิว 3D ด้วยกริด $50 \times 50$ (`float32` ขนาดเพียง `10 KB`) ทำให้สร้าง Mesh เสร็จภายใน `< 150 ms` |
| **5. Frontend VCR State Store** | ห้ามเก็บเฟรมใน React State Array ของ JSON Objects | จัดเก็บข้อมูลทุกเฟรมใน **Flat TypedArrays** (`Float32Array`, `Uint16Array`) และอ่านค่าด้วย Pointer Index ใน `useFrame()` |
| **6. WebGL Draw Calls** | $\le 15$ Draw Calls ต่อฉาก 3D (60 FPS คงที่) | ใช้ `THREE.InstancedMesh` สำหรับวาดจุดข้อมูล/โหนดประสาทนับพันจุดใน 1 Draw Call และเรียก `.dispose()` ทุกครั้งที่สลับโมเดล |
| **7. CPU / GPU Worker Threads** | `max_workers = max(1, physical_cores - 1)` | เว้นอย่างน้อย 1 CPU Core ไว้ให้ FastAPI Event Loop เสมอ เพื่อให้ API ตอบสนองไวและยกเลิกงาน (`Cancel Job`) ได้ทันที |
| **8. Process & Port Lifecycle** | `0` Zombie Processes / Temp Cache $\le 1.0\text{ GB}$ | สคริปต์ Launcher ต้องดักจับ `SIGINT`/`SIGTERM`/`EXIT` เพื่อปิดทั้งพอร์ต `30211` และ `30222` พร้อมกัน และมีระบบกวาดล้าง `.cache/` |

---

## 5. สารบัญแผนการพัฒนาทั้ง 40 Phases (Complete 40-Phase Blueprint Index)

### 🧱 Part 1: Base Platform, DDD Core, First Simple Model, 3D VCR Studio, What-If, `.mlv` & Compute Engine (Phases 01 - 17)
1. [phase-01-scaffolding-and-ddd-core.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-01-scaffolding-and-ddd-core.md) — โครงสร้าง Monorepo 2 โฟลเดอร์ 2 พอร์ต (`30211` & `30222`), DDD Shared `core/` Kernel & Offline Setup
2. [phase-02-dataset-ingestion-and-memory-guard.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-02-dataset-ingestion-and-memory-guard.md) — ระบบโหลด CSV/XLSX/Parquet, Data Profiler, `float32` Downcaster & `.cache/` Parquet Memory Guard
3. [phase-03-model-registry-and-schema-introspection.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-03-model-registry-and-schema-introspection.md) — `BaseModelAdapter`, `ModelRegistry` & ระบบสกัด Feature Schema อัตโนมัติ 100%
4. [phase-04-first-model-linear-regression-slice.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-04-first-model-linear-regression-slice.md) — โมเดลพื้นฐานตัวแรก (`LinearRegression`) สำหรับทดสอบท่อการทำงาน End-to-End ครบวงจร
5. [phase-05-column-mapping-and-type-casting.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-05-column-mapping-and-type-casting.md) — ระบบจับคู่ Column $\leftrightarrow$ Feature อัตโนมัติ (Fuzzy Match), Type Casting & Missing Value Guard
6. [phase-06-data-selector-and-trajectory-queue.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-06-data-selector-and-trajectory-queue.md) — ตาราง Virtualized เลือกข้อมูลหลายแถว, ระบบกรองเงื่อนไข & การจัดลำดับคิวการเล่น (Trajectory Queue)
7. [phase-07-global-evaluation-and-error-slicer.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-07-global-evaluation-and-error-slicer.md) — Global Evaluation Lens (Interactive Confusion Matrix & Residual Error Slicer คลิกเลือกแถวที่ทายผิดเข้าคิวทันที)
8. [phase-08-3d-stage-engine-and-camera-system.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-08-3d-stage-engine-and-camera-system.md) — 3D WebGL Stage, ระบบสลับมุมกล้อง 1-Click Presets, เส้นประทิ้งดิ่งตั้งฉาก & เงาตกกระทบพื้น
9. [phase-09-3d-projection-and-surface-mesh-generator.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-09-3d-projection-and-surface-mesh-generator.md) — เครื่องยนต์คำนวณพื้นผิว 3D Mesh ($50 \times 50$ Axis Explorer Slice & 3D PCA Global Projection)
10. [phase-10-vcr-playback-studio-and-scrubber.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-10-vcr-playback-studio-and-scrubber.md) — ระบบควบคุมการเล่นวิดีโอ (Play, Pause, Step, Loop, Speed, 60FPS Interpolation Scrubber & หางแสง Comet Trail)
11. [phase-11-smart-multitrack-timeline-and-bookmarks.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-11-smart-multitrack-timeline-and-bookmarks.md) — Smart Multi-Track Timeline (Confidence Strip, หมุดชี้จุดผิดปกติอัตโนมัติ Anomaly Pins & Bookmarks)
12. [phase-12-explainability-hud-and-waterfall.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-12-explainability-hud-and-waterfall.md) — ป้ายข้อมูลลอยตามลูกบอลใน 3D (Floating HUD) & กราฟ Waterfall อธิบายอิทธิพลรายฟีเจอร์
13. [phase-13-what-if-sandbox-and-counterfactuals.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-13-what-if-sandbox-and-counterfactuals.md) — Interactive "What-If" Sandbox, Counterfactual Finder & ลูกบอลเงา 3D Ghost Sphere
14. [phase-14-modern-studio-ux-and-command-palette.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-14-modern-studio-ux-and-command-palette.md) — 3D Click-to-Scrub, Command Palette (`Ctrl+K`), Cinema Mode, Minimap & Feature Label Aliasing
15. [phase-15-mlv-file-format-baker-and-loader.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-15-mlv-file-format-baker-and-loader.md) — ฟอร์แมตไฟล์ `.mlv` (MsgPack + Zstandard Binary Container) บันทึกและเปิดดูทันทีไม่ต้องคำนวณซ้ำ
16. [phase-16-rich-export-video-and-enriched-csv.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-16-rich-export-video-and-enriched-csv.md) — ระบบส่งออก Enriched CSV/Excel, อัดคลิปวิดีโอการเคลื่อนที่ 3D (`.webm`/`.mp4`) & ถ่ายภาพ 4K Snapshot
17. [phase-17-compute-engine-cpu-gpu-and-chunking.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-17-compute-engine-cpu-gpu-and-chunking.md) — ระบบจัดการ Multi-core CPU, ตรวจจับ GPU & ประมวลผลข้อมูลขนาดใหญ่ทีละ Chunk พร้อม Progress Bar

### 📐 Part 2: Linear & Generalized Linear Models Integration (Phases 18 - 21)
18. [phase-18-model-logistic-regression.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-18-model-logistic-regression.md) — Logistic Regression (Binary & Multinomial Softmax) & พื้นผิวคลื่น 3D Sigmoid S-Surface
19. [phase-19-model-ridge-and-lasso.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-19-model-ridge-and-lasso.md) — Ridge ($L_2$) & Lasso ($L_1$) พร้อมการไฮไลต์แกนฟีเจอร์ที่ถูกตัดน้ำหนักเป็นศูนย์ (Zeroed-Out Axes)
20. [phase-20-model-elasticnet.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-20-model-elasticnet.md) — ElasticNet & การวิเคราะห์สัดส่วนผสม $L_1/L_2$ Regularization Geometry
21. [phase-21-model-sgd-classifier-regressor.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-21-model-sgd-classifier-regressor.md) — SGD Classifier / Regressor (Hinge, Log-Loss, Modified Huber) & ระนาบขอบเขต Margin

### 🎯 Part 3: Distance, Kernel & Probabilistic Models Integration (Phases 22 - 26)
22. [phase-22-model-knn-radar-sphere.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-22-model-knn-radar-sphere.md) — K-Nearest Neighbors (`KNeighborsClassifier`/`Regressor`) & ลูกบอลเรดาร์ 3 มิติ + เส้นใยเชื่อมเพื่อนบ้าน $K$ จุด
23. [phase-23-model-svm-kernel-surface.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-23-model-svm-kernel-surface.md) — Support Vector Machines (`SVC`, `SVR`, `LinearSVC`) & พื้นผิวโค้ง RBF Saddle + เสาแสง Support Vectors
24. [phase-24-model-gaussian-naive-bayes.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-24-model-gaussian-naive-bayes.md) — Gaussian Naive Bayes (`GaussianNB`) & ภูเขาระฆังคว่ำความน่าจะเป็น 3 มิติ (3D Gaussian Bell Hills)
25. [phase-25-model-multinomial-bernoulli-nb.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-25-model-multinomial-bernoulli-nb.md) — Multinomial & Bernoulli Naive Bayes & แท่งสะสมความน่าจะเป็น Log-Likelihood Towers
26. [phase-26-model-lda-and-qda.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-26-model-lda-and-qda.md) — Linear & Quadratic Discriminant Analysis (`LDA`/`QDA`) & ทรงรีความแปรปรวนร่วม 3 มิติ (3D Covariance Ellipsoids)

### 🌲 Part 4: Decision Trees & Bagging Ensembles Integration (Phases 27 - 30)
27. [phase-27-model-decision-tree-voxel.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-27-model-decision-tree-voxel.md) — Decision Tree (`DecisionTreeClassifier`/`Regressor`), `decision_path` & ภูมิประเทศขั้นบันได 3D Voxel Terraces
28. [phase-28-model-random-forest-consensus.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-28-model-random-forest-consensus.md) — Random Forest (`RandomForestClassifier`/`Regressor`) & ลานโหวตของต้นไม้ทั้งป่าใน 3D (Ensemble Voting Arena)
29. [phase-29-model-extra-trees.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-29-model-extra-trees.md) — Extra Trees (`ExtraTreesClassifier`/`Regressor`) & วงรัศมีแสดงความแปรปรวนของการสุ่มจุดตัด
30. [phase-30-model-adaboost-cascade.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-30-model-adaboost-cascade.md) — AdaBoost (`AdaBoostClassifier`/`Regressor`) & ลำดับการแก้ไขคำตอบทีละขั้น (`staged_predict` Cascade)

### 🚀 Part 5: Gradient Boosting Heavyweights Integration (Phases 31 - 35)
31. [phase-31-model-sklearn-gbm.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-31-model-sklearn-gbm.md) — Scikit-Learn Gradient Boosting (`GradientBoostingClassifier`/`Regressor`) & เส้นทางสะสมค่า Residual
32. [phase-32-model-hist-gradient-boosting.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-32-model-hist-gradient-boosting.md) — HistGradientBoosting (`HistGradientBoostingClassifier`/`Regressor`) & ขอบเขตถังข้อมูล 3D Histogram Bins
33. [phase-33-model-xgboost.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-33-model-xgboost.md) — XGBoost (`.pkl`, `.json`, `.ubj`), C++ Exact TreeSHAP (`pred_contribs`) & 3D Boosted Leaf Flow
34. [phase-34-model-lightgbm.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-34-model-lightgbm.md) — LightGBM (`.pkl`, `.txt`, `.model`), Native TreeSHAP & โครงสร้างการแตกกิ่งแบบไม่สมมาตร Leaf-wise 3D
35. [phase-35-model-catboost.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-35-model-catboost.md) — CatBoost (`.cbm`, `.pkl`), Native Categorical `Pool` & ตารางสมมาตร 3D Symmetric Oblivious Tree Grid

### 🧠 Part 6: Neural Networks, Pipelines & Universal ONNX Integration (Phases 36 - 39)
36. [phase-36-model-mlp-neural-network.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-36-model-mlp-neural-network.md) — Multi-Layer Perceptron (`MLPClassifier`/`Regressor`), Top-32 Synapse Brain 3D & Dead ReLU Detector
37. [phase-37-sklearn-pipeline-unwrapper.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-37-sklearn-pipeline-unwrapper.md) — Scikit-Learn `Pipeline` & `ColumnTransformer` Unwrapper, Step Tracer & Raw vs Scaled 3D Space Toggle
38. [phase-38-onnx-classical-ml-engine.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-38-onnx-classical-ml-engine.md) — ONNX Runtime Classical ML Inspector (`ai.onnx.ml`: `TreeEnsemble`, `Linear`, `SVM`, `ZipMap` Unpacker)
39. [phase-39-onnx-deep-learning-intermediate.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-39-onnx-deep-learning-intermediate.md) — ONNX Deep Learning Intermediate Layer Extractor (In-Memory Graph Injection) & 3D Latent Space PCA Manifold

### 🏁 Part 7: Final Packaging, 20-Model Demo Suite, Multi-Model Compare & Hardening (Phase 40)
40. [phase-40-offline-packaging-and-stress-test.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-40-offline-packaging-and-stress-test.md) — สคริปต์รัน Offline บน Windows/Mac/Linux, 20-Model Instant Demo Suite, โหมดเปรียบเทียบ 2 โมเดล (Dual `.mlv` Compare) & 100,000-Row Stress Test


---

## 6. ตารางสรุปเรขาคณิต 3 มิติ และกลไกการแกะโครงสร้างภายในของทั้ง 20+ ตระกูลโมเดล (20+ Model Families 3D & Explainability Matrix)

| ลำดับ | ตระกูลโมเดล (Model Family) | ไฟล์ที่รองรับ | รูปทรงเรขาคณิตบนฉาก 3D (3D Stage Geometry) | กลไกสกัดคำอธิบายภายใน (Internal Flow & Attribution) |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Linear Regression** (Phase 04) | `.pkl`, `.joblib` | ระนาบเอียง 3 มิติ (Tilted Flat Hyperplane) + เส้น Residual ทิ้งดิ่ง | สมการแม่นยำ $\phi_j = \beta_j (x_j - \bar{x}_j)$ |
| **02** | **Logistic Regression** (Phase 18) | `.pkl`, `.joblib` | พื้นผิวคลื่นตัว S (3D Sigmoid S-Surface) / Softmax Terraces | Log-Odds Contribution $\beta_{c,j} x_j$ |
| **03** | **Ridge ($L_2$) & Lasso ($L_1$)** (Phase 19) | `.pkl`, `.joblib` | ระนาบเอียงพร้อมแกนที่ถูกตัดเป็นศูนย์ (Zeroed-Out Feature Axes) | สัมประสิทธิ์ $\beta_j x_j$ + ป้ายเตือน Feature Sparsity |
| **04** | **ElasticNet** (Phase 20) | `.pkl`, `.joblib` | ระนาบผสม $L_1 + L_2$ พร้อมแถบวัด `l1_ratio` | สัดส่วนการหดตัว $L_1$ vs $L_2$ Penalty |
| **05** | **SGD Classifier / Regressor** (Phase 21) | `.pkl`, `.joblib` | ระนาบตัดสินใจ + ขอบเขต Margin $\pm 1/\|\mathbf{w}\|$ | ระยะห่างตั้งฉากจาก Hyperplane (`decision_function`) |
| **06** | **K-Nearest Neighbors (KNN)** (Phase 22) | `.pkl`, `.joblib` | ลูกบอลเรดาร์ 3 มิติ + เส้นใยเชื่อมเพื่อนบ้าน $K$ จุด (`kneighbors`) | น้ำหนักโหวตตามระยะห่าง $w_k = 1 / (d_k + \epsilon)$ |
| **07** | **SVM (`SVC`, `SVR`, `LinearSVC`)** (Phase 23) | `.pkl`, `.joblib` | พื้นผิวอานม้าโค้งมน (RBF Kernel Saddle) + เสาแสง Support Vectors | แรงดึงดูดจาก Support Vectors $\alpha_m y_m K(\mathbf{x}_m, \mathbf{x})$ |
| **08** | **Gaussian Naive Bayes** (Phase 24) | `.pkl`, `.joblib` | ภูเขาระฆังคว่ำความน่าจะเป็น 3 มิติ (3D Gaussian Bell Hills) | ผลรวม Log-Likelihood $-\frac{(x_j - \mu_{c,j})^2}{2\sigma^2_{c,j}}$ |
| **09** | **Multinomial & Bernoulli NB** (Phase 25) | `.pkl`, `.joblib` | หอคอยแท่งความน่าจะเป็นสะสม (3D Log-Likelihood Towers) | คะแนน Evidence รายคำ/รายบิต $x_j \log P(f_j|c)$ |
| **10** | **LDA & QDA** (Phase 26) | `.pkl`, `.joblib` | ทรงรีความแปรปรวนร่วม 3 มิติ (3D Covariance Ellipsoids) | ระยะห่าง Mahalanobis $(\mathbf{x}-\boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}_c^{-1} (\mathbf{x}-\boldsymbol{\mu}_c)$ |
| **11** | **Decision Tree** (Phase 27) | `.pkl`, `.joblib` | ภูมิประเทศขั้นบันได 3 มิติ (3D Voxel Stepped Terraces) | เส้นทางเดินผ่านโหนดจริง (`decision_path` & `apply`) |
| **12** | **Random Forest** (Phase 28) | `.pkl`, `.joblib` | ขั้นบันไดความละเอียดสูง + ลานโหวตต้นไม้ทั้งป่า (Consensus Arena) | คะแนนโหวตของ `estimators_` ทุกต้น + Tree Contributions |
| **13** | **Extra Trees** (Phase 29) | `.pkl`, `.joblib` | ขั้นบันไดเฉลี่ยที่เรียบเนียนขึ้น + วงรัศมีความแปรปรวนของการสุ่มตัด | ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของคำตอบรายต้น |
| **14** | **AdaBoost** (Phase 30) | `.pkl`, `.joblib` | ลำดับการแก้ไขคำตอบทีละขั้น (Staged Boosting Cascade) | การสะสมคะแนนผ่าน `staged_predict_proba(X)` |
| **15** | **Scikit-Learn GBM** (Phase 31) | `.pkl`, `.joblib` | พื้นผิวการสะสมค่า Residual (Additive Residual Terrain) | การไต่ระดับ Log-Odds / Residual ทีละ Stage |
| **16** | **HistGradientBoosting** (Phase 32) | `.pkl`, `.joblib` | ขอบเขตถังข้อมูล 255 ถัง (3D Quantized Histogram Bin Grid) | ดัชนีถังข้อมูล `_bin_mapper` + Fast Tree Attribution |
| **17** | **XGBoost** (Phase 33) | `.pkl`, `.json`, `.ubj` | ภูมิประเทศขั้นบันได XGBoost + ลูกศรแรงผลัก TreeSHAP | C++ Native `pred_contribs=True` & `pred_leaf=True` |
| **18** | **LightGBM** (Phase 34) | `.pkl`, `.txt`, `.model` | ภูมิประเทศแตกกิ่งไม่สมมาตร (Asymmetric Leaf-wise Terrain) | C++ Native `pred_contrib=True` & `pred_leaf=True` |
| **19** | **CatBoost** (Phase 35) | `.cbm`, `.pkl` | ตารางหมากรุกสมมาตร 3 มิติ (Symmetric Oblivious Grid) | Bitwise Leaf Index + C++ `ShapValues` + Native Cat Axis |
| **20** | **MLP Neural Network** (Phase 36) | `.pkl`, `.joblib` | โครงข่ายประสาทลอยฟ้า 3 มิติ (3D Floating Synapse Brain) + Surface | Layer-by-Layer $A^{(l)}$, Dead ReLU & Input $\times$ Gradient |
| **21** | **Sklearn `Pipeline` / `CT`** (Phase 37) | `.pkl`, `.joblib` | อุโมงค์แปลงข้อมูล + ฉาก 3D ของ Estimator ตัวสุดท้าย (Raw/Scaled) | Step-by-Step Value Tracer & Lineage Attribution Sum |
| **22** | **ONNX Classical ML** (Phase 38) | `.onnx` | ปรับเปลี่ยนรูปทรง 3D อัตโนมัติตามโหนด `ai.onnx.ml` | Decompiled `TreeEnsemble` / `Linear` / `SVM` Attributes |
| **23** | **ONNX Deep Learning** (Phase 39) | `.onnx` | พื้นผิว 3D คู่กับกลุ่มหมอกดาวในอวกาศแฝง (3D Latent PCA Manifold) | In-Memory Graph Output Injection + Latent SVD |

---

## 7. เกณฑ์การตรวจรับคุณภาพระดับระบบ (Master Quality Gate Checklist)
- [ ] **Architecture Purity**: โค้ดใน `backend/app/domains/` แยก Bounded Context ชัดเจน ไม่มีการ Import ข้ามโดเมนโดยไม่ผ่าน `app/core/` หรือ Application Orchestrator
- [ ] **Memory Safety**: ทดสอบกับข้อมูลขนาด `100,000` แถว โดย Peak RAM ของ Backend ไม่เกิน `600 MB` และไม่มี Memory Leak หลังรันต่อเนื่อง 50 รอบ
- [ ] **60 FPS Real-Time Scrubbing**: การลากแถบเวลา VCR และการหมุนกล้อง 3D ทั้งในโหมดปกติและโหมดเปรียบเทียบ 2 โมเดล (Dual `.mlv` Compare) รักษาความลื่นไหลที่ 60 FPS
- [ ] **100% Air-Gapped Offline**: เปิดใช้งานผ่าน `start-windows.bat`, `start-windows.ps1`, และ `start-unix.sh` ได้โดยไม่ต้องต่ออินเทอร์เน็ต และปิดโปรเซสลูกสะอาดหมดจดเมื่อเลิกใช้งาน
