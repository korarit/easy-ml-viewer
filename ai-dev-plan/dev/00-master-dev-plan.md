# Master Development Plan — Easy ML Viewer (56 Phases Production Blueprint)

เอกสารฉบับนี้คือ **พิมพ์เขียวแม่บท (Master Production Blueprint)** ของโปรเจกต์ **Easy ML Viewer** ซึ่งควบคุมลำดับการพัฒนาทั้งหมด **56 Phases** โดยจัดเรียงตาม **ลำดับการพึ่งพาทางวิศวกรรมจริง (Strict Dependency-First Order)** เพื่อให้ทีมพัฒนาหรือ AI Agent สามารถลงมือสร้างระบบทีละเฟสได้อย่างมั่นคง ทดสอบได้จริงในทุกก้าว และไม่เกิดปัญหาคอขวดทางหน่วยความจำหรือการรื้อโครงสร้างย้อนหลัง

---

## 1. ภาพรวมกลยุทธ์การเชื่อมโยงทั้ง 56 Phases (Dependency-First Integration Strategy)

การพัฒนาแบ่งออกเป็น **9 ช่วงหลัก (9 Strategic Parts)** ที่ต่อยอดกันเป็นชั้นๆ ดังนี้:

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
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| PART 8 (Phases 41 - 48): 2D INTUITIVE STUDIO & DUAL-MODE VISUALIZATION SUITE (2D <-> 3D TOGGLE)    |
|  - Phase 41: Dual-Mode 2D/3D Stage Engine, Orthographic Rig, Vector Grid & Universal Boundary Gen  |
|  - Phase 42: 2D Geometries for Linear Models (OLS, Logistic, Ridge, Lasso, ElasticNet, SGD)        |
|  - Phase 43: 2D Geometries for Distance & Kernel Models (KNN Radar, SVM Curved RBF & SV Rings)     |
|  - Phase 44: 2D Geometries for Probabilistic & Discriminant Models (Gaussian NB, Discrete NB, QDA)|
|  - Phase 45: 2D Geometries for Decision Trees & Ensembles (Decision Tree, Random Forest, AdaBoost) |
|  - Phase 46: 2D Geometries for Gradient Boosting Heavyweights (GBM, HistGBM, XGBoost, Light, Cat) |
|  - Phase 47: 2D Geometries for Neural Networks, Pipelines & Universal ONNX (MLP, Pipeline, ONNX)   |
|  - Phase 48: Dual-Mode Synchronized Compare (2D/3D), 2D .mlv Persistence, SVG Export & Hardening  |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
| PART 9 (Phases 49 - 56): 2D POINT INTELLIGENCE & ADVANCED DATA REPRESENTATION SUITE               |
|  - Phase 49: Universal 2D Point Intelligence Core Engine (Comet Trails, Dual-Ring, Error, Hexbin) |
|  - Phase 50: Point Intelligence for Linear Models (Orthogonal Spokes & Margin Corridor Bands)     |
|  - Phase 51: Point Intelligence for Distance & Kernel (KNN Radar Trails & SVM Support Halos)      |
|  - Phase 52: Point Intelligence for Probabilistic & Discriminant (Entropy Flow & Centroid Spokes) |
|  - Phase 53: Point Intelligence for Trees & Ensembles (Forest Consensus Rings & Manhattan Steps)  |
|  - Phase 54: Point Intelligence for Gradient Boosting (TreeSHAP Push Trails & Pseudo-Residuals)   |
|  - Phase 55: Point Intelligence for Neural & ONNX (Dual Space Trails & Data Desert Detectors)     |
|  - Phase 56: Point Intelligence Unified Controls, .mlv Persistence, SVG/Video Export & Hardening  |
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
| **`core/` (Shared Kernel)** | ตั้งค่า Cross-platform `pathlib.Path`, ระบบ Exception มาตรฐาน, Memory Guard (`psutil`), Binary MsgPack+Zstd Codec, และ Process/Thread Pool | Phase 01, 15, 17, 40, 48, 56 |
| **`dataset/`** | โหลดไฟล์ `.csv`, `.xlsx`, `.parquet`, วิเคราะห์สถิติคอลัมน์ (Profiler), แคสต์ตัวเลขเป็น `float32`, และจัดเก็บลง `.cache/*.parquet` | Phase 02, 06, 17 |
| **`model_engine/`** | นิยาม `BaseModelAdapter`, `ModelRegistry`, ระบบสกัด Schema อัตโนมัติ, และรวม Adapter เฉพาะทางของโมเดลทั้ง 20+ ตระกูล | Phase 03, 04, 18–39, 42–47, 50–55 |
| **`mapping/`** | จับคู่คอลัมน์ CSV เข้ากับ Feature ของโมเดลด้วยอัลกอริทึม Fuzzy Matching, ตรวจสอบความเข้ากันได้ของชนิดข้อมูล และจัดการ Categorical Encoding | Phase 05, 35, 37 |
| **`simulation/`** | คำนวณพื้นผิว 3D Mesh ($50 \times 50$), คำนวณ Trajectory Queue, สกัดค่าอิทธิพลรายฟีเจอร์ (SHAP/Contributions), Contour 2D, Point Intelligence และ What-If Sandbox | Phase 06–14, 17, 41, 49 |
| **`recording/`** | บีบอัดผลลัพธ์การจำลองทั้งหมดเป็นไฟล์ `.mlv` (MsgPack + Zstd), โหลดไฟล์ `.mlv` แบบ Instant Playback, ส่งออกคลิป `.webm`/`.mp4`, และเปรียบเทียบ 2 โมเดล | Phase 15, 16, 40, 48, 56 |

---

## 3. ตารางสรุป UI/UX Flow ภาพรวมตั้งแต่ต้นจนจบ (End-to-End Studio UX Flow Matrix)

| ขั้นตอน (Stage) | หน้าจอ / โมดูล UI (Screen / Component) | การกระทำของผู้ใช้ (User Action) | การตอบสนองของระบบและ 3D/2D Stage (System Feedback) |
| :--- | :--- | :--- | :--- |
| **Step 1: Launch & Entry** | **Welcome Hub & Instant Demo Gallery** | อัปโหลดไฟล์โมเดล + ข้อมูล หรือคลิกปุ่ม `"Try 20-Model Demo Gallery"` / `"Open .mlv"` | ตรวจสอบชนิดโมเดลและไฟล์ข้อมูลทันที หรือเปิดไฟล์ `.mlv` เข้าสู่ฉาก 3D/2D ภายใน `< 0.5s` |
| **Step 2: Schema & Mapping** | **Smart Column Mapper Drawer** | ตรวจสอบเส้นเชื่อมจับคู่ `CSV Column <-> Model Feature` (แก้ไขหากต้องการ) | ระบบทำ Auto-Fuzzy Match ให้อัตโนมัติ แจ้งเตือนคอลัมน์ที่มีค่าว่าง (`NaN`) หรือชนิดข้อมูลไม่ตรง |
| **Step 3: Global Triage** | **Global Evaluation Lens & Error Slicer** | คลิกช่อง `False Positives` ใน Confusion Matrix หรือลากคลุมจุดที่ทายผิดสูงในกราฟ Residual | ระบบคัดกรองเฉพาะแถวที่มีปัญหาเข้าสู่คิวการเล่น (Trajectory Queue) ทันทีโดยไม่ต้องไล่หาเอง |
| **Step 4: 3D/2D Stage & Dual Mode** | **Viewport Toolbar (`V`)** | สลับมุมมอง `[ 2D Planar ]` $\leftrightarrow$ `[ 3D Spatial ]` เลือกแกน $X, Y$ | สร้างพื้นผิว 3D หรือแผนที่เส้นขอบเขต 2D Decision Boundary + ความหนาแน่น Heatmap |
| **Step 5: Point Intelligence** | **Point Display Toolbar (`T`, `R`, `E`, `H`)** | เปิดโหมด `Comet Trails`, `Dual-Ring`, `Errors Only`, หรือ `Hexbin Density` | จุดข้อมูลแสดงความสัมพันธ์ตามเวลา, วงแหวนแยกค่าจริง/ทำนาย, ก้าน Residual, และรังผึ้ง 6 เหลี่ยม |
| **Step 6: VCR Playback** | **60 FPS VCR Transport & Multi-Track Timeline** | กด `Play / Pause`, ลากแถบ Scrubber, หรือคลิกหมุด `Anomaly Pin` บน Timeline | ลูกบอลแสงเคลื่อนที่ผ่านพื้นผิว 3D/2D ที่ 60 FPS พร้อมหางแสง Comet Trail และเส้นประทิ้งดิ่งตั้งฉาก |
| **Step 7: Explainability** | **Floating HUD & Model Geometry Overlays** | ดูป้ายข้อมูลที่ลอยตามลูกบอล หรือดูกราฟ Waterfall ด้านขวา | แสดงสมการภายในของโมเดล (เวกเตอร์น้ำหนัก, เรดาร์ KNN, กิ่งต้นไม้, หรือ TreeSHAP Push Vectors) |
| **Step 8: What-If Lab** | **What-If Sandbox & Counterfactual Ghost** | ลากสไลเดอร์ปรับค่าตัวแปร หรือกดปุ่ม `"Find Minimum Change to Flip"` | ลูกบอลเงา **Ghost Sphere** วิ่งแยกออกจากลูกบอลจริงข้ามเส้นแบ่งเขตเพื่อแสดงว่าต้องเปลี่ยนค่าเท่าใดผลลัพธ์จึงพลิก |
| **Step 9: Save & Compare** | **`.mlv` Baker, Video Exporter & Dual Compare** | กดบันทึกเป็นไฟล์ `.mlv`, อัดวิดีโอ `.mp4`, หรือเปิดเทียบ 2 โมเดลซ้าย-ขวา | บันทึกไฟล์ `.mlv` ขนาดกะทัดรัด หรือซิงค์ฉาก 2D/3D ซ้าย-ขวาเพื่อชี้จุดที่สองโมเดลทายขัดแย้งกัน |

---

## 4. ตารางควบคุมงบประมาณหน่วยความจำและทรัพยากรเครื่องมาตรฐาน (Global Memory Budget & Resource Optimization Contract)

ทุกเฟสตั้งแต่ Phase 01 ถึง Phase 56 **ต้องปฏิบัติตามข้อกำหนดด้านหน่วยความจำและประสิทธิภาพตารางนี้อย่างเคร่งครัด**:

| องค์ประกอบ (Subsystem / Resource) | เพดานควบคุมสูงสุด (Strict Limit / Budget) | กลไกบังคับใช้ทางวิศวกรรม (Mandatory Engineering Enforcement) |
| :--- | :--- | :--- |
| **1. Numeric Data Precision** | ห้ามใช้ `float64` ใน Pipeline หลัก (`float32` เท่านั้น) | แคสต์ทุกคอลัมน์ตัวเลขเป็น `np.float32` และดัชนีคลาส/ใบไม้เป็น `np.uint16` / `np.int32` ตั้งแต่วินาทีที่อ่านไฟล์ (ลด RAM 50%) |
| **2. Dataset Storage Cache** | Peak RAM ตอนโหลดไฟล์ CSV/Excel $\le 250\text{ MB}$ | แปลงไฟล์ `.csv`/`.xlsx` เก็บลงดิสก์ใน `.cache/*.parquet` แล้วอ่านเฉพาะคอลัมน์และช่วงแถวที่ใช้งานผ่าน PyArrow Memory-Mapping |
| **3. Inference Chunk Size** | สูงสุด `2,500` แถวต่อ 1 Chunk | ห้ามส่งข้อมูลทั้งไฟล์ (เช่น 100,000 แถว) เข้า `predict()` หรือ `SHAP` รวดเดียว ต้องตัดแบ่งทีละ `2,500` แถวใน `ComputePool` เสมอ |
| **4. 3D Surface Grid Resolution** | มาตรฐาน $50 \times 50$ (`2,500` จุด) / สูงสุด $80 \times 80$ | คำนวณพื้นผิว 3D ด้วยกริด $50 \times 50$ (`float32` ขนาดเพียง `10 KB`) ทำให้สร้าง Mesh เสร็จภายใน `< 150 ms` |
| **5. 2D Contour Grid Resolution** | มาตรฐาน $120 \times 120$ (`14,400` จุด) | Vectorized Marching Squares คำนวณเสร็จใน $< 80$ ms ใน Background Thread |
| **6. Frontend VCR State Store** | ห้ามเก็บเฟรมใน React State Array ของ JSON Objects | จัดเก็บข้อมูลทุกเฟรมใน **Flat TypedArrays** (`Float32Array`, `Uint16Array`) และอ่านค่าด้วย Pointer Index ใน `useFrame()` |
| **7. WebGL Draw Calls & SVG Nodes** | $\le 15$ Draw Calls ต่อฉาก 3D / จุด 2D ผ่าน Instanced/Batching (60 FPS คงที่) | ใช้ `THREE.InstancedMesh` สำหรับจุด 3D และ SVG Path Batching / Hexbin Aggregation สำหรับระนาบ 2D |
| **8. CPU / GPU Worker Threads** | `max_workers = max(1, physical_cores - 1)` | เว้นอย่างน้อย 1 CPU Core ไว้ให้ FastAPI Event Loop เสมอ เพื่อให้ API ตอบสนองไวและยกเลิกงาน (`Cancel Job`) ได้ทันที |
| **9. Process & Port Lifecycle** | `0` Zombie Processes / Temp Cache $\le 1.0\text{ GB}$ | สคริปต์ Launcher ต้องดักจับ `SIGINT`/`SIGTERM`/`EXIT` เพื่อปิดทั้งพอร์ต `30211` และ `30222` พร้อมกัน และมีระบบกวาดล้าง `.cache/` |

---

## 5. สารบัญแผนการพัฒนาทั้ง 56 Phases (Complete 56-Phase Blueprint Index)

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

### 🗺️ Part 8: 2D Intuitive Studio & Dual-Mode Visualization Suite (Phases 41 - 48)
41. [phase-41-dual-mode-2d-3d-stage-engine-and-contour-generator.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-41-dual-mode-2d-3d-stage-engine-and-contour-generator.md) — เครื่องยนต์แสดงผลแบบสองโหมด (2D Planar $\leftrightarrow$ 3D Spatial), กล้อง Orthographic Rig, Vector Grid & อัลกอริทึม Marching Squares Decision Boundary
42. [phase-42-model-group-linear-2d-geometry.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-42-model-group-linear-2d-geometry.md) — เรขาคณิต 2 มิติสำหรับกลุ่มโมเดล Linear (Linear Regression, Logistic, Ridge, Lasso Zero-Axis, ElasticNet & SGD Margin Corridor)
43. [phase-43-model-group-distance-kernel-2d-geometry.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-43-model-group-distance-kernel-2d-geometry.md) — เรขาคณิต 2 มิติสำหรับกลุ่ม Distance & Kernel (KNN 2D Radar Circle & Spider Lines, SVM Curved RBF Boundary & Support Vector Halos)
44. [phase-44-model-group-probabilistic-discriminant-2d-geometry.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-44-model-group-probabilistic-discriminant-2d-geometry.md) — เรขาคณิต 2 มิติสำหรับกลุ่ม Probabilistic & Discriminant (Gaussian NB Bivariate Ellipses $1\sigma-3\sigma$, Discrete NB Steps & LDA/QDA Conic Section Curves)
45. [phase-45-model-group-trees-ensembles-2d-geometry.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-45-model-group-trees-ensembles-2d-geometry.md) — เรขาคณิต 2 มิติสำหรับกลุ่ม Decision Trees & Bagging (Decision Tree Axis-Aligned Partitions & Active Leaf Box, Random Forest Consensus & AdaBoost Cascade)
46. [phase-46-model-group-gradient-boosting-2d-geometry.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-46-model-group-gradient-boosting-2d-geometry.md) — เรขาคณิต 2 มิติสำหรับกลุ่ม Gradient Boosting (XGBoost 2D TreeSHAP Vectors, CatBoost Oblivious Checkerboard Grid, HistGBM 255-Bin Grid & LightGBM Polygons)
47. [phase-47-model-group-neural-pipeline-onnx-2d-geometry.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-47-model-group-neural-pipeline-onnx-2d-geometry.md) — เรขาคณิต 2 มิติสำหรับกลุ่ม Neural Networks, Pipelines & ONNX (MLP Hidden Neuron Hyperplanes & Dead ReLUs, Pipeline Raw/Scaled Space Toggle & ONNX 2D Latent SVD Plane)
48. [phase-48-dual-mode-compare-mlv-baker-export-and-hardening.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-48-dual-mode-compare-mlv-baker-export-and-hardening.md) — โหมดเปรียบเทียบ 2 โมเดลในระนาบ 2D (Conflict Zones), บันทึกแคช 2D ในไฟล์ `.mlv`, ส่งออก Vector SVG/4K Image/Video & 100,000-Row 2D Stress Test

### 🔬 Part 9: 2D Point Intelligence & Advanced Data Representation Suite (Phases 49 - 56)
49. [phase-49-2d-point-intelligence-core-engine.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-49-2d-point-intelligence-core-engine.md) — โครงสร้างพื้นฐานกลาง Universal 2D Point Intelligence (Comet Trails, Concentric Dual-Rings, Error Lens & Hexbin Grid Tessellation Engine)
50. [phase-50-point-intelligence-linear-models.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-50-point-intelligence-linear-models.md) — 2D Point Intelligence สำหรับโมเดล Linear (Orthogonal Decision Plane Spokes, Margin Corridor Violation Rings & Vertical Residual Stems)
51. [phase-51-point-intelligence-distance-kernel-models.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-51-point-intelligence-distance-kernel-models.md) — 2D Point Intelligence สำหรับโมเดล Distance & Kernel (KNN Radar Swapping Trails, Support Vector Diamond Halos & SVR $\epsilon$-Tube Slack Spokes)
52. [phase-52-point-intelligence-probabilistic-models.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-52-point-intelligence-probabilistic-models.md) — 2D Point Intelligence สำหรับโมเดล Probabilistic & Discriminant (Shannon Entropy Trail Fading, Centroid Attractor Spokes & Gaussian Divergence Hexbins)
53. [phase-53-point-intelligence-trees-ensembles.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-53-point-intelligence-trees-ensembles.md) — 2D Point Intelligence สำหรับโมเดล Trees & Ensembles (Forest Voting Agreement Variance Rings, Manhattan Step Trails & AdaBoost Weight Scaling)
54. [phase-54-point-intelligence-gradient-boosting.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-54-point-intelligence-gradient-boosting.md) — 2D Point Intelligence สำหรับโมเดล Gradient Boosting (Staged TreeSHAP Push Vector Trails, Functional Margin Dual-Rings & Pseudo-Residual Spokes)
55. [phase-55-point-intelligence-neural-pipeline-onnx.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-55-point-intelligence-neural-pipeline-onnx.md) — 2D Point Intelligence สำหรับ Neural Networks, Pipelines & ONNX (Dual Space Raw/Scaled Trails, Softmax Logit Gap Rings & Data Desert Detectors)
56. [phase-56-point-intelligence-hardening-and-persistence.md](file:///e:/easy-ml-viewer/ai-dev-plan/dev/phase-56-point-intelligence-hardening-and-persistence.md) — แผงควบคุมแบบรวมศูนย์ (`T`, `R`, `E`, `H`), บันทึกแทร็ก Point Intelligence ลงไฟล์ `.mlv`, ส่งออกภาพเวกเตอร์ SVG / วิดีโอ WebM & Hardening

---

## 6. ตารางสรุปเรขาคณิต 3 มิติ, เรขาคณิต 2 มิติ และกลไกการแกะโครงสร้างภายในของทั้ง 20+ ตระกูลโมเดล (20+ Model Families 3D/2D & Explainability Matrix)

| ลำดับ | ตระกูลโมเดล (Model Family) | ไฟล์ที่รองรับ | รูปทรงเรขาคณิต 3D (3D Stage) | รูปทรงเรขาคณิต 2D (2D Planar Mode) | Point Intelligence 2D (Phases 49–55) | กลไกสกัดคำอธิบายภายใน (Attribution) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | **Linear Regression** (Phase 04, 50) | `.pkl`, `.joblib` | ระนาบเอียง 3 มิติ (Tilted Hyperplane) | เส้นสมการ $w_1 x_1 + w_2 x_2 + b = 0$ + กราฟ 2D Slope | Vertical Residual Stems & Bounded Hexbins | $\phi_j = \beta_j (x_j - \bar{x}_j)$ |
| **02** | **Logistic Regression** (Phase 18, 50) | `.pkl`, `.joblib` | คลื่น S-Curve / Softmax Terraces | เส้นแบ่งเขต $P=0.5$ + Sigmoid Density Gradient Heatmap | Probability Confidence Ring & Orthogonal Spokes | Log-Odds Contribution $\beta_{c,j} x_j$ |
| **03** | **Ridge ($L_2$) & Lasso ($L_1$)** (Phase 19, 50) | `.pkl`, `.joblib` | ระนาบเอียงพร้อมแกนที่ถูกตัดเป็นศูนย์ | เส้นแบ่งเขตขนานแกนฟีเจอร์ที่เป็นศูนย์ ($\emptyset$ Zero-Axis) | Sparsity-Aware Hexbins & Zero-Axis Badges | สัมประสิทธิ์ $\beta_j x_j$ + Sparsity Flag |
| **04** | **ElasticNet** (Phase 20, 50) | `.pkl`, `.joblib` | ระนาบผสม $L_1 + L_2$ | เส้นตรง Regularized Line + แถบวัดสัดส่วน $L_1/L_2$ | Regularization Balance Spokes | สัดส่วนการหดตัว $L_1$ vs $L_2$ |
| **05** | **SGD Classifier / Regressor** (Phase 21, 50) | `.pkl`, `.joblib` | ระนาบตัดสินใจ + ขอบเขต Margin | เส้น Decision Line + แถบทางเดิน Margin Corridor $w^Tx+b=\pm 1$ | Pulsing Amber Margin Violation Rings | ระยะห่างตั้งฉาก (`decision_function`) |
| **06** | **KNN** (Phase 22, 51) | `.pkl`, `.joblib` | ลูกบอลเรดาร์ 3D + ใยแมงมุม | วงกลมเรดาร์ 2D ($R_t = d_{(K)}$) + เส้นใยแสง 2D เชื่อม $K$ จุด | Dynamic Neighbor Swap Trails & Vote Arc Rings | น้ำหนัก $w_k = 1 / (d_k + \epsilon)$ |
| **07** | **SVM (`SVC`/`SVR`)** (Phase 23, 51) | `.pkl`, `.joblib` | พื้นผิวอานม้า RBF Saddle + เสาแสง | เส้นโค้ง RBF Boundary + วงแหวนเพชร Support Vectors | Support Vector Diamond Halos & SVR $\epsilon$-Spokes | แรงดึงดูด $\alpha_m y_m K(\mathbf{x}_m, \mathbf{x})$ |
| **08** | **Gaussian Naive Bayes** (Phase 24, 52) | `.pkl`, `.joblib` | ภูเขาระฆังคว่ำ 3D Bell Hills | วงรีความน่าจะเป็น Bivariate Normal ($1\sigma, 2\sigma, 3\sigma$) | Entropy Color Trails & Centroid Attractor Spokes | ผลรวม Log-Likelihood $-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}$ |
| **09** | **Multinomial & Bernoulli NB** (Phase 25, 52) | `.pkl`, `.joblib` | หอคอยแท่งสะสมความน่าจะเป็น 3D | แผนผังเส้นชั้นบันได Log-Likelihood ขั้นบันได 2 มิติ | Empirical vs NB Theoretical Divergence Hexbins | คะแนน Evidence รายคำ $x_j \log P(f_j|c)$ |
| **10** | **LDA & QDA** (Phase 26, 52) | `.pkl`, `.joblib` | ทรงรีความแปรปรวนร่วม 3 มิติ | LDA: เส้นตรง / QDA: เส้นโค้งภาคตัดกรวย + วงรี $\boldsymbol{\Sigma}_c$ | Mahalanobis Distance Ellipsoid Halos | Mahalanobis Distance $(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})$ |
| **11** | **Decision Tree** (Phase 27, 53) | `.pkl`, `.joblib` | ขั้นบันได 3D Voxel Terraces | เส้นตัดตั้งฉากแกน (Orthogonal Partitions) + ไฮไลต์กล่อง Leaf | Manhattan Step Trails & Leaf Minority Spokes | เส้นทางเดินผ่านโหนด (`decision_path`) |
| **12** | **Random Forest** (Phase 28, 53) | `.pkl`, `.joblib` | ขั้นบันไดละเอียด + ลานโหวต 3D | เส้นตัดซ้อนทับกันหลายต้น + แถบสี Consensus Heatmap | Forest Voting Consensus Variance Rings | คะแนนโหวตรายต้น + Tree Contributions |
| **13** | **Extra Trees** (Phase 29, 53) | `.pkl`, `.joblib` | ขั้นบันไดเฉลี่ย + วงรัศมีความแปรปรวน | เส้นตัดสุ่มพร้อมแถบสีความน่าจะเป็นที่เรียบเนียนขึ้น | Randomized Cut Variance Dual-Rings | ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน |
| **14** | **AdaBoost** (Phase 30, 53) | `.pkl`, `.joblib` | ลำดับการแก้คำตอบทีละขั้น Cascade | สไลเดอร์ดูวิวัฒนาการเส้นแบ่งเขตตามลำดับ Weak Learner | Instance Sample Weight Ring Scaling ($\omega_i$) | การสะสมคะแนน `staged_predict_proba` |
| **15** | **Scikit-Learn GBM** (Phase 31, 54) | `.pkl`, `.joblib` | พื้นผิวการสะสมค่า Residual 3D | แผนที่การสะสมคะแนน Log-Odds / Residual 2 มิติ | Staged Residual Descent Trails & Pseudo-Residuals | การไต่ระดับ Residual ทีละ Stage |
| **16** | **HistGradientBoosting** (Phase 32, 54) | `.pkl`, `.joblib` | ขอบเขตถังข้อมูล 3D Histogram Bins | เส้นกริดแสดงขอบเขตถังข้อมูล 255 Bins ตัดกับ Boundary | Quantized 256-Bin Snapped Hexbin Grid | ดัชนีถังข้อมูล `_bin_mapper` |
| **17** | **XGBoost** (Phase 33, 54) | `.pkl`, `.json`, `.ubj` | ภูมิประเทศขั้นบันได + แรงผลัก 3D | รูปหลายเหลี่ยมใบไม้ + ลูกศรแรงผลัก 2D TreeSHAP Vectors | Cumulative TreeSHAP Push Vector Trails | C++ `pred_contribs` & `pred_leaf` |
| **18** | **LightGBM** (Phase 34, 54) | `.pkl`, `.txt`, `.model` | ภูมิประเทศแตกกิ่งไม่สมมาตร 3D | ขอบเขตใบไม้รูปทรงอิสระ (Leaf-wise Polygons 2D) | Leaf-wise Margin Gradient Dual-Rings | C++ Native TreeSHAP |
| **19** | **CatBoost** (Phase 35, 54) | `.cbm`, `.pkl` | ตารางหมากรุกสมมาตร 3 มิติ | ตารางหมากรุกสมมาตร 2 มิติ (2D Oblivious Grid) | Oblivious Grid Cell Density Hexbins | Bitwise Leaf Index + C++ SHAP |
| **20** | **MLP Neural Network** (Phase 36, 55) | `.pkl`, `.joblib` | โครงข่ายประสาทลอยฟ้า 3 มิติ | ระนาบตัดของ Hidden Neurons + เส้นแบ่งเขตโค้งมนเรียบ | Softmax Logit Gap Rings & Loss Gradient Spokes | Layer Activation $A^{(l)}$ & Gradients |
| **21** | **Sklearn `Pipeline`** (Phase 37, 55) | `.pkl`, `.joblib` | อุโมงค์แปลงข้อมูล 3 มิติ | ตัวสลับมุมมอง [Raw Space] $\leftrightarrow$ [Scaled Space] 2D | Dual Space Raw/Scaled Trajectory Trails | Step-by-Step Value Tracer |
| **22** | **ONNX Classical ML** (Phase 38, 55) | `.onnx` | ปรับเปลี่ยนรูปทรงตามโหนด `ai.onnx.ml` | Universal 2D Contour Map ตามชนิดโหนด ONNX | Operator Output Spread Rings | Decompiled Operators & Attributes |
| **23** | **ONNX Deep Learning** (Phase 39, 55) | `.onnx` | พื้นผิว 3D + กลุ่มหมอกดาว Latent | ระนาบฉายภาพแฝง 2D Latent SVD/PCA Projection Plane | Data Desert Hazard Badges & Latent Geodesics | In-Memory Graph Output Injection |

---

## 7. เกณฑ์การตรวจรับคุณภาพระดับระบบ (Master Quality Gate Checklist)
- [ ] **Architecture Purity**: โค้ดใน `backend/app/domains/` แยก Bounded Context ชัดเจน ไม่มีการ Import ข้ามโดเมนโดยไม่ผ่าน `app/core/` หรือ Application Orchestrator
- [ ] **Memory Safety**: ทดสอบกับข้อมูลขนาด `100,000` แถว โดย Peak RAM ของ Backend ไม่เกิน `600 MB` และไม่มี Memory Leak หลังรันต่อเนื่อง 50 รอบ
- [ ] **60 FPS Real-Time Scrubbing**: การลากแถบเวลา VCR และการหมุนกล้อง 3D ทั้งในโหมดปกติและโหมดเปรียบเทียบ 2 โมเดล (Dual `.mlv` Compare) รวมถึงโหมด 2D Point Intelligence รักษาความลื่นไหลที่ 60 FPS
- [ ] **100% Air-Gapped Offline**: เปิดใช้งานผ่าน `start-windows.bat`, `start-windows.ps1`, และ `start-unix.sh` ได้โดยไม่ต้องต่ออินเทอร์เน็ต และปิดโปรเซสลูกสะอาดหมดจดเมื่อเลิกใช้งาน
