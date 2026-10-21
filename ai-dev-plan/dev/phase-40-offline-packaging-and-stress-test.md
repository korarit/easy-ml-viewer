# Phase 40: Cross-Platform Offline Launchers (Windows/Mac/Linux), 20-Model Demo Suite, Multi-Recording Compare & Hardening

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
ปิดท้ายการพัฒนาโปรเจกต์ในเฟสที่ 40 โดยหลอมรวม Base Platform (Phases 01–17) และตระกูลโมเดลทั้ง 20+ ตระกูล (Phases 18–39) เข้าเป็นผลิตภัณฑ์ระดับ Production ที่ทำงานได้แบบ **100% Air-Gapped Offline** บนทุกระบบปฏิบัติการ (**Windows, macOS, และ Linux**) พร้อมระบบสคริปต์เปิดโปรแกรมคลิกเดียวที่จัดการ Process Lifecycle & Graceful Port Cleanup อย่างรัดกุม, ชุดโมเดลตัวอย่างสำเร็จรูป 20 ตระกูล, โหมดเปรียบเทียบ 2 โมเดลพร้อมกัน (**Synchronized Dual `.mlv` Compare**), และชุดทดสอบความทนทานระดับ 100,000 แถว (**Stress Benchmark Suite**)

### ขอบเขตการพัฒนา (Scope)
1. **Cross-Platform Offline Launchers & Process Lifecycle Manager**: สคริปต์ `start-windows.bat`, `start-windows.ps1`, และ `start-unix.sh` ที่ตรวจสอบพอร์ตค้าง, รัน Backend (`30211`) และ Frontend (`30222`), และดักจับสัญญาณปิดโปรแกรม (`SIGINT`/`SIGTERM`/Window Close) เพื่อฆ่า Child Processes ทั้งหมดไม่ให้เหลือ Zombie ค้างพอร์ต
2. **20-Model Instant Demo Suite Generator (`backend/scripts/generate_all_20_demos.py`)**: สร้างไฟล์โมเดลครบทั้ง 20+ ตระกูล พร้อมชุดข้อมูล `.csv` และไฟล์บันทึก `.mlv` สำเร็จรูปสำหรับปุ่ม **"Instant Demo Gallery"** บนหน้าแรก
3. **Synchronized Dual `.mlv` Split-Screen Comparison Studio**: โหลดไฟล์ `.mlv` 2 ไฟล์ขึ้นมาเล่นเปรียบเทียบกันแบบซ้าย-ขวา (เช่น `RandomForest.mlv` vs `XGBoost.mlv`) ด้วยแถบเวลา VCR และมุมกล้อง 3D ที่ซิงค์กันแบบเฟรมต่อเฟรม พร้อมระบบตรวจจับแถวที่สองโมเดลทายขัดแย้งกัน (**Prediction Disagreement Highlighter**)
4. **100,000-Row Stress Benchmark & 100% Offline Audit Suite**: ชุดทดสอบอัตโนมัติที่ยืนยันเพดานหน่วยความจำ, อัตราเฟรม 60 FPS, และการไม่เชื่อมต่ออินเทอร์เน็ตภายนอก `127.0.0.1`

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 ระบบจัดการ Process Lifecycle & Graceful Port Cleanup ข้ามแพลตฟอร์ม
ปัญหาคลาสสิกของแอปพลิเคชันแบบ 2 พอร์ต (`30211` และ `30222`) คือเมื่อผู้ใช้ปิดหน้าต่าง Terminal กะทันหัน โปรเซสลูก (`uvicorn` หรือ `node/vite`) อาจกลายเป็น Orphan/Zombie Process ที่ยึดพอร์ตค้างไว้ทำให้เปิดโปรแกรมครั้งถัดไปไม่ติด ระบบจึงออกแบบกลไกป้องกัน 3 ชั้น:
1. **Pre-Flight Stale Port Sweeper**:
   - ก่อนเริ่มรัน สคริปต์จะตรวจสอบว่าพอร์ต `30211` หรือ `30222` ถูกใช้งานอยู่หรือไม่ หากพบว่าเป็นโปรเซสค้างของ Easy ML Viewer จะเสนอให้สั่งเคลียร์พอร์ต (`netstat -ano` + `taskkill /F /PID` บน Windows หรือ `lsof -ti:30211 | xargs kill -9` บน macOS/Linux)
2. **Signal Trapping & Process Tree Termination**:
   - บน **macOS / Linux (`start-unix.sh`)**: ใช้ `trap 'kill 0' SIGINT SIGTERM EXIT` เพื่อส่งสัญญาณปิดไปยัง Process Group ทั้งหมด (`backend_pid` และ `frontend_pid`) ทันทีที่ผู้ใช้กด `Ctrl+C` หรือปิดเทอร์มินัล
   - บน **Windows (`start-windows.ps1` / `start-windows.bat`)**: ใช้ PowerShell `try { ... } finally { Stop-Process -Id $backendProc.Id, $frontendProc.Id -Force -ErrorAction SilentlyContinue }` ควบคู่กับ `taskkill /T /F /PID` เพื่อกวาดล้าง Process Tree ทั้งยวง
3. **Backend Parent-Watchdog & Temp Cache Sweeper**:
   - มี Endpoint `DELETE /api/v1/system/cache` และระบบ Auto-Clean ลบไฟล์ `.parquet` ชั่วคราวใน `.cache/` ที่มีอายุเกิน 24 ชั่วโมงเมื่อเริ่มเปิดเซิร์ฟเวอร์

### 2.2 คณิตศาสตร์และกลไกของโหมดเปรียบเทียบ 2 โมเดล (Synchronized Dual `.mlv` Compare)
เมื่อผู้ใช้เปิดไฟล์บันทึก $\mathcal{R}_A$ (โมเดล $A$) และ $\mathcal{R}_B$ (โมเดล $B$) บนชุดข้อมูลขนาด $N$ แถวเดียวกัน:
1. **Frame-Locked VCR Synchronization**:
   - ทั้งสอง Viewport อ่านตัวแปรเวลา `currentFrameFloat` ($t \in [0, N-1]$) ตัวเดียวกันจาก `useDualCompareStore` ทำให้การกด Play, Pause, หรือลาก Scrubber ขยับลูกบอลแสงบนพื้นผิว 3D ทั้งฝั่งซ้ายและฝั่งขวาพร้อมกันในระดับ Sub-millisecond
2. **Disagreement & Divergence Metric Computation**:
   - สำหรับงาน **Classification**: คำนวณดัชนีความขัดแย้ง (Prediction Disagreement Mask $D_i$) และระยะห่างของการแจกแจงความน่าจะเป็น (Jensen-Shannon / Total Variation Divergence $\Delta P_i$):
     $$
     D_i = \mathbb{I}\left(\hat{y}^{(A)}_i \ne \hat{y}^{(B)}_i\right), \qquad \Delta P_i = \frac{1}{2} \sum_{c=1}^{C} \left|P_A(y=c|\mathbf{x}_i) - P_B(y=c|\mathbf{x}_i)\right| \in [0, 1]
     $$
   - สำหรับงาน **Regression**: คำนวณความต่างสัมบูรณ์ $\Delta \hat{y}_i = \left|\hat{y}^{(A)}_i - \hat{y}^{(B)}_i\right|$
3. **Disagreement Timeline Pins**:
   - แถบ Timeline ด้านล่างจะปักหมุดสีม่วงแดง (`Disagreement Pin`) ณ เฟรมที่ $D_i = 1$ (เช่น โมเดล $A$ บอกว่าอนุมัติสินเชื่อ แต่โมเดล $B$ บอกว่าปฏิเสธ) เพื่อให้ผู้ใช้คลิกกระโดดไปวิเคราะห์จุดที่สองโมเดลคิดไม่เหมือนกันได้ทันที!

### 2.3 ชุดทดสอบความทนทาน 100,000 แถว (100,000-Row Stress Benchmark Architecture)
เพื่อพิสูจน์ว่าระบบรองรับข้อมูลขนาดใหญ่ได้โดยไม่ค้าง ชุดทดสอบ `backend/tests/stress/test_100k_benchmark.py` จะจำลองชุดข้อมูลสังเคราะห์ขนาด `100,000` แถว $\times$ `25` คอลัมน์:
- ตรวจสอบว่า `DatasetIngestionService` แปลงข้อมูลลง `.cache/*.parquet` (`float32`) โดยใช้ Peak RAM Delta ไม่เกิน `250 MB`
- ตรวจสอบว่าการดึง Trajectory Queue ขนาด `5,000` เฟรม และการคำนวณพื้นผิว 3D Mesh ($50 \times 50 = 2,500$ จุด) เสร็จสมบูรณ์ภายในเวลา `< 1.5 วินาที`
- ตรวจสอบว่าในฝั่ง Frontend การเรนเดอร์ผ่าน `THREE.InstancedMesh` และ TypedArray Scrubber รักษา Frame Time เฉลี่ย `< 16.6 ms` (60 FPS) โดยไม่เกิด WebGL Context Lost

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)
1. **Instant 20-Model Demo Gallery Modal (บนหน้าแรก)**:
   - การ์ดสวยงามแบ่งตาม 6 หมวดหมู่ (Linear, Distance/Kernel, Probabilistic, Trees/Bagging, Gradient Boosting, Neural/Pipeline/ONNX) คลิกการ์ดใดก็เปิดดูฉาก 3D ของโมเดลนั้นได้ทันทีภายใน 0.5 วินาที
2. **Split-Screen Dual `.mlv` Comparison Studio**:
   - หน้าจอแบ่งครึ่งซ้าย (Model A) และขวา (Model B) พร้อมสวิตช์ `[x] Sync 3D Camera Orbit` (หมุนกล้องฝั่งซ้ายแล้วฝั่งขวาหมุนตามในมุมเดียวกันเป๊ะ) และ `[x] Sync VCR Scrubber`
3. **Disagreement Track & "Jump to Next Conflict" Button**:
   - แถบสีพิเศษบน Timeline ที่สว่างขึ้นเฉพาะช่วงที่ Model A กับ Model B ทายผลต่างกัน พร้อมปุ่มลัด `[Jump to Next Disagreement (Alt+D)]`
4. **System Resource & Offline Status Pill (มุมขวาบนของ Top Bar)**:
   - แสดงสถานะ `[Offline Mode: 100% Local]` คู่กับมาตรวัดการใช้ RAM ของ Backend, สถานะ GPU/CPU Worker, และปุ่ม `[Clear Temp Cache]`

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

### UX Flow
1. ผู้ใช้ดับเบิลคลิกไฟล์ `start-windows.bat` (หรือรัน `./start-unix.sh` บน Mac/Linux)
2. สคริปต์ตรวจสอบพอร์ต `30211` และ `30222`, เปิด Backend และ Frontend, แล้วเปิดเบราว์เซอร์เข้าสู่ `http://127.0.0.1:30222` อัตโนมัติ
3. ผู้ใช้คลิกปุ่ม `[Compare 2 Models (.mlv)]` แล้วเลือก `RandomForest.mlv` คู่กับ `CatBoost.mlv`
4. หน้าจอแสดงฉาก 3D เปรียบเทียบกันแบบซ้าย-ขวา ผู้ใช้กดปุ่ม `[Next Conflict]` ระบบเลื่อน Scrubber ไปยังแถวที่ `#142` ซึ่ง Random Forest ทายว่า `Class 0 (0.54)` แต่ CatBoost ทายว่า `Class 1 (0.78)` พร้อมแสดง Waterfall เทียบกันหมัดต่อหมัด

### ASCII Wireframe
```text
+--------------------------------------------------------------------------------------------------+
| [Easy ML Viewer]  DUAL MODEL COMPARE STUDIO     [x] Sync 3D Camera  [x] Sync VCR   [Offline 100%]|
| Left (Model A): RandomForest.mlv (Acc: 91.2%)   | Right (Model B): CatBoost.mlv (Acc: 94.1%)     |
+-------------------------------------------------+------------------------------------------------+
| 3D STAGE A: Random Forest Voxel Terraces        | 3D STAGE B: CatBoost Symmetric Oblivious Grid  |
|                                                 |                                                |
|      Z (Prob)                                   |      Z (Prob)                                  |
|      ^      +-------+-------+                   |      ^     +-------+-------+-------+           |
|  1.0 |     /       /       /|                   |  1.0 |    /       /  (*)  /       /|           |
|  0.5 |    +-------+-------+ | <-- Row #142      |  0.5 |   +-------+-------+-------+ |           |
|      |    |  (*)  |       | +     Prob: 0.46    |      |   |       |  0.78 |       | +           |
|  0.0 +----+-------+-------+---> X (Class 0!)    |  0.0 +---+-------+-------+-------+---> X       |
|                                                 |          (Class 1! -> CONFLICT DETECTED)       |
+-------------------------------------------------+------------------------------------------------+
| SIDE-BY-SIDE WATERFALL COMPARISON (Row #142 — Actual Label: Class 1 [CatBoost Wins!])            |
| Feature       | Model A (RandomForest) Contrib  | Model B (CatBoost) Contrib   | Delta (B - A)   |
| occupation    | +0.04 [+]                       | +0.31 [+++++++] (Native Cat) | +0.27           |
| income        | -0.12 [---]                     | -0.05 [-]                    | +0.07           |
+--------------------------------------------------------------------------------------------------+
| Conflict Track: [....|..||....*....|.......] 18 Disagreements Found  [<< Prev] [Next Conflict >>]|
| [|<] [<] [ || Pause ] [>] [>|]    Frame: 0142 / 1000   [======*=================]  Speed: [1.0x] |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD แยกรายชั้นพร้อมชื่อไฟล์และ Code Signatures

### 5.1 Root Launchers & Scripts
- **`start-windows.bat`** & **`start-windows.ps1`**: สคริปต์ตรวจสอบพอร์ต, เปิด `uvicorn` (`127.0.0.1:30211`) + `vite` (`127.0.0.1:30222`), และจัดการ Clean Shutdown เมื่อปิดหน้าต่าง
- **`start-unix.sh`**: สคริปต์ Bash สำหรับ macOS/Linux พร้อม `trap` ปิด Process Group
- **`backend/scripts/generate_all_20_demos.py`**:
  ```python
  def generate_all_demo_artifacts(output_dir: Path) -> None:
      """Trains and exports all 20+ model families (.pkl, .cbm, .ubj, .txt, .onnx),
      reference datasets (.csv), and pre-baked .mlv recordings."""
      ...
  ```

### 5.2 Backend (`backend/app/domains/recording/` & `system/`)
- **`app/domains/recording/application/dual_compare_service.py`**
  ```python
  from dataclasses import dataclass
  import numpy as np
  from app.domains.recording.domain.entities import MLVRecordingPayload

  @dataclass(slots=True)
  class DualComparisonSummary:
      total_aligned_frames: int
      disagreement_indices: np.ndarray       # uint32 indices where pred_A != pred_B
      probability_divergence: np.ndarray     # float32 [0..1] per frame
      agreement_rate: float

  class DualRecordingCompareService:
      def compare_recordings(
          self, rec_a: MLVRecordingPayload, rec_b: MLVRecordingPayload
      ) -> DualComparisonSummary: ...
  ```

### 5.3 Frontend (`frontend/src/domains/recording/` & `simulation/`)
- **`src/domains/recording/store/useDualCompareStore.ts`**: จัดการสถานะโมเดลซ้าย/ขวา, รายการดัชนีความขัดแย้ง `disagreementIndices`, และสถานะล็อกมุมกล้อง `isCameraSynced`
- **`src/domains/simulation/components/compare/DualStageCompareWorkspace.tsx`**: คอมโพเนนต์หน้าจอคู่ที่เชื่อม `OrbitControls` ของ Canvas ซ้ายและขวาเข้าด้วยกันผ่าน Event Ref โดยไม่เกิด Infinite Re-render Loop

---

## 6. แผนการตรวจสอบหน่วยความจำและเพิ่มประสิทธิภาพทรัพยากรเครื่อง (Memory Check Plan & Process Resource Optimization)
1. **Shared WebGL Geometry & Buffer Disposal in Dual View**:
   - การเปิด 2 ฉาก 3D พร้อมกันต้องระวังโควตา WebGL Context ของเบราว์เซอร์ (ปกติจำกัดไม่เกิน 8–16 Contexts ต่อแท็บ) ระบบสามารถใช้ `@react-three/drei` `<View>` Scissor Портบน `<Canvas>` เดียวกันเพื่อวาด 2 ฉาก 3D ด้วย WebGL Context เดียว! ช่วยลดการใช้ VRAM ลง 45% และป้องกันปัญหา Context Lost อย่างเด็ดขาด
2. **Strict Memory Budget Verification (`psutil.Process().memory_info().rss`)**:
   - ในชุดทดสอบ 100,000 แถว บังคับตรวจสอบว่า RSS Memory ของโปรเซส Backend ตลอดการประมวลผล Chunking ต้องไม่เกิน `600 MB` และลดลงกลับมาใกล้ระดับ Baseline หลังเรียก `gc.collect()`
3. **Temp `.cache/` Quota Enforcement**:
   - ระบบตรวจสอบขนาดโฟลเดอร์ `.cache/` ทุกครั้งที่เริ่มงาน หากขนาดรวมเกิน `1.0 GB` จะลบไฟล์ `.parquet` ชั่วคราวที่เก่าที่สุดออกโดยอัตโนมัติ (LRU Cache Eviction)

---

## 7. ตารางข้อควรระวังและ Edge Cases (Caveats & Edge Case Matrix)

| กรณีปัญหา (Edge Case) | สาเหตุเชิงเทคนิค | แนวทางป้องกันและแก้ไขในระบบ (Mitigation Strategy) |
| :--- | :--- | :--- |
| **1. Zombie Process ค้างพอร์ต `30211` หรือ `30222`** | ผู้ใช้กดกากบาทปิดหน้าต่าง Terminal โดยไม่กด `Ctrl+C` | ใช้ `trap ... EXIT` ใน `start-unix.sh` และใช้ PowerShell Job/Process Object Cleanup ใน `start-windows.ps1` พร้อมระบบเช็กพอร์ตซ้ำก่อนรัน |
| **2. เปรียบเทียบ 2 ไฟล์ `.mlv` ที่มีจำนวนแถวไม่เท่ากัน** | ผู้ใช้เผลอเลือกไฟล์ `.mlv` ที่อัดมาจากคนละชุดข้อมูล (เช่น 500 แถว vs 1,200 แถว) | ตัดเทียบเฉพาะจำนวนเฟรมที่สั้นกว่า $N_{\min} = \min(N_A, N_B)$ และแสดงแบนเนอร์แจ้งเตือน `"Frame count mismatch: truncating comparison to first 500 frames"` |
| **3. การซิงค์มุมกล้อง 3D เกิด Infinite Loop** | กล้อง A ขยับ $\to$ สั่งกล้อง B ขยับ $\to$ กล้อง B ยิง Event กลับมาสั่งกล้อง A | ใช้ตัวแปร `activeControlSourceRef = useRef<'A' | 'B' | null>(null)` เพื่อให้อัปเดตเฉพาะฝั่งตรงข้ามเมื่อเกิด `onChange` จากเมาส์ของผู้ใช้จริงเท่านั้น |
| **4. ฟอนต์หรือไอคอนไม่แสดงเมื่อตัดอินเทอร์เน็ต 100%** | เผลอเรียกใช้ Google Fonts CDN หรือ External Icon URL | บันเดิลไฟล์ฟอนต์ `.woff2` (`Inter`, `JetBrains Mono`) และไอคอน SVG (`lucide-react`) ไว้ใน `frontend/public/fonts/` และตรวจเช็กด้วย Offline Network Audit Test |

---

## 8. เกณฑ์การตรวจรับงาน (Definition of Done Checklist)
- [ ] สคริปต์ `start-windows.bat`, `start-windows.ps1`, และ `start-unix.sh` สามารถเปิดระบบทั้ง Backend (`30211`) และ Frontend (`30222`) และปิดโปรเซสลูกทั้งหมดได้อย่างสะอาดเมื่อเลิกใช้งาน
- [ ] สคริปต์ `generate_all_20_demos.py` สร้างโมเดลตัวอย่างครบทั้ง 20+ ตระกูลและสามารถกดเปิดเล่นจากหน้า **Instant Demo Gallery** ได้ทันทีโดยไม่ต้องตั้งค่าใดๆ
- [ ] โหมด **Synchronized Dual `.mlv` Compare** สามารถเล่นเปรียบเทียบ 2 โมเดลพร้อมกัน ซิงค์ทั้งมุมกล้อง 3D และแถบเวลา VCR รวมถึงกระโดดไปยังเฟรมที่ทายขัดแย้งกันได้ถูกต้อง
- [ ] ผ่านชุดทดสอบ **100,000-Row Stress Benchmark** โดยรักษาความเร็วการเล่น VCR ที่ 60 FPS และใช้ RAM ไม่เกินเพดานที่กำหนด
- [ ] ผ่านการตรวจสอบ **100% Air-Gapped Offline Mode** (ไม่มี Network Request ใดๆ ออกนอก `localhost` / `127.0.0.1`)
