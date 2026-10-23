# Phase 56: 2D Point Intelligence Unified Controls, .mlv Multi-Track Recording & Hardening

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
บูรณาการระบบ **2D Point Intelligence (A, B, C, D)** ทั้งหมดจาก Phase 49 ถึง Phase 55 เข้าเป็นส่วนหนึ่งของระบบสตูดิโอหลักอย่างสมบูรณ์แบบ ได้แก่:
1. **Unified Point Intelligence Toolbar & Global Hotkeys**: แผงควบคุมคีย์ลัดสากลบนระนาบ 2 มิติ (`T` สำหรับ Trails, `R` สำหรับ Dual-Rings, `E` สำหรับ Error Lens, `H` สำหรับ Hexbin Grid) พร้อมเมนู Quick Settings สำหรับปรับขนาดหางและโหมดรังผึ้ง
2. **.mlv Multi-Track Binary Serialization**: ขยายสเปกไฟล์ `.mlv` (Machine Learning Video) เพื่อจัดเก็บแทร็กข้อมูล Point Intelligence (Hexbin Tessellation Cache, Error Indices, Comet Trail Trajectory) ด้วย MsgPack + Zstandard ทำให้เปิดไฟล์ย้อนหลังได้ทันที 100% Offline โดยไม่ต้องคำนวณใหม่
3. **Vector SVG & 60 FPS Video Export Integration**: ส่งออกภาพเวกเตอร์ความละเอียดสูง SVG และบันทึกวิดีโอ WebM 60 FPS ที่บันทึกการเคลื่อนที่ของหาง Comet Trail และการขยับของวงแหวน Dual-Ring ได้อย่างคมชัดระดับพิกเซล
4. **End-to-End Hardening & Performance Gate**: ทดสอบความทนทานภายใต้ข้อมูลขนาดใหญ่ ($N = 10,000$ แถว) ป้องกัน Memory Leak 100% และรับประกันความเร็วคงที่ 60 FPS บนทุกเบราว์เซอร์

---

## 2. โครงสร้างไฟล์และสเปกการจัดเก็บ .mlv v2 (`backend/app/core/binary_codec.py`)

### 2.1 โครงสร้างแทร็กข้อมูล Point Intelligence ในไฟล์ `.mlv`
```text
MLV Container v2.0 (MsgPack + Zstd)
├── header (magic="MLV2", version=2, frame_count=T, model_family="...")
├── tracks/
│   ├── surface_3d_track
│   ├── contour_2d_track
│   └── point_intelligence_2d_track/
│       ├── flags: { trails: bool, dual_ring: bool, error_lens: bool, hexbin: bool }
│       ├── hexbin_grid: { radius: float, cells: [ { cx, cy, count, error_rate, dom_class } ] }
│       ├── density_contours: [ { threshold, segments } ]
│       └── frame_point_states: [ [ { x, y, actual, pred, residual, is_error } ] ]
```

### 2.2 ประสิทธิภาพการบีบอัด (Compression Benchmark Target)
- ขนาดของแทร็ก Point Intelligence สำหรับข้อมูล 1,000 แถว $\times$ 100 เฟรม ต้องมีขนาดหลังบีบอัดด้วย Zstandard $< 150$ KB
- เวลาในการถอดรหัส (Decode Time) ในฝั่ง Frontend ต้อง $< 10$ ms

---

## 3. UI Feature List & คีย์ลัดสากล (`frontend/src/domains/stage3d/`)

```text
+--------------------------------------------------------------------------------------------------+
| 2D POINT INTELLIGENCE SHORTCUTS & CONTROLS MATRIX                                                |
+--------------------------+-----------------------+-----------------------------------------------+
| Feature                  | Hotkey                | พฤติกรรมการทำงาน                              |
+--------------------------+-----------------------+-----------------------------------------------+
| Toggle Comet Trails      | T                     | เปิด/ปิดหางดาวตก Comet Trail และ Ghost Dots   |
| Cycle Trail Length       | Shift + T             | สลับความยาวหาง: 5 เฟรม -> 10 -> 20 -> All     |
| Toggle Dual-Ring Glyphs  | R                     | สลับระหว่างจุดปกติ กับ วงแหวนคู่ Dual-Ring    |
| Toggle Error Lens        | E                     | กรองแสดงเฉพาะจุดที่โมเดลทายผิด (Errors Only)  |
| Toggle Hexbin Density    | H                     | เปิด/ปิดแผนที่รังผึ้ง 6 เหลี่ยม (Hexbin Grid) |
| Cycle Hexbin Metric      | Shift + H             | สลับโหมดสีรังผึ้ง: Count -> Error % -> Class  |
+--------------------------+-----------------------+-----------------------------------------------+
```

---

## 4. UX Flow & ASCII Wireframe ของแผงควบคุมหลัก

```text
+--------------------------------------------------------------------------------------------------+
| [2D PLANAR STUDIO]  X: Annual_Income  |  Y: Credit_Score           [Export SVG] [Record WebM]    |
| +----------------------------------------------------------------------------------------------+ |
| | [● Trails (T: 10f)]  [● Dual-Ring (R)]  [○ Errors Only (E)]  [● Hexbin: Error% (Shift+H)]     | |
| +----------------------------------------------------------------------------------------------+ |
|                                                                                                  |
|   1.0 |    ⬡ [0% Error]                                                                          |
|       |         \                                                                                |
|   0.8 |          ⬡⬡⬡ [35% Error Zone]                                                           |
|       |            \                                                                             |
|   0.6 |             \  (Decision Boundary Polyline)                                              |
|       |              \                                                                           |
|   0.4 |               \     ~~~~~~~●-●-◉ [Active Frame #48]                                      |
|       |                \                                                                         |
|   0.2 |                 \          ⬡ [100% Error Hotspot: Bright Red Hexagon]                    |
|       |                  \                                                                       |
|   0.0 +----------------------------------------------------------------------------------------+ |
|       0.0             0.2             0.4             0.6             0.8            1.0         |
| [◄◄] [► Play] [►►] [=====●==================================================] Frame 48/200 60FPS|
+--------------------------------------------------------------------------------------------------+
```

---

## 5. การส่งออกไฟล์เวกเตอร์ SVG และวิดีโอ WebM (Export Suite Integration)
1. **SVG Vector Exporter (`frontend/src/domains/export-studio/components/svgBuilder.ts`)**:
   - เรนเดอร์เลเยอร์ Hexbin Grid, เส้นขอบเขต Boundary, และจุด Dual-Ring เป็นเวกเตอร์แท้ 100%
   - สามารถนำไฟล์ SVG ไปเปิดใน Adobe Illustrator, Figma หรือพิมพ์ลงรายงานวิจัยได้ที่ความคมชัดไม่สิ้นสุด
2. **Offline 60 FPS WebM Recorder (`Stage3DCaptureRegistry`)**:
   - บันทึกการเล่นย้อนหลังของ VCR ควบคู่กับหาง Comet Trail ที่เคลื่อนไหวตามเวลาแบบ Lossless Frame-by-Frame

---

## 6. Definition of Done (DoD)
1. คีย์ลัดสากล `T`, `R`, `E`, `H` ทำงานได้อย่างสมบูรณ์แบบโดยไม่ขัดแย้งกับคีย์ลัดอื่นในระบบ
2. ไฟล์ `.mlv` สามารถบันทึกและโหลดแทร็ก Point Intelligence ย้อนหลังได้โดยผลลัพธ์ตรงกัน 100%
3. ส่งออกภาพเวกเตอร์ SVG ได้ถูกต้องครบถ้วนทุกเลเยอร์
4. ผ่าน Stress Test ข้อมูล $10,000$ แถว โดยไม่มีหน่วยความจำรั่วไหล (Zero Memory Leaks)

