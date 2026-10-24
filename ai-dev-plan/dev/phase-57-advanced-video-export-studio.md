# Phase 57: Advanced Multi-Layout Video Export Studio (2D/3D Dual-Dimension, Why/What-If PiP, Independent Overrides & MP4/WebM)

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
ยกระดับขีดความสามารถของระบบส่งออกวิดีโอ (**Video Export Studio**) ใน Easy ML Viewer จากเดิมใน Phase 16 และ Phase 48 ที่จับภาพสดจากแคนวาสหน้าจอ (Screen Canvas Capture) ให้กลายเป็น **สตูดิโอเรนเดอร์วิดีโอระดับมืออาชีพแบบแยกอิสระ (Isolated Headless Video Rendering Studio)** โดยตอบโจทย์การนำเสนอผลงานวิจัย, การสาธิตโมเดลในการประชุมทางธุรกิจ, และการจัดทำสื่อการสอนด้วยความสามารถหลัก 5 ด้าน:

1. **อิสระในการเลือกมิติการแสดงผล (2D Planar vs 3D Spatial Dimension Toggle)**:
   - ผู้ใช้สามารถเลือกว่าจะเรนเดอร์วิดีโอเป็นมุมมอง **3D Spatial Mesh** หรือ **2D Planar Decision Map** ได้อย่างอิสระ โดยไม่ต้องเปลี่ยนโหมด Viewport ในหน้าจอหลัก
2. **การฝังพาเนลอธิบายผล (Embedded Why Panel & What-If Sandbox in Video)**:
   - เรนเดอร์ข้อมูลเชิงลึกควบคู่ไปกับพื้นผิวการตัดสินใจของโมเดลในเฟรมวิดีโอเดียวกัน:
     - **Why Panel**: กราฟแท่ง Waterfall แสดงค่าน้ำหนักอิทธิพลของฟีเจอร์ (SHAP / Local Feature Contributions) ของจุดข้อมูลที่กำลังเคลื่อนที่ในแต่ละเฟรม
     - **What-If Panel**: ป้ายแสดงสถานะการจำลองสมมติ, ค่าการกระจัด ($\Delta \mathbf{x}$), และผลต่างคำทำนายเทียบกับจุดอ้างอิงจริง
   - รองรับ 2 เลย์เอาต์: **Split Studio (จอแยก 70/30 ด้านขวา)** และ **Picture-in-Picture (PiP ลอยมุมขวาล่าง)**
3. **การตั้งค่าอิสระแยกจากหน้าจอพรีวิว (Independent Theme, Angle & Speed Overrides)**:
   - **Theme Override**: เลือกระบายสีเป็น **Dark Theme**, **Light Theme** (เหมาะสำหรับพิมพ์เอกสาร/สไลด์นำเสนอสีขาว), หรือ **Match UI** โดยไม่ส่งผลกระทบต่อธีมของสตูดิโอที่ผู้ใช้กำลังเปิดอยู่
   - **Camera Angle Override (สำหรับ 3D)**: เลือกมุมกล้องได้อิสระ (`Isometric`, `Top XY`, `Front XZ`, `Side YZ`, หรือ `Follow Ball`) แม้ในหน้าจอพรีวิวจะหมุนกล้องค้างไว้ที่มุมอื่น
   - **Playback Speed & Framerate Override**: กำหนดความเร็วในการเรนเดอร์ (`0.25x Slow-Mo`, `0.5x`, `1.0x`, `2.0x`, `4.0x`) และอัตราเฟรม (`30 FPS` หรือ `60 FPS`) โดยไม่อิงกับความเร็ว VCR บนหน้าจอ
4. **รองรับฟอร์แมตไฟล์สากล (Universal MP4 & WebM Support)**:
   - รองรับไฟล์ **`.mp4` (H.264 / AVC1 + AAC)** แท้ 100% Offline สำหรับเปิดบน PowerPoint, Keynote, QuickTime, iOS, และเบราว์เซอร์ทุกแพลตฟอร์ม
   - ยังคงรองรับไฟล์ **`.webm` (VP9 / Opus)** คุณภาพสูงขนาดกะทัดรัดเช่นเดิม
5. **การเรนเดอร์แบบเฟรมต่อเฟรมไร้การกระตุก (Deterministic Off-Screen Compositor)**:
   - ใช้วิธีกรอเฟรมทีละก้าวแบบ Deterministic Tick (`seekToFrame(i)`) ควบคู่กับ Virtual Canvas เพื่อรับประกันว่าวิดีโอที่ได้จะ **ไม่มีการ Drop Frame (Zero Frame Drop 60 FPS)** แม้เครื่องคอมพิวเตอร์จะประมวลผลโมเดลหนักก็ตาม

---

## 2. กลไกคณิตศาสตร์และสถาปัตยกรรมการเรนเดอร์ (Rendering Architecture & Mathematics)

### 2.1 เลย์เอาต์และเรขาคณิตการรวมภาพ 2D/3D (Multi-Layout Compositor)
ระบบสร้างแคนวาสเสมือน (**Virtual Composite Canvas**) ความละเอียด Full HD ($1920 \times 1080$), 4K ($3840 \times 2160$), หรือ Square ($1080 \times 1080$):

```text
+---------------------------------------------------------------------------------------------------+
| COMPOSITE CANVAS (1920 x 1080) — LAYOUT OPTION A: SPLIT STUDIO (70% STAGE / 30% WHY-IF PANEL)     |
+-------------------------------------------------------------------+-------------------------------+
| PRIMARY VISUALIZER (3D or 2D) (Width: 1344px, Height: 1080px)     | TELEMETRY DOCK (Width: 576px) |
|                                                                   |                               |
|   - 3D Surface Mesh or 2D Contour Decision Map                    | [WHY PANEL: LOCAL SHAP]       |
|   - Theme: Dark / Light Override                                  | Feature_1  █████████ +0.42    |
|   - Active Camera Angle (Isometric / Top / Front / Side)          | Feature_2  ████ -0.18         |
|   - Point Intelligence: Comet Trails & Dual-Rings                 | Feature_3  ██ +0.08           |
|                                                                   +-------------------------------+
|                                                                   | [WHAT-IF SANDBOX]             |
|                                                                   | Input Row: #42  | Pred: 0.89  |
|                                                                   | Counterfactual Flip: -12.4%   |
|                                                                   +-------------------------------+
|   [TIMELINE HUD] Frame 0048/0200 · 60.0 FPS · Speed: 1.0x         | [METRICS BADGE] Loss: 0.012   |
+-------------------------------------------------------------------+-------------------------------+
```

```text
+---------------------------------------------------------------------------------------------------+
| COMPOSITE CANVAS (1920 x 1080) — LAYOUT OPTION B: PICTURE-IN-PICTURE (PiP)                        |
+---------------------------------------------------------------------------------------------------+
| FULLSCREEN VISUALIZER (1920 x 1080)                                                               |
|                                                                                                   |
|                                                               +---------------------------------+ |
|                                                               | FROSTED GLASS PiP (520 x 360)   | |
|                                                               | Why: Feature_1 (+0.42)          | |
|                                                               | What-If: Flip Delta = -12.4%    | |
|                                                               | Current y_hat: 0.892 (Class 1)  | |
|                                                               +---------------------------------+ |
+---------------------------------------------------------------------------------------------------+
```

### 2.2 การจัดการชุดสี Dark / Light Mode Override เชิงโปรแกรม
เพื่อให้วิดีโอที่ส่งออกมีธีมตรงตามที่ผู้ใช้ต้องการ โดยไม่กระพริบหน้าจอหลัก ระบบจำลอง **Color Palette Token Matrix**:

$$\mathbf{C}_{\text{render}} = \begin{cases}
\mathbf{C}_{\text{dark}} & \text{เมื่อเลือก Dark Theme} \\
\mathbf{C}_{\text{light}} & \text{เมื่อเลือก Light Theme} \\
\mathbf{C}_{\text{current\_ui}} & \text{เมื่อเลือก Match UI}
\end{cases}$$

โดยตารางแปลงโทนสีสำหรับ Light Mode ประกอบด้วย:
- Background: `#f8fafc` (Slate-50)
- Grid & Wireframe: `#cbd5e1` (Slate-300)
- Surface Opacity: ปรับ Ambient Light เป็น $1.4\times$ และ Directional Light เป็น $1.2\times$
- Text & HUD: `#0f172a` (Slate-900) พร้อม Contrast Shadow

### 2.3 การส่งออกเป็นไฟล์ MP4 แท้ (Offline MP4 Muxer & Fallback Architecture)
1. **In-Browser WebCodecs + `mp4-muxer` (Primary Fast Path)**:
   - ใช้เบราว์เซอร์ API `VideoEncoder` (Codec: `avc1.640028` หรือ `avc1.4d002a`) แปลงภาพจาก Canvas ทีละเฟรมเป็น H.264 NAL units
   - แพ็กเกจลงโครงสร้างกล่อง MP4 (`ftyp`, `moov`, `mdat`) ผ่านโมดูล JavaScript น้ำหนักเบาในหน่วยความจำ ไม่ต้องพึ่ง External CLI
2. **Backend Native PyAV / FFmpeg Transcoder (Fallback Path)**:
   - หากเบราว์เซอร์รุ่นเก่าไม่รองรับ WebCodecs หรือการตั้งค่าความละเอียดสูงเกินขีดความสามารถ GPU:
   - ส่งสตรีมเฟรม WebM/Raw เข้าเอนด์พอยต์ `POST /api/v1/export/transcode-mp4` ของ FastAPI
   - ฝั่ง Backend ใช้ `av` (PyAV library ที่ผูกกับ FFmpeg 100% Offline) ทำการ Re-mux หรือ H.264 Fast Encode ได้ที่ความเร็ว $> 120$ FPS

---

## 3. สถาปัตยกรรมข้อมูลและ DTO Schemas (`frontend/` & `backend/`)

```typescript
// frontend/src/domains/export-studio/types/advancedVideoExport.types.ts

export type VideoDimension = '3d' | '2d';
export type VideoLayout = 'split_studio' | 'pip' | 'pure_stage';
export type VideoTheme = 'match_ui' | 'dark' | 'light';
export type VideoCameraPreset = 'isometric' | 'top_xy' | 'front_xz' | 'side_yz' | 'follow_ball';
export type VideoFormat = 'mp4' | 'webm';
export type VideoResolution = '1080p' | '720p' | 'square_1080';

export interface AdvancedVideoExportOptions {
  dimension: VideoDimension;
  layout: VideoLayout;
  includeWhyPanel: boolean;
  includeWhatIfPanel: boolean;
  theme: VideoTheme;
  cameraPreset: VideoCameraPreset;
  speedMultiplier: number; // 0.25, 0.5, 1.0, 2.0, 4.0
  fps: 30 | 60;
  format: VideoFormat;
  resolution: VideoResolution;
  bitrateMbps: number;
  rangeStartFrame: number;
  rangeEndFrame: number;
}
```

```python
# backend/app/presentation/routers/video_transcode_router.py
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/export", tags=["Export Studio"])

class TranscodeResponse(BaseModel):
    download_url: str
    duration_sec: float
    output_format: str
    file_size_bytes: int
```

---

## 4. UI Feature Matrix & หน้าจอ ExportStudioModal (`frontend/src/domains/export-studio/`)

```text
+--------------------------------------------------------------------------------------------------+
| EXPORT STUDIO — TAB 2: ADVANCED VIDEO CLIP (.MP4 / .WEBM)                                        |
+--------------------------------------------------------------------------------------------------+
| 1. Dimension & Layout:                                                                           |
|    Dimension:       [● 3D Spatial]  [○ 2D Planar]                                                |
|    Panel Layout:    [● Split Studio (70/30)]  [○ PiP Bottom-Right]  [○ Pure Stage (No Panels)]   |
|    Analytics Data:  [✓] Include Why Panel (SHAP Waterfall)   [✓] Include What-If Delta Badge     |
|                                                                                                  |
| 2. Visual Overrides (Independent from Preview):                                                  |
|    Theme Style:     [○ Match UI]  [● Dark Mode]  [○ Light Mode (Paper/Slides)]                   |
|    Camera Angle:    [● Isometric]  [○ Top XY]  [○ Front XZ]  [○ Side YZ]  [○ Follow Ball]        |
|    Playback Speed:  [○ 0.25x Slow]  [○ 0.5x]  [● 1.0x Normal]  [○ 2.0x Fast]  [○ 4.0x]           |
|                                                                                                  |
| 3. Output Format & Quality:                                                                      |
|    Format:          [● MP4 (H.264 Universal)]  [○ WebM (VP9)]                                    |
|    Resolution:      [● Full HD 1080p (1920x1080)]  [○ 720p]  [○ Square 1080x1080]               |
|    Frame Rate:      [○ 30 FPS]  [● 60 FPS (Ultra Smooth)]                                        |
|    Frame Range:     From Frame [ 0 ] to [ 200 ]  (Total 201 frames · Est. Duration: 6.7s)       |
+--------------------------------------------------------------------------------------------------+
| [ PROGRESS: Rendering Frame 48 / 201 (24%) · Encoding H.264 MP4 ]                                |
| [ CANCEL ]                                                             [ START OFFLINE RENDER ]  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. แผนการทำงานและสเต็ปการเรนเดอร์ (Execution Pipeline)

1. **Step 1: Setup Off-Screen Render Target**:
   - สร้างแคนวาสขนาด $1920 \times 1080$ ในหน่วยความจำ (Off-screen Canvas)
   - หากเป็นโหมด 3D: ผูกกับ Second Three.js WebGLRenderTarget ปรับมุมกล้องและแสงตาม `cameraPreset` และ `theme` ที่เลือก
   - หากเป็นโหมด 2D: ผูกกับ Second 2D Vector & Heatmap Context ปรับขนาดให้พอดีกับกรอบ 70%
2. **Step 2: Deterministic Frame Scrubbing Loop**:
   - วนลูป $i = \text{rangeStart} \dots \text{rangeEnd}$:
     - อัปเดตพิกัดลูกบอลและ Trajectory สำหรับเฟรม $i$
     - สกัดข้อมูล Explainability สำหรับเฟรม $i$ (Top-5 Feature Contributions)
     - สกัดข้อมูล What-If ถ้ามีการเปิดใช้งาน
     - วาด Primary Stage ลงบนฝั่งซ้าย (0px ถึง 1344px)
     - วาด Why Panel + What-If Dock ลงบนฝั่งขวา (1344px ถึง 1920px) พร้อมข้อมูลตัวเลขจริง
     - ดึง Image Frame ส่งเข้า Video Encoder
3. **Step 3: Muxing & Packaging**:
   - เมื่อครบทุกเฟรม แพ็กเกจข้อมูลเป็น Blob ไฟล์ `.mp4` หรือ `.webm`
   - เรียกฟังก์ชันดาวน์โหลดของเบราว์เซอร์ทันที 100% Offline

---

## 6. Definition of Done (DoD)
1. สามารถเลือกส่งออกวิดีโอเป็น **3D หรือ 2D** ได้อย่างอิสระโดยไม่กระทบ Viewport ปัจจุบัน
2. วิดีโอที่ได้มี **Why Panel** และ/หรือ **What-If Panel** ปรากฏร่วมอยู่ในเฟรมอย่างสวยงามคมชัด
3. สามารถเลือกธีม **Dark / Light**, มุมกล้อง, และความเร็วในการเล่นแยกจากหน้าจอสดได้สำเร็จ
4. ส่งออกไฟล์นามสกุล **`.mp4` (H.264)** ที่เปิดเล่นบน Windows Media Player, QuickTime, และ PowerPoint ได้อย่างราบรื่น
5. การเรนเดอร์วิดีโอ 200 เฟรม ที่ 1080p 60 FPS เสร็จสมบูรณ์ภายในเวลาไม่เกิน $15$ วินาที และไม่มีเฟรมตกหล่น
