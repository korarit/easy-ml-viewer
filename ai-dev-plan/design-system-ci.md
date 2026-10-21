# Easy ML Viewer — Corporate Identity (CI) & Visual Design System
> **Design Philosophy**: *"Precision Engineering Lab"* — สตูดิโอวิเคราะห์และควบคุมโมเดล Machine Learning ในรูปแบบเครื่องมือช่างระดับไฮเอนด์ (Tactile Studio Hardware) สะอาด สุขุม หนักแน่น และอ่านข้อมูลได้แม่นยำสูงทั้งใน **Dark Mode** และ **Light Mode**

---

## 1. ปรัชญาการออกแบบและแถลงการณ์ต่อต้าน "AI Cringe" (Design Manifesto & Anti-AI-Trope)

โปรเจกต์นี้ปฏิเสธรูปแบบดีไซน์แบบพิมพ์นิยมที่มักถูกสร้างโดย AI ทั่วไป (Anti-AI-Cringe):
- ❌ **ไม่ใช้แสงนีออนเรืองแสงพร่ำเพรื่อ**: ห้ามใส่ `box-shadow: 0 0 25px cyan/magenta` บนขอบปุ่มหรือแผงควบคุม เพราะทำให้ตาล้าและดูเหมือนเกมอาร์เคดราคาถูก
- ❌ **ไม่ใช้พื้นหลังดำสนิทแบบ OLED (`#000000`) ตัดกับตัวหนังสือสว่างจ้า**: เพราะทำให้เกิด Contrast Harshness ส่งผลให้อ่านตัวเลขตารางและกราฟต่อเนื่องเกิน 15 นาทีไม่ได้
- ❌ **ไม่ใส่องค์ประกอบไซไฟปลอม (Fake Sci-Fi Fluff)**: ไม่มีเส้น HUD หมุนๆ วงแหวนประดับที่ไม่มีข้อมูล หรือเอฟเฟกต์โฮโลแกรมลอยไปมาอย่างไร้ความหมาย

### ✨ แกนหลัก 4 ประการของ "Precision Engineering Lab"
1. **Functional Semantics First**: สีทุกสี กราฟิกทุกเส้น ต้องสื่อความหมายทางคณิตศาสตร์หรือสถิติเท่านั้น (เช่น สีส้ม = Anomaly, สีเขียว = Class 0, สีแดงอมชมพู = Class 1, สีม่วงลาเวนเดอร์ = สมมุติฐาน What-If)
2. **Tactile Hardware Feel (สัมผัสแผงเครื่องมือจริง)**: ได้แรงบันดาลใจจาก *Teenage Engineering*, *Dieter Rams (Braun)*, *Linear.app*, *Ableton Live*, และ *DaVinci Resolve* — ปุ่มกดมีมิติของแผ่นโลหะ/พลาสติกเนื้อด้าน (Matte Finish), สวิตช์และสไลเดอร์ตอบสนองแม่นยำ, ไทม์ไลน์ VCR มีขีดสเกลไม้บรรทัดวิศวกรรม
3. **High Ergonomic Density**: การจัดวางข้อมูลมีความกระชับ (Compact Spacing) ไม่เปลืองพื้นที่หน้าจอด้วย Padding ที่กว้างเกินความจำเป็น เพื่อให้วิศวกรและนักวิจัยเห็นทั้งโมเดล 3D, ตารางข้อมูล, และแถบ SHAP ได้พร้อมกันในหน้าจอเดียว
4. **100% Air-Gapped & Crisp Vector Precision**: ฟอนต์, ไอคอน SVG, และ 3D Shaders ทั้งหมดบันเดิลอยู่ในเครื่อง 100% เส้นแบ่งทุกเส้นคมชัดระดับ 1 Physical Pixel (`border border-slate-700/50` / `border-slate-200`)

---

## 2. ระบบตัวอักษรและตัวเลข (Typography System)

ระบบฟอนต์ถูกบันเดิลเป็นไฟล์ `.woff2` ในโฟลเดอร์ `frontend/public/fonts/` เพื่อการทำงานแบบออฟไลน์สมบูรณ์แบบ:

```text
frontend/public/fonts/
├── Inter-Regular.woff2
├── Inter-Medium.woff2
├── Inter-SemiBold.woff2
├── JetBrainsMono-Regular.woff2
└── JetBrainsMono-Medium.woff2
```

### การจัดสรรฟอนต์ตามบริบทการใช้งาน (Font Hierarchy)

| บริบท (Context) | ตระกูลฟอนต์ (Font Family) | ขนาด & น้ำหนัก | ฟังก์ชันการแสดงผล |
| :--- | :--- | :--- | :--- |
| **Headline / Section Titles** | `Inter` (sans-serif) | 14px / 16px (SemiBold 600) | หัวข้อโมดูล, ชื่อโมเดลที่กำลังรัน, ชื่อสเต็ปบน Navigation |
| **UI Body, Labels, Tooltips** | `Inter` (sans-serif) | 12px / 13px (Regular 400, Medium 500) | ข้อความอธิบาย, ปุ่มกด, เมนูตัวเลือก, แท็บ |
| **Telemetry, Coordinates, VCR Time** | `JetBrains Mono` (monospace) | 11px / 12px (Medium 500) | พิกัด $X, Y, Z$, เวลาของเฟรม `00:01:24.05`, ค่า Loss, RAM, Latency |
| **Feature Names & Equations** | `JetBrains Mono` (monospace) | 11px (Regular 400) | ชื่อคอลัมน์ CSV, สมการ Decision Tree Split, ค่า SHAP Value |

> [!TIP]
> ตัวเลขโมโนสเปซต้องเปิดใช้งาน CSS `font-feature-settings: 'tnum' 1, 'zero' 1;` เสมอ เพื่อให้ความกว้างของตัวเลขคงที่ (Tabular Numerals) ป้องกันไม่ให้ตัวเลขสั่นหรือขยับไปมาขณะลูกบอล 3D กำลังเคลื่อนที่ใน VCR

---

## 3. ตารางชุดสีมาตรฐาน (Design Tokens & Color Palette Matrix)

### 3.1 พื้นผิวหลัก (Surfaces & Structural Tokens)

```text
+----------------------------------------------------------------------------------------------------+
| SURFACE ELEVATION HIERARCHY                                                                        |
| Level 0: Viewport / Background Canvas (พื้นล่างสุด)                                                 |
| Level 1: Studio Sidebar & Bottom Deck (แถบควบคุมที่ติดขอบหน้าจอ)                                    |
| Level 2: Cards, Drawers, Virtualized Data Table (แผ่นการ์ดและตารางที่ยกตัวขึ้นมา)                   |
| Level 3: Modals, Floating 3D HUD, Tooltips (หน้าต่างลอยที่อยู่บนสุด)                               |
+----------------------------------------------------------------------------------------------------+
```

| Token Semantic Name | Dark Mode ("Slate Obsidian Lab") | Light Mode ("Engineering Paper") | คำอธิบายการนำไปใช้งาน |
| :--- | :--- | :--- | :--- |
| `--bg-canvas` | `#0B0F17` (Deep Slate Navy) | `#F8FAFC` (Clean Technical Paper) | พื้นหลังหลักของหน้าจอและฉาก 3D Viewport (ไม่แสบตา) |
| `--surface-panel` | `#111827` (Matte Obsidian) | `#FFFFFF` (Pure Card White) | พื้นหลังของ Sidebars, Top Bar, Bottom VCR Deck |
| `--surface-elevated` | `#1E293B` (Elevated Card) | `#F1F5F9` (Muted Warm Slate) | แผ่นการ์ดย่อย, กล่องพารามิเตอร์, เซลล์ตารางที่ถูก Hover |
| `--surface-overlay` | `rgba(15, 23, 42, 0.85)` | `rgba(255, 255, 255, 0.88)` | กระจก Frosted Glass ของป้าย 3D HUD (Backdrop Blur 12px) |
| `--border-subtle` | `#1E293B` (Slate Hairline 1px) | `#E2E8F0` (Crisp Light Gray) | เส้นแบ่งขอบแผงควบคุมและคอลัมน์ตาราง |
| `--border-strong` | `#334155` (Active Outline) | `#CBD5E1` (Defined Edge) | ขอบของการ์ดที่ถูกเลือก หรือเส้นตารางที่กำลัง Focus |
| `--text-primary` | `#F8FAFC` (Slate 50 — 96% Lum) | `#0F172A` (Slate 900 — Deep Ink) | ข้อความหลัก, หัวข้อ, ค่าตัวเลขสำคัญ |
| `--text-secondary` | `#94A3B8` (Slate 400) | `#475569` (Slate 600) | ข้อความคำอธิบาย, หน่วยนับ, สถิติประกอบ |
| `--text-muted` | `#64748B` (Slate 500) | `#94A3B8` (Slate 400) | ข้อความที่ไม่แอกทีฟ, Hotkey hints, สถานะ Locked |

---

### 3.2 สีแสดงข้อมูลและฟังก์ชันสถิติ (Data Semantics & Machine Learning Tokens)

สีเหล่านี้มี **นิยามทางคณิตศาสตร์และตรรกะที่ตรงกันทั้งใน Dark และ Light Mode** โดยปรับค่าความสว่าง (Luminance) ให้เหมาะสมกับพื้นหลัง:

| Semantic Token | Dark Mode Hex | Light Mode Hex | ความหมายทาง Machine Learning (Mathematical Meaning) |
| :--- | :--- | :--- | :--- |
| **`--ml-primary`** | `#3B82F6` (Precision Cobalt) | `#1D4ED8` (Deep Blue) | เส้นทางเดินของโมเดล (Trajectory Path), ปุ่ม Action หลัก, สเต็ปที่กำลังทำงาน |
| **`--ml-class-0`** | `#10B981` (Sage Emerald) | `#059669` (Forest Green) | คลาส 0, ค่าจริง (Ground Truth Pass), ฝั่งความน่าจะเป็น Negative/Safe |
| **`--ml-class-1`** | `#F43F5E` (Coral Crimson) | `#E11D48` (Rose Red) | คลาส 1, ค่าทายเป้าหมาย (Target Class), จุดที่ทายผิดพลาด (Residual Error) |
| **`--ml-anomaly`** | `#F59E0B` (Amber Gold) | `#D97706` (Warm Amber) | หมุด Anomaly บนไทม์ไลน์, จุดที่สองโมเดลทายขัดแย้งกัน (Conflict Frame) |
| **`--ml-whatif-ghost`** | `#8B5CF6` (Lavender Iris) | `#7C3AED` (Deep Violet) | ลูกบอลสมมุติ **Counterfactual Ghost Sphere**, เส้นความต่าง What-If Delta |
| **`--ml-wireframe-grid`** | `rgba(148, 163, 184, 0.18)` | `rgba(100, 116, 139, 0.22)` | เส้นตาข่ายพิกัดบนพื้นผิว 3D Decision Surface และพื้นห้องแล็บ |
| **`--ml-drop-line`** | `rgba(56, 189, 248, 0.45)` | `rgba(2, 132, 199, 0.50)` | เส้นประทิ้งดิ่งตั้งฉาก (Orthogonal Drop-line) ชี้ลงระนาบ $X\text{-}Y$ |

---

## 4. รายละเอียดส่วนประกอบ UI สำคัญ (Tactile Hardware Component Specs)

### 4.1 VCR Transport Control Deck (`VCRControlBar.tsx`)
แรงบันดาลใจจากแผงควบคุมเทปรีลสตูดิโอ (Reel-to-Reel Tape Deck):
- **ปุ่มหลัก**: `[<< Step Back]`, `[Play / Pause]`, `[Step Forward >>]`, `[Loop]`, `[Speed 0.5x/1x/2x]`
  - ตัวปุ่มใช้สไตล์สี่เหลี่ยมมุมมนเล็กน้อย (`rounded-md`), พื้นผิวสี Matte Slate
  - ขณะกด (`:active`) ยุบตัวลง 1px พร้อมเสียงคลิกสไตล์ไมโครสวิตช์ (Micro-switch tactile feel)
  - ห้ามใช้เอฟเฟกต์ไฟนีออนกะพริบ ใช้เพียงจุดไฟ LED สีฟ้าจุดเล็กๆ 4px (`bg-blue-500 rounded-full`) เมื่ออยู่ในสถานะ Playing
- **Scrubber Slider & Timecode Ruler**:
  - แถบไม้บรรทัดแบ่งสเกลเฟรมด้วยเส้น Ticks สั้น-ยาว (เหมือนสเกลวัดมุมในแบบร่างวิศวกรรม)
  - หัวเลื่อน Scrubber เป็นแท่งโลหะทรงลิ่มแนวตั้งสีขาว Off-white พร้อมเส้นเล็งสีฟ้า
  - ตัวเลขเฟรมและเปอร์เซ็นต์แสดงผลด้วย `JetBrains Mono`: `[ FRAME 0148 / 1250 | 11.8% | 60.0 FPS ]`

```text
+---------------------------------------------------------------------------------------------------------+
| [ |< ] [ < ] [ ▶ PLAY ] [ > ] [ >| ]  [ 🔁 Loop ]  [ 1.0x ▾ ] |  00:02:14.05  [ FRAME 0148 / 1250 ]     |
| +-----------------------------------------------------------------------------------------------------+ |
| |  |...|...|...|...|...|...▲...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|  | |
| |  [Conf Strip: ████████████░░░░░░░░░░████████████████]   📍 Anomaly #1 (#082)   📍 Conflict (#412)   | |
| +-----------------------------------------------------------------------------------------------------+ |
+---------------------------------------------------------------------------------------------------------+
```

### 4.2 3D Floating HUD & Measurement Reticle (`3DFloatingHUD.tsx`)
- ลอยอยู่เหนือลูกบอลแสงข้อมูล $24\text{ px}$ พร้อมเส้นลีดเดอร์ (Leader Line) ชี้ลงสู่จุดศูนย์กลางลูกบอล
- พื้นหลังเป็นแผ่นกระจกฝ้าสังเคราะห์ (Frosted Glass: `backdrop-blur-md bg-slate-900/80` ใน Dark Mode หรือ `bg-white/90` ใน Light Mode)
- แสดงสมการย่อยและการตัดสินใจเฉพาะหน้า:
  - บรรทัดที่ 1: ดัชนีแถวและชื่อโมเดล `Row #148 | RandomForestClassifier`
  - บรรทัดที่ 2: ค่าพิกัดฟีเจอร์ปัจจุบัน `[X: 4.82, Y: 1.35, Z: 0.91]`
  - บรรทัดที่ 3: ความน่าจะเป็นพร้อมแท่งแถบสีบาง 2px: `P(Class 1): 88.4% [████████░░]`

### 4.3 What-If Sandbox & Counterfactual Ghost Sphere
- เมื่อผู้ใช้ขยับสไลเดอร์ตัวแปรในแผง What-If:
  - **ลูกบอลจริง (Real Data Sphere)**: คงอยู่ที่ตำแหน่งเดิมบนพื้นผิว 3D เป็นทรงกลมสีทึบ
  - **ลูกบอลเงาสมมุติ (Ghost Sphere)**: สร้างเป็นทรงกลมโครงตาข่ายโปร่งแสงสีม่วงลาเวนเดอร์ (`--ml-whatif-ghost`) วิ่งแยกตัวออกไปตามค่าที่เปลี่ยนแปลง
  - มีเส้นประเชื่อมต่อระหว่างจุดจริงกับจุดสมมุติ พร้อมป้ายระยะห่างทางคณิตศาสตร์ (Euclidean / L1 Distance $\Delta d = 1.42$)

### 4.4 Data Table & Virtualized List (`DatasetTableView.tsx`)
- หัวตาราง (Table Header) ตรึงอยู่กับที่ (Sticky), ตัวอักษรตัวพิมพ์เล็กกึ่งหนา (`font-medium text-xs tracking-wider uppercase`)
- เส้นแบ่งเซลล์บาง 1px คมชัด
- สลับสีแถวคี่-คู่แบบบางเบา (`even:bg-slate-800/20` ใน Dark Mode หรือ `even:bg-slate-50` ใน Light Mode)
- เมื่อคลิกเลือกแถว มีเส้นกรอบเน้นสีน้ำเงิน Cobalt (`ring-1 ring-blue-500`) ชัดเจนโดยไม่เปลี่ยนสีพื้นหลังจนอ่านข้อความไม่ออก

---

## 5. การตั้งค่า CSS Variables สำหรับ Tailwind CSS v4 (`studio.css`)

ไฟล์นี้จะถูกนำไปติดตั้งที่ `frontend/src/core/theme/studio.css`:

```css
@import "tailwindcss";

@layer base {
  /* ---------------------------------------------------- */
  /* 🌙 DARK THEME (DEFAULT: SLATE OBSIDIAN LAB)           */
  /* ---------------------------------------------------- */
  :root, [data-theme="dark"] {
    --bg-canvas: #0B0F17;
    --surface-panel: #111827;
    --surface-elevated: #1E293B;
    --surface-overlay: rgba(15, 23, 42, 0.85);

    --border-subtle: #1E293B;
    --border-strong: #334155;

    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;

    --ml-primary: #3B82F6;
    --ml-class-0: #10B981;
    --ml-class-1: #F43F5E;
    --ml-anomaly: #F59E0B;
    --ml-whatif-ghost: #8B5CF6;
    --ml-wireframe-grid: rgba(148, 163, 184, 0.18);
    --ml-drop-line: rgba(56, 189, 248, 0.45);

    --stage-vignette-start: #0B0F17;
    --stage-vignette-end: #06090E;
    color-scheme: dark;
  }

  /* ---------------------------------------------------- */
  /* ☀️ LIGHT THEME (ENGINEERING PAPER & BLUEPRINT)        */
  /* ---------------------------------------------------- */
  [data-theme="light"] {
    --bg-canvas: #F8FAFC;
    --surface-panel: #FFFFFF;
    --surface-elevated: #F1F5F9;
    --surface-overlay: rgba(255, 255, 255, 0.88);

    --border-subtle: #E2E8F0;
    --border-strong: #CBD5E1;

    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;

    --ml-primary: #1D4ED8;
    --ml-class-0: #059669;
    --ml-class-1: #E11D48;
    --ml-anomaly: #D97706;
    --ml-whatif-ghost: #7C3AED;
    --ml-wireframe-grid: rgba(100, 116, 139, 0.22);
    --ml-drop-line: rgba(2, 132, 199, 0.50);

    --stage-vignette-start: #F8FAFC;
    --stage-vignette-end: #E2E8F0;
    color-scheme: light;
  }
}

/* Tabular Numerals สำหรับข้อมูลสถิติและตัวเลข 3D ทั้งหมด */
.font-mono-tabular {
  font-family: 'JetBrains Mono', monospace;
  font-feature-settings: 'tnum' 1, 'zero' 1;
}
```

---

## 6. กฎและข้อห้ามในการออกแบบ (Strict Design System Rules)

1. **ห้ามใช้ Border Radius ขนาดใหญ่แบบการ์ตูน**: ขอบปุ่มและการ์ดใช้สูงสุดไม่เกิน `rounded-md` ($6\text{ px}$) หรือ `rounded-lg` ($8\text{ px}$) ห้ามใช้ `rounded-3xl` หรือทรงแคปซูลขนาดใหญ่เกินความจำเป็น เพื่อคงความรู้สึกของเครื่องมือช่าง
2. **ห้ามใช้แอนิเมชันแบบเด้งดึ๋ง (No Bouncy Physics)**: Transition ของ UI ใช้ความเร็ว $100\text{ ms} - 150\text{ ms}$ แบบ `cubic-bezier(0.16, 1, 0.3, 1)` (Snappy & Mechanical) เพื่อให้แอปตอบสนองนิ้วมือทันที
3. **การสลับ Dark / Light Mode ใน 3D Canvas**:
   - เมื่อผู้ใช้สลับโหมด สีพื้นหลังของ Three.js Canvas, แสง Ambient Light (`THREE.AmbientLight`), และสีตารางพื้นห้อง (`GridHelper`) จะต้องเปลี่ยนค่าความสว่างตาม CSS Tokens ทันทีโดยไม่ต้องโหลดฉากใหม่
4. **ความคมชัดของตัวหนังสือต้องผ่านเกณฑ์ WCAG AAA**: อัตราส่วน Contrast Ratio ของข้อความหลักต้องไม่ต่ำกว่า $7:1$ บนทั้งสองโหมด เพื่อรองรับการทำงานต่อเนื่องของผู้ใช้โดยไม่เกิดอาการเมื่อยล้าสายตา

