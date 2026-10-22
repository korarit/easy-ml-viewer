# Phase 48: Dual-Mode Synchronized Compare (2D/3D), 2D .mlv Persistence, High-Res Vector/Video Export & Hardening

## 1. วัตถุประสงค์และขอบเขตของเฟส (Objective & Scope)
ปิดท้ายชุดพัฒนาโหมด 2 มิติ (Part 8: 2D Intuitive Studio & Dual-Mode Visualization Suite) ในเฟสที่ 48 โดยหลอมรวมระบบ 2D Planar Mode เข้ากับระบบหลักทั้งหมดของโปรเจกต์ ได้แก่:
1. **Synchronized Dual-Model Compare in 2D**: โหมดเปรียบเทียบ 2 โมเดลในมุมมอง 2 มิติ ทั้งแบบแบ่งจอซ้าย-ขวา (**Split-Screen 2D Sync**) และแบบซ้อนทับบนระนาบเดียวกัน (**Superimposed 2D Decision Boundaries**) พร้อมเน้นย้ำพื้นที่ที่สองโมเดลตัดสินใจขัดแย้งกัน (**2D Disagreement Area Highlighter**)
2. **2D Cache Integration in `.mlv` Recording Container**: บันทึกเส้นขอบเขต Boundary Polylines และ Heatmap 2 มิติลงในไฟล์ `.mlv` (MsgPack + Zstandard) เพื่อให้เปิดดูซ้ำและสลับโหมด 2D/3D ได้ทันทีโดยไม่ต้องคำนวณใหม่
3. **Publication-Quality 2D Vector (SVG) & High-Res Snapshot/Video Export**: ส่งออกแผนภาพ 2D Decision Boundary เป็นไฟล์ **Vector SVG** สำหรับงานวิจัย/รายงาน พร้อมส่งออกภาพ 4K PNG และคลิปวิดีโอ 2D WebM/MP4
4. **100,000-Row Stress Benchmark & 100% Offline Audit for 2D Mode**: ตรวจสอบความทนทานของโหมด 2D ด้วยชุดข้อมูล 100,000 แถว รับประกันความเร็ว 60 FPS และการทำงานแบบ Offline 100%

---

## 2. กลไกการทำงานเชิงลึกและคณิตศาสตร์/โครงสร้างภายใน (Deep Technical & Algorithmic Mechanism)

### 2.1 คณิตศาสตร์การเปรียบเทียบ 2 โมเดลในระนาบ 2 มิติ (2D Dual-Model Comparison)
เมื่อผู้ใช้เปรียบเทียบโมเดล $A$ (เช่น `RandomForest`) และโมเดล $B$ (เช่น `SVM-RBF`) บนระนาบฟีเจอร์เดียวกัน $(X_1, X_2)$:
1. **การซ้อนทับเส้นขอบเขตการตัดสินใจ (Superimposed Decision Boundaries)**:
   - เส้นแบ่งเขตของโมเดล $A$ วาดด้วยสีน้ำเงินคราม (`#38bdf8`) และโมเดล $B$ วาดด้วยสีส้มอำพัน (`#fb923c`)
   - ทำให้เห็นความแตกต่างของพฤติกรรมการตัดสินใจได้อย่างชัดเจน เช่น โมเดล $A$ ตัดเป็นขั้นบันไดเหลี่ยมตามฟีเจอร์ ขณะที่โมเดล $B$ เป็นเส้นโค้งเรียบ
2. **พื้นที่ความขัดแย้ง 2 มิติ (2D Disagreement Area Mask $\mathcal{R}_{\text{conflict}}$)**:
   - สำหรับแต่ละพิกเซลบนกริดประเมิน $(x_i, y_j)$:
     $$\text{Conflict}(x_i, y_j) = \mathbb{I}\left(\hat{y}_A(x_i, y_j) \ne \hat{y}_B(x_i, y_j)\right)$$
   - บริเวณที่ $\text{Conflict} = 1$ จะถูกระบายด้วยลวดลายแถบเฉียงสีม่วงแดง (**Magenta Diagonal Hatch Pattern**) พร้อมแสดงสัดส่วนพื้นที่ขัดแย้งเชิงเปอร์เซ็นต์:
     $$\text{Disagreement Area \%} = \frac{\sum_{i,j} \text{Conflict}(x_i, y_j)}{N_x \times N_y} \times 100\%$$
3. **Synchronized Dual 2D VCR Playback**:
   - แถบ Scrubber ควบคุมเป้าเล็งวงแหวน 2 มิติของทั้งสองโมเดลพร้อมกัน หากเฟรมใดที่โมเดล $A$ และ $B$ ทายผลไม่ตรงกัน บนไทม์ไลน์จะปักหมุดสีม่วง `Disagreement Pin` และในจอ 2D เป้าเล็งจะสว่างวาบขึ้น

### 2.2 โครงสร้างส่วนขยายของไฟล์ `.mlv` (2D Boundary Payload Extension)
ในคอนเทนเนอร์ `.mlv` (Phase 15) จะเพิ่มบล็อกข้อมูล `contour_2d_cache`:
```python
# โครงสร้างภายใน .mlv MsgPack Payload
mlv_payload = {
    "version": "2.1.0",
    "metadata": { ... },
    "surface_3d": { ... },
    "contour_2d_cache": {
        "x_feature": "income",
        "y_feature": "debt_ratio",
        "grid_resolution": 120,
        "heatmap_float32": b"...",      # Zstandard-compressed float32 bytes
        "boundary_polylines": [         # รายการเส้นแบ่งเขตแบบเวกเตอร์
            {"class_id": 1, "points": [[-2.1, 0.4], [-1.9, 0.42], ...]}
        ],
        "model_specific_2d": { ... }    # ข้อมูลเฉพาะ เช่น Tree Partitions หรือ SVM SV Indices
    },
    "frames": [ ... ]
}
```
ผลลัพธ์: ขนาดไฟล์ `.mlv` เพิ่มขึ้นเพียง **$\sim 45\text{ KB}$** แต่ผู้ใช้สามารถเปิดไฟล์ขึ้นมาแล้วกดสลับ 2D $\leftrightarrow$ 3D ได้ทันทีใน 0.05 วินาทีโดยไม่ต้องรอคำนวณใหม่!

### 2.3 การส่งออกเวกเตอร์ SVG ความละเอียดสูง (SVG Export Pipeline)
ระบบสร้างไฟล์ SVG คุณภาพสูงสำหรับการพิมพ์หรือใส่ในบทความวิชาการ โดยประกอบด้วย:
- `<svg viewBox="0 0 1200 900">`
- เลเยอร์พื้นหลัง Heatmap แปลงเป็น Base64 PNG แบบฝังใน `<image>`
- เลเยอร์เส้น Decision Boundary วาดด้วย `<path d="M... L..." stroke="white" stroke-width="2.5" />`
- เลเยอร์จุดข้อมูล Trajectory Points วาดด้วย `<circle cx="..." cy="..." r="4" fill="..." />`
- ป้ายชื่อแกนและสเกลตัวเลขตัวอักษรคมชัดด้วย `<text font-family="Inter">`

---

## 3. UI Feature List อย่างละเอียด (Detailed UI Feature List)

### 3.1 Dual 2D Compare Studio
- **Layout Switcher**:
  - `[Side-by-Side 2D]`: จอคู่ซ้าย (Model A) และขวา (Model B) ซิงค์การแพนและซูม
  - `[Superimposed 2D Overlay]`: รวมทั้งสองโมเดลลงบนจอเดียวเพื่อดูเส้นตัดกันตรงๆ
- **Disagreement Overlay Toggle**:
  - ปุ่ม `[x] Highlight Conflict Zones (Striped Magenta)` พร้อมตัวเลขระบุ เช่น `14.8% Disagreement Area`
  - ปุ่มลัด `[Next Conflict (Alt+D)]` กรอ VCR ไปยังเฟรมที่โมเดลทั้งสองทายต่างกัน

### 3.2 2D Export Studio Controls
- **Export Formats Dropdown**:
  - `Vector SVG (.svg)`: เส้นคมชัด ซูมได้ไม่แตก พร้อม Legend และป้ายแกน
  - `High-Res Image (.png)`: เลือกความละเอียดได้ (`1x 1080p`, `2x 2K`, `4x 4K`)
  - `Video (.webm / .mp4)`: บันทึกแอนิเมชันลูกบอลวิ่งบนแผนที่ 2D ตามช่วงเวลาที่กำหนด
- **Watermark & Appearance Options**:
  - ตัวเลือกสไตล์ CI: `Precision Dark Mode` หรือ `Paper Clean Light Mode`

---

## 4. UX Flow & ASCII Wireframe ของหน้าจอ

```text
+--------------------------------------------------------------------------------------------------+
| DUAL 2D COMPARE | Model A: RandomForest (Blue) vs Model B: SVM-RBF (Orange)   [Offline 100%]     |
+--------------------------------------------------------------------------------------------------+
| Y: Debt Ratio                                                  CONFLICT AREA: 12.4% (Alt+D: Next)|
|   0.90 |                                                                                         |
|        |           / / / / / / / / / / / / / / / / / / / / / / / (Conflict Zone: A=Approve, B=Deny)
|   0.75 |          / / / / / / / / / / / / / / / / / / / / / / /                                  |
|        |   +-----------------------+   <- Model A (RandomForest Stepped Boundary)                |
|   0.60 |   |                       |                                                             |
|        |   |   (................)  |   <- Model B (SVM Curved Boundary)                          |
|   0.45 |   |  (                  ) |                                                             |
|        |   +-+                    +------+                                                       |
|   0.30 |     |                           |       ● Frame #0084 (DISAGREEMENT: A=0, B=1)          |
|   0.15 |     |                           |                                                       |
|   0.05 +-----+---------------------------+-------------------------> X: Income                  |
|                                                                                                  |
| [|<] [<<] [ ▶ PLAY ] [>>] [>|]  Timeline: ──●────⚡───[Conflict #3]────●─────   [Export SVG]      |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. สถาปัตยกรรม DDD & โค้ดต้นแบบ (DDD Code Prototypes)

### 5.1 โครงสร้างไฟล์ในระบบ
```text
backend/app/
├── domains/recording/
│   ├── contour_2d_baker.py          # เข้ารหัส 2D Contour ลงในไฟล์ .mlv
│   └── svg_export_generator.py      # สร้างไฟล์ SVG Vector คุณภาพสูง
frontend/src/domains/
├── recording/
│   └── components/DualCompare2DCanvas.tsx # ฉากเปรียบเทียบ 2 โมเดลในโหมด 2D
└── export-studio/
    └── components/SvgExportModal.tsx      # หน้าต่างดาวน์โหลด SVG
```

### 5.2 โค้ดต้นแบบ: `svg_export_generator.py`
```python
from typing import List, Dict, Any

class SvgExportGenerator:
    """สร้างไฟล์ SVG สำหรับแผนภาพ 2D Decision Boundary แบบ Standalone 100%"""

    @staticmethod
    def generate_svg(
        x_label: str,
        y_label: str,
        boundary_segments: List[List[float]],
        data_points: List[Dict[str, Any]],
        width: int = 1200,
        height: int = 900
    ) -> str:
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#0b0f14"/>',
            # วาดกริดและไม้บรรทัด
            '<g id="grid" stroke="#1e293b" stroke-width="1">',
            '</g>',
            # วาดเส้นแบ่งเขต Decision Boundary
            '<g id="decision-boundaries" stroke="#ffffff" stroke-width="2.5" fill="none">',
        ]
        for seg in boundary_segments:
            x1, y1, x2, y2 = seg
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        svg_parts.append('</g>')

        # วาดจุดข้อมูล Trajectory Points
        svg_parts.append('<g id="trajectory-points">')
        for pt in data_points:
            color = "#10b981" if pt["actual"] == 0 else "#f43f5e"
            svg_parts.append(f'<circle cx="{pt["x"]}" cy="{pt["y"]}" r="4" fill="{color}" opacity="0.85"/>')
        svg_parts.append('</g>')

        # ป้ายชื่อแกน
        svg_parts.append(f'<text x="{width/2}" y="{height - 20}" fill="#94a3b8" font-family="Inter" font-size="14" text-anchor="middle">{x_label}</text>')
        svg_parts.append(f'<text x="30" y="{height/2}" fill="#94a3b8" font-family="Inter" font-size="14" text-anchor="middle" transform="rotate(-90, 30, {height/2})">{y_label}</text>')
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
```

---

## 6. เกณฑ์การตรวจรับคุณภาพเฉพาะเฟส (Definition of Done & Quality Gates)
- [ ] โหมด Dual 2D Compare สามารถแสดงผลเส้นแบ่งเขตของ 2 โมเดลพร้อมกันได้อย่างถูกต้อง และระบายไฮไลต์พื้นที่ขัดแย้ง (Conflict Area) ได้แม่นยำ
- [ ] ไฟล์ `.mlv` ที่บันทึกจากระบบสามารถจัดเก็บข้อมูล 2D Contour ได้อย่างสมบูรณ์ และเมื่อเปิดไฟล์ขึ้นมาใหม่สามารถสลับดู 2D ได้ในเวลา $< 0.1$ วินาที
- [ ] การส่งออกไฟล์ Vector SVG สร้างภาพที่มีโครงสร้างสมบูรณ์ เปิดดูได้ในทุกเบราว์เซอร์หรือโปรแกรมกราฟิก (เช่น Illustrator, Inkscape) โดยไม่ต้องต่อเน็ต
- [ ] ผ่านการทดสอบ Stress Test 100,000 แถวในโหมด 2D โดยอัตราเฟรมไม่ตกต่ำกว่า 60 FPS
- [ ] ยืนยันการทำงานแบบ 100% Offline บน Windows, macOS, และ Linux

