# Traffic Sign Recognition — YOLOv8

### Real-time detection using deep learning | Python + YOLOv8 + OpenCV

\---

## Setup \& Run (4 commands)

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Generate training dataset (6,000 images total)
python generate\_dataset.py

# 3. Train YOLOv8 on your dataset (\\\~30 min CPU, \\\~5 min GPU)
python train.py

# 4. Run live detection from webcam
python detect.py
```

\---

## How YOLOv8 Works (for your mentor)

```
Camera Frame (640×640)
       │
       ▼
  YOLOv8 Backbone (CSPDarknet)
  ─────────────────────────────
  Extracts feature maps at 3 scales:
    • 80×80  grid → small objects
    • 40×40  grid → medium objects
    • 20×20  grid → large objects
       │
       ▼
  YOLOv8 Head (Detection)
  ────────────────────────
  Each grid cell predicts:
    • Is there an object here?       (objectness)
    • What class is it?              (20 sign classes)
    • Where exactly is it?           (x, y, w, h)
    • How confident?                 (0–1)
       │
       ▼
  NMS — Non-Maximum Suppression
  ──────────────────────────────
  Removes duplicate overlapping boxes,
  keeps only the best detection per object
       │
       ▼
  Final detections drawn on frame
```

## Transfer Learning (why this is accurate)

YOLOv8s is pretrained on COCO dataset (330,000 images, 80 object classes).
It already knows how to detect edges, shapes, textures, and objects.

We **fine-tune** it on our 6,000 synthetic traffic sign images.
The model adapts its knowledge to recognize our specific 20 signs.
This is why training takes minutes instead of days.

\---

## Project Structure

```
traffic\\\_sign\\\_yolo/
├── generate\\\_dataset.py    ← Creates 6,000 training images
├── train.py               ← Fine-tunes YOLOv8 on your dataset
├── detect.py              ← Live webcam detection
├── requirements.txt
│
├── dataset/               ← Created by generate\\\_dataset.py
│   ├── images/train/      ← 6,000 training images
│   ├── images/val/        ← 1,200 validation images
│   ├── labels/train/      ← YOLO format labels (.txt)
│   ├── labels/val/
│   └── data.yaml          ← YOLO config
│
└── runs/traffic\\\_signs/    ← Created by train.py
    └── weights/
        ├── best.pt        ← Best model (use this)
        └── last.pt        ← Final epoch model
```

\---

## 20 Sign Classes

STOP, NO\_ENTRY, SPEED\_LIMIT, NO\_OVERTAKING, NO\_PARKING,
ONE\_WAY, TURN\_LEFT, TURN\_RIGHT, SCHOOL\_AHEAD, CURVE\_AHEAD,
ROAD\_WORK\_AHEAD, NARROW\_ROAD, ROUNDABOUT\_AHEAD, SPEED\_BREAKER,
HOSPITAL, PARKING, PETROL\_PUMP, BUS\_STOP, REST\_AREA, TOILET

\---

## Controls (in detection window)

|Key|Action|
|-|-|
|`Q` / `ESC`|Quit|
|`S`|Save screenshot|
|`+`|Increase confidence threshold|
|`-`|Decrease confidence threshold|

\---

## Improving Accuracy

**More epochs** — edit `train.py`, change `epochs=50` to `epochs=100`

**More training data** — edit `generate\\\_dataset.py`:

```python
SAMPLES\\\_TRAIN = 500   # was 300
SAMPLES\\\_VAL   = 100   # was 60
```

**Use a GPU** — if you have an NVIDIA GPU, edit `train.py`:

```python
device = 0    # instead of "cpu"
```

GPU training is 10-20x faster.

**Use a larger model** — edit `train.py`:

```python
model = YOLO("yolov8m.pt")   # medium  — more accurate, slower
model = YOLO("yolov8l.pt")   # large   — most accurate, needs GPU
```

