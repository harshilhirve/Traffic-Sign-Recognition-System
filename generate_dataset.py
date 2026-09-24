"""
============================================================
  generate_dataset.py

  YOLO DATASET GENERATOR
  ──────────────────────
  Generates synthetic training images for all 20 traffic
  sign classes in YOLO format.

  YOLO label format (one .txt per image):
    <class_id> <x_center> <y_center> <width> <height>
  All values normalized 0-1 relative to image size.

  Output structure:
    dataset/
      images/train/   ← training images (.jpg)
      images/val/     ← validation images (.jpg)
      labels/train/   ← YOLO label files (.txt)
      labels/val/     ← YOLO label files (.txt)
      data.yaml       ← YOLO config file

  Run: python generate_dataset.py
============================================================
"""

import cv2
import numpy as np
import os
import math
import random
import shutil

# ── Configuration ─────────────────────────────────────────
IMG_W, IMG_H   = 640, 640      # YOLO standard input size
SAMPLES_TRAIN  = 300           # Training images per class
SAMPLES_VAL    = 60            # Validation images per class
SIGN_SIZE_MIN  = 100           # Min sign size in pixels
SIGN_SIZE_MAX  = 300           # Max sign size in pixels
OUTPUT_DIR     = "dataset"

# ── 20 Sign Classes ───────────────────────────────────────
CLASSES = [
    "STOP",
    "NO_ENTRY",
    "SPEED_LIMIT",
    "NO_OVERTAKING",
    "NO_PARKING",
    "ONE_WAY",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SCHOOL_AHEAD",
    "CURVE_AHEAD",
    "ROAD_WORK_AHEAD",
    "NARROW_ROAD",
    "ROUNDABOUT_AHEAD",
    "SPEED_BREAKER",
    "HOSPITAL",
    "PARKING",
    "PETROL_PUMP",
    "BUS_STOP",
    "REST_AREA",
    "TOILET",
]

# ── Sign visual definitions ────────────────────────────────
SIGN_DEFS = {
    "STOP":             {"shape":"octagon",   "bg":(30,30,210),   "border":(255,255,255), "text":"STOP",    "text_c":(255,255,255)},
    "NO_ENTRY":         {"shape":"circle",    "bg":(30,30,210),   "border":(255,255,255), "text":None,      "text_c":(255,255,255), "sym":"no_entry"},
    "SPEED_LIMIT":      {"shape":"circle",    "bg":(255,255,255), "border":(30,30,210),   "text":"60",      "text_c":(20,20,20)},
    "NO_OVERTAKING":    {"shape":"circle",    "bg":(255,255,255), "border":(30,30,210),   "text":None,      "text_c":(20,20,20),    "sym":"no_overtaking"},
    "NO_PARKING":       {"shape":"circle",    "bg":(255,255,255), "border":(30,30,210),   "text":"P",       "text_c":(30,30,210),   "sym":"cross"},
    "ONE_WAY":          {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"ONE WAY", "text_c":(255,255,255), "sym":"arrow_r"},
    "TURN_LEFT":        {"shape":"circle",    "bg":(210,30,30),   "border":(255,255,255), "text":None,      "text_c":(255,255,255), "sym":"arrow_l"},
    "TURN_RIGHT":       {"shape":"circle",    "bg":(210,30,30),   "border":(255,255,255), "text":None,      "text_c":(255,255,255), "sym":"arrow_r"},
    "SCHOOL_AHEAD":     {"shape":"triangle",  "bg":(0,210,210),   "border":(20,20,20),    "text":"SCHOOL",  "text_c":(20,20,20)},
    "CURVE_AHEAD":      {"shape":"triangle",  "bg":(0,210,210),   "border":(20,20,20),    "text":None,      "text_c":(20,20,20),    "sym":"curve"},
    "ROAD_WORK_AHEAD":  {"shape":"triangle",  "bg":(0,140,255),   "border":(20,20,20),    "text":None,      "text_c":(20,20,20),    "sym":"shovel"},
    "NARROW_ROAD":      {"shape":"triangle",  "bg":(0,210,210),   "border":(20,20,20),    "text":None,      "text_c":(20,20,20),    "sym":"narrow"},
    "ROUNDABOUT_AHEAD": {"shape":"circle",    "bg":(210,30,30),   "border":(255,255,255), "text":None,      "text_c":(255,255,255), "sym":"roundabout"},
    "SPEED_BREAKER":    {"shape":"triangle",  "bg":(0,210,210),   "border":(20,20,20),    "text":"BUMP",    "text_c":(20,20,20)},
    "HOSPITAL":         {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"H",       "text_c":(255,255,255)},
    "PARKING":          {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"P",       "text_c":(255,255,255)},
    "PETROL_PUMP":      {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"FUEL",    "text_c":(255,255,255)},
    "BUS_STOP":         {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"BUS",     "text_c":(255,255,255)},
    "REST_AREA":        {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"REST",    "text_c":(255,255,255)},
    "TOILET":           {"shape":"rectangle", "bg":(210,30,30),   "border":(255,255,255), "text":"WC",      "text_c":(255,255,255)},
}


# ─────────────────────────────────────────────────────────
#  BACKGROUND GENERATORS
# ─────────────────────────────────────────────────────────

def make_background():
    """Generate a random realistic background."""
    bg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    kind = random.choice(["road","sky","urban","field","night"])

    if kind == "road":
        bg[:] = [random.randint(50,80)]*3
        # Road markings
        for y in range(0, IMG_H, random.randint(40,80)):
            cv2.line(bg,(IMG_W//2,y),(IMG_W//2,y+30),(200,200,200),3)
        # Grass verges
        bg[:, :random.randint(20,100)] = [30, random.randint(80,140), 30]
        bg[:, IMG_W-random.randint(20,100):] = [30, random.randint(80,140), 30]

    elif kind == "sky":
        sky_c = [random.randint(120,200), random.randint(160,220), random.randint(200,255)]
        bg[:] = sky_c
        # Clouds
        for _ in range(random.randint(2,5)):
            cx,cy = random.randint(0,IMG_W), random.randint(0,IMG_H//3)
            cv2.ellipse(bg,(cx,cy),(random.randint(40,100),random.randint(15,35)),0,0,360,(230,235,240),-1)

    elif kind == "urban":
        bg[:] = [random.randint(60,120)]*3
        # Buildings
        for _ in range(random.randint(3,7)):
            bx = random.randint(0,IMG_W-80)
            bh = random.randint(100,400)
            bw = random.randint(50,120)
            c = random.randint(80,160)
            cv2.rectangle(bg,(bx,IMG_H-bh),(bx+bw,IMG_H),(c,c,c+10),-1)
            # Windows
            for wy in range(IMG_H-bh+10, IMG_H-10, 25):
                for wx in range(bx+5, bx+bw-5, 18):
                    cv2.rectangle(bg,(wx,wy),(wx+10,wy+15),(200,210,240),-1)

    elif kind == "field":
        bg[:IMG_H//2] = [random.randint(120,190), random.randint(160,220), random.randint(200,255)]
        bg[IMG_H//2:] = [30, random.randint(100,160), 30]

    elif kind == "night":
        bg[:] = [random.randint(5,30)]*3
        # Stars
        for _ in range(random.randint(50,150)):
            sx,sy = random.randint(0,IMG_W), random.randint(0,IMG_H//2)
            cv2.circle(bg,(sx,sy),1,(200,200,220),-1)
        bg[IMG_H//2:] = [random.randint(20,50)]*3

    # Add noise
    noise = np.random.randint(-15,15,(IMG_H,IMG_W,3),dtype=np.int16)
    bg = np.clip(bg.astype(np.int16)+noise,0,255).astype(np.uint8)
    return bg


# ─────────────────────────────────────────────────────────
#  SIGN RENDERER
# ─────────────────────────────────────────────────────────

def draw_sign(defn, size):
    """Draw a single traffic sign onto a transparent canvas."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size//2, size//2
    r = int(size * 0.42)
    bg   = tuple(int(c) for c in defn["bg"])
    bord = tuple(int(c) for c in defn["border"])
    bw   = max(3, size//16)

    shape = defn["shape"]
    if shape == "octagon":
        pts = np.array([[cx+int(r*math.cos(math.pi/8+i*math.pi/4)),
                         cy+int(r*math.sin(math.pi/8+i*math.pi/4))] for i in range(8)], np.int32)
        cv2.fillPoly(canvas,[pts],bg); cv2.polylines(canvas,[pts],True,bord,bw)
    elif shape == "circle":
        cv2.circle(canvas,(cx,cy),r,bg,-1); cv2.circle(canvas,(cx,cy),r,bord,bw)
    elif shape == "triangle":
        pts = np.array([[cx,cy-r],[cx-int(r*.87),cy+r//2],[cx+int(r*.87),cy+r//2]],np.int32)
        cv2.fillPoly(canvas,[pts],bg); cv2.polylines(canvas,[pts],True,bord,bw)
    elif shape == "rectangle":
        cv2.rectangle(canvas,(cx-r,cy-int(r*.6)),(cx+r,cy+int(r*.6)),bg,-1)
        cv2.rectangle(canvas,(cx-r,cy-int(r*.6)),(cx+r,cy+int(r*.6)),bord,bw)

    # Symbol
    sym = defn.get("sym")
    sc  = tuple(int(c) for c in defn["text_c"])
    lw  = max(2, size//18)
    if sym == "no_entry":
        bh = max(4,r//3)
        cv2.rectangle(canvas,(cx-r+14,cy-bh),(cx+r-14,cy+bh),sc,-1)
    elif sym == "arrow_r":
        pts = np.array([[cx-r//2,cy-r//5],[cx+r//6,cy-r//5],[cx+r//6,cy-r//2],
                        [cx+r//2,cy],[cx+r//6,cy+r//2],[cx+r//6,cy+r//5],[cx-r//2,cy+r//5]],np.int32)
        cv2.fillPoly(canvas,[pts],sc)
    elif sym == "arrow_l":
        pts = np.array([[cx+r//2,cy-r//5],[cx-r//6,cy-r//5],[cx-r//6,cy-r//2],
                        [cx-r//2,cy],[cx-r//6,cy+r//2],[cx-r//6,cy+r//5],[cx+r//2,cy+r//5]],np.int32)
        cv2.fillPoly(canvas,[pts],sc)
    elif sym == "cross":
        cv2.line(canvas,(cx-r+12,cy-r+12),(cx+r-12,cy+r-12),(30,30,200),lw)
        cv2.line(canvas,(cx+r-12,cy-r+12),(cx-r+12,cy+r-12),(30,30,200),lw)
    elif sym == "no_overtaking":
        cv2.circle(canvas,(cx-r//4,cy),r//3,(120,120,120),-1)
        cv2.circle(canvas,(cx+r//4,cy),r//3,(30,30,200),-1)
        cv2.line(canvas,(cx-r+10,cy-r+10),(cx+r-10,cy+r-10),(30,30,200),lw)
    elif sym == "roundabout":
        cv2.circle(canvas,(cx,cy),r//2,sc,lw)
        for a in [0,120,240]:
            rad=math.radians(a); ax=int(cx+(r//2)*math.cos(rad)); ay=int(cy+(r//2)*math.sin(rad))
            cv2.circle(canvas,(ax,ay),max(3,size//20),sc,-1)
    elif sym == "curve":
        pts=np.array([[cx-r//3,cy+r//3],[cx-r//3,cy-r//4],[cx,cy-r//2],[cx+r//3,cy-r//4],[cx+r//3,cy+r//3]],np.int32)
        cv2.polylines(canvas,[pts],False,sc,lw)
    elif sym == "shovel":
        cv2.line(canvas,(cx,cy-r//2),(cx,cy+r//2),sc,lw)
        cv2.ellipse(canvas,(cx,cy-r//4),(r//4,r//4),0,0,180,sc,lw)
    elif sym == "narrow":
        lx,rx=cx-r//2,cx+r//2
        cv2.line(canvas,(lx,cy-r//3),(lx,cy+r//3),sc,lw)
        cv2.line(canvas,(rx,cy-r//3),(rx,cy+r//3),sc,lw)
        cv2.arrowedLine(canvas,(lx-2,cy),(cx-6,cy),sc,lw,tipLength=0.4)
        cv2.arrowedLine(canvas,(rx+2,cy),(cx+6,cy),sc,lw,tipLength=0.4)

    # Text
    text = defn.get("text")
    if text:
        font=cv2.FONT_HERSHEY_SIMPLEX
        for fs in [1.2,0.9,0.7,0.5,0.38]:
            (tw,th),_=cv2.getTextSize(text,font,fs,2)
            if tw < r*1.7 and th < r*0.85: break
        tx,ty=cx-tw//2,cy+th//2
        cv2.putText(canvas,text,(tx+2,ty+2),font,fs,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(canvas,text,(tx,ty),font,fs,sc,2,cv2.LINE_AA)

    return canvas


def augment_sign(sign_img):
    """Apply random augmentations to a sign image."""
    out = sign_img.copy()
    # Brightness
    beta = random.randint(-60, 60)
    out = np.clip(out.astype(np.int16)+beta,0,255).astype(np.uint8)
    # Noise
    if random.random() > 0.4:
        n = np.random.normal(0, random.uniform(3,20), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16)+n,0,255).astype(np.uint8)
    # Blur
    if random.random() > 0.5:
        k = random.choice([3,5])
        out = cv2.GaussianBlur(out,(k,k),0)
    # Rotation
    if random.random() > 0.3:
        angle = random.uniform(-20,20)
        h,w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
        out = cv2.warpAffine(out,M,(w,h),borderMode=cv2.BORDER_REFLECT)
    return out


def make_sample(class_name, class_id):
    """
    Create one training sample:
    - Random background
    - Sign placed at random position and size
    - Returns (image, yolo_label_string)
    """
    bg = make_background()

    # Random sign size
    sign_size = random.randint(SIGN_SIZE_MIN, SIGN_SIZE_MAX)
    defn = SIGN_DEFS[class_name]
    sign = draw_sign(defn, sign_size)
    sign = augment_sign(sign)

    # Random position (keep sign fully inside image)
    max_x = IMG_W - sign_size
    max_y = IMG_H - sign_size
    if max_x <= 0 or max_y <= 0:
        sign_size = min(IMG_W, IMG_H) - 20
        sign = cv2.resize(sign, (sign_size, sign_size))
        max_x = IMG_W - sign_size
        max_y = IMG_H - sign_size

    px = random.randint(0, max(1, max_x))
    py = random.randint(0, max(1, max_y))

    # Paste sign onto background
    bg[py:py+sign_size, px:px+sign_size] = sign

    # YOLO label: normalized cx, cy, w, h
    cx = (px + sign_size/2) / IMG_W
    cy = (py + sign_size/2) / IMG_H
    nw = sign_size / IMG_W
    nh = sign_size / IMG_H
    label = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

    return bg, label


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

def generate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, OUTPUT_DIR)

    # Clean and recreate
    if os.path.exists(out):
        shutil.rmtree(out)
    for d in ["images/train","images/val","labels/train","labels/val"]:
        os.makedirs(os.path.join(out, d))

    total = len(CLASSES) * (SAMPLES_TRAIN + SAMPLES_VAL)
    print("="*60)
    print("  YOLO DATASET GENERATOR")
    print(f"  {len(CLASSES)} classes × {SAMPLES_TRAIN} train + {SAMPLES_VAL} val")
    print(f"  = {total} total images at {IMG_W}×{IMG_H}px")
    print("="*60)

    for class_id, class_name in enumerate(CLASSES):
        # Training samples
        for i in range(SAMPLES_TRAIN):
            img, label = make_sample(class_name, class_id)
            name = f"{class_name}_{i:04d}"
            cv2.imwrite(os.path.join(out,"images","train",f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(out,"labels","train",f"{name}.txt"),"w") as f:
                f.write(label+"\n")

        # Validation samples
        for i in range(SAMPLES_VAL):
            img, label = make_sample(class_name, class_id)
            name = f"{class_name}_val_{i:04d}"
            cv2.imwrite(os.path.join(out,"images","val",f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(out,"labels","val",f"{name}.txt"),"w") as f:
                f.write(label+"\n")

        print(f"  [{class_id+1:2d}/20] {class_name:25s}  train:{SAMPLES_TRAIN}  val:{SAMPLES_VAL}")

    # Write data.yaml — YOLO config file
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print()
    print(f"  data.yaml saved → {yaml_path}")
    print()
    print("  Next step:  python train.py")
    print("="*60)


if __name__ == "__main__":
    generate()
