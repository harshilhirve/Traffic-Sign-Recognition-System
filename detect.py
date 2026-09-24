"""
============================================================
  detect.py

  REAL-TIME TRAFFIC SIGN DETECTOR
  ────────────────────────────────
  Uses your trained YOLOv8 model to detect traffic signs
  live from your laptop webcam.

  Hold a sign image on your phone in front of the camera.
  Detections appear as colored boxes with labels.

  Controls:
    Q / ESC  — Quit
    S        — Save screenshot
    +/-      — Increase/decrease confidence threshold

  Run: python detect.py
============================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import glob

# ── Sign info: category, action, alert for each class ─────
SIGN_INFO = {
    "STOP":             {"cat":"Mandatory",     "action":"Stop completely",             "critical":True,  "alert":"STOP! Come to a complete halt!"},
    "NO_ENTRY":         {"cat":"Prohibitory",   "action":"Do not enter",                "critical":True,  "alert":"WARNING: No Entry — wrong direction!"},
    "SPEED_LIMIT":      {"cat":"Regulatory",    "action":"Do not exceed speed limit",   "critical":False, "alert":""},
    "NO_OVERTAKING":    {"cat":"Prohibitory",   "action":"Do not overtake",             "critical":False, "alert":""},
    "NO_PARKING":       {"cat":"Prohibitory",   "action":"Parking not permitted",       "critical":False, "alert":""},
    "ONE_WAY":          {"cat":"Mandatory",     "action":"Drive in arrow direction only","critical":False, "alert":""},
    "TURN_LEFT":        {"cat":"Mandatory",     "action":"Turn left ahead",             "critical":False, "alert":""},
    "TURN_RIGHT":       {"cat":"Mandatory",     "action":"Turn right ahead",            "critical":False, "alert":""},
    "SCHOOL_AHEAD":     {"cat":"Warning",       "action":"Slow down — school zone",     "critical":True,  "alert":"CAUTION: School Zone — watch for children!"},
    "CURVE_AHEAD":      {"cat":"Warning",       "action":"Reduce speed — sharp curve",  "critical":False, "alert":""},
    "ROAD_WORK_AHEAD":  {"cat":"Warning",       "action":"Slow down — road works",      "critical":False, "alert":""},
    "NARROW_ROAD":      {"cat":"Warning",       "action":"Road narrows ahead",          "critical":False, "alert":""},
    "ROUNDABOUT_AHEAD": {"cat":"Mandatory",     "action":"Yield in roundabout",         "critical":False, "alert":""},
    "SPEED_BREAKER":    {"cat":"Warning",       "action":"Speed bump ahead — slow down","critical":False, "alert":""},
    "HOSPITAL":         {"cat":"Informational", "action":"Hospital nearby",             "critical":False, "alert":""},
    "PARKING":          {"cat":"Informational", "action":"Parking available",           "critical":False, "alert":""},
    "PETROL_PUMP":      {"cat":"Informational", "action":"Fuel station nearby",         "critical":False, "alert":""},
    "BUS_STOP":         {"cat":"Informational", "action":"Bus stop ahead",              "critical":False, "alert":""},
    "REST_AREA":        {"cat":"Informational", "action":"Rest area ahead",             "critical":False, "alert":""},
    "TOILET":           {"cat":"Informational", "action":"Restroom facilities nearby",  "critical":False, "alert":""},
}

CAT_COLORS = {
    "Mandatory":     (200,  80,  20),
    "Prohibitory":   ( 30,  30, 210),
    "Warning":       (  0, 160, 255),
    "Regulatory":    ( 30,  30, 180),
    "Informational": (180, 120,  20),
}

# ── Helpers ───────────────────────────────────────────────

def txt(img, text, x, y, scale, color, thick=1):
    cv2.putText(img,text,(x+1,y+1),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def fill_rect(img, x1,y1,x2,y2, color, alpha=0.80):
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def find_model():
    """Find the best trained model or fall back to base yolov8s."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for trained model
    patterns = [
        os.path.join(script_dir,"runs","traffic_signs","weights","best.pt"),
        os.path.join(script_dir,"runs","*","weights","best.pt"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]

    # Fallback: use base YOLOv8s (detects general objects, not our signs)
    print("[WARN] Trained model not found — using base yolov8s.pt")
    print("       For traffic sign detection, run:  python train.py")
    return "yolov8s.pt"


# ─────────────────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────────────────

def run():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("  Run:  pip install ultralytics")
        sys.exit(1)

    # Load model
    model_path = find_model()
    print(f"[MODEL] Loading: {model_path}")
    model = YOLO(model_path)
    print("[MODEL] Ready.")
    print()
    print("Controls:  Q/ESC=Quit  S=Save  +/-=Confidence threshold")
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  Try changing VideoCapture(0) to VideoCapture(1) in detect.py")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf_thresh = 0.40       # minimum confidence to show detection
    total_det   = 0
    fps_t       = time.time()
    fps_count   = 0
    fps         = 0.0
    os.makedirs("screenshots", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame lost.")
            break

        # FPS
        fps_count += 1
        if time.time()-fps_t >= 0.5:
            fps = fps_count/(time.time()-fps_t)
            fps_count = 0; fps_t = time.time()

        display = frame.copy()

        # ── YOLO inference ────────────────────────────────
        results = model(frame, conf=conf_thresh, verbose=False)[0]

        best = None   # track highest-confidence detection for side panel

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # Get class name
            cls_name = model.names[cls_id]
            info = SIGN_INFO.get(cls_name, {"cat":"Unknown","action":"—","critical":False,"alert":""})
            color = CAT_COLORS.get(info["cat"], (150,150,150))

            # Draw bounding box
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)

            # Label bar above box
            label = f"{cls_name.replace('_',' ')}  {conf:.0%}"
            (lw,lh),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            fill_rect(display,x1,y1-22,x1+lw+10,y1,( 15,15,15),0.85)
            txt(display,label,x1+5,y1-7,0.45,color,1)

            # Alert bar below box for critical signs
            if info["critical"] and info["alert"]:
                fill_rect(display,x1,y2,x2,y2+22,(20,20,160),0.88)
                txt(display,info["alert"][:45],x1+5,y2+15,0.38,(255,255,255),1)

            # Track best detection
            if best is None or conf > best[1]:
                best = (cls_name, conf, info)

        if results.boxes:
            total_det += len(results.boxes)

        # ── Side info panel ───────────────────────────────
        h, w = display.shape[:2]
        pw = 280
        fill_rect(display, w-pw-10, 52, w-10, h-10, (15,18,24), 0.88)
        cv2.rectangle(display,(w-pw-10,52),(w-10,h-10),(50,60,70),1)
        py = 72

        if best:
            cls_name, conf, info = best
            color = CAT_COLORS.get(info["cat"],(150,150,150))
            txt(display, cls_name.replace("_"," "), w-pw+5, py, 0.50, (255,255,255), 1); py+=22
            txt(display, info["cat"],              w-pw+5, py, 0.36, color, 1);          py+=20
            cv2.line(display,(w-pw+5,py),(w-15,py),(50,60,70),1); py+=12

            # Confidence bar
            txt(display,"CONFIDENCE",w-pw+5,py,0.30,(100,120,130),1); py+=14
            bw = pw-20
            cv2.rectangle(display,(w-pw+5,py),(w-pw+5+bw,py+7),(40,50,60),-1)
            fc = int(bw*conf)
            bc = (0,200,100) if conf>=0.75 else (0,165,255) if conf>=0.55 else (0,80,220)
            cv2.rectangle(display,(w-pw+5,py),(w-pw+5+fc,py+7),bc,-1)
            txt(display,f"{conf:.0%}",w-pw+5+bw-30,py+6,0.33,(255,255,255),1); py+=20

            cv2.line(display,(w-pw+5,py),(w-15,py),(50,60,70),1); py+=12
            txt(display,"ACTION",w-pw+5,py,0.30,(100,120,130),1); py+=14

            # Word-wrap action text
            words = info["action"].split()
            line=""
            for word in words:
                test=line+(" " if line else "")+word
                (tw,_),_=cv2.getTextSize(test,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)
                if tw>bw: txt(display,line,w-pw+5,py,0.36,(0,200,130),1); py+=15; line=word
                else: line=test
            if line: txt(display,line,w-pw+5,py,0.36,(0,200,130),1); py+=15

            if info["critical"]:
                py+=6
                fill_rect(display,w-pw+5,py,w-15,py+28,(20,20,140),0.90)
                cv2.rectangle(display,(w-pw+5,py),(w-15,py+28),(80,80,220),1)
                txt(display,info["alert"][:30],w-pw+10,py+12,0.33,(255,255,255),1)
                if len(info["alert"])>30:
                    txt(display,info["alert"][30:60],w-pw+10,py+24,0.33,(255,255,255),1)
        else:
            txt(display,"No sign detected",w-pw+5,py,0.38,(80,90,100),1); py+=20
            txt(display,"Hold sign to camera",w-pw+5,py,0.33,(60,70,80),1)

        # ── Header bar ────────────────────────────────────
        fill_rect(display,0,0,w,44,(10,14,20),0.88)
        cv2.line(display,(0,44),(w,44),(0,200,100),1)
        txt(display,"TRAFFIC SIGN RECOGNITION — YOLOv8",12,16,0.45,(0,200,100),1)
        txt(display,f"FPS:{fps:.0f}  CONF:{conf_thresh:.0%}  TOTAL:{total_det}",12,34,0.35,(80,100,110),1)

        # ── Controls bar ──────────────────────────────────
        fill_rect(display,0,h-30,w,h,(10,14,20),0.88)
        cv2.line(display,(0,h-30),(w,h-30),(50,60,70),1)
        hints=[("Q","Quit"),("S","Save"),("+","Conf+"),("-","Conf-")]
        cx2=10
        for k,lbl in hints:
            (kw,_),_=cv2.getTextSize(k,cv2.FONT_HERSHEY_SIMPLEX,0.32,1)
            cv2.rectangle(display,(cx2-2,h-24),(cx2+kw+6,h-8),(40,50,60),-1)
            txt(display,k,cx2+2,h-11,0.32,(220,225,230),1); cx2+=kw+12
            (lw2,_),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.32,1)
            txt(display,lbl,cx2,h-11,0.32,(80,90,100),1); cx2+=lw2+20

        cv2.imshow("Traffic Sign Recognition — YOLOv8", display)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"screenshots/det_{ts}.jpg"
            cv2.imwrite(fname, display)
            print(f"[SAVE] {fname}")
        elif key == ord('+'):
            conf_thresh = min(0.95, conf_thresh+0.05)
            print(f"[CONF] Threshold: {conf_thresh:.0%}")
        elif key == ord('-'):
            conf_thresh = max(0.10, conf_thresh-0.05)
            print(f"[CONF] Threshold: {conf_thresh:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Session ended — {total_det} total detections")


if __name__ == "__main__":
    run()
