import cv2
from ultralytics import YOLO

MODEL_PATH = "dataset/runs/detect/train-2/weights/best_named.pt"
CONFIDENCE = 0.40

print("\nLoading traffic sign detection model...")
print(f"Model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("\nClass names:")
for class_id, name in model.names.items():
    print(f"{class_id}: {name}")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("\nERROR: Could not open camera.")
    raise SystemExit

print("\nCamera opened successfully.")
print("Traffic sign detection is running.")
print("Press Q or ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("\nERROR: Could not read camera frame.")
        break

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Sign Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nDetection stopped.")
