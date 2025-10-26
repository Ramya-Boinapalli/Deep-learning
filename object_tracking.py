import cv2
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = 0  # 0 = default webcam
MODEL_PATH = "yolov8n.pt"  # Replace with your custom model if available

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# INITIALIZE VIDEO CAPTURE
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Cannot open webcam!")

# -----------------------------
# READ FIRST FRAME
# -----------------------------
ret, frame = cap.read()
if not ret:
    raise Exception("Cannot read first frame from webcam!")

# -----------------------------
# DETECT FIRST OBJECT
# -----------------------------
results = model(frame)

if len(results[0].boxes) == 0:
    raise Exception("No object detected in the first frame!")

# Take first detected object
bbox = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
x, y, x2, y2 = bbox
w, h = x2 - x, y2 - y
bbox = (int(x), int(y), int(w), int(h))

# -----------------------------
# INITIALIZE TRACKER (KCF)
# -----------------------------
tracker = cv2.legacy.TrackerKCF_create()
tracker.init(frame, bbox)

# -----------------------------
# VIDEO LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from webcam!")
        break

    # UPDATE TRACKER
    success, bbox = tracker.update(frame)
    
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking lost! Re-detecting...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Re-detect using YOLO
        results = model(frame)
        if len(results[0].boxes) > 0:
            bbox = results[0].boxes.xyxy[0].cpu().numpy()
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            bbox = (int(x), int(y), int(w), int(h))
            tracker = cv2.legacy.TrackerKCF_create()
            tracker.init(frame, bbox)

    # DISPLAY
    cv2.imshow("YOLOv8 + KCF Tracker", frame)
    
    # EXIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("⏹️ Stopped by user.")
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
