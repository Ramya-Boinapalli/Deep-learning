r'''
import cv2
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------
# Input video (or 0 for webcam)
VIDEO_PATH = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\WhatsApp Video 2025-10-22 at 4.26.33 PM.mp4"

# YOLOv8 pose model path
MODEL_PATH = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\yolov8n-pose.pt"

# Optional: Save annotated output
SAVE_OUTPUT = True
OUTPUT_PATH = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\pose_estimation_output.mp4"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# INITIALIZE VIDEO CAPTURE
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception(f"❌ Cannot open video: {VIDEO_PATH}")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# -----------------------------
# VIDEO WRITER (optional)
# -----------------------------
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# -----------------------------
# VIDEO LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video.")
        break

    # Run pose estimation
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Save output if enabled
    if SAVE_OUTPUT:
        video_writer.write(annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("⏹️ Stopped by user.")
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
if SAVE_OUTPUT:
    video_writer.release()
cv2.destroyAllWindows()

if SAVE_OUTPUT:
    print(f"🎯 Output saved to: {OUTPUT_PATH}")
'''

import cv2
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\WhatsApp Video 2025-10-22 at 4.26.33 PM.mp4"
MODEL_PATH = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\yolov8n-pose.pt"
OUTPUT_IMAGE_PATH = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\pose_output_image.jpg"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# CAPTURE FIRST FRAME
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception(f"Cannot open video: {VIDEO_PATH}")

ret, frame = cap.read()
if not ret:
    raise Exception("❌ Could not read first frame from video!")

# -----------------------------
# RUN POSE ESTIMATION
# -----------------------------
results = model(frame)

# Annotate frame
annotated_frame = results[0].plot()

# -----------------------------
# SAVE OUTPUT IMAGE
# -----------------------------
cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
print(f"✅ Pose image saved to: {OUTPUT_IMAGE_PATH}")

# -----------------------------
# SHOW IMAGE (optional)
# -----------------------------
cv2.imshow("Pose Estimation (First Frame)", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()



