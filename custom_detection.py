import cv2
from ultralytics import YOLO
import os

# -----------------------------
# Paths
# -----------------------------
video_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\WhatsApp Video 2025-10-22 at 4.26.33 PM.mp4"
output_video_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\customer_video_output.mp4"
output_frames_dir = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\output_frames"

# Create folder to save frames
os.makedirs(output_frames_dir, exist_ok=True)

# -----------------------------
# Load YOLOv8 model
# -----------------------------
model = YOLO("yolov8n.pt")  # Replace with your custom model if available

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Cannot open video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# -----------------------------
# Prepare output video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Write annotated frame to video
    video_writer.write(annotated_frame)

    # Save frame as image
    frame_file = os.path.join(output_frames_dir, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(frame_file, annotated_frame)

    # Optional: show live detection
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video stopped by user.")
        break

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

# -----------------------------
# Release resources
# -----------------------------
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"✅ Output video saved to: {output_video_path}")
print(f"✅ Annotated frames saved to folder: {output_frames_dir}")
