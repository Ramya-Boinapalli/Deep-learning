import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import time

# ----------------------------
# Video Path
# ----------------------------
video_path = r"C:\Users\Ramya\VS Code Project\DS_10am\DL_env\YOLO\pulse_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # default fallback
print(f"Video FPS: {fps}")

# ----------------------------
# Select ROI manually
# ----------------------------
ret, frame = cap.read()
if not ret:
    print("❌ Error: Cannot read first frame.")
    exit()

roi = cv2.selectROI("Select Wrist ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Wrist ROI")
x1, y1, w, h = roi
x2, y2 = x1 + w, y1 + h

green_signal = []

# ----------------------------
# Helper: Bandpass Filter
# ----------------------------
def bandpass_filter(signal, low=0.8, high=3.0, fs=30):
    nyquist = 0.5 * fs
    low_cut = low / nyquist
    high_cut = high / nyquist
    b, a = butter(3, [low_cut, high_cut], btype='band')
    return filtfilt(b, a, signal)

# ----------------------------
# Process for 1 Minute (60 seconds)
# ----------------------------
start_time = time.time()
while True:
    ret, frame = cap.read()

    # 🔁 Loop the video if it ends before 60 seconds
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    roi_frame = frame[y1:y2, x1:x2]
    if roi_frame.size == 0:
        continue

    # Mean green channel
    green_mean = np.mean(roi_frame[:, :, 1])
    green_signal.append(green_mean)

    # Show video feed
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    elapsed = time.time() - start_time
    cv2.putText(frame, f"Recording: {int(elapsed)}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Wrist Pulse Detection", frame)

    # Stop after 60 seconds
    if elapsed >= 60:
        print("✅ Recording complete — 1 minute finished.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Stopped manually.")
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------
# Post-processing: Compute BPM
# ----------------------------
green_signal = np.array(green_signal)
if len(green_signal) < 2:
    print("❌ Not enough data to calculate BPM.")
    exit()

filtered_signal = bandpass_filter(green_signal, fs=fps)
peaks, _ = find_peaks(filtered_signal, distance=int(fps * 0.5))
bpm = len(peaks) * 60 / (len(filtered_signal) / fps)
print(f"❤️ Estimated BPM: {bpm:.2f}")
