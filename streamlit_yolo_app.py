# streamlit_yolo_app.py
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np

st.set_page_config(page_title="YOLOv8 Streamlit App", layout="wide")
st.title("YOLOv8 - Object Detection / Pose / Segmentation")

# -----------------------------
# Model selection
# -----------------------------
model_option = st.selectbox("Select YOLOv8 Model", ["Detection", "Segmentation", "Pose"])
if model_option == "Detection":
    model_path = "yolov8n.pt"
elif model_option == "Segmentation":
    model_path = "yolov8n-seg.pt"
else:
    model_path = "yolov8n-pose.pt"

model = YOLO(model_path)
st.success(f"Loaded model: {model_path}")

# -----------------------------
# Video or webcam input
# -----------------------------
input_option = st.radio("Input Source", ["Webcam", "Upload Video"])
if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    else:
        st.warning("Please upload a video file.")
        st.stop()
else:
    video_path = 0  # webcam

# -----------------------------
# Video processing
# -----------------------------
run_button = st.button("Start Detection")
if run_button:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video source!")
        st.stop()

    # Video writer (optional)
    save_output = st.checkbox("Save output video")
    if save_output:
        output_file = "yolo_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Write output
        if save_output:
            video_writer.write(annotated_frame)

        # Convert BGR to RGB for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

        # Stop button
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_output:
        video_writer.release()
        st.success(f"Output video saved as {output_file}")
    st.success("Detection completed!")
