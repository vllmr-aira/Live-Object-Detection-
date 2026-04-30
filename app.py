import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

# Create folder for saved images
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

# Cache model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection, Counting & Alerts")

# Target object for alert
TARGET_OBJECT = "person"

# Frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    boxes = results[0].boxes
    names = model.names

    object_counts = {}
    alert_triggered = False

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]

            # Count objects
            object_counts[label] = object_counts.get(label, 0) + 1

            # Trigger alert
            if label == TARGET_OBJECT:
                alert_triggered = True

    # Annotate frame
    annotated_frame = results[0].plot()

    # Add count text overlay
    y_offset = 30
    for obj, count in object_counts.items():
        text = f"{obj}: {count}"
        cv2.putText(annotated_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_offset += 30

    # Save frame if alert triggered
    if alert_triggered:
        timestamp = int(time.time())
        filename = f"{SAVE_DIR}/alert_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)