import os
import cv2
import time
import pafy
import numpy as np
from ultralytics import YOLO

# ✅ Set PAFY to use yt-dlp (Fixes YouTube video loading issue)
os.environ["PAFY_BACKEND"] = "internal"

# Load YOLOv8 pose estimation model
model = YOLO("yolov8n-pose.pt")  

# 🎥 YouTube Video URL (Change this to your desired video)
youtube_url = "https://www.youtube.com/watch?v=I4AI4G5fHxY"
use_youtube = True  # ✅ Set False to use local video or webcam

# 📌 Load Video Source (YouTube / Local Video / Webcam)
if use_youtube:
    try:
        video = pafy.new(youtube_url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
        print(f"🎥 Streaming YouTube video: {youtube_url}")
    except Exception as e:
        print(f"❌ Error loading YouTube video: {e}")
        exit()
else:
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or "fall_video.mp4" for a local video

# 🚨 Alert Counter (Ensures alert is triggered only after 20 frames of fall detection)
alert_var = 0  

def process_detections(results, frame):
    """
    Process YOLO detections and check for fall conditions.
    """
    global alert_var
    fall_alert_list = []  # Store IDs of falling people
    centroid_dict = {}  # Store person detections

    for result in results:
        for idx, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
            label = model.names[int(result.boxes.cls[idx])]  # Get class label

            if label == "person":
                centroid_dict[idx] = (x1, y1, x2, y2)

    # 🔍 Check Fall Conditions
    for idx, (x1, y1, x2, y2) in centroid_dict.items():
        width = x2 - x1
        height = y2 - y1
        difference = height - width  # 📌 Height should be greater than width for a standing person

        if difference < 0:  # 🚨 Fall detected (Person is lying down)
            fall_alert_list.append(idx)

    # 📌 Draw Bounding Boxes
    for idx, (x1, y1, x2, y2) in centroid_dict.items():
        color = (0, 255, 0)  # Green for standing
        if idx in fall_alert_list:
            color = (0, 0, 255)  # Red for fall
