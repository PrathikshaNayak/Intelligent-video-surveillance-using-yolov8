from ultralytics import YOLO
import yt_dlp
import cv2
import os

# Your trained model
model = YOLO('firedetection.pt')  # Update the model path

# YouTube video URL
youtube_url = "https://youtu.be/ZT0pmw0jqZQ?si=9jtl9_v0ODgrTpap"

# Set yt_dlp options to download
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'youtube_video.mp4',  # Save as this file
}

# Download video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([youtube_url])

# Now open the downloaded file
cap = cv2.VideoCapture('youtube_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream ended")
        break

    # Run YOLOv8 detection
    results = model.predict(frame, imgsz=640, conf=0.25)
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow('Fire Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Optionally delete video after viewing
os.remove('youtube_video.mp4')
