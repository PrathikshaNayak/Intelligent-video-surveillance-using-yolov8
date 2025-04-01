from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano (fast and lightweight)

# Open video file or webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or a YouTube/IP link

# Define colors for object classes
color_dict = {
    "person": (0, 255, 255), "bicycle": (238, 123, 158), "car": (24, 245, 217),
    "motorbike": (224, 119, 227), "bus": (179, 50, 247), "truck": (82, 42, 106),
    "boat": (201, 25, 52), "traffic light": (62, 17, 209), "stop sign": (199, 113, 167),
    "bench": (161, 83, 182), "cat": (100, 64, 151), "dog": (156, 116, 171),
    "horse": (88, 9, 123), "elephant": (74, 90, 143), "zebra": (26, 101, 131),
    "laptop": (159, 149, 163), "mouse": (148, 148, 87), "remote": (171, 107, 183),
    "keyboard": (33, 154, 135), "cell phone": (206, 209, 108), "microwave": (206, 209, 108),
    "oven": (97, 246, 15), "sink": (157, 58, 24), "refrigerator": (117, 145, 137)
}

# Set up video output (optional)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_video = "output.avi"
save_output = False  # Change to True to save video
if save_output:
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (frame_width, frame_height))

def draw_detections(results, frame):
    """
    Draw bounding boxes around detected objects.
    """
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
            label = model.names[int(cls)]  # Get class label
            confidence = round(float(conf), 2)  # Confidence score

            color = color_dict.get(label, (255, 255, 255))  # Get color for object or default to white
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bounding box
            cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Main loop
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform object detection
    processed_frame = draw_detections(results, frame)

    # Show output
    fps = round(1 / (time.time() - start_time), 2)
    cv2.putText(processed_frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("YOLOv8 Object Detection", processed_frame)

    # Save video if enabled
    if save_output:
        out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
