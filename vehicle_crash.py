from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano (fast and lightweight)

# Open video file or webcam
cap = cv2.VideoCapture("crash_video.mp4")  # Replace with 0 for webcam

alert_var = 0  # Counter for crash alert

def is_overlap(box1, box2):
    """
    Check if two bounding boxes overlap (possible crash detection).
    :param box1, box2: Bounding box coordinates (x1, y1, x2, y2)
    :return: True if boxes overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if not (x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2):
        return True
    return False

def process_detections(results, frame):
    """
    Process YOLO detections and check for vehicle crashes.
    """
    global alert_var
    vehicle_boxes = []  # Store bounding boxes of detected vehicles
    crash_alert_list = []  # Store vehicles involved in crashes

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
            label = model.names[int(cls)]  # Get class label

            if label in ["car", "truck", "bus"]:  # Detect vehicles only
                vehicle_boxes.append((x1, y1, x2, y2))

    # Check for crashes (overlapping bounding boxes)
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            if is_overlap(vehicle_boxes[i], vehicle_boxes[j]):
                crash_alert_list.append(vehicle_boxes[i])
                crash_alert_list.append(vehicle_boxes[j])

    # Draw bounding boxes
    for box in vehicle_boxes:
        color = (0, 255, 0)  # Green for normal vehicles
        if box in crash_alert_list:
            color = (0, 0, 255)  # Red for crashes
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display crash detection message
    if crash_alert_list:
        text = "Crash Detected"
        if alert_var >= 8:  # Alert after 8 continuous frames
            cv2.imwrite("crash_alert.jpg", frame)  # Save image
            print("🚨 Crash detected! Alert generated!")  # Call alert function here if needed
        alert_var += 1
    else:
        text = "Crash Not Detected"
        alert_var = 0  # Reset alert counter if no crashes detected

    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) if crash_alert_list else (0, 255, 0), 2)

    return frame

# Main loop
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform detection
    processed_frame = process_detections(results, frame)

    # Show output
    fps = round(1 / (time.time() - start_time), 2)
    cv2.putText(processed_frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Vehicle Crash Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
