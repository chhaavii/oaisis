import cv2
import numpy as np
import pyttsx3
import time
from collections import deque

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Slower speech rate
last_spoken = {}
min_speak_interval = 3  # Minimum seconds between speaking the same object

# Detection history for stability
detection_history = deque(maxlen=5)  # Keep last 5 detections
DETECTION_INTERVAL = 1.0  # Seconds between detections
last_detection_time = 0

def speak(text):
    """Speak the given text if not spoken recently"""
    current_time = time.time()
    if text not in last_spoken or (current_time - last_spoken[text]) > min_speak_interval:
        try:
            engine.say(text)
            engine.runAndWait()
            last_spoken[text] = current_time
            print(f"Speaking: {text}")  # Debug output
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)  # Try alternative camera index
    if not cap.isOpened():
        print("Error: Could not open any webcam")
        exit()

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
                             'MobileNetSSD_deploy.caffemodel')

# List of class names the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "cell phone"]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Starting object detection...")
print("Press 'q' to quit")

while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    (h, w) = frame.shape[:2]
    
    # Only process detection every DETECTION_INTERVAL seconds
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        last_detection_time = current_time
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            0.007843, 
            (300, 300), 
            127.5
        )
        net.setInput(blob)
        detections = net.forward()
        
        # Clear previous detections
        detection_history.clear()
        
        # Process new detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Higher confidence threshold for more stable results
            if confidence > 0.6:  # Increased from 0.2 to 0.6
                idx = int(detections[0, 0, i, 1])
                class_name = CLASSES[idx] if idx < len(CLASSES) else "unknown"
                detection_history.append((class_name, confidence))
                
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the prediction on the frame
                color = (0, 255, 0) if class_name == "cell phone" else COLORS[idx % len(COLORS)]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence*100:.1f}%"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Only speak for high confidence detections
                if confidence > 0.7:  # Higher threshold for speaking
                    speak_text = f"{class_name} detected"
                    speak(speak_text)
    
    # Show the output frame
    cv2.imshow("Object Detection", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
engine.stop()