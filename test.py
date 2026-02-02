import cv2
import mediapipe as mp
import socket
import time
import numpy as np

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
PYNQ_IP = "192.168.2.99"  # CHANGE THIS
UDP_PORT = 5005
SEND_DATA = True         # Set to True when ready
MODEL_PATH = 'hand_landmarker.task' # Ensure this file exists!

# ----------------------------------------
# NEW MEDIAPIPE TASKS SETUP
# ----------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the latest result asynchronously
latest_result = None

# The Async Callback Function
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def is_fist(hand_landmarks):
    # Calculate distance between Wrist (0) and Index Tip (8)
    # Note: In MediaPipe Tasks, landmarks are accessed by index
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]
    
    # Euclidean distance (normalized 0.0 to 1.0)
    dist = ((wrist.x - index_tip.x)**2 + (wrist.y - index_tip.y)**2)**0.5
    
    # Threshold: < 0.2 usually means fingers are curled in
    return dist < 0.2

# Initialize the Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)

# ----------------------------------------
# UDP SETUP
# ----------------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Starting Vision Tracking (Tasks API)... Press 'q' to quit.")

# Start capturing
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # 1. Pre-process Image
        # Flip for "Mirror" feel
        frame = cv2.flip(frame, 1)
        # Convert to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 2. Inference (Async)
        # We must provide a timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # 3. Process Results (if available from callback)
        if latest_result and latest_result.hand_landmarks:
            # Get the first hand detected
            hand_landmarks = latest_result.hand_landmarks[0]
            
            # Extract Wrist (Index 0)
            wrist = hand_landmarks[0] # Note: tasks API uses index, not enum like solutions

            # Get Gesture State
            # 1 = Fist (Closed), 0 = Open
            # We send this as the 3rd number in our packet
            grab_state = 1 if is_fist(hand_landmarks) else 0
            
            # Payload: "x, y, grab_state"
            payload = f"{wrist.x:.3f},{wrist.y:.3f},{grab_state}"
            
            if SEND_DATA:
                sock.sendto(payload.encode(), (PYNQ_IP, UDP_PORT))

            # ----------------------------------------
            # VISUALIZATION
            # ----------------------------------------
            # Draw landmarks manually (since drawing_utils is also deprecated/different)
            h, w, _ = frame.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)

                        # Visual Debug
            status_text = "GRIP!" if grab_state else "OPEN"
            color = (0, 0, 255) if grab_state else (0, 255, 0)
            cv2.putText(frame, status_text, (cx, cy - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw Wrist
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Target: {payload}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw Skeleton (Simple loop)
            # Just drawing points for simplicity in this demo
            for landmark in hand_landmarks:
                lx, ly = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)

        cv2.imshow('Kinetic Mirror (Tasks API)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()