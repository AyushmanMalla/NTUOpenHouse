# import tensorflow as tf
# import tensorflow_hub as hub
# import cv2
# import numpy as np

# # --- CONFIGURATION ---
# MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
# INPUT_SIZE = 256 
# CONF_THRESH = 0.3

# # --- SKELETON CONNECTIONS ---
# # These are the pairs of keypoints that make up the bones
# KEYPOINT_EDGES = [
#     (0, 1), (0, 2), (1, 3), (2, 4),           # Face / Head
#     (0, 5), (0, 6),                           # Nose to Shoulders
#     (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
#     (5, 7), (7, 9),                           # Left Arm
#     (6, 8), (8, 10),                          # Right Arm
#     (11, 13), (13, 15),                       # Left Leg
#     (12, 14), (14, 16)                        # Right Leg
# ]

# def main():
#     print("Loading MoveNet...")
#     model = hub.load(MODEL_URL)
#     movenet = model.signatures['serving_default']
#     print("Loaded.")

#     cap = cv2.VideoCapture(0)
#     print("Running Skeleton Tracking Demo. Press 'q' to exit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret: 
#             break
        
#         h, w = frame.shape[:2]

#         # --- PREPROCESSING ---
#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Resize with padding to maintain aspect ratio, then cast to int32 for MoveNet
#         img_tensor = tf.expand_dims(rgb_frame, axis=0)
#         img_tensor = tf.image.resize_with_pad(img_tensor, INPUT_SIZE, INPUT_SIZE)
#         input_tensor = tf.cast(img_tensor, dtype=tf.int32)
        
#         # --- INFERENCE ---
#         outputs = movenet(input_tensor)
#         # MoveNet returns shape: [1, 1, 17, 3] -> [batch, person, keypoint, (y, x, score)]
#         keypoints = outputs['output_0'].numpy()[0, 0, :, :]

#         # Extract points and convert normalized coordinates (0.0 - 1.0) to pixel coordinates
#         points_px = []
#         for kp in keypoints:
#             ky, kx, score = kp
#             cx, cy = int(kx * w), int(ky * h)
#             points_px.append((cx, cy, score))

#         # --- DRAWING BONES (LINES) ---
#         for edge in KEYPOINT_EDGES:
#             p1, p2 = edge
#             # Only draw the bone if BOTH keypoints are confident enough
#             if points_px[p1][2] > CONF_THRESH and points_px[p2][2] > CONF_THRESH:
#                 cv2.line(frame, points_px[p1][:2], points_px[p2][:2], (0, 255, 0), 2)

#         # --- DRAWING JOINTS (CIRCLES) ---
#         for cx, cy, score in points_px:
#             if score > CONF_THRESH:
#                 cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#         # Visuals
#         cv2.putText(frame, "MoveNet Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
#         cv2.imshow("Kinetic Tracker", frame)
#         if cv2.waitKey(1) == ord('q'): 
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"
INPUT_SIZE = 256  # Must be a multiple of 32 for Multipose
CONF_THRESH = 0.3
MAX_SUBJECTS = 2  # Hard limit for tracking

# --- SKELETON CONNECTIONS ---
KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Face / Head
    (0, 5), (0, 6),                           # Nose to Shoulders
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (5, 7), (7, 9),                           # Left Arm
    (6, 8), (8, 10),                          # Right Arm
    (11, 13), (13, 15),                       # Left Leg
    (12, 14), (14, 16)                        # Right Leg
]

# Distinct colors to separate the subjects visually
SUBJECT_COLORS = [
    {"bone": (0, 255, 0), "joint": (0, 0, 255)},     # Person 1: Green/Red
    {"bone": (255, 255, 0), "joint": (255, 0, 255)}  # Person 2: Cyan/Magenta
]

def main():
    print("Loading MoveNet MultiPose...")
    model = hub.load(MODEL_URL)
    movenet = model.signatures['serving_default']
    print("Loaded.")

    cap = cv2.VideoCapture(0)
    print("Running Multi-Subject Tracking Demo. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        h, w = frame.shape[:2]

        # --- PREPROCESSING ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = tf.expand_dims(rgb_frame, axis=0)
        
        # Multipose requires dimensions to be a multiple of 32
        img_tensor = tf.image.resize_with_pad(img_tensor, INPUT_SIZE, INPUT_SIZE)
        input_tensor = tf.cast(img_tensor, dtype=tf.int32)
        
        # --- INFERENCE ---
        outputs = movenet(input_tensor)
        
        # MultiPose outputs a tensor of shape: [1, 6, 56] -> [batch, person, data]
        # Data structure per person: 17 keypoints (y, x, score) = 51 values + bounding box (ymin, xmin, ymax, xmax, score) = 5 values
        poses = outputs['output_0'].numpy()[0] 

        # Sort the 6 potential poses by the bounding box confidence score (index 55) in descending order
        poses = sorted(poses, key=lambda x: x[55], reverse=True)

        subjects_tracked = 0

        # --- DRAWING ---
        for i, pose in enumerate(poses):
            if subjects_tracked >= MAX_SUBJECTS:
                break # Stop if we hit your 2-subject limit
                
            box_score = pose[55]
            if box_score < CONF_THRESH:
                continue # Skip background noise/empty slots
            
            subjects_tracked += 1
            colors = SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
            
            # Extract the 17 keypoints (first 51 values) and reshape to a readable (17, 3) matrix
            keypoints = pose[:51].reshape((17, 3))
            
            points_px = []
            for kp in keypoints:
                ky, kx, score = kp
                
                # Accurately map coordinates back from the padded 256x256 square to your original webcam frame
                scale = max(w, h)
                pad_x = (scale - w) / 2.0
                pad_y = (scale - h) / 2.0
                
                cx = int((kx * scale) - pad_x)
                cy = int((ky * scale) - pad_y)
                points_px.append((cx, cy, score))

            # Draw Bones
            for edge in KEYPOINT_EDGES:
                p1, p2 = edge
                if points_px[p1][2] > CONF_THRESH and points_px[p2][2] > CONF_THRESH:
                    cv2.line(frame, points_px[p1][:2], points_px[p2][:2], colors["bone"], 2)

            # Draw Joints
            for cx, cy, score in points_px:
                if score > CONF_THRESH:
                    cv2.circle(frame, (cx, cy), 5, colors["joint"], -1)

        # Visuals
        cv2.putText(frame, f"MultiPose Tracker (Tracking: {subjects_tracked})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Kinetic Tracker", frame)
        if cv2.waitKey(1) == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()