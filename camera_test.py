import cv2
import time
import os

def test_camera_headless(device_id=0):
    print(f"Opening camera {device_id}...")
    cap = cv2.VideoCapture(device_id)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Error: Could not open video device {device_id}")
        return

    print("Camera opened. Capturing 10 frames for sanity check...")
    
    start_time = time.time()
    success_count = 0
    last_frame = None

    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            last_frame = frame
            # Print frame stats to prove we have real data (not just black pixels)
            mean_val = frame.mean()
            print(f"Frame {i+1}: Shape={frame.shape}, Mean Pixel Intensity={mean_val:.2f}")
        else:
            print(f"Frame {i+1}: Failed to grab.")

    end_time = time.time()
    
    # Calculate Stats
    duration = end_time - start_time
    fps = success_count / duration if duration > 0 else 0
    print(f"\nCaptured {success_count} frames in {duration:.2f} seconds.")
    print(f"Estimated FPS: {fps:.2f}")

    # Save validation image
    if last_frame is not None:
        filename = "camera_sanity.jpg"
        cv2.imwrite(filename, last_frame)
        print(f"Saved sanity check image to '{os.getcwd()}/{filename}'")
    
    cap.release()

if __name__ == "__main__":
    test_camera_headless(0)