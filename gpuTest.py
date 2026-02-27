import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time
import math
import serial

# --- CONFIGURATION ---
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256 
CONF_THRESH = 0.3
SERIAL_PORT = "/dev/ttyACM0" 
BAUD_RATE = 115200
PARALLEL_THRESH = 60 

# --- EMA FILTER ---
# Acts as a mathematical shock absorber for jittery CV data
class EMAFilter:
    def __init__(self, alpha=0.15):
        # Alpha controls smoothness. 
        # 0.01 = Very slow/smooth. 1.0 = Instant/jittery.
        self.alpha = alpha
        self.val = None
        
    def __call__(self, x):
        if self.val is None:
            self.val = x
        else:
            self.val = self.alpha * x + (1.0 - self.alpha) * self.val
        return self.val

class SimpleRobotLink:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.last_sent = [-1, -1, -1, -1, -1, -1] 
        
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2) 
            print(f"Connected to Braccio on {port}")
        except Exception as e:
            print(f"Connection Failed (Sim Mode): {e}")

    def send_target(self, m1, m2, m3, m4, m5, m6):
        # 1. Enforce Hard Limits
        m1 = int(np.clip(m1, 40, 165))
        m2 = int(180)
        m3 = int(np.clip(m3, 90, 180))
        m4 = int(np.clip(m4, 0, 150))
        m5 = int(np.clip(m5, 0, 180))
        m6 = int(np.clip(m6, 10, 73))
        
        target = [m1, m2, m3, m4, m5, m6]
        
        # 2. ONLY send if the integer target has actually changed
        if target != self.last_sent:
            cmd = f"P{m1},{m2},{m3},{m4},{m5},{m6}\n"
            if self.ser:
                try:
                    self.ser.write(cmd.encode())
                except Exception as e:
                    print(f"UART Error: {e}")
            self.last_sent = target
            
        return target

def map_val(x, in_min, in_max, out_min, out_max):
    x = max(in_min, min(in_max, x))
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def main():
    print("Loading MoveNet...")
    model = hub.load(MODEL_URL)
    movenet = model.signatures['serving_default']
    
    robot = SimpleRobotLink(SERIAL_PORT, BAUD_RATE)
    cap = cv2.VideoCapture(0)
    
    # Initialize our shock absorbers
    filter_m4 = EMAFilter(alpha=0.15)
    filter_m3 = EMAFilter(alpha=0.15)
    
    print("Running. Raise arm parallel to ground to activate.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]

        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = tf.cast(tf.image.resize_with_pad(img, INPUT_SIZE, INPUT_SIZE), dtype=tf.int32)
        input_tensor = tf.expand_dims(img, axis=0)
        
        outputs = movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]

        shldr = (int(keypoints[6][1]*w), int(keypoints[6][0]*h), keypoints[6][2])
        elbow = (int(keypoints[8][1]*w), int(keypoints[8][0]*h), keypoints[8][2])
        wrist = (int(keypoints[10][1]*w), int(keypoints[10][0]*h), keypoints[10][2])

        # Updated Rest Position
        t_m1, t_m2, t_m3, t_m4, t_m5, t_m6 = 40, 180, 90, 75, 180, 73
        state = "RESTING"

        if shldr[2] > CONF_THRESH and elbow[2] > CONF_THRESH and wrist[2] > CONF_THRESH:
            cv2.line(frame, shldr[:2], elbow[:2], (255,255,0), 2)
            cv2.line(frame, elbow[:2], wrist[:2], (0,255,0), 2)
            cv2.circle(frame, shldr[:2], 5, (0,0,255), -1)
            cv2.circle(frame, elbow[:2], 5, (255,255,0), -1)
            cv2.circle(frame, wrist[:2], 5, (0,255,0), -1)

            # --- 1. ACTIVE TRIGGER FIX ---
            # We ONLY check the elbow's height relative to the shoulder. 
            # The wrist is now free to move freely as it folds.
            y_diff_elbow = abs(elbow[1] - shldr[1])
            
            if y_diff_elbow < PARALLEL_THRESH:
                state = "ACTIVE"
                
                # M4: Base Swing (Mirroring X coordinate using WRIST)
                rel_x = shldr[0] - wrist[0]
                raw_m4 = map_val(rel_x, -200, 200, 150, 0)
                t_m4 = filter_m4(raw_m4)

                # --- 2. FORESHORTENING & ELBOW FIX ---
                # Calculate the 2D pixel lengths of the upper arm and forearm
                len_upper = math.dist(shldr[:2], elbow[:2])
                len_forearm = math.dist(elbow[:2], wrist[:2])
                
                # Failsafe: If pixel lengths are very short, the arm is pointing AT the camera.
                # The math will be pure noise, so we force the human angle to 0.0 (straight).
                if len_upper < 40 or len_forearm < 40:
                    elbow_angle_deg = 0.0 
                else:
                    # Calculate the 2D rotational angle of each arm segment
                    angle_upper = math.atan2(elbow[1] - shldr[1], elbow[0] - shldr[0])
                    angle_forearm = math.atan2(wrist[1] - elbow[1], wrist[0] - elbow[0])
                    
                    # Find the angular deviation (0 = perfectly straight, 140+ = fully folded)
                    angle_diff = math.degrees(angle_forearm - angle_upper)
                    elbow_angle_deg = abs(angle_diff) % 360
                    if elbow_angle_deg > 180:
                        elbow_angle_deg = 360 - elbow_angle_deg
                
                # Map the human fold angle (0 straight -> 140 folded) to the robot (90 straight -> 180 folded)
                raw_m3 = map_val(elbow_angle_deg, 0, 140, 90, 180)
                t_m3 = filter_m3(raw_m3)
            else:
                state = "RESTING"
                # Reset filters gently when dropping to rest state
                filter_m4.val = None
                filter_m3.val = None
        else:
            state = "LOST"

        
        # Send Target to Arduino
        sent_pose = robot.send_target(t_m1, t_m2, t_m3, t_m4, t_m5, t_m6)
        
        # Visuals
        color = (0, 255, 0) if state == "ACTIVE" else (0, 255, 255)
        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Target Sent: {sent_pose}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Kinetic Mirror", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()