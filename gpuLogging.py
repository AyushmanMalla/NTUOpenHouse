import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time
import csv
import math

# --- CONFIGURATION ---
# We use 'Thunder' because it's higher accuracy (like your FPGA target likely is)
# If you want faster but less accurate, change to "lightning" and input size 192
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
INPUT_SIZE = 256 
CONF_THRESH = 0.3
LOG_FILE = "robot_data.csv"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200


class BraccioSender:
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2) # Wait for Arduino reset
            print(f"Connected to Braccio on {port}")
        except:
            print("UART FAIL: Running in Simulation Mode")
            self.ser = None
        self.last_sent = 0
        self.interval = 0.03 # ~30 commands/sec max

    def send(self, m1, m2, m3, m4, m5, m6):
        if time.time() - self.last_sent < self.interval: return
        
        # Clamp Values
        m1 = max(0, min(180, int(m1)))
        m2 = max(15, min(165, int(m2)))
        m3 = max(0, min(180, int(m3)))
        
        # Protocol: P90,45,180,180,90,10\n
        cmd = f"P{m1},{m2},{m3},{m4},{m5},{m6}\n"
        
        if self.ser:
            self.ser.write(cmd.encode())
        self.last_sent = time.time()

class OneEuroFilter:
    def __init__(self, min_cutoff=0.1, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
    

# --- ROBOT MAPPING HELPERS ---
def get_angle(a, b, c):
    """Calculates angle ABC (in degrees)."""
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0: return 0.0
    
    # Clip to avoid numerical errors causing NaN
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def map_val(x, in_min, in_max, out_min, out_max):
    # Clamp input first
    x = max(in_min, min(in_max, x))
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def main():
    print("Loading MoveNet from TensorFlow Hub...")
    model = hub.load(MODEL_URL)
    movenet = model.signatures['serving_default']
    print("Model Loaded.")

    cap = cv2.VideoCapture(0)
    
    # Setup CSV Logging
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Base_Angle", "Shoulder_Angle", "Elbow_Angle", "Wrist_Conf"])

    print("Starting Stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Preprocess
        # Resize and pad to square (MoveNet expects square input)
        # For simplicity in profiling, we just resize (distorts slightly but fine for testing)
        input_image = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = tf.convert_to_tensor(input_image, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)

        # 2. Inference
        outputs = movenet(input_image)
        # Shape: [1, 1, 17, 3] -> (y, x, score)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]

        # 3. Extract & Visualize
        h, w = frame.shape[:2]
        
        # Indices: 6=R_Shoulder, 8=R_Elbow, 10=R_Wrist
        shldr_raw = keypoints[6]
        elbow_raw = keypoints[8]
        wrist_raw = keypoints[10]
        
        # Convert to pixels
        shldr = (int(shldr_raw[1] * w), int(shldr_raw[0] * h), shldr_raw[2])
        elbow = (int(elbow_raw[1] * w), int(elbow_raw[0] * h), elbow_raw[2])
        wrist = (int(wrist_raw[1] * w), int(wrist_raw[0] * h), wrist_raw[2])

        # Draw Skeleton
        if shldr[2] > CONF_THRESH and wrist[2] > CONF_THRESH:
            cv2.line(frame, shldr[:2], elbow[:2], (0, 255, 0), 2)
            cv2.line(frame, elbow[:2], wrist[:2], (0, 255, 0), 2)
            cv2.circle(frame, shldr[:2], 5, (0, 0, 255), -1)
            cv2.circle(frame, wrist[:2], 5, (255, 0, 0), -1)

            # --- VIRTUAL ROBOT LOGIC (Scenario B) ---
            
            # M1: Base (Left/Right)
            # Logic: Relative X position of wrist vs shoulder
            # Wrist 150px Left (-150) -> Robot Right (160 deg)
            # Wrist 150px Right (+150) -> Robot Left (20 deg)
            rel_x = wrist[0] - shldr[0]
            m1_base = map_val(rel_x, -150, 150, 160, 20)

            # M2: Shoulder (Up/Down)
            # Logic: Relative Y position
            # Wrist same height as shoulder (0) -> Arm Up (90 deg)
            # Wrist below shoulder (200) -> Arm Down/Forward (20 deg)
            rel_y = wrist[1] - shldr[1]
            m2_shoulder = map_val(rel_y, -50, 200, 110, 20)

            # M3: Elbow (Extension/Distance)
            # Logic: Euclidean distance
            # Close (50px) -> Bent (0 deg)
            # Far (250px) -> Straight (170 deg)
            dist = math.sqrt(rel_x**2 + rel_y**2)
            m3_elbow = map_val(dist, 50, 250, 10, 170)

            # Display "Virtual Robot" state
            cv2.putText(frame, f"Base (M1): {int(m1_base)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Shldr (M2): {int(m2_shoulder)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Elbow (M3): {int(m3_elbow)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Log to file
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), int(m1_base), int(m2_shoulder), int(m3_elbow), wrist[2]])

        cv2.imshow("Robot Logic Profiler", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved to {LOG_FILE}")

if __name__ == "__main__":
    main()