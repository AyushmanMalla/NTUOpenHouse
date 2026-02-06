import cv2
import numpy as np
import vart
import xir
import time
import os
import shutil

# --- CONFIGURATION ---
MODEL_PATH = "/home/ubuntu/Kria-PYNQ/movenet_kr260_vai25.xmodel"
CAMERA_ID = 0
INPUT_W, INPUT_H = 192, 192
GRID_W, GRID_H = 48, 48
STRIDE = INPUT_W / GRID_W  # Should be 4
NUM_KEYPOINTS = 17
OUTPUT_DIR = "output_frames"
FRAMES_TO_CAPTURE = 50  # Capture 50 frames for the test

def get_dpu_subgraph(graph):
    """
    Recursively find the subgraph designated for the DPU.
    """
    root = graph.get_root_subgraph()
    
    # Helper function to traverse children
    def find_dpu(subgraph):
        if subgraph.has_attr("device"):
            if subgraph.get_attr("device").upper() == "DPU":
                return subgraph
        
        for child in subgraph.get_children():
            result = find_dpu(child)
            if result is not None:
                return result
        return None
    return find_dpu(root)

def preprocess(frame, fix_point):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) / 127.5
    scale = 2 ** fix_point
    img = img * scale
    return np.expand_dims(img.astype(np.int8), axis=0)

def decode_keypoints(heatmaps, regs, scale_y, scale_x):
    keypoints = []
    for k in range(NUM_KEYPOINTS):
        heatmap_channel = heatmaps[:, :, k]
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap_channel)
        grid_x, grid_y = max_loc
        
        reg_y = regs[grid_y, grid_x, 2*k]
        reg_x = regs[grid_y, grid_x, 2*k + 1]
        
        final_y = (grid_y + reg_y) * STRIDE
        final_x = (grid_x + reg_x) * STRIDE
        
        screen_y = int(final_y * scale_y)
        screen_x = int(final_x * scale_x)
        keypoints.append((screen_x, screen_y, max_val))
    return keypoints

# Modified "Stable" Decoder
# def decode_keypoints_coarse(heatmaps, scale_y, scale_x):
#     keypoints = []
#     for k in range(17):
#         heatmap_channel = heatmaps[:, :, k]
#         _, max_val, _, max_loc = cv2.minMaxLoc(heatmap_channel)
#         grid_x, grid_y = max_loc
        
#         # IGNORE REGS for now. Just map grid to screen.
#         final_y = grid_y * 4 # STRIDE is 4
#         final_x = grid_x * 4 
        
#         screen_y = int(final_y * scale_y)
#         screen_x = int(final_x * scale_x)
#         keypoints.append((screen_x, screen_y, max_val))
#     return keypoints

def draw_skeleton(frame, keypoints):
    EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
        (13, 15), (12, 14), (14, 16)
    ]
    for kp in keypoints:
        x, y, conf = kp
        color = (0, 255, 0) if conf > 0.3 else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color, -1)
        
    for edge in EDGES:
        idx1, idx2 = edge
        x1, y1, c1 = keypoints[idx1]
        x2, y2, c2 = keypoints[idx2]
        if c1 > 0.3 and c2 > 0.3:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

# --- MAIN EXECUTION ---
def main():
    # 1. Setup Output Directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

    # 2. Initialize DPU
    print("Loading model...")
    g = xir.Graph.deserialize(MODEL_PATH)
    runner = vart.Runner.create_runner(get_dpu_subgraph(g), "run")
    
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    in_fix_pos = input_tensors[0].get_attr("fix_point")
    
    heatmap_idx = next(i for i, t in enumerate(output_tensors) if t.dims[-1] == 17)
    regs_idx = next(i for i, t in enumerate(output_tensors) if t.dims[-1] == 34 and "regs" in t.name)
    
    heatmap_scale = 2 ** (-1 * output_tensors[heatmap_idx].get_attr("fix_point"))
    regs_scale = 2 ** (-1 * output_tensors[regs_idx].get_attr("fix_point"))

    # 3. Initialize Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Pre-allocate buffers
    input_data = [np.empty(input_tensors[0].dims, dtype=np.int8)]
    output_data = [np.empty(t.dims, dtype=np.int8) for t in output_tensors]
    
    print(f"Starting capture of {FRAMES_TO_CAPTURE} frames...")
    start_time = time.time()
    
    for i in range(FRAMES_TO_CAPTURE):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        h_raw, w_raw = frame.shape[:2]
        scale_y = h_raw / INPUT_H
        scale_x = w_raw / INPUT_W
        
        # Inference
        input_data[0] = preprocess(frame, in_fix_pos)
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)
        
        # Decode
        heatmaps = output_data[heatmap_idx][0].astype(np.float32) * heatmap_scale
        regs = output_data[regs_idx][0].astype(np.float32) * regs_scale
        keypoints = decode_keypoints(heatmaps, regs, scale_y, scale_x)
        # keypoints = decode_keypoints_coarse(heatmaps, scale_y, scale_x)
        
        # Visualize
        draw_skeleton(frame, keypoints)
        
        # Save Frame
        filename = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
        cv2.imwrite(filename, frame)
        
        if i % 10 == 0:
            print(f"Saved {filename}")

    total_time = time.time() - start_time
    fps = FRAMES_TO_CAPTURE / total_time
    print(f"\nDone! Average FPS: {fps:.2f}")
    print(f"Check the '{OUTPUT_DIR}' folder for your images.")
    
    cap.release()

if __name__ == "__main__":
    main()