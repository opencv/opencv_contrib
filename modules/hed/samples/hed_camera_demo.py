import cv2 as cv
import numpy as np
import argparse
import time
import os
import sys

# Define standard search paths for HED model files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROTO = os.path.join(SCRIPT_DIR, "../../../../opencv/data/deploy.prototxt")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "../../../../opencv/data/hed_pretrained_bsds.caffemodel")

# Fallbacks for absolute paths if running outside the standard build structure
ALT_PROTO = r"n:\dev-stuff\opencvchange\opencv\data\deploy.prototxt"
ALT_MODEL = r"n:\dev-stuff\opencvchange\opencv\data\hed_pretrained_bsds.caffemodel"

# Parse arguments
parser = argparse.ArgumentParser(
    description='Real-time Holistically-Nested Edge Detection (HED) webcam demo on CPU. '
                'Provides interactive GUI controls and comparative visualization.'
)
parser.add_argument('--input', help='Path to video file or image. Skip to use webcam.', default=0)
parser.add_argument('--prototxt', help='Path to deploy.prototxt', default=None)
parser.add_argument('--caffemodel', help='Path to caffemodel', default=None)
parser.add_argument('--width', help='Initial inference width', default=384, type=int)
parser.add_argument('--height', help='Initial inference height', default=384, type=int)
args = parser.parse_args()

# -------------------------------------------------------------
# 1. Resolve Model Paths
# -------------------------------------------------------------
proto_path = args.prototxt
model_path = args.caffemodel

if not proto_path:
    if os.path.exists(DEFAULT_PROTO):
        proto_path = DEFAULT_PROTO
    elif os.path.exists(ALT_PROTO):
        proto_path = ALT_PROTO
    else:
        print(f"Error: Prototxt not found at {DEFAULT_PROTO} or {ALT_PROTO}")
        print("Please specify path using --prototxt")
        sys.exit(1)

if not model_path:
    if os.path.exists(DEFAULT_MODEL):
        model_path = DEFAULT_MODEL
    elif os.path.exists(ALT_MODEL):
        model_path = ALT_MODEL
    else:
        print(f"Error: Caffemodel not found at {DEFAULT_MODEL} or {ALT_MODEL}")
        print("Please specify path using --caffemodel")
        sys.exit(1)

print(f"[INFO] Using Prototxt:   {proto_path}")
print(f"[INFO] Using Caffemodel: {model_path}")

# -------------------------------------------------------------
# 2. Register Custom Caffe Crop Layer (CPU target)
# -------------------------------------------------------------
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

# Register the crop layer with OpenCV's DNN engine
cv.dnn_registerLayer('Crop', CropLayer)

# -------------------------------------------------------------
# 3. Vectorized Non-Maximum Suppression for Edge Thinning
# -------------------------------------------------------------
def edge_thinning_nms(edge_map, threshold=0.1):
    """
    Applies Non-Maximum Suppression (NMS) on a soft edge probability map
    by calculating local gradient directions and suppressing non-peak pixels.
    """
    # Compute gradients along x and y directions
    dx = cv.Sobel(edge_map, cv.CV_32F, 1, 0, ksize=3)
    dy = cv.Sobel(edge_map, cv.CV_32F, 0, 1, ksize=3)
    
    # Compute angles and convert to degrees
    angle = np.arctan2(dy, dx) * (180.0 / np.pi)
    angle[angle < 0] += 180
    
    h, w = edge_map.shape
    nms = np.zeros((h, w), dtype=np.float32)
    
    # Pad image to simplify neighborhood access at boundaries
    padded = np.pad(edge_map, 1, mode='constant', constant_values=0)
    
    # Slices for 8-neighborhood
    N  = padded[0:-2, 1:-1]
    S  = padded[2:, 1:-1]
    E  = padded[1:-1, 2:]
    W  = padded[1:-1, 0:-2]
    NE = padded[0:-2, 2:]
    SW = padded[2:, 0:-2]
    NW = padded[0:-2, 0:-2]
    SE = padded[2:, 2:]
    
    # Determine the gradient sector
    mask0   = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle <= 180))
    mask45  = (angle >= 22.5) & (angle < 67.5)
    mask90  = (angle >= 67.5) & (angle < 112.5)
    mask135 = (angle >= 112.5) & (angle < 157.5)
    
    keep = np.zeros_like(edge_map, dtype=bool)
    
    # Sector 0 (Horizontal gradient: check East/West)
    keep |= mask0 & (edge_map >= W) & (edge_map >= E)
    # Sector 1 (45-degree gradient: check NorthWest/SouthEast)
    keep |= mask45 & (edge_map >= NW) & (edge_map >= SE)
    # Sector 2 (Vertical gradient: check North/South)
    keep |= mask90 & (edge_map >= N) & (edge_map >= S)
    # Sector 3 (135-degree gradient: check NorthEast/SouthWest)
    keep |= mask135 & (edge_map >= NE) & (edge_map >= SW)
    
    nms[keep] = edge_map[keep]
    
    # Apply soft thresholding
    if threshold > 0:
        nms[nms < threshold] = 0
        
    return nms

# -------------------------------------------------------------
# 4. Initialize Network (CPU mode)
# -------------------------------------------------------------
print("[INFO] Loading network model (Running on CPU)...")
net = cv.dnn.readNet(proto_path, model_path)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Start camera / video capture
print("[INFO] Initializing video stream...")
source = args.input
if isinstance(source, str) and source.isdigit():
    source = int(source)

cap = cv.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source: {args.input}")
    sys.exit(1)

# Default operational parameters
inf_w, inf_h = args.width, args.height
apply_nms = True
nms_threshold = 0.15
canny_thresh1 = 100
canny_thresh2 = 200

# Window configuration
window_name = "Capstone Demo: HED Real-time comparative analysis (CPU)"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 1280, 960)

print("\n" + "="*50)
print("INTERACTIVE KEYBOARD CONTROLS:")
print("  [q] : Quit demo")
print("  [r] : Cycle inference resolution (256 -> 384 -> 500)")
print("  [t] : Toggle Edge Thinning (NMS)")
print("  [+ / -] : Increase/Decrease edge thinning threshold")
print("  [s] : Save a comparison screenshot")
print("="*50 + "\n")

frame_count = 0
fps_start_time = time.time()
fps = 0.0

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video stream.")
        break
        
    h_orig, w_orig = frame.shape[:2]
    
    # -------------------------------------------------------------
    # A. Classic Canny Edge Detection
    # -------------------------------------------------------------
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, canny_thresh1, canny_thresh2)
    # Convert Canny output to BGR for display concatenation
    canny_bgr = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
    
    # -------------------------------------------------------------
    # B. HED Inference via OpenCV DNN (CPU)
    # -------------------------------------------------------------
    # Create input blob with VGG BGR training means
    blob = cv.dnn.blobFromImage(
        frame, 
        scalefactor=1.0, 
        size=(inf_w, inf_h),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, 
        crop=False
    )
    
    net.setInput(blob)
    
    inf_start = time.time()
    hed_out = net.forward()
    inf_time = (time.time() - inf_start) * 1000.0  # Milliseconds
    
    # Post-process raw network output (shape: [1, 1, H, W])
    hed_map = hed_out[0, 0]
    
    # Resize output back to original camera frame resolution
    hed_resized = cv.resize(hed_map, (w_orig, h_orig))
    
    # Scale from float [0, 1] to uint8 [0, 255]
    hed_u8 = (hed_resized * 255).astype(np.uint8)
    hed_bgr = cv.cvtColor(hed_u8, cv.COLOR_GRAY2BGR)
    
    # -------------------------------------------------------------
    # C. HED Edge Thinning (NMS)
    # -------------------------------------------------------------
    if apply_nms:
        # Perform NMS on the resized float map
        thinned = edge_thinning_nms(hed_resized, threshold=nms_threshold)
        thinned_u8 = (thinned * 255).astype(np.uint8)
        thinned_bgr = cv.cvtColor(thinned_u8, cv.COLOR_GRAY2BGR)
    else:
        # If disabled, show a blank or bypass image
        thinned_bgr = np.zeros_like(frame)
        cv.putText(thinned_bgr, "NMS Thinning Disabled", (w_orig // 4, h_orig // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # -------------------------------------------------------------
    # D. Overlay Text & Assemble comparative 2x2 Grid
    # -------------------------------------------------------------
    font = cv.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    
    # 1. Panel 1: Original Image + Live Stats
    orig_annotated = frame.copy()
    stats = [
        f"Input: {w_orig}x{h_orig}",
        f"Inference Res: {inf_w}x{inf_h}",
        f"Active Device: CPU",
        f"DNN Latency: {inf_time:.1f} ms",
        f"Frame Rate: {fps:.1f} FPS"
    ]
    for idx, stat in enumerate(stats):
        cv.putText(orig_annotated, stat, (15, 30 + idx * 25), font, 0.6, bg_color, 4, cv.LINE_AA)
        cv.putText(orig_annotated, stat, (15, 30 + idx * 25), font, 0.6, text_color, 15, cv.LINE_AA)
        cv.putText(orig_annotated, stat, (15, 30 + idx * 25), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        
    cv.putText(orig_annotated, "[1] Original Camera Feed", (15, h_orig - 15), font, 0.7, (0, 255, 255), 2)
    
    # 2. Panel 2: Classic Canny
    cv.putText(canny_bgr, "[2] Canny Edge Detection", (15, h_orig - 15), font, 0.7, (0, 255, 255), 2)
    
    # 3. Panel 3: Raw HED
    cv.putText(hed_bgr, "[3] Raw HED (Probability Map)", (15, h_orig - 15), font, 0.7, (0, 255, 255), 2)
    
    # 4. Panel 4: Thinned HED
    cv.putText(thinned_bgr, f"[4] Thinned HED (NMS Thresh: {nms_threshold:.2f})", (15, h_orig - 15), font, 0.7, (0, 255, 255), 2)
    
    # Construct 2x2 Grid
    top_row = np.hstack((orig_annotated, canny_bgr))
    bottom_row = np.hstack((hed_bgr, thinned_bgr))
    grid = np.vstack((top_row, bottom_row))
    
    # Scale grid down if it exceeds a typical monitor's resolution
    display_scale = 0.75
    h_grid, w_grid = grid.shape[:2]
    grid_resized = cv.resize(grid, (int(w_grid * display_scale), int(h_grid * display_scale)))
    
    cv.imshow(window_name, grid_resized)
    
    # Calculate global FPS
    frame_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start_time = time.time()
        
    # -------------------------------------------------------------
    # E. Key Handling
    # -------------------------------------------------------------
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Exiting...")
        break
        
    elif key == ord('r'):
        # Cycle resolutions: 256 -> 384 -> 500
        if inf_w == 256:
            inf_w, inf_h = 384, 384
        elif inf_w == 384:
            inf_w, inf_h = 500, 500
        else:
            inf_w, inf_h = 256, 256
        print(f"[ACTION] Changed network inference size to: {inf_w}x{inf_h}")
        
    elif key == ord('t'):
        apply_nms = not apply_nms
        print(f"[ACTION] Toggled NMS Edge Thinning: {apply_nms}")
        
    elif key == ord('+') or key == ord('='):
        nms_threshold = min(0.95, nms_threshold + 0.05)
        print(f"[ACTION] Increased NMS threshold to: {nms_threshold:.2f}")
        
    elif key == ord('-') or key == ord('_'):
        nms_threshold = max(0.01, nms_threshold - 0.05)
        print(f"[ACTION] Decreased NMS threshold to: {nms_threshold:.2f}")
        
    elif key == ord('s'):
        filename = f"comparison_screenshot_{int(time.time())}.png"
        cv.imwrite(filename, grid)
        print(f"[ACTION] Saved full screenshot as: {filename}")

# Cleanup
cap.release()
cv.destroyAllWindows()
