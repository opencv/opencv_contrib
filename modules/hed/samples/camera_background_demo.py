import cv2 as cv
import numpy as np
import argparse
import time
import os
import sys
import urllib.request

# Define standard search paths for HED model files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROTO = os.path.join(SCRIPT_DIR, "../../../../opencv/data/deploy.prototxt")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "../../../../opencv/data/hed_pretrained_bsds.caffemodel")

# Fallbacks for absolute paths if running outside the standard build structure
ALT_PROTO = r"n:\dev-stuff\opencvchange\opencv\data\deploy.prototxt"
ALT_MODEL = r"n:\dev-stuff\opencvchange\opencv\data\hed_pretrained_bsds.caffemodel"

# Parse arguments
parser = argparse.ArgumentParser(
    description='Real-time HED vs Canny Virtual Background Replacement Demo. '
                'Compares edge-based semantic segmentation using HED vs Canny.'
)
parser.add_argument('--input', help='Path to video file or image. Skip to use webcam.', default=0)
parser.add_argument('--prototxt', help='Path to deploy.prototxt', default=None)
parser.add_argument('--caffemodel', help='Path to caffemodel', default=None)
parser.add_argument('--width', help='Inference width', default=384, type=int)
parser.add_argument('--height', help='Inference height', default=384, type=int)
args = parser.parse_args()

# -------------------------------------------------------------
# 1. Resolve Model and Background Image Paths
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
        sys.exit(1)

if not model_path:
    if os.path.exists(DEFAULT_MODEL):
        model_path = DEFAULT_MODEL
    elif os.path.exists(ALT_MODEL):
        model_path = ALT_MODEL
    else:
        print(f"Error: Caffemodel not found at {DEFAULT_MODEL} or {ALT_MODEL}")
        sys.exit(1)

bg_save_path = os.path.join(SCRIPT_DIR, "background.jpg")

def download_background(save_path):
    # Public URLs of beautiful high-res background images (office room/beach)
    urls = [
        "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=800&q=80", # Office Room
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80"  # Beach
    ]
    if os.path.exists(save_path):
        print(f"[INFO] Virtual background image found locally: {save_path}")
        return True
    
    print("[INFO] Downloading virtual background image from Unsplash...")
    for url in urls:
        try:
            # Set a user-agent to prevent HTTP 403 Forbidden from some CDNs
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[SUCCESS] Background saved to: {save_path}")
            return True
        except Exception as e:
            print(f"[WARNING] Failed downloading from {url}: {e}")
            
    print("[WARNING] Could not download background image. Falling back to generated virtual pattern.")
    return False

def generate_fallback_background(h, w):
    # Generates a professional vertical gradient background with a grid overlay
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = 0
        g = int(140 * (y / h))
        b = int(220 * (1.0 - y / h))
        background[y, :, :] = [b, g, r] # BGR format
    # Add thin digital gridlines
    for x in range(0, w, 40):
        cv.line(background, (x, 0), (x, h), (75, 75, 75), 1)
    for y in range(0, h, 40):
        cv.line(background, (0, y), (w, y), (75, 75, 75), 1)
    return background

# Ensure background image is present
has_bg_file = download_background(bg_save_path)

# -------------------------------------------------------------
# 2. Register Custom Caffe Crop Layer (Required for CPU inference)
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

# Register standard Caffe Crop Layer
cv.dnn_registerLayer('Crop', CropLayer)

# -------------------------------------------------------------
# 3. Load Model and Initialize Video Stream
# -------------------------------------------------------------
print("[INFO] Loading HED network...")
net = cv.dnn.readNet(proto_path, model_path)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

source = args.input
if isinstance(source, str) and source.isdigit():
    source = int(source)

cap = cv.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source: {args.input}")
    sys.exit(1)

# Default operational parameters
inf_w, inf_h = args.width, args.height
hed_edge_threshold = 0.12  # Lower threshold for better neck/jaw boundary detection
canny_thresh1 = 50
canny_thresh2 = 120

# Window configuration
window_name = "Capstone Demo: Edge-guided Virtual Background replacement (HED vs Canny)"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 1280, 960)

print("\n" + "="*50)
print("INTERACTIVE KEYBOARD CONTROLS:")
print("  [q] : Quit demo")
print("  [+ / -] : Adjust HED edge threshold (currently {:.2f})".format(hed_edge_threshold))
print("  [s] : Save screenshot")
print("="*50 + "\n")

frame_count = 0
fps_start_time = time.time()
fps = 0.0

# Read initial frame to get sizes and load virtual background
ret, frame = cap.read()
if not ret:
    print("[ERROR] Failed to read from camera.")
    sys.exit(1)
    
h_orig, w_orig = frame.shape[:2]

if has_bg_file:
    bg_img = cv.imread(bg_save_path)
    if bg_img is None:
        bg_img = generate_fallback_background(h_orig, w_orig)
    else:
        bg_img = cv.resize(bg_img, (w_orig, h_orig))
else:
    bg_img = generate_fallback_background(h_orig, w_orig)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # -------------------------------------------------------------
    # A. Canny Background Separation
    # -------------------------------------------------------------
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, canny_thresh1, canny_thresh2)
    
    # 1. Close gaps in Canny edges using morphological closing
    canny_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    canny_closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, canny_kernel)
    canny_dilated = cv.dilate(canny_closed, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    
    # 2. Draw border lines ONLY at left, right, and bottom (NOT top)
    # This seals the open shoulder boundaries at the borders without enclosing the background
    cv.line(canny_dilated, (0, h_orig - 1), (w_orig - 1, h_orig - 1), 255, 5)
    cv.line(canny_dilated, (0, 0), (0, h_orig - 1), 255, 5)
    cv.line(canny_dilated, (w_orig - 1, 0), (w_orig - 1, h_orig - 1), 255, 5)
    
    # 3. Find external contours. Since the top is open, the background is not enclosed.
    # The only major closed external contour will be the user's silhouette.
    canny_contours, _ = cv.findContours(canny_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    canny_mask_clean = np.zeros_like(canny)
    if canny_contours:
        # Keep only the single largest external contour (the user's silhouette)
        largest_canny = max(canny_contours, key=cv.contourArea)
        # Filling this external contour completely fills the user's interior, ignoring internal facial edges
        cv.drawContours(canny_mask_clean, [largest_canny], -1, 255, -1)
        
    # 4. Smooth mask & blend
    canny_mask_smooth = cv.GaussianBlur(canny_mask_clean, (25, 25), 0) / 255.0
    canny_mask_smooth = np.expand_dims(canny_mask_smooth, axis=2)
    canny_blended = (frame * canny_mask_smooth + bg_img * (1.0 - canny_mask_smooth)).astype(np.uint8)
    
    # -------------------------------------------------------------
    # B. HED Background Separation
    # -------------------------------------------------------------
    # Create input blob
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
    inf_time = (time.time() - inf_start) * 1000.0
    
    # Resize raw probability output back to original camera size
    hed_map = hed_out[0, 0]
    hed_resized = cv.resize(hed_map, (w_orig, h_orig))
    
    # Binarize the probability map to get boundaries
    hed_edges = (hed_resized > hed_edge_threshold).astype(np.uint8) * 255
    hed_edges_bgr = cv.cvtColor(hed_edges, cv.COLOR_GRAY2BGR)
    
    # 1. Close gaps in HED semantic boundaries
    hed_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    hed_closed = cv.morphologyEx(hed_edges, cv.MORPH_CLOSE, hed_kernel)
    hed_dilated = cv.dilate(hed_closed, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    
    # 2. Draw border lines ONLY at left, right, and bottom (NOT top)
    # This seals the open shoulder boundaries at the borders without enclosing the background
    cv.line(hed_dilated, (0, h_orig - 1), (w_orig - 1, h_orig - 1), 255, 5)
    cv.line(hed_dilated, (0, 0), (0, h_orig - 1), 255, 5)
    cv.line(hed_dilated, (w_orig - 1, 0), (w_orig - 1, h_orig - 1), 255, 5)
    
    # 3. Find external contours. Since the top is open, the background is not enclosed.
    # The only major closed external contour will be the user's silhouette.
    hed_contours, _ = cv.findContours(hed_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    hed_mask_clean = np.zeros_like(hed_edges)
    if hed_contours:
        # Keep only the single largest external contour (the user's silhouette)
        largest_hed = max(hed_contours, key=cv.contourArea)
        # Filling this external contour completely fills the user's interior (face, shirt, dress),
        # ignoring internal facial/clothing texture edges and preventing transparency issues.
        cv.drawContours(hed_mask_clean, [largest_hed], -1, 255, -1)
    
    # 4. Smooth mask & blend
    hed_mask_smooth = cv.GaussianBlur(hed_mask_clean, (25, 25), 0) / 255.0
    hed_mask_smooth = np.expand_dims(hed_mask_smooth, axis=2)
    hed_blended = (frame * hed_mask_smooth + bg_img * (1.0 - hed_mask_smooth)).astype(np.uint8)
    
    # -------------------------------------------------------------
    # C. Build 2x2 Grid View
    # -------------------------------------------------------------
    font = cv.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    
    # Panel 1: Original image + Stats overlay
    p1 = frame.copy()
    stats = [
        f"Resolution: {w_orig}x{h_orig}",
        f"HED Size: {inf_w}x{inf_h}",
        f"HED Latency: {inf_time:.1f} ms",
        f"Frame Rate: {fps:.1f} FPS"
    ]
    for idx, stat in enumerate(stats):
        cv.putText(p1, stat, (15, 30 + idx * 25), font, 0.6, bg_color, 4, cv.LINE_AA)
        cv.putText(p1, stat, (15, 30 + idx * 25), font, 0.6, text_color, 15, cv.LINE_AA)
        cv.putText(p1, stat, (15, 30 + idx * 25), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(p1, "[1] Original Camera Feed", (15, h_orig - 15), font, 0.7, (0, 255, 255), 2)
    
    # Panel 2: Raw HED Edges
    p2 = hed_edges_bgr.copy()
    cv.putText(p2, f"[2] HED Semantic Boundaries (Thresh: {hed_edge_threshold:.2f})", (15, h_orig - 15), font, 0.7, (0, 255, 255), 2)
    
    # Panel 3: Canny BG replacement (shows noise and background leaks)
    p3 = canny_blended.copy()
    cv.putText(p3, "[3] Canny Separation (Noisy Contours)", (15, h_orig - 15), font, 0.7, (0, 0, 255), 2)
    
    # Panel 4: HED BG replacement (clean semantic contour separation)
    p4 = hed_blended.copy()
    cv.putText(p4, "[4] HED Separation (Semantic Contours)", (15, h_orig - 15), font, 0.7, (0, 255, 0), 2)
    
    # Assemble grid
    top_row = np.hstack((p1, p2))
    bottom_row = np.hstack((p3, p4))
    grid = np.vstack((top_row, bottom_row))
    
    # Scale grid for display
    display_scale = 0.75
    h_grid, w_grid = grid.shape[:2]
    grid_resized = cv.resize(grid, (int(w_grid * display_scale), int(h_grid * display_scale)))
    
    cv.imshow(window_name, grid_resized)
    
    # Calculate FPS
    frame_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start_time = time.time()
        
    # Keyboard interaction
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        hed_edge_threshold = min(0.95, hed_edge_threshold + 0.05)
        print(f"[ACTION] Increased HED edge binarization threshold to: {hed_edge_threshold:.2f}")
    elif key == ord('-') or key == ord('_'):
        hed_edge_threshold = max(0.01, hed_edge_threshold - 0.05)
        print(f"[ACTION] Decreased HED edge binarization threshold to: {hed_edge_threshold:.2f}")
    elif key == ord('s'):
        filename = f"bg_separation_comparison_{int(time.time())}.png"
        cv.imwrite(filename, grid)
        print(f"[ACTION] Saved comparison grid screenshot: {filename}")

cap.release()
cv.destroyAllWindows()
