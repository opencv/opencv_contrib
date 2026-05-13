# OpenCV SLAM Module - Running Examples Guide

This guide explains how to compile and run the SLAM module example programs.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Preparing Test Data](#2-preparing-test-data)
3. [Compiling Samples](#3-compiling-samples)
4. [Running Examples](#4-running-examples)
5. [Expected Output](#5-expected-output)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### 1.1 Installed Components

Ensure the following are installed:

```bash
# Check OpenCV with SLAM module
ls -la /usr/local/lib/libcv_slam.so

# Check FBoW vocabulary
ls -la /usr/local/share/fbow/

# Check test data
ls -la /path/to/opencv_contrib_slam/modules/slam/testdata/
```

### 1.2 Required Files

You need three things to run the examples:

| File | Description | Location |
|------|-------------|----------|
| **Config YAML** | Camera parameters | `modules/slam/testdata/config/` |
| **Vocabulary** | BoW vocabulary (ORB) | `modules/slam/testdata/vocab/orb_vocab.fbow` |
| **Image Sequence** | EuRoC format images | Download from EuRoC dataset |

### 1.3 Download EuRoC Dataset

The examples use the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):

```bash
# Download MH01 dataset (smallest, ~1GB)
mkdir -p ~/datasets/EuRoC
cd ~/datasets/EuRoC

# Option 1: Download directly
wget https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_dataset/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip

# Option 2: Use rsync from server
rsync -av --progress robot@asrl.eng.cam.ac.uk::asrl datasets/ijrr_euroc_dataset/
```

---

## 2. Preparing Test Data

### 2.1 Check Existing Test Data

```bash
# List available config files
ls modules/slam/testdata/config/

# List available vocabulary
ls modules/slam/testdata/vocab/
```

### 2.2 Download ORB Vocabulary (if missing)

If `orb_vocab.fbow` is not present:

```bash
# Download from FBoW repository
wget https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow
mv orb_vocab.fbow modules/slam/testdata/vocab/
```

### 2.3 Dataset Directory Structure

Expected structure:
```
EuRoC/
└── MH_01_easy/
    └── mav0/
        └── cam0/
            ├── data/
            │   ├── 1403636579763555584.png
            │   ├── 1403636579813555584.png
            │   └── ...
            └── data.csv
```

---

## 3. Compiling Samples

### 3.1 Using the Standalone CMakeLists.txt

The `samples/cpp/CMakeLists.txt` is configured for your local paths:

```bash
cd modules/slam/samples/cpp

# Edit CMakeLists.txt to update paths if needed:
# - OPENCV_BUILD_DIR: Path to your OpenCV build directory
# - OPENCV_CONTRIB_SLAM_DIR: Path to opencv_contrib_slam

# Create build directory
mkdir -p build
cd build

# Configure
cmake ..

# Compile
make -j$(nproc)
```

### 3.2 Manual Compilation

If CMake fails, compile manually:

```bash
cd modules/slam/samples/cpp

g++ -std=c++17 \
    -I/home/user/opencv_contrib_slam/modules/slam/include \
    -I/home/user/opencv/modules/core/include \
    -I/home/user/opencv/modules/imgcodecs/include \
    -I/home/user/opencv/modules/features2d/include \
    -I/home/user/opencv/modules/calib3d/include \
    -I/home/user/opencv/modules/highgui/include \
    -I/home/user/build-opencv \
    -L/home/user/build-opencv/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_features2d \
    -lopencv_calib3d -lopencv_highgui -lcv_slam \
    full_slam.cpp -o example_full_slam
```

### 3.3 Build Directory Contents

After compilation, you should have:

```
modules/slam/samples/cpp/build/
├── example_full_slam       # Full SLAM pipeline
├── example_localization    # Localization mode (if exists)
├── example_map_save_load   # Map persistence test (if exists)
└── ...
```

---

## 4. Running Examples

### 4.1 Full SLAM Pipeline

Run complete SLAM on an image sequence:

```bash
cd modules/slam/samples/cpp/build

./example_full_slam \
    ../../testdata/config/euroc_mh01.yaml \
    ../../testdata/vocab/orb_vocab.fbow \
    /path/to/EuRoC/MH_01_easy/mav0/cam0/data \
    /tmp/slam_output
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `config.yaml` | Camera configuration file |
| `vocab.fbow` | BoW vocabulary file |
| `image_dir` | Directory containing images |
| `output_dir` | Directory for output files |

### 4.2 Frontend Only (Fast Test)

Run visual odometry frontend without backend optimization:

```bash
./run_frontend_mh01 \
    ../../testdata/config/euroc_mh01.yaml \
    ../../testdata/vocab/orb_vocab.fbow \
    /path/to/EuRoC/MH_01_easy/mav0/cam0/data \
    /tmp/frontend_output
```

### 4.3 Localization Mode

Load existing map and localize within it:

```bash
./example_localization_mode \
    ../../testdata/config/euroc_mh01.yaml \
    ../../testdata/vocab/orb_vocab.fbow \
    /tmp/slam_output/map.msgpack \
    /path/to/EuRoC/MH_01_easy/mav0/cam0/data
```

### 4.4 Map Save/Load Test

Build a map, save it, then load for localization:

```bash
# Step 1: Build and save map
./example_full_slam \
    ../../testdata/config/euroc_mh01.yaml \
    ../../testdata/vocab/orb_vocab.fbow \
    /path/to/EuRoC/MH_01_easy/mav0/cam0/data \
    /tmp/slam_output

# Step 2: Load and localize (requires localization_mode example)
./example_localization_mode \
    ../../testdata/config/euroc_mh01.yaml \
    ../../testdata/vocab/orb_vocab.fbow \
    /tmp/slam_output/map.msgpack \
    /path/to/EuRoC/MH_01_easy/mav0/cam0/data
```

---

## 5. Expected Output

### 5.1 Full SLAM Output

```
Loading images from: /path/to/EuRoC/MH_01_easy/mav0/cam0/data
Found 2341 images

Initializing SLAM system...
Processing 2341 frames...
Frame 0/2341 (tracked: 1, 45.2 ms)
Frame 100/2341 (tracked: 98, 23.1 ms)
Frame 200/2341 (tracked: 195, 21.8 ms)
...

=== SLAM Statistics ===
Tracked frames: 2320/2341 (99.1%)
Average FPS: 42.3
Average processing time: 23.6 ms

Saving map to: /tmp/slam_output/map.msgpack
Map saved successfully
Saving trajectory to: /tmp/slam_output/trajectory.txt
Trajectory saved successfully
Map contains 15420 3D points

Shutting down SLAM system...
Done!
```

### 5.2 Output Files

| File | Description | Format |
|------|-------------|--------|
| `map.msgpack` | Serialized map | MessagePack binary |
| `trajectory.txt` | Camera poses | TUM trajectory format |

### 5.3 Trajectory Format (TUM)

```
# timestamp tx ty tz qx qy qz qw
1403636579.763555 0.0012 -0.0023 0.0156 0.001 -0.002 0.003 0.999
1403636579.823555 0.0014 -0.0021 0.0158 0.002 -0.001 0.002 0.999
...
```

---

## 6. Troubleshooting

### 6.1 "No images found"

```bash
# Check image directory
ls /path/to/EuRoC/MH_01_easy/mav0/cam0/data/*.png | head

# If using data.csv, ensure it's present
cat /path/to/EuRoC/MH_01_easy/mav0/cam0/data.csv | head
```

### 6.2 "Vocabulary file not found"

```bash
# Verify vocabulary exists
ls -la modules/slam/testdata/vocab/orb_vocab.fbow

# Or download
wget https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow
```

### 6.3 "Library not found" / "undefined symbol"

```bash
# Update library cache
sudo ldconfig

# Set library path if needed
export LD_LIBRARY_PATH=/path/to/build-opencv/lib:$LD_LIBRARY_PATH
```

### 6.4 Low tracking rate / "tracking lost"

Possible causes:
- Motion blur in images
- Poor lighting
- Fast rotation
- Insufficient texture

Solutions:
- Use a higher quality dataset (e.g., MH_01 instead of MH_05)
- Reduce feature threshold: Edit config.yaml to increase `Features.max_features`
- Enable backend optimization: `slam->setBackendEnabled(true, 10)`

### 6.5 Config file not found

```bash
# Use absolute path
./example_full_slam \
    /full/path/to/opencv_contrib_slam/modules/slam/testdata/config/euroc_mh01.yaml \
    ...
```

---

## Configuration File Format

Create your own `camera.yaml`:

```yaml
%YAML:1.0
Camera.name: "EuRoC monocular"
Camera.setup: monocular        # monocular, stereo, rgbd
Camera.model: perspective      # perspective, fisheye, equirectangular
Camera.fx: 458.654
Camera.fy: 457.296
Camera.cx: 367.215
Camera.cy: 248.375
Camera.k1: -0.28340811
Camera.k2: 0.07395907
Camera.p1: 0.0
Camera.p2: 0.0
Camera.fps: 20.0
Camera.width: 752
Camera.height: 480
```

---

## Quick Start Summary

```bash
# 1. Build the example
cd modules/slam/samples/cpp/build
make

# 2. Run full SLAM
./example_full_slam \
    ../../testdata/config/euroc_mh01.yaml \
    ../../testdata/vocab/orb_vocab.fbow \
    ~/datasets/EuRoC/MH_01_easy/mav0/cam0/data \
    /tmp/output

# 3. Check results
ls -la /tmp/output/
cat /tmp/output/trajectory.txt | head
```

---

For more information, see:
- [INSTALL.md](./INSTALL.md) - Installation guide
- [README.md](./README.md) - Module documentation
- [BENCHMARK.md](./BENCHMARK.md) - Performance benchmarks
