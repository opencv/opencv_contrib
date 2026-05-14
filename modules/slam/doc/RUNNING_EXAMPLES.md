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

### 1.1 Built with Samples Enabled

When building OpenCV with SLAM module, ensure `BUILD_EXAMPLES=ON`:

```bash
cmake -D BUILD_EXAMPLES=ON -D BUILD_TESTS=ON ...
make -j$(nproc)
sudo make install
```

This will compile SLAM samples as `example_slam_*` targets.

### 1.2 Required Files

You need three things to run the examples:

| File | Description | Location |
|------|-------------|----------|
| **Config YAML** | Camera parameters | `samples/data/config/` |
| **Vocabulary** | BoW vocabulary (ORB) | `samples/data/vocab/orb_vocab.fbow` |
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
ls opencv_contrib_slam/modules/slam/samples/data/config/

# List available vocabulary
ls opencv_contrib_slam/modules/slam/samples/data/vocab/
```

### 2.2 Download ORB Vocabulary (if missing)

If `orb_vocab.fbow` is not present:

```bash
# Download from FBoW repository
wget https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow
mv orb_vocab.fbow opencv_contrib_slam/modules/slam/samples/data/vocab/
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

### 3.1 Standard OpenCV Sample Build

When you build OpenCV with `BUILD_EXAMPLES=ON`, SLAM module samples are automatically compiled:

```bash
cd ~/opencv/build

# Build all examples (including SLAM)
make -j$(nproc)

# Or build only SLAM examples
make example_slam_full_slam
make example_slam_localization
make example_slam_frontend
```

### 3.2 Build Directory Contents

After compilation, SLAM examples are located in the build directory:

```
~/opencv/build/bin/
├── example_slam_full_slam      # Full SLAM pipeline
├── example_slam_localization   # Localization mode
├── example_slam_frontend       # Frontend test
└── ...
```

---

## 4. Running Examples

### 4.1 Full SLAM Pipeline

Run complete SLAM on an image sequence:

```bash
cd ~/opencv/build/bin

./example_slam_full_slam \
    ../samples/data/config/euroc_mh01.yaml \
    ../samples/data/vocab/orb_vocab.fbow \
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
cd ~/opencv/build/bin

# Step 1: Build and save map
./example_slam_full_slam \
    ../samples/data/config/euroc_mh01.yaml \
    ../samples/data/vocab/orb_vocab.fbow \
    /path/to/EuRoC/MH_01_easy/mav0/cam0/data \
    /tmp/slam_output

# Step 2: Load and localize
./example_slam_localization \
    ../samples/data/config/euroc_mh01.yaml \
    ../samples/data/vocab/orb_vocab.fbow \
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
ls -la opencv_contrib_slam/modules/slam/samples/data/vocab/orb_vocab.fbow

# Or download
wget https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow
```

### 6.3 "Library not found" / "undefined symbol"

```bash
# Update library cache
sudo ldconfig

# Set library path if needed
export LD_LIBRARY_PATH=~/opencv/build/lib:$LD_LIBRARY_PATH
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
# 1. Build OpenCV with SLAM examples enabled
cmake -D BUILD_EXAMPLES=ON ... && make -j$(nproc)

# 2. Run full SLAM
cd ~/opencv/build/bin
./example_slam_full_slam \
    ../samples/data/config/euroc_mh01.yaml \
    ../samples/data/vocab/orb_vocab.fbow \
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
