# OpenCV SLAM Module Installation Guide

This guide provides step-by-step instructions for installing OpenCV with the SLAM module from scratch on Ubuntu 20.04/22.04.

## Table of Contents

1. [System Dependencies](#1-system-dependencies)
2. [Core Dependencies](#2-core-dependencies)
3. [Optional Dependencies](#3-optional-dependencies)
4. [Building OpenCV with SLAM Module](#4-building-opencv-with-slam-module)
5. [Verification](#5-verification)

---

## 1. System Dependencies

```bash
sudo apt update
sudo apt upgrade -y

# Essential build tools
sudo apt install -y build-essential cmake git pkg-config

# GUI and image display dependencies
sudo apt install -y libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libv4l-dev v4l-utils libxvidcore-dev libx264-dev libjpeg-dev libpng-dev
sudo apt install -y libtiff-dev libatlas-base-dev gfortran

# Python development (optional, for Python bindings)
sudo apt install -y python3-dev python3-pip python3-numpy
```

## 2. Core Dependencies

### 2.1 Eigen3 (Required)

Eigen3 >= 3.3 is required for linear algebra operations.

```bash
# Install from Ubuntu repositories (version 3.3.9 on Ubuntu 22.04)
sudo apt install -y libeigen3-dev

# Or install from source for a specific version
cd ~
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install
sudo ldconfig
```

### 2.2 yaml-cpp (Required)

yaml-cpp >= 0.6 is required for configuration file parsing.

```bash
# Install from Ubuntu repositories
sudo apt install -y libyaml-cpp-dev

# Or install from source for newer version
cd ~
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
git checkout yaml-cpp-0.7.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DYAML_BUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 2.3 nlohmann_json (Required)

nlohmann_json >= 3.2 is required for JSON serialization.

```bash
# Install from Ubuntu repositories
sudo apt install -y nlohmann-json3-dev

# Or install from source
cd ~
git clone https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DJSON_BuildTests=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 2.4 SQLite3 (Required)

```bash
sudo apt install -y libsqlite3-dev
```

### 2.5 g2o (Required)

g2o >= 20230223_git is required for graph optimization. The recommended version is specifically **20230223_git**.

```bash
cd ~
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
git checkout 20230223_git
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_EXAMPLES=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 2.6 SuiteSparse and CXSparse (Required for g2o)

```bash
sudo apt install -y libsuitesparse-dev libcxsparse-dev
```

## 3. Optional Dependencies

### 3.1 FBoW (Feature-based Bag of Words) - Default

FBoW is the default BoW framework. Install either FBoW or DBoW2.

```bash
cd ~
git clone https://github.com/rmiquelma/fbow.git
cd fbow
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3.2 DBoW2 (Alternative to FBoW)

If you prefer DBoW2 over FBoW, set `BOW_FRAMEWORK=DBoW2` during CMake configuration.

```bash
cd ~
git clone https://github.com/dorian3d/DBoW2.git
cd DBoW2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3.3 Pangolin (Optional - for visualization GUI)

Pangolin provides visualization capabilities but is optional.

```bash
# Install Pangolin dependencies
sudo apt install -y libglew-dev libxkbcommon-dev libxkbcommon-x11-dev wayland-protocols

cd ~
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PANGOLIN_PYTHON=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3.4 GTSAM (Optional - for advanced backend)

GTSAM provides an alternative backend for optimization. Enable with `-DUSE_GTSAM=ON`.

```bash
# Install GTSAM dependencies
sudo apt install -y libboost-all-dev

cd ~
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout gtsam-4.1.1
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_GTSAM_UNSTABLE=OFF \
    -DGTSAM_BUILD_SHARED=ON
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3.5 OpenMP (Optional - for parallel processing)

```bash
sudo apt install -y libomp-dev
```

### 3.6 ArUCO (Optional - for marker-based localization)

```bash
# ArUCO is part of opencv_contrib, will be built automatically if opencv_aruco is enabled
# No additional installation required
```

### 3.7 CUDA (Optional - for GPU acceleration)

If you have an NVIDIA GPU and want CUDA acceleration:

```bash
# Install CUDA Toolkit (follow NVIDIA's official instructions)
# Then install cuda_efficient_features from: https://github.com/alanlazearistizabal/cuda_efficient_features
cd ~
git clone https://github.com/alanlazearistizabal/cuda_efficient_features.git
cd cuda_efficient_features
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 4. Building OpenCV with SLAM Module

### 4.1 Clone OpenCV and OpenCV Contrib Repositories

```bash
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Checkout compatible versions
cd opencv
git checkout 4.8.0
cd ../opencv_contrib
git checkout 4.8.0
```

### 4.2 Configure with CMake

```bash
cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_EXECUTABLE=$(which python3) \
    ..
```

### 4.3 Enable SLAM Module Specific Options

For full SLAM functionality, additional CMake options are available:

```bash
# Reconfigure with SLAM-specific options
cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=ON \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BOW_FRAMEWORK=FBoW \
    -D USE_GTSAM=OFF \
    -D USE_OPENMP=OFF \
    -D USE_SSE_ORB=OFF \
    -D USE_ARUCO=ON \
    ..
```

**CMake Options for SLAM Module:**

| Option | Default | Description |
|--------|---------|-------------|
| `BOW_FRAMEWORK` | FBoW | BoW framework: FBoW or DBoW2 |
| `USE_GTSAM` | OFF | Enable GTSAM backend |
| `USE_OPENMP` | OFF | Enable OpenMP parallel processing |
| `USE_SSE_ORB` | OFF | Enable SSE3 for ORB extraction |
| `USE_SSE_FP_MATH` | OFF | Enable SSE for floating-point math |
| `USE_CUDA_EFFICIENT_DESCRIPTORS` | OFF | Enable CUDA acceleration |
| `USE_ARUCO` | ON | Enable ArUCO marker support |
| `USE_ARUCO_NANO` | OFF | Enable ArUCO nano support |
| `USE_TRACY` | OFF | Enable Tracy profiler |
| `ENABLE_TRACE_LEVEL_LOG` | OFF | Enable verbose logging |

### 4.4 Build and Install

```bash
# Build with all available cores
make -j$(nproc)

# Install
sudo make install

# Update library cache
sudo ldconfig

# Set OpenCV environment variables
echo 'export OpenCV_DIR=/usr/local/lib/cmake/opencv4' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 5. Verification

### 5.1 Check OpenCV Installation

```bash
pkg-config --modversion opencv4
```

### 5.2 Verify SLAM Module

```python
import cv2

# Check if SLAM module is available
print("OpenCV version:", cv2.__version__)
print("Available modules:", cv2.getBuildInformation())

# Try importing slam module
try:
    slam = cv2.slam
    print("\nSLAM module loaded successfully!")
    print("Available functions:", dir(slam))
except AttributeError:
    print("\nSLAM module not found!")
```

### 5.3 Run SLAM Samples

After building with samples enabled:

```bash
# The slam samples will be installed to:
# /usr/local/share/opencv4/samples/

# Run an example (if built)
./bin/opencv_example
```

### 5.4 Build and Run Tests

```bash
# Rebuild with tests enabled
cmake -D BUILD_TESTS=ON ..
make -j$(nproc)

# Run SLAM module tests
./bin/opencv_test_slam
```

## Troubleshooting

### Eigen3 Not Found

```bash
# Ensure Eigen3 is installed and CMake can find it
sudo apt install -y libeigen3-dev
# If still not found, specify the path:
cmake -D EIGEN_INCLUDE_DIR=/usr/include/eigen3 ..
```

### g2o Linking Errors

```bash
# Ensure g2o is installed correctly
pkg-config --libs g2o
# If not found, add to CMake:
cmake -D CMAKE_PREFIX_PATH=/usr/local ..
```

### FBoW/DBoW2 Not Found

```bash
# For FBoW
cmake -D CMAKE_PREFIX_PATH=~/fbow/build ..
# For DBoW2
cmake -D CMAKE_PREFIX_PATH=~/DBoW2/build ..
```

### CUDA Not Found

```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Summary of Dependencies

| Dependency | Version | Required | Default |
|------------|---------|----------|---------|
| Eigen3 | >= 3.3 | Yes | System |
| yaml-cpp | >= 0.6 | Yes | System |
| nlohmann_json | >= 3.2 | Yes | System |
| g2o | 20230223_git | Yes | Source |
| SQLite3 | any | Yes | System |
| FBoW or DBoW2 | any | Yes | FBoW |
| Pangolin | any | No | Not installed |
| GTSAM | any | No | Not installed |
| OpenMP | any | No | Not installed |
| CUDA | any | No | Not installed |
| ArUCO | any | No | Enabled |

## Quick Install Script

For a minimal installation with core dependencies:

```bash
#!/bin/bash
set -e

# Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libeigen3-dev \
    libyaml-cpp-dev nlohmann-json3-dev libsqlite3-dev libsuitesparse-dev \
    libcxsparse-dev

# Install g2o
cd ~
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o && git checkout 20230223_git
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc) && sudo make install && cd ~

# Install FBoW
git clone https://github.com/rmiquelma/fbow.git
cd fbow && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install && cd ~

# Build OpenCV with SLAM
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && git checkout 4.8.0 && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D BOW_FRAMEWORK=FBoW \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=ON \
    ..
make -j$(nproc) && sudo make install && sudo ldconfig

echo "OpenCV with SLAM module installed successfully!"
```

---

For more information, visit:
- OpenCV: https://github.com/opencv/opencv
- OpenCV Contrib: https://github.com/opencv/opencv_contrib
- OpenCV SLAM: https://github.com/QueenofUSSR/opencv_contrib_slam
