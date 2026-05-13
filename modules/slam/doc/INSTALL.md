# OpenCV SLAM Module Installation Guide

This guide provides step-by-step instructions for installing OpenCV with the SLAM module from scratch on Ubuntu 20.04/22.04.

## Table of Contents

1. [System Dependencies](#1-system-dependencies)
2. [Core Dependencies](#2-core-dependencies)
3. [Optional Dependencies](#3-optional-dependencies)
4. [Building OpenCV with SLAM Module](#4-building-opencv-with-slam-module)
5. [Verification](#5-verification)
6. [Running Examples](#6-running-examples)
7. [Arch Linux Installation](#7-arch-linux-installation)

---

## 7. Arch Linux Installation

On Arch Linux (or Arch-based WSL), use `pacman` instead of `apt`:

```bash
# Install core dependencies via pacman
sudo pacman -S --noconfirm base-devel cmake git pkg-config \
    eigen yaml-cpp nlohmann-json sqlite suitesparse

# Note: Arch's Eigen version is 5.x, but g2o requires Eigen 3.x
# Install Eigen 3.4 from source:
cd /tmp
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/home/$USER/local
make -j$(nproc) && make install

# Install g2o (specify Eigen 3.4 path):
cd /tmp/g2o && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DEIGEN3_INCLUDE_DIR=/home/$USER/local/include/eigen3 \
    -DBUILD_SHARED_LIBS=ON -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc) && sudo make install

# Install FBoW
cd /tmp && git clone https://github.com/rmiquelma/fbow.git
cd fbow && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install

# Build OpenCV with SLAM (same as Ubuntu)
```

### Arch-Specific Notes

| Issue | Solution |
|-------|----------|
| Eigen 5.x incompatible with g2o | Install Eigen 3.4 from source |
| GitHub clone timeout | Configure git proxy or use mirror |
| sudo password prompt timeout | Use `sudo -n` or run commands as root |

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
