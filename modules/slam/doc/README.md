# cv_slam Detailed Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Relocalization Guide](#relocalization-guide)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [FAQ](#faq)

---

## Introduction

cv_slam is a visual SLAM (Simultaneous Localization and Mapping) module for OpenCV. It supports monocular, stereo, and RGBD cameras with various camera models.

### Key Features

- **Camera models**: Perspective, fisheye, equirectangular
- **Map persistence**: Save and load pre-built maps
- **Localization mode**: Re-localize in known environments
- **Loop closure**: Detect and correct accumulated drift
- **Backend optimization**: Real-time sliding window bundle adjustment
- **Modular design**: Swap feature detectors and matchers

### System Requirements

- OpenCV 4.5+
- C++17 compiler
- Eigen3
- g2o (for backend optimization)
- FBoW (for loop closure)

---

## Installation

### Build with OpenCV Contrib

```bash
# Clone repositories
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Copy SLAM module
cp -r opencv_contrib_slam/modules/slam opencv_contrib/modules/

# Build
cd opencv/build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      -DBUILD_opencv_slam=ON \
      ..
make -j4
```

### Standalone Build

```bash
cd opencv_contrib_slam/modules/slam/build
cmake -DBUILD_SAMPLES=ON ..
make -j4
```

---

## Basic Usage

### Minimal Example

```cpp
#include <opencv2/slam.hpp>
#include <opencv2/features2d.hpp>

int main() {
    // Create detector and matcher
    auto orb = cv::ORB::create(1000);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    // Configure
    cv::vo::VOConfig config;
    config.camera_config_file = "camera.yaml";
    config.vocab_file = "orb_vocab.fbow";
    
    // Create SLAM system
    auto slam = cv::vo::VisualOdometry::create(config, orb, matcher);
    
    // Process frames
    for (const auto& frame : frames) {
        auto pose = slam->processFrame(frame.image, frame.timestamp);
        if (pose.has_value()) {
            // Use pose
        }
    }
    
    // Cleanup
    slam->release();
    return 0;
}
```

### Configuration File

Create `camera.yaml` for your camera:

```yaml
%YAML:1.0
Camera.name: "EuRoC monocular"
Camera.setup: monocular
Camera.model: perspective
Camera.fx: 458.654
Camera.fy: 457.296
Camera.cx: 367.215
Camera.cy: 248.375
Camera.k1: -0.28340811
Camera.k2: 0.07395907
Camera.fps: 20.0
Camera.width: 752
Camera.height: 480
```

---

## Advanced Features

### Map Save/Load

```cpp
// Save map after SLAM
slam->saveMap("map.msgpack");

// Load map for localization
auto slam2 = cv::vo::VisualOdometry::create(config, orb, matcher);
slam2->loadMap("map.msgpack");
```

### Localization Mode

```cpp
// Switch to localization mode (no new mapping)
slam->setMode(cv::vo::SLAMMode::LOCALIZATION);

// Process frames for localization only
for (const auto& frame : frames) {
    auto pose = slam->processFrame(img, timestamp);
}
```

---

## Relocalization Guide

### Overview

Relocalization allows the system to recover its position after tracking loss or when starting in a pre-built map.

### When to Use Relocalization

| Scenario | Method | Description |
|----------|--------|-------------|
| **Tracking lost** | Automatic | System tries to relocalize automatically |
| **Starting in known map** | Manual | Provide initial pose estimate |
| **Global localization** | Manual | Try multiple pose hypotheses |

### Method 1: Automatic Relocalization

When tracking is lost, the system automatically tries to relocalize:

```cpp
auto pose = slam->processFrame(image, timestamp);
if (!pose.has_value()) {
    // Tracking lost - system will try to relocalize automatically
    // Continue processing frames
}
```

**How it works:**
1. System detects tracking loss
2. BoW place recognition finds candidate keyframes
3. Geometric verification validates candidates
4. If successful, tracking resumes

### Method 2: Relocalization by Pose Estimate

If you have an approximate pose (e.g., from GPS, IMU, or user input):

```cpp
// Create pose estimate (4x4 transformation matrix)
cv::Matx44d pose_cw = ...;  // Camera to world transformation

// Request relocalization
bool success = slam->relocalize_by_pose(pose_cw);

if (success) {
    std::cout << "Relocalization successful!" << std::endl;
} else {
    std::cout << "Relocalization failed - pose too far from map" << std::endl;
}
```

**Requirements:**
- Pose must be within ~5 meters of actual position
- Orientation must be within ~30 degrees

### Method 3: 2D Relocalization (with Ground Plane)

For ground robots or vehicles with known ground plane:

```cpp
// Provide pose + ground normal vector
cv::Matx44d pose_cw = ...;
cv::Vec3d normal_vector(0, -1, 0);  // Y-down coordinate system

bool success = slam->relocalize_by_pose2D(pose_cw, normal_vector);
```

**Advantages:**
- More robust than full 3D relocalization
- Faster convergence
- Better for ground robots

### Complete Relocalization Workflow

```cpp
#include <opencv2/slam.hpp>
#include <opencv2/features2d.hpp>

int main() {
    // 1. Create SLAM system
    auto orb = cv::ORB::create(1000);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    cv::vo::VOConfig config;
    config.camera_config_file = "camera.yaml";
    config.vocab_file = "orb_vocab.fbow";
    
    auto slam = cv::vo::VisualOdometry::create(config, orb, matcher);
    
    // 2. Load pre-built map
    slam->loadMap("map.msgpack");
    
    // 3. Switch to localization mode
    slam->setMode(cv::vo::SLAMMode::LOCALIZATION);
    
    // 4. Process first frame
    cv::Mat image = cv::imread("frame.png", cv::IMREAD_GRAYSCALE);
    auto pose = slam->processFrame(image, timestamp);
    
    // 5. If tracking fails, try relocalization
    if (!pose.has_value()) {
        std::cout << "Tracking lost, attempting relocalization..." << std::endl;
        
        // Option A: Wait for automatic relocalization
        // Just continue processing frames
        
        // Option B: Provide manual pose estimate
        cv::Matx44d initial_pose = ...;  // From GPS/IMU/etc.
        slam->relocalize_by_pose(initial_pose);
    }
    
    // 6. Continue processing
    while (processing) {
        auto pose = slam->processFrame(image, timestamp);
        if (pose.has_value()) {
            // Use pose for navigation, etc.
        }
    }
    
    slam->release();
    return 0;
}
```

### Troubleshooting Relocalization

| Problem | Cause | Solution |
|---------|-------|----------|
| **Relocalization fails** | Too far from map | Move closer to mapped area |
| **Wrong position** | Ambiguous environment | Provide better pose estimate |
| **Slow relocalization** | Large map | Use place recognition priors |
| **Frequent tracking loss** | Poor features | Increase lighting/texture |

### Best Practices

1. **Build comprehensive maps**: More keyframes = better relocalization
2. **Use loop closure**: Reduces drift, improves map consistency
3. **Provide good initial pose**: Even rough estimate helps
4. **Continue processing**: Don't stop when tracking is lost
5. **Test in similar conditions**: Lighting, viewpoint should match mapping

---

### Enable/Disable Components

```cpp
// Disable backend optimization (faster, less accurate)
slam->setBackendEnabled(false);

// Disable loop closure (no drift correction)
slam->setLoopClosureEnabled(false);
```

### Custom Feature Detector

```cpp
// Use SIFT instead of ORB
auto sift = cv::SIFT::create(1000);
auto matcher = cv::BFMatcher::create(cv::NORM_L2);

auto slam = cv::vo::VisualOdometry::create(config, sift, matcher);
```

---

## API Reference

### VisualOdometry Class

#### Core Methods

| Method | Description |
|--------|-------------|
| `processFrame(img, timestamp)` | Process a single frame |
| `getMapPoints()` | Get all 3D map points |
| `getCurrentPose()` | Get current camera pose |
| `getTrajectory()` | Get full trajectory |

#### Control Methods

| Method | Description |
|--------|-------------|
| `setMode(mode)` | Set SLAM/LOCALIZATION mode |
| `getMode()` | Get current mode |
| `setBackendEnabled(enable)` | Enable/disable backend BA |
| `setLoopClosureEnabled(enable)` | Enable/disable loop closure |
| `reset()` | Reset the system |
| `release()` | Cleanup and release resources |

#### Map Methods

| Method | Description |
|--------|-------------|
| `saveMap(path)` | Save map to file |
| `loadMap(path)` | Load map from file |
| `saveTrajectory(path, format)` | Save trajectory |

### VOConfig Structure

```cpp
struct VOConfig {
    std::string camera_config_file;  // Camera YAML file
    std::string vocab_file;          // BoW vocabulary file
    bool enable_backend;             // Enable backend BA
    bool enable_loop_closure;        // Enable loop closure
    int local_ba_window_size;        // BA window size
    // ... more parameters
};
```

---

## Examples

### Full SLAM Pipeline

See `samples/cpp/full_slam.cpp`:

```bash
./example_full_slam camera.yaml orb_vocab.fbow /path/to/images /output
```

### Localization Mode

See `samples/cpp/localization_mode.cpp`:

```bash
./example_localization_mode camera.yaml orb_vocab.fbow map.msgpack /path/to/images
```

### Map Save/Load Test

See `samples/cpp/map_save_load.cpp`:

```bash
./example_map_save_load camera.yaml orb_vocab.fbow /path/to/images /output
```

---

## FAQ

### Q: What vocabulary file should I use?

Download ORB vocabulary from:
- [FBoW ORB Vocabulary](https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow)

### Q: My tracking rate is low. How to improve?

Try:
1. Reduce `max_features` in config (e.g., 500 instead of 1000)
2. Disable backend: `setBackendEnabled(false)`
3. Use ORB instead of SIFT/AKAZE

### Q: How to use with stereo cameras?

Set `Camera.setup: stereo` in config and provide baseline:

```yaml
Camera.setup: stereo
Camera.baseline: 0.11  # meters
```

### Q: Can I use learning-based features?

Yes! Any `cv::Feature2D` subclass works:

```cpp
auto sift = cv::SIFT::create();
auto superpoint = cv::SuperPoint::create();  // If available
```

### Q: How accurate is the localization?

On EuRoC MH01 dataset:
- ATE RMSE: ~0.023m (with loop closure)
- RPE RMSE: ~0.007m
- Tracking rate: ~99%

See [BENCHMARK.md](../../Docs/Project/BENCHMARK.md) for detailed results.

---

## Troubleshooting

### Tracking Lost Frequently

- Check camera calibration
- Increase feature detection threshold
- Ensure sufficient texture in environment

### Map Save/Load Fails

- Ensure write permissions for output directory
- Check vocabulary file path is correct
- Verify map file is not corrupted

### High CPU Usage

- Disable visualization
- Reduce `local_ba_window_size`
- Use fewer features

---

## License

BSD-2-Clause license

Based on [stella_vslam](https://github.com/stella-cv/stella_vslam)
