# OpenCV VSLAM Module

Monocular, stereo, and RGBD visual SLAM system for OpenCV.

## Features

- **Multiple camera models**: Perspective, fisheye, equirectangular
- **Map save/load**: Store and reuse pre-built maps
- **Localization mode**: Re-localize in pre-built maps
- **Loop closure**: Detect and correct drift using place recognition
- **Backend optimization**: Sliding window bundle adjustment
- **Modular design**: Pluggable feature detectors and matchers

## Quick Start

```cpp
#include <opencv2/slam.hpp>
#include <opencv2/features2d.hpp>

// Create feature detector and matcher
auto orb = cv::ORB::create(1000);
auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

// Configure SLAM
cv::vo::VOConfig config;
config.camera_config_file = "camera.yaml";
config.vocab_file = "orb_vocab.fbow";
config.enable_backend = true;
config.enable_loop_closure = true;

// Create SLAM system
auto slam = cv::vo::VisualOdometry::create(config, orb, matcher);

// Process frames
for (const auto& frame : frames) {
    auto pose = slam->processFrame(frame.image, frame.timestamp);
    if (pose.has_value()) {
        // Use pose
    }
}

// Save map
slam->saveMap("map.json");

// Cleanup
slam->release();
```

## Examples

See `samples/cpp/` directory for complete examples:

- `full_slam.cpp` - Complete SLAM pipeline
- `localization_mode.cpp` - Localization with pre-built map
- `map_save_load.cpp` - Map save/load demonstration

## Build

```bash
cd opencv_contrib_slam/modules/slam/build
cmake -DBUILD_SAMPLES=ON ..
make -j4
```

## Documentation

- [Detailed documentation](doc/README.md)
- [Benchmarks](docs/BENCHMARK.md)

## License

BSD-2-Clause license

Based on [stella_vslam](https://github.com/stella-cv/stella_vslam) (BSD-2-Clause)
