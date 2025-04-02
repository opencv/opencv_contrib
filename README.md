# OpenCV Fractal Markers (GSoC 2025 Proposal)

Implementation of hierarchical marker detection from:  
[Fractal Markers: A New Approach for Long-Range Marker Pose Estimation Under Occlusion](https://ieeexplore.ieee.org/document/8698871)

## API Proposal
```cpp
vector<vector<Point2f>> corners;
detectFractalMarkers(image, getFractalDictionary(), corners);


## Project Phases
- [x] Phase 1: API Design ( Doxygen documented -->(`aruco/include/opencv2/aruco/fractal_markers.hpp`)  )
- [ ] Phase 2: Core Detection Implementation  
- [ ] Phase 3: Occlusion Handling & Optimization  


Note: This is my first OpenCV contribution. Feedback on API design and implementation approach is welcome!