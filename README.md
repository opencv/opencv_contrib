# OpenCV Fractal Markers (GSoC 2025 Proposal)

**Project**: OpenCV idea 13 : Integrate Fractal ArUco into OpenCV
**Paper**: [Fractal Markers: A New Approach for Long-Range Marker Pose Estimation Under Occlusion](https://ieeexplore.ieee.org/document/8698871)

## API Proposal
```cpp
// Current implementation:
void detectFractalMarkers(InputArray image,
    const Ptr<FractalDictionary>& fractalDict,
    OutputArrayOfArrays corners,
    const Ptr<DetectorParameters>& params = nullptr);


