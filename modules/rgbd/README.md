 RGB-Depth Processing module
============================

Contains a collection of depth processing algorithms:
* Linemod 3D object recognition
* Fast surface normals and 3D plane finding
* 3D visual odometry
* KinectFusion

Note that the KinectFusion algorithm was patented and its use may be restricted by following (but not limited to) list of patents:

* _US20120196679A1_  Real-Time Camera Tracking Using Depth Maps
* _US20120194644A1_  Mobile Camera Localization Using Depth Maps
* _US20120194516A1_  Three-Dimensional Environment Reconstruction
* _US8401225B2_  Moving object segmentation using depth images

Since OpenCV's license imposes different restrictions on usage please consult a legal before using this algorithm any way.

That's why you need to set the OPENCV_ENABLE_NONFREE option in CMake to use KinectFusion.
