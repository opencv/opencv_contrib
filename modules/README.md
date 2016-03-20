An overview of the contrib modules and a small explanation
----------------------------------------------------------

This list gives an overview of all modules available inside the contrib repository.
These are also the correct names for disabling the building of a specific module by adding

```
$ cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -D BUILD_opencv_reponame=OFF <opencv_source_directory>
```

1. **opencv_adas**: Advanced Driver Assistance Systems module with Forward Collision Warning.

2. **opencv_bgsegm**: Improved Adaptive Background Mixture Model for Real-time Tracking / Visual Tracking of Human Visitors under Variable-Lighting Conditions.

3. **opencv_bioinspired**: Biologically inspired vision models and derivated tools.

4. **opencv_ ccalib**: Custom Calibration Pattern for 3D reconstruction.

5. **opencv_cvv**: GUI for Interactive Visual Debugging of Computer Vision Programs.

6. **opencv_datasets**: Interface for interfacing with existing computer vision databases.

7. **opencv_datasettools**: Tools for working with different datasets.

8. **opencv_face**: Recently added face recognition software which is not yet stabilized.

9. **opencv_latentsvm**: Implementation of the LatentSVM detector algorithm.

10. **opencv_line_descriptor**: Binary descriptors for lines extracted from an image.

11. **opencv_matlab**: OpenCV Matlab Code Generator.

12. **opencv_optflow**: Optical Flow Algorithms for tracking points.

13. **opencv_reg**: Image Registration module.

14. **opencv_rgbd**: RGB-Depth Processing module.

15. **opencv_saliency**: Saliency API, understanding where humans focus given a scene.

16. **opencv_surface_matching**: Surface Matching Algorithm Through 3D Features.

17. **opencv_text**: Scene Text Detection and Recognition in Natural Scene Images.

18. **opencv_tracking**: Long-term optical tracking API.

19. **opencv_xfeatures2d**: Extra 2D Features Framework containing experimental and non-free 2D feature algorithms.

20. **opencv_ximgproc**: Extended Image Processing: Structured Forests / Domain Transform Filter / Guided Filter / Adaptive Manifold Filter / Joint Bilateral Filter / Superpixels.

21. **opencv_xobjdetect**: Integral Channel Features Detector Framework.

22. **opencv_xphoto**: Additional photo processing algorithms: Color balance / Denoising / Inpainting.

23. **opencv_stereo**: Stereo Correspondence done with different descriptors: Census / CS-Census / MCT / BRIEF / MV.

24. **opencv_hdf**: Hierarchical Data Format I/O.

25. **opencv_fuzzy**: New module focused on the fuzzy image processing.
