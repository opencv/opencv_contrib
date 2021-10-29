An overview of the opencv_contrib modules
-----------------------------------------

This list gives an overview of all modules available inside the contrib repository.
To turn off building one of these module repositories, set the names in bold below to <reponame>

```
$ cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -D BUILD_opencv_<reponame>=OFF <opencv_source_directory>
```

- **alphamat**: Computer Vision based Alpha Matting -- Given an input image and a trimap, generate an alpha matte.

- **aruco**: ArUco and ChArUco Markers -- Augmented reality ArUco marker and "ChARUco" markers where ArUco markers embedded inside the white areas of the checker board.

- **barcode**: Barcode detecting and decoding methods.

- **bgsegm**: Background segmentation algorithm combining statistical background image estimation and per-pixel Bayesian segmentation.

- **bioinspired**: Biological Vision -- Biologically inspired vision model: minimize noise and luminance variance, transient event segmentation, high dynamic range tone mapping methods.

- **ccalib**: Custom Calibration -- Patterns for 3D reconstruction, omnidirectional camera calibration, random pattern calibration and multi-camera calibration.

- **cnn_3dobj**: Deep Object Recognition and Pose -- Uses Caffe Deep Neural Net library to build, train and test a CNN model of visual object recognition and pose.

- **cvv**: Computer Vision Debugger -- Simple code that you can add to your program that pops up a GUI allowing you to interactively and visually debug computer vision programs.

- **datasets**: Datasets Reader -- Code for reading existing computer vision databases and samples of using the readers to train, test and run using that dataset's data.

- **dnn_objdetect**: Object Detection using CNNs -- Implements compact CNN Model for object detection. Trained using Caffe but uses opencv_dnn module.

- **dnn_superres**: Superresolution using CNNs -- Contains four trained convolutional neural networks to upscale images.

- **dnns_easily_fooled**: Subvert DNNs -- This code can use the activations in a network to fool the networks into recognizing something else.

- **dpm**: Deformable Part Model -- Felzenszwalb's Cascade with deformable parts object recognition code.

- **face**: Face Recognition -- Face recognition techniques: Eigen, Fisher and Local Binary Pattern Histograms LBPH methods.

- **freetype**: Drawing text using freetype and harfbuzz.

- **fuzzy**: Fuzzy Logic in Vision -- Fuzzy logic image transform and inverse; Fuzzy image processing.

- **hdf**: Hierarchical Data Storage -- This module contains I/O routines for Hierarchical Data Format: https://en.m.wikipedia.org/wiki/Hierarchical_Data_Format meant to store large amounts of data.

- **hfs**: Hierarchical Feature Selection for Efficient Image Segmentation -- This module contains an efficient algorithm to segment an image.

- **img_hash**: This module contains algorithms to extract hash of an image allowing to efficiently estimate similarity between images.

- **intensity_transform**: The module brings implementations of intensity transformation algorithms to adjust image contrast.

- **julia**: Julia language wrappers with samples and tests.

- **line_descriptor**: Line Segment Extract and Match -- Methods of extracting, describing and matching line segments using binary descriptors.

- **matlab**: Matlab Interface -- OpenCV Matlab Mex wrapper code generator for certain opencv core modules.

- **mcc**: Macbeth Color Chart detector -- Find and return color patch location in MacBeth color calibration charts.

- **optflow**: Optical Flow -- Algorithms for running and evaluating deepflow, simpleflow, sparsetodenseflow and motion templates (silhouette flow).

- **ovis**: OGRE 3D Visualiser -- allows you to render 3D data using the OGRE 3D engine.

- **phase_unwrapping**: Quality-guided phase unwrapping.

- **plot**: Plotting -- The plot module allows you to easily plot data in 1D or 2D.

- **quality**: Image Quality Analysis (IQA) API.

- **rapid**: Silhouette based 3D object tracking.

- **reg**: Image Registration -- Pixels based image registration for precise alignment. Follows the paper "Image Alignment and Stitching: A Tutorial", by Richard Szeliski.

- **rgbd**: RGB-Depth Processing module -- Linemod 3D object recognition; Fast surface normals and 3D plane finding. 3D visual odometry. 3d reconstruction using KinectFusion.

- **saliency**: Saliency API -- Where humans would look in a scene. Has routines for static, motion and "objectness" saliency.

- **sfm**: Structure from Motion -- This module contains algorithms to perform 3d reconstruction from 2d images. The core of the module is a light version of Libmv.

- **shape**: Shape Distance and Matching

- **stereo**: Stereo Correspondence -- Stereo matching done with different descriptors: Census / CS-Census / MCT / BRIEF / MV and dense stereo correspondence using Quasi Dense Stereo method.

- **structured_light**: Structured Light Use -- How to generate and project gray code patterns and use them to find dense depth in a scene.

- **superres**: Super Resolution

- **surface_matching**: Point Pair Features -- Implements 3d object detection and localization using multimodal point pair features.

- **text**: Scene Text Detection and Recognition -- This module contains algorithms to perform text detection, words segmentation and text recognition in a visual scene.

- **tracking**: Vision Based Object Tracking -- Use and/or evaluate different visual object tracking techniques.

- **videostab**: Video Stabilization

- **viz**: 3D Visualizer

- **wechat_qrcode**: WeChat QR code detector for detecting and parsing QR code.

- **xfeatures2d**: Features2D extra -- Extra 2D Features Framework containing experimental and non-free 2D feature detector/descriptor algorithms. SURF, BRIEF, Censure, Freak, LUCID, Daisy, Self-similar.

- **ximgproc**: Extended Image Processing -- Structured Forests / Domain Transform Filter / Guided Filter / Adaptive Manifold Filter / Joint Bilateral Filter / Superpixels / Ridge Detection Filter.

- **xobjdetect**: Boosted 2D Object Detection -- Uses a Waldboost cascade and local binary patterns computed as integral features for 2D object detection.

- **xphoto**: Extra Computational Photography -- Additional photo processing algorithms: Color balance / Denoising / Inpainting.
