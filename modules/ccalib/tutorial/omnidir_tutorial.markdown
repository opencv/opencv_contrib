Omnidirectional Cameara Calibration {#tutorial_omnidir_calib_main}
======================

This module includes calibration, rectification and stereo reconstruction of omnidirectional camearas. The camera model is from this paper:

*C. Mei and P. Rives, Single view point omnidirectional camera calibration from planar grids, in ICRA 2007.*

The model is capable of describing catadioptric cameras, which may have 360 degrees of field of view. Also, it can be used for fisheye cameras.

The implementation of the calibration part is based on Li's calibration toolbox:

*B. Li, L. Heng, K. Kevin and M. Pollefeys, "A Multiple-Camera System Calibration Toolbox Using A Feature Descriptor-Based Calibration Pattern", in IROS 2013.*

This tutorial will introduce the following parts of omnidirectional camera calibartion module:

-    calibrate a single camera.
-    calibrate a system with multiple cameras.
-    rectify images so that large distoration is removed.
-    reconstruct 3D from two stereo images, with large filed of view.
-    comparison with fisheye model in opencv/calib3d/

Single Camera Calibration
---------------------

The first step to calibrate camera is to get a calibration object and take some photos. The most common object is checkerboard, other objects may also be available for this module further. The physical length of checkerboard square is required.

Next extract checkerboard corners and get their positions in all images by Opencv function *findChessboardCorners* or by hand (if you do want to make sure that corners are perfectly extracted). Save the positions in images in imagePoints, with vector of Mat type of CV_64FC2. That is, you get a 1XN or NX1 CV_64FC2 mat for each image, N is the number of corners. Then set the world coordinate to one of the four extreme corners and set xy plane as the checkerboard plane so that the position of checkerboard corners in world frame can be determined. Save the points in world coordiante in objectPoints, with vector of Mat type of CV_64FC3. Each element in vector objectPoints and imagePoints must come from the same image.

In the folder *data*, the file *omni_calib_data.xml* stores an example of objectPoints, imagePoints and imageSize. Use the following code to load them:

@code{.cpp}
cv::FileStorage fs("omni_calib_data.xml", cv::FileStorage::READ);
std::vector<cv::Mat> objectPoints, imagePoints;
cv::Size imgSize;
fs["objectPoints"] >> objectPoints;
fs["imagePoints"] >> imagePoints;
fs["imageSize"] >> imgSize;
@endcode

Then run the calibration function like:

@code{.cpp}
double rms = omnidir::calibrate(objectPoints, imagePoints, size, K, xi, D, om, t, flags, critia)
@endcode

The variable *size* of tyep Size is the size of images. *flags* is a enumeration for some features, including:

-    CALIB_USE_GUESS: initialize camera parameters by input K, xi, D, om, t.
-    CALIB_FIX_SKEW, CALIB_FIX_K1, CALIB_FIX_K2, CALIB_FIX_P1, CALIB_FIX_P2, CALIB_FIX_XI, CALIB_FIX_GAMMA, CALIB_FIX_CENTER: fix the corresponding parameters during calibration, you can use 'plus' operator to set multiple features. For example, CALIB_FIX_SKEW+CALIB_FIX_K1 means fix skew and K1.

*K*, *xi*, *D*, *om*, *t* are output internal and external parameters. The returned value *rms* is the root mean square of reprojection errors.

critia is the stopping critia during optimization, set it to be, for example, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 0.0001), which means using 200 iterations and stopping when relative change is smaller than 0.0001.

Image Rectificaiton
---------------------------

Omnidirectional images have very large distortion, making it is not compatible with human's eye balls. Here is an example of omnidirectional image of 360 degrees of horizontal field of view.

![image](img/sample.jpg)

After rectification, a perspective like view is generated. Here is one example to run image rectification in this module:

@code{.cpp}
omnidir::undistortImage(distorted, undistorted, K, D, xi, int flags, Knew, new_size)
@endcode

The variable *distorted* and *undistorted* are the origional image and rectified image perspectively. *K*, *D*, *xi* are camera parameters. *KNew* and *new_size* are the camera matrix and image size for rectified image. *flags* is the rectification type, it can be:

-    RECTIFY_PERSPECTIVE: rectify to perspective images, which will lose some filed of view.
-    RECTIFY_CYLINDRICAL: rectify to cylindrical images, it preserves all view.
-    RECTIFY_LONGLATI: rectify to longitude-latitude map like a world map of the earth. This rectification can be used to stereo reconstruction.

The following three images are three types of rectified images discribed above:

![image](img/sample_rec_per.jpg)

![image](img/sample_rec_cyl.jpg)

![image](img/sample_rec_log.jpg)

It can be observed that perspective rectified image perserves only a little field of view, while other two perserves all.