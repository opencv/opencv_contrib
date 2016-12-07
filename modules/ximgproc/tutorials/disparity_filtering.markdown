Disparity map post-filtering {#tutorial_ximgproc_disparity_filtering}
============================

Introduction
------------

Stereo matching algorithms, especially highly-optimized ones that are intended for real-time processing
on CPU, tend to make quite a few errors on challenging sequences. These errors are usually concentrated
in uniform texture-less areas, half-occlusions and regions near depth discontinuities. One way of dealing
with stereo-matching errors is to use various techniques of detecting potentially inaccurate disparity
values and invalidate them, therefore making the disparity map semi-sparse. Several such techniques are
already implemented in the StereoBM and StereoSGBM algorithms. Another way would be to use some kind of
filtering procedure to align the disparity map edges with those of the source image and to propagate
the disparity values from high- to low-confidence regions like half-occlusions. Recent advances in
edge-aware filtering have enabled performing such post-filtering under the constraints of real-time
processing on CPU.

In this tutorial you will learn how to use the disparity map post-filtering to improve the results
of StereoBM and StereoSGBM algorithms.

Source Stereoscopic Image
-------------------------

![Left view](images/ambush_5_left.jpg)
![Right view](images/ambush_5_right.jpg)

Source Code
-----------

We will be using snippets from the example application, that can be downloaded [here ](https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp).

Explanation
-----------

The provided example has several options that yield different trade-offs between the speed and
the quality of the resulting disparity map. Both the speed and the quality are measured if the user
has provided the ground-truth disparity map. In this tutorial we will take a detailed look at the
default pipeline, that was designed to provide the best possible quality under the constraints of
real-time processing on CPU.

-#  **Load left and right views**
    @snippet ximgproc/samples/disparity_filtering.cpp load_views
    We start by loading the source stereopair. For this tutorial we will take a somewhat challenging
    example from the MPI-Sintel dataset with a lot of texture-less regions.

-#  **Prepare the views for matching**
    @snippet ximgproc/samples/disparity_filtering.cpp downscale
    We perform downscaling of the views to speed-up the matching stage at the cost of minor
    quality degradation. To get the best possible quality downscaling should be avoided.

-#  **Perform matching and create the filter instance**
    @snippet ximgproc/samples/disparity_filtering.cpp matching
    We are using StereoBM for faster processing. If speed is not critical, though,
    StereoSGBM would provide better quality. The filter instance is created by providing
    the StereoMatcher instance that we intend to use. Another matcher instance is
    returned by the createRightMatcher function. These two matcher instances are then
    used to compute disparity maps both for the left and right views, that are required
    by the filter.

-#  **Perform filtering**
    @snippet ximgproc/samples/disparity_filtering.cpp filtering
    Disparity maps computed by the respective matcher instances, as well as the source left view
    are passed to the filter. Note that we are using the original non-downscaled view to guide the
    filtering process. The disparity map is automatically upscaled in an edge-aware fashion to match
    the original view resolution. The result is stored in filtered_disp.

-#  **Visualize the disparity maps**
    @snippet ximgproc/samples/disparity_filtering.cpp visualization
    We use a convenience function getDisparityVis to visualize the disparity maps. The second parameter
    defines the contrast (all disparity values are scaled by this value in the visualization).

Results
-------

![Result of the StereoBM](images/ambush_5_bm.png)
![Result of the demonstrated pipeline (StereoBM on downscaled views with post-filtering)](images/ambush_5_bm_with_filter.png)
