Training Model Analysis {#tutorial_model_analysis}
=============

Goal
----

In this tutorial you will learn how to

-   Extract feature from particular image.
-   Have a meaningful comparation on the extracted feature.

Code
----

@include cnn_3dobj/samples/model_analysis.cpp

Explanation
-----------

Here is the general structure of the program:

-   Sample which is most closest in pose to reference image and also the same class.
    @code{.cpp}
    ref_img.push_back(ref_img1);
    @endcode

-   Sample which is less closest in pose to reference image and also the same class.
    @code{.cpp}
    ref_img.push_back(ref_img2);
    @endcode

-   Sample which is very close in pose to reference image but not the same class.
    @code{.cpp}
    ref_img.push_back(ref_img3);
    @endcode

-   Initialize a net work with Device.
    @code{.cpp}
    cv::cnn_3dobj::descriptorExtractor descriptor(device, dev_id);
    @endcode
-   Load net with the caffe trained net work parameter and structure.
    @code{.cpp}
    if (strcmp(mean_file.c_str(), "no") == 0)
        descriptor.loadNet(network_forIMG, caffemodel);
    else
        descriptor.loadNet(network_forIMG, caffemodel, mean_file);
    @endcode

-   Have comparations on the distance between reference image and 3 other images
    distance between closest sample and reference image should be smallest and
    distance between sample in another class and reference image should be largest.
    @code{.cpp}
    if (matches[0] < matches[1] && matches[0] < matches[2])
        pose_pass = true;
    if (matches[1] < matches[2])
        class_pass = true;
    @endcode
Results
-------
