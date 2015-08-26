Classify {#tutorial_classify}
===============

Goal
----

In this tutorial you will learn how to

-   How to extract feature from an image
-   How to extract features from images under a given root path
-   How to make a prediction using reference images and target image

Code
----

You can download the code from [here ](https://github.com/Wangyida/opencv_contrib/blob/cnn_3dobj/samples/demo_classify.cpp).
@include cnn_3dobj/samples/demo_classify.cpp

Explanation
-----------

Here is the general structure of the program:

-   Initialize a net work with Device.
    @code{.cpp}
    cv::cnn_3dobj::descriptorExtractor descriptor(device);
    @endcode

-   Load net with the caffe trained net work parameter and structure.
    @code{.cpp}
    if (strcmp(mean_file.c_str(), "no") == 0)
        descriptor.loadNet(network_forIMG, caffemodel);
    else
        descriptor.loadNet(network_forIMG, caffemodel, mean_file);
    @endcode

-   List the file names under a given path.
    @code{.cpp}
    listDir(src_dir.c_str(), name_gallery, false);
    for (unsigned int i = 0; i < name_gallery.size(); i++)
    {
        name_gallery[i] = src_dir + name_gallery[i];
    }
    @endcode

-   Extract feature from a set of images.
    @code{.cpp}
    descriptor.extract(img_gallery, feature_reference, feature_blob);
    @endcode

-   Initialize a matcher which using L2 distance.
    @code{.cpp}
    cv::BFMatcher matcher(NORM_L2);
    std::vector<std::vector<cv::DMatch> > matches;
    @endcode

-   Have a KNN match on the target and reference images.
    @code{.cpp}
    matcher.knnMatch(feature_test, feature_reference, matches, num_candidate);
    @endcode

-   Print features of the reference images.
    @code{.cpp}std::cout << std::endl << "---------- Features of target image: " << target_img << "----------" << endl << feature_test << std::endl;
    @endcode
Results
-------
