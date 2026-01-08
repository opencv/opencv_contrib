Interactive Visual Debugging of Computer Vision applications {#tutorial_cvv_introduction}
============================================================

What is the most common way to debug computer vision applications? Usually the answer is temporary,
hacked together, custom code that must be removed from the code for release compilation.

In this tutorial we will show how to use the visual debugging features of the **cvv** module
(*opencv2/cvv.hpp*) instead.

Goals
-----

In this tutorial you will learn how to:

-   Add cvv debug calls to your application
-   Use the visual debug GUI
-   Enable and disable the visual debug features during compilation (with zero runtime overhead when
    disabled)

Code
----

The example code

-   captures images (*videoio*), e.g. from a webcam,
-   applies some filters to each image (*imgproc*),
-   detects image features and matches them to the previous image (*features2d*).

If the program is compiled without visual debugging (see CMakeLists.txt below) the only result is
some information printed to the command line. We want to demonstrate how much debugging or
development functionality is added by just a few lines of *cvv* commands.

@includelineno cvv/samples/cvv_demo.cpp

@code{.cmake}
cmake_minimum_required(VERSION 2.8)

project(cvvisual_test)

SET(CMAKE_PREFIX_PATH ~/software/opencv/install)

SET(CMAKE_CXX_COMPILER "g++-4.8")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -pthread -Wall -Werror")

# (un)set: cmake -DCVV_DEBUG_MODE=OFF ..
OPTION(CVV_DEBUG_MODE "cvvisual-debug-mode" ON)
if(CVV_DEBUG_MODE MATCHES ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCVVISUAL_DEBUGMODE")
endif()


FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cvvt main.cpp)
target_link_libraries(cvvt
  opencv_core opencv_videoio opencv_imgproc opencv_features2d
  opencv_cvv
)
@endcode

Explanation
-----------

-#  We compile the program either using the above CmakeLists.txt with Option *CVV_DEBUG_MODE=ON*
    (*cmake -DCVV_DEBUG_MODE=ON*) or by adding the corresponding define *CVVISUAL_DEBUGMODE* to
    our compiler (e.g. *g++ -DCVVISUAL_DEBUGMODE*).
-#  The first cvv call simply shows the image (similar to *imshow*) with the imgIdString as comment.
    @code{.cpp}
    cvv::showImage(imgRead, CVVISUAL_LOCATION, imgIdString.c_str());
    @endcode
    The image is added to the overview tab in the visual debug GUI and the cvv call blocks.

    ![image](images/01_overview_single.jpg)

    The image can then be selected and viewed

    ![image](images/02_single_image_view.jpg)

    Whenever you want to continue in the code, i.e. unblock the cvv call, you can either continue
    until the next cvv call (*Step*), continue until the last cvv call (*\>\>*) or run the
    application until it exists (*Close*).

    We decide to press the green *Step* button.

-#  The next cvv calls are used to debug all kinds of filter operations, i.e. operations that take a
    picture as input and return a picture as output.
    @code{.cpp}
    cvv::debugFilter(imgRead, imgGray, CVVISUAL_LOCATION, "to gray");
    @endcode
    As with every cvv call, you first end up in the overview.

    ![image](images/03_overview_two.jpg)

    We decide not to care about the conversion to gray scale and press *Step*.
    @code{.cpp}
    cvv::debugFilter(imgGray, imgGraySmooth, CVVISUAL_LOCATION, "smoothed");
    @endcode
    If you open the filter call, you will end up in the so called "DefaultFilterView". Both images
    are shown next to each other and you can (synchronized) zoom into them.

    ![image](images/04_default_filter_view.jpg)

    When you go to very high zoom levels, each pixel is annotated with its numeric values.

    ![image](images/05_default_filter_view_high_zoom.jpg)

    We press *Step* twice and have a look at the dilated image.
    @code{.cpp}
    cvv::debugFilter(imgEdges, imgEdgesDilated, CVVISUAL_LOCATION, "dilated edges");
    @endcode
    The DefaultFilterView showing both images

    ![image](images/06_default_filter_view_edges.jpg)

    Now we use the *View* selector in the top right and select the "DualFilterView". We select
    "Changed Pixels" as filter and apply it (middle image).

    ![image](images/07_dual_filter_view_edges.jpg)

    After we had a close look at these images, perhaps using different views, filters or other GUI
    features, we decide to let the program run through. Therefore we press the yellow *\>\>* button.

    The program will block at
    @code{.cpp}
    cvv::finalShow();
    @endcode
    and display the overview with everything that was passed to cvv in the meantime.

    ![image](images/08_overview_all.jpg)

-#  The cvv debugDMatch call is used in a situation where there are two images each with a set of
    descriptors that are matched to each other.

    We pass both images, both sets of keypoints and their matching to the visual debug module.
    @code{.cpp}
    cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, allMatchIdString.c_str());
    @endcode
    Since we want to have a look at matches, we use the filter capabilities (*\#type match*) in the
    overview to only show match calls.

    ![image](images/09_overview_filtered_type_match.jpg)

    We want to have a closer look at one of them, e.g. to tune our parameters that use the matching.
    The view has various settings how to display keypoints and matches. Furthermore, there is a
    mouseover tooltip.

    ![image](images/10_line_match_view.jpg)

    We see (visual debugging!) that there are many bad matches. We decide that only 70% of the
    matches should be shown - those 70% with the lowest match distance.

    ![image](images/11_line_match_view_portion_selector.jpg)

    Having successfully reduced the visual distraction, we want to see more clearly what changed
    between the two images. We select the "TranslationMatchView" that shows to where the keypoint
    was matched in a different way.

    ![image](images/12_translation_match_view_portion_selector.jpg)

    It is easy to see that the cup was moved to the left during the two images.

    Although, cvv is all about interactively *seeing* the computer vision bugs, this is complemented
    by a "RawView" that allows to have a look at the underlying numeric data.

    ![image](images/13_raw_view.jpg)

-#  There are many more useful features contained in the cvv GUI. For instance, one can group the
    overview tab.

    ![image](images/14_overview_group_by_line.jpg)

Result
------

-   By adding a view expressive lines to our computer vision program we can interactively debug it
    through different visualizations.
-   Once we are done developing/debugging we do not have to remove those lines. We simply disable
    cvv debugging (*cmake -DCVV_DEBUG_MODE=OFF* or g++ without *-DCVVISUAL_DEBUGMODE*) and our
    programs runs without any debug overhead.

Enjoy computer vision!
