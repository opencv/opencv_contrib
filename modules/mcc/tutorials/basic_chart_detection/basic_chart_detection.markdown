Detecting colorcheckers using basic algorithms{#tutorial_mcc_basic_chart_detection}
===========================

In this tutorial you will learn how to use the 'mcc' module to detect colorcharts in a image.
Here we will only use the basic detection algorithm. In the next tutorial you will see how you
can improve detection accuracy using a neural network.

Building
----

When building OpenCV, run the following command to build all the contrib module:

```make
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules/
```

Or only build the mcc module:

```make
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules/mcc
```

Or make sure you check the mcc module in the GUI version of CMake: cmake-gui.

Source Code of the sample
-----------

```
run
<path_of_your_opencv_build_directory>/bin/example_mcc_chart_detection -t=<type_of_chart> -v=<optional_path_to_video_if_not_provided_webcam_will_be_used.mp4> --ci=<optional_camera_id_needed_only_if_video_not_provided> --nc=<optional_maximum_number_of_charts_to_look_for>
```

* -t=#  is the chart type where 0 (Standard), 1 (DigitalSG), 2 (Vinyl)
* --ci=#  is the camera ID where 0 (default is the main camera), 1 (secondary camera) etc
* --nc=#  By default its values is 1 which means only the best chart will be detected

Examples:

```
Run a movie on a standard macbeth chart:
/home/opencv/build/bin/example_mcc_chart_detection -t=0 -v=mcc24.mp4

Or run on a vinyl macbeth chart from camera 0:
/home/opencv/build/bin/example_mcc_chart_detection -t=2 --ci=0

Or run on a vinyl macbeth chart, detecting the best 5 charts(Detections can be less than 5 but never more):
/home/opencv/build/bin/example_mcc_chart_detection -t=2 --ci=0 --nc=5

```


@includelineno mcc/samples/chart_detection.cpp

Explanation
-----------

-#  **Set header and namespaces**
    @code{.cpp}
    #include <opencv2/mcc.hpp>
    using namespace std;
    using namespace cv;
    using namespace mcc;
    @endcode

    If you want you can set the namespace like the code above.
-#  **Create the detector object**
    @code{.cpp}
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    @endcode

    This is just to create the object.
-#  **Run the detector**
    @code{.cpp}
    detector->process(image, chartType);
    @endcode

    If the detector successfully detects atleast one chart, it return true otherwise it returns false. In the above given code we print a failure message if no chart were detected. Otherwise if it were successful, the list of colorcharts is stored inside the detector itself, we will see in the next step on how to extract it. By default it will detect atmost one chart, but you can tune the third parameter, nc(maximum number of charts), for detecting more charts.
-#  **Get List of ColorCheckers**
    @code{.cpp}
    std::vector<cv::Ptr<mcc::CChecker>> checkers;
    detector->getListColorChecker(checkers);
    @endcode

    All the colorcheckers that were detected are now stored in the 'checkers' vector.

-#  **Draw the colorcheckers back to the image**
    @code{.cpp}

    for(Ptr<mcc::CChecker> checker : checkers)
    {
        // current checker
        Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
        cdraw->draw(image);
    }
    @endcode

    Loop through all the checkers one by one and then draw them.
