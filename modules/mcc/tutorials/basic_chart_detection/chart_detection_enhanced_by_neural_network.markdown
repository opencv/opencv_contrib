Detecting colorcheckers using neural network{#tutorial_mcc_chart_detection_enhanced_by_neural_network}
===========================

In this tutorial you will learn how to use the neural network to boost up the accuracy of the chart detection algorithm.

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

You can run the sample code by doing

```run
<path_of_your_opencv_build_directory>/bin/example_mcc_chart_detection_with_network -t=<type_of_chart> -m=<path_to_neural_network> -pb=<path_to_models_pbtxt> -v=<optional_path_to_video_if_not_provided_webcam_will_be_used.mp4> --ci=<optional_camera_id_needed_only_if_video_not_provided> --nc=<optional_maximum_number_of_charts_in_image> --use_gpu <optional_should_gpu_be_used>

``'

* -t=#  is the chart type where 0 (Standard), 1 (DigitalSG), 2 (Vinyl)
* --ci=#  is the camera ID where 0 (default is the main camera), 1 (secondary camera) etc
* --nc=#  By default its values is 1 which means only the best chart will be detected

Example:

```
Simple run on CPU (GPU wont be used)
/home/opencv/build/bin/example_mcc_chart_detection_with_network -t=0 -m=/home/model.pb --pb=/home/model.pbtxt -v=mcc24.mp4
```

```
To run on GPU
/home/opencv/build/bin/example_mcc_chart_detection_with_network -t=0 -m=/home/model.pb --pb=/home/model.pbtxt -v=mcc24.mp4 --use_gpu

To run on GPU and detect the best 5 charts (Detections can be less than 5 but not more than 5)
/home/opencv/build/bin/example_mcc_chart_detection_with_network -t=0 -m=/home/model.pb --pb=/home/model.pbtxt -v=mcc24.mp4 --use_gpu --nc=5
```

@includelineno mcc/samples/chart_detection_with_network.cpp


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

-#  **Load the model**
    @code{.cpp}
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path, pbtxt_path);

    @endcode

    Load the model, here the model supplied with model was trained in tensorflow so we are loading it in tensorflow, but if you have some other model trained in some other framework you can use that also.

-#  **(Optional) Set the dnn backend to CUDA**
    @code{.cpp}
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    @endcode

    Models run much faster on CUDA, so use CUDA if possible.
-#  **Run the detector**
    @code{.cpp}
    detector->process(image, chartType, max_number_charts_in_image, true);
    @endcode

    If the detector successfully detects atleast one chart, it return true otherwise it returns false. In the above given code we print a failure message if no chart were detected. Otherwise if it were successful, the list of colorcharts is stored inside the detector itself, we will see in the next step on how to extract it. The fourth parameter is for deciding whether to use the net or not.
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
