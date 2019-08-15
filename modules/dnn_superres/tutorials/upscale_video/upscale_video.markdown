Upscaling video {#tutorial_dnn_superres_upscale_video}
===========================

In this tutorial you will learn how to use the 'dnn_superres' interface to upscale video via pre-trained neural networks.

Building
----

When building OpenCV, run the following command to build the 'dnn_superres' module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_superres=ON <opencv_source_dir>
```

Or make sure you check the dnn_superres module in the GUI version of CMake: cmake-gui.

Source Code of the sample
-----------

@includelineno dnn_superres/samples/dnn_superres_video.cpp

Explanation
-----------

-#  **Set header and namespaces**
    @code{.cpp}
    #include <opencv2/dnn_superres.hpp>
    using namespace std;
    using namespace cv;
    using namespace dnn_superres;
    @endcode
-#  **Create the Dnn Superres object**
    @code{.cpp}
    DnnSuperResImpl sr;
    @endcode
    Instantiate a dnn super-resolution object.
-#  **Read the model**
    @code{.cpp}
    path = "models/ESPCN_x2.pb"
    sr.readModel(path);
    @endcode
    Read the model from the given path.
-#  **Set the model**
    @code{.cpp}
    sr.setModel("espcn", 2);
    @endcode
    Sets the algorithm and scaling factor.
-#  **Upscale a video**
    @code{.cpp}
    sr.upsampleVideo(input_path, output_path);
    @endcode
    Now we can upscale a video. Input path is the path to the video file to be upscaled, and output_path is the upscaled video file that will be saved.
