Upscaling images: multi-output {#tutorial_dnn_superres_upscale_image_multi}
===========================

In this tutorial you will learn how to use the 'dnn_superres' interface to upscale an image via a multi-output pre-trained neural network.
OpenCVs dnn module supports accessing multiple nodes in one inference, if the names of the nodes are given.
Currently there is one model included that is capable of giving more output in one inference run, that is the LapSRN model.
LapSRN supports multiple outputs with one forward pass. It can now support 2x, 4x, 8x, and (2x, 4x) and (2x, 4x, 8x) super-resolution.
The uploaded trained model files have the following output node names:
- 2x model: NCHW_output
- 4x model: NCHW_output_2x, NCHW_output_4x
- 8x model: NCHW_output_2x, NCHW_output_4x, NCHW_output_8x

Building
----

When building OpenCV, run the following command to build all the contrib module:

```make
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules/
```

Or only build the dnn_superres module:

```make
cmake -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules/dnn_superres
```

Or make sure you check the dnn_superres module in the GUI version of CMake: cmake-gui.

Source Code of the sample
-----------

Run the sample code with the following command

```run
./bin/example_dnn_superres_dnn_superres_multioutput path/to/image.png 2,4 NCHW_output_2x,NCHW_output_4x \
path/to/opencv_contrib/modules/dnn_superres/models/LapSRN_x4.pb
```


@includelineno dnn_superres/samples/dnn_superres_multioutput.cpp

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
    path = "models/LapSRN_x8.pb"
    sr.readModel(path);
    @endcode

    Read the model from the given path.

-#  **Set the model**
    @code{.cpp}
    sr.setModel("lapsrn", 8);
    @endcode

    Sets the algorithm and scaling factor. The last (largest) scaling factor should be given here.

-#  **Give the node names and scaling factors**
     @code{.cpp}
     std::vector<int> scales{2, 4, 8}
     std::vector<int> node_names{'NCHW_output_2x','NCHW_output_4x','NCHW_output_8x'}
     @endcode

     Set the scaling factors, and the output node names in the model.

-#  **Upscale an image**
    @code{.cpp}
    Mat img = cv::imread(img_path);
    std::vector<Mat> outputs;
    sr.upsampleMultioutput(img, outputs, scales, node_names);
    @endcode

    Run the inference. The output images will be stored in a Mat vector.
