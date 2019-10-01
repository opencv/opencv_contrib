Upscaling images: single-output {#tutorial_dnn_superres_upscale_image_single}
===========================

In this tutorial you will learn how to use the 'dnn_superres' interface to upscale an image via pre-trained neural networks.

Building
----

When building OpenCV, run the following command to build the 'dnn_superres' module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_superres=ON <opencv_source_dir>
```

Or make sure you check the dnn_superres module in the GUI version of CMake: cmake-gui.

Source Code of the sample
-----------

@includelineno dnn_superres/samples/dnn_superres.cpp

Explanation
-----------

-#  **Set header and namespaces**
    @code{.cpp}
    #include <opencv2/dnn_superres.hpp>
    using namespace std;
    using namespace cv;
    using namespace dnn;
    using namespace dnn_superres;
    @endcode

    If you want you can set the namespace like the code above.
-#  **Create the Dnn Superres object**
    @code{.cpp}
    DnnSuperResImpl sr;
    @endcode

    This is just to create the object, register the custom dnn layers and get access to the class functions.
-#  **Read the model**
    @code{.cpp}
    path = "models/FSRCNN_x2.pb"
    sr.readModel(path);
    @endcode

    This reads the TensorFlow model from the .pb file. Here 'path' is one of the pre-trained Tensorflow models' path file. You can download the models from OpenCV's GitHub, in the 'dnn_superres' module.
-#  **Set the model**
    @code{.cpp}
    sr.setModel("fsrcnn", 2);
    @endcode

    Depending on the model you want to run, you have to set the algorithm and upscale factor. This is to know the desired algorithm and scale, even if you change the .pb file's name. For example: if you chose FSRCNN_x2.pb, your algorithm and scale will be 'fsrcnn' and 2, respectively. (Other algorithm options include "edsr", "espcn" and "lapsrn".)
-#  **Upscale an image**
    @code{.cpp}
    Mat img = cv::imread(img_path);
    Mat img_new;
    sr.upsample(img, img_new);
    @endcode

    Now we can upscale any image. Load an image via the standard 'imread' function and create a new Mat for the destination image. Then simple
    upscale. Your upscaled image is located in 'img_new'.

Original: ![](images/input.jpg)
Upscaled Image via FSRCNN: ![](images/fsrcnnOutput.jpg)
Upscaled Image via Bicubic Interpolation: ![](images/bicubicOutput.jpg)