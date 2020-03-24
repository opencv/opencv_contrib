Upscaling images: single-output {#tutorial_dnn_superres_upscale_image_single}
===========================

In this tutorial you will learn how to use the 'dnn_superres' interface to upscale an image via pre-trained neural networks. It works in C++ and Python.

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

You can run the sample code by doing

```run
<path_of_your_opencv_build_directory>/bin/example_dnn_superres_dnn_superres <path_to_image.png> <algo_string> <upscale_int> <model_path.pb>
```

Example:

```run
/home/opencv/build/bin/example_dnn_superres_dnn_superres /home/image.png edsr 2 /home/EDSR_x2.pb
```

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

An example in python
-----------
@code{.py}
import cv2
from cv2 import dnn_superres

# Create an SR object - only function that differs from c++ code
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('./image.png')

# Read the desired model
path = "EDSR_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("./upscaled.png", result)
@endcode


Original: ![](images/input.jpg)
Upscaled Image via FSRCNN: ![](images/fsrcnnOutput.jpg)
Upscaled Image via Bicubic Interpolation: ![](images/bicubicOutput.jpg)