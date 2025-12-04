Tutorials for data augmentation module {#tutorial_table_of_content_imgaug}
===============================================================

Data augmentation techniques are widely used in deep learning training to expand
the training samples and overcome overfitting problem. imgaug module in OpenCV is
implemented in pure C++ and powered with efficient OpenCV image processing operations,
so it runs much faster and more efficient than Python-based implementations.

With the binding generator provided by OpenCV, imgaug can be used not only from C++, but also from
different languages like Python, Java, etc. Conversely, you can also adopt your code
easily from other languages to C++, which is especially useful when you want to deploy
a model with its data preprocessing pipeline from Python to production environment in C++.

-   @subpage tutorial_imgaug_basic_usage

    *Compatibility:* >= OpenCV 4.0

    *Author:* Chuyang Zhao

    Basic usage of imgaug module. Perform data augmentation on images.

-   @subpage tutorial_imgaug_object_detection

    *Compatibility:* >= OpenCV 4.0

    *Author:* Chuyang Zhao

    Use imgaug to perform data augmentation for object detection task.

-   @subpage tutorial_imgaug_pytorch

    *Compatibility:* >= OpenCV 4.0

    *Author:* Chuyang Zhao

    Use imgaug with PyTorch for different computer vision tasks.