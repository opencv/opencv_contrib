Data augmentation with imgaug in object detection {#tutorial_imgaug_object_detection}
==============================

@tableofcontents

@prev_tutorial{tutorial_imgaug_basic_usage}
@next_tutorial{tutorial_imgaug_pytorch}

|    |    |
| -: | :- |
| Author | Chuyang Zhao |
| Compatibility | OpenCV >= 4.0 |


Introduction
------
In the previous tutorial, we demonstrate how to use imgaug to perform transforms on pure images.
In some tasks, the inputs contains not only images but also the annotations. We extend the imgaug
module to support most of the main stream computer vision tasks. Here we demonstrate how to use imgaug for
object detection.


Goal
----
In this tutorial, you will learn:
- How to use imgaug to perform data augmentation for data in object detection task


The inputs of object detection task contain source input image, the annotated bounding boxes, and the class labels
for each bounding box. In C++, the input image is represented as cv::Mat, the annotated bounding boxes can be represented
as `std::vector<cv::Rect>` in which each bounding box is represented as a cv::Rect. The annotated labels for objects in
bounding boxes can be represented as `std::vector<int>`.

The data augmentation methods for object detection are implemented under namespace cv::imgaug::det, you can
find more details of all implemented methods in documentation cv::imgaug::det.


Usage
-----
### Apply single data augmentation method
@add_toggle_cpp

To use the imgaug module in object detection, we need to include the header file:

@code{.cpp}
#include <opencv2/imgaug.hpp>
@endcode

Take random flip as an example, we first initialize the cv::imgaug::det::RandomRotation instance by:

@code{.cpp}
imgaug::det::RandomRotation aug(Vec2d(-30, 30));
@endcode

The first argument cv::Vec2d(-30, 30) is the degree range in which the rotation degree will be uniformly sampled from.

Then we read the source image and load its annotation data, which including bounding boxes and class labels.
In the following example, the annotation data contains two bounding boxes and two class labels:

@code{.cpp}
Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);

std::vector<Rect> bboxes{
Rect{112, 40, 249, 343},
Rect{61, 273, 113, 228}
};

std::vector<int> labels {1, 2};
@endcode

The bounding boxes on the source image is as follows:

![](images/det_src.jpg)

Then we call random rotation operation on the given image and its annotations by imgaug::det::RandomRotation::call:

@code{.cpp}
aug.call(src, dst, bboxes, labels);
@endcode

The augmented image and its annotation are as follows:

![](images/det_rotation_out.jpg)

Complete code of this example:
@include imgaug/samples/det_sample.cpp.

@end_toggle

@add_toggle_python

In Python, you should import the following package:

@code{.py}
from cv2 import imgaug
@endcode

Be aware the data augmentation methods for object detection are all in submodule `cv2.imgaug.det`.

Take random flip as an example, we first initialize the cv::imgaug::det::RandomRotation instance by:

@code{.py}
aug = imgaug.det.RandomRotation((-30, 30))
@endcode

The first argument (-30, 30) is the degree range in which the rotation degree will be uniformly sampled from.

Then we read the source image and load its annotation data, which including bounding boxes and class labels.
In the following example, the annotation data contains two bounding boxes and two class labels:

@code{.py}
src = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)

bboxes = [
    (112, 40, 249, 343),
    (61, 273, 113, 228)
]

labels = [1, 2]
@endcode

@note We represent the bounding box with a four-elements tuple (x, y, w, h),
in which x and y are the coordinates of the top left corner of the bounding box,
w and h are the width and height of the bounding box. The binding generator will
convert the tuple into cv::Rect in C++. Please make sure the elements in the tuple
is in the right order.

The bounding boxes on the source image is as follows:

![](images/det_src.png)

Then we call random rotation operation on the given image and its annotations by imgaug::det::RandomRotation::call:

@code{.py}
dst = aug.call(src, bboxes, labels)
@endcode

The augmented image and its annotation are as follows:

![](images/det_rotation_out.png)

Complete code of this example:
@include imgaug/samples/det_sample.cpp

@end_toggle

### Compose multiple data augmentation methods
@add_toggle_cpp
Compose multiple data augmentation methods into one in object detection module (cv::imgaug::det) is similar to basic imgaug module (cv::imgaug).
We also need to initialize multiple data augmentation instances in imgaug::det :

@code{.cpp}
imgaug::det::RandomRotation randomRotation(Vec2d(-30, 30));
imgaug::det::RandomFlip randomFlip(1);
imgaug::det::Resize resize(Size(224, 224));
@endcode

Different from data augmentation classes in cv::imgaug, data augmentation classes in cv::imgaug::det are inherited from base class
cv::imgaug::det::Transform, so we need to use pointer of type cv::imgaug::det::Transform to store the address of each data augmentation
instances in det module. We store their pointers in a vector and then initialize the imgaug::det::Compose class with this vector:

@code{.cpp}
std::vector<Ptr<imgaug::det::Transform> > transforms {&randomRotation, &randomFlip, &resize};
imgaug::det::Compose aug(transforms);
@endcode

@warning You cannot compose data augmentation methods in cv::imgaug::det module with methods in cv::imgaug module,
because they do not inherit from the same base class. You can only compose methods in the same module.

Then we can call the compose method on the given image and its annotation as follows:

@code{.cpp}
aug.call(src, dst, bboxes, labels);
@endcode

The augmented image and its annotation are as follows:

![](images/det_compose_out.png)

Complete code of this example:
@include imgaug/samples/det_compose_sample.cpp

@end_toggle

@add_toggle_python
Compose multiple data augmentation methods into one in object detection module (cv::imgaug::det) is similar to basic imgaug module (cv::imgaug).
We also need to initialize multiple data augmentation instances in imgaug::det :

@code{.py}
randomRotation = imgaug.det.RandomRotation((-30, 30))
randomFlip = imgaug.det.RandomFlip(1)
resize = imgaug.det.Resize((224, 224))
@endcode

We save all these methods in a list `transforms` as parameter to initialize Compose class.

@code{.py}
transforms = [randomRotation, randomFlip, resize]
aug = Compose(transforms)
@endcode

@warning You cannot compose data augmentation methods in cv::imgaug::det module with methods in cv::imgaug module,
because they do not inherit from the same base class. You can only compose methods in the same module.

Then we can call the compose method on the given image and its annotation as follows:

@code{.py}
dst = aug.call(src, bboxes, labels)
@endcode

The augmented image and its annotation are as follows:

![](images/det_compose_out.png)

Complete code of this example:
@include imgaug/samples/det_compose_sample.cpp

@end_toggle