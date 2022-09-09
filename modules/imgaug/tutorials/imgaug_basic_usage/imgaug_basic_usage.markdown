Data augmentation with imgaug {#tutorial_imgaug_basic_usage}
==============================

Goal
----
In this tutorial, you will learn:
- How to use imgaug to perform data augmentation for images
- How to use imgaug to perform data augmentation for data in object detection task

Basics
------
From [Wikipedia](https://en.wikipedia.org/wiki/Data_augmentation), **data augmentation** are techniques used to increase the amount of data
by adding slightly modified copies of already existing data or newly created synthetic data from existing data.
It acts as a regularizer and helps reduce overfitting when training a machine learning model.

In a narrow sense, data augmentation is to perform some sort of transforms on given images and generate the modified
images as additional training data, but broadly speaking, data augmentation can perform not only on images.
For computer vision tasks like object detection and semantic segmentation, the inputs contains not only images
but also annotation on the source images. So in these tasks, data augmentation should be able to perform transforms on 
all these data. 

The imgaug module implemented in OpenCV takes both these requirements into account. You can use imgaug module
for a wide range of computer vision tasks. Detailed usages and currently supported tasks are shown below.
The imgaug module in OpenCV is implemented in pure C++ and is backended with OpenCV efficient image processing operations,
so it runs much faster and more efficient than the existing Python-based implementation. Powered with OpenCV, imgaug module
is cross-platform and can convert to other language easily. This is especially useful when we want to 
deploy our model along with its data preprocessing pipeline to production environment for better inference speed. 
With this feature, we can also use imgaug in other devices such as embed system and mobile phone easily.

Usage
-----
### Basic Usage
In this section, I will use some methods in imgaug to demonstrate how to use imgaug to perform data augmentation on images.
For the details of all the methods in imgaug, please refer to the documentation @ref cv::imgaug .

#### Apply single data augmentation method
In C++ environment, you should include the header file `#include <opencv2/imgaug.hpp>` to use imgaug. 

We call the constructor of the data augmentation class to get its initialized instance.
Here we get the instance of `RandomCrop` to perform random crop on the given images. `RandomCrop` requires parameter `sz`
which is the size of the cropped area on the given image, here we pass cv::Size(300, 300) for this parameter.

```c++
imgaug::RandomCrop randomCrop(cv::Size(300, 300)); 
```


Then we read the source image in format cv::Mat and performs the data augmentation operation on it by calling `randomCrop.call()` function.

```c++
Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
Mat dst;
randomCrop.call(src, dst);
```

We can show what we get:
```c++
imshow("result", dst);
waitKey(0);
```

![](images/random_crop_out.jpg)

#### Compose multiple data augmentation methods
To compose multiple data augmentation methods into one, firstly you need to 
initialize the data augmentation classes you want to use later:
```c++
imgaug::RandomCrop randomCrop(cv::Size(300, 300));
imgaug::RandomFlip randomFlip(1);
imgaug::Resize resize(cv::Size(224, 224));
```

Because in @ref cv::imgaug::Compose , we call each data augmentation method by the pointer of their
parent class @ref cv::imgaug::Transform. We need to use a vector of type cv::Ptr<cv::imgaug::Transform> to 
store the addresses of all data augmentation instances.

```c++
std::vector<Ptr<imgaug::Transform> > transforms {&randomCrop, &randomFlip, &resize};
```

Then we construct the Compose class by passing `transforms` as the required argument.

```c++
imgaug::Compose aug(transforms);
```

We call the compose method the same way as normal data augmentation methods. The composed
method will call all the methods in `transforms` on the given image sequentially:

```c++
Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
Mat dst;
aug.call(src, dst);
```

Here is the result we get:

![](images/compose_out.jpg)

#### Change the seed of random number generator
In imgaug, we use @ref cv::imgaug::rng as our random number generator. Commonly, when
we don't manually set the initial state of rng, its initial state will be set to the tick count 
when it was initialized. But you can also manually set the seed of the rng by call cv::imgaug::setSeed():
```c++
int seed = 1234;
imgaug::setSeed(seed);
```

### Used in Object Detection
In the previous section, we demonstrate how to use imgaug to perform transforms on pure images.
In some tasks, the inputs contains not only images but also the annotations. We extend the imgaug
module to support most of the main stream computer vision tasks. Here we demonstrate how to use imgaug for
object detection.

The inputs of object detection task contains source input image, the annotated bounding boxes, and the class labels
for each bounding box. In C++, the input image is represented as cv::Mat, the annotated bounding boxes can be represented
as std::vector<cv::Rect> in which each bounding box is represented as a cv::Rect. The annotated labels for objects in 
bounding boxes can be represented as std::vector<int>.

The data augmentation methods for object detection are implemented under namespace cv::imgaug::det, you can 
find more details of all implemented methods in documentation @ref cv::imgaug::det.

To use the imgaug module in object detection, we need to include the header file <opencv2/imgaug.hpp>. 
Take random flip as an example, we first initialize the cv::imgaug::det::RandomRotation instance:

```c++
imgaug::det::RandomRotation aug(Vec2d(-30, 30));
```

Then we read the source image and load its annotation data, including bounding boxes and class labels.
In the following example, the annotation contains two bounding boxes and two class labels:

```c++
Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);

std::vector<Rect> bboxes{
        Rect{112, 40, 249, 343},
        Rect{61, 273, 113, 228}
};

std::vector<int> labels {1, 2};
```

The bounding boxes on the source image is as follows:

![](images/det_src.png)

Then we call random rotation operation on the given image and its annotations by imgaug::det::RandomRotation::call:

```c++
aug.call(src, dst, bboxes, labels);
```

The augmented image and its annotation are as follows:

![](images/det_rotation_out.png)

Full code of this example can be found at imgaug/samples/det_sample.cpp.

#### Compose multiple data augmentation methods
Compose multiple data augmentation methods into one in object detection module (cv::imgaug::det) is similar to basic imgaug module (cv::imgaug).
We also need to initialize multiple data augmentation instances in imgaug::det :

```c++
imgaug::det::RandomRotation randomRotation(Vec2d(-30, 30));
imgaug::det::RandomFlip randomFlip(1);
imgaug::det::Resize resize(Size(224, 224));
```

Different from data augmentation classes in cv::imgaug, data augmentation classes in cv::imgaug::det are inherited from base class
cv::imgaug::det::Transform, so we need to use pointer of type cv::imgaug::det::Transform to store the address of each data augmentation 
instances in det module. We store their pointers in a vector and then initialize the imgaug::det::Compose class with this vector:

```c++
std::vector<Ptr<imgaug::det::Transform> > transforms {&randomRotation, &randomFlip, &resize};
imgaug::det::Compose aug(transforms);
```

Then we can call the compose method on the given image and its annotation as follows:

```c++
aug.call(src, dst, bboxes, labels);
```

The augmented image and its annotation are as follows:

![](images/det_compose_out.png)

