Data augmentation with imgaug {#tutorial_imgaug_basic_usage}
============================================================

@tableofcontents

@next_tutorial{tutorial_imgaug_object_detection}

|    |    |
| -: | :- |
| Author | Chuyang Zhao |
| Compatibility | OpenCV >= 4.0 |


Introduction
------
From [Wikipedia](https://en.wikipedia.org/wiki/Data_augmentation), **data augmentation** are techniques used to increase the amount of data
by adding slightly modified copies of already existing data or newly created synthetic data from existing data.
It acts as a regularizer and helps reduce overfitting when training a machine learning model.

In a narrow sense, data augmentation is to perform some sort of transforms on given images and generate the modified
images as additional training data, but broadly speaking, data augmentation can perform not only on images.
For computer vision tasks like object detection and semantic segmentation, the inputs contain not only images
but also annotation on the source images. So in these tasks, data augmentation should be able to perform transforms on
all these data.

The imgaug module implemented in OpenCV takes both these requirements into account. You can use the imgaug module
for a wide range of computer vision tasks.
The imgaug module in OpenCV is implemented in pure C++ and is backend with OpenCV efficient image processing operations,
so it runs much faster and more efficiently than other existing Python-based implementation such as torchvision. Powered with OpenCV, the imgaug module
is cross-platform and can convert to other languages easily. This is especially useful when we want to
deploy our model along with its data preprocessing pipeline to the production environment for better inference speed.
With this feature, we can also use imgaug on other devices such as embedded systems and mobile phones easily.

Goal
----
In this tutorial, you will learn:
- How to use **imgaug** to perform data augmentation for images
- How to compose multiple methods into one data augmentation method
- How to change the seed of the random number generator used in **imgaug**


Usage
-----
In this section, I will use some methods in imgaug to demonstrate how to use imgaug to perform data augmentation on images.
For the details of all the methods in imgaug, please refer to the documentation @ref cv::imgaug .

### Apply single data augmentation method
@add_toggle_cpp
In C++ environment, to use imgaug module you should include the header file:

@code{.cpp}
#include <opencv2/imgaug.hpp>
@endcode

We call the constructor of the data augmentation class to get its initialized instance.
Here we get the instance of cv::imgaug::RandomCrop to perform random crop on the given images. cv::imgaug::RandomCrop requires parameter `sz`
which is the size of the cropped area on the given image, here we pass cv::Size(300, 300) for this parameter.

@code{.cpp}
imgaug::RandomCrop randomCrop(cv::Size(300, 300));
@endcode

Then we read the source image in format cv::Mat and performs the data augmentation operation on it by calling cv::imgaug::RandomCrop::call function.

@code{.cpp}
Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
Mat dst;
randomCrop.call(src, dst);
@endcode

The original image is as follows:

![](images/lena.jpg)

You can display the augmented image after applying random crop by:

@code{.cpp}
imshow("result", dst);
waitKey(0);
@endcode

![](images/random_crop_out.jpg)

@end_toggle

@add_toggle_python
In Python, to use imgaug module you should import the following package:

@code{.py}
from cv2 import imgaug
@endcode

We call the constructor of the data augmentation class to get its initialized instance.
Here we get an instance of **cv::imgaug::RandomCrop** to perform random crop on the given images. **cv::imgaug::RandomCrop** requires a parameter `sz`
which is the size of the cropped area on the given image, here we pass a two-elements tuple `(300, 300)` for this parameter.

@code{.py}
randomCrop = imgaug.RandomCrop(sz=(300, 300))
@endcode

Then we read the source image with **cv::imread** and performs the data augmentation operation on it by calling **cv::imgaug::RandomCrop::call** function.

@code{.py}
src = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)
dst = randomCrop.call(src)
@endcode

The original image is as follows:

![](images/lena.jpg)

You can display the augmented image after applying random crop by:

@code{.py}
cv2.imshow("result", dst)
cv2.waitKey(0)
@endcode

![](images/random_crop_out.jpg)

@end_toggle

### Compose multiple data augmentation methods
@add_toggle_cpp
To compose multiple data augmentation methods into one, firstly you need to
initialize the data augmentation classes you want to use later:

@code{.cpp}
imgaug::RandomCrop randomCrop(cv::Size(300, 300));
imgaug::RandomFlip randomFlip(1);
imgaug::Resize resize(cv::Size(224, 224));
@endcode

Because in **cv::imgaug::Compose**, we call each data augmentation method by the pointer of their
base class **cv::imgaug::Transform**. We need to use a vector of type **cv::Ptr<cv::imgaug::Transform>** to
store the addresses of all data augmentation instances.

@code{.cpp}
std::vector<Ptr<imgaug::Transform> > transforms {&randomCrop, &randomFlip, &resize};
@endcode

Then we construct the **cv::imgaug::Compose** class by passing `transforms` as the required argument.

@code{.cpp}
imgaug::Compose aug(transforms);
@endcode

We call the compose method the same way as normal data augmentation methods. The composed
method will call all the methods in `transforms` on the given image sequentially:

@code{.cpp}
Mat src = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
Mat dst;
aug.call(src, dst);
@endcode

Here is the result we get:

![](images/compose_out.jpg)

@end_toggle

@add_toggle_python
To compose multiple data augmentation methods into one, firstly you need to
initialize the data augmentation classes you want to use later:

@code{.py}
randomCrop = imgaug.RandomCrop((300, 300))
randomFlip = imgaug.RandomFlip(1)
resize = imgaug.Resize((224, 224))
@endcode

We store all data augmentation instances in a list.

@code{.py}
transforms = [randomCrop, randomFlip, resize]
@endcode

Then we initialize the cv::imgaug::Compose class by passing the list of all data augmentation instances as the argument.

@code{.py}
aug = imgaug.Compose(transforms)
@endcode

We call the compose method the same way as normal data augmentation methods.
The composed method will apply all the data augmentation methods in transforms list to the given image sequentially.

@code{.py}
src = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)
dst = aug.call(src)
@endcode

Here is the result we get:

![](images/compose_out.jpg)

@end_toggle

### Change the seed of random number generator
@add_toggle_cpp
In imgaug, we use **cv::imgaug::rng** as our random number generator. The role of rng is to generate
random numbers for some random methods. For example, in cv::imgaug::RandomCrop we need to generate the coordinates
of the upper-left corner of the cropped rectangle randomly, in which we will use `rng` to generate random
numbers in valid range. When a random number is generated by `rng`, the internal state of `rng` will change.
Thus, we probably won't get the same result when we call the same method again. In the above process, the most
important thing is the initial state of `rng`, which determines the subsequent numbers `rng` generated. So in some
cases if you want to replicate other one's results, or if you want to make sure the random values generated will be
different the next time you run the same program. You can manually set the initial state of the `rng` by calling
**cv::imgaug::setSeed**. By default, if you don't manually set the initial state of `rng`, its initial state will be
set to the tick count since it was first initialized.

@code{.cpp}
int seed = 1234;
imgaug::setSeed(seed);
@endcode

@end_toggle

@add_toggle_python
In imgaug, we use **cv::imgaug::rng** as our random number generator. The role of rng is to generate
random numbers for some random methods. For example, in cv::imgaug::RandomCrop we need to generate the coordinates
of the upper-left corner of the cropped rectangle randomly, in which we will use `rng` to generate random
numbers in valid range. When a random number is generated by `rng`, the internal state of `rng` will change.
Thus, we probably won't get the same result when we call the same method again. In the above process, the most
important thing is the initial state of `rng`, which determines the subsequent numbers `rng` generated. So in some
cases if you want to replicate other one's results, or if you want to make sure the random values generated will be
different the next time you run the same program. You can manually set the initial state of the `rng` by calling
**cv::imgaug::setSeed**. By default, if you don't manually set the initial state of `rng`, its initial state will be
set to the tick count since it was first initialized.

@code{.py}
seed = 1234
imgaug.setSeed(seed)
@endcode

@end_toggle