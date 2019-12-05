Inpainting using F-transform {#tutorial_fuzzy_inpainting}
=============

Goal
====
In this tutorial, you will learn how image inpainting using F-transform works. It consists in:

-   basic theory behind,
-   three different algorithms.

Introduction
====
The goal of this tutorial is to show that the inverse F-transform can be used for image reconstruction. By the image reconstruction, we mean a reconstruction of a corrupted image where corruption is everything that the original image does not include. It can be noise, text, scratch, etc. Proposal is to solve the problem of reconstruction with the help of an approximation technique. This means that we will be looking for an approximating image which is close to the given one and at the same time, does not contain what we recognize as the corruption. This task is called _image inpainting_.

Fuzzy transform application
====
As I shown in previous tutorial, F-transform is a tool of fuzzy mathematics highly usable in image processing. Let me rewrite the formula using kernel \f$g\f$ introduced before as well:

\f[
    F^0_{kl}=\frac{\sum_{x=0}^{2h+1}\sum_{y=0}^{2h+1} \iota_{kl}(x,y) g(x,y)}{\sum_{x=0}^{2h+1}\sum_{y=0}^{2h+1} g(x,y)},
\f]

where \f$\iota_{kl} \subset I\f$ centered to pixel \f$(k \cdot h,l \cdot h)\f$ and \f$g\f$ is a kernel. For purpose of image processing, a binary mask \f$S\f$ is used such as

\f[
	g^s_{kl} = g \circ s_{kl}
\f]

where \f$s_{k,l} \subset S\f$. Subarea \f$s\f$ of mask \f$S\f$ corresponds with subarea \f$\iota\f$ of image \f$I\f$. Operator \f$\circ\f$ is element-wise matrix multiplication (Hadamard product). Formula is updated to

\f[
    F^0_{kl}=\frac{\sum_{x=0}^{2h+1}\sum_{y=0}^{2h+1} \iota_{kl}(x,y) g^s(x,y)}{\sum_{x=0}^{2h+1}\sum_{y=0}^{2h+1} g^s(x,y)}.
\f]

More details can be found in related papers.

Code
====

@include fuzzy/samples/fuzzy_inpainting.cpp

Explanation
====
The sample below demonstrates the usage of image inpainting. Three artificial images are created using the same input and three different type of corruption. In the real life usage, the input image will be already presented but here we created it by ourselves.

First of all, we must load our image and three masks used for artificial damage creation.

@code{.cpp}
    // Input image
    Mat I = imread("input.png");

    // Various masks
    Mat mask1 = imread("mask1.png", IMREAD_GRAYSCALE);
    Mat mask2 = imread("mask2.png", IMREAD_GRAYSCALE);
    Mat mask3 = imread("mask3.png", IMREAD_GRAYSCALE);
@endcode

> See that mask must be loaded as `IMREAD_GRAYSCALE`.

In the next step, the masks are used for damaging our input image.

@code{.cpp}
    // Apply the damage
    Mat input1, input2, input3;

    I.copyTo(input1, mask1);
    I.copyTo(input2, mask2);
    I.copyTo(input3, mask3);
@endcode

Using the masks, we applied three different kind of corruption on the same input image. Here is the result.

![input1, input2 and input3](images/fuzzy_inp_input.jpg)

> Do not forget that in real life usage, images `input1`, `input2` and `input3` are created naturally and used as the input directly.

Declaration of output images follows. In the following lines, the method of inpainting is applied. Let me explain three different algorithms one by one.

First of them is `ONE_STEP`.

@code{.cpp}
    ft::inpaint(input1, mask1, output1, 2, ft::LINEAR, ft::ONE_STEP);
@endcode

The `ONE_STEP` algorithm simply compute direct F-transform ignoring damaged parts using kernel with radius `2` (as specified in the method calling). Inverse F-transform fill up the missing area using values from the components nearby. It is up to you to choose radius which is big enough.

Second is `MULTI_STEP`.

@code{.cpp}
    ft::inpaint(input2, mask2, output2, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(input3, mask3, output3, 2, ft::LINEAR, ft::MULTI_STEP);
@endcode

`MULTI_STEP` algorithm works in the same way but defined radius (`2` in this case) is automatically increased if it is found insufficient. If you want to fill up the hole and you are not sure how big radius you need, you can choose `MULTI_STEP` and let the computer decide. The lowest possible will be found.

Last one is `ITERATIVE`.

@code{.cpp}
    ft::inpaint(input3, mask3, output4, 2, ft::LINEAR, ft::ITERATIVE);
@endcode

Best choice in majority of cases is `ITERATIVE`. This way of processing use small radius of basic functions for small kind of damage and higher ones for bigger holes.

![output1 (ONE_STEP), output2 (MULTI_STEP), output3 (MULTI_STEP), output4 (ITERATIVE)](images/fuzzy_inp_output.jpg)
