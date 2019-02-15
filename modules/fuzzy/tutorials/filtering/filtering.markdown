Filtering using F-transform {#tutorial_fuzzy_filtering}
=============

Goal
====
This tutorial demonstrates to you how to use F-transform for image filtering. You will see:

-   basic theory behind,
-   illustration of different settings.

Fuzzy transform application
====
As I shown in previous tutorial, F-transform is a tool of fuzzy mathematics highly usable in image processing. Let me rewrite the formula using kernel \f$g\f$ introduced before as well:

\f[
    F^0_{kl}=\frac{\sum_{x=0}^{2h+1}\sum_{y=0}^{2h+1} \iota_{kl}(x,y) g(x,y)}{\sum_{x=0}^{2h+1}\sum_{y=0}^{2h+1} g(x,y)},
\f]

where \f$\iota_{kl} \subset I\f$ centered to pixel \f$(k \cdot h,l \cdot h)\f$ and \f$g\f$ is a kernel. More details can be found in related papers.

Code
====
@include fuzzy/samples/fuzzy_filtering.cpp

Explanation
====
Image filtering changes input in a defined way to enhance or simply change some concrete feature. Let me demonstrate some simple blur.

As a first step, we load input image.

@code{.cpp}
    // Input image
    Mat I = imread("input.png");
@endcode

Following the F-transform formula, we must specify a kernel.

@code{.cpp}
    // Kernel cretion
    Mat kernel1, kernel2;

    ft::createKernel(ft::LINEAR, 3, kernel1, 3);
    ft::createKernel(ft::LINEAR, 100, kernel2, 3);
@endcode

> So now, we have two kernels that differ in `radius`. Bigger radius leads to bigger blur.

The filtering itself is applied as shown below.

@code{.cpp}
    // Filtering
    Mat output1, output2;

    ft::filter(I, kernel1, output1);
    ft::filter(I, kernel2, output2);
@endcode

Output images look as follows.

![input, output1 (radius 3), output2 (radius 100)](images/fuzzy_filt_output.jpg)