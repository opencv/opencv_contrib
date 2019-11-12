Image Inpainting {#tutorial_xphoto_inpainting}
================

Introduction
------------
In this tutorial we will show how to use the algorithm Rapid Frequency Selective Reconstructiom (FSR) for image inpainting.

Basics
------
Image Inpainting is the process of reconstructing damaged or missing parts of an image.
This is achieved by replacing distorted pixels by pixels similar to the neighboring ones. There are several algorithms for inpainting, using different approaches for such replacement.

One of those algorithms is called **Rapid Frequency Selectice Reconstruction (FSR)**.
FSR reconstructs image signals by exploiting the property that small areas of images can be represented sparsely in the Fourier domain. See @cite GenserPCS2018 and @cite SeilerTIP2015 for details.

FSR can be utilized for the following areas of application:

-# **Error Concealment (Inpainting)**:
    The sampling mask indicates the missing pixels of the distorted input image to be reconstructed.

-# **Non-Regular Sampling**:
    For more information on how to choose a good sampling mask, please review @cite GroscheICIP2018 and @cite GroscheIST2018.

Example
-------
The following sample code shows how to use FSR for inpainting.
The non-zero pixels of the error mask indicate valid image area, while zero pixels indicate area to be reconstructed.
You can create an arbitrary mask manually using tools like Paint or GIMP. Start with a plain white image and draw some distortions in black.

  @code{.cpp}

  #include <opencv2/opencv.hpp>
  #include <opencv2/xphoto/inpainting.hpp>
  #include <iostream>

  using namespace cv;

  int main(int argc, char** argv)
  {
      // read image and error pattern
      Mat original_, mask_;
      original_ = imread("images/kodim22.png");
      mask_ = imread("images/pattern_random.png", IMREAD_GRAYSCALE);

      // make sure that mask and source image have the same size
      Mat mask;
      resize(mask_, mask, original_.size(), 0.0, 0.0, cv::INTER_NEAREST);

      // distort image
      Mat im_distorted(original_.size(), original_.type(), Scalar::all(0));
      original_.copyTo(im_distorted, mask); // copy valid pixels only (i.e. non-zero pixels in mask)

      // reconstruct the distorted image
      // choose quality profile fast (xphoto::INPAINT_FSR_FAST) or best (xphoto::INPAINT_FSR_BEST)
      Mat reconstructed;
      xphoto::inpaint(im_distorted, mask, reconstructed, xphoto::INPAINT_FSR_FAST);

      imshow("orignal image", original_);
      imshow("distorted image", im_distorted);
      imshow("reconstructed image", reconstructed);
      waitKey();

      return 0;
  }
  @endcode

Original and distorted image:
![image](images/originalVSdistorted.jpg)

Reconstruction:
![image](images/reconstructed_fastVSbest.jpg)

Left image: fast quality profile (run time 8 seconds). Right image: best quality profile (1 minute 51 seconds).

Additional Resources
--------------------
[Comparison of FSR to existing inpainting methods in OpenCV](https://github.com/opencv/opencv_contrib/files/3730212/inpainting_comparison.pdf)
