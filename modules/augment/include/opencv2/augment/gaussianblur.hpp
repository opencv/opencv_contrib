// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_AUGMENT_GAUSSIANBLUR_HPP
#define OPENCV_AUGMENT_GAUSSIANBLUR_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W GaussianBlur : public Transform
{
public:
    /* @brief Constructor to initialize the gaussian blur transformation with a range of kernel sizes and range of sigma values
       @param minKernelSize the min size of gaussian blur kernel
       @param maxKernelSize the max size of gaussian blur kernel
       @param minSigmaX the min standard deviation in the x direction of the gaussian blur kernel
       @param maxSigmaX the max standard deviation in the x direction of the gaussian blur kernel
       @param minSigmaY the min standard deviation in the y direction of the gaussian blur kernel
       @param maxSigmaY the max standard deviation in the y direction of the gaussian blur kernel
    */
    CV_WRAP GaussianBlur(Size minKernelSize, Size maxKernelSize, float minSigmaX, float maxSigmaX, float minSigmaY, float maxSigmaY);

    /* @brief Constructor to initialize the gaussian blur transformation with a range of kernel sizes and range of sigma values (sigma in both directions is the same)
       @param minKernelSize the min size of gaussian blur kernel
       @param maxKernelSize the max size of gaussian blur kernel
       @param minSigma the min standard deviation in both directions of the gaussian blur kernel
       @param maxSigma the max standard deviation in both directions of the gaussian blur kernel
    */
    CV_WRAP GaussianBlur(int minKernelSize, int maxKernelSize, float minSigma, float maxSigma);

    /* @brief Constructor to initialize the gaussian blur transformation with a specific kernel size and a same standard deviation in both directions
       @param kernelSize the size of gaussian blur kernel
       @param sigma the standard deviation in both directions of the gaussian blur kernel
    */
    CV_WRAP GaussianBlur(int kernelSize, float sigma);

    /* @brief Constructor to initialize the gaussian blur transformation with a specific kernel size and sigmaX and sigmaY
       @param kernelSize the size of gaussian blur kernel
       @param sigmaX the standard deviation in x direction of the gaussian blur kernel
       @param sigmaY the standard deviation in y direction of the gaussian blur kernel
    */
    CV_WRAP GaussianBlur(Size kernelSize, float sigmaX, float sigmaY = 0);


    /* @brief Apply the gaussian blue to a single image
        @param src Input image to be blurred
        @param dst Output (blurred) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst) override;

    /* @brief Apply the transformation for a rectangle
       @param box Rect2f consisting of (x1, y1, w, h) corresponding to (top left point, size)
    */
    virtual Rect2f rectangle(const Rect2f& src) override;

    /* @brief choose an random kernel size and sigma from the specified ranges to apply in next transformations
    */
    void virtual init(const Mat&) override;

private :
    Size minKernelSize;
    Size maxKernelSize;
    float minSigmaX;
    float minSigmaY;
    float maxSigmaX;
    float maxSigmaY;
    Size kernelSize;
    float sigmaX;
    float sigmaY;
    bool sameXY;
};

}} //namespace cv::augment

#endif
