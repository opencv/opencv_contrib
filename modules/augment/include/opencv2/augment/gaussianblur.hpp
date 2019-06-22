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
       @param minKernelSize the min sigma value of the gaussian blur kernel
       @param maxKernelSize the max sigma value of the gaussian blur kernel
    */
    CV_WRAP GaussianBlur(int minKernelSize, int maxKernelSize, float minSigma, float maxSigma);

    /* @brief Constructor to initialize the gaussian blur transformation with a range of kernel sizes and sigma will be decided based on kernel size
       @param minKernelSize the min size of gaussian blur kernel
       @param maxKernelSize the max size of gaussian blur kernel
    */
    CV_WRAP GaussianBlur(int minKernelSize, int maxKernelSize);

    /* @brief Apply the gaussian blue to a single image
        @param src Input image to be blurred
        @param dst Output (blurred) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst);

    /* @brief choose an random kernel size and sigma from the specified ranges to apply in next transformations
    */
    void virtual init(const Mat& srcImage);

private :
    int minKernelSize;
    int maxKernelSize;
    float minSigma;
    float maxSigma;
    int currentKernelSize;
    float currentSigma;

};

}} //namespace cv::augment

#endif
