// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_AVERAGEBLUR_HPP
#define OPENCV_AUGMENT_AVERAGEBLUR_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W AverageBlur : public Transform
{
public:
    /* @brief Constructor to initialize the average blur transformation with a range of kernel sizes
       @param minKernelSize the min size of average blur kernel
       @param maxKernelSize the max size of average blur kernel
       @param borderType the std::vector containing the different types of padding used
    */
    CV_WRAP AverageBlur(Size minKernelSize, Size maxKernelSize,const std::vector<int>& borderTypes = std::vector<int>());

    /* @brief Constructor to initialize the average blur transformation with a range of kernel sizes
       @param minKernelSize the min size of average blur kernel
       @param maxKernelSize the max size of average blur kernel
       @param borderType the type of padding used
    */
    CV_WRAP AverageBlur(int minKernelSize, int maxKernelSize, int borderType = BORDER_DEFAULT);

    /* @brief Constructor to initialize the average blur transformation with a specific kernel size
       @param kernelSize the size of average blur kernel
       @param borderType the type of padding used
    */
    CV_WRAP AverageBlur(Size kernelSize, int borderType = BORDER_DEFAULT);

    /* @brief Constructor to initialize the average blur transformation with a specific kernel size
       @param kernelSize the size of average blur kernel
    */
    CV_WRAP AverageBlur(int kernelSize);

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
    Size kernelSize;
    bool sameXY;
    std::vector<int> borderTypes;
    int borderType;

};

}} //namespace cv::augment

#endif
