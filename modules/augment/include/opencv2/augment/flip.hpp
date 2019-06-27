// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_FLIP_HPP
#define OPENCV_AUGMENT_FLIP_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W FlipHorizontal : public Transform
{
public:
    /* @brief Constructor
    */
    CV_WRAP FlipHorizontal();

    /* @brief Apply the horizontal flipping to a single image
        @param _src Input image to be flipped
        @param _dst Output (flipped) image
    */
    CV_WRAP void image(InputArray _src, OutputArray _dst);

    /* @brief Apply the flipping for a single point
        @param src Input point to be flipped
    */
    Point2f point(const Point2f& src);

};


class CV_EXPORTS_W FlipVertical : public Transform
{
public:
    /* @brief Constructor
    */
    CV_WRAP FlipVertical();

    /* @brief Apply the vertical flipping to a single image
        @param _src Input image to be flipped
        @param _dst Output (flipped) image
    */
    CV_WRAP virtual void image(InputArray _src, OutputArray _dst);

    /* @brief Apply the flipping for a single point
        @param src Input point to be flipped
    */
    Point2f virtual point(const Point2f& src);

};

}} //namespace cv::augnment

#endif
