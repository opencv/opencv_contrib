// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_CROPTOFIXEDSIZE_HPP
#define OPENCV_AUGMENT_CROPTOFIXEDSIZE_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W CropToFixedSize : public Transform
{
public:
    /* @brief Constructor
        @param width the width the images will be cropped to
        @param width the height the images will be cropped to
    */
    CV_WRAP CropToFixedSize(int width, int height);

    /* @brief Apply the cropping to a single image
        @param src Input image to be cropped
        @param dst Output (cropped) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst);

    /* @brief Apply the cropping transformation a single point
        @param src Input point to be mapped to the new coordinates after cropping
    */
    Point2f virtual point(const Point2f& src);

    /* @brief choose the center of the random cropping
    */
    void virtual init(const Mat& srcImage);

private:
    int width, height;
    int originX, originY;
        
};

}} //namespace cv::augment

#endif
