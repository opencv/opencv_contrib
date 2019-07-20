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

    /* @brief Constructor
       @param width the width the images will be cropped to
       @param width the height the images will be cropped to
       @param origin is the coordinate of the new origin
    */
    CV_WRAP CropToFixedSize(int width, int height, Point origin);

    /* @brief Apply the cropping to a single image
       @param src Input image to be cropped
       @param dst Output (cropped) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst) override;

    /* @brief Apply the cropping transformation a single point
       @param src Input point to be mapped to the new coordinates after cropping
    */
    Point2f virtual point(const Point2f& src) override;

    /* @brief Apply the transformation for a rectangle
       @param box Rect2f consisting of (x1, y1, w, h) corresponding to (top left point, size)
    */
    virtual Rect2f rectangle(const Rect2f& src) override;

    /* @brief choose the center of the random cropping
    */
    void virtual init(const Mat& srcImage) override;

private:
    int width, height;
    int originX, originY;
    bool randomXY;
};

}} //namespace cv::augment

#endif
