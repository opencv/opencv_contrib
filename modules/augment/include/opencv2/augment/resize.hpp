// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_RESIZE_HPP
#define OPENCV_AUGMENT_RESIZE_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W Resize : public Transform
{
public:
    /* @brief Constructor
       @param size the size the images will be resized to
       @param interpolations vector of the interpolation techniques to be chosen at random
    */
    CV_WRAP Resize(Size size, const std::vector<int>& interpolations = std::vector<int>());

    /* @brief Constructor
       @param size the size the images will be resized to
       @param interpolation specific interpolation type to be used
    */
    CV_WRAP Resize(Size size, int interpolation);

    /* @brief Apply the resizing to a single image
       @param src Input image to be resized
       @param dst Output (resized) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst) override;

    /* @brief Apply the resizing to a single mask
      @param src Input mask to be resized
      @param dst Output (resized) mask
    */
    CV_WRAP virtual void mask(InputArray src, OutputArray dst) override;

    /* @brief Apply the resizing transformation a single point
       @param src Input point to be mapped to the new coordinates after resizing
    */
    Point2f virtual point(const Point2f& src) override;

    /* @brief Apply the transformation for a rectangle
       @param box Rect2f consisting of (x1, y1, w, h) corresponding to (top left point, size)
    */
    virtual Rect2f rectangle(const Rect2f& src) override;

    void virtual init(const Mat& srcImage) override;

private :
    Size size;
    std::vector<int> interpolations;
    int interpolation;

};

}} //namespace cv::augment

#endif
