// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_SHIFT_HPP
#define OPENCV_AUGMENT_SHIFT_HPP
#include <opencv2/augment/transform.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W Shift : public Transform
{
public:
    /* @brief Constructor
       @param amount the percentage of the image shifted (same shift for both dimensions)
    */
    CV_WRAP Shift(float amount);

    /* @brief Constructor
       @param amount the percentage of the image shifted (x,y)
    */
    CV_WRAP Shift(Vec2f amount);

    /* @brief Constructor to specify range of percentages to shift with
       @param minAmount the minimum percentage to shift with (x,y)
       @param maxAmount the maximum percentage to shift with (x,y)
    */
    CV_WRAP Shift(Vec2f minAmount, Vec2f maxAmount);

    /* @brief Constructor to specify range of percentages to shift with
       @param minAmount the minimum percentage to shift with (same shift range for both dimensions)
       @param maxAmount the maximum percentage to shift with (same shift range for both dimensions)
    */
    CV_WRAP Shift(float minAmount, float maxAmount);

    /* @brief Apply the shifting to a single image
       @param src Input image to be shifted
       @param dst Output (shifted) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst) override;

    /* @brief Apply the shifting for a single point
       @param src Input point to be shifted
    */
    Point2f virtual point(const Point2f& src) override;

    /* @brief Apply the transformation for a rectangle
       @param box Rect2f consisting of (x1, y1, w, h) corresponding to (top left point, size)
    */
    virtual Rect2f rectangle(const Rect2f& src) override;

    /* @brief choose number of rows and columns to be shifted
    */
    void virtual init(const Mat& srcImage) override;

private :
  int rowsShifted, columnsShifted;
  Mat translationMat;
  Vec2f minAmount, maxAmount, amount;
  bool randomAmount;

};

}} //namespace cv::augment

#endif
