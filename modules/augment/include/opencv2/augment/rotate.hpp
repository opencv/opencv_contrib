// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_ROTATE_HPP
#define OPENCV_AUGMENT_ROTATE_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W Rotate : public Transform
{
public:
    /* @brief Constructor
        @param _angleRange the range of angle (in degrees) values to rotate the image
    */
    CV_WRAP Rotate(float minAngle, float maxAngle);

    /* @brief Constructor to initialize the rotation transformation with a specific angle (in degrees)
    */
    CV_WRAP Rotate(float angle);

    /* @brief Apply the rotation to a single image
        @param _src Input image to be flipped
        @param _dst Output (rotated) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst);

    /* @brief Apply the rotation for a single point
        @param _src Input point to be rotated
    */
    Point2f virtual point(const Point2f& src);

    /* @brief choose an angle from the specified range to apply in next transformations
    */
    void virtual init(const Mat& srcImage);

private :
    float minAngle, maxAngle;
    Mat rotationMat;

};

}} //namespace cv::augment

#endif
