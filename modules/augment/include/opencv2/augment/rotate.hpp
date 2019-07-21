// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_ROTATE_HPP
#define OPENCV_AUGMENT_ROTATE_HPP
#include <opencv2/augment/transform.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W Rotate : public Transform
{
public:
    /* @brief Constructor
        @param minAngle the minimum angle of rotation
        @param maxAngle the maximum angle of rotation
        @param interpolations the vector of interpolations used randomly in rotation
        @param borderType the vector of types of the border used when the image is rotated
        @param borderValue the value used in case of consant border
   */
    CV_WRAP Rotate(float minAngle, float maxAngle, std::vector<int> interpolations = std::vector<int>(), std::vector<int> borderTypes = std::vector<int>(), const Scalar&  borderValue=0);

    /* @brief Constructor to initialize the rotation transformation with a specific angle (in degrees)
       @param angle the angle used in rotation
       @param interpolation the interpolation type used in rotation
       @param borderType the type of image border when rotated
       @param borderValue the value used in case of constant border
    */
    CV_WRAP Rotate(float angle, int interpolation=INTER_LINEAR, int borderType=BORDER_CONSTANT, int borderValue=0);

    /* @brief Apply the rotation to a single image
        @param src Input image to be rotated
        @param dst Output (rotated) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst) override;

    /* @brief Apply the rotation for a single point
        @param src Input point to be rotated
    */
    Point2f virtual point(const Point2f& src) override;

    /* @brief choose an angle from the specified range to apply in next transformations
    */
    void virtual init(const Mat& srcImage) override;

private :
    float minAngle, maxAngle, angle;
    Mat rotationMat;
    std::vector<int> interpolations, borderTypes;
    int interpolation, borderType;
    Scalar borderValue;
    bool random;

};

}} //namespace cv::augment

#endif
