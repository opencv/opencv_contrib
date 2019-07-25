// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_UNIFORMNOISE_HPP
#define OPENCV_AUGMENT_UNIFORMNOISE_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W UniformNoise : public Transform
{
public:
    /* @brief Constructor to initialize the noise noise transformation with a specific mean and stddev
       @param low the lowest value of the noise added
       @param high the highest value of the noise added
    */
    CV_WRAP UniformNoise(float low, float high);

    /* @brief Apply the uniform noise to a single image
       @param src Input image that the noise will be added to
       @param dst Output (noisy) image
    */
    CV_WRAP virtual void image(InputArray src, OutputArray dst) override;

    /* @brief Apply the transformation for a rectangle
       @param box Rect2f consisting of (x1, y1, w, h) corresponding to (top left point, size)
    */
    virtual Rect2f rectangle(const Rect2f& src) override;

    /* @brief choose an random kernel size and sigma from the specified ranges to apply in next transformations
    */
    void virtual init(const Mat& srcImage) override;

private :
  float low, high;
  Mat noise;

};

}} //namespace cv::augment

#endif
