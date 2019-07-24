// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_GAUSSIANNOISE_HPP
#define OPENCV_AUGMENT_GAUSSIANNOISE_HPP
#include <opencv2/augment/transform.hpp>

namespace cv { namespace augment {

class CV_EXPORTS_W GaussianNoise : public Transform
{
public:
    /* @brief Constructor to initialize the gaussian noise transformation with a specific mean and stddev
       @param mean the mean of the gaussian noise
       @param stddev the standard deviation of the gaussian noise
    */
    CV_WRAP GaussianNoise(float mean, float stddev);

    /* @brief Apply the gaussian noise to a single image
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
    void virtual init(const Mat& srcImage) override;

private :
  float mean, stddev;
  Mat noise;

};

}} //namespace cv::augment

#endif
