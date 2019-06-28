// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_AUGMENT_AUGMENTER_HPP
#define OPENCV_AUGMENT_AUGMENTER_HPP
#include <opencv2/augment/transform.hpp>
#include <tuple>
namespace cv { namespace augment {

class CV_EXPORTS_W Augmenter
{
public:
    /* @brief Constructor
    */
    CV_WRAP Augmenter();

    /* @brief add a transformation to the augmenter
    */
    CV_WRAP void add(Ptr<Transform> t, float prob=1.f);

private:
    std::vector< std::tuple<Ptr<Transform> , float> > transformations;
};

}} //namespace cv::augment

#endif
