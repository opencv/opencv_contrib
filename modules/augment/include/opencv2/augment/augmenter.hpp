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
       @param t the transformation to be added to the augmenter
       @param prob the probability of applying the transformation t
    */
    CV_WRAP void add(Ptr<Transform> t, float prob=1.f);

    /* @brief apply the transformations to vector of images
       @param imgs the images that the transformations will be applied to
    */
    CV_WRAP std::vector<Mat> applyImages(const std::vector<Mat>& imgs);

    ///* @brief apply the transformations to both images and masks
    //   @param imgs the images that the transformations will be applied to
    //   @param masks the masks of the images to apply the same transformations to
    //*/
    //CV_WRAP std::tuple<std::vector<Mat>, std::vector<Mat>> applyImagesWithMasks(const std::vector<Mat>& imgs, const std::vector<Mat>& masks);
private:
    std::vector<Ptr<Transform>> transformations;
    std::vector<float> probs;
};

}} //namespace cv::augment

#endif
