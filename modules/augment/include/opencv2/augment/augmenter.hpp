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
       @param imgs the images that to be augmented
       @param dstImgs the augmented images 
    */
    CV_WRAP void applyImages(const std::vector<Mat>& imgs, OutputArrayOfArrays dstImgs);

     /* @brief apply the transformations to vector of images and corresponding masks
        @param imgs the images to be augmented
        @param dstImgs the augmented images
        @param masks the masks to be augmented
        @param dstMasks the augmented masks
     */
    CV_WRAP void applyImagesWithMasks(const std::vector<Mat>& imgs, const std::vector<Mat>& masks, OutputArrayOfArrays dstImgs, OutputArrayOfArrays dstMasks);


private:
    std::vector<Ptr<Transform>> transformations;
    std::vector<float> probs;
};

}} //namespace cv::augment

#endif
