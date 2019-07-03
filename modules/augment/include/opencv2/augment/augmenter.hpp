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
    CV_WRAP void applyImages(InputArrayOfArrays imgs, OutputArrayOfArrays dstImgs);

     /* @brief apply the transformations to vector of images and corresponding masks
        @param imgs the images to be augmented
        @param dstImgs the augmented images
        @param masks the masks to be augmented
        @param dstMasks the augmented masks
     */
    CV_WRAP void applyImagesWithMasks(InputArrayOfArrays imgs, InputArrayOfArrays masks, OutputArrayOfArrays dstImgs, OutputArrayOfArrays dstMasks);

    /* @brief apply the transformations to vector of images and corresponding points
        @param imgs the images to be augmented
        @param dstImgs the augmented images
        @param points the points to be augmented
        @param dstPoints the augmented points
     */
    CV_WRAP void applyImagesWithPoints(InputArrayOfArrays imgs, InputArrayOfArrays points, OutputArrayOfArrays dstImgs, OutputArrayOfArrays dstPoints);

    /* @brief apply the transformations to vector of images and corresponding points
       @param imgs the images to be augmented
       @param dstImgs the augmented images
       @param rects the rectangles to be augmented
       @param dstRects the augmented rectangles
    */
    CV_WRAP void applyImagesWithRectangles(InputArrayOfArrays imgs, InputArrayOfArrays rects, OutputArrayOfArrays dstImgs, OutputArrayOfArrays dstRects);


private:
    std::vector<Ptr<Transform>> transformations;
    std::vector<float> probs;
};

}} //namespace cv::augment

#endif
