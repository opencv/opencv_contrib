/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_MSER_HPP
#define OPENCV_FASTCV_MSER_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief MSER blob detector for grayscale images
 *
 */
class CV_EXPORTS_W FCVMSER
{
public:

    /**
     * @brief Structure containing additional information about found contour
     *
     */
    struct ContourData
    {
        uint32_t variation;   //!< Variation of a contour from previous grey level
        int32_t  polarity;    //!< Polarity for a contour. This value is 1 if this is a MSER+ region, -1 if this is a MSER- region.
        uint32_t nodeId;      //!< Node ID for a contour
        uint32_t nodeCounter; //!< Node counter for a contour
    };

    /**
     * @brief Creates MSER detector
     *
     * @param imgSize Image size. Image width has to be greater than 50, and image height has to be greater than 5.
     * @param numNeighbors Number of neighbors in contours, can be 4 or 8
     * @param delta Delta to be used in MSER algorithm (the difference in grayscale values
                    within which the region is stable ).
                    Typical value range [0.8 8], typical value 2
     * @param minArea Minimum area (number of pixels) of a mser contour.
                      Typical value range [10 50], typical value 30
     * @param maxArea Maximum area (number of pixels) of a  mser contour.
                      Typical value 14400 or 0.25*width*height
     * @param maxVariation Maximum variation in grayscale between 2 levels allowed.
                           Typical value range [0.1 1.0], typical value 0.15
     * @param minDiversity Minimum diversity in grayscale between 2 levels allowed.
                           Typical value range [0.1 1.0], typical value 0.2
     * @return Feature detector object ready for detection
     */
    CV_WRAP static Ptr<FCVMSER> create( const cv::Size& imgSize,
                                        int numNeighbors = 4,
                                        int delta = 2,
                                        int minArea = 30,
                                        int maxArea = 14400,
                                        float maxVariation = 0.15f,
                                        float minDiversity = 0.2f);

    /**
     * @brief This is an overload for detect() function
     *
     * @param src Source image of type CV_8UC1. Image width has to be greater than 50, and image height has to be greater than 5.
                 Pixels at the image boundary are not processed. If boundary pixels are important
                for a particular application, please consider padding the input image with dummy
                pixels of one pixel wide.
    * @param contours Array containing found contours
    */
    CV_WRAP virtual void detect(InputArray src, std::vector<std::vector<Point>>& contours) = 0;

    /**
     * @brief This is an overload for detect() function
     *
     * @param src Source image of type CV_8UC1. Image width has to be greater than 50, and image height has to be greater than 5.
                 Pixels at the image boundary are not processed. If boundary pixels are important
                for a particular application, please consider padding the input image with dummy
                pixels of one pixel wide.
    * @param contours Array containing found contours
    * @param boundingBoxes Array containing bounding boxes of found contours
    */
    CV_WRAP virtual void detect(InputArray src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes) = 0;

    /**
     * @brief Runs MSER blob detector on the grayscale image
     *
     * @param src Source image of type CV_8UC1. Image width has to be greater than 50, and image height has to be greater than 5.
                 Pixels at the image boundary are not processed. If boundary pixels are important
                for a particular application, please consider padding the input image with dummy
                pixels of one pixel wide.
    * @param contours Array containing found contours
    * @param boundingBoxes Array containing bounding boxes of found contours
    * @param contourData Array containing additional information about found contours
    */
    virtual void detect(InputArray src, std::vector<std::vector<Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
                        std::vector<ContourData>& contourData) = 0;

    CV_WRAP virtual cv::Size getImgSize() = 0;
    CV_WRAP virtual int getNumNeighbors() = 0;
    CV_WRAP virtual int getDelta()        = 0;
    CV_WRAP virtual int getMinArea()      = 0;
    CV_WRAP virtual int getMaxArea()      = 0;
    CV_WRAP virtual float getMaxVariation() = 0;
    CV_WRAP virtual float getMinDiversity() = 0;

    virtual ~FCVMSER() {}
};

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_MSER_HPP
