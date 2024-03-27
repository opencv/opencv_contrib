/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this
 license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without
 modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright
 notice,
 //     this list of conditions and the following disclaimer in the
 documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote
 products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is"
 and
 // any express or implied warranties, including, but not limited to, the
 implied
 // warranties of merchantability and fitness for a particular purpose are
 disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any
 direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __OPENCV_SINUSOIDAL_COMPLEMENTARY_GRAYCODE_HPP__
#define __OPENCV_SINUSOIDAL_COMPLEMENTARY_GRAYCODE_HPP__

#include "opencv2/core.hpp"
#include "opencv2/structured_light/structured_light.hpp"

namespace cv {
namespace structured_light {
//! @addtogroup structured_light
//! @{

/** @brief Class implementing the Sinusoidal Complementary Gray-code pattern,
 based on @cite Zhang Q.
 *
 *  The resulting pattern consists of a sinusoidal pattern and a Gray code
 pattern corresponding to the period width.
 *
 *  The entire projection sequence contains the sine fringe sequence and the
 Gray code fringe sequence.
 *  For an image with a format (WIDTH, HEIGHT), a vertical sinusoidal fringe
 with a period of N has a period width of
 *  w = WIDTH / N.The algorithm uses log 2 (N) vertical Gray code patterns and
 an additional sequence Gray code pattern to
 *  avoid edge dephasing errors caused by the sampling theorem.The same goes for
 horizontal stripes.

 *
 *  For an image in 1280*720 format, a phase-shifted fringe pattern needs to be
 generated for 32 periods, with a period width of 1280 / 32 = 40 and
 *  the required number of Gray code patterns being log 2 (32) = 5 + 1 = 6.the
 algorithm can be applied to sinusoidal complementary Gray codes of any number
 of steps and any bits,
 *  as long as their period widths satisfy the above principle.
 */
class CV_EXPORTS_W SinusCompleGrayCodePattern : public StructuredLightPattern {
  public:
    /** @brief Parameters of StructuredLightPattern constructor.
     *  @param width Projector's width. Default value is 1280.
     *  @param height Projector's height. Default value is 720.
     */
    struct CV_EXPORTS Params {
        Params();
        int width;
        int height;
        int nbrOfPeriods;
        int shiftTime;
        int minDisparity;
        int maxDisparity;
        bool horizontal;
        float confidenceThreshold;
        float maxCost;
    };

    /** @brief Constructor
     @param parameters SinusCompleGrayCodePattern parameters
     SinusCompleGrayCodePattern::Params: the width and the height of the
     projector.
     */
    static Ptr<SinusCompleGrayCodePattern>
    create(const SinusCompleGrayCodePattern::Params &parameters =
               SinusCompleGrayCodePattern::Params());
    /**
     * @brief Compute a confidence map from sinusoidal patterns.
     * @param patternImages Input data to compute the confidence map.
     * @param confidenceMap confidence map obtained through PSP.
     */
    CV_WRAP
    virtual void computeConfidenceMap(InputArrayOfArrays patternImages,
                                      OutputArray confidenceMap) const = 0;
    /**
     * @brief Compute a wrapped phase map from sinusoidal patterns.
     * @param patternImages Input data to compute the wrapped phase map.
     * @param wrappedPhaseMap Wrapped phase map obtained through PSP.
     */
    CV_WRAP
    virtual void computePhaseMap(InputArrayOfArrays patternImages,
                                 OutputArray wrappedPhaseMap) const = 0;
    /**
     * @brief Compute a floor map from complementary graycode patterns and
     * wrappedPhaseMap.
     * @param patternImages Input data to compute the floor map.
     * @param confidenceMap Input data to threshold gray code img, we use
     * confidence map because that we set confidence map is same as texture map,
     * A = B.
     * @param wrappedPhaseMap Input data to help us select K1 or K2.
     * @param floorMap Floor map obtained through complementary graycode and
     * wrappedPhaseMap.
     */
    CV_WRAP
    virtual void computeFloorMap(InputArrayOfArrays patternImages,
                                 InputArray confidenceMap,
                                 InputArray wrappedPhaseMap,
                                 OutputArray floorMap) const = 0;
    /**
     * @brief Unwrap the wrapped phase map to remove phase ambiguities.
     * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
     * @param unwrappedPhaseMap The unwrapped phase map used to find
     * correspondences between the two devices.
     * @param shadowMask Mask used to discard shadow regions.
     * @param confidenceThreshod confidence threshod to discard invalid data.
     */
    CV_WRAP
    virtual void unwrapPhaseMap(InputArray wrappedPhaseMap, InputArray floorMap,
                                OutputArray unwrappedPhaseMap,
                                InputArray shadowMask = noArray()) const = 0;
    /**
     * @brief compute disparity from left unwrap map and right unwrap map.
     * @param lhsUnwrapMap left unwrap map.
     * @param rhsUnwrapMap right unwrap map.
     * @param disparityMap dispairty map that computed.
     */
    CV_WRAP
    virtual void computeDisparity(InputArray lhsUnwrapMap,
                                  InputArray rhsUnwrapMap,
                                  OutputArray disparityMap) const = 0;
};

//! @}

} // namespace structured_light
} // namespace cv
#endif
