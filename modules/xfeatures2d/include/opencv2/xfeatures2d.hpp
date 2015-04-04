/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_XFEATURES2D_HPP__
#define __OPENCV_XFEATURES2D_HPP__

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

/** @defgroup xfeatures2d Extra 2D Features Framework
@{
    @defgroup xfeatures2d_experiment Experimental 2D Features Algorithms

This section describes experimental algorithms for 2d feature detection.

    @defgroup xfeatures2d_nonfree Non-free 2D Features Algorithms

This section describes two popular algorithms for 2d feature detection, SIFT and SURF, that are
known to be patented. Use them at your own risk.

@}
*/

namespace cv
{
namespace xfeatures2d
{

//! @addtogroup xfeatures2d_experiment
//! @{

/** @brief Class implementing the FREAK (*Fast Retina Keypoint*) keypoint descriptor, described in @cite AOV12 .

The algorithm propose a novel keypoint descriptor inspired by the human visual system and more
precisely the retina, coined Fast Retina Key- point (FREAK). A cascade of binary strings is
computed by efficiently comparing image intensities over a retinal sampling pattern. FREAKs are in
general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK.
They are competitive alternatives to existing keypoints in particular for embedded applications.

@note
   -   An example on how to use the FREAK descriptor can be found at
        opencv_source_code/samples/cpp/freak_demo.cpp
 */
class CV_EXPORTS FREAK : public Feature2D
{
public:

    enum
    {
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
    };

    /**
    @param orientationNormalized Enable orientation normalization.
    @param scaleNormalized Enable scale normalization.
    @param patternScale Scaling of the description pattern.
    @param nOctaves Number of octaves covered by the detected keypoints.
    @param selectedPairs (Optional) user defined selected pairs indexes,
     */
    static Ptr<FREAK> create(bool orientationNormalized = true,
                             bool scaleNormalized = true,
                             float patternScale = 22.0f,
                             int nOctaves = 4,
                             const std::vector<int>& selectedPairs = std::vector<int>());
};


/** @brief The class implements the keypoint detector introduced by @cite Agrawal08, synonym of StarDetector. :
 */
class CV_EXPORTS StarDetector : public FeatureDetector
{
public:
    //! the full constructor
    static Ptr<StarDetector> create(int maxSize=45, int responseThreshold=30,
                         int lineThresholdProjected=10,
                         int lineThresholdBinarized=8,
                         int suppressNonmaxSize=5);
};

/*
 * BRIEF Descriptor
 */

/** @brief Class for computing BRIEF descriptors described in @cite calon2010 .

@note
   -   A complete BRIEF extractor sample can be found at
        opencv_source_code/samples/cpp/brief_match_test.cpp

 */
class CV_EXPORTS BriefDescriptorExtractor : public DescriptorExtractor
{
public:
    static Ptr<BriefDescriptorExtractor> create( int bytes = 32 );
};

/** @overload */
CV_EXPORTS void AGAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true );

/** @brief Detects corners using the AGAST algorithm

@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the four neighborhoods as defined in the paper:
AgastFeatureDetector::AGAST_5_8, AgastFeatureDetector::AGAST_7_12d,
AgastFeatureDetector::AGAST_7_12s, AgastFeatureDetector::OAST_9_16

Detects corners using the AGAST algorithm by @cite mair2010_agast .

 */
CV_EXPORTS void AGAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression, int type );

/** @brief Wrapping class for feature detection using the AGAST method. :
 */
class CV_EXPORTS_W AgastFeatureDetector : public Feature2D
{
public:
    enum
    {
        AGAST_5_8 = 0, AGAST_7_12d = 1, AGAST_7_12s = 2, OAST_9_16 = 3,
        THRESHOLD = 10000, NONMAX_SUPPRESSION = 10001,
    };

    CV_WRAP static Ptr<AgastFeatureDetector> create( int threshold=10,
                                                     bool nonmaxSuppression=true,
                                                     int type=AgastFeatureDetector::OAST_9_16 );

    CV_WRAP virtual void setThreshold(int threshold) = 0;
    CV_WRAP virtual int getThreshold() const = 0;

    CV_WRAP virtual void setNonmaxSuppression(bool f) = 0;
    CV_WRAP virtual bool getNonmaxSuppression() const = 0;

    CV_WRAP virtual void setType(int type) = 0;
    CV_WRAP virtual int getType() const = 0;
};

/** @brief Class implementing the locally uniform comparison image descriptor, described in @cite LUCID

An image descriptor that can be computed very fast, while being
about as robust as, for example, SURF or BRIEF.
 */
class CV_EXPORTS LUCID : public DescriptorExtractor
{
public:
    /**
     * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
     * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
     */
    static Ptr<LUCID> create(const int lucid_kernel, const int blur_kernel);
};



//! @}

}
}

#endif
