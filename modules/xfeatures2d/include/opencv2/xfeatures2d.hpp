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
class CV_EXPORTS_W FREAK : public Feature2D
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
    CV_WRAP static Ptr<FREAK> create(bool orientationNormalized = true,
                             bool scaleNormalized = true,
                             float patternScale = 22.0f,
                             int nOctaves = 4,
                             const std::vector<int>& selectedPairs = std::vector<int>());
};


/** @brief The class implements the keypoint detector introduced by @cite Agrawal08, synonym of StarDetector. :
 */
class CV_EXPORTS_W StarDetector : public Feature2D
{
public:
    //! the full constructor
    CV_WRAP static Ptr<StarDetector> create(int maxSize=45, int responseThreshold=30,
                         int lineThresholdProjected=10,
                         int lineThresholdBinarized=8,
                         int suppressNonmaxSize=5);
};

/*
 * BRIEF Descriptor
 */

/** @brief Class for computing BRIEF descriptors described in @cite calon2010 .

@param bytes legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
@param use_orientation sample patterns using keypoints orientation, disabled by default.

 */
class CV_EXPORTS_W BriefDescriptorExtractor : public Feature2D
{
public:
    CV_WRAP static Ptr<BriefDescriptorExtractor> create( int bytes = 32, bool use_orientation = false );
};

/** @brief Class implementing the locally uniform comparison image descriptor, described in @cite LUCID

An image descriptor that can be computed very fast, while being
about as robust as, for example, SURF or BRIEF.
 */
class CV_EXPORTS_W LUCID : public Feature2D
{
public:
    /**
     * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
     * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
     */
    CV_WRAP static Ptr<LUCID> create(const int lucid_kernel, const int blur_kernel);
};


/*
* LATCH Descriptor
*/

/** latch Class for computing the LATCH descriptor.
If you find this code useful, please add a reference to the following paper in your work:
Gil Levi and Tal Hassner, "LATCH: Learned Arrangements of Three Patch Codes", arXiv preprint arXiv:1501.03719, 15 Jan. 2015

LATCH is a binary descriptor based on learned comparisons of triplets of image patches.

* bytes is the size of the descriptor - can be 64, 32, 16, 8, 4, 2 or 1
* rotationInvariance - whether or not the descriptor should compansate for orientation changes.
* half_ssd_size - the size of half of the mini-patches size. For example, if we would like to compare triplets of patches of size 7x7x
    then the half_ssd_size should be (7-1)/2 = 3.

Note: the descriptor can be coupled with any keypoint extractor. The only demand is that if you use set rotationInvariance = True then
    you will have to use an extractor which estimates the patch orientation (in degrees). Examples for such extractors are ORB and SIFT.

Note: a complete example can be found under /samples/cpp/tutorial_code/xfeatures2D/latch_match.cpp

*/
class CV_EXPORTS_W LATCH : public Feature2D
{
public:
    CV_WRAP static Ptr<LATCH> create(int bytes = 32, bool rotationInvariance = true, int half_ssd_size=3);
};

/** @brief Class implementing DAISY descriptor, described in @cite Tola10

@param radius radius of the descriptor at the initial scale
@param q_radius amount of radial range division quantity
@param q_theta amount of angular range division quantity
@param q_hist amount of gradient orientations range division quantity
@param norm choose descriptors normalization type, where
DAISY::NRM_NONE will not do any normalization (default),
DAISY::NRM_PARTIAL mean that histograms are normalized independently for L2 norm equal to 1.0,
DAISY::NRM_FULL mean that descriptors are normalized for L2 norm equal to 1.0,
DAISY::NRM_SIFT mean that descriptors are normalized for L2 norm equal to 1.0 but no individual one is bigger than 0.154 as in SIFT
@param H optional 3x3 homography matrix used to warp the grid of daisy but sampling keypoints remains unwarped on image
@param interpolation switch to disable interpolation for speed improvement at minor quality loss
@param use_orientation sample patterns using keypoints orientation, disabled by default.

 */
class CV_EXPORTS_W DAISY : public Feature2D
{
public:
    enum
    {
        NRM_NONE = 100, NRM_PARTIAL = 101, NRM_FULL = 102, NRM_SIFT = 103,
    };
    CV_WRAP static Ptr<DAISY> create( float radius = 15, int q_radius = 3, int q_theta = 8,
                int q_hist = 8, int norm = DAISY::NRM_NONE, InputArray H = noArray(),
                bool interpolation = true, bool use_orientation = false );

    /** @overload
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) = 0;

    virtual void compute( InputArrayOfArrays images,
                          std::vector<std::vector<KeyPoint> >& keypoints,
                          OutputArrayOfArrays descriptors );

    /** @overload
     * @param image image to extract descriptors
     * @param roi region of interest within image
     * @param descriptors resulted descriptors array for roi image pixels
     */
    virtual void compute( InputArray image, Rect roi, OutputArray descriptors ) = 0;

    /**@overload
     * @param image image to extract descriptors
     * @param descriptors resulted descriptors array for all image pixels
     */
    virtual void compute( InputArray image, OutputArray descriptors ) = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void GetDescriptor( double y, double x, int orientation, float* descriptor ) const = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     * @param H homography matrix for warped grid
     */
    virtual bool GetDescriptor( double y, double x, int orientation, float* descriptor, double* H ) const = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void GetUnnormalizedDescriptor( double y, double x, int orientation, float* descriptor ) const = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     * @param H homography matrix for warped grid
     */
    virtual bool GetUnnormalizedDescriptor( double y, double x, int orientation, float* descriptor , double *H ) const = 0;

};

/** @brief Class implementing the MSD (*Maximal Self-Dissimilarity*) keypoint detector, described in @cite Tombari14.

The algorithm implements a novel interest point detector stemming from the intuition that image patches
which are highly dissimilar over a relatively large extent of their surroundings hold the property of
being repeatable and distinctive. This concept of "contextual self-dissimilarity" reverses the key
paradigm of recent successful techniques such as the Local Self-Similarity descriptor and the Non-Local
Means filter, which build upon the presence of similar - rather than dissimilar - patches. Moreover,
it extends to contextual information the local self-dissimilarity notion embedded in established
detectors of corner-like interest points, thereby achieving enhanced repeatability, distinctiveness and
localization accuracy.

*/

class CV_EXPORTS_W MSDDetector : public Feature2D {

public:

    static Ptr<MSDDetector> create(int m_patch_radius = 3, int m_search_area_radius = 5,
            int m_nms_radius = 5, int m_nms_scale_radius = 0, float m_th_saliency = 250.0f, int m_kNN = 4,
            float m_scale_factor = 1.25f, int m_n_scales = -1, bool m_compute_orientation = false);
};

/** @brief Class implementing VGG (Oxford Visual Geometry Group) descriptor trained end to end
using "Descriptor Learning Using Convex Optimisation" (DLCO) aparatus described in @cite Simonyan14.

@param desc type of descriptor to use, VGG::VGG_120 is default (120 dimensions float)
Available types are VGG::VGG_120, VGG::VGG_80, VGG::VGG_64, VGG::VGG_48
@param isigma gaussian kernel value for image blur (default is 1.4f)
@param img_normalize use image sample intensity normalization (enabled by default)
@param use_orientation sample patterns using keypoints orientation, enabled by default
@param scale_factor adjust the sampling window of detected keypoints to 64.0f (VGG sampling window)
6.25f is default and fits for KAZE, SURF detected keypoints window ratio
6.75f should be the scale for SIFT detected keypoints window ratio
5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
0.75f should be the scale for ORB keypoints ratio

@param dsc_normalize clamp descriptors to 255 and convert to uchar CV_8UC1 (disabled by default)

 */
class CV_EXPORTS_W VGG : public Feature2D
{
public:

    CV_WRAP enum
    {
        VGG_120 = 100, VGG_80 = 101, VGG_64 = 102, VGG_48 = 103,
    };

    CV_WRAP static Ptr<VGG> create( int desc = VGG::VGG_120, float isigma = 1.4f,
                                    bool img_normalize = true, bool use_scale_orientation = true,
                                    float scale_factor = 6.25f, bool dsc_normalize = false );
    /**
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    CV_WRAP virtual void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) = 0;

};

//! @}

}
}

#endif
