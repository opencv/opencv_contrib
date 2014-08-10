/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENCV_EDGEFILTER_HPP__
#define __OPENCV_EDGEFILTER_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{

enum EdgeAwareFiltersList
{
    DTF_NC,
    DTF_IC,
    DTF_RF,

    GUIDED_FILTER,
    AM_FILTER
};


/*Interface for DT filters*/
class CV_EXPORTS_W DTFilter : public Algorithm
{
public:

    CV_WRAP virtual void filter(InputArray src, OutputArray dst, int dDepth = -1) = 0;
};

typedef Ptr<DTFilter> DTFilterPtr;

/*Fabric function for DT filters*/
CV_EXPORTS_W
Ptr<DTFilter> createDTFilter(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

/*One-line DT filter call*/
CV_EXPORTS_W
void dtFilter(InputArray guide, InputArray src, OutputArray dst, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/*Interface for Guided Filter*/
class CV_EXPORTS_W GuidedFilter : public Algorithm
{
public:

    CV_WRAP virtual void filter(InputArray src, OutputArray dst, int dDepth = -1) = 0;
};

/*Fabric function for Guided Filter*/
CV_EXPORTS_W Ptr<GuidedFilter> createGuidedFilter(InputArray guide, int radius, double eps);

/*One-line Guided Filter call*/
CV_EXPORTS_W void guidedFilter(InputArray guide, InputArray src, OutputArray dst, int radius, double eps, int dDepth = -1);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

class CV_EXPORTS_W AdaptiveManifoldFilter : public Algorithm
{
public:
    /**
        * @brief Apply High-dimensional filtering using adaptive manifolds
        * @param src       Input image to be filtered.
        * @param dst       Adaptive-manifold filter response.
        * @param src_joint Image for joint filtering (optional).
        */
    CV_WRAP virtual void filter(InputArray src, OutputArray dst, InputArray joint = noArray()) = 0;

    CV_WRAP virtual void collectGarbage() = 0;

    static Ptr<AdaptiveManifoldFilter> create();
};

//Fabric function for AM filter algorithm
CV_EXPORTS_W Ptr<AdaptiveManifoldFilter> createAMFilter(double sigma_s, double sigma_r, bool adjust_outliers = false);

//One-line Adaptive Manifold filter call
CV_EXPORTS_W void amFilter(InputArray joint, InputArray src, OutputArray dst, double sigma_s, double sigma_r, bool adjust_outliers = false);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

CV_EXPORTS_W
void jointBilateralFilter(InputArray joint, InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT);

}
#endif
#endif
