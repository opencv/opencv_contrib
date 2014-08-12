/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *  
 *  
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *  
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *  
 *  *Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *  
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *  
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_EDGEFILTER_HPP__
#define __OPENCV_EDGEFILTER_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
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
class CV_EXPORTS DTFilter : public Algorithm
{
public:

    virtual void filter(InputArray src, OutputArray dst, int dDepth = -1) = 0;
};

typedef Ptr<DTFilter> DTFilterPtr;

/*Fabric function for DT filters*/
CV_EXPORTS
Ptr<DTFilter> createDTFilter(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

/*One-line DT filter call*/
CV_EXPORTS
void dtFilter(InputArray guide, InputArray src, OutputArray dst, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/*Interface for Guided Filter*/
class CV_EXPORTS GuidedFilter : public Algorithm
{
public:

    virtual void filter(InputArray src, OutputArray dst, int dDepth = -1) = 0;
};

/*Fabric function for Guided Filter*/
CV_EXPORTS Ptr<GuidedFilter> createGuidedFilter(InputArray guide, int radius, double eps);

/*One-line Guided Filter call*/
CV_EXPORTS void guidedFilter(InputArray guide, InputArray src, OutputArray dst, int radius, double eps, int dDepth = -1);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

class CV_EXPORTS AdaptiveManifoldFilter : public Algorithm
{
public:
    /**
        * @brief Apply High-dimensional filtering using adaptive manifolds
        * @param src       Input image to be filtered.
        * @param dst       Adaptive-manifold filter response.
        * @param src_joint Image for joint filtering (optional).
        */
    virtual void filter(InputArray src, OutputArray dst, InputArray joint = noArray()) = 0;

    virtual void collectGarbage() = 0;

    static Ptr<AdaptiveManifoldFilter> create();
};

//Fabric function for AM filter algorithm
CV_EXPORTS Ptr<AdaptiveManifoldFilter> createAMFilter(double sigma_s, double sigma_r, bool adjust_outliers = false);

//One-line Adaptive Manifold filter call
CV_EXPORTS void amFilter(InputArray joint, InputArray src, OutputArray dst, double sigma_s, double sigma_r, bool adjust_outliers = false);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

CV_EXPORTS
void jointBilateralFilter(InputArray joint, InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT);

}
}
#endif
#endif
