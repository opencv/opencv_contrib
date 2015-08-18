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

#ifndef __OPENCV_SPARSEMATCHDENSIFIER_HPP__
#define __OPENCV_SPARSEMATCHDENSIFIER_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>
#include <vector>

using namespace std;

namespace cv {
namespace ximgproc {

struct CV_EXPORTS_W SparseMatch
{
    Point2f reference_image_pos;
    Point2f target_image_pos;

    CV_WRAP SparseMatch(Point2f ref_point, Point2f target_point);
};

class CV_EXPORTS_W SparseMatchInterpolator : public Algorithm
{
public:
    CV_WRAP virtual void interpolate(InputArray reference_image, InputArray target_image, InputArray matches, OutputArray dense_flow) = 0;
};

class CV_EXPORTS_W EdgeAwareInterpolator : public SparseMatchInterpolator
{
public:
    CV_WRAP virtual void setInlierEps(float eps) = 0;
};

CV_EXPORTS_W
Ptr<EdgeAwareInterpolator> createEdgeAwareInterpolator();
}
}
#endif
#endif
