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

#include "precomp.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cv
{
namespace ximgproc
{
    void rollingGuidanceFilter(InputArray src_, OutputArray dst_, int d,
                               double sigmaColor, double sigmaSpace,  int numOfIter, int borderType)
    {
        CV_Assert(!src_.empty());

        Mat guidance = src_.getMat();
        Mat src = src_.getMat();

        CV_Assert(src.size() == guidance.size());
        CV_Assert(src.depth() == guidance.depth() && (src.depth() == CV_8U || src.depth() == CV_32F) );

        if (sigmaColor <= 0)
            sigmaColor = 1;
        if (sigmaSpace <= 0)
            sigmaSpace = 1;

        dst_.create(src.size(), src.type());
        Mat dst = dst_.getMat();

        if (src.data == guidance.data)
            guidance = guidance.clone();
        if (dst.data == src.data)
            src = src.clone();

        int srcCnNum = src.channels();

        if (srcCnNum == 1 || srcCnNum == 3)
        {
            while(numOfIter--){
                jointBilateralFilter(guidance, src, guidance, d, sigmaColor, sigmaSpace, borderType);
            }
            guidance.copyTo(dst_);
        }
        else
        {
            CV_Error(Error::BadNumChannels, "Unsupported number of channels");
        }
    }
}
}
