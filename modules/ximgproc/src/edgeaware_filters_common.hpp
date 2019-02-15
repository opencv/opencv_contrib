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
 *  * Redistributions of source code must retain the above copyright notice,
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

#ifndef __EDGEAWAREFILTERS_COMMON_HPP__
#define __EDGEAWAREFILTERS_COMMON_HPP__
#ifdef __cplusplus

namespace cv
{
namespace ximgproc
{

Ptr<DTFilter> createDTFilterRF(InputArray adistHor, InputArray adistVert, double sigmaSpatial, double sigmaColor, int numIters);

int getTotalNumberOfChannels(InputArrayOfArrays src);

void checkSameSizeAndDepth(InputArrayOfArrays src, Size &sz, int &depth);

namespace intrinsics
{
    void add_(float *dst, float *src1, int w);

    void mul(float *dst, float *src1, float *src2, int w);

    void mul(float *dst, float *src1, float src2, int w);

    //dst = alpha*src + beta
    void mad(float *dst, float *src1, float alpha, float beta, int w);

    void add_mul(float *dst, float *src1, float *src2, int w);

    void sub_mul(float *dst, float *src1, float *src2, int w);

    void sub_mad(float *dst, float *src1, float *src2, float c0, int w);

    void det_2x2(float *dst, float *a00, float *a01, float *a10, float *a11, int w);

    void div_det_2x2(float *a00, float *a01, float *a11, int w);

    void div_1x(float *a1, float *b1, int w);

    void inv_self(float *src, int w);


    void sqr_(float *dst, float *src1, int w);

    void sqrt_(float *dst, float *src, int w);

    void sqr_dif(float *dst, float *src1, float *src2, int w);

    void add_sqr_dif(float *dst, float *src1, float *src2, int w);

    void add_sqr(float *dst, float *src1, int w);

    void min_(float *dst, float *src1, float *src2, int w);

    void rf_vert_row_pass(float *curRow, float *prevRow, float alphaVal, int w);
}

}
}

#endif
#endif
