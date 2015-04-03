/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, University of Nizhny Novgorod, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "_lsvmc_function.h"
namespace cv
{
namespace lsvm
{

float calcM    (int k,int di,int dj, const CvLSVMFeaturePyramidCascade * H, const CvLSVMFilterObjectCascade *filter){
    int i, j;
    float m = 0.0f;
    for(j = dj; j < dj + filter->sizeY; j++){
        for(i = di * H->pyramid[k]->numFeatures; i < (di + filter->sizeX) * H->pyramid[k]->numFeatures; i++){
             m += H->pyramid[k]->map[(j * H->pyramid[k]->sizeX     ) * H->pyramid[k]->numFeatures + i] * 
                  filter ->H        [((j - dj) * filter->sizeX - di) * H->pyramid[k]->numFeatures + i];            
        }
    }
    return m;
}
float calcM_PCA(int k,int di,int dj, const CvLSVMFeaturePyramidCascade * H, const CvLSVMFilterObjectCascade *filter){
    int i, j;
    float m = 0.0f;
    for(j = dj; j < dj + filter->sizeY; j++){
        for(i = di * H->pyramid[k]->numFeatures; i < (di + filter->sizeX) * H->pyramid[k]->numFeatures; i++){
            m += H->pyramid[k]->map[(j * H->pyramid[k]->sizeX     ) * H->pyramid[k]->numFeatures + i] * 
                 filter ->H_PCA    [((j - dj) * filter->sizeX - di) * H->pyramid[k]->numFeatures + i];
        }
    }

    return m;
}
float calcM_PCA_cash(int k,int di,int dj, const CvLSVMFeaturePyramidCascade * H, const CvLSVMFilterObjectCascade *filter, float * cashM, int * maskM, int step){
    int i, j, n;
    float m = 0.0f;
    float tmp1, tmp2, tmp3, tmp4;
    float res;
    int pos;
    float *a, *b;

    pos = dj * step + di;

    if(!((maskM[pos / (sizeof(int) * 8)]) & (1 << pos % (sizeof(int) * 8))))
    {
        for(j = dj; j < dj + filter->sizeY; j++)
        {
            a = H->pyramid[k]->map + (j * H->pyramid[k]->sizeX) * H->pyramid[k]->numFeatures
              + di * H->pyramid[k]->numFeatures;
            b = filter ->H_PCA + (j - dj) * filter->sizeX * H->pyramid[k]->numFeatures;
            n = ((di + filter->sizeX) * H->pyramid[k]->numFeatures) - 
              (di * H->pyramid[k]->numFeatures);
            
            res = 0.0f;
            tmp1 = 0.0f; tmp2 = 0.0f; tmp3 = 0.0f; tmp4 = 0.0f;

            for (i = 0; i < (n >> 2); ++i)
            {
                tmp1 += a[4 * i + 0] * b[4 * i + 0];
                tmp2 += a[4 * i + 1] * b[4 * i + 1];
                tmp3 += a[4 * i + 2] * b[4 * i + 2];
                tmp4 += a[4 * i + 3] * b[4 * i + 3];
            }
            
            for (i = (n >> 2) << 2; i < n; ++i) //?
            {
                res += a[i] * b[i];
            }

            res += tmp1 + tmp2 + tmp3 + tmp4;

            m += res;
        }

        cashM[pos                    ]  = m;
        maskM[pos / (sizeof(int) * 8)] |= 1 << pos % (sizeof(int) * 8);
    }
    else
    {
        m = cashM[pos];
    }
    return m;
}
float calcFine (const CvLSVMFilterObjectCascade *filter, int di, int dj){
    return filter->fineFunction[0] * di      + filter->fineFunction[1] * dj + 
           filter->fineFunction[2] * di * di + filter->fineFunction[3] * dj * dj;
}
}
}
