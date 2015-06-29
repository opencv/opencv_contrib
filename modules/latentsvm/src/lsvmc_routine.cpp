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
#include "_lsvmc_routine.h"
namespace cv
{
namespace lsvm
{
int allocFilterObject(CvLSVMFilterObjectCascade **obj, const int sizeX,
                      const int sizeY, const int numFeatures) 
{
    int i;
    (*obj) = (CvLSVMFilterObjectCascade *)malloc(sizeof(CvLSVMFilterObjectCascade));
    (*obj)->sizeX           = sizeX;
    (*obj)->sizeY           = sizeY;
    (*obj)->numFeatures     = numFeatures;
    (*obj)->fineFunction[0] = 0.0f;
    (*obj)->fineFunction[1] = 0.0f;
    (*obj)->fineFunction[2] = 0.0f;
    (*obj)->fineFunction[3] = 0.0f;
    (*obj)->V.x         = 0;
    (*obj)->V.y         = 0;
    (*obj)->V.l         = 0;
    (*obj)->H = (float *) malloc(sizeof (float) * 
                                (sizeX * sizeY  * numFeatures));
    for(i = 0; i < sizeX * sizeY * numFeatures; i++)
    {
        (*obj)->H[i] = 0.0f;
    }
    return LATENT_SVM_OK;
}
int freeFilterObject (CvLSVMFilterObjectCascade **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->H);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeatureMapObject(CvLSVMFeatureMapCascade **obj, const int sizeX, 
                          const int sizeY, const int numFeatures)
{
    int i;
    (*obj) = (CvLSVMFeatureMapCascade *)malloc(sizeof(CvLSVMFeatureMapCascade));
    (*obj)->sizeX       = sizeX;
    (*obj)->sizeY       = sizeY;
    (*obj)->numFeatures = numFeatures;
    (*obj)->map = (float *) malloc(sizeof (float) * 
                                  (sizeX * sizeY  * numFeatures));
    for(i = 0; i < sizeX * sizeY * numFeatures; i++)
    {
        (*obj)->map[i] = 0.0f;
    }
    return LATENT_SVM_OK;
}
int freeFeatureMapObject (CvLSVMFeatureMapCascade **obj)
{
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    free((*obj)->map);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}

int allocFeaturePyramidObject(CvLSVMFeaturePyramidCascade **obj,
                              const int numLevels) 
{
    (*obj) = (CvLSVMFeaturePyramidCascade *)malloc(sizeof(CvLSVMFeaturePyramidCascade));
    (*obj)->numLevels = numLevels;
    (*obj)->pyramid    = (CvLSVMFeatureMapCascade **)malloc(
                         sizeof(CvLSVMFeatureMapCascade *) * numLevels);
    return LATENT_SVM_OK;
}

int freeFeaturePyramidObject (CvLSVMFeaturePyramidCascade **obj)
{
    int i; 
    if(*obj == NULL) return LATENT_SVM_MEM_NULL;
    for(i = 0; i < (*obj)->numLevels; i++)
    {
        freeFeatureMapObject(&((*obj)->pyramid[i]));
    }
    free((*obj)->pyramid);
    free(*obj);
    (*obj) = NULL;
    return LATENT_SVM_OK;
}
}
}
