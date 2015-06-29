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

#ifndef LSVM_PARSER
#define LSVM_PARSER

#include "_lsvmc_types.h"

#define MODEL    1
#define P        2
#define COMP     3
#define SCORE    4
#define RFILTER  100
#define PFILTERs 101
#define PFILTER  200
#define SIZEX    150
#define SIZEY    151
#define WEIGHTS  152
#define TAGV     300
#define Vx       350
#define Vy       351
#define TAGD     400
#define Dx       451
#define Dy       452
#define Dxx      453
#define Dyy      454
#define BTAG     500

#define PCA          5
#define WEIGHTSPCA   162
#define CASCADE_Th   163
#define HYPOTHES_PCA 164
#define DEFORM_PCA   165
#define HYPOTHES     166
#define DEFORM       167

#define PCACOEFF     6

#define STEP_END 1000

#define EMODEL    (STEP_END + MODEL)
#define EP        (STEP_END + P)
#define ECOMP     (STEP_END + COMP)
#define ESCORE    (STEP_END + SCORE)
#define ERFILTER  (STEP_END + RFILTER)
#define EPFILTERs (STEP_END + PFILTERs)
#define EPFILTER  (STEP_END + PFILTER)
#define ESIZEX    (STEP_END + SIZEX)
#define ESIZEY    (STEP_END + SIZEY)
#define EWEIGHTS  (STEP_END + WEIGHTS)
#define ETAGV     (STEP_END + TAGV)
#define EVx       (STEP_END + Vx)
#define EVy       (STEP_END + Vy)
#define ETAGD     (STEP_END + TAGD)
#define EDx       (STEP_END + Dx)
#define EDy       (STEP_END + Dy)
#define EDxx      (STEP_END + Dxx)
#define EDyy      (STEP_END + Dyy)
#define EBTAG     (STEP_END + BTAG)

#define EPCA          (STEP_END + PCA)
#define EWEIGHTSPCA   (STEP_END + WEIGHTSPCA)
#define ECASCADE_Th   (STEP_END + CASCADE_Th)
#define EHYPOTHES_PCA (STEP_END + HYPOTHES_PCA)
#define EDEFORM_PCA   (STEP_END + DEFORM_PCA)
#define EHYPOTHES     (STEP_END + HYPOTHES)
#define EDEFORM       (STEP_END + DEFORM)

#define EPCACOEFF     (STEP_END + PCACOEFF)

namespace cv
{
namespace lsvm
{

    int loadModel(
             // input parametr
              const char *modelPath,// model path
             
              // output parametrs
              CvLSVMFilterObjectCascade ***filters,
              int *kFilters, 
              int *kComponents, 
              int **kPartFilters, 
              float **b,
              float *scoreThreshold,
              float ** PCAcoeff);
}
}
#endif
