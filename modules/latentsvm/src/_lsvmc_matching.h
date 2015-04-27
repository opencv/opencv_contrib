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

/*****************************************************************************/
/*                      Matching procedure API                               */
/*****************************************************************************/
//
#ifndef _LSVM_MATCHING_H_
#define _LSVM_MATCHING_H_

#include "_lsvmc_latentsvm.h"
#include "_lsvmc_error.h"
#include "_lsvmc_routine.h"

namespace cv
{
namespace lsvm
{


/*
// Computation border size for feature map
//
// API
// int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);
// INPUT
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// bx                - border size (X-direction)
// by                - border size (Y-direction)
// RESULT
// Error status
*/
int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);

/*
// Addition nullable border to the feature map
//
// API
// int addNullableBorder(featureMap *map, int bx, int by);
// INPUT
// map               - feature map
// bx                - border size (X-direction)
// by                - border size (Y-direction)
// OUTPUT
// RESULT
// Error status
*/
int addNullableBorder(CvLSVMFeatureMapCascade *map, int bx, int by);

/*
// Perform non-maximum suppression algorithm (described in original paper)
// to remove "similar" bounding boxes
//
// API
// int nonMaximumSuppression(int numBoxes, const CvPoint *points, 
                             const CvPoint *oppositePoints, const float *score,
                             float overlapThreshold, 
                             int *numBoxesout, CvPoint **pointsOut, 
                             CvPoint **oppositePointsOut, float **scoreOut);
// INPUT
// numBoxes          - number of bounding boxes
// points            - array of left top corner coordinates
// oppositePoints    - array of right bottom corner coordinates
// score             - array of detection scores
// overlapThreshold  - threshold: bounding box is removed if overlap part 
					   is greater than passed value
// OUTPUT
// numBoxesOut       - the number of bounding boxes algorithm returns
// pointsOut         - array of left top corner coordinates
// oppositePointsOut - array of right bottom corner coordinates
// scoreOut          - array of detection scores
// RESULT
// Error status
*/
int nonMaximumSuppression(int numBoxes, const CvPoint *points, 
                          const CvPoint *oppositePoints, const float *score,
                          float overlapThreshold, 
                          int *numBoxesOut, CvPoint **pointsOut, 
                          CvPoint **oppositePointsOut, float **scoreOut);
int getMaxFilterDims(const CvLSVMFilterObjectCascade **filters, int kComponents,
                     const int *kPartFilters, 
                     unsigned int *maxXBorder, unsigned int *maxYBorder);
//}

int getMaxFilterDims(const CvLSVMFilterObjectCascade **filters, int kComponents,
                     const int *kPartFilters, 
                     unsigned int *maxXBorder, unsigned int *maxYBorder);
}
}
#endif
