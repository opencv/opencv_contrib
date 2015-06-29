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
/*                      Latent SVM prediction API                            */
/*****************************************************************************/

#ifndef _LATENTSVM_H_
#define _LATENTSVM_H_

#include <stdio.h>
#include "_lsvmc_types.h"
#include "_lsvmc_error.h"
#include "_lsvmc_routine.h"

namespace cv
{
namespace lsvm
{

//////////////////////////////////////////////////////////////
// Building feature pyramid
// (pyramid constructed both contrast and non-contrast image)
//////////////////////////////////////////////////////////////

void FeaturePyramid32(CvLSVMFeaturePyramidCascade* H, int maxX, int maxY);

/*
// Creation PCA feature pyramid
//
// API
// featurePyramid* createPCA_FeaturePyramid(featurePyramid* H);

// INPUT
// H                 - feature pyramid     
// OUTPUT
// RESULT
// PCA feature pyramid
*/
CvLSVMFeaturePyramidCascade* createPCA_FeaturePyramid(CvLSVMFeaturePyramidCascade* H, 
                                               CvLatentSvmDetectorCascade* detector, 
                                               int maxX, int maxY);

/*
// Getting feature pyramid  
//
// API
// int getFeaturePyramid(IplImage * image, const CvLSVMFilterObjectCascade **all_F, 
                      const int n_f,
                      const int lambda, const int k, 
                      const int startX, const int startY, 
                      const int W, const int H, featurePyramid **maps);
// INPUT
// image             - image
// lambda            - resize scale
// k                 - size of cells
// startX            - X coordinate of the image rectangle to search
// startY            - Y coordinate of the image rectangle to search
// W                 - width of the image rectangle to search
// H                 - height of the image rectangle to search
// OUTPUT
// maps              - feature maps for all levels
// RESULT
// Error status
*/
int getFeaturePyramid(IplImage * image, CvLSVMFeaturePyramidCascade **maps);

/*
// Getting feature map for the selected subimage  
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int getFeatureMaps(const IplImage * image, const int k, CvLSVMFeatureMapCascade **map);


/*
// Feature map Normalization and Truncation 
//
// API
// int normalizationAndTruncationFeatureMaps(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
int normalizeAndTruncate(CvLSVMFeatureMapCascade *map, const float alfa);

/*
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int PCAFeatureMaps(CvLSVMFeatureMapCascade *map);

//////////////////////////////////////////////////////////////
// search object
//////////////////////////////////////////////////////////////

/*
// Transformation filter displacement from the block space 
// to the space of pixels at the initial image
//
// API
// int convertPoints(int countLevel, int lambda, 
                     int initialImageLevel,
                     CvPoint *points, int *levels, 
                     CvPoint **partsDisplacement, int kPoints, int n, 
                     int maxXBorder,
                     int maxYBorder);
// INPUT
// countLevel        - the number of levels in the feature pyramid
// lambda            - method parameter
// initialImageLevel - level of feature pyramid that contains feature map
                       for initial image
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// partsDisplacement - displacement of part filters (in the block space)
// kPoints           - number of root filter positions
// n                 - number of part filters
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// points            - the set of root filter positions (in the space of pixels)
// partsDisplacement - displacement of part filters (in the space of pixels)
// RESULT
// Error status
*/
int convertPoints(int countLevel, int lambda, 
                  int initialImageLevel,
                  CvPoint *points, int *levels, 
                  CvPoint **partsDisplacement, int kPoints, int n, 
                  int maxXBorder,
                  int maxYBorder);

/*
// Elimination boxes that are outside the image boudaries
//
// API
// int clippingBoxes(int width, int height, 
                     CvPoint *points, int kPoints);
// INPUT
// width             - image wediht
// height            - image heigth
// points            - a set of points (coordinates of top left or
                       bottom right corners)
// kPoints           - points number
// OUTPUT
// points            - updated points (if coordinates less than zero then
                       set zero coordinate, if coordinates more than image 
                       size then set coordinates equal image size)
// RESULT
// Error status
*/
int clippingBoxes(int width, int height, 
                  CvPoint *points, int kPoints);

/*
// Creation feature pyramid with nullable border
//
// API
// featurePyramid* createFeaturePyramidWithBorder(const IplImage *image,
                                                  int maxXBorder, int maxYBorder);

// INPUT
// image             - initial image     
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// RESULT
// Feature pyramid with nullable border
*/
CvLSVMFeaturePyramidCascade* createFeaturePyramidWithBorder(IplImage *image,
                                               int maxXBorder, int maxYBorder);

/*
// Computation root filters displacement and values of score function
//
// API
// int searchObjectThresholdSomeComponents(const featurePyramid *H,
                                           const CvLSVMFilterObjectCascade **filters, 
                                           int kComponents, const int *kPartFilters,
                                           const float *b, float scoreThreshold,
                                           CvPoint **points, CvPoint **oppPoints,
                                           float **score, int *kPoints);
// INPUT
// H                 - feature pyramid
// filters           - filters (root filter then it's part filters, etc.)
// kComponents       - root filters number
// kPartFilters      - array of part filters number for each component
// b                 - array of linear terms
// scoreThreshold    - score threshold
// OUTPUT
// points            - root filters displacement (top left corners)
// oppPoints         - root filters displacement (bottom right corners)
// score             - array of score values
// kPoints           - number of boxes
// RESULT
// Error status
*/
int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramidCascade *H,
										const CvLSVMFeaturePyramidCascade *H_PCA,
                                        const CvLSVMFilterObjectCascade **filters, 
                                        int kComponents, const int *kPartFilters,
                                        const float *b, float scoreThreshold,
                                        CvPoint **points, CvPoint **oppPoints,
                                        float **score, int *kPoints);

/*
// Compute opposite point for filter box
//
// API
// int getOppositePoint(CvPoint point,
                        int sizeX, int sizeY,
                        float step, int degree,
                        CvPoint *oppositePoint);

// INPUT
// point             - coordinates of filter top left corner
                       (in the space of pixels)
// (sizeX, sizeY)    - filter dimension in the block space
// step              - scaling factor
// degree            - degree of the scaling factor
// OUTPUT
// oppositePoint     - coordinates of filter bottom corner
                       (in the space of pixels)
// RESULT
// Error status
*/
int getOppositePoint(CvPoint point,
                     int sizeX, int sizeY,
                     float step, int degree,
                     CvPoint *oppositePoint);

/*
// Drawing root filter boxes
//
// API
// int showRootFilterBoxes(const IplImage *image,
                           const CvLSVMFilterObjectCascade *filter, 
                           CvPoint *points, int *levels, int kPoints,
                           CvScalar color, int thickness, 
                           int line_type, int shift);
// INPUT
// image             - initial image
// filter            - root filter object
// points            - a set of points
// levels            - levels of feature pyramid
// kPoints           - number of points
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// window contained initial image and filter boxes
// RESULT
// Error status
*/
int showRootFilterBoxes(IplImage *image,
                        const CvLSVMFilterObjectCascade *filter, 
                        CvPoint *points, int *levels, int kPoints,
                        CvScalar color, int thickness, 
                        int line_type, int shift);

/*
// Drawing part filter boxes
//
// API
// int showPartFilterBoxes(const IplImage *image,
                           const CvLSVMFilterObjectCascade *filter, 
                           CvPoint *points, int *levels, int kPoints,
                           CvScalar color, int thickness, 
                           int line_type, int shift);
// INPUT
// image             - initial image
// filters           - a set of part filters
// n                 - number of part filters
// partsDisplacement - a set of points
// levels            - levels of feature pyramid
// kPoints           - number of foot filter positions
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// window contained initial image and filter boxes
// RESULT
// Error status
*/
int showPartFilterBoxes(IplImage *image,
                        const CvLSVMFilterObjectCascade **filters,
                        int n, CvPoint **partsDisplacement, 
                        int *levels, int kPoints,
                        CvScalar color, int thickness, 
                        int line_type, int shift);

/*
// Drawing boxes
//
// API
// int showBoxes(const IplImage *img, 
                 const CvPoint *points, const CvPoint *oppositePoints, int kPoints, 
                 CvScalar color, int thickness, int line_type, int shift);
// INPUT
// img               - initial image
// points            - top left corner coordinates
// oppositePoints    - right bottom corner coordinates
// kPoints           - points number
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// RESULT
// Error status
*/
int showBoxes(IplImage *img, 
              const CvPoint *points, const CvPoint *oppositePoints, int kPoints, 
              CvScalar color, int thickness, int line_type, int shift);
}
}
#endif
