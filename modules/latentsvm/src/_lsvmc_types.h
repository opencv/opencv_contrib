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

#ifndef SVM_TYPE
#define SVM_TYPE

#include "float.h"

#define PI    CV_PI

#define EPS 0.000001

#define F_MAX FLT_MAX
#define F_MIN -FLT_MAX

// The number of elements in bin
// The number of sectors in gradient histogram building
#define NUM_SECTOR 9

// The number of levels in image resize procedure
// We need Lambda levels to resize image twice
#define LAMBDA 10

// Block size. Used in feature pyramid building procedure
#define SIDE_LENGTH 8

#define VAL_OF_TRUNCATE 0.2f 
namespace cv
{
namespace lsvm
{
//////////////////////////////////////////////////////////////
// main data structures                                     //
//////////////////////////////////////////////////////////////

// data type: STRUCT CvObjectDetection
// structure contains the bounding box and confidence level for detected object
// rect					- bounding box for a detected object
// score				- confidence level

typedef struct CvObjectDetection
{
    cv::Rect rect;
    float score;
} CvObjectDetection;


// DataType: STRUCT featureMap
// FEATURE MAP DESCRIPTION
//   Rectangular map (sizeX x sizeY), 
//   every cell stores feature vector (dimension = numFeatures)
// map             - matrix of feature vectors
//                   to set and get feature vectors (i,j) 
//                   used formula map[(j * sizeX + i) * p + k], where
//                   k - component of feature vector in cell (i, j)
typedef struct{
    int sizeX;
    int sizeY;
    int numFeatures;
    float *map;
} CvLSVMFeatureMapCascade;

// DataType: STRUCT featurePyramid
//
// numLevels    - number of levels in the feature pyramid
// pyramid      - array of pointers to feature map at different levels
typedef struct{
    int numLevels;
    CvLSVMFeatureMapCascade **pyramid;
} CvLSVMFeaturePyramidCascade;

// DataType: STRUCT filterDisposition
// The structure stores preliminary results in optimization process
// with objective function D 
//
// x            - array with X coordinates of optimization problems solutions
// y            - array with Y coordinates of optimization problems solutions
// score        - array with optimal objective values
typedef struct{
    float *score;
    int *x;
    int *y;
} CvLSVMFilterDisposition;

// DataType: STRUCT position
// Structure describes the position of the filter in the feature pyramid
// l - level in the feature pyramid
// (x, y) - coordinate in level l

typedef struct CvLSVMFilterPosition
{
    int x;
    int y;
    int l;
} CvLSVMFilterPosition;

// DataType: STRUCT filterObject
// Description of the filter, which corresponds to the part of the object
// V               - ideal (penalty = 0) position of the partial filter
//                   from the root filter position (V_i in the paper)
// penaltyFunction - vector describes penalty function (d_i in the paper)
//                   pf[0] * x + pf[1] * y + pf[2] * x^2 + pf[3] * y^2
// FILTER DESCRIPTION
//   Rectangular map (sizeX x sizeY),
//   every cell stores feature vector (dimension = p)
// H               - matrix of feature vectors
//                   to set and get feature vectors (i,j)
//                   used formula H[(j * sizeX + i) * p + k], where
//                   k - component of feature vector in cell (i, j)
// END OF FILTER DESCRIPTION

typedef struct CvLSVMFilterObjectCascade{
    CvLSVMFilterPosition V;
    float fineFunction[4];
    int sizeX;
    int sizeY;
    int numFeatures;
    float *H;
    float *H_PCA;
    float Hypothesis, Deformation;
    float Hypothesis_PCA, Deformation_PCA;
    int deltaX;
    int deltaY;
} CvLSVMFilterObjectCascade;

// data type: STRUCT CvLatentSvmDetector
// structure contains internal representation of trained Latent SVM detector
// num_filters			- total number of filters (root plus part) in model
// num_components		- number of components in model
// num_part_filters		- array containing number of part filters for each component
// filters				- root and part filters for all model components
// b					- biases for all model components
// score_threshold		- confidence level threshold

typedef struct CvLatentSvmDetectorCascade
{
    int num_filters;
    int num_components;
    int* num_part_filters;
    CvLSVMFilterObjectCascade** filters;
    float* b;
    float score_threshold;
    float *pca;
    int pca_size;
} CvLatentSvmDetectorCascade;
}
}
#endif
