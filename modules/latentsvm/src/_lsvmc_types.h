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
namespace lsvmc
{

//////////////////////////////////////////////////////////////
// main data structures                                     //
//////////////////////////////////////////////////////////////

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
} CvLSVMFeatureMapCaskade;

// DataType: STRUCT featurePyramid
//
// numLevels    - number of levels in the feature pyramid
// pyramid      - array of pointers to feature map at different levels
typedef struct{
    int numLevels;
    CvLSVMFeatureMapCaskade **pyramid;
} CvLSVMFeaturePyramidCaskade;

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
}
}
#endif
