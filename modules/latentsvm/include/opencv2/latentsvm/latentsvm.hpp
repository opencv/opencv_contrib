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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_OBJDETECT_HPP__
#define __OPENCV_OBJDETECT_HPP__

#include "opencv2/core/core.hpp"

#ifdef __cplusplus
#include <map>
#include <vector>
#endif

extern "C" {



/****************************************************************************************\
*                         Latent SVM Object Detection functions                          *
\****************************************************************************************/

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
typedef struct CvLSVMFilterObjectCaskade{
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
} CvLSVMFilterObjectCaskade;

// data type: STRUCT CvLatentSvmDetector
// structure contains internal representation of trained Latent SVM detector
// num_filters			- total number of filters (root plus part) in model
// num_components		- number of components in model
// num_part_filters		- array containing number of part filters for each component
// filters				- root and part filters for all model components
// b					- biases for all model components
// score_threshold		- confidence level threshold
typedef struct CvLatentSvmDetectorCaskade
{
    int num_filters;
    int num_components;
    int* num_part_filters;
    CvLSVMFilterObjectCaskade** filters;
    float* b;
    float score_threshold;
    float *pca;
    int pca_size;
}
CvLatentSvmDetectorCaskade;

// data type: STRUCT CvObjectDetection
// structure contains the bounding box and confidence level for detected object
// rect					- bounding box for a detected object
// score				- confidence level
typedef struct CvObjectDetection
{
    CvRect rect;
    float score;
} CvObjectDetection;
}
//////////////// Object Detection using Latent SVM //////////////


/*
// load trained detector from a file
//
// API
// CvLatentSvmDetector* cvLoadLatentSvmDetector(const char* filename);
// INPUT
// filename				- path to the file containing the parameters of
                        - trained Latent SVM detector
// OUTPUT
// trained Latent SVM detector in internal representation
*/
namespace cv
{
namespace lsvmc
{
CVAPI(CvLatentSvmDetectorCaskade*) cvLoadLatentSvmDetectorCaskade(const char* filename);

/*
// release memory allocated for CvLatentSvmDetector structure
//
// API
// void cvReleaseLatentSvmDetector(CvLatentSvmDetector** detector);
// INPUT
// detector				- CvLatentSvmDetector structure to be released
// OUTPUT
*/
CVAPI(void) cvReleaseLatentSvmDetectorCaskade(CvLatentSvmDetectorCaskade** detector);
/*
// find rectangular regions in the given image that are likely
// to contain objects and corresponding confidence levels
//
// API
// CvSeq* cvLatentSvmDetectObjects(const IplImage* image,
//									CvLatentSvmDetector* detector,
//									CvMemStorage* storage,
//									float overlap_threshold = 0.5f,
//                                  int numThreads = -1);
// INPUT
// image				- image to detect objects in
// detector				- Latent SVM detector in internal representation
// storage				- memory storage to store the resultant sequence
//							of the object candidate rectangles
// overlap_threshold	- threshold for the non-maximum suppression algorithm
                           = 0.5f [here will be the reference to original paper]
// OUTPUT
// sequence of detected objects (bounding boxes and confidence levels stored in CvObjectDetection structures)
*/
CVAPI(CvSeq*) cvLatentSvmDetectObjectsCaskade(IplImage* image,
                                CvLatentSvmDetectorCaskade* detector,
                                CvMemStorage* storage,
                                float overlap_threshold CV_DEFAULT(0.5f));
}
}
#ifdef __cplusplus



namespace cv
{

///////////////////////////// Object Detection ////////////////////////////

/*
 * This is a class wrapping up the structure CvLatentSvmDetector and functions working with it.
 * The class goals are:
 * 1) provide c++ interface;
 * 2) make it possible to load and detect more than one class (model) unlike CvLatentSvmDetector.
 */

namespace lsvmc
{
class CV_EXPORTS LatentSvmDetector
{
public:
    struct CV_EXPORTS ObjectDetection
    {
        ObjectDetection();
        ObjectDetection( const Rect& rect, float score, int classID=-1 );
        Rect rect;
        float score;
        int classID;
    };

    LatentSvmDetector();
    LatentSvmDetector( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );
    virtual ~LatentSvmDetector();

    virtual void clear();
    virtual bool empty() const;
    bool load( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );

    virtual void detect( const Mat& image,
                         vector<ObjectDetection>& objectDetections,
                         float overlapThreshold=0.5f);

    const vector<string>& getClassNames() const;
    size_t getClassCount() const;

private:
    vector<CvLatentSvmDetectorCaskade*> detectors;
    vector<string> classNames;
};
}
} // namespace cv

#endif

#endif
