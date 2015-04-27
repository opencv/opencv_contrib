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
#include "_lsvmc_parser.h"
#include "_lsvmc_matching.h"
namespace cv
{
namespace lsvm
{

std::string extractModelName( const std::string& filename );

const int pca_size = 31;

CvLatentSvmDetectorCascade* cvLoadLatentSvmDetectorCascade(const char* filename);
void cvReleaseLatentSvmDetectorCascade(CvLatentSvmDetectorCascade** detector);
CvSeq* cvLatentSvmDetectObjectsCascade(IplImage* image,
                                CvLatentSvmDetectorCascade* detector,
                                CvMemStorage* storage,
                                float overlap_threshold);

/*
// load trained detector from a file
//
// API
// CvLatentSvmDetectorCascade* cvLoadLatentSvmDetector(const char* filename);
// INPUT
// filename             - path to the file containing the parameters of
//                      - trained Latent SVM detector
// OUTPUT
// trained Latent SVM detector in internal representation
*/
CvLatentSvmDetectorCascade* cvLoadLatentSvmDetectorCascade(const char* filename)
{
    CvLatentSvmDetectorCascade* detector = 0;
    CvLSVMFilterObjectCascade** filters = 0;
    int kFilters = 0;
    int kComponents = 0;
    int* kPartFilters = 0;
    float* b = 0;
    float scoreThreshold = 0.f;
    int err_code = 0;
	float* PCAcoeff = 0;

    err_code = loadModel(filename, &filters, &kFilters, &kComponents, &kPartFilters, &b, &scoreThreshold, &PCAcoeff);
    if (err_code != LATENT_SVM_OK) return 0;

    detector = (CvLatentSvmDetectorCascade*)malloc(sizeof(CvLatentSvmDetectorCascade));
    detector->filters = filters;
    detector->b = b;
    detector->num_components = kComponents;
    detector->num_filters = kFilters;
    detector->num_part_filters = kPartFilters;
    detector->score_threshold = scoreThreshold;
	  detector->pca = PCAcoeff;
    detector->pca_size = pca_size;

    return detector;
}

/*
// release memory allocated for CvLatentSvmDetectorCascade structure
//
// API
// void cvReleaseLatentSvmDetector(CvLatentSvmDetectorCascade** detector);
// INPUT
// detector             - CvLatentSvmDetectorCascade structure to be released
// OUTPUT
*/
void cvReleaseLatentSvmDetectorCascade(CvLatentSvmDetectorCascade** detector)
{
    free((*detector)->b);
    free((*detector)->num_part_filters);
    for (int i = 0; i < (*detector)->num_filters; i++)
    {
        free((*detector)->filters[i]->H);
        free((*detector)->filters[i]);
    }
    free((*detector)->filters);
	free((*detector)->pca);
    free((*detector));
    *detector = 0;
}

/*
// find rectangular regions in the given image that are likely
// to contain objects and corresponding confidence levels
//
// API
// CvSeq* cvLatentSvmDetectObjects(const IplImage* image,
//                                  CvLatentSvmDetectorCascade* detector,
//                                  CvMemStorage* storage,
//                                  float overlap_threshold = 0.5f);
// INPUT
// image                - image to detect objects in
// detector             - Latent SVM detector in internal representation
// storage              - memory storage to store the resultant sequence
//                          of the object candidate rectangles
// overlap_threshold    - threshold for the non-maximum suppression algorithm [here will be the reference to original paper]
// OUTPUT
// sequence of detected objects (bounding boxes and confidence levels stored in CvObjectDetection structures)
*/
CvSeq* cvLatentSvmDetectObjectsCascade(IplImage* image,
                                CvLatentSvmDetectorCascade* detector,
                                CvMemStorage* storage,
                                float overlap_threshold)
{
    CvLSVMFeaturePyramidCascade *H = 0;
	CvLSVMFeaturePyramidCascade *H_PCA = 0;
    CvPoint *points = 0, *oppPoints = 0;
    int kPoints = 0;
    float *score = 0;
    unsigned int maxXBorder = 0, maxYBorder = 0;
    int numBoxesOut = 0;
    CvPoint *pointsOut = 0;
    CvPoint *oppPointsOut = 0;
    float *scoreOut = 0;
    CvSeq* result_seq = 0;
    int error = 0;

    if(image->nChannels == 3)
        cvCvtColor(image, image, CV_BGR2RGB);

    // Getting maximum filter dimensions
    getMaxFilterDims((const CvLSVMFilterObjectCascade**)(detector->filters), detector->num_components,
                     detector->num_part_filters, &maxXBorder, &maxYBorder);
    // Create feature pyramid with nullable border
    H = createFeaturePyramidWithBorder(image, maxXBorder, maxYBorder);
	
	// Create PCA feature pyramid
    H_PCA = createPCA_FeaturePyramid(H, detector, maxXBorder, maxYBorder);
    
    FeaturePyramid32(H, maxXBorder, maxYBorder);
	
    // Search object
    error = searchObjectThresholdSomeComponents(H, H_PCA,(const CvLSVMFilterObjectCascade**)(detector->filters),
        detector->num_components, detector->num_part_filters, detector->b, detector->score_threshold,
        &points, &oppPoints, &score, &kPoints);
    if (error != LATENT_SVM_OK)
    {
        return NULL;
    }
    // Clipping boxes
    clippingBoxes(image->width, image->height, points, kPoints);
    clippingBoxes(image->width, image->height, oppPoints, kPoints);
    // NMS procedure
    nonMaximumSuppression(kPoints, points, oppPoints, score, overlap_threshold,
                &numBoxesOut, &pointsOut, &oppPointsOut, &scoreOut);

    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvObjectDetection), storage );

    for (int i = 0; i < numBoxesOut; i++)
    {
        CvObjectDetection detection;
        detection.score = scoreOut[i];
        detection.rect.x = pointsOut[i].x;
        detection.rect.y = pointsOut[i].y;
        detection.rect.width = oppPointsOut[i].x - pointsOut[i].x;
        detection.rect.height = oppPointsOut[i].y - pointsOut[i].y;
        cvSeqPush(result_seq, &detection);
    }

    if(image->nChannels == 3)
        cvCvtColor(image, image, CV_RGB2BGR);

    freeFeaturePyramidObject(&H);
	freeFeaturePyramidObject(&H_PCA);
    free(points);
    free(oppPoints);
    free(score);

    return result_seq;
}

class LSVMDetectorImpl : public LSVMDetector
{
public:

    LSVMDetectorImpl( const std::vector<std::string>& filenames, const std::vector<std::string>& classNames=std::vector<std::string>() );
    ~LSVMDetectorImpl();

    bool isEmpty() const;

    void detect(cv::Mat const &image, CV_OUT std::vector<ObjectDetection>& objects, float overlapThreshold=0.5f);

    const std::vector<std::string>& getClassNames() const;
    size_t getClassCount() const;

private:
    std::vector<CvLatentSvmDetectorCascade*> detectors;
    std::vector<std::string> classNames;
};

cv::Ptr<LSVMDetector> LSVMDetector::create(std::vector<std::string> const &filenames,
                                     std::vector<std::string> const &classNames)
{
    return cv::makePtr<LSVMDetectorImpl>(filenames, classNames);
}

LSVMDetectorImpl::ObjectDetection::ObjectDetection() : score(0.f), classID(-1) {}

LSVMDetectorImpl::ObjectDetection::ObjectDetection( const Rect& _rect, float _score, int _classID ) :
    rect(_rect), score(_score), classID(_classID) {}


LSVMDetectorImpl::LSVMDetectorImpl( const std::vector<std::string>& filenames, const std::vector<std::string>& _classNames )
{
    for( size_t i = 0; i < filenames.size(); i++ )
    {
        const std::string filename = filenames[i];
        if( filename.length() < 5 || filename.substr(filename.length()-4, 4) != ".xml" )
            continue;

        CvLatentSvmDetectorCascade* detector = cvLoadLatentSvmDetectorCascade( filename.c_str() );
        if( detector )
        {
            detectors.push_back( detector );
            if( _classNames.empty() )
            {
                classNames.push_back( extractModelName(filenames[i]) );
            }
            else
                classNames.push_back( _classNames[i] );
        }
    }
}

LSVMDetectorImpl::~LSVMDetectorImpl()
{
    for(size_t i = 0; i < detectors.size(); i++)
      cv::lsvm::cvReleaseLatentSvmDetectorCascade(&detectors[i]);
}

bool LSVMDetectorImpl::isEmpty() const
{
    return detectors.empty();
}

const std::vector<std::string>& LSVMDetectorImpl::getClassNames() const
{
    return classNames;
}

size_t LSVMDetectorImpl::getClassCount() const
{
    return classNames.size();
}

std::string extractModelName( const std::string& filename )
{
    size_t startPos = filename.rfind('/');
    if( startPos == std::string::npos )
        startPos = filename.rfind('\\');

    if( startPos == std::string::npos )
        startPos = 0;
    else
        startPos++;

    const int extentionSize = 4; //.xml

    int substrLength = (int)(filename.size() - startPos - extentionSize);

    return filename.substr(startPos, substrLength);
}

void LSVMDetectorImpl::detect( cv::Mat const &image,
                               std::vector<ObjectDetection> &objectDetections,
                               float overlapThreshold)
{
    objectDetections.clear();
    
    for( size_t classID = 0; classID < detectors.size(); classID++ )
    {
        IplImage image_ipl = image;
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* detections = cv::lsvm::cvLatentSvmDetectObjectsCascade( &image_ipl, (CvLatentSvmDetectorCascade*)(detectors[classID]), storage, overlapThreshold);

        // convert results
        objectDetections.reserve( objectDetections.size() + detections->total );
        for( int detectionIdx = 0; detectionIdx < detections->total; detectionIdx++ )
        {
            CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( detections, detectionIdx );
            objectDetections.push_back( ObjectDetection(Rect(detection.rect), detection.score, (int)classID) );
        }

        cvReleaseMemStorage( &storage );
    }
}

} // namespace cv
}
