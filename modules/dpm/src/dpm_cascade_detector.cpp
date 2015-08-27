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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "dpm_cascade.hpp"

using namespace std;

namespace cv
{
namespace dpm
{

class DPMDetectorImpl : public DPMDetector
{
public:

    DPMDetectorImpl( const vector<string>& filenames, const vector<string>& classNames=vector<string>() );
    ~DPMDetectorImpl();

    bool isEmpty() const;

    void detect(Mat &image, CV_OUT vector<ObjectDetection>& objects);

    const vector<string>& getClassNames() const;
    size_t getClassCount() const;
    string extractModelName( const string& filename );

private:
    vector< Ptr<DPMCascade> > detectors;
    vector<string> classNames;
};

Ptr<DPMDetector> DPMDetector::create(vector<string> const &filenames,
                                     vector<string> const &classNames)
{
    return makePtr<DPMDetectorImpl>(filenames, classNames);
}

DPMDetectorImpl::ObjectDetection::ObjectDetection()
    : score(0.f), classID(-1) {}

DPMDetectorImpl::ObjectDetection::ObjectDetection( const Rect& _rect, float _score, int _classID )
    : rect(_rect), score(_score), classID(_classID) {}


DPMDetectorImpl::DPMDetectorImpl( const vector<string>& filenames,
        const vector<string>& _classNames )
{
    for( size_t i = 0; i < filenames.size(); i++ )
    {
        const string filename = filenames[i];
        if( filename.length() < 5 || filename.substr(filename.length()-4, 4) != ".xml" )
            continue;

        Ptr<DPMCascade> detector = makePtr<DPMCascade>();

        // initialization
        detector->loadCascadeModel( filename.c_str() );

        if( detector )
        {
            detectors.push_back( detector );
            if( _classNames.empty() )
            {
                classNames.push_back( extractModelName(filename));
            }
            else
                classNames.push_back( _classNames[i] );
        }
    }
}

DPMDetectorImpl::~DPMDetectorImpl()
{
}

bool DPMDetectorImpl::isEmpty() const
{
    return detectors.empty();
}

const vector<string>& DPMDetectorImpl::getClassNames() const
{
    return classNames;
}

size_t DPMDetectorImpl::getClassCount() const
{
    return classNames.size();
}

string DPMDetectorImpl::extractModelName( const string& filename )
{
    size_t startPos = filename.rfind('/');
    if( startPos == string::npos )
        startPos = filename.rfind('\\');

    if( startPos == string::npos )
        startPos = 0;
    else
        startPos++;

    const int extentionSize = 4; //.xml

    int substrLength = (int)(filename.size() - startPos - extentionSize);

    return filename.substr(startPos, substrLength);
}

void DPMDetectorImpl::detect( Mat &image,
        vector<ObjectDetection> &objectDetections)
{
    objectDetections.clear();

    for( size_t classID = 0; classID < detectors.size(); classID++ )
    {
        // detect objects
        vector< vector<double> > detections;
        detections = detectors[classID]->detect(image);

        for (unsigned int i = 0; i < detections.size(); i++)
        {
            ObjectDetection ds = ObjectDetection();
            int s = (int)detections[i].size() - 1;
            ds.score = (float)detections[i][s];
            int x1 = (int)detections[i][0];
            int y1 = (int)detections[i][1];
            int w = (int)detections[i][2] - x1 + 1;
            int h = (int)detections[i][3] - y1 + 1;
            ds.rect = Rect(x1, y1, w, h);
            ds.classID = (int)classID;

            objectDetections.push_back(ds);
        }
    }
}

} // namespace cv
}
