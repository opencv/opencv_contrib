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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#include "opencv2/video/tracking.hpp"
#include <algorithm>

//these should become parameters
#define POINTNUM 20

#define HYPO(a,b) (t1=(a),t2=(b),sqrt(t1*t1+t2*t2))

namespace cv
{

/*
 *  TrackerMedianFlow
 */

class TrackerMedianFlowModel : public TrackerModel
{
 public:
  TrackerMedianFlowModel(){}
  Rect getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect boundingBox){boundingBox_=boundingBox;}
  Mat getImage(){return image_;}
  void setImage(Mat image){image_=image;}
  bool setTrackerStateEstimator( Ptr<TrackerStateEstimator> /*trackerStateEstimator*/ ){return false;}
  void modelEstimation( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdate(){}
  bool runStateEstimator(){return false;}
  void setLastTargetState( const Ptr<TrackerTargetState>& /*lastTargetState*/ ){}
  Ptr<TrackerTargetState> getLastTargetState() const{return trajectory.back();}
  const std::vector<ConfidenceMap>& getConfidenceMaps() const{return confidenceMaps;}
  const ConfidenceMap& getLastConfidenceMap() const{return currentConfidenceMap;}
  Ptr<TrackerStateEstimator> getTrackerStateEstimator() const{return stateEstimator;}
 private:
  void clearCurrentConfidenceMap(){}
 protected:
  Rect boundingBox_;
  Mat image_;
  std::vector<ConfidenceMap> confidenceMaps;
  Ptr<TrackerStateEstimator> stateEstimator;
  ConfidenceMap currentConfidenceMap;
  Trajectory trajectory;
  int maxCMLength;
  void modelEstimationImpl( const std::vector<Mat>& responses ){}
  void modelUpdateImpl(){}
};

class MedianFlowCore{
 public:
     static Rect medianFlowImpl(Mat oldImage,Mat newImage,Rect oldBox,TrackerMedianFlow::Params params);
 private:
     static Rect vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect& oldRect);
     //FIXME: this can be optimized: current method uses sort->select approach, there are O(n) selection algo for median
     static float getMedian(std::vector<float> values,int size);
     static float dist(Point2f p1,Point2f p2);
};

/*
 * Parameters
 */
TrackerMedianFlow::Params::Params()
{
}

void TrackerMedianFlow::Params::read( const cv::FileNode& /*fn*/ )
{
  //numClassifiers = fn["numClassifiers"];
}

void TrackerMedianFlow::Params::write( cv::FileStorage& /*fs*/ ) const
{
  //fs << "numClassifiers" << numClassifiers;
}

/*
 * Constructor
 */
TrackerMedianFlow::TrackerMedianFlow( const TrackerMedianFlow::Params &parameters) :
    params( parameters )
{
  isInit = false;
}

/*
 * Destructor
 */
TrackerMedianFlow::~TrackerMedianFlow()
{
}

void TrackerMedianFlow::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerMedianFlow::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

bool TrackerMedianFlow::initImpl( const Mat& image, const Rect& boundingBox )
{
    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel());
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlow::updateImpl( const Mat& image, Rect& boundingBox )
{
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();
    Rect oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    boundingBox=MedianFlowCore::medianFlowImpl(oldImage,image,oldBox,params);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

Rect MedianFlowCore::medianFlowImpl(Mat oldImage,Mat newImage,Rect oldBox,TrackerMedianFlow::Params params){
    //make grid 20x20
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;
    for(int i=0;i<POINTNUM;i++){
        for(int j=0;j<POINTNUM;j++){
                pointsToTrackOld.push_back(Point2f(oldBox.x+(1.0*oldBox.width/POINTNUM)*i,oldBox.y+(1.0*oldBox.height/POINTNUM)*j));
        }
    }
    std::vector<uchar> status;
    Mat errors;
    calcOpticalFlowPyrLK(oldImage,newImage,pointsToTrackOld,pointsToTrackNew,status,errors);
    for(int i=0;i<pointsToTrackOld.size();i++){
        if(status[i]==0){
            pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
            pointsToTrackNew.erase(pointsToTrackNew.begin()+i);
            status.erase(status.begin()+i);
            i--;
        }
    }
    for(int i=0;i<pointsToTrackNew.size();i++){
        printf("(%f,%f)vs(%f,%f)\n",pointsToTrackOld[i].x,pointsToTrackOld[i].y,pointsToTrackNew[i].x,pointsToTrackNew[i].y);
    }
    //for every point:
    //      compute FB error
    //      compute NCC
    // filter
    // vote
    CV_Assert(pointsToTrackOld.size()>0);
    return vote(pointsToTrackOld,pointsToTrackNew,oldBox);
}
Rect MedianFlowCore::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect& oldRect){
    float t1,t2;
    Rect newRect;
    Point newCenter(oldRect.x+oldRect.width/2,oldRect.y+oldRect.height/2);
    int n=oldPoints.size();
    std::vector<float> buf(n*(n-1));

    for(int i=0;i<n;i++){  buf[i]=newPoints[i].x-oldPoints[i].x;  }
    newCenter.x+=getMedian(buf,n);
    printf("shift_x=%f\n",getMedian(buf,n));
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].y-oldPoints[i].y;  }
    newCenter.y+=getMedian(buf,n);
    printf("shift_y=%f\n",getMedian(buf,n));

    if(oldPoints.size()==1){
        newRect.x=newCenter.x-oldRect.width/2;
        newRect.y=newCenter.y-oldRect.height/2;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    float nd,od;
    for(int i=0,ctr=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            nd=HYPO(newPoints[i].x-newPoints[j].x,newPoints[i].y-newPoints[j].y);
            od=HYPO(oldPoints[i].x-oldPoints[j].x,oldPoints[i].y-oldPoints[j].y);
            buf[ctr]=nd/od;
            ctr++;
        }
    }

    float scale=getMedian(buf,n*(n-1));
    printf("scale=%f\n",scale);
    newRect.x=newCenter.x-scale*oldRect.width/2;
    newRect.y=newCenter.y-scale*oldRect.height/2;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    printf("\n");
    return newRect;
}
float MedianFlowCore::getMedian(std::vector<float> values,int size){
    std::sort(values.begin(),values.begin()+size);
    return values[size/2];
}
} /* namespace cv */
