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
#include <limits.h>

//these should become parameters
#define POINTNUM 20

#define HYPO(a,b) (t1=(a),t2=(b),sqrt(t1*t1+t2*t2))
#define SAME(a,b) (norm((a)-(b))==0)

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
  void setImage(const Mat& image){image.copyTo(image_);}
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
     static float getMedian(std::vector<float> values,int size=-1);
     static float dist(Point2f p1,Point2f p2);
     static std::string type2str(int type);
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

bool TrackerMedianFlow::initImpl( const Mat& image, const Rect& boundingBox ){
    /*for(int s=140, i=s;i<s+60;i++){for(int sj=280, j=sj;j<sj+10;j++){
        printf("(%d,%d) pixel (%d,%d,%d)\n",i,j,image.at<Vec3b>(j,i).val[0],image.at<Vec3b>(j,i).val[1],image.at<Vec3b>(j,i).val[2]);
    }}*/
    /*int i=165,j=284;
    printf("\n");
    printf("(%d,%d) pixel (%d,%d,%d)\n",i,j,image.at<Vec3b>(j,i).val[0],image.at<Vec3b>(j,i).val[1],image.at<Vec3b>(j,i).val[2]);*/

    /*Mat oldImage_gray;
    cvtColor( image, oldImage_gray, CV_BGR2GRAY );
    std::vector<Point2f> features;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10);
    goodFeaturesToTrack(oldImage_gray, features, 500, 0.01, 10, Mat(), 3, 0, 0.04);
    cornerSubPix(oldImage_gray,features, subPixWinSize, Size(-1,-1), termcrit);
    for(int i=0;i<features.size();i++){
        printf("fea #%d -- (%d,%d)\n",i,(int)features[i].x,(int)features[i].y);
    }
    printf("%dx%d\n",oldImage_gray.cols,oldImage_gray.rows);
    exit(0);*/

    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel());
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlow::updateImpl( const Mat& image, Rect& boundingBox ){
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();

    /*int i=165,j=284;
    printf("\n");
    printf("(%d,%d) pixel (%d,%d,%d)\n",i,j,image.at<Vec3b>(j,i).val[0],image.at<Vec3b>(j,i).val[1],image.at<Vec3b>(j,i).val[2]);
    printf("\n");
    printf("(%d,%d) old pixel (%d,%d,%d)\n",i,j,oldImage.at<Vec3b>(j,i).val[0],oldImage.at<Vec3b>(j,i).val[1],oldImage.at<Vec3b>(j,i).val[2]);*/

    /*if(SAME(image,oldImage)){
        printf("same in updateImpl()!\n");
    }else{
        printf("diff in updateImpl()!\n");
    }
    exit(0);*/

    Rect oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    boundingBox=MedianFlowCore::medianFlowImpl(oldImage,image,oldBox,params);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

std::string MedianFlowCore::type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
Rect MedianFlowCore::medianFlowImpl(Mat oldImage,Mat newImage,Rect oldBox,TrackerMedianFlow::Params params){
    //make grid 20x20
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;
    float t1,t2;
    for(int i=0;i<POINTNUM;i++){
        for(int j=0;j<POINTNUM;j++){
                pointsToTrackOld.push_back(Point2f(oldBox.x+(1.0*oldBox.width/POINTNUM)*i,oldBox.y+(1.0*oldBox.height/POINTNUM)*j));
        }
    }

    Mat oldImage_gray,newImage_gray;
    cvtColor( oldImage, oldImage_gray, CV_BGR2GRAY );
    cvtColor( newImage, newImage_gray, CV_BGR2GRAY );

    /*pointsToTrackOld.clear();
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10);
    goodFeaturesToTrack(oldImage_gray, pointsToTrackOld, 500, 0.01, 10, Mat(), 3, 0, 0.04);
    cornerSubPix(oldImage_gray,pointsToTrackOld, subPixWinSize, Size(-1,-1), termcrit);
    for(int i=0;i<pointsToTrackOld.size();i++){
        printf("fea #%d -- (%d,%d)\n",i,(int)pointsToTrackOld[i].x,(int)pointsToTrackOld[i].y);
    }*/

    std::vector<uchar> status(pointsToTrackOld.size());
    std::vector<float> errors(pointsToTrackOld.size());
    calcOpticalFlowPyrLK(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors);
    for(int i=0;i<pointsToTrackOld.size();i++){
        if(status[i]==0){
            pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
            pointsToTrackNew.erase(pointsToTrackNew.begin()+i);
            status.erase(status.begin()+i);
            i--;
        }
    }
    printf("\t%d after LK forward\n",pointsToTrackOld.size());

    //      compute FB error
    std::vector<float> FBerror(pointsToTrackOld.size());
    std::vector<Point2f> pointsToTrackReprojection;
    calcOpticalFlowPyrLK(newImage_gray,oldImage_gray,pointsToTrackNew,pointsToTrackReprojection,status,errors);
    for(int i=0;i<pointsToTrackOld.size();i++){
        if(status[i]==0){
            FBerror[i]=FLT_MAX;
        }else{
            FBerror[i]=HYPO(pointsToTrackOld[i].x-pointsToTrackReprojection[i].x,pointsToTrackOld[i].y-pointsToTrackReprojection[i].y);
        }
    }
    float FBerrorMedian=getMedian(FBerror);
    printf("FBerrorMedian=%f\n",FBerrorMedian);

    //      compute NCC -- TODO

    // filter
    for(int i=0;i<pointsToTrackOld.size();i++){
        if(FBerror[i]>FBerrorMedian){
            pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
            pointsToTrackNew.erase(pointsToTrackNew.begin()+i);
            status.erase(status.begin()+i);
            FBerror.erase(FBerror.begin()+i);
            i--;
        }
    }
    // vote
    printf("\t%d after LK backward\n",pointsToTrackOld.size());

    CV_Assert(pointsToTrackOld.size()>0);
    return vote(pointsToTrackOld,pointsToTrackNew,oldBox);
}
Rect MedianFlowCore::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect& oldRect){
    float t1,t2;
    Rect newRect;
    Point newCenter(oldRect.x+oldRect.width/2,oldRect.y+oldRect.height/2);
    int n=oldPoints.size();
    std::vector<float> buf(n*(n-1));

    if(oldPoints.size()==1){
        newRect.x=newCenter.x-oldRect.width/2;
        newRect.y=newCenter.y-oldRect.height/2;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

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
    printf("oldRect.width=%d\n oldRect.height=%d\n newCenter.x=%d\n newCenter.y=%d\n", oldRect.width,oldRect.height,newCenter.x,newCenter.y);
    newRect.x=newCenter.x-scale*oldRect.width/2;
    newRect.y=newCenter.y-scale*oldRect.height/2;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    printf("[%d %d %d %d]\n",newRect.x,newRect.y,newRect.x+newRect.width,newRect.y+newRect.height);
    if(newRect.x<=0){
        exit(0);
    }
    return newRect;
}
float MedianFlowCore::getMedian(std::vector<float> values,int size){
    if(size==-1){
        size=values.size();
    }
    std::sort(values.begin(),values.begin()+size);
    return values[size/2];
}
} /* namespace cv */
