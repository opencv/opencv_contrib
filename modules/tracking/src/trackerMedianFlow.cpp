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
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <limits.h>

namespace cv
{

/*
 *  TrackerMedianFlow
 */
/*
 * TODO:
 * take all parameters out 
 * employ NCC
 * bring more work to constructor -- TODO
 * (if bad, try floating-point output)
 *              add "non-detected" answer in algo
 *              asessment framework
 *
 *
 * FIXME:
 * when patch is cut from image to compute NCC, there can be problem with size
 * optimize (allocation<-->reallocation)
 * optimize (remove vector.erase() calls)
 *       bring "out" all the parameters to TrackerMedianFlow::Param
 */

class MedianFlowCore{
 public:
     MedianFlowCore(TrackerMedianFlow::Params paramsIn):termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.3){params=paramsIn;}
     Rect2d medianFlowImpl(Mat oldImage,Mat newImage,Rect2d oldBox);
 private:
     Rect2d vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect);
     //FIXME: this can be optimized: current method uses sort->select approach, there are O(n) selection algo for median; besides
          //it makes copy all the time
     template<typename T>
     T getMedian( std::vector<T>& values,int size=-1);
     float dist(Point2f p1,Point2f p2);
     std::string type2str(int type);
     void computeStatistics(std::vector<float>& data,int size=-1);
     void check_FB(const Mat& oldImage,const Mat& newImage,
             const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
     void check_NCC(const Mat& oldImage,const Mat& newImage,
             const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
     inline double l2distance(Point2f p1,Point2f p2);

     TrackerMedianFlow::Params params;
     TermCriteria termcrit;
};

class TrackerMedianFlowModel : public TrackerModel{
 public:
  TrackerMedianFlowModel(TrackerMedianFlow::Params params):medianFlow(params){}
  MedianFlowCore* getMedianFlowCore(){return &medianFlow;}
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  Mat getImage(){return image_;}
  void setImage(const Mat& image){image.copyTo(image_);}
 protected:
  MedianFlowCore medianFlow;
  Rect2d boundingBox_;
  Mat image_;
  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}
};

/*
 * Parameters
 */
TrackerMedianFlow::Params::Params(){
    pointsInGrid=10;
}

void TrackerMedianFlow::Params::read( const cv::FileNode& fn ){
  pointsInGrid=fn["pointsInGrid"];
}

void TrackerMedianFlow::Params::write( cv::FileStorage& fs ) const{
  fs << "pointsInGrid" << pointsInGrid;
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
    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel(params));
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlow::updateImpl( const Mat& image, Rect& boundingBox ){
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();

    Rect2d oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    boundingBox=(((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getMedianFlowCore())->
        medianFlowImpl(oldImage,image,oldBox);
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
Rect2d MedianFlowCore::medianFlowImpl(Mat oldImage,Mat newImage,Rect2d oldBox){
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;

    Mat oldImage_gray,newImage_gray;
    cvtColor( oldImage, oldImage_gray, COLOR_BGR2GRAY );
    cvtColor( newImage, newImage_gray, COLOR_BGR2GRAY );

    //"open ended" grid
    for(int i=0;i<params.pointsInGrid;i++){
        for(int j=0;j<params.pointsInGrid;j++){
                pointsToTrackOld.push_back(
                        Point2f(oldBox.x+((1.0*oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid,
                        oldBox.y+((1.0*oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid));
        }
    }

    std::vector<uchar> status(pointsToTrackOld.size());
    std::vector<float> errors(pointsToTrackOld.size());
    calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(3,3),5,termcrit,0);
    printf("\t%d after LK forward\n",pointsToTrackOld.size());

    std::vector<bool> filter_status;
    check_FB(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,filter_status);
    check_NCC(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,filter_status);

    // filter
    for(int i=0;i<pointsToTrackOld.size();i++){
        if(!filter_status[i]){
            pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
            pointsToTrackNew.erase(pointsToTrackNew.begin()+i);
            filter_status.erase(filter_status.begin()+i);
            i--;
        }
    }
    printf("\t%d after LK backward\n",pointsToTrackOld.size());

    // vote
    CV_Assert(pointsToTrackOld.size()>0);
    Rect2d newBddBox=vote(pointsToTrackOld,pointsToTrackNew,oldBox);

    return newBddBox;
}

Rect2d MedianFlowCore::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect){
    static int iteration=0;//FIXME -- we don't want this static var in final release
    Rect2d newRect;
    Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
    int n=oldPoints.size();
    std::vector<double> buf(std::max(n*(n-1)/2,3),0.0);

    if(oldPoints.size()==1){
        newRect.x=oldRect.x+newPoints[0].x-oldPoints[0].x;
        newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    double xshift=0,yshift=0;
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].x-oldPoints[i].x;  }
    xshift=getMedian(buf,n);
    newCenter.x+=xshift;
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].y-oldPoints[i].y;  }
    yshift=getMedian(buf,n);
    newCenter.y+=yshift;

    if(oldPoints.size()==1){
        newRect.x=newCenter.x-oldRect.width/2.0;
        newRect.y=newCenter.y-oldRect.height/2.0;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    float nd,od;
    for(int i=0,ctr=0;i<n;i++){
        for(int j=0;j<i;j++){
            nd=l2distance(newPoints[i],newPoints[j]);
            od=l2distance(oldPoints[i],oldPoints[j]);
            buf[ctr]=(od==0)?0:(nd/od);
            ctr++;
        }
    }

    double scale=getMedian(buf,n*(n-1)/2);
    printf("iter %d %f %f %f\n",iteration,xshift,yshift,scale);
    newRect.x=newCenter.x-scale*oldRect.width/2.0;
    newRect.y=newCenter.y-scale*oldRect.height/2.0;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    /*if(newRect.x<=0){
        exit(0);
    }*/
    printf("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height);
    printf("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height);

    iteration++;
    return newRect;
}

template<typename T>
T MedianFlowCore::getMedian(std::vector<T>& values,int size){
    if(size==-1){
        size=values.size();
    }
    std::vector<T> copy(values.begin(),values.begin()+size);
    std::sort(copy.begin(),copy.end());
    if(size%2==0){
        return (copy[size/2-1]+copy[size/2])/2.0;
    }else{
        return copy[(size-1)/2];
    }
}

void MedianFlowCore::computeStatistics(std::vector<float>& data,int size){
    int binnum=10;
    if(size==-1){
        size=data.size();
    }
    float mini=*std::min_element(data.begin(),data.begin()+size),maxi=*std::max_element(data.begin(),data.begin()+size);
    std::vector<int> bins(binnum,(int)0);
    for(int i=0;i<size;i++){
        bins[std::min((int)(binnum*(data[i]-mini)/(maxi-mini)),binnum-1)]++;
    }
    for(int i=0;i<binnum;i++){
        printf("[%4f,%4f] -- %4d\n",mini+(maxi-mini)/binnum*i,mini+(maxi-mini)/binnum*(i+1),bins[i]);
    }
}
double MedianFlowCore::l2distance(Point2f p1,Point2f p2){
    double dx=p1.x-p2.x, dy=p1.y-p2.y;
    return sqrt(dx*dx+dy*dy);
}
void MedianFlowCore::check_FB(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status){

    if(status.size()==0){
        status=std::vector<bool>(oldPoints.size(),true);
    }

    std::vector<uchar> LKstatus(oldPoints.size());
    std::vector<float> errors(oldPoints.size());
    std::vector<double> FBerror(oldPoints.size());
    std::vector<Point2f> pointsToTrackReprojection;
    calcOpticalFlowPyrLK(newImage, oldImage,newPoints,pointsToTrackReprojection,LKstatus,errors,Size(3,3),5,termcrit,0);

    for(int i=0;i<oldPoints.size();i++){
        FBerror[i]=l2distance(oldPoints[i],pointsToTrackReprojection[i]);
    }
    double FBerrorMedian=getMedian(FBerror);
    printf("point median=%f\n",FBerrorMedian);
    printf("FBerrorMedian=%f\n",FBerrorMedian);
    for(int i=0;i<oldPoints.size();i++){
        status[i]=(FBerror[i]<FBerrorMedian);
    }
}
void MedianFlowCore::check_NCC(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status){

    std::vector<float> NCC(oldPoints.size(),0.0);
    Size patch(30,30);
    Mat p1,p2;
    Mat_<float> res(1,1);

	for (int i = 0; i < oldPoints.size(); i++) {
		getRectSubPix( oldImage, patch, oldPoints[i],p1);
		getRectSubPix( newImage, patch, newPoints[i],p2);
		matchTemplate( p1,p2, res, CV_TM_CCOEFF );
		NCC[i] = res.at<float>(0,0);
	}
	float median = getMedian(NCC);
	for(int i = 0; i < oldPoints.size(); i++) {
        status[i] = status[i] && (NCC[i]>median);
	}
}
} /* namespace cv */
