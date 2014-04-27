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
//debug headers start
#include "opencv2/highgui.hpp"
//debug headers end

#define SAME(a,b) (norm((a)-(b))==0)

namespace cv
{

/*
 *  TrackerMedianFlow
 */
/*
 * TODO:
 * real videos
 *
 * final version:
 *      remove opencv_highgui from CMakeLists.txt
 *      remove all debug headers in this file (see above)
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
     static float getMedian( std::vector<float>& values,int size=-1);
     static float dist(Point2f p1,Point2f p2);
     static std::string type2str(int type);
     static void computeStatistics(std::vector<float>& data,int size=-1);
     static void displayPoint(Mat& image, Point2f pt,String title);
     inline static float l2distance(Point2f p1,Point2f p2);
};

/*
 * Parameters
 */
TrackerMedianFlow::Params::Params(){
    pointsInGrid=20;
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
    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel());
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlow::updateImpl( const Mat& image, Rect& boundingBox ){
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();

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
    float t1,t2;
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;

    Mat oldImage_gray,newImage_gray;
    cvtColor( oldImage, oldImage_gray, COLOR_BGR2GRAY );
    cvtColor( newImage, newImage_gray, COLOR_BGR2GRAY );

    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

    if(true){
        for(int i=0;i<params.pointsInGrid;i++){
            for(int j=0;j<params.pointsInGrid;j++){
                    pointsToTrackOld.push_back(Point2f(oldBox.x+(1.0*oldBox.width/params.pointsInGrid)*i,
                                oldBox.y+(1.0*oldBox.height/params.pointsInGrid)*j));
            }
        }

        std::vector<uchar> status(pointsToTrackOld.size());
        std::vector<float> errors(pointsToTrackOld.size());
        calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(3,3),3,termcrit,0,0.001);
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
                FBerror[i]=l2distance(pointsToTrackOld[i],pointsToTrackReprojection[i]);
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
        printf("\t%d after LK backward\n",pointsToTrackOld.size());
    }else{
        Size subPixWinSize(10,10),winSize(31,31);
        goodFeaturesToTrack(oldImage_gray, pointsToTrackOld, 500, 0.01, 10, Mat(), 3, 0, 0.04);
        cornerSubPix(oldImage_gray, pointsToTrackOld, subPixWinSize, Size(-1,-1), termcrit);
        printf("\t%d feature points\n",pointsToTrackOld.size());

        for(int i=0;i<pointsToTrackOld.size();i++){
            if(pointsToTrackOld[i].x<oldBox.x || pointsToTrackOld[i].x>(oldBox.x+oldBox.width)||
                    pointsToTrackOld[i].y<oldBox.y || pointsToTrackOld[i].y>(oldBox.y+oldBox.height)){
                pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
                i--;
            }
        }
        printf("\t%d after filtering \n",pointsToTrackOld.size());
        if(pointsToTrackOld.size()<4){
            //exit(0);
        }

        std::vector<uchar> status(pointsToTrackOld.size());
        std::vector<float> errors(pointsToTrackOld.size());
        calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(3,3),3,termcrit,0,0.001);
            for(int i=0;i<pointsToTrackOld.size();i++){
                printf("(%f,%f) --> (%f,%f)\n",
                    pointsToTrackOld[i].x,pointsToTrackOld[i].y,pointsToTrackNew[i].x,pointsToTrackNew[i].y);
            }
            printf("\n");
    }

    //FIXME: debug block
    if(!true){
        displayPoint(oldImage_gray,pointsToTrackOld[pointsToTrackNew.size()-1],"make me sway");
        displayPoint(newImage_gray,pointsToTrackNew[pointsToTrackNew.size()-1],"sway me more");
        waitKey(0);
        //exit(0);
    }

    // vote
    CV_Assert(pointsToTrackOld.size()>0);
    Rect newBddBox=vote(pointsToTrackOld,pointsToTrackNew,oldBox);

    if(!true){
        Mat imago;
        newImage_gray.copyTo(imago);
        rectangle( imago, newBddBox, 150, 2, 1 );
        imshow("make me sway",imago);
        waitKey(0);
    }

    return newBddBox;
}
Rect MedianFlowCore::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect& oldRect){
    Rect newRect;
    Point newCenter(oldRect.x+oldRect.width/2,oldRect.y+oldRect.height/2);
    int n=oldPoints.size();
    std::vector<float> buf(n*(n-1));

        printf("line %d with %d\n",__LINE__,n);fflush(stdout);
    if(!true){//FIXME: debug block
        printf("line %d with %d\n",__LINE__,n);fflush(stdout);
        for(int i=0;i<n;i++){
            buf[i]=newPoints[i].x;
        }
        newRect.x=*std::min_element(buf.begin(),buf.begin()+n);
        newRect.width=(*std::max_element(buf.begin(),buf.begin()+n))-newRect.x;
        for(int i=0;i<n;i++){
            buf[i]=newPoints[i].y;
        }
        newRect.y=*std::min_element(buf.begin(),buf.begin()+n);
        newRect.height=(*std::max_element(buf.begin(),buf.begin()+n))-newRect.y;
        return newRect;
    }

    if(oldPoints.size()==1){
        newRect.x=oldRect.x+newPoints[0].x-oldPoints[0].x;
        newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    //FIXME: debug block
    printf("X SHIFT\n");
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].x-oldPoints[i].x;  }
    computeStatistics(buf,n);
    printf("Y SHIFT\n");
    for(int i=0;i<n;i++){  buf[i]=newPoints[i].y-oldPoints[i].y;  }
    computeStatistics(buf,n);

    for(int i=0;i<n;i++){  buf[i]=newPoints[i].x-oldPoints[i].x;  }
    newCenter.x+=getMedian(buf,n);
    printf("shift_x=%f\n",getMedian(buf,n));
    if(false && getMedian(buf,n)<15){
        printf("STOP!");
        exit(0);
    }
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
            nd=l2distance(newPoints[i],newPoints[j]);
            od=l2distance(oldPoints[i],oldPoints[j]);
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
    //exit(0);
    return newRect;
}
float MedianFlowCore::getMedian(std::vector<float>& values,int size){
    if(size==-1){
        size=values.size();
    }
    std::sort(values.begin(),values.begin()+size);
    return values[size/2];
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
void MedianFlowCore::displayPoint(Mat& image, Point2f pt,String title){
    static int i=0;
    printf("point to draw: (%f,%f)\n",pt.x,pt.y);
    const int dim=10;
    CV_Assert(dim%2==0);
    Point cutPoint(pt.x-dim/2,pt.y-dim/2);
    Rect cutFrame;
    cutFrame.x=cutPoint.x; cutFrame.y=cutPoint.y;
    pt.x-=cutPoint.x; pt.y-=cutPoint.y;
    cutFrame.width=cutFrame.height=dim;

    Mat res;
    const int scale=30;
    resize(image(cutFrame),res,Size(dim*scale,dim*scale));
    pt.x*=scale; pt.y*=scale;
    circle(res,pt,3,std::max(0,200-(i++)),-1);
    
    imshow(title,res);
}
float MedianFlowCore::l2distance(Point2f p1,Point2f p2){
    float dx=p1.x-p2.x, dy=p1.y-p2.y;
    return sqrt(dx*dx+dy*dy);
}
} /* namespace cv */
