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

#undef ALEX_DEBUG
#ifdef ALEX_DEBUG
#define dfprintf(x) fprintf x
#define dprintf(x) printf x
#else
#define dfprintf(x)
#define dprintf(x)
#endif

/*
 *  TrackerMedianFlow
 */
/*
 * TODO:
 * add "non-detected" answer in algo --> test it with 2 rects --> frame-by-frame debug in TLD --> test it!!
 * take all parameters out
 *              asessment framework
 *
 *
 * FIXME:
 * when patch is cut from image to compute NCC, there can be problem with size
 * optimize (allocation<-->reallocation)
 * optimize (remove vector.erase() calls)
 *       bring "out" all the parameters to TrackerMedianFlow::Param
 */

class TrackerMedianFlowImpl : public TrackerMedianFlow{
 public:
     TrackerMedianFlowImpl(TrackerMedianFlow::Params paramsIn):termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.3){params=paramsIn;isInit=false;}
     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;
 private:
     bool initImpl( const Mat& image, const Rect2d& boundingBox );
     bool updateImpl( const Mat& image, Rect2d& boundingBox );
     bool medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox);
     Rect2d vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD);
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
  TrackerMedianFlowModel(TrackerMedianFlow::Params /*params*/){}
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  Mat getImage(){return image_;}
  void setImage(const Mat& image){image.copyTo(image_);}
 protected:
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

void TrackerMedianFlowImpl::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerMedianFlowImpl::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

Ptr<TrackerMedianFlow> TrackerMedianFlow::createTracker(const TrackerMedianFlow::Params &parameters){
    return Ptr<TrackerMedianFlowImpl>(new TrackerMedianFlowImpl(parameters));
}

bool TrackerMedianFlowImpl::initImpl( const Mat& image, const Rect2d& boundingBox ){
    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel(params));
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlowImpl::updateImpl( const Mat& image, Rect2d& boundingBox ){
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();

    Rect2d oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    if(!medianFlowImpl(oldImage,image,oldBox)){
        return false;
    }
    boundingBox=oldBox;
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(oldBox);
    return true;
}

std::string TrackerMedianFlowImpl::type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = (uchar)(1 + (type >> CV_CN_SHIFT));

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
bool TrackerMedianFlowImpl::medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox){
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;

    Mat oldImage_gray,newImage_gray;
    if (oldImage.channels() != 1)
        cvtColor( oldImage, oldImage_gray, COLOR_BGR2GRAY );
    else
        oldImage.copyTo(oldImage_gray);

    if (newImage.channels() != 1)
        cvtColor( newImage, newImage_gray, COLOR_BGR2GRAY );
    else
        newImage.copyTo(newImage_gray);

    //"open ended" grid
    for(int i=0;i<params.pointsInGrid;i++){
        for(int j=0;j<params.pointsInGrid;j++){
                pointsToTrackOld.push_back(
                        Point2f((float)(oldBox.x+((1.0*oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid),
                        (float)(oldBox.y+((1.0*oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid)));
        }
    }

    std::vector<uchar> status(pointsToTrackOld.size());
    std::vector<float> errors(pointsToTrackOld.size());
    calcOpticalFlowPyrLK(oldImage_gray, newImage_gray,pointsToTrackOld,pointsToTrackNew,status,errors,Size(3,3),5,termcrit,0);
    dprintf(("\t%d after LK forward\n",(int)pointsToTrackOld.size()));

    std::vector<Point2f> di;
    for(int i=0;i<(int)pointsToTrackOld.size();i++){
        if(status[i]==1){
            di.push_back(pointsToTrackNew[i]-pointsToTrackOld[i]);
        }
    }

    std::vector<bool> filter_status;
    check_FB(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,filter_status);
    check_NCC(oldImage_gray,newImage_gray,pointsToTrackOld,pointsToTrackNew,filter_status);

    // filter
    for(int i=0;i<(int)pointsToTrackOld.size();i++){
        if(!filter_status[i]){
            pointsToTrackOld.erase(pointsToTrackOld.begin()+i);
            pointsToTrackNew.erase(pointsToTrackNew.begin()+i);
            filter_status.erase(filter_status.begin()+i);
            i--;
        }
    }
    dprintf(("\t%d after LK backward\n",(int)pointsToTrackOld.size()));

    if(pointsToTrackOld.size()==0 || di.size()==0){
        return false;
    }
    Point2f mDisplacement;
    oldBox=vote(pointsToTrackOld,pointsToTrackNew,oldBox,mDisplacement);

    std::vector<double> displacements;
    for(int i=0;i<(int)di.size();i++){
        di[i]-=mDisplacement;
        displacements.push_back(sqrt(di[i].ddot(di[i])));
    }
    if(getMedian(displacements,(int)displacements.size())>10){
        return false;
    }

    return true;
}

Rect2d TrackerMedianFlowImpl::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD){
    static int iteration=0;//FIXME -- we don't want this static var in final release
    Rect2d newRect;
    Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
    int n=(int)oldPoints.size();
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
    mD=Point2f((float)xshift,(float)yshift);

    if(oldPoints.size()==1){
        newRect.x=newCenter.x-oldRect.width/2.0;
        newRect.y=newCenter.y-oldRect.height/2.0;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        return newRect;
    }

    double nd,od;
    for(int i=0,ctr=0;i<n;i++){
        for(int j=0;j<i;j++){
            nd=l2distance(newPoints[i],newPoints[j]);
            od=l2distance(oldPoints[i],oldPoints[j]);
            buf[ctr]=(od==0.0)?0.0:(nd/od);
            ctr++;
        }
    }

    double scale=getMedian(buf,n*(n-1)/2);
    dprintf(("iter %d %f %f %f\n",iteration,xshift,yshift,scale));
    newRect.x=newCenter.x-scale*oldRect.width/2.0;
    newRect.y=newCenter.y-scale*oldRect.height/2.0;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    /*if(newRect.x<=0){
        exit(0);
    }*/
    dprintf(("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height));
    dprintf(("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height));

    iteration++;
    return newRect;
}

template<typename T>
T TrackerMedianFlowImpl::getMedian(std::vector<T>& values,int size){
    if(size==-1){
        size=(int)values.size();
    }
    std::vector<T> copy(values.begin(),values.begin()+size);
    std::sort(copy.begin(),copy.end());
    if(size%2==0){
        return (copy[size/2-1]+copy[size/2])/((T)2.0);
    }else{
        return copy[(size-1)/2];
    }
}

void TrackerMedianFlowImpl::computeStatistics(std::vector<float>& data,int size){
    int binnum=10;
    if(size==-1){
        size=(int)data.size();
    }
    float mini=*std::min_element(data.begin(),data.begin()+size),maxi=*std::max_element(data.begin(),data.begin()+size);
    std::vector<int> bins(binnum,(int)0);
    for(int i=0;i<size;i++){
        bins[std::min((int)(binnum*(data[i]-mini)/(maxi-mini)),binnum-1)]++;
    }
    for(int i=0;i<binnum;i++){
        dprintf(("[%4f,%4f] -- %4d\n",mini+(maxi-mini)/binnum*i,mini+(maxi-mini)/binnum*(i+1),bins[i]));
    }
}
double TrackerMedianFlowImpl::l2distance(Point2f p1,Point2f p2){
    double dx=p1.x-p2.x, dy=p1.y-p2.y;
    return sqrt(dx*dx+dy*dy);
}
void TrackerMedianFlowImpl::check_FB(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status){

    if(status.size()==0){
        status=std::vector<bool>(oldPoints.size(),true);
    }

    std::vector<uchar> LKstatus(oldPoints.size());
    std::vector<float> errors(oldPoints.size());
    std::vector<double> FBerror(oldPoints.size());
    std::vector<Point2f> pointsToTrackReprojection;
    calcOpticalFlowPyrLK(newImage, oldImage,newPoints,pointsToTrackReprojection,LKstatus,errors,Size(3,3),5,termcrit,0);

    for(int i=0;i<(int)oldPoints.size();i++){
        FBerror[i]=l2distance(oldPoints[i],pointsToTrackReprojection[i]);
    }
    double FBerrorMedian=getMedian(FBerror);
    dprintf(("point median=%f\n",FBerrorMedian));
    dprintf(("FBerrorMedian=%f\n",FBerrorMedian));
    for(int i=0;i<(int)oldPoints.size();i++){
        status[i]=(FBerror[i]<FBerrorMedian);
    }
}
void TrackerMedianFlowImpl::check_NCC(const Mat& oldImage,const Mat& newImage,
        const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status){

    std::vector<float> NCC(oldPoints.size(),0.0);
    Size patch(30,30);
    Mat p1,p2;

	for (int i = 0; i < (int)oldPoints.size(); i++) {
		getRectSubPix( oldImage, patch, oldPoints[i],p1);
		getRectSubPix( newImage, patch, newPoints[i],p2);

        const int N=900;
        double s1=sum(p1)(0),s2=sum(p2)(0);
        double n1=norm(p1),n2=norm(p2);
        double prod=p1.dot(p2);
        double sq1=sqrt(n1*n1-s1*s1/N),sq2=sqrt(n2*n2-s2*s2/N);
        double ares=(sq2==0)?sq1/abs(sq1):(prod-s1*s2/N)/sq1/sq2;

		NCC[i] = (float)ares;
	}
	float median = getMedian(NCC);
	for(int i = 0; i < (int)oldPoints.size(); i++) {
        status[i] = status[i] && (NCC[i]>median);
	}
}
} /* namespace cv */
