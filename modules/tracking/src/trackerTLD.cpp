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

using namespace cv;

/*
 * FIXME(optimize):
 *   do not erase rectangles and generate offline? (maybe permute to put at beginning)
*/
/*ask Kalal: 
 *  init_model:negative_patches  -- all?
 *  posterior: 0/0
 *  sampling: how many base classifiers?
 *  initial model: why 20
 *  scanGrid low overlap
 *  rotated rect in initial model
 */

namespace cv
{

class TLDDetector{
public:
    void generateScanGrid(int cols,int rows, Rect2d initBox,std::vector<Rect2d>& res);
    void patchVariance(const Mat& img,std::vector<Rect2d>& res,double originalVariance);
    static double variance(const Mat& img);
    static inline double overlap(const Rect2d& r1,const Rect2d& r2);
    static void getClosestN(std::vector<Rect2d>& scanGrid,Rect2d bBox,int n,std::vector<Rect2d>& res);
protected:
    //double NCC(const Mat& img,const RotatedRect& r1,const RotatedRect& r2);// TODO -- 3
};

class TrackerTLDModel : public TrackerModel{
 public:
  TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox);
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  double getOriginalVariance(){return originalVariance_;}
 protected:
  void resample(const Mat& img,const RotatedRect& r2,Mat_<double>& samples);
  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}
  Rect2d boundingBox_;
  double originalVariance_;
  std::vector<Mat_<double> > positiveExamples,negativeExamples;
  RNG rng;
};

/*
 * Parameters
 */
TrackerTLD::Params::Params(){
}

void TrackerTLD::Params::read( const cv::FileNode& fn ){
}

void TrackerTLD::Params::write( cv::FileStorage& fs ) const{
}

/*
 * Constructor
 */
TrackerTLD::TrackerTLD( const TrackerTLD::Params &parameters) :
    params( parameters ){
  isInit = false;
}

/*
 * Destructor
 */
TrackerTLD::~TrackerTLD()
{
}

void TrackerTLD::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerTLD::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

bool TrackerTLD::initImpl(const Mat& image, const Rect& boundingBox ){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    model=Ptr<TrackerTLDModel>(new TrackerTLDModel(params,image_gray,boundingBox));
    return true;
}

bool TrackerTLD::updateImpl( const Mat& image, Rect& boundingBox ){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    TLDDetector detector;
    std::vector<Rect2d> scanGrid;
    detector.generateScanGrid(image.cols,image.rows,tldModel->getBoundingBox(),scanGrid);
    printf("%d frames after generateScanGrid() for %dx%d\n",scanGrid.size(),image.rows,image.cols);

    //best overlap around 92%
    /*double m=0;
    for(int i=0;i<scanGrid.size();i++){
        double overlap=TLDDetector::overlap(scanGrid[i],boundingBox);
        if(overlap>m){m=overlap;}
    }
    printf("best overlap: %f\n",m);*/

    detector.patchVariance(image_gray,scanGrid,tldModel->getOriginalVariance());
    printf("%d frames after patchVariance()\n",scanGrid.size());
    printf("\n");
    return true;
}

TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox){
    boundingBox_=boundingBox;
    originalVariance_=TLDDetector::variance(image(boundingBox));
    std::vector<Rect2d> scanGrid,closest(10);
    TLDDetector detector;
    detector.generateScanGrid(image.cols,image.rows,boundingBox,scanGrid);
    detector.getClosestN(scanGrid,boundingBox,10,closest);
    positiveExamples.reserve(200);
    exit(0);
    Point2f center;
    Size2f size;
    for(int i=0;i<closest.size();i++){
        for(int j=0;j<20;j++){
            Mat_<double> standardPatch(15,15);
            center.x=closest[i].x+closest[i].width*(0.5+rng.uniform(-0.01,0.01));
            center.y=closest[i].y+closest[i].height*(0.5+rng.uniform(-0.01,0.01));
            size.width=closest[i].width*rng.uniform((double)0.99,(double)1.01);
            size.height=closest[i].height*rng.uniform((double)0.99,(double)1.01);
            resample(image,RotatedRect(center,size,rng.uniform((double)-10.0,(double)10.0)),standardPatch);
            for(int y=0;y<standardPatch.rows;y++){
                for(int x=0;x<standardPatch.cols;x++){
                    standardPatch(x,y)+=rng.gaussian(5.0);
                }
            }
            positiveExamples.push_back(standardPatch);
        }
    }
    //std::vector<Mat_<double> > positiveExamples,negativeExamples; -- 0 < overlap < 0.2
}

void TLDDetector::getClosestN(std::vector<Rect2d>& scanGrid,Rect2d bBox,int n,std::vector<Rect2d>& res){
    if(n>=scanGrid.size()){
        res.assign(scanGrid.begin(),scanGrid.end());
        return;
    }
    std::vector<double> overlaps(n,0.0);
    res.assign(scanGrid.begin(),scanGrid.begin()+n);
    for(int i=0;i<n;i++){
        overlaps[i]=overlap(res[i],bBox);
    }
    int i, j;
    double otmp;
    Rect2d rtmp;
    for (i = 1; i < n; i++){
        j = i;
        while (j > 0 && overlaps[j - 1] > overlaps[j]) {
            otmp = overlaps[j];overlaps[j] = overlaps[j - 1];overlaps[j - 1] = otmp;
            rtmp = res[j];res[j] = res[j - 1];res[j - 1] = rtmp;
            j--;
        }
    }

    double o=0.0;
    for(int i=n;i<scanGrid.size();i++){
        if((o=overlap(scanGrid[i],bBox))<=overlaps[0]){
            continue;
        }
        int j=0;
        for(j=0;j<n && overlaps[j]<o;j++);
        j--;
        for(int k=0;k<j;overlaps[k]=overlaps[k+1],res[k]=res[k+1],k++);
        overlaps[j]=o;res[j]=scanGrid[i];
    }
}

double TLDDetector::variance(const Mat& img){
    double p=0,p2=0;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            p+=img.at<uchar>(i,j);
            p2+=img.at<uchar>(i,j)*img.at<uchar>(i,j);
        }
    }
    p/=(img.cols*img.rows);
    p2/=(img.cols*img.rows);
    return p2-p*p;
}

void TLDDetector::generateScanGrid(int cols,int rows, Rect2d initBox,std::vector<Rect2d>& res){
    res.clear();
    //scales step: 1.2; hor step: 10% of width; verstep: 10% of height; minsize: 20pix
    for(double h=initBox.height, w=initBox.width;h<=cols && w<=rows;){
        for(double x=0;(x+w)<=cols;x+=(0.1*w)){
            for(double y=0;(y+h)<=rows;y+=(0.1*h)){
                res.push_back(Rect2d(x,y,w,h));
            }
        }
        if(h<=initBox.height){
            h/=1.2; w/=1.2;
            if(h<20 || w<20){
                h=initBox.height*1.2; w=initBox.width*1.2;
                CV_Assert(h>initBox.height || w>initBox.width);
                return;
            }
        }else{
            h*=1.2; w*=1.2;
        }
    }
}

void TLDDetector::patchVariance(const Mat& img,std::vector<Rect2d>& res,double originalVariance){
    for(int i=0;i<res.size();i++){
        if(variance(img(res[i]))<0.5*originalVariance){
            res.erase(res.begin()+i);
            i--;
        }
    }
}

double TLDDetector::overlap(const Rect2d& r1,const Rect2d& r2){
    double a1=r1.area(), a2=r2.area(), a0=(r1&r2).area();
    return a0/(a1+a2-a0);
}

void TrackerTLDModel::resample(const Mat& img,const RotatedRect& r2,Mat_<double>& samples){
    Point2f vertices[4];
    r2.points(vertices);
    float dx1=vertices[1].x-vertices[0].x,
          dy1=vertices[1].y-vertices[0].y,
          dx2=vertices[3].x-vertices[0].x,
          dy2=vertices[3].y-vertices[0].y;
    for(int i=0;i<samples.rows;i++){
        for(int j=0;j<samples.cols;j++){
            float x=vertices[0].x+dx1*j/samples.cols+dx2*i/samples.rows,
                  y=vertices[0].y+dy1*j/samples.cols+dy2*i/samples.rows;
            int ix=cvFloor(x),iy=cvFloor(y);
            float tx=x-ix,ty=y-iy;
            float a = img.at<uchar>(iy,ix) * (1.0 - tx) + img.at<uchar>(iy,ix+1)* tx;
            float b = img.at<uchar>(iy+1,ix)* (1.0 - tx) + img.at<uchar>(iy+1,ix+1) * tx;
            samples(i,j)=a * (1.0 - ty) + b * ty;
        }
    }
    //generate grid
    //interpolation
}

} /* namespace cv */
