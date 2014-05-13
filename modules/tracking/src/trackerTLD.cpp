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
#include "TLD.hpp"

#define CLIP(x,a,b) MIN(MAX((x),(a)),(b))

using namespace cv;

/*
 * FIXME(optimize):
 *   do not erase rectangles and generate offline? (maybe permute to put at beginning)
*/
/*ask Kalal: 
 * ./bin/example_tracking_tracker TLD ../TrackerChallenge/test.avi 0 5,110,25,130 > out.txt
 *
 *  init_model:negative_patches  -- all?
 *  posterior: 0/0
 *  sampling: how many base classifiers?
 *  initial model: why 20
 *  scanGrid low overlap
 *  rotated rect in initial model
 */

namespace cv
{


static inline double overlap(const Rect2d& r1,const Rect2d& r2);
static void resample(const Mat& img,const RotatedRect& r2,Mat_<double>& samples);
static void resample(const Mat& img,const Rect2d& r2,Mat_<double>& samples);
static void getClosestN(std::vector<Rect2d>& scanGrid,Rect2d bBox,int n,std::vector<Rect2d>& res);
static double variance(const Mat& img);
double NCC(Mat_<double> patch1,Mat_<double> patch2);

class TLDDetector{
public:
    TLDDetector(const Mat& img,const TrackerTLD::Params& params):img_(img){}
    void generateScanGrid(Rect2d initBox,std::vector<Rect2d>& res);
    void patchVariance(std::vector<Rect2d>& res,double originalVariance);
    void ensembleClassifier(const Mat& blurredImg,std::vector<TLDEnsembleClassifier>& classifiers,std::vector<Rect2d>& res);
protected:
    const Mat img_;
};

class TrackerTLDModel : public TrackerModel{
 public:
  TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox);
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  double getOriginalVariance(){return originalVariance_;}
  std::vector<TLDEnsembleClassifier>* getClassifiers(){return &classifiers;}
  double Sr(const Mat_<double> patch);
 protected:
  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}
  Rect2d boundingBox_;
  double originalVariance_;
  std::vector<Mat_<double> > positiveExamples,negativeExamples;
  RNG rng;
  std::vector<TLDEnsembleClassifier> classifiers;
};

//debug functions and variables
Rect2d etalon(14.0,110.0,20.0,20.0);
static void myassert(const Mat& img){
    int count=0;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(img.at<uchar>(i,j)==0){
                count++;
            }
        }
    }
    printf("black: %d out of %d (%f)\n",count,img.rows*img.cols,1.0*count/img.rows/img.cols);
}
void printPatch(const Mat_<double>& standardPatch){
    for(int i=0;i<standardPatch.rows;i++){
        for(int j=0;j<standardPatch.cols;j++){
            printf("%5.2f, ",standardPatch(i,j));
        }
        printf("\n");
    }
}
std::string type2str(const Mat& mat) {
  int type=mat.type();
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
    myassert(image(boundingBox));
    model=Ptr<TrackerTLDModel>(new TrackerTLDModel(params,image_gray,boundingBox));
    return true;
}

bool TrackerTLD::updateImpl(const Mat& image, Rect& boundingBox){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    Mat image_blurred;
    GaussianBlur(image_gray,image_blurred,Size(3,3),0.0);
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    TLDDetector detector(image_gray,params);
    std::vector<Rect2d> scanGrid;

    detector.generateScanGrid(tldModel->getBoundingBox(),scanGrid);
    printf("%d frames after generateScanGrid() for %dx%d\n",scanGrid.size(),image_gray.rows,image_gray.cols);

    //best overlap around 92%
    /*double m=0;
    for(int i=0;i<scanGrid.size();i++){
        double overlap=TLDDetector::overlap(scanGrid[i],boundingBox);
        if(overlap>m){m=overlap;}
    }
    printf("best overlap: %f\n",m);*/

    double o=0.0;
    int omax=0;
    myassert(image_gray(etalon));
    myassert(image_blurred(etalon));

    detector.patchVariance(scanGrid,tldModel->getOriginalVariance());
    printf("%d frames after patchVariance()\n",scanGrid.size());

    detector.ensembleClassifier(image_blurred,*(tldModel->getClassifiers()),scanGrid);
    printf("%d frames after ensembleClassifier()\n",scanGrid.size());

    Mat_<double> standardPatch(15,15);
    float maxSr=0.0;
    int maxIndex=0;
    double tmpSr=0.0;
    for(int i=0;i<scanGrid.size();i++){
        resample(image_gray,scanGrid[i],standardPatch);
        double tmpSr=tldModel->Sr(standardPatch);
        if(tmpSr>maxSr){
            maxSr=tmpSr;
            maxIndex=i;
        }
    }

    boundingBox=Rect(scanGrid[maxIndex].x,scanGrid[maxIndex].y,scanGrid[maxIndex].width,scanGrid[maxIndex].height);
    return true;
}

TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox){
    boundingBox_=boundingBox;
    originalVariance_=variance(image(boundingBox));
    std::vector<Rect2d> scanGrid,closest(10);

    TLDDetector detector(image,params);
    detector.generateScanGrid(boundingBox,scanGrid);
    getClosestN(scanGrid,boundingBox,10,closest);

    Mat image_blurred;
    Mat_<double> blurredPatch(15,15);
    GaussianBlur(image,image_blurred,Size(3,3),0.0);
    for(int i=0;i<200;i++){
        classifiers.push_back(TLDEnsembleClassifier(i+1));
    }

    positiveExamples.reserve(200);
    Point2f center;
    Size2f size;
    for(int i=0;i<closest.size();i++){
        for(int j=0;j<20;j++){
            Mat_<double> standardPatch(15,15);
            center.x=closest[i].x+closest[i].width*(0.5+rng.uniform(-0.01,0.01));
            center.y=closest[i].y+closest[i].height*(0.5+rng.uniform(-0.01,0.01));
            size.width=closest[i].width*rng.uniform((double)0.99,(double)1.01);
            size.height=closest[i].height*rng.uniform((double)0.99,(double)1.01);
            float angle=rng.uniform((double)-10.0,(double)10.0);

            resample(image,RotatedRect(center,size,angle),standardPatch);
            for(int y=0;y<standardPatch.rows;y++){
                for(int x=0;x<standardPatch.cols;x++){
                    standardPatch(x,y)+=rng.gaussian(5.0);
                }
            }
            positiveExamples.push_back(standardPatch);

            resample(image_blurred,RotatedRect(center,size,angle),blurredPatch);
            for(int k=0;k<classifiers.size();k++){
                classifiers[k].integrate(blurredPatch,true);
            }
        }
    }

    negativeExamples.clear();
    const int negMax=200;
    negativeExamples.reserve(negMax);
    std::vector<int> indices;
    indices.reserve(negMax);
    while(negativeExamples.size()<negMax){
        int i=rng.uniform((int)0,(int)scanGrid.size());
        if(std::find(indices.begin(),indices.end(),i)==indices.end() && overlap(boundingBox,scanGrid[i])<0.2){
            Mat_<double> standardPatch(15,15);
            resample(image,scanGrid[i],standardPatch);
            negativeExamples.push_back(standardPatch);

            resample(image_blurred,scanGrid[i],blurredPatch);
            for(int k=0;k<classifiers.size();k++){
                classifiers[k].integrate(blurredPatch,false);
            }
        }
    }
    for(int i=rng.uniform((int)0,(int)negativeExamples.size());negativeExamples.size()>400;i=rng.uniform((int)0,(int)negativeExamples.size())){
        negativeExamples.erase(negativeExamples.begin()+i);
    }
    printf("positive patches: %d\nnegative patches: %d\n",positiveExamples.size(),negativeExamples.size());
}

void getClosestN(std::vector<Rect2d>& scanGrid,Rect2d bBox,int n,std::vector<Rect2d>& res){
    if(n>=scanGrid.size()){
        res.assign(scanGrid.begin(),scanGrid.end());
        return;
    }
    std::vector<double> overlaps(n,0.0);
    res.assign(scanGrid.begin(),scanGrid.begin()+n);
    for(int i=0;i<n;i++){
        overlaps[i]=overlap(res[i],bBox);
    }
    double otmp;
    Rect2d rtmp;
    for (int i = 1; i < n; i++){
        int j = i;
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

double variance(const Mat& img){
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

void TLDDetector::generateScanGrid(Rect2d initBox,std::vector<Rect2d>& res){
    int cols=img_.cols, rows=img_.rows;
    res.clear();
    //scales step: 1.2; hor step: 10% of width; verstep: 10% of height; minsize: 20pix
    for(double h=initBox.height, w=initBox.width;h<cols && w<rows;){
        for(double x=0;(x+w)<cols;x+=(0.1*w)){
            for(double y=0;(y+h)<rows;y+=(0.1*h)){
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

void TLDDetector::patchVariance(std::vector<Rect2d>& res,double originalVariance){
    for(int i=0;i<res.size();i++){
        if(variance(img_(res[i]))<0.5*originalVariance){
            res.erase(res.begin()+i);
            i--;
        }
    }
}
void TLDDetector::ensembleClassifier(const Mat& blurredImg,std::vector<TLDEnsembleClassifier>& classifiers,std::vector<Rect2d>& res){
    Mat_<double> standardPatch(15,15);
    for(int i=0;i<res.size();i++){
        double p=0.0;
        resample(blurredImg,res[i],standardPatch);
        for(int j=0;j<classifiers.size();j++){
            p+=classifiers[j].posteriorProbability(standardPatch);
        }
        p/=classifiers.size();

        if(p<=0.5){
            res.erase(res.begin()+i);
            i--;
        }
    }
}

double TrackerTLDModel::Sr(const Mat_<double> patch){
    double splus=0.0;
    for(int i=0;i<positiveExamples.size();i++){
        splus=MAX(splus,0.5*(NCC(positiveExamples[i],patch)+1.0));
    }
    double sminus=0.0;
    for(int i=0;i<negativeExamples.size();i++){
        sminus=MAX(sminus,0.5*(NCC(negativeExamples[i],patch)+1.0));
    }
    if(splus+sminus==0.0){
        return 0.0;
    }
    return splus/(sminus+splus);
}

double NCC(Mat_<double> patch1,Mat_<double> patch2){
    CV_Assert(patch1.rows=patch2.rows);
    CV_Assert(patch1.cols=patch2.cols);
    return patch1.dot(patch2)/norm(patch1)/norm(patch2);
}

double overlap(const Rect2d& r1,const Rect2d& r2){
    double a1=r1.area(), a2=r2.area(), a0=(r1&r2).area();
    return a0/(a1+a2-a0);
}

void resample(const Mat& img,const RotatedRect& r2,Mat_<double>& samples){
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
            float a=img.at<uchar>(CLIP(iy,0,img.cols-1),CLIP(ix,0,img.rows-1))*(1.0-tx)+
                img.at<uchar>(CLIP(iy,0,img.cols-1),CLIP(ix+1,0,img.rows-1))* tx;
            float b=img.at<uchar>(CLIP(iy+1,0,img.cols-1),CLIP(ix,0,img.rows-1))*(1.0-tx)+
                img.at<uchar>(CLIP(iy+1,0,img.cols-1),CLIP(ix+1,0,img.rows-1))* tx;
            samples(i,j)=(double)a * (1.0 - ty) + b * ty;
        }
    }
}
void resample(const Mat& img,const Rect2d& r2,Mat_<double>& samples){
    Point2f center(r2.x+r2.width/2,r2.y+r2.height/2);
    return resample(img,RotatedRect(center,Size2f(r2.width,r2.height),0.0),samples);
}

} /* namespace cv */
