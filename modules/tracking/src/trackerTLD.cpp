/*///////////////////////////////////////////////////////////////////////////////////////
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
#include "time.h"
#include <algorithm>
#include <limits.h>
#include "TLD.hpp"

#define HOW_MANY_CLASSIFIERS 20
#define THETA_NN 0.5
#define CORE_THRESHOLD 0.5

using namespace cv;

/*
 * FIXME(optimize):
 * recovery from occlusion
 * TODO
 *   finish learning
 *   TLD no found
 *   test!!!
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


class TrackerTLDModel;

class TLDDetector : public TrackerTLD::Private{
public:
    TLDDetector(const TrackerTLD::Params& params,int rows,int cols,Rect2d initBox);
    ~TLDDetector(){}
    const std::vector<Rect2d>& generateScanGrid()const{return scanGrid;}
    void setModel(Ptr<TrackerModel> model_in){model=model_in;}
    bool detect(const Mat& img,const Mat& imgBlurred,Rect2d& res);
    bool getNextRect(Rect2d& rect_out,bool& isObject_out,bool& shouldBeIntegrated_out,bool reset=false);
protected:
    int patchVariance(const Mat& img,double originalVariance,int size);
    int ensembleClassifier(const Mat& blurredImg,std::vector<TLDEnsembleClassifier>& classifiers,int size);
    std::vector<Rect2d> scanGrid;
    TrackerTLD::Params params_;
    Ptr<TrackerModel> model;
    int scanGridPos;
    std::vector<bool> isObject,shouldBeIntegrated;
};

class Pexpert{
public:
    Pexpert(const Mat& img,Rect2d& resultBox,const TLDDetector* detector,TrackerTLD::Params params):img_(img),resultBox_(resultBox),
        detector_(detector),params_(params){}
    bool operator()(Rect2d box){return false;}
    int additionalExamples(std::vector<Mat_<double> >& examples);
protected:
    RNG rng;
    Mat img_;
    const TLDDetector* detector_;
    Rect2d resultBox_;
    TrackerTLD::Params params_;
};

class Nexpert{
public:
    Nexpert(const Mat& img,Rect2d& resultBox,const TLDDetector* detector,TrackerTLD::Params params):img_(img),resultBox_(resultBox),
        detector_(detector),params_(params){}
    bool operator()(Rect2d box);
    int additionalExamples(std::vector<Mat_<double> >& examples){return 0;}
protected:
    Mat img_;
    const TLDDetector* detector_;
    Rect2d resultBox_;
    TrackerTLD::Params params_;
};

template <class T,class Tparams>
class TrackerProxyImpl : public TrackerProxy{
public:
    TrackerProxyImpl(Tparams params=Tparams()):params_(params){}
    bool init( const Mat& image, const Rect2d& boundingBox ){
        trackerPtr=Ptr<T>(new T(params_));
        return trackerPtr->init(image,boundingBox);
    }
    bool update( const Mat& image,Rect2d& boundingBox){
        return false;//FIXME
        return trackerPtr->update(image,boundingBox);
    }
private:
    Ptr<T> trackerPtr;
    Tparams params_;
    Rect2d boundingBox_;
};

class TrackerTLDModel : public TrackerModel{
 public:
  TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox,TLDDetector* detector);
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  double getOriginalVariance(){return originalVariance_;}
  std::vector<TLDEnsembleClassifier>* getClassifiers(){return &classifiers;}
  double Sr(const Mat_<double> patch);
  double Sc(const Mat_<double> patch);
  void integrate(Mat& img,Mat& imgBlurred,Rect2d box,bool isPositive);
  void integrate(Mat_<double> e,bool isPositive);
 protected:
  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}
  Rect2d boundingBox_;
  double originalVariance_;
  std::vector<Mat_<double> > positiveExamples,negativeExamples;
  RNG rng;
  std::vector<TLDEnsembleClassifier> classifiers;
};


TrackerTLD::Params::Params(){
}

void TrackerTLD::Params::read( const cv::FileNode& fn ){
}

void TrackerTLD::Params::write( cv::FileStorage& fs ) const{
}

TrackerTLD::TrackerTLD( const TrackerTLD::Params &parameters) :
    params( parameters ){
  isInit = false;
  privateInfo.push_back(Ptr<TrackerProxyImpl<TrackerMedianFlow,TrackerMedianFlow::Params> >(
              new TrackerProxyImpl<TrackerMedianFlow,TrackerMedianFlow::Params>()));
}

TrackerTLD::~TrackerTLD(){
}

void TrackerTLD::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerTLD::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

bool TrackerTLD::initImpl(const Mat& image, const Rect2d& boundingBox ){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    TLDDetector* detector=new TLDDetector(params,image.rows,image.cols,boundingBox);
    privateInfo.push_back(Ptr<TLDDetector>(detector));
    privateInfo.push_back(Ptr<WrapperBool>(new WrapperBool(false)));
    privateInfo.push_back(Ptr<WrapperBool>(new WrapperBool(false)));
    model=Ptr<TrackerTLDModel>(new TrackerTLDModel(params,image_gray,boundingBox,detector));
    detector->setModel(model);
    ((TrackerProxy*)static_cast<Private*>(privateInfo[0]))->init(image,boundingBox);
    return true;
}

bool TrackerTLD::updateImpl(const Mat& image, Rect2d& boundingBox){
    Mat image_gray;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    Mat image_blurred;
    GaussianBlur(image_gray,image_blurred,Size(3,3),0.0);
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    TrackerProxy* trackerProxy=(TrackerProxy*)static_cast<Private*>(privateInfo[0]);
    TLDDetector* detector=((TLDDetector*)static_cast<TrackerTLD::Private*>(privateInfo[1]));
    WrapperBool* confidentPtr=((WrapperBool*)static_cast<TrackerTLD::Private*>(privateInfo[2]));
    WrapperBool* failedLastTimePtr=((WrapperBool*)static_cast<TrackerTLD::Private*>(privateInfo[3]));
    Mat_<double> standardPatch(15,15);

    //best overlap around 92%
    /*double m=0;
    for(int i=0;i<scanGrid.size();i++){
        double overlap=TLDDetector::overlap(scanGrid[i],boundingBox);
        if(overlap>m){m=overlap;}
    }
    printf("best overlap: %f\n",m);*/

    Rect2d tmpCandid=boundingBox;
    std::vector<Rect2d> candidates;
    std::vector<double> candidatesRes;
    bool trackerNeedsReInit=false;
    for(int i=0;i<2;i++){
        if(((i==0)&&!(failedLastTimePtr->get())&&trackerProxy->update(image,tmpCandid)) || 
                ((i==1)&&(detector->detect(image_gray,image_blurred,tmpCandid)))){
            candidates.push_back(tmpCandid);
            resample(image_gray,tmpCandid,standardPatch);
            candidatesRes.push_back(tldModel->Sc(standardPatch));
        }else{
            if(i==0){
                trackerNeedsReInit=true;
            }
        }
    }
    printf("candidates.size()=%d\n",candidates.size());
    std::vector<double>::iterator it;
    if((it=std::max_element(candidatesRes.begin(),candidatesRes.end()))==candidatesRes.end()){
        confidentPtr->set(false);
        failedLastTimePtr->set(true);
        return false;
    }else{
        boundingBox=candidates[it-candidatesRes.begin()];
        failedLastTimePtr->set(false);
        if(trackerNeedsReInit || it!=candidatesRes.begin()){
            trackerProxy->init(image,boundingBox);
        }
    }

    if(*it > CORE_THRESHOLD){
        confidentPtr->set(true);
    }

    if(false && confidentPtr->get()){
        Pexpert pExpert(image_gray,boundingBox,detector,params);
        Nexpert nExpert(image_gray,boundingBox,detector,params);
        bool isObject,shouldBeIntegrated,expertResult;
        std::vector<Mat_<double> > examples;
        examples.reserve(100);
        Rect2d rect;
        detector->getNextRect(rect,isObject,shouldBeIntegrated,true);
        do{
            if(isObject){
                expertResult=nExpert(rect);
            }else{
                expertResult=pExpert(rect);
            }
            if(shouldBeIntegrated || (expertResult!=isObject)){
                tldModel->integrate(image_gray,image_gray,rect,expertResult);
            }
        }while(detector->getNextRect(rect,isObject,shouldBeIntegrated));
        pExpert.additionalExamples(examples);
        for(int i=0;i<examples.size();i++){
            tldModel->integrate(examples[i],true);
        }
        examples.clear();
        nExpert.additionalExamples(examples);
        for(int i=0;i<examples.size();i++){
            tldModel->integrate(examples[i],false);
        }
    }

    return true;
}

TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox,TLDDetector* detector){
    boundingBox_=boundingBox;
    originalVariance_=variance(image(boundingBox));
    std::vector<Rect2d> scanGrid=detector->generateScanGrid(),closest(10);

    getClosestN(scanGrid,boundingBox,10,closest);

    Mat image_blurred;
    Mat_<double> blurredPatch(15,15);
    GaussianBlur(image,image_blurred,Size(3,3),0.0);
    for(int i=0;i<HOW_MANY_CLASSIFIERS;i++){
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


TLDDetector::TLDDetector(const TrackerTLD::Params& params,int rows,int cols,Rect2d initBox){
    scanGrid.clear();
    //scales step: 1.2; hor step: 10% of width; verstep: 10% of height; minsize: 20pix
    for(double h=initBox.height, w=initBox.width;h<cols && w<rows;){
        for(double x=0;(x+w)<=(cols-1.0);x+=(0.1*w)){
            for(double y=0;(y+h)<=(rows-1.0);y+=(0.1*h)){
                scanGrid.push_back(Rect2d(x,y,w,h));
            }
        }
        if(h<=initBox.height){
            h/=1.2; w/=1.2;
            if(h<20 || w<20){
                h=initBox.height*1.2; w=initBox.width*1.2;
                CV_Assert(h>initBox.height || w>initBox.width);
            }
        }else{
            h*=1.2; w*=1.2;
        }
    }
    printf("%d rects in scanGrid\n",scanGrid.size());
}

bool TLDDetector::detect(const Mat& img,const Mat& imgBlurred,Rect2d& res){
    int remains=0;
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));

    START_TICK("patch variance")
    remains=patchVariance(img,tldModel->getOriginalVariance(),scanGrid.size());
    END_TICK("patch variance")
    printf("remains %d rects\n",remains);

    START_TICK("ensembleClassifier")
    remains=ensembleClassifier(imgBlurred,*(tldModel->getClassifiers()),remains);
    END_TICK("ensembleClassifier")
    printf("remains %d rects\n",remains);

    Mat_<double> standardPatch(15,15);
    float maxSc=0.0;
    Rect2d maxScRect;
    double tmp=0.0;
    int iSc=-1;
    START_TICK("NCC")
    isObject.resize(remains);
    shouldBeIntegrated.resize(remains);
    for(int i=0;i<remains;i++){
        resample(img,scanGrid[i],standardPatch);
        tmp=tldModel->Sr(standardPatch);
        isObject[i]=(tmp>THETA_NN);
        shouldBeIntegrated[i]=(abs(tmp-THETA_NN)<0.1);
        if(!isObject[i]){
            continue;
        }
        tmp=tldModel->Sc(standardPatch);
        if(tmp>maxSc){
            iSc=i;
            maxSc=tmp;
            maxScRect=scanGrid[i];
        }
    }
    END_TICK("NCC")
    if(iSc==-1){
        return false;
    }
    printf("iSc=%d\n",iSc);

    res=maxScRect;
    return true;
}

bool TLDDetector::getNextRect(Rect2d& rect_out,bool& isObject_out,bool& shouldBeIntegrated_out,bool reset){
    if(reset){
        scanGridPos=0;
    }
    if(scanGridPos>=isObject.size()){
        return false;
    }
    rect_out=scanGrid[scanGridPos];
    isObject_out=isObject[scanGridPos];
    shouldBeIntegrated_out=shouldBeIntegrated[scanGridPos];
    scanGridPos++;
    return true;
}

int TLDDetector::patchVariance(const Mat& img,double originalVariance,int size){
    Mat_<unsigned int> intImgP(img.rows,img.cols),intImgP2(img.rows,img.cols);

    intImgP(0,0)=img.at<uchar>(0,0);
    for(int j=1;j<intImgP.cols;j++){intImgP(0,j)=intImgP(0,j-1)+img.at<uchar>(0,j);}
    for(int i=1;i<intImgP.rows;i++){intImgP(i,0)=intImgP(i-1,0)+img.at<uchar>(i,0);}
    for(int i=1;i<intImgP.rows;i++){for(int j=1;j<intImgP.cols;j++){
            intImgP(i,j)=intImgP(i,j-1)-intImgP(i-1,j-1)+intImgP(i-1,j)+img.at<uchar>(i,j);}}

    unsigned int p;
    p=img.at<uchar>(0,0);intImgP2(0,0)=p*p;
    for(int j=1;j<intImgP2.cols;j++){p=img.at<uchar>(0,j);intImgP2(0,j)=intImgP2(0,j-1)+p*p;}
    for(int i=1;i<intImgP2.rows;i++){p=img.at<uchar>(i,0);intImgP2(i,0)=intImgP2(i-1,0)+p*p;}
    for(int i=1;i<intImgP2.rows;i++){for(int j=1;j<intImgP2.cols;j++){p=img.at<uchar>(i,j);
            intImgP2(i,j)=intImgP2(i,j-1)-intImgP2(i-1,j-1)+intImgP2(i-1,j)+p*p;}}

    int i=0,j=0;
    Rect2d tmp;

    for(;i<size && !(variance(intImgP,intImgP2,img,(scanGrid[i]))<0.5*originalVariance);i++);

    for(j=i+1;j<size;j++){
        if(!(variance(intImgP,intImgP2,img,(scanGrid[j]))<0.5*originalVariance)){
            tmp=scanGrid[i];
            scanGrid[i]=scanGrid[j];
            scanGrid[j]=tmp;
            i++;
        }
    }

    return MIN(size,i+1);
}

int TLDDetector::ensembleClassifier(const Mat& blurredImg,std::vector<TLDEnsembleClassifier>& classifiers,int size){
    Mat_<double> standardPatch(15,15);
    int i=0,j=0;
    Rect2d tmp;

    for(;i<size;i++){
        double p=0.0;
        resample(blurredImg,scanGrid[i],standardPatch);
        for(int k=0;k<classifiers.size();k++){
            p+=classifiers[k].posteriorProbability(standardPatch);
        }
        p/=classifiers.size();

        if(p<=0.5){
            break;
        }
    }

    for(j=i+1;j<size;j++){
        double p=0.0;
        resample(blurredImg,scanGrid[j],standardPatch);
        for(int k=0;k<classifiers.size();k++){
            p+=classifiers[k].posteriorProbability(standardPatch);
        }
        p/=classifiers.size();

        if(!(p<=0.5)){
            tmp=scanGrid[i];
            scanGrid[i]=scanGrid[j];
            scanGrid[j]=tmp;
            i++;
        }
    }

    return MIN(size,i+1);
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

double TrackerTLDModel::Sc(const Mat_<double> patch){
    double splus=0.0;
    for(int i=0;i<((positiveExamples.size()+1)/2);i++){
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

void TrackerTLDModel::integrate(Mat& img,Mat& imgBlurred,Rect2d box,bool isPositive){
    Mat_<double> standardPatch(15,15),blurredPatch(15,15);
    resample(img,box,standardPatch);
    if(isPositive){
        positiveExamples.push_back(standardPatch);
    }else{
        negativeExamples.push_back(standardPatch);
    }

    resample(imgBlurred,box,blurredPatch);
    for(int i=0;i<classifiers.size();i++){
        classifiers[i].integrate(blurredPatch,isPositive);
    }
}
void TrackerTLDModel::integrate(Mat_<double> e,bool isPositive){
    if(isPositive){
        positiveExamples.push_back(e);
    }else{
        negativeExamples.push_back(e);
    }
    for(int i=0;i<classifiers.size();i++){
        classifiers[i].integrate(e,isPositive);
    }
}

int Pexpert::additionalExamples(std::vector<Mat_<double> >& examples){
    examples.clear();
    examples.reserve(100);
    std::vector<Rect2d> closest,scanGrid=detector_->generateScanGrid();
    closest.reserve(10);
    getClosestN(scanGrid,resultBox_,10,closest);

    Point2f center;
    Size2f size;
    for(int i=0;i<closest.size();i++){
        for(int j=0;j<10;j++){
            Mat_<double> standardPatch(15,15);
            center.x=closest[i].x+closest[i].width*(0.5+rng.uniform(-0.01,0.01));
            center.y=closest[i].y+closest[i].height*(0.5+rng.uniform(-0.01,0.01));
            size.width=closest[i].width*rng.uniform((double)0.99,(double)1.01);
            size.height=closest[i].height*rng.uniform((double)0.99,(double)1.01);
            float angle=rng.uniform((double)-5.0,(double)5.0);

            resample(img_,RotatedRect(center,size,angle),standardPatch);
            for(int y=0;y<standardPatch.rows;y++){
                for(int x=0;x<standardPatch.cols;x++){
                    standardPatch(x,y)+=rng.gaussian(5.0);
                }
            }
            examples.push_back(standardPatch);
        }
    }
    return 0;
    //Mat img_;
    //const TLDDetector* detector_;
    //Rect2d resultBox_;
    //TrackerTLD::Params params_;
}

bool Nexpert::operator()(Rect2d box){
    if(overlap(resultBox_,box)<0.2){
        return false;
    }
    return true;
}
/*void fast_ensemble(){
   const uchar* ptr0 = blurred_img.at<uchar>(y,x);
   // grab the grid points
   for(int i = 0; i < 80; i++) buf[i] = ptr0[ofs[i]];
   // compute binary words, look at histograms.
   double sum = 0;
   const uchar* pairs = pairs_tab;
   const int* P = hist;
   const int word_size = 13;
   for(base_idx = 0; base_idx < base_n; base_idx++, pairs += word_size*2, P += (1<<word_size)*2 ){
          int word = (buf[pairs[0]] < buf[pairs[1]]) +
                           (buf[pairs[2]] < buf[pairs[3]])*2 +
                           ...
                           (buf[pairs[24]] < buf[pairs[25]])*4096;
         int p = P[word*2], n = P[word*2+1];
         sum += p + n > 0 ? (double)p/(p + n) : 0.;
   }
   sum /= base_n;
}*/

} /* namespace cv */
