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
#include "opencv2/highgui.hpp"

#define THETA_NN 0.5
#define CORE_THRESHOLD 0.5
#define NEG_EXAMPLES_IN_INIT_MODEL 300
#define MAX_EXAMPLES_IN_MODEL 500
#define MEASURES_PER_CLASSIFIER 13
#define BLUR_AS_VADIM
#undef CLOSED_LOOP
static const cv::Size GaussBlurKernelSize(3,3);

using namespace cv;
using namespace tld;

/*
 * FIXME(optimize):
 *      no median
 *      direct formula in resamples
 * FIXME(issues)
 *      THETA_NN 0.5<->0.6 dramatic change vs video 6 !!
 * TODO:
 *      schoolPC: codec, libopencv-dev
 *      fix pushbot ->pick commits -> compare_branches->all in 1->resubmit
 *      ||video(0.5<->0.6) --> debug if box size is less than 20 --> (remove ensemble self-loop) --> (try inter_area for resize)
 *      perfect PN
 *
 *      vadim:
 *
 *      blurred in TrackerTLDModel()
 *      private members: pull_approval --> update master --> merge --> redo --> report
 *
 *      warpAffine -- ?
*/

/* design decisions:
 * blur --> resize (vs. resize-->blur) in detect(), ensembleClassifier stage
 * no random gauss noise, when making examples for ensembleClassifier
 */

namespace cv
{

class TLDDetector;
class MyMouseCallbackDEBUG{
public:
    MyMouseCallbackDEBUG( Mat& img, Mat& imgBlurred,TLDDetector* detector):img_(img),imgBlurred_(imgBlurred),detector_(detector){}
    static void onMouse( int event, int x, int y, int, void* obj){
        ((MyMouseCallbackDEBUG*)obj)->onMouse(event,x,y);
    }
    MyMouseCallbackDEBUG& operator= (const MyMouseCallbackDEBUG& /*other*/){return *this;}
private:
    void onMouse( int event, int x, int y);
    Mat& img_,imgBlurred_;
    TLDDetector* detector_;
};

class Data : public TrackerTLD::Private{
public:
    Data(Rect2d initBox);
    Size getMinSize(){return minSize;}
    double getScale(){return scale;}
    bool confident;
    bool failedLastTime;
    int frameNum;
    void printme(FILE*  port=stdout);
private:
    double scale;
    Size minSize;
};

class TrackerTLDModel;

class TLDDetector : public TrackerTLD::Private{
public:
    TLDDetector(const TrackerTLD::Params& params,Ptr<TrackerModel>model_in):model(model_in),params_(params){}
    ~TLDDetector(){}
    static void generateScanGrid(int rows,int cols,Size initBox,std::vector<Rect2d>& res,bool withScaling=false);
    bool detect(const Mat& img,const Mat& imgBlurred,Rect2d& res,std::vector<Rect2d>& rect,std::vector<bool>& isObject,
            std::vector<bool>& shouldBeIntegrated);
protected:
    friend class MyMouseCallbackDEBUG;
    Ptr<TrackerModel> model;
    inline bool patchVariance(Mat_<double>& intImgP,Mat_<double>& intImgP2,double originalVariance,Point pt,Size size);
    bool ensembleClassifier(const uchar* data,int rowstep){return ensembleClassifierNum(data,rowstep)>0.5;}
    double ensembleClassifierNum(const uchar* data,int rowstep);
    TrackerTLD::Params params_;
};

class Pexpert{
public:
    Pexpert(const Mat& img,const Mat& imgBlurred,Rect2d& resultBox,const TLDDetector* detector,TrackerTLD::Params params,Size initSize):
        img_(img),imgBlurred_(imgBlurred),resultBox_(resultBox),detector_(detector),params_(params),initSize_(initSize){}
    bool operator()(Rect2d /*box*/){return false;}
    int additionalExamples(std::vector<Mat_<uchar> >& examplesForModel,std::vector<Mat_<uchar> >& examplesForEnsemble);
protected:
    Mat img_,imgBlurred_;
    Rect2d resultBox_;
    const TLDDetector* detector_;
    TrackerTLD::Params params_;
    RNG rng;
    Size initSize_;
};

class Nexpert{
public:
    Nexpert(const Mat& img,Rect2d& resultBox,const TLDDetector* detector,TrackerTLD::Params params):img_(img),resultBox_(resultBox),
        detector_(detector),params_(params){}
    bool operator()(Rect2d box);
    int additionalExamples(std::vector<Mat_<uchar> >& examplesForModel,std::vector<Mat_<uchar> >& examplesForEnsemble){
        examplesForModel.clear();examplesForEnsemble.clear();return 0;}
protected:
    Mat img_;
    Rect2d resultBox_;
    const TLDDetector* detector_;
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
        return trackerPtr->update(image,boundingBox);
    }
private:
    Ptr<T> trackerPtr;
    Tparams params_;
    Rect2d boundingBox_;
};

class TrackerTLDModel : public TrackerModel{
 public:
  TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox,Size minSize);
  Rect2d getBoundingBox(){return boundingBox_;}
  void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
  double getOriginalVariance(){return originalVariance_;}
  std::vector<TLDEnsembleClassifier>* getClassifiers(){return &classifiers;}
  double Sr(const Mat_<uchar> patch);
  double Sc(const Mat_<uchar> patch);
  void integrateRelabeled(Mat& img,Mat& imgBlurred,const std::vector<Rect2d>& box,const std::vector<bool>& isPositive,
          const std::vector<bool>& alsoIntoModel);
  void integrateAdditional(const std::vector<Mat_<uchar> >& eForModel,const std::vector<Mat_<uchar> >& eForEnsemble,bool isPositive);
  Size getMinSize(){return minSize_;}
  void printme(FILE*  port=stdout);
 protected:
  Size minSize_;
  unsigned int timeStampPositiveNext,timeStampNegativeNext;
  TrackerTLD::Params params_;
  void pushIntoModel(const Mat_<uchar>& example,bool positive);
  void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
  void modelUpdateImpl(){}
  Rect2d boundingBox_;
  double originalVariance_;
  std::vector<Mat_<uchar> > positiveExamples,negativeExamples;
  std::vector<unsigned int> timeStampsPositive,timeStampsNegative;
  RNG rng;
  std::vector<TLDEnsembleClassifier> classifiers;
};

class TrackerTLDImpl : public TrackerTLD
{
 public:
  TrackerTLDImpl( const TrackerTLD::Params &parameters = TrackerTLD::Params() );
  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:

  bool initImpl( const Mat& image, const Rect2d& boundingBox );
  bool updateImpl( const Mat& image, Rect2d& boundingBox );

  TrackerTLD::Params params;
  Ptr<Data> data;
  Ptr<TrackerProxy> tracker;
  Ptr<TLDDetector> detector;
};


TrackerTLD::Params::Params(){
}

void TrackerTLD::Params::read( const cv::FileNode& /*fn*/ ){
}

void TrackerTLD::Params::write( cv::FileStorage& /*fs*/ ) const{
}

Ptr<TrackerTLD> TrackerTLD::createTracker(const TrackerTLD::Params &parameters){
    return Ptr<TrackerTLDImpl>(new TrackerTLDImpl(parameters));
}

TrackerTLDImpl::TrackerTLDImpl( const TrackerTLD::Params &parameters) :
    params( parameters ){
  isInit = false;
  tracker=Ptr<TrackerProxyImpl<TrackerMedianFlow,TrackerMedianFlow::Params> >(
              new TrackerProxyImpl<TrackerMedianFlow,TrackerMedianFlow::Params>());
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
    tracker->init(image,boundingBox);
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    data=new Data(boundingBox);
    double scale=data->getScale();
    Rect2d myBoundingBox=boundingBox;
    if(scale>1.0){
        Mat image_proxy;
        resize(image_gray,image_proxy,Size(cvRound(image.cols*scale),cvRound(image.rows*scale)));
        image_proxy.copyTo(image_gray);
        myBoundingBox.x*=scale;
        myBoundingBox.y*=scale;
        myBoundingBox.width*=scale;
        myBoundingBox.height*=scale;
    }
    model=Ptr<TrackerTLDModel>(new TrackerTLDModel(params,image_gray,myBoundingBox,data->getMinSize()));
    detector=new TLDDetector(params,model);
    data->confident=false;
    data->failedLastTime=false;

#if !1
        dprintf(("here I am\n"));
        Mat image_blurred;
        GaussianBlur(image_gray,image_blurred,GaussBlurKernelSize,0.0);
        MyMouseCallbackDEBUG* callback=new MyMouseCallbackDEBUG(image_gray,image_blurred,detector);
        imshow("picker",image_gray);
        setMouseCallback( "picker", MyMouseCallbackDEBUG::onMouse, (void*)callback);
        waitKey();
#endif
    return true;
}

bool TrackerTLD::updateImpl(const Mat& image, Rect2d& boundingBox){
    Mat image_gray,image_blurred,imageForDetector;
    cvtColor( image, image_gray, COLOR_BGR2GRAY );
    Data* data=((Data*)static_cast<TrackerTLD::Private*>(privateInfo[2]));
    double scale=data->getScale();
    if(scale>1.0){
        resize(image_gray,imageForDetector,Size(cvRound(image.cols*scale),cvRound(image.rows*scale)));
    }else{
        imageForDetector=image_gray;
    }
    GaussianBlur(imageForDetector,image_blurred,GaussBlurKernelSize,0.0);
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    TrackerProxy* trackerProxy=(TrackerProxy*)static_cast<Private*>(privateInfo[0]);
    TLDDetector* detector=((TLDDetector*)static_cast<TrackerTLD::Private*>(privateInfo[1]));
    data->frameNum++;
    Mat_<uchar> standardPatch(15,15);
    std::vector<Rect2d> detectorResults;
    std::vector<bool> isObject,shouldBeIntegrated;
    //best overlap around 92%

    Rect2d tmpCandid=boundingBox;
    std::vector<Rect2d> candidates;
    std::vector<double> candidatesRes;
    bool trackerNeedsReInit=false;
    for(int i=0;i<2;i++){
        if(((i==0)&&!(data->failedLastTime)&&trackerProxy->update(image,tmpCandid)) || 
                ((i==1)&&(detector->detect(imageForDetector,image_blurred,tmpCandid,detectorResults,isObject,shouldBeIntegrated)))){
            candidates.push_back(tmpCandid);
            if(i==0){
                resample(image_gray,tmpCandid,standardPatch);
            }else{
                resample(imageForDetector,tmpCandid,standardPatch);
            }
            candidatesRes.push_back(tldModel->Sc(standardPatch));
        }else{
            if(i==0){
                trackerNeedsReInit=true;
            }
        }
    }

    std::vector<double>::iterator it=std::max_element(candidatesRes.begin(),candidatesRes.end());

    dfprintf((stdout,"scale=%f\n",log(1.0*boundingBox.width/(data->getMinSize()).width)/log(1.2)));
    for(int i=0;i<(int)candidatesRes.size();i++){
        dprintf(("\tcandidatesRes[%d]=%f\n",i,candidatesRes[i]));
    }
    data->printme();
    tldModel->printme(stdout);
#if !1
    if(data->frameNum==82){
        dprintf(("here I am\n"));
        MyMouseCallbackDEBUG* callback=new MyMouseCallbackDEBUG(imageForDetector,image_blurred,detector);
        imshow("picker",imageForDetector);
        setMouseCallback( "picker", MyMouseCallbackDEBUG::onMouse, (void*)callback);
        waitKey();
    }
#endif

    if(it==candidatesRes.end()){
        data->confident=false;
        data->failedLastTime=true;
        return false;
    }else{
        boundingBox=candidates[it-candidatesRes.begin()];
        data->failedLastTime=false;
        if(trackerNeedsReInit || it!=candidatesRes.begin()){
            trackerProxy->init(image,boundingBox);
        }
    }

    if(!false && it!=candidatesRes.end()){
        resample(imageForDetector,candidates[it-candidatesRes.begin()],standardPatch);
        dfprintf((stderr,"%d %f %f\n",data->frameNum,tldModel->Sc(standardPatch),tldModel->Sr(standardPatch)));
        if(candidatesRes.size()==2 &&  it==(candidatesRes.begin()+1))
            dfprintf((stderr,"detector WON\n"));
    }else{
        dfprintf((stderr,"%d x x\n",data->frameNum));
    }

    if(*it > CORE_THRESHOLD){
        data->confident=true;
    }

    if(data->confident){
        Pexpert pExpert(imageForDetector,image_blurred,boundingBox,detector,params,data->getMinSize());
        Nexpert nExpert(imageForDetector,boundingBox,detector,params);
        bool expertResult;
        std::vector<Mat_<uchar> > examplesForModel,examplesForEnsemble;
        examplesForModel.reserve(100);examplesForEnsemble.reserve(100);
        int negRelabeled=0;
        for(int i=0;i<(int)detectorResults.size();i++){
            if(isObject[i]){
                expertResult=nExpert(detectorResults[i]);
                if(expertResult!=isObject[i]){negRelabeled++;}
            }else{
                expertResult=pExpert(detectorResults[i]);
            }

            shouldBeIntegrated[i]=shouldBeIntegrated[i] || (isObject[i]!=expertResult);
            isObject[i]=expertResult;
        }
        tldModel->integrateRelabeled(imageForDetector,image_blurred,detectorResults,isObject,shouldBeIntegrated);
        dprintf(("%d relabeled by nExpert\n",negRelabeled));
        pExpert.additionalExamples(examplesForModel,examplesForEnsemble);
        tldModel->integrateAdditional(examplesForModel,examplesForEnsemble,true);
        examplesForModel.clear();examplesForEnsemble.clear();
        nExpert.additionalExamples(examplesForModel,examplesForEnsemble);
        tldModel->integrateAdditional(examplesForModel,examplesForEnsemble,false);
    }else{
#ifdef CLOSED_LOOP
        tldModel->integrateRelabeled(imageForDetector,image_blurred,detectorResults,isObject,shouldBeIntegrated);
#endif
    }

    return true;
}

TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params,const Mat& image, const Rect2d& boundingBox,Size minSize):minSize_(minSize),
timeStampPositiveNext(0),timeStampNegativeNext(0),params_(params){
    boundingBox_=boundingBox;
    originalVariance_=variance(image(boundingBox));
    std::vector<Rect2d> closest(10),scanGrid;
    Mat scaledImg,blurredImg,image_blurred;

    double scale=scaleAndBlur(image,cvRound(log(1.0*boundingBox.width/(minSize.width))/log(1.2)),scaledImg,blurredImg,GaussBlurKernelSize);
    GaussianBlur(image,image_blurred,GaussBlurKernelSize,0.0);
    TLDDetector::generateScanGrid(image.rows,image.cols,minSize,scanGrid);
    getClosestN(scanGrid,Rect2d(boundingBox.x/scale,boundingBox.y/scale,boundingBox.width/scale,boundingBox.height/scale),10,closest);

    Mat_<uchar> blurredPatch(minSize);
    for(int i=0,howMany=TLDEnsembleClassifier::getMaxOrdinal();i<howMany;i++){
        classifiers.push_back(TLDEnsembleClassifier(i+1,minSize,MEASURES_PER_CLASSIFIER));
    }

    positiveExamples.reserve(200);
    Point2f center;
    Size2f size;
    for(int i=0;i<(int)closest.size();i++){
        for(int j=0;j<20;j++){
            Mat_<uchar> standardPatch(15,15);
            center.x=(float)(closest[i].x+closest[i].width*(0.5+rng.uniform(-0.01,0.01)));
            center.y=(float)(closest[i].y+closest[i].height*(0.5+rng.uniform(-0.01,0.01)));
            size.width=(float)(closest[i].width*rng.uniform((double)0.99,(double)1.01));
            size.height=(float)(closest[i].height*rng.uniform((double)0.99,(double)1.01));
            float angle=(float)rng.uniform(-10.0,10.0);

            resample(scaledImg,RotatedRect(center,size,angle),standardPatch);
            
            for(int y=0;y<standardPatch.rows;y++){
                for(int x=0;x<standardPatch.cols;x++){
                    standardPatch(x,y)+=(uchar)rng.gaussian(5.0);
                }
            }

#ifdef BLUR_AS_VADIM
            GaussianBlur(standardPatch,blurredPatch,GaussBlurKernelSize,0.0);
#else
            resample(blurredImg,RotatedRect(center,size,angle),blurredPatch);
#endif
            pushIntoModel(standardPatch,true);
            for(int k=0;k<(int)classifiers.size();k++){
                classifiers[k].integrate(blurredPatch,true);
            }
        }
    }

    TLDDetector::generateScanGrid(image.rows,image.cols,minSize,scanGrid,true);
    negativeExamples.clear();
    negativeExamples.reserve(NEG_EXAMPLES_IN_INIT_MODEL);
    std::vector<int> indices;
    indices.reserve(NEG_EXAMPLES_IN_INIT_MODEL);
    while(negativeExamples.size()<NEG_EXAMPLES_IN_INIT_MODEL){
        int i=rng.uniform((int)0,(int)scanGrid.size());
        if(std::find(indices.begin(),indices.end(),i)==indices.end() && overlap(boundingBox,scanGrid[i])<0.2){
            Mat_<uchar> standardPatch(15,15);
            resample(image,scanGrid[i],standardPatch);
            pushIntoModel(standardPatch,false);

            resample(image_blurred,scanGrid[i],blurredPatch);
            for(int k=0;k<(int)classifiers.size();k++){
                classifiers[k].integrate(blurredPatch,false);
            }
        }
    }
    dprintf(("positive patches: %d\nnegative patches: %d\n",(int)positiveExamples.size(),(int)negativeExamples.size()));
}

void TLDDetector::generateScanGrid(int rows,int cols,Size initBox,std::vector<Rect2d>& res,bool withScaling){
    res.clear();
    //scales step: 1.2; hor step: 10% of width; verstep: 10% of height; minsize: 20pix
    for(double h=initBox.height, w=initBox.width;h<cols && w<rows;){
        for(double x=0;(x+w)<=(cols-1.0);x+=(0.1*w)){
            for(double y=0;(y+h)<=(rows-1.0);y+=(0.1*h)){
                res.push_back(Rect2d(x,y,w,h));
            }
        }
        if(withScaling){
            if(h<=initBox.height){
                h/=1.2; w/=1.2;
                if(h<20 || w<20){
                    h=initBox.height*1.2; w=initBox.width*1.2;
                    CV_Assert(h>initBox.height || w>initBox.width);
                }
            }else{
                h*=1.2; w*=1.2;
            }
        }else{
            break;
        }
    }
    dprintf(("%d rects in res\n",(int)res.size()));
}

bool TLDDetector::detect(const Mat& img,const Mat& imgBlurred,Rect2d& res,std::vector<Rect2d>& rect,std::vector<bool>& isObject,
        std::vector<bool>& shouldBeIntegrated){
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    Size initSize=tldModel->getMinSize();
    rect.clear();
    isObject.clear();
    shouldBeIntegrated.clear();

    Mat resized_img,blurred_img;
    Mat_<uchar> standardPatch(15,15);
    img.copyTo(resized_img);
    imgBlurred.copyTo(blurred_img);
    double originalVariance=tldModel->getOriginalVariance();;
    int dx=initSize.width/10,dy=initSize.height/10;
    Size2d size=img.size();
    double scale=1.0;
    int total=0,pass=0;
    int npos=0,nneg=0;
    double tmp=0,maxSc=-5.0;
    Rect2d maxScRect;
    START_TICK("detector");
    do{
        Mat_<double> intImgP,intImgP2;
        computeIntegralImages(resized_img,intImgP,intImgP2);

        for(int i=0,imax=cvFloor((0.0+resized_img.cols-initSize.width)/dx);i<imax;i++){
            for(int j=0,jmax=cvFloor((0.0+resized_img.rows-initSize.height)/dy);j<jmax;j++){
                total++;
                if(!patchVariance(intImgP,intImgP2,originalVariance,Point(dx*i,dy*j),initSize)){
                    continue;
                }
                if(!ensembleClassifier(&blurred_img.at<uchar>(dy*j,dx*i),(int)blurred_img.step[0])){
                    continue;
                }
                pass++;

                rect.push_back(Rect2d(dx*i*scale,dy*j*scale,initSize.width*scale,initSize.height*scale));
                resample(resized_img,Rect2d(Point(dx*i,dy*j),initSize),standardPatch);
                tmp=tldModel->Sr(standardPatch);
                isObject.push_back(tmp>THETA_NN);
                shouldBeIntegrated.push_back(abs(tmp-THETA_NN)<0.1);
                if(!isObject[isObject.size()-1]){
                    nneg++;
                    continue;
                }else{
                    npos++;
                }
                tmp=tldModel->Sc(standardPatch);
                if(tmp>maxSc){
                    maxSc=tmp;
                    maxScRect=rect[rect.size()-1];
                }
            }
        }

        size.width/=1.2;
        size.height/=1.2;
        scale*=1.2;
        resize(img,resized_img,size);
        GaussianBlur(resized_img,blurred_img,GaussBlurKernelSize,0.0f);
    }while(size.width>=initSize.width && size.height>=initSize.height);
    END_TICK("detector");

    dfprintf((stdout,"after NCC: nneg=%d npos=%d\n",nneg,npos));
#if !0
        std::vector<Rect2d> poss,negs;
        for(int i=0;i<(int)rect.size();i++){
            if(isObject[i])
                poss.push_back(rect[i]);
            else
                negs.push_back(rect[i]);
        }
        dfprintf((stdout,"%d pos and %d neg\n",(int)poss.size(),(int)negs.size()));
        drawWithRects(img,negs,poss);
#endif
#if !1
        std::vector<Rect2d> scanGrid;
        generateScanGrid(img.rows,img.cols,initSize,scanGrid);
        std::vector<double> results;
        Mat_<uchar> standardPatch_inner(15,15);
        for(int i=0;i<(int)scanGrid.size();i++){
            resample(img,scanGrid[i],standardPatch_inner);
            results.push_back(tldModel->Sr(standardPatch_inner));
        }
        std::vector<double>::iterator it=std::max_element(results.begin(),results.end());
        Mat image;
        img.copyTo(image);
        rectangle( image,scanGrid[it-results.begin()], 255, 1, 1 );
        imshow("img",image);
        waitKey();
#endif
#if !1
        Mat image;
        img.copyTo(image);
        rectangle( image,res, 255, 1, 1 );
        for(int i=0;i<(int)rect.size();i++){
          rectangle( image,rect[i], 0, 1, 1 );
        }
        imshow("img",image);
        waitKey();
#endif

    dfprintf((stdout,"%d after ensemble\n",pass));
    if(maxSc<0){
        return false;
    }
    res=maxScRect;
    return true;
}

bool TLDDetector::patchVariance(Mat_<double>& intImgP,Mat_<double>& intImgP2,double originalVariance,Point pt,Size size){
    return variance(intImgP,intImgP2,Rect(pt.x,pt.y,size.width,size.height)) >= 0.5*originalVariance;
}

double TLDDetector::ensembleClassifierNum(const uchar* data,int rowstep){
    TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(model));
    std::vector<TLDEnsembleClassifier>* classifiers=tldModel->getClassifiers();
    double p=0;
    for(int k=0;k<(int)classifiers->size();k++){
        p+=(*classifiers)[k].posteriorProbability(data,rowstep);
    }
    p/=classifiers->size();
    return p;
}

double TrackerTLDModel::Sr(const Mat_<uchar> patch){
    double splus=0.0;
    for(int i=0;i<(int)positiveExamples.size();i++){
        splus=MAX(splus,0.5*(NCC(positiveExamples[i],patch)+1.0));
    }
    double sminus=0.0;
    for(int i=0;i<(int)negativeExamples.size();i++){
        sminus=MAX(sminus,0.5*(NCC(negativeExamples[i],patch)+1.0));
    }
    if(splus+sminus==0.0){
        return 0.0;
    }
    return splus/(sminus+splus);
}

double TrackerTLDModel::Sc(const Mat_<uchar> patch){
    double splus=0.0;
    int med=getMedian(timeStampsPositive);
    for(int i=0;i<(int)positiveExamples.size();i++){
        if((int)timeStampsPositive[i]<=med){
            splus=MAX(splus,0.5*(NCC(positiveExamples[i],patch)+1.0));
        }
    }
    double sminus=0.0;
    for(int i=0;i<(int)negativeExamples.size();i++){
        sminus=MAX(sminus,0.5*(NCC(negativeExamples[i],patch)+1.0));
    }
    if(splus+sminus==0.0){
        return 0.0;
    }
    return splus/(sminus+splus);
}

void TrackerTLDModel::integrateRelabeled(Mat& img,Mat& imgBlurred,const std::vector<Rect2d>& box,const std::vector<bool>& isPositive,
          const std::vector<bool>& alsoIntoModel){
    Mat_<uchar> standardPatch(15,15),blurredPatch(minSize_);
    int positiveIntoModel=0,negativeIntoModel=0,positiveIntoEnsemble=0,negativeIntoEnsemble=0;
    for(int k=0;k<(int)box.size();k++){
        if(alsoIntoModel[k]){
            resample(img,box[k],standardPatch);
            if(isPositive[k]){
                positiveIntoModel++;
                pushIntoModel(standardPatch,true);
            }else{
                negativeIntoModel++;
                pushIntoModel(standardPatch,false);
            }
        }

#ifdef CLOSED_LOOP
        if(alsoIntoModel[k] || (isPositive[k]==false)){
#else
        if(alsoIntoModel[k]){
#endif
            resample(imgBlurred,box[k],blurredPatch);
            if(isPositive[k]){
                positiveIntoEnsemble++;
            }else{
                negativeIntoEnsemble++;
            }
            for(int i=0;i<(int)classifiers.size();i++){
                classifiers[i].integrate(blurredPatch,isPositive[k]);
            }
        }
    }
    if(negativeIntoModel>0)
        dfprintf((stdout,"negativeIntoModel=%d ",negativeIntoModel));
    if(positiveIntoModel>0)
        dfprintf((stdout,"positiveIntoModel=%d ",positiveIntoModel));
    if(negativeIntoEnsemble>0)
        dfprintf((stdout,"negativeIntoEnsemble=%d ",negativeIntoEnsemble));
    if(positiveIntoEnsemble>0)
        dfprintf((stdout,"positiveIntoEnsemble=%d ",positiveIntoEnsemble));
    dfprintf((stdout,"\n"));
}

void TrackerTLDModel::integrateAdditional(const std::vector<Mat_<uchar> >& eForModel,const std::vector<Mat_<uchar> >& eForEnsemble,bool isPositive){
    int positiveIntoModel=0,negativeIntoModel=0,positiveIntoEnsemble=0,negativeIntoEnsemble=0;
    for(int k=0;k<(int)eForModel.size();k++){
        double sr=Sr(eForModel[k]);
        if((sr>THETA_NN)!=isPositive){
            if(isPositive){
                positiveIntoModel++;
                pushIntoModel(eForModel[k],true);
            }else{
                negativeIntoModel++;
                pushIntoModel(eForModel[k],false);
            }
        }
        double p=0;
        for(int i=0;i<(int)classifiers.size();i++){
            p+=classifiers[i].posteriorProbability(eForEnsemble[k].data,(int)eForEnsemble[k].step[0]);
        }
        p/=classifiers.size();
        if((p>0.5)!=isPositive){
            if(isPositive){
                positiveIntoEnsemble++;
            }else{
                negativeIntoEnsemble++;
            }
            for(int i=0;i<(int)classifiers.size();i++){
                classifiers[i].integrate(eForEnsemble[k],isPositive);
            }
        }
    }
    if(negativeIntoModel>0)
        dfprintf((stdout,"negativeIntoModel=%d ",negativeIntoModel));
    if(positiveIntoModel>0)
        dfprintf((stdout,"positiveIntoModel=%d ",positiveIntoModel));
    if(negativeIntoEnsemble>0)
        dfprintf((stdout,"negativeIntoEnsemble=%d ",negativeIntoEnsemble));
    if(positiveIntoEnsemble>0)
        dfprintf((stdout,"positiveIntoEnsemble=%d ",positiveIntoEnsemble));
    dfprintf((stdout,"\n"));
}

int Pexpert::additionalExamples(std::vector<Mat_<uchar> >& examplesForModel,std::vector<Mat_<uchar> >& examplesForEnsemble){
    examplesForModel.clear();examplesForEnsemble.clear();
    examplesForModel.reserve(100);examplesForEnsemble.reserve(100);

    std::vector<Rect2d> closest(10),scanGrid;
    Mat scaledImg,blurredImg;

    double scale=scaleAndBlur(img_,cvRound(log(1.0*resultBox_.width/(initSize_.width))/log(1.2)),scaledImg,blurredImg,GaussBlurKernelSize);
    TLDDetector::generateScanGrid(img_.rows,img_.cols,initSize_,scanGrid);
    getClosestN(scanGrid,Rect2d(resultBox_.x/scale,resultBox_.y/scale,resultBox_.width/scale,resultBox_.height/scale),10,closest);

    Point2f center;
    Size2f size;
    for(int i=0;i<(int)closest.size();i++){
        for(int j=0;j<10;j++){
            Mat_<uchar> standardPatch(15,15),blurredPatch(initSize_);
            center.x=(float)(closest[i].x+closest[i].width*(0.5+rng.uniform(-0.01,0.01)));
            center.y=(float)(closest[i].y+closest[i].height*(0.5+rng.uniform(-0.01,0.01)));
            size.width=(float)(closest[i].width*rng.uniform((double)0.99,(double)1.01));
            size.height=(float)(closest[i].height*rng.uniform((double)0.99,(double)1.01));
            float angle=(float)rng.uniform(-5.0,5.0);

#ifdef BLUR_AS_VADIM
            GaussianBlur(standardPatch,blurredPatch,GaussBlurKernelSize,0.0);
#else
            resample(blurredImg,RotatedRect(center,size,angle),blurredPatch);
#endif
            resample(scaledImg,RotatedRect(center,size,angle),standardPatch);
            for(int y=0;y<standardPatch.rows;y++){
                for(int x=0;x<standardPatch.cols;x++){
                    standardPatch(x,y)+=(uchar)rng.gaussian(5.0);
                }
            }
            examplesForModel.push_back(standardPatch);
            examplesForEnsemble.push_back(blurredPatch);
        }
    }
    return 0;
}

bool Nexpert::operator()(Rect2d box){
    if(overlap(resultBox_,box)<0.2){
        return false;
    }
    return true;
}

Data::Data(Rect2d initBox){
    double minDim=MIN(initBox.width,initBox.height);
    scale = 20.0/minDim;
    minSize.width=(int)(initBox.width*20.0/minDim);
    minSize.height=(int)(initBox.height*20.0/minDim);
    frameNum=0;
    dprintf(("minSize= %dx%d\n",minSize.width,minSize.height));
}

void Data::printme(FILE*  port){
    dfprintf((port,"Data:\n"));
    dfprintf((port,"\tframeNum=%d\n",frameNum));
    dfprintf((port,"\tconfident=%s\n",confident?"true":"false"));
    dfprintf((port,"\tfailedLastTime=%s\n",failedLastTime?"true":"false"));
    dfprintf((port,"\tminSize=%dx%d\n",minSize.width,minSize.height));
}
void TrackerTLDModel::printme(FILE*  port){
    dfprintf((port,"TrackerTLDModel:\n"));
    dfprintf((port,"\tpositiveExamples.size()=%d\n",(int)positiveExamples.size()));
    dfprintf((port,"\tnegativeExamples.size()=%d\n",(int)negativeExamples.size()));
}
void MyMouseCallbackDEBUG::onMouse( int event, int x, int y){
    if(event== EVENT_LBUTTONDOWN){
        Mat imgCanvas;
        img_.copyTo(imgCanvas);
        TrackerTLDModel* tldModel=((TrackerTLDModel*)static_cast<TrackerModel*>(detector_->model));
        Size initSize=tldModel->getMinSize();
        Mat_<uchar> standardPatch(15,15);
        double originalVariance=tldModel->getOriginalVariance();;
        double tmp;

        Mat resized_img,blurred_img;
        double scale=1.2;
        //double scale=1.2*1.2*1.2*1.2;
        Size2d size(img_.cols/scale,img_.rows/scale);
        resize(img_,resized_img,size);
        resize(imgBlurred_,blurred_img,size);

        Mat_<double> intImgP,intImgP2;
        detector_->computeIntegralImages(resized_img,intImgP,intImgP2);

        int dx=initSize.width/10, dy=initSize.height/10,
            i=(int)(x/scale/dx), j=(int)(y/scale/dy);

        dfprintf((stderr,"patchVariance=%s\n",(detector_->patchVariance(intImgP,intImgP2,originalVariance,Point(dx*i,dy*j),initSize))?"true":"false"));
        dfprintf((stderr,"p=%f\n",(detector_->ensembleClassifierNum(&blurred_img.at<uchar>(dy*j,dx*i),(int)blurred_img.step[0]))));
        fprintf(stderr,"ensembleClassifier=%s\n",
                (detector_->ensembleClassifier(&blurred_img.at<uchar>(dy*j,dx*i),(int)blurred_img.step[0]))?"true":"false");

        resample(resized_img,Rect2d(Point(dx*i,dy*j),initSize),standardPatch);
        tmp=tldModel->Sr(standardPatch);
        dfprintf((stderr,"Sr=%f\n",tmp));
        dfprintf((stderr,"isObject=%s\n",(tmp>THETA_NN)?"true":"false"));
        dfprintf((stderr,"shouldBeIntegrated=%s\n",(abs(tmp-THETA_NN)<0.1)?"true":"false"));
        dfprintf((stderr,"Sc=%f\n",tldModel->Sc(standardPatch)));

        rectangle(imgCanvas,Rect2d(Point2d(scale*dx*i,scale*dy*j),Size2d(initSize.width*scale,initSize.height*scale)), 0, 2, 1 );
        imshow("picker",imgCanvas);
        waitKey();
    }
}
void TrackerTLDModel::pushIntoModel(const Mat_<uchar>& example,bool positive){
    std::vector<Mat_<uchar> >* proxyV;
    unsigned int* proxyN;
    std::vector<unsigned int>* proxyT;
    if(positive){
        proxyV=&positiveExamples;
        proxyN=&timeStampPositiveNext;
        proxyT=&timeStampsPositive;
    }else{
        proxyV=&negativeExamples;
        proxyN=&timeStampNegativeNext;
        proxyT=&timeStampsNegative;
    }
    if(proxyV->size()<MAX_EXAMPLES_IN_MODEL){
        proxyV->push_back(example);
        proxyT->push_back(*proxyN);
    }else{
        int index=rng.uniform((int)0,(int)proxyV->size());
        (*proxyV)[index]=example;
        (*proxyT)[index]=(*proxyN);
    }
    (*proxyN)++;
}

} /* namespace cv */
