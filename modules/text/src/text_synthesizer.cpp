#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include "opencv2/text/text_synthesizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <errno.h>
#include <map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_QT5GUI
#include <QImage>
#include <QFont>
#include <QPainter>
#include <QFontDatabase>
#include <QFontMetrics>
#endif


//TODO FIND apropriate
#define CV_IMWRITE_JPEG_QUALITY 1
#define CV_LOAD_IMAGE_COLOR 1
namespace cv{
namespace text{

namespace {
//Unnamed namespace with auxiliary classes and functions used for quick computation
template <typename T> T min_ (T v1, T v2) {
    return (v1 < v2) * v1 + (v1 >= v2) * v2;
}

template <typename T> T max_(T v1, T v2) {
    return (v1 > v2)* v1 + (v1 <= v2) * v2;
}

template <typename P,typename BL_A,typename BL> void blendRGBA(Mat& out, const Mat &in1, const Mat& in2){
    CV_Assert (out.cols == in1.cols && out.cols == in2.cols);
    CV_Assert (out.rows == in1.rows && out.rows == in2.rows);
    CV_Assert (out.channels() == 4 && in1.channels() == 4 && in2.channels() == 4);
    int lineWidth=out.cols * 4;
    BL blend;
    BL_A blendA;
    for(int y = 0; y < out.rows; y++){
        const P* in1B = in1.ptr<P> (y) ;
        const P* in1G = in1.ptr<P> (y) + 1;
        const P* in1R = in1.ptr<P> (y) + 2;
        const P* in1A = in1.ptr<P> (y) + 3;

        const P* in2B = in2.ptr<P> (y);
        const P* in2G = in2.ptr<P> (y) + 1;
        const P* in2R = in2.ptr<P> (y) + 2;
        const P* in2A = in2.ptr<P> (y) + 3;

        P* outB = out.ptr<P> (y);
        P* outG = out.ptr<P> (y) + 1;
        P* outR = out.ptr<P> (y) + 2;
        P* outA = out.ptr<P> (y) + 3;

        for(int x = 0; x < lineWidth; x += 4){
            outB[x] = blend(in1B + x, in1A + x, in2B + x, in2A + x);
            outG[x] = blend(in1G + x, in1A + x, in2G + x, in2A + x);
            outR[x] = blend(in1R + x, in1A + x, in2R + x, in2A + x);
            outA[x] = blendA(in1A[x], in2A[x]);
        }
    }
}

#ifdef HAVE_QT5GUI
std::map<int,int> initQt2CvScriptCodeMap () ;
std::map<int,int> initQt2CvScriptCodeMap () {
    std::map<int,int> res;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ANY] = QFontDatabase::Any;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_LATIN] = QFontDatabase::Latin;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GREEK] = QFontDatabase::Greek;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_CYRILLIC] = QFontDatabase::Cyrillic;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ARMENIAN] = QFontDatabase::Armenian;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ARABIC] = QFontDatabase::Arabic;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_HEBREW] = QFontDatabase::Hebrew;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_SYRIAC] = QFontDatabase::Syriac;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_THAANA] = QFontDatabase::Thaana;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_DEVANAGARI] = QFontDatabase::Devanagari;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_BENGALI] = QFontDatabase::Bengali;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GURMUKHI] = QFontDatabase::Gurmukhi;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GUJARATI] = QFontDatabase::Gujarati;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ORIYA] = QFontDatabase::Oriya;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_TAMIL] = QFontDatabase::Tamil;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_TELUGU] = QFontDatabase::Telugu;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_KANNADA] = QFontDatabase::Kannada;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_MALAYALAM] = QFontDatabase::Malayalam;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_SINHALA] = QFontDatabase::Sinhala;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_THAI] = QFontDatabase::Thai;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_LAO] = QFontDatabase::Lao;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_TIBETAN] = QFontDatabase::Tibetan;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_MYANMAR] = QFontDatabase::Myanmar;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GEORGIAN] = QFontDatabase::Georgian;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_KHMER] = QFontDatabase::Khmer;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_CHINESE_SIMPLIFIED] = QFontDatabase::SimplifiedChinese;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_CHINESE_TRADITIONAL] = QFontDatabase::TraditionalChinese;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_JAPANESE] = QFontDatabase::Japanese;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_KOREAM] = QFontDatabase::Korean;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_VIETNAMESE] = QFontDatabase::Vietnamese;
    return res;
}


int getQt2CvScriptCode(int cvScriptCode);
int getQt2CvScriptCode(int cvScriptCode){
    static std::map<int,int> m(initQt2CvScriptCodeMap());
    if(m.find(cvScriptCode)!=m.end()){
        return m[cvScriptCode];
    }else{
        CV_Error(Error::StsError,"Unknown script_code");
        return 0;
    }
}
#endif //HAVE_QT5GUI


std::map<int,String> initScriptCode2StringMap();
std::map<int,String> initScriptCode2StringMap(){
    std::map<int,String> res;
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ANY]="Any";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_LATIN]="Latin";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GREEK]="Greek";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_CYRILLIC]="Cyrillic";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ARMENIAN]="Armenian";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ARABIC]="Arabic";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_HEBREW]="Hebrew";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_SYRIAC]="Syriac";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_THAANA]="Thaana";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_DEVANAGARI]="Devanagari";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_BENGALI]="Bengali";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GURMUKHI]="Gurmukhi";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GUJARATI]="Gujarati";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_ORIYA]="Oriya";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_TAMIL]="Tamil";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_TELUGU]="Telugu";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_KANNADA]="Kannada";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_MALAYALAM]="Malayalam";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_SINHALA]="Sinhala";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_THAI]="Thai";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_LAO]="Lao";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_TIBETAN]="Tibetan";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_MYANMAR]="Myanmar";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_GEORGIAN]="Georgian";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_KHMER]="Khmer";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_CHINESE_SIMPLIFIED]="SimplifiedChinese";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_CHINESE_TRADITIONAL]="TraditionalChinese";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_JAPANESE]="Japanese";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_KOREAM]="Korean";
    res[CV_TEXT_SYNTHESIZER_SCRIPT_VIETNAMESE]="Vietnamese";
    return res;
}


String getCvScriptCode2String(int cvScriptCode);
String getCvScriptCode2String(int cvScriptCode){
    static std::map<int,String> m(initScriptCode2StringMap());
    if(m.find(cvScriptCode)!=m.end()){
        return m[cvScriptCode];
    }else{
        CV_Error(Error::StsError,"Unknown script_code");
        return "Error";
    }
}


}//unnamed namespace
void blendWeighted(Mat& out,Mat& top,Mat& bottom,float topMask,float bottomMask);
void blendWeighted(Mat& out,Mat& top,Mat& bottom,float topMask,float bottomMask){
    if(out.channels( )==3 && top.channels( )==3 && bottom.channels( )==3 ){
        for(int y=0;y<out.rows;y++){
            float* outR=out.ptr<float>(y);
            float* outG=out.ptr<float>(y)+1;
            float* outB=out.ptr<float>(y)+2;

            float* topR=top.ptr<float>(y);
            float* topG=top.ptr<float>(y)+1;
            float* topB=top.ptr<float>(y)+2;

            float* bottomR=bottom.ptr<float>(y);
            float* bottomG=bottom.ptr<float>(y)+1;
            float* bottomB=bottom.ptr<float>(y)+2;

            for(int x=0;x<out.cols;x++){
                int x3=x*3;
                outR[x3]=topR[x3]*topMask+bottomR[x3]*bottomMask;
                outG[x3]=topG[x3]*topMask+bottomG[x3]*bottomMask;
                outB[x3]=topB[x3]*topMask+bottomB[x3]*bottomMask;
            }
        }
        return;
    }
    if(out.channels( )==1 && top.channels( )==1 && bottom.channels( )==1 ){
        for(int y=0;y<out.rows;y++){
            float* outG=out.ptr<float>(y);
            float* topG=top.ptr<float>(y);
            float* bottomG=bottom.ptr<float>(y);
            for(int x=0;x<out.cols;x++){
                outG[x]=topG[x]*topMask+bottomG[x]*bottomMask;
            }
        }
        return;
    }
    CV_Error(Error::StsError,"Images must all be either CV_32FC1 CV_32FC32");
}

void blendWeighted(Mat& out,Mat& top,Mat& bottom,Mat& topMask_,Mat& bottomMask_);
void blendWeighted(Mat& out,Mat& top,Mat& bottom,Mat& topMask_,Mat& bottomMask_){
    for(int y=0;y<out.rows;y++){
        float* outR=out.ptr<float>(y);
        float* outG=out.ptr<float>(y)+1;
        float* outB=out.ptr<float>(y)+2;

        float* topR=top.ptr<float>(y);
        float* topG=top.ptr<float>(y)+1;
        float* topB=top.ptr<float>(y)+2;

        float* bottomR=bottom.ptr<float>(y);
        float* bottomG=bottom.ptr<float>(y)+1;
        float* bottomB=bottom.ptr<float>(y)+2;

        float* topMask=topMask_.ptr<float>(y);
        float* bottomMask=bottomMask_.ptr<float>(y);

        for(int x=0;x<out.cols;x++){
            int x3=x*3;
            outR[x3]=topR[x3]*topMask[x]+bottomR[x3]*bottomMask[x];
            outG[x3]=topG[x3]*topMask[x]+bottomG[x3]*bottomMask[x];
            outB[x3]=topB[x3]*topMask[x]+bottomB[x3]*bottomMask[x];
        }
    }
}

void blendOverlay(Mat& out,Mat& top,Mat& bottom,Mat& topMask);
void blendOverlay(Mat& out,Mat& top,Mat& bottom,Mat& topMask){
    for(int y=0;y<out.rows;y++){
        float* outR=out.ptr<float>(y);
        float* outG=out.ptr<float>(y)+1;
        float* outB=out.ptr<float>(y)+2;

        float* topR=top.ptr<float>(y);
        float* topG=top.ptr<float>(y)+1;
        float* topB=top.ptr<float>(y)+2;

        float* bottomR=bottom.ptr<float>(y);
        float* bottomG=bottom.ptr<float>(y)+1;
        float* bottomB=bottom.ptr<float>(y)+2;

        float* mask=topMask.ptr<float>(y);

        for(int x=0;x<out.cols;x++){
            int x3=x*3;
            outR[x3]=topR[x3]*mask[x]+bottomR[x3]*(1-mask[x]);
            outG[x3]=topG[x3]*mask[x]+bottomG[x3]*(1-mask[x]);
            outB[x3]=topB[x3]*mask[x]+bottomB[x3]*(1-mask[x]);
        }
    }
}

void blendOverlay(Mat& out,Scalar topCol,Scalar bottomCol,Mat& topMask);
void blendOverlay(Mat& out,Scalar topCol,Scalar bottomCol,Mat& topMask){
    float topR=float(topCol[0]);
    float topG=float(topCol[1]);
    float topB=float(topCol[2]);

    float bottomR=float(bottomCol[0]);
    float bottomG=float(bottomCol[1]);
    float bottomB=float(bottomCol[2]);

    for(int y=0;y<out.rows;y++){
        float* outR=out.ptr<float>(y);
        float* outG=out.ptr<float>(y)+1;
        float* outB=out.ptr<float>(y)+2;

        float* mask=topMask.ptr<float>(y);

        for(int x=0;x<out.cols;x++){
            int x3=x*3;
            outR[x3]=topR*mask[x]+bottomR*(1-mask[x]);
            outG[x3]=topG*mask[x]+bottomG*(1-mask[x]);
            outB[x3]=topB*mask[x]+bottomB*(1-mask[x]);
        }
    }
}


TextSynthesizer::TextSynthesizer(int maxSampleWidth,int sampleHeight):
    resHeight_(sampleHeight),maxResWidth_(maxSampleWidth)
{
    underlineProbabillity_=0.05;
    italicProbabillity_=.1;
    boldProbabillity_=.1;
    maxPerspectiveDistortion_=20;

    shadowProbabillity_=.3;
    maxShadowOpacity_=.7;
    maxShadowSize_=3;
    maxShadowHoffset_=2;
    maxShadowVoffset_=2;

    borderProbabillity_=.5;
    maxBorderSize_=2;

    curvingProbabillity_=.1;
    maxHeightDistortionPercentage_=5;
    maxCurveArch_=.1;

    finalBlendAlpha_=.3;
    finalBlendProb_=.1;
    compressionNoiseProb_=.3;
}

class TextSynthesizerQtImpl: public TextSynthesizer{
protected:
    bool rndProbUnder(double v){
        return (this->rng_.next()%10000)<(10000*v);
    }

    void updateFontNameList(std::vector<String>& fntList){
#ifdef HAVE_QT5GUI
        fntList.clear();
        QStringList lst=this->fntDb_->families(QFontDatabase::WritingSystem(getQt2CvScriptCode(this->script_)));
        for(int k=0;k<lst.size();k++){
            fntList.push_back(lst[k].toUtf8().constData());
        }
#else
        fntList.clear();
#endif
    }

    void modifyAvailableFonts(std::vector<String>& fntList){
        std::vector<String> dbList;
        this->updateFontNameList(dbList);
        for(size_t k =0;k<fntList.size();k++){
            if(std::find(dbList.begin(), dbList.end(), fntList[k]) == dbList.end()){
                CV_Error(Error::StsError,"The font name list must only contain fonts in your system");
            }
        }
        this->availableFonts_=fntList;
    }

#ifdef HAVE_QT5GUI
    QFont generateFont(){
        CV_Assert(this->availableFonts_.size());
        QFont fnt(this->availableFonts_[rng_.next() % this->availableFonts_.size()].c_str());
        fnt.setPixelSize(this->resHeight_-2*this->txtPad_);
        if(this->rndProbUnder(this->underlineProbabillity_)){
            fnt.setUnderline(true);
        }else{
            fnt.setUnderline(false);
        }
        if(this->rndProbUnder(this->boldProbabillity_)){
            fnt.setBold(true);
        }else{
            fnt.setBold(false);
        }
        if(this->rndProbUnder(this->italicProbabillity_)){
            fnt.setItalic(true);
        }else{
            fnt.setItalic(false);
        }
        return fnt;
    }
#endif
    void generateTxtPatch(Mat& output,Mat& outputMask,String caption){
        const int maxTxtWidth=this->maxResWidth_;
        Mat textImg;
        textImg =cv::Mat(this->resHeight_,maxTxtWidth,CV_8UC3,Scalar_<uchar>(0,0,0));
#ifdef HAVE_QT5GUI
        QImage qimg((unsigned char*)(textImg.data), textImg.cols, textImg.rows, textImg.step, QImage::Format_RGB888);
        QPainter qp(&qimg);
        qp.setPen(QColor(255,255,255));
        QFont fnt=this->generateFont();
        QFontMetrics fntMtr(fnt,qp.device());
        QRect bbox=fntMtr.tightBoundingRect(caption.c_str());
        qp.setFont(fnt);
        qp.drawText(QPoint(txtPad_,txtPad_+ bbox.height()), caption.c_str());
        qp.end();
        textImg=textImg.colRange(0,min( bbox.width()+2*txtPad_,maxTxtWidth-1));
#else
        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 1;
        int thickness = 2;
        int baseline = 0;
        Size textSize = getTextSize(caption, fontFace, fontScale, thickness, &baseline);
        putText(textImg, caption, Point(this->txtPad_,this->resHeight_-this->txtPad_),
                FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, Scalar_<uchar>(255,255,255), thickness, 8);
        textImg=textImg.colRange(0,min( textSize.width+2*txtPad_,maxTxtWidth-1));
        //TODO Warn without throuwing an exception
#endif
        Mat textGrayImg;
        cvtColor(textImg,textGrayImg,COLOR_RGB2GRAY);
        //Obtaining color triplet
        int colorTriplet=this->rng_.next()%this->colorClusters_.rows;
        uchar* cVal=this->colorClusters_.ptr<uchar>(colorTriplet);
        Scalar_<float> fgText(float(cVal[0]/255.0),float(cVal[1]/255.0),float(cVal[2]/255.0));
        Scalar_<float> fgBorder(float(cVal[3]/255.0),float(cVal[4]/255.0),float(cVal[5]/255.0));
        Scalar_<float> fgShadow(float(cVal[6]/255.0),float(cVal[7]/255.0),float(cVal[8]/255.0));

        Mat floatTxt;Mat floatBorder;Mat floatShadow;
        textGrayImg.convertTo(floatTxt, CV_32FC1, 1.0/255.0);

        //Sampling uniform distributionfor sizes
        int borderSize=(this->rng_.next()%this->maxBorderSize_)*this->rndProbUnder(this->borderProbabillity_);
        int shadowSize=(this->rng_.next()%this->maxShadowSize_)*this->rndProbUnder(this->shadowProbabillity_);
        int voffset=(this->rng_.next()%(shadowSize*2+1))-shadowSize;
        int hoffset=(this->rng_.next()%(shadowSize*2+1))-shadowSize;
        float shadowOpacity=float(((this->rng_.next()%10000)*maxShadowOpacity_)/10000.0);

        //generating shadows
        generateDilation(floatBorder,floatTxt,borderSize,0,0);
        generateDilation(floatShadow,floatBorder,shadowSize,voffset,hoffset);

        Mat floatBordered=Mat(floatTxt.rows,floatTxt.cols,CV_32FC3);
        Mat floatShadowed=Mat(floatTxt.rows,floatTxt.cols,CV_32FC3);
        Mat floatMixed=Mat(floatTxt.rows,floatTxt.cols,CV_32FC3);
        Mat floatMask=Mat(floatTxt.rows,floatTxt.cols,CV_32FC1);

        blendOverlay(floatBordered,fgText,fgBorder, floatTxt);
        blendOverlay(floatShadowed,fgShadow,fgShadow, floatTxt);
        blendOverlay(floatMixed,floatBordered,floatShadowed, floatBorder);
        blendWeighted(floatMask,floatShadow,floatBorder, shadowOpacity,1-shadowOpacity);
        floatMixed.copyTo(output);floatMask.copyTo(outputMask);
    }

    String getScriptName() {
        return getCvScriptCode2String(this->script_);
    }

    void generateDilation(Mat& outputImg,
                          const Mat& inputImg,int dilationSize, int horizOffset,int vertOffset){
        //erosion is defined as a negative dilation size
        if (dilationSize==0) {
            inputImg.copyTo(outputImg);
        } else {
            if (dilationSize > 0) {
                if(horizOffset==0 && vertOffset==0){
                    dilate(inputImg,outputImg,Mat(),Point(-1, -1),dilationSize);
                }else{
                    Mat tmpMat;
                    dilate(inputImg,tmpMat,Mat(),Point(-1, -1),dilationSize);
                    outputImg=Mat(inputImg.rows,inputImg.cols,inputImg.type(),Scalar(0));
                    int validWidth=inputImg.cols-abs(horizOffset);
                    int validHeight=inputImg.rows-abs(vertOffset);
                    tmpMat(Rect(max(0,-horizOffset),max(0,-vertOffset), validWidth,validHeight)).
                            copyTo(outputImg(Rect(max(0,horizOffset),max(0,vertOffset), validWidth,validHeight)));
                }
            }else{
                if(horizOffset==0 && vertOffset==0){
                    dilate(inputImg,outputImg,Mat(),Point(-1, -1),-dilationSize);
                }else{
                    Mat tmpMat;
                    erode(inputImg,tmpMat,Mat(),Point(-1, -1),-dilationSize);
                    outputImg=Mat(inputImg.rows,inputImg.cols,inputImg.type(),Scalar(0));
                    int validWidth=inputImg.cols-abs(horizOffset);
                    int validHeight=inputImg.rows-abs(vertOffset);
                    tmpMat(Rect(max(0,-horizOffset),max(0,-vertOffset), validWidth,validHeight)).
                            copyTo(outputImg(Rect(max(0,horizOffset),max(0,vertOffset), validWidth,validHeight)));
                }
            }
        }
    }

    void randomlyDistortPerspective(const Mat& inputImg,Mat& outputImg){
        int N=int(this->maxPerspectiveDistortion_);
        if(N>0){
            float xa=this->rng_.next()%N;
            float xb=this->rng_.next()%N;
            float xc=this->rng_.next()%N;
            float xd=this->rng_.next()%N;

            float ya=this->rng_.next()%N;
            float yb=this->rng_.next()%N;
            float yc=this->rng_.next()%N;
            float yd=this->rng_.next()%N;

            float left=min_<float>(xa,xd);
            float top=min_<float>(ya,yb);
            float right=100-min_<float>(xb,xc);
            float bottom=100-min_<float>(yc,yd);

            float horizCoef;
            float vertCoef;

            if(right-left<bottom-top){
                vertCoef=(1+bottom-top)/(1+right-left);
                horizCoef=0;
            }else{
                vertCoef=0;
                horizCoef=(1+right-left)/(1+bottom-top);
            }

            xa=0+xa*horizCoef;xd=0+xd*horizCoef;
            xb=100-xb*horizCoef;xc=100-xc*horizCoef;
            ya=ya*vertCoef;yb=yb*vertCoef;
            yc=100-yc*vertCoef;yd=100-yd*vertCoef;

            std::vector<Point> src(4);std::vector<Point> dst(4);
            src[0]=Point2f(0,0);src[1]=Point2f(100,0);src[2]=Point2f(100,100);src[3]=Point2f(0,100);
            dst[0]=Point2f(xa,ya);
            dst[1]=Point2f(xb,yb);
            dst[2]=Point2f(xc,yc);
            dst[3]=Point2f(xd,yd);
            Mat h=findHomography(src,dst);
            warpPerspective(inputImg,outputImg,h,inputImg.size());
        }else{
            outputImg=inputImg;
        }
    }

    void addCurveDeformation(const Mat& inputImg,Mat& outputImg){
        if (this->rndProbUnder(this->curvingProbabillity_)){
            Mat X=Mat(inputImg.rows,inputImg.cols,CV_32FC1);
            Mat Y=Mat(inputImg.rows,inputImg.cols,CV_32FC1);
            int xAdd=-int(this->rng_.next()%inputImg.cols);
            float xMult=(this->rng_.next()%10000)*float(maxCurveArch_)/10000;
            int sign=(this->rng_.next()%2)?-1:1;
            for(int y=0;y<inputImg.rows;y++){
                float* xRow=X.ptr<float>(y);
                float* yRow=Y.ptr<float>(y);
                for(int x=0;x<inputImg.cols;x++){
                    xRow[x]=float(x);
                    yRow[x] = float(y + sign * cos((x + xAdd) * xMult)
                                    * maxHeightDistortionPercentage_ - sign
                                    * maxHeightDistortionPercentage_);
                }
            }
            remap(inputImg,outputImg,X,Y,INTER_LINEAR);
        }else{
            outputImg=inputImg;
        }
    }

    void addCompressionArtifacts(Mat& img){
        if(this->rndProbUnder(this->compressionNoiseProb_)){
            std::vector<uchar> buffer;
            std::vector<int> parameters;
            parameters.push_back(CV_IMWRITE_JPEG_QUALITY);
            parameters.push_back(this->rng_.next() % 100);
            Mat ucharImg;
            img.convertTo(ucharImg,CV_8UC3,255);
            imencode(".jpg",ucharImg,buffer,parameters);
            ucharImg=imdecode(buffer,CV_LOAD_IMAGE_COLOR);
            ucharImg.convertTo(img,CV_32FC3,1.0/255);
        }
    }

    void initColorClusters(){
        this->colorClusters_=Mat(4,3,CV_8UC3,Scalar(32,32,32));

        this->colorClusters_.at<Vec3b>(0, 0)=Vec3b(192,32,32);
        this->colorClusters_.at<Vec3b>(0, 1)=Vec3b(192,255,32);
        this->colorClusters_.at<Vec3b>(0, 2)=Vec3b(0,32,32);

        this->colorClusters_.at<Vec3b>(0, 0)=Vec3b(0,32,192);
        this->colorClusters_.at<Vec3b>(0, 1)=Vec3b(0,255,32);
        this->colorClusters_.at<Vec3b>(0, 2)=Vec3b(0,0,64);

        this->colorClusters_.at<Vec3b>(0, 0)=Vec3b(128,128,128);
        this->colorClusters_.at<Vec3b>(0, 1)=Vec3b(255,255,255);
        this->colorClusters_.at<Vec3b>(0, 2)=Vec3b(0,0,0);

        this->colorClusters_.at<Vec3b>(0, 0)=Vec3b(255,255,255);
        this->colorClusters_.at<Vec3b>(0, 1)=Vec3b(128,128,128);
        this->colorClusters_.at<Vec3b>(0, 2)=Vec3b(0,0,0);
    }

    RNG rng_;//Randon number generator used for all distributions
    int txtPad_;
#ifdef HAVE_QT5GUI
    Ptr<QFontDatabase> fntDb_;
#endif
    std::vector<String> availableFonts_;
    std::vector<String> availableBgSampleFiles_;
    std::vector<Mat> availableBgSampleImages_;
    Mat colorClusters_;
    int script_;
 public:
    TextSynthesizerQtImpl(int script,
                          int maxSampleWidth = 400,
                          int sampleHeight = 50,
                          uint64 rndState = 0)
        : TextSynthesizer(maxSampleWidth, sampleHeight)
        , rng_(rndState != 0 ? rndState:std::time(NULL))
        , txtPad_(10) {
#ifdef HAVE_QT5GUI
        CV_Assert(initQt2CvScriptCodeMap().count(script));//making sure script is a valid script code
#endif
        this->script_=script;
        //QT needs to be initialised. Highgui does this
        namedWindow("__w");
        waitKey(1);
        destroyWindow("__w");
#ifdef HAVE_QT5GUI
        this->fntDb_ = Ptr<QFontDatabase>(new QFontDatabase());
#endif
        this->updateFontNameList(this->availableFonts_);
        this->initColorClusters();
    }

    void getRandomSeed (OutputArray res) const {
        Mat tmpMat(1,8,CV_8UC1);
        tmpMat.ptr<uint64>(0)[0] = this->rng_.state;
        tmpMat.copyTo(res);
    }

    void setRandomSeed (Mat state) {
        CV_Assert (state.rows == 1 && state.cols == 8);
        CV_Assert (state.depth() == CV_8U && state.channels() == 1);
        this->rng_.state=state.ptr<uint64>(0)[0];
    }

    void generateBgSample(CV_OUT Mat& sample){
        if(this->availableBgSampleImages_.size()!=0){
            Mat& img=availableBgSampleImages_[this->rng_.next()%availableBgSampleImages_.size()];
            int left=this->rng_.next()%(img.cols-maxResWidth_);
            int top=this->rng_.next()%(img.rows-resHeight_);
            img.colRange(Range(left,left+maxResWidth_)).rowRange(Range(top,top+resHeight_)).copyTo(sample);
        }else{
            if(this->availableBgSampleFiles_.size()==0){
                Mat res(this->resHeight_,this->maxResWidth_,CV_8UC3);
                this->rng_.fill(res,RNG::UNIFORM,0,256);
                res.copyTo(sample);
            }else{
                Mat img;
                img=imread(this->availableBgSampleFiles_[this->rng_.next()%availableBgSampleFiles_.size()].c_str(),IMREAD_COLOR);
                CV_Assert(img.data != NULL);
                CV_Assert(img.cols>maxResWidth_ && img.rows> resHeight_);
                int left=this->rng_.next()%(img.cols-maxResWidth_);
                int top=this->rng_.next()%(img.rows-resHeight_);
                img.colRange(Range(left,left+maxResWidth_)).rowRange(Range(top,top+resHeight_)).copyTo(sample);
            }
        }
        if(sample.channels()==4){
            Mat rgb;
            cvtColor(sample,rgb,COLOR_RGBA2RGB);
            sample=rgb;
        }
        if(sample.channels()==1){
            Mat rgb;
            cvtColor(sample,rgb,COLOR_GRAY2RGB);
            sample=rgb;
        }
    }

    void generateTxtSample(String caption,CV_OUT Mat& sample,CV_OUT Mat& sampleMask){
        generateTxtPatch(sample,sampleMask,caption);
    }

    void generateSample(String caption,CV_OUT Mat & sample){
        Mat txtSample;
        Mat txtCurved;
        Mat txtDistorted;
        Mat bgSample;
        Mat bgResized;
        Mat txtMask;
        Mat txtMerged;
        Mat floatBg;
        std::vector<Mat> txtChannels;
        generateTxtPatch(txtSample,txtMask,caption);

        split(txtSample,txtChannels);
        txtChannels.push_back(txtMask);
        merge(txtChannels,txtMerged);
        addCurveDeformation(txtMerged,txtCurved);
        randomlyDistortPerspective(txtCurved,txtDistorted);
        split(txtDistorted,txtChannels);
        txtMask=txtChannels[3];
        txtChannels.pop_back();
        merge(txtChannels,txtSample);

        generateBgSample(bgSample);
        bgSample.convertTo(floatBg, CV_32FC3, 1.0/255.0);
        bgResized=floatBg.colRange(0,txtSample.cols);
        sample=Mat(txtDistorted.rows,txtDistorted.cols,CV_32FC3);

        blendOverlay(sample,txtSample,bgResized,txtMask);
        float blendAlpha=float(this->finalBlendAlpha_*(this->rng_.next()%1000)/1000.0);
        if(this->rndProbUnder(this->finalBlendProb_)){
            blendWeighted(sample,sample,bgResized,1-blendAlpha,blendAlpha);
        }
        addCompressionArtifacts(sample);
    }

    void getColorClusters(CV_OUT Mat& clusters) const {
        this->colorClusters_.copyTo(clusters);
    }

    void setColorClusters(Mat clusters){
        CV_Assert(clusters.type()==CV_8UC3);
        CV_Assert(clusters.cols==3);
        clusters.copyTo(this->colorClusters_);
    }

    std::vector<String> listAvailableFonts() const {
        std::vector<String> res;
        res=this->availableFonts_;
        return res;
    }

    virtual void addBgSampleImage(const Mat& inImg){
        CV_Assert(inImg.cols>maxResWidth_ && inImg.rows> resHeight_);
        Mat img;
        switch(inImg.type()){
        case CV_8UC1: {
            cvtColor(inImg, img, COLOR_GRAY2RGBA);
            break;
        }
        case CV_8UC3: {
            cvtColor(inImg, img, COLOR_RGB2RGBA);
            break;
        }
        case CV_8UC4: {
            inImg.copyTo(img);
            break;
        }
        default:{
            CV_Error(Error::StsError,
                     "Only uchar images of 1, 3, or 4 channels are accepted");
        }
        }
        this->availableBgSampleImages_.push_back(img);
    }

    void addFontFiles(const std::vector<cv::String>& fntList){
#ifdef HAVE_QT5GUI
        for(size_t n=0;n<fntList.size();n++){
            int addFontSucces=this->fntDb_->addApplicationFont(fntList[n].c_str());
            if(addFontSucces==-1){
                CV_Error(Error::StsError,"Failed to load ttf font. QT5 currently doesn't support this under X11");
            }
        }
        this->updateFontNameList(this->availableFonts_);
#else
        CV_Assert(fntList.size()>0);//to supress compilation warning
        CV_Error(Error::StsError,"QT5 not available, TextSynthesiser is not fully functional.");
#endif
    }

    std::vector<String> listBgSampleFiles(){
        std::vector<String> res(this->availableBgSampleFiles_.size());
        std::copy(this->availableBgSampleFiles_.begin(),this->availableBgSampleFiles_.end(),res.begin());
        return res;
    }
};

Ptr<TextSynthesizer> TextSynthesizer::create(int sampleHeight, int maxWidth, int script){
    Ptr<TextSynthesizer> res(new TextSynthesizerQtImpl(script, maxWidth,sampleHeight));
    return res;
}

}  //namespace text
}  //namespace cv
