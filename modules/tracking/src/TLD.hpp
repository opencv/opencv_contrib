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

namespace cv
{

//debug functions and variables
#define MEASURE_TIME(a) {\
    clock_t start;float milisec=0.0;\
    start=clock();{a} milisec=1000.0*(clock()-start)/CLOCKS_PER_SEC;\
    printf("%-90s took %f milis\n",#a,milisec); }
#define START_TICK(name) { clock_t start;float milisec=0.0; start=clock();
#define END_TICK(name) milisec=1000.0*(clock()-start)/CLOCKS_PER_SEC;\
    printf("%s took %f milis\n",name,milisec); }
extern Rect2d etalon;
void myassert(const Mat& img);
void printPatch(const Mat_<uchar>& standardPatch);
std::string type2str(const Mat& mat);
void drawWithRects(const Mat& img,std::vector<Rect2d>& blackOnes,Rect2d whiteOne=Rect2d(-1.0,-1.0,-1.0,-1.0));
void drawWithRects(const Mat& img,std::vector<Rect2d>& blackOnes,std::vector<Rect2d>& whiteOnes);

//aux functions and variables
#define CLIP(x,a,b) MIN(MAX((x),(a)),(b))
inline double overlap(const Rect2d& r1,const Rect2d& r2);
void resample(const Mat& img,const RotatedRect& r2,Mat_<uchar>& samples);
void resample(const Mat& img,const Rect2d& r2,Mat_<uchar>& samples);
double variance(const Mat& img);
double variance(Mat_<unsigned int>& intImgP,Mat_<unsigned int>& intImgP2,Rect box);
double NCC(Mat_<uchar> patch1,Mat_<uchar> patch2);
void getClosestN(std::vector<Rect2d>& scanGrid,Rect2d bBox,int n,std::vector<Rect2d>& res);
double scaleAndBlur(const Mat& originalImg,int scale,Mat& scaledImg,Mat& blurredImg,Size GaussBlurKernelSize);

class TLDEnsembleClassifier{
public:
    TLDEnsembleClassifier(int ordinal,Size size);
    void integrate(Mat_<uchar> patch,bool isPositive);
    double posteriorProbability(const uchar* data,int rowstep)const;
    static int getMaxOrdinal();
private:
    static int getGridSize();
    inline void stepPrefSuff(uchar* arr,int len);
    void preinit(int ordinal);
    unsigned short int code(const uchar* data,int rowstep)const;
    unsigned int pos[8192],neg[8192];//8192=2^13
    uchar x1[13],y1[13],x2[13],y2[13];
};

class TrackerProxy : public TrackerTLD::Private{
public:
    virtual bool init( const Mat& image, const Rect2d& boundingBox)=0;
    virtual bool update(const Mat& image, Rect2d& boundingBox)=0;
    virtual ~TrackerProxy(){}
};

}
