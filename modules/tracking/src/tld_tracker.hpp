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
#include<algorithm>
#include<limits.h>

namespace cv {namespace tld
{

//debug functions and variables
#define ALEX_DEBUG
#ifdef ALEX_DEBUG
#define dfprintf(x) fprintf x
#define dprintf(x) printf x
#else
#define dfprintf(x)
#define dprintf(x)
#endif
#define MEASURE_TIME(a)\
{\
    clock_t start; float milisec = 0.0; \
    start = clock(); {a} milisec = 1000.0 * (clock() - start) / CLOCKS_PER_SEC; \
    dprintf(("%-90s took %f milis\n", #a, milisec));\
}
#define HERE dprintf(("line %d\n", __LINE__)); fflush(stderr);
#define START_TICK(name)\
{ \
    clock_t start; double milisec = 0.0; start = clock();
#define END_TICK(name) milisec = 1000.0 * (clock() - start) / CLOCKS_PER_SEC; \
    dprintf(("%s took %f milis\n", name, milisec)); \
}
extern Rect2d etalon;
void myassert(const Mat& img);
void printPatch(const Mat_<uchar>& standardPatch);
std::string type2str(const Mat& mat);
void drawWithRects(const Mat& img, std::vector<Rect2d>& blackOnes, Rect2d whiteOne = Rect2d(-1.0, -1.0, -1.0, -1.0));
void drawWithRects(const Mat& img, std::vector<Rect2d>& blackOnes, std::vector<Rect2d>& whiteOnes, String fileName = "");

//aux functions and variables
template<typename T> inline T CLIP(T x, T a, T b){ return std::min(std::max(x, a), b); }
/** Computes overlap between the two given rectangles. Overlap is computed as ratio of rectangles' intersection to that
 * of their union.*/
double overlap(const Rect2d& r1, const Rect2d& r2);
/** Resamples the area surrounded by r2 in img so it matches the size of samples, where it is written.*/
void resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples);
/** Specialization of resample() for rectangles without retation for better performance and simplicity.*/
void resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples);
/** Computes the variance of single given image.*/
double variance(const Mat& img);
/** Computes normalized corellation coefficient between the two patches (they should be
 * of the same size).*/
double NCC(const Mat_<uchar>& patch1, const Mat_<uchar>& patch2);
void getClosestN(std::vector<Rect2d>& scanGrid, Rect2d bBox, int n, std::vector<Rect2d>& res);
double scaleAndBlur(const Mat& originalImg, int scale, Mat& scaledImg, Mat& blurredImg, Size GaussBlurKernelSize, double scaleStep);
int getMedian(const std::vector<int>& values, int size = -1);

class TLDEnsembleClassifier
{
public:
    static int makeClassifiers(Size size, int measurePerClassifier, int gridSize, std::vector<TLDEnsembleClassifier>& classifiers);
    void integrate(const Mat_<uchar>& patch, bool isPositive);
    double posteriorProbability(const uchar* data, int rowstep) const;
    double posteriorProbabilityFast(const uchar* data) const;
    void prepareClassifier(int rowstep);
private:
    TLDEnsembleClassifier(const std::vector<Vec4b>& meas, int beg, int end);
    static void stepPrefSuff(std::vector<Vec4b> & arr, int pos, int len, int gridSize);
    int code(const uchar* data, int rowstep) const;
    int codeFast(const uchar* data) const;
    std::vector<Point2i> posAndNeg;
    std::vector<Vec4b> measurements;
    std::vector<Point2i> offset;
    int lastStep_;
};

class TrackerProxy
{
public:
    virtual bool init(const Mat& image, const Rect2d& boundingBox) = 0;
    virtual bool update(const Mat& image, Rect2d& boundingBox) = 0;
    virtual ~TrackerProxy(){}
};

}}
