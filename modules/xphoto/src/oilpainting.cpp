// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/xphoto.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

template<class T>
class Vec3fTo {
public :
    cv::Vec3f a;
    Vec3fTo(cv::Vec3f x) {
        a = x;
    };
    T extract();
    cv::Vec3f make(int);
};

template<>
uchar Vec3fTo<uchar>::extract()
{
    return static_cast<uchar>(a[0]);
}

template<>
cv::Vec3b Vec3fTo<cv::Vec3b>::extract()
{
    return a;
}

template<>
cv::Vec3f Vec3fTo<uchar>::make(int x)
{
    return cv::Vec3f((a*x)/x);
}

template<>
cv::Vec3f Vec3fTo<cv::Vec3b>::make(int x)
{
    return cv::Vec3f(static_cast<float>(static_cast<int>(a[0]*x)/x),
        static_cast<float>(static_cast<int>(a[1] * x) / x),
        static_cast<float>(static_cast<int>(a[2] * x) / x));
}

namespace cv
{
namespace xphoto
{
template<typename Type>
class ParallelOilPainting : public ParallelLoopBody
{
private:
    Mat & imgSrc;
    Mat &dst;
    Mat &imgLuminance;
    int halfsize;
    int dynRatio;

public:
    ParallelOilPainting<Type>(Mat& img, Mat &d, Mat &iLuminance, int r,int k) :
        imgSrc(img),
        dst(d),
        imgLuminance(iLuminance),
        halfsize(r),
        dynRatio(k)
    {}
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        std::vector<int> histogram(256);
        std::vector<Vec3f> meanBGR(256);

        for (int y = range.start; y < range.end; y++)
        {
            Type *vDst = dst.ptr<Type>(y);
            for (int x = 0; x < imgSrc.cols; x++, vDst++)
            {
                if (x == 0)
                {
                    histogram.assign(256, 0);
                    meanBGR.assign(256, Vec3f(0,0,0));
                    for (int yy = -halfsize; yy <= halfsize; yy++)
                    {
                        if (y + yy >= 0 && y + yy < imgSrc.rows)
                        {
                            Type *vPtr = imgSrc.ptr<Type>(y + yy) + x - 0;
                            uchar *uc = imgLuminance.ptr(y + yy) + x - 0;
                            for (int xx = 0; xx <= halfsize; xx++, vPtr++, uc++)
                            {
                                if (x + xx >= 0 && x + xx < imgSrc.cols)
                                {
                                    histogram[*uc]++;
                                    meanBGR[*uc] += Vec3fTo<Type>(*vPtr).make(dynRatio);
                                }
                            }
                        }
                    }

                }
                else
                {
                    for (int yy = -halfsize; yy <= halfsize; yy++)
                    {
                        if (y + yy >= 0 && y + yy < imgSrc.rows)
                        {
                            Type *vPtr = imgSrc.ptr<Type>(y + yy) + x - halfsize - 1;
                            uchar *uc = imgLuminance.ptr(y + yy) + x - halfsize - 1;
                            int xx = -halfsize - 1;
                            if (x + xx >= 0 && x + xx < imgSrc.cols)
                            {
                                histogram[*uc]--;
                                meanBGR[*uc] -= Vec3fTo<Type>(*vPtr).make(dynRatio);
                            }
                            vPtr = imgSrc.ptr<Type>(y + yy) + x + halfsize;
                            uc = imgLuminance.ptr(y + yy) + x + halfsize;
                            xx = halfsize;
                            if (x + xx >= 0 && x + xx < imgSrc.cols)
                            {
                                histogram[*uc]++;
                                meanBGR[*uc] += Vec3fTo<Type>(*vPtr).make(dynRatio);
                            }
                        }
                    }
                }
                auto pos = distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end()));
                *vDst = Vec3fTo<Type>(meanBGR[pos] / histogram[pos]).extract();
            }
        }
    }
};

void oilPainting(InputArray src, OutputArray dst, int size, int dynValue)
{
    oilPainting(src, dst, size, dynValue, COLOR_BGR2GRAY);
}

void oilPainting(InputArray _src, OutputArray _dst, int size, int dynValue,int code)
{
    CV_CheckType(_src.type(), _src.type() == CV_8UC1 || _src.type() == CV_8UC3, "only 1 or 3 channels (CV_8UC)");
    CV_Assert(_src.kind() == _InputArray::MAT);
    CV_Assert(size >= 1);
    CV_CheckGT(dynValue , 0,"dynValue must be  0");
    CV_CheckLT(dynValue, 128, "dynValue must less than 128 ");
    Mat src = _src.getMat();
    Mat lum,dst(_src.size(),_src.type());
    if (src.type() == CV_8UC3)
    {
        cvtColor(_src, lum, code);
        if (lum.channels() > 1)
        {
            extractChannel(lum, lum, 0);
        }
    }
    else
        lum = src.clone();
    double dratio = 1 / double(dynValue);
    lum.forEach<uchar>([=](uchar &pixel, const int * /*position*/) { pixel = saturate_cast<uchar>(cvRound(pixel * dratio)); });
    if (_src.type() == CV_8UC1)
    {
        ParallelOilPainting<uchar> oilAlgo(src, dst, lum, size, dynValue);
        parallel_for_(Range(0, src.rows), oilAlgo);
    }
    else
    {
        ParallelOilPainting<Vec3b> oilAlgo(src, dst, lum, size, dynValue);
        parallel_for_(Range(0, src.rows), oilAlgo);
    }
    dst.copyTo(_dst);
    dst = (dst  / dynValue) * dynValue;
}
}
}
