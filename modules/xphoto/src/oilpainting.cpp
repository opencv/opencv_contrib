/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "opencv2/xphoto.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{
namespace xphoto
{
    class ParallelOilPainting : public ParallelLoopBody
    {
    private:
        Mat & imgSrc;
        Mat &dst;
        Mat &imgLuminance;
        int halfsize;

    public:
        ParallelOilPainting(Mat& img, Mat &d, Mat &iLuminance, int r) :
            imgSrc(img),
            dst(d),
            imgLuminance(iLuminance),
            halfsize(r)
        {}
        virtual void operator()(const Range& range) const CV_OVERRIDE
        {
            std::vector<int> histogram(256);
            std::vector<Vec3f> meanBGR(256);

            for (int y = range.start; y < range.end; y++)
            {
                Vec3b *vDst = (Vec3b *)dst.ptr(y);
                for (int x = 0; x < imgSrc.cols; x++, vDst++)
                {
                    if (x == 0)
                    {
                        histogram.assign(256, 0);
                        meanBGR.assign(256, Vec3f(0, 0, 0));
                        for (int yy = -halfsize; yy <= halfsize; yy++)
                        {
                            if (y + yy >= 0 && y + yy < imgSrc.rows)
                            {
                                Vec3b *vPtr = (Vec3b *)imgSrc.ptr(y + yy) + x - 0;
                                uchar *uc = imgLuminance.ptr(y + yy) + x - 0;
                                for (int xx = 0; xx <= halfsize; xx++, vPtr++, uc++)
                                {
                                    if (x + xx >= 0 && x + xx < imgSrc.cols)
                                    {
                                        histogram[*uc]++;
                                        meanBGR[*uc] += *vPtr;
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
                                Vec3b *vPtr = (Vec3b *)imgSrc.ptr(y + yy) + x - halfsize - 1;
                                uchar *uc = imgLuminance.ptr(y + yy) + x - halfsize - 1;
                                int xx = -halfsize - 1;
                                if (x + xx >= 0 && x + xx < imgSrc.cols)
                                {
                                    histogram[*uc]--;
                                    meanBGR[*uc] -= *vPtr;
                                }
                                vPtr = (Vec3b *)imgSrc.ptr(y + yy) + x + halfsize;
                                uc = imgLuminance.ptr(y + yy) + x + halfsize;
                                xx = halfsize;
                                if (x + xx >= 0 && x + xx < imgSrc.cols)
                                {
                                    histogram[*uc]++;
                                    meanBGR[*uc] += *vPtr;
                                }
                            }
                        }
                    }
                    int64 pos = distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end()));
                    *vDst = meanBGR[pos] / histogram[pos];
                }
            }
        }
    };

    class Quotient {
        uchar q;
    public:
        Quotient(int v) { q = static_cast<uchar>(v); };
        void operator ()(uchar &pixel, const int * ) const {
            pixel = pixel / q;
        }
    };

    void oilPainting(InputArray _src, OutputArray _dst, int size, int dynValue)
    {
        oilPainting(_src, _dst, size, dynValue, COLOR_BGR2GRAY);
    }

    void oilPainting(InputArray _src, OutputArray _dst, int size, int dynValue,int code)
    {
        CV_Assert(_src.kind() == _InputArray::MAT && size>=1 && dynValue>0 && dynValue<128);
        Mat src = _src.getMat();
        CV_Assert(src.type()==CV_8UC1 || src.type() == CV_8UC3);
        Mat lum,dst(src.size(),src.type());
        if (src.type() == CV_8UC3)
        {
            cvtColor(src, lum, code);
            if (lum.channels() > 1)
            {
                extractChannel(lum, lum, 0);
            }
        }
        else
            lum = src.clone();
        lum.forEach<uchar>(Quotient(dynValue));
        ParallelOilPainting oilAlgo(src, dst, lum, size);
        parallel_for_(Range(0,src.rows), oilAlgo, getNumThreads());
        dst.copyTo(_dst);
        dst = (dst  / dynValue) * dynValue;
    }
}
}
