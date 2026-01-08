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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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

#include <algorithm>
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xphoto.hpp"

namespace cv
{
namespace xphoto
{

template <typename T>
void balanceWhiteSimple(std::vector<Mat_<T> > &src, Mat &dst, const float inputMin, const float inputMax,
                        const float outputMin, const float outputMax, const float p)
{
    /********************* Simple white balance *********************/
    const float s1 = p; // low quantile
    const float s2 = p; // high quantile

    int nElements = src[0].depth() == CV_8U ? 256 : 4096;

    float minValue0 = inputMin;
    float maxValue0 = inputMax;

    // deal with cv::calcHist (exclusive upper bound)
    if (src[0].depth() == CV_32F || src[0].depth() == CV_64F) // floating
    {
        maxValue0 += MIN((inputMax - inputMin) / (nElements - 1), 1);
        if (inputMax == inputMin) // single value
            maxValue0 += 1;
    }
    else // integer
    {
        maxValue0 += 1;
    }

    float interval = (maxValue0 - minValue0) / float(nElements);

    for (size_t i = 0; i < src.size(); ++i)
    {
        float minValue = minValue0;
        float maxValue = maxValue0;

        Mat img = src[i].reshape(1);
        Mat hist;
        int channels[] = {0};
        int histSize[] = {nElements};
        float inputRange[] = {minValue, maxValue};
        const float *ranges[] = {inputRange};

        calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

        int total = int(src[i].total());

        int p1 = 0, p2 = nElements - 1;
        int n1 = 0, n2 = total;

        // searching for s1 and s2
        while (n1 + hist.at<float>(p1) < s1 * total / 100.0f)
        {
            n1 += saturate_cast<int>(hist.at<float>(p1++));
            minValue += interval;
        }

        while (n2 - hist.at<float>(p2) > (100.0f - s2) * total / 100.0f)
        {
            n2 -= saturate_cast<int>(hist.at<float>(p2--));
            maxValue -= interval;
        }

        src[i] = (outputMax - outputMin) * (src[i] - minValue) / (maxValue - minValue) + outputMin;
    }
    /****************************************************************/

    dst.create(/**/ src[0].size(), CV_MAKETYPE(src[0].depth(), int(src.size())) /**/);
    cv::merge(src, dst);
}

class SimpleWBImpl CV_FINAL : public SimpleWB
{
  private:
    float inputMin, inputMax, outputMin, outputMax, p;

  public:
    SimpleWBImpl()
    {
        inputMin = 0.0f;
        inputMax = 255.0f;
        outputMin = 0.0f;
        outputMax = 255.0f;
        p = 2.0f;
    }

    float getInputMin() const CV_OVERRIDE { return inputMin; }
    void setInputMin(float val) CV_OVERRIDE { inputMin = val; }

    float getInputMax() const CV_OVERRIDE { return inputMax; }
    void setInputMax(float val) CV_OVERRIDE { inputMax = val; }

    float getOutputMin() const CV_OVERRIDE { return outputMin; }
    void setOutputMin(float val) CV_OVERRIDE { outputMin = val; }

    float getOutputMax() const CV_OVERRIDE { return outputMax; }
    void setOutputMax(float val) CV_OVERRIDE { outputMax = val; }

    float getP() const CV_OVERRIDE { return p; }
    void setP(float val) CV_OVERRIDE { p = val; }

    void balanceWhite(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_Assert(!_src.empty());
        CV_Assert(_src.depth() == CV_8U || _src.depth() == CV_16S || _src.depth() == CV_32S || _src.depth() == CV_32F);
        Mat src = _src.getMat();
        Mat &dst = _dst.getMatRef();

        switch (src.depth())
        {
        case CV_8U:
        {
            std::vector<Mat_<uchar> > mv;
            split(src, mv);
            balanceWhiteSimple(mv, dst, inputMin, inputMax, outputMin, outputMax, p);
            break;
        }
        case CV_16S:
        {
            std::vector<Mat_<short> > mv;
            split(src, mv);
            balanceWhiteSimple(mv, dst, inputMin, inputMax, outputMin, outputMax, p);
            break;
        }
        case CV_32S:
        {
            std::vector<Mat_<int> > mv;
            split(src, mv);
            balanceWhiteSimple(mv, dst, inputMin, inputMax, outputMin, outputMax, p);
            break;
        }
        case CV_32F:
        {
            std::vector<Mat_<float> > mv;
            split(src, mv);
            balanceWhiteSimple(mv, dst, inputMin, inputMax, outputMin, outputMax, p);
            break;
        }
        }
    }
};

Ptr<SimpleWB> createSimpleWB() { return makePtr<SimpleWBImpl>(); }
}
}
