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
#include <iterator>
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

    int depth = 2; // depth of histogram tree
    if (src[0].depth() != CV_8U)
        ++depth;
    int bins = 16; // number of bins at each histogram level

    int nElements = int(pow((float)bins, (float)depth));
    // number of elements in histogram tree

    for (size_t i = 0; i < src.size(); ++i)
    {
        std::vector<int> hist(nElements, 0);

        typename Mat_<T>::iterator beginIt = src[i].begin();
        typename Mat_<T>::iterator endIt = src[i].end();

        for (typename Mat_<T>::iterator it = beginIt; it != endIt; ++it)
        // histogram filling
        {
            int pos = 0;
            float minValue = inputMin - 0.5f;
            float maxValue = inputMax + 0.5f;
            T val = *it;

            float interval = float(maxValue - minValue) / bins;

            for (int j = 0; j < depth; ++j)
            {
                int currentBin = int((val - minValue + 1e-4f) / interval);
                ++hist[pos + currentBin];

                pos = (pos + currentBin) * bins;

                minValue = minValue + currentBin * interval;
                maxValue = minValue + interval;

                interval /= bins;
            }
        }

        int total = int(src[i].total());

        int p1 = 0, p2 = bins - 1;
        int n1 = 0, n2 = total;

        float minValue = inputMin - 0.5f;
        float maxValue = inputMax + 0.5f;

        float interval = (maxValue - minValue) / float(bins);

        for (int j = 0; j < depth; ++j)
        // searching for s1 and s2
        {
            while (n1 + hist[p1] < s1 * total / 100.0f)
            {
                n1 += hist[p1++];
                minValue += interval;
            }
            p1 *= bins;

            while (n2 - hist[p2] > (100.0f - s2) * total / 100.0f)
            {
                n2 -= hist[p2--];
                maxValue -= interval;
            }
            p2 = (p2 + 1) * bins - 1;

            interval /= bins;
        }

        src[i] = (outputMax - outputMin) * (src[i] - minValue) / (maxValue - minValue) + outputMin;
    }
    /****************************************************************/

    dst.create(/**/ src[0].size(), CV_MAKETYPE(src[0].depth(), int(src.size())) /**/);
    cv::merge(src, dst);
}

class SimpleWBImpl : public SimpleWB
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

    float getInputMin() const { return inputMin; }
    void setInputMin(float val) { inputMin = val; }

    float getInputMax() const { return inputMax; }
    void setInputMax(float val) { inputMax = val; }

    float getOutputMin() const { return outputMin; }
    void setOutputMin(float val) { outputMin = val; }

    float getOutputMax() const { return outputMax; }
    void setOutputMax(float val) { outputMax = val; }

    float getP() const { return p; }
    void setP(float val) { p = val; }

    void balanceWhite(InputArray _src, OutputArray _dst)
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
