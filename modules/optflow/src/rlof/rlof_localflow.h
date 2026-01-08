// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _RLOF_LOCALFLOW_H_
#define _RLOF_LOCALFLOW_H_
#include <limits>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow/rlofflow.hpp"
//! Fast median estimation method based on @cite Tibshirani2008. This implementation relates to http://www.stat.cmu.edu/~ryantibs/median/
using namespace cv;
template<typename T>
T quickselect(const Mat & inp, int k)
{
    unsigned long i;
    unsigned long ir;
    unsigned long j;
    unsigned long l;
    unsigned long mid;
    Mat values = inp.clone();
    T a;

    l = 0;
    ir = MAX(values.rows, values.cols) - 1;
    while(true)
    {
        if (ir <= l + 1)
        {
            if (ir == l + 1 && values.at<T>(ir) < values.at<T>(l))
                std::swap(values.at<T>(l), values.at<T>(ir));
            return values.at<T>(k);
        }
        else
        {
            mid = (l + ir) >> 1;
            std::swap(values.at<T>(mid), values.at<T>(l+1));
            if (values.at<T>(l) > values.at<T>(ir))
                std::swap(values.at<T>(l), values.at<T>(ir));
            if (values.at<T>(l+1) > values.at<T>(ir))
                std::swap(values.at<T>(l+1), values.at<T>(ir));
            if (values.at<T>(l) > values.at<T>(l+1))
                std::swap(values.at<T>(l), values.at<T>(l+1));
            i = l + 1;
            j = ir;
            a = values.at<T>(l+1);
            while (true)
            {
                do
                {
                    i++;
                }
                while (values.at<T>(i) < a);
                do
                {
                    j--;
                }
                while (values.at<T>(j) > a);
                if (j < i) break;
                std::swap(values.at<T>(i), values.at<T>(j));
            }
            values.at<T>(l+1) = values.at<T>(j);
            values.at<T>(j) = a;
            if (j >= static_cast<unsigned long>(k)) ir = j - 1;
            if (j <= static_cast<unsigned long>(k)) l = i;
        }
    }
}

namespace cv {
namespace optflow {

class CImageBuffer
{
public:
    CImageBuffer()
        : m_Overwrite(true)
    {}
    void setGrayFromRGB(const cv::Mat & inp)
    {
        if(m_Overwrite)
            cv::cvtColor(inp, m_Image, cv::COLOR_BGR2GRAY);
    }
    void setImage(const cv::Mat & inp)
    {
        if(m_Overwrite)
            inp.copyTo(m_Image);
    }
    void setBlurFromRGB(const cv::Mat & inp)
    {
        if(m_Overwrite)
            cv::GaussianBlur(inp, m_BlurredImage, cv::Size(7,7), -1);
    }

    int buildPyramid(cv::Size winSize, int maxLevel, float levelScale[2], bool withBlurredImage = false);
    cv::Mat & getImage(int level) {return m_ImagePyramid[level];}

    std::vector<cv::Mat>     m_ImagePyramid;
    cv::Mat                  m_BlurredImage;
    cv::Mat                  m_Image;
    std::vector<cv::Mat>     m_CrossPyramid;
    int                      m_maxLevel;
    bool                     m_Overwrite;
};

void calcLocalOpticalFlow(
    const Mat prevImage,
    const Mat currImage,
    Ptr<CImageBuffer>  prevPyramids[2],
    Ptr<CImageBuffer>  currPyramids[2],
    const std::vector<Point2f> & prevPoints,
    std::vector<Point2f> & currPoints,
    const RLOFOpticalFlowParameter & param);

}} // namespace
#endif
