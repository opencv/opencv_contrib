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

#ifndef OPENCV_TRACKER_CSRT_SEGMENTATION
#define OPENCV_TRACKER_CSRT_SEGMENTATION
#include "precomp.hpp"

namespace cv
{
class Histogram
{
public:
    int m_numBinsPerDim;
    int m_numDim;

    Histogram() : m_numBinsPerDim(0), m_numDim(0) {}
    Histogram(int numDimensions, int numBinsPerDimension = 8);
    void extractForegroundHistogram(std::vector<cv::Mat> & imgChannels,
            cv::Mat weights, bool useMatWeights, int x1, int y1, int x2, int y2);
    void extractBackGroundHistogram(std::vector<cv::Mat> & imgChannels,
            int x1, int y1, int x2, int y2, int outer_x1, int outer_y1,
            int outer_x2, int outer_y2);
    cv::Mat backProject(std::vector<cv::Mat> & imgChannels);
    std::vector<double> getHistogramVector();
    void setHistogramVector(double *vector);

private:
    int p_size;
    std::vector<double> p_bins;
    std::vector<int> p_dimIdCoef;

    inline double kernelProfile_Epanechnikov(double x)
        { return (x <= 1) ? (2.0/CV_PI)*(1-x) : 0; }
};


class Segment
{
public:
    static std::pair<cv::Mat, cv::Mat> computePosteriors(std::vector<cv::Mat> & imgChannels,
            int x1, int y1, int x2, int y2, cv::Mat weights, cv::Mat fgPrior,
            cv::Mat bgPrior, const Histogram &fgHistPrior, int numBinsPerChannel = 16);
    static std::pair<cv::Mat, cv::Mat> computePosteriors2(std::vector<cv::Mat> & imgChannels,
            int x1, int y1, int x2, int y2, double p_b, cv::Mat weights, cv::Mat
            fgPrior, cv::Mat bgPrior, Histogram hist_target, Histogram hist_background);
    static std::pair<cv::Mat, cv::Mat> computePosteriors2(std::vector<cv::Mat> &imgChannels,
            cv::Mat fgPrior, cv::Mat bgPrior,
            Histogram hist_target, Histogram hist_background, int numBinsPerChannel);

private:
    static std::pair<cv::Mat, cv::Mat> getRegularizedSegmentation(cv::Mat & prob_o,
            cv::Mat & prob_b, cv::Mat &prior_o, cv::Mat &prior_b);

    inline static double gaussian(double x2, double y2, double std2){
        return exp(-(x2 + y2)/(2*std2))/(2*CV_PI*std2);
    }

    double p_b;
    double p_o;

};

}//cv namespace
#endif
