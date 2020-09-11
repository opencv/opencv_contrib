// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#ifndef __OPENCV_MCC_LINEARIZE_HPP__
#define __OPENCV_MCC_LINEARIZE_HPP__

#include "opencv2/mcc/color.hpp"

namespace cv
{
namespace ccm
{

/* *\ brief Enum of the possible types of linearization.
*/
enum LINEAR_TYPE
{
    IDENTITY_,
    GAMMA,
    COLORPOLYFIT,
    COLORLOGPOLYFIT,
    GRAYPOLYFIT,
    GRAYLOGPOLYFIT
};

/* *\ brief Polyfit model.
*/
class Polyfit
{
public:
    int deg;
    Mat p;

    Polyfit() {};

    /* *\ brief Polyfit method.
    https://en.wikipedia.org/wiki/Polynomial_regression
    polynomial: yi = a0 + a1*xi + a2*xi^2 + ... + an*xi^deg (i = 1,2,...,n)
    and deduct: Ax = y
    See linear.pdf for details
    */
    Polyfit(Mat x, Mat y, int deg_) :deg(deg_)
    {
        int n = x.cols * x.rows * x.channels();
        x = x.reshape(1, n);
        y = y.reshape(1, n);
        Mat_<double> A = Mat_<double>::ones(n, deg + 1);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 1; j < A.cols; ++j)
            {
                A.at<double>(i, j) = x.at<double>(i) * A.at<double>(i, j - 1);
            }
        }
        Mat y_(y);
        cv::solve(A, y_, p, DECOMP_SVD);
    }

    virtual ~Polyfit() {};

    Mat operator()(const Mat& inp)
    {
        return elementWise(inp, [this](double x)->double {return fromEW(x); });
    };

private:
    double fromEW(double x)
    {
        double res = 0;
        for (int d = 0; d <= deg; ++d)
        {
            res += pow(x, d) * p.at<double>(d, 0);
        }
        return res;
    };
};

/* *\ brief Logpolyfit model.
*/
class LogPolyfit
{
public:
    int deg;
    Polyfit p;

    LogPolyfit() {};

    /* *\ brief Logpolyfit method.
    */
    LogPolyfit(Mat x, Mat y, int deg_) :deg(deg_)
    {
        Mat mask_ = (x > 0) & (y > 0);
        Mat src_, dst_, s_, d_;
        src_ = maskCopyTo(x, mask_);
        dst_ = maskCopyTo(y, mask_);
        log(src_, s_);
        log(dst_, d_);
        p = Polyfit(s_, d_, deg);
    }

    virtual ~LogPolyfit() {};

    Mat operator()(const Mat& inp)
    {
        Mat mask_ = inp >= 0;
        Mat y, y_, res;
        log(inp, y);
        y = p(y);
        exp(y, y_);
        y_.copyTo(res, mask_);
        return res;
    };
};

/* *\ brief Linearization base.
*/
class Linear
{
public:
    Linear() {};

    virtual ~Linear() {};

    /* *\ brief Inference.
       *\ param inp the input array, type of cv::Mat.
    */
    virtual Mat linearize(Mat inp)
    {
        return inp;
    };

    /* *\brief Evaluate linearization model.
    */
    virtual void value(void) {};
};


/* *\ brief Linearization identity.
   *        make no change.
*/
class LinearIdentity : public Linear {};

/* *\ brief Linearization gamma correction.
*/
class LinearGamma : public Linear
{
public:
    double gamma;

    LinearGamma(double gamma_) :gamma(gamma_) {};

    Mat linearize(Mat inp) CV_OVERRIDE
    {
        return gammaCorrection(inp, gamma);
    };
};

/* *\ brief Linearization.
   *        Grayscale polynomial fitting.
*/
template <class T>
class LinearGray :public Linear
{
public:
    int deg;
    T p;

    LinearGray(int deg_, Mat src, Color dst, Mat mask, RGBBase_ cs) :deg(deg_)
    {
        dst.getGray();
        Mat lear_gray_mask = mask & dst.grays;

        // the grayscale function is approximate for src is in relative color space.
        src = rgb2gray(maskCopyTo(src, lear_gray_mask));
        Mat dst_ = maskCopyTo(dst.toGray(cs.io), lear_gray_mask);
        calc(src, dst_);
    }

    /* *\ brief monotonically increase is not guaranteed.
       *\ param src the input array, type of cv::Mat.
       *\ param dst the input array, type of cv::Mat.
    */
    void calc(const Mat& src, const Mat& dst)
    {
        p = T(src, dst, deg);
    };

    Mat linearize(Mat inp) CV_OVERRIDE
    {
        return p(inp);
    };
};

/* *\ brief Linearization.
   *        Fitting channels respectively.
*/
template <class T>
class LinearColor :public Linear
{
public:
    int deg;
    T pr;
    T pg;
    T pb;

    LinearColor(int deg_, Mat src_, Color dst, Mat mask, RGBBase_ cs) :deg(deg_)
    {
        Mat src = maskCopyTo(src_, mask);
        Mat dst_ = maskCopyTo(dst.to(*cs.l).colors, mask);
        calc(src, dst_);
    }

    void calc(const Mat& src, const Mat& dst)
    {
        Mat schannels[3];
        Mat dchannels[3];
        split(src, schannels);
        split(dst, dchannels);
        pr = T(schannels[0], dchannels[0], deg);
        pg = T(schannels[1], dchannels[1], deg);
        pb = T(schannels[2], dchannels[2], deg);
    };

    Mat linearize(Mat inp) CV_OVERRIDE
    {
        Mat channels[3];
        split(inp, channels);
        std::vector<Mat> channel;
        Mat res;
        merge(std::vector<Mat>{ pr(channels[0]), pg(channels[1]), pb(channels[2]) }, res);
        return res;
    };
};

/* *\ brief Get linearization method.
   *        used in ccm model.
   *\ param gamma used in LinearGamma.
   *\ param deg degrees.
   *\ param src the input array, type of cv::Mat.
   *\ param dst the input array, type of cv::Mat.
   *\ param mask the input array, type of cv::Mat.
   *\ param cs type of RGBBase_.
   *\ param linear_type type of linear.
*/
std::shared_ptr<Linear>  getLinear(double gamma, int deg, Mat src, Color dst, Mat mask, RGBBase_ cs, LINEAR_TYPE linear_type);
std::shared_ptr<Linear>  getLinear(double gamma, int deg, Mat src, Color dst, Mat mask, RGBBase_ cs, LINEAR_TYPE linear_type)
{
    std::shared_ptr<Linear> p = std::make_shared<Linear>();
    switch (linear_type)
    {
    case cv::ccm::IDENTITY_:
        p.reset(new LinearIdentity());
        break;
    case cv::ccm::GAMMA:
        p.reset(new LinearGamma(gamma));
        break;
    case cv::ccm::COLORPOLYFIT:
        p.reset(new LinearColor<Polyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::COLORLOGPOLYFIT:
        p.reset(new LinearColor<LogPolyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::GRAYPOLYFIT:
        p.reset(new LinearGray<Polyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::GRAYLOGPOLYFIT:
        p.reset(new LinearGray<LogPolyfit>(deg, src, dst, mask, cs));
        break;
    default:
        throw std::invalid_argument{ "Wrong linear_type!" };
        break;
    }
    return p;
};

} // namespace ccm
} // namespace cv


#endif