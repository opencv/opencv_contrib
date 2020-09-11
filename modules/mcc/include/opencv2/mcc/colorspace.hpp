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

#ifndef __OPENCV_MCC_COLORSPACE_HPP__
#define __OPENCV_MCC_COLORSPACE_HPP__

#include <vector>
#include <string>
#include <iostream>
#include "opencv2/mcc/io.hpp"
#include "opencv2/mcc/operations.hpp"
#include "opencv2/mcc/utils.hpp"

namespace cv
{
namespace ccm
{
/* *\ brief Basic class for ColorSpace.
*/
class ColorSpace
{
public:
    IO io;
    std::string type;
    bool linear;
    Operations to;
    Operations from;
    ColorSpace* l;
    ColorSpace* nl;

    ColorSpace() {};

    ColorSpace(IO io_, std::string type_, bool linear_) :io(io_), type(type_), linear(linear_) {};

    virtual ~ColorSpace()
    {
        l = 0;
        nl = 0;
    };

    virtual bool relate(const ColorSpace& other) const
    {
        return (type == other.type) && (io == other.io);
    };

    virtual Operations relation(const ColorSpace& /*other*/) const
    {
        return IDENTITY_OPS;
    };

    bool operator<(const ColorSpace& other)const
    {
        return (io < other.io || (io == other.io && type < other.type) || (io == other.io && type == other.type && linear < other.linear));
    }
};

/* *\ brief Base of RGB color space;
   *        the argument values are from AdobeRGB;
   *        Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
*/
class RGBBase_ : public ColorSpace
{
public:
    // primaries
    double xr;
    double yr;
    double xg;
    double yg;
    double xb;
    double yb;
    MatFunc toL;
    MatFunc fromL;
    Mat M_to;
    Mat M_from;

    using ColorSpace::ColorSpace;

    /* *\ brief There are 3 kinds of relationships for RGB:
       *        1. Different types;    - no operation
       *        1. Same type, same linear; - copy
       *        2. Same type, different linear, self is nonlinear; - 2 toL
       *        3. Same type, different linear, self is linear - 3 fromL
       *\ param other type of ColorSpace.
       *\ return Operations.
    */
    Operations relation(const ColorSpace& other) const CV_OVERRIDE
    {
        if (linear == other.linear)
        {
            return IDENTITY_OPS;
        }
        if (linear)
        {
            return Operations({ Operation(fromL) });
        }
        return Operations({ Operation(toL) });
    };

    /* *\ brief Initial operations.
    */
    void init()
    {
        setParameter();
        calLinear();
        calM();
        calOperations();
    }

    /* *\ brief Produce color space instance with linear and non-linear versions.
       *\ param rgbl type of RGBBase_.
    */
    void bind(RGBBase_& rgbl)
    {
        init();
        rgbl.init();
        l = &rgbl;
        rgbl.l = &rgbl;
        nl = this;
        rgbl.nl = this;
    }

private:
    virtual void setParameter() {};

    /* *\ brief Calculation of M_RGBL2XYZ_base.
       *        see ColorSpace.pdf for details.
    */
    virtual void calM()
    {
        Mat XYZr, XYZg, XYZb, XYZ_rgbl, Srgb;
        XYZr = Mat(xyY2XYZ({ xr, yr }), true);
        XYZg = Mat(xyY2XYZ({ xg, yg }), true);
        XYZb = Mat(xyY2XYZ({ xb, yb }), true);
        merge(std::vector<Mat>{ XYZr, XYZg, XYZb }, XYZ_rgbl);
        XYZ_rgbl = XYZ_rgbl.reshape(1, XYZ_rgbl.rows);
        Mat XYZw = Mat(illuminants.find(io)->second, true);
        solve(XYZ_rgbl, XYZw, Srgb);
        merge(std::vector<Mat>{ Srgb.at<double>(0)* XYZr,
            Srgb.at<double>(1)* XYZg,
            Srgb.at<double>(2)* XYZb }, M_to);
        M_to = M_to.reshape(1, M_to.rows);
        M_from = M_to.inv();
    };

    /* *\ brief operations to or from XYZ.
    */
    virtual void calOperations()
    {
        // rgb -> rgbl
        toL = [this](Mat rgb)->Mat {return toLFunc(rgb); };

        // rgbl -> rgb
        fromL = [this](Mat rgbl)->Mat {return fromLFunc(rgbl); };

        if (linear)
        {
            to = Operations({ Operation(M_to.t()) });
            from = Operations({ Operation(M_from.t()) });
        }
        else
        {
            to = Operations({ Operation(toL), Operation(M_to.t()) });
            from = Operations({ Operation(M_from.t()), Operation(fromL) });
        }
    }

    virtual void calLinear() {}

    virtual Mat toLFunc(Mat& /*rgb*/)
    {
        return Mat();
    };

    virtual Mat fromLFunc(Mat& /*rgbl*/)
    {
        return Mat();
    };

};

/* *\ brief Base of Adobe RGB color space;
*/
class AdobeRGBBase_ : public RGBBase_
{
public:
    using RGBBase_::RGBBase_;
    double gamma;

private:
    Mat toLFunc(Mat& rgb) CV_OVERRIDE
    {
        return gammaCorrection(rgb, gamma);
    }

    Mat fromLFunc(Mat& rgbl) CV_OVERRIDE
    {
        return gammaCorrection(rgbl, 1. / gamma);
    }
};

/* *\ brief Base of sRGB color space;
*/
class sRGBBase_ : public RGBBase_
{
public:
    using RGBBase_::RGBBase_;
    double a;
    double gamma;
    double alpha;
    double beta;
    double phi;
    double K0;

private:
    /* *\ brief linearization parameters
       *        see ColorSpace.pdf for details;
    */
    virtual void calLinear() CV_OVERRIDE
    {
        alpha = a + 1;
        K0 = a / (gamma - 1);
        phi = (pow(alpha, gamma) * pow(gamma - 1, gamma - 1)) / (pow(a, gamma - 1) * pow(gamma, gamma));
        beta = K0 / phi;
    }

    /* *\ brief Used by toLFunc.
    */
    double toLFuncEW(double& x)
    {
        if (x > K0)
        {
            return pow(((x + alpha - 1) / alpha), gamma);
        }
        else if (x >= -K0)
        {
            return x / phi;
        }
        else
        {
            return -(pow(((-x + alpha - 1) / alpha), gamma));
        }
    }

    /* *\ brief Linearization.
       *        see ColorSpace.pdf for details.
       *\ param rgb the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    Mat toLFunc(Mat& rgb) CV_OVERRIDE
    {
        return elementWise(rgb, [this](double a_)->double {return toLFuncEW(a_); });
    }

    /* *\ brief Used by fromLFunc.
    */
    double fromLFuncEW(double& x)
    {
        if (x > beta)
        {
            return alpha * pow(x, 1 / gamma) - (alpha - 1);
        }
        else if (x >= -beta)
        {
            return x * phi;
        }
        else
        {
            return -(alpha * pow(-x, 1 / gamma) - (alpha - 1));
        }
    }

    /* *\ brief Delinearization.
       *        see ColorSpace.pdf for details.
       *\ param rgbl the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    Mat fromLFunc(Mat& rgbl) CV_OVERRIDE
    {
        return elementWise(rgbl, [this](double a_)->double {return fromLFuncEW(a_); });
    }
};

/* *\ brief sRGB color space.
   *        data from https://en.wikipedia.org/wiki/SRGB.
*/
class sRGB_ :public sRGBBase_
{
public:
    sRGB_(bool linear_) :sRGBBase_(D65_2, "sRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.64;
        yr = 0.33;
        xg = 0.3;
        yg = 0.6;
        xb = 0.15;
        yb = 0.06;
        a = 0.055;
        gamma = 2.4;
    }
};

/* *\ brief Adobe RGB color space.
*/
class AdobeRGB_ : public AdobeRGBBase_
{
public:
    AdobeRGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "AdobeRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.64;
        yr = 0.33;
        xg = 0.21;
        yg = 0.71;
        xb = 0.15;
        yb = 0.06;
        gamma = 2.2;
    }
};

/* *\ brief Wide-gamut RGB color space.
   *        data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space.
*/
class WideGamutRGB_ : public AdobeRGBBase_
{
public:
    WideGamutRGB_(bool linear_ = false) :AdobeRGBBase_(D50_2, "WideGamutRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.7347;
        yr = 0.2653;
        xg = 0.1152;
        yg = 0.8264;
        xb = 0.1566;
        yb = 0.0177;
        gamma = 2.2;
    }
};

/* *\ brief ProPhoto RGB color space.
   *        data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space.
*/
class ProPhotoRGB_ : public AdobeRGBBase_
{
public:
    ProPhotoRGB_(bool linear_ = false) :AdobeRGBBase_(D50_2, "ProPhotoRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.734699;
        yr = 0.265301;
        xg = 0.159597;
        yg = 0.840403;
        xb = 0.036598;
        yb = 0.000105;
        gamma = 1.8;
    }
};

/* *\ brief DCI-P3 RGB color space.
   *        data from https://en.wikipedia.org/wiki/DCI-P3.
*/
class DCI_P3_RGB_ : public AdobeRGBBase_
{
public:
    DCI_P3_RGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "DCI_P3_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.68;
        yr = 0.32;
        xg = 0.265;
        yg = 0.69;
        xb = 0.15;
        yb = 0.06;
        gamma = 2.2;
    }
};

/* *\ brief Apple RGB color space.
   *        data from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html.
*/
class AppleRGB_ : public AdobeRGBBase_
{
public:
    AppleRGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "AppleRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.625;
        yr = 0.34;
        xg = 0.28;
        yg = 0.595;
        xb = 0.155;
        yb = 0.07;
        gamma = 1.8;
    }
};

/* *\ brief REC_709 RGB color space.
   *        data from https://en.wikipedia.org/wiki/Rec._709.
*/
class REC_709_RGB_ : public sRGBBase_
{
public:
    REC_709_RGB_(bool linear_) :sRGBBase_(D65_2, "REC_709_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.64;
        yr = 0.33;
        xg = 0.3;
        yg = 0.6;
        xb = 0.15;
        yb = 0.06;
        a = 0.099;
        gamma = 1 / 0.45;
    }
};

/* *\ brief REC_2020 RGB color space.
   *        data from https://en.wikipedia.org/wiki/Rec._2020.
*/
class REC_2020_RGB_ : public sRGBBase_
{
public:
    REC_2020_RGB_(bool linear_) :sRGBBase_(D65_2, "REC_2020_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE
    {
        xr = 0.708;
        yr = 0.292;
        xg = 0.17;
        yg = 0.797;
        xb = 0.131;
        yb = 0.046;
        a = 0.09929682680944;
        gamma = 1 / 0.45;
    }
};


sRGB_ sRGB(false), sRGBL(true);
AdobeRGB_ AdobeRGB(false), AdobeRGBL(true);
WideGamutRGB_ WideGamutRGB(false), WideGamutRGBL(true);
ProPhotoRGB_ ProPhotoRGB(false), ProPhotoRGBL(true);
DCI_P3_RGB_ DCI_P3_RGB(false), DCI_P3_RGBL(true);
AppleRGB_ AppleRGB(false), AppleRGBL(true);
REC_709_RGB_ REC_709_RGB(false), REC_709_RGBL(true);
REC_2020_RGB_ REC_2020_RGB(false), REC_2020_RGBL(true);

/* *\ brief Bind RGB with RGBL.
*/
class ColorSpaceInitial
{
public:
    ColorSpaceInitial()
    {
        sRGB.bind(sRGBL);
        AdobeRGB.bind(AdobeRGBL);
        WideGamutRGB.bind(WideGamutRGBL);
        ProPhotoRGB.bind(ProPhotoRGBL);
        DCI_P3_RGB.bind(DCI_P3_RGBL);
        AppleRGB.bind(AppleRGBL);
        REC_709_RGB.bind(REC_709_RGBL);
        REC_2020_RGB.bind(REC_2020_RGBL);

    }
};

ColorSpaceInitial color_space_initial;

/* *\ brief Enum of the possible types of CAMs.
*/
enum CAM
{
    IDENTITY,
    VON_KRIES,
    BRADFORD
};

static std::map <std::tuple<IO, IO, CAM>, Mat > cams;
const static Mat Von_Kries = (Mat_<double>(3, 3) << 0.40024, 0.7076, -0.08081, -0.2263, 1.16532, 0.0457, 0., 0., 0.91822);
const static Mat Bradford = (Mat_<double>(3, 3) << 0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296);
const static std::map <CAM, std::vector< Mat >> MAs = {
    {IDENTITY , { Mat::eye(cv::Size(3,3),CV_64FC1) , Mat::eye(cv::Size(3,3),CV_64FC1)} },
    {VON_KRIES, { Von_Kries ,Von_Kries.inv() }},
    {BRADFORD, { Bradford ,Bradford.inv() }}
};

/* *\ brief XYZ color space.
   *        Chromatic adaption matrices.
*/
class XYZ :public ColorSpace
{
public:
    XYZ(IO io_) : ColorSpace(io_, "XYZ", true) {};
    Operations cam(IO dio, CAM method = BRADFORD)
    {
        return (io == dio) ? Operations() : Operations({ Operation(cam_(io, dio, method).t()) });
    }

private:
    /* *\ brief Get cam.
       *\ param sio the input IO of src.
       *\ param dio the input IO of dst.
       *\ param method type of CAM.
       *\ return the output array, type of cv::Mat.
    */
    Mat cam_(IO sio, IO dio, CAM method = BRADFORD) const
    {
        if (sio == dio)
        {
            return Mat::eye(cv::Size(3, 3), CV_64FC1);
        }
        if (cams.count(std::make_tuple(dio, sio, method)) == 1)
        {
            return cams[std::make_tuple(dio, sio, method)];
        }

        // Function from http ://www.brucelindbloom.com/index.html?ColorCheckerRGB.html.
        Mat XYZws = Mat(illuminants.find(dio)->second);
        Mat XYZWd = Mat(illuminants.find(sio)->second);
        Mat MA = MAs.at(method)[0];
        Mat MA_inv = MAs.at(method)[1];
        Mat M = MA_inv * Mat::diag((MA * XYZws) / (MA * XYZWd)) * MA;
        cams[std::make_tuple(dio, sio, method)] = M;
        cams[std::make_tuple(sio, dio, method)] = M.inv();
        return M;
    }
};

/* *\ brief Define XYZ_D65_2 and XYZ_D50_2.
*/
const XYZ XYZ_D65_2(D65_2);
const XYZ XYZ_D50_2(D50_2);

/* *\ brief Lab color space.
*/
class Lab :public ColorSpace
{
public:
    Lab(IO io_) : ColorSpace(io_, "Lab", true)
    {
        to = { Operation([this](Mat src)->Mat {return tosrc(src); }) };
        from = { Operation([this](Mat src)->Mat {return fromsrc(src); }) };
    }

private:
    static constexpr double delta = (6. / 29.);
    static constexpr double m = 1. / (3. * delta * delta);
    static constexpr double t0 = delta * delta * delta;
    static constexpr double c = 4. / 29.;

    cv::Vec3d fromxyz(cv::Vec3d& xyz)
    {
        double x = xyz[0] / illuminants.find(io)->second[0], y = xyz[1] / illuminants.find(io)->second[1], z = xyz[2] / illuminants.find(io)->second[2];
        auto f = [](double t)->double { return t > t0 ? std::cbrt(t) : (m * t + c); };
        double fx = f(x), fy = f(y), fz = f(z);
        return { 116. * fy - 16. ,500 * (fx - fy),200 * (fy - fz) };
    }

    /* *\ brief Calculate From.
       *\ param src the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    Mat fromsrc(Mat& src)
    {
        return channelWise(src, [this](cv::Vec3d a)->cv::Vec3d {return fromxyz(a); });
    }

    cv::Vec3d tolab(cv::Vec3d& lab)
    {
        auto f_inv = [](double t)->double {return t > delta ? pow(t, 3.0) : (t - c) / m; };
        double L = (lab[0] + 16.) / 116., a = lab[1] / 500., b = lab[2] / 200.;
        return { illuminants.find(io)->second[0] * f_inv(L + a),illuminants.find(io)->second[1] * f_inv(L),illuminants.find(io)->second[2] * f_inv(L - b) };
    }

    /* *\ brief Calculate To.
       *\ param src the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    Mat tosrc(Mat& src)
    {
        return channelWise(src, [this](cv::Vec3d a)->cv::Vec3d {return tolab(a); });
    }
};

/* *\ brief Define Lab_D65_2 and Lab_D50_2.
*/
const Lab Lab_D65_2(D65_2);
const Lab Lab_D50_2(D50_2);

} // namespace ccm
} // namespace cv


#endif