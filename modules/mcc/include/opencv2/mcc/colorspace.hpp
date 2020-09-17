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
class CV_EXPORTS_W ColorSpace
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

    virtual bool relate(const ColorSpace& other) const;

    virtual Operations relation(const ColorSpace& /*other*/) const;

    bool operator<(const ColorSpace& other)const;
};

/* *\ brief Base of RGB color space;
   *        the argument values are from AdobeRGB;
   *        Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
*/
class CV_EXPORTS_W RGBBase_ : public ColorSpace
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
    Operations relation(const ColorSpace& other) const CV_OVERRIDE;

    /* *\ brief Initial operations.
    */
    void init();
    /* *\ brief Produce color space instance with linear and non-linear versions.
       *\ param rgbl type of RGBBase_.
    */
    void bind(RGBBase_& rgbl);

private:
    virtual void setParameter() {};

    /* *\ brief Calculation of M_RGBL2XYZ_base.
       *        see ColorSpace.pdf for details.
    */
    virtual void calM();

    /* *\ brief operations to or from XYZ.
    */
    virtual void calOperations();

    virtual void calLinear() {};

    virtual Mat toLFunc(Mat& /*rgb*/);

    virtual Mat fromLFunc(Mat& /*rgbl*/);

};

/* *\ brief Base of Adobe RGB color space;
*/
class CV_EXPORTS_W AdobeRGBBase_ : public RGBBase_
{
public:
    using RGBBase_::RGBBase_;
    double gamma;

private:
    Mat toLFunc(Mat& rgb) CV_OVERRIDE;
    Mat fromLFunc(Mat& rgbl) CV_OVERRIDE;
};

/* *\ brief Base of sRGB color space;
*/
class CV_EXPORTS_W sRGBBase_ : public RGBBase_
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
    virtual void calLinear() CV_OVERRIDE;
    /* *\ brief Used by toLFunc.
    */
    double toLFuncEW(double& x);

    /* *\ brief Linearization.
       *        see ColorSpace.pdf for details.
       *\ param rgb the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    Mat toLFunc(Mat& rgb) CV_OVERRIDE;

    /* *\ brief Used by fromLFunc.
    */
    double fromLFuncEW(double& x);
    /* *\ brief Delinearization.
       *        see ColorSpace.pdf for details.
       *\ param rgbl the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    Mat fromLFunc(Mat& rgbl) CV_OVERRIDE;
};

/* *\ brief sRGB color space.
   *        data from https://en.wikipedia.org/wiki/SRGB.
*/
class CV_EXPORTS_W sRGB_ :public sRGBBase_
{
public:
    sRGB_(bool linear_) :sRGBBase_(D65_2, "sRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief Adobe RGB color space.
*/
class CV_EXPORTS_W AdobeRGB_ : public AdobeRGBBase_
{
public:
    AdobeRGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "AdobeRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief Wide-gamut RGB color space.
   *        data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space.
*/
class CV_EXPORTS_W WideGamutRGB_ : public AdobeRGBBase_
{
public:
    WideGamutRGB_(bool linear_ = false) :AdobeRGBBase_(D50_2, "WideGamutRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief ProPhoto RGB color space.
   *        data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space.
*/
class CV_EXPORTS_W ProPhotoRGB_ : public AdobeRGBBase_
{
public:
    ProPhotoRGB_(bool linear_ = false) :AdobeRGBBase_(D50_2, "ProPhotoRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief DCI-P3 RGB color space.
   *        data from https://en.wikipedia.org/wiki/DCI-P3.
*/
class CV_EXPORTS_W DCI_P3_RGB_ : public AdobeRGBBase_
{
public:
    DCI_P3_RGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "DCI_P3_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief Apple RGB color space.
   *        data from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html.
*/
class CV_EXPORTS_W AppleRGB_ : public AdobeRGBBase_
{
public:
    AppleRGB_(bool linear_ = false) :AdobeRGBBase_(D65_2, "AppleRGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief REC_709 RGB color space.
   *        data from https://en.wikipedia.org/wiki/Rec._709.
*/
class CV_EXPORTS_W REC_709_RGB_ : public sRGBBase_
{
public:
    REC_709_RGB_(bool linear_) :sRGBBase_(D65_2, "REC_709_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};

/* *\ brief REC_2020 RGB color space.
   *        data from https://en.wikipedia.org/wiki/Rec._2020.
*/
class CV_EXPORTS_W REC_2020_RGB_ : public sRGBBase_
{
public:
    REC_2020_RGB_(bool linear_) :sRGBBase_(D65_2, "REC_2020_RGB", linear_) {};

private:
    void setParameter() CV_OVERRIDE;
};


CV_EXPORTS_W extern sRGB_ sRGB, sRGBL;
CV_EXPORTS_W extern AdobeRGB_ AdobeRGB, AdobeRGBL;
CV_EXPORTS_W extern WideGamutRGB_ WideGamutRGB, WideGamutRGBL;
CV_EXPORTS_W extern ProPhotoRGB_ ProPhotoRGB, ProPhotoRGBL;
CV_EXPORTS_W extern DCI_P3_RGB_ DCI_P3_RGB, DCI_P3_RGBL;
CV_EXPORTS_W extern AppleRGB_ AppleRGB, AppleRGBL;
CV_EXPORTS_W extern REC_709_RGB_ REC_709_RGB, REC_709_RGBL;
CV_EXPORTS_W extern REC_2020_RGB_ REC_2020_RGB, REC_2020_RGBL;

/* *\ brief Bind RGB with RGBL.
*/
class CV_EXPORTS_W ColorSpaceInitial
{
public:
    ColorSpaceInitial();
};

extern ColorSpaceInitial color_space_initial;

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
class CV_EXPORTS_W  XYZ :public ColorSpace
{
public:
    XYZ(IO io_) : ColorSpace(io_, "XYZ", true) {};
    Operations cam(IO dio, CAM method = BRADFORD);
private:
    /* *\ brief Get cam.
       *\ param sio the input IO of src.
       *\ param dio the input IO of dst.
       *\ param method type of CAM.
       *\ return the output array, type of cv::Mat.
    */
    Mat cam_(IO sio, IO dio, CAM method = BRADFORD) const;
};

/* *\ brief Define XYZ_D65_2 and XYZ_D50_2.
*/
const XYZ XYZ_D65_2(D65_2);
const XYZ XYZ_D50_2(D50_2);

/* *\ brief Lab color space.
*/
class CV_EXPORTS_W Lab :public ColorSpace
{
public:
    Lab(IO io_);
private:
    static constexpr double delta = (6. / 29.);
    static constexpr double m = 1. / (3. * delta * delta);
    static constexpr double t0 = delta * delta * delta;
    static constexpr double c = 4. / 29.;

    cv::Vec3d fromxyz(cv::Vec3d& xyz);

    /* *\ brief Calculate From.
       *\ param src the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    Mat fromsrc(Mat& src);

    cv::Vec3d tolab(cv::Vec3d& lab);

    /* *\ brief Calculate To.
       *\ param src the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    Mat tosrc(Mat& src);
};

/* *\ brief Define Lab_D65_2 and Lab_D50_2.
*/
const Lab Lab_D65_2(D65_2);
const Lab Lab_D50_2(D50_2);

} // namespace ccm
} // namespace cv


#endif
