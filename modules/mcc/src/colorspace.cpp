// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
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

#include "precomp.hpp"

namespace cv {
namespace ccm {

static std::map<IO, std::vector<double>> illuminants = {
  {A_2, { 1.098466069456375, 1, 0.3558228003436005 }},
  {A_10, { 1.111420406956693, 1, 0.3519978321919493 }},
  {D50_2, { 0.9642119944211994, 1, 0.8251882845188288 }},
  {D50_10, { 0.9672062750333777, 1, 0.8142801513128616 }},
  {D55_2, { 0.956797052643698, 1, 0.9214805860173273 }},
  {D55_10, { 0.9579665682254781, 1, 0.9092525159847462 }},
  {D65_2, { 0.95047, 1., 1.08883 }},
  {D65_10, { 0.94811, 1., 1.07304 }},
  {D75_2, { 0.9497220898840717, 1, 1.226393520724154 }},
  {D75_10, { 0.9441713925645873, 1, 1.2064272211720228 }},
  {E_2, { 1., 1., 1. }},
  {E_10, { 1., 1., 1. }},
};

/* *\ brief Basic class for ColorSpace.
 */
bool ColorSpace::relate(const ColorSpace& other) const {
  return (type == other.type) && (io == other.io);
};

Operations ColorSpace::relation(const ColorSpace& /*other*/) const {
  return IDENTITY_OPS;
};

bool ColorSpace::operator<(const ColorSpace& other) const {
  return (io < other.io || (io == other.io && type < other.type) ||
          (io == other.io && type == other.type && linear < other.linear));
}

/* *\ brief Base of RGB color space;
 *        the argument values are from AdobeRGB;
 *        Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
 */
Operations RGBBase_::relation(const ColorSpace& other) const {
  if (linear == other.linear) {
    return IDENTITY_OPS;
  }
  if (linear) {
    return Operations({Operation(fromL)});
  }
  return Operations({Operation(toL)});
};

/* *\ brief Initial operations.
 */
void RGBBase_::init() {
  setParameter();
  calLinear();
  calM();
  calOperations();
}

/* *\ brief Produce color space instance with linear and non-linear versions.
 *\ param rgbl type of RGBBase_.
 */
void RGBBase_::bind(RGBBase_& rgbl) {
  init();
  rgbl.init();
  l = &rgbl;
  rgbl.l = &rgbl;
  nl = this;
  rgbl.nl = this;
}

/* *\ brief Calculation of M_RGBL2XYZ_base.
 *        see ColorSpace.pdf for details.
 */
void RGBBase_::calM() {
  Mat XYZr, XYZg, XYZb, XYZ_rgbl, Srgb;
  XYZr = Mat(xyY2XYZ({xr, yr}), true);
  XYZg = Mat(xyY2XYZ({xg, yg}), true);
  XYZb = Mat(xyY2XYZ({xb, yb}), true);
  merge(std::vector<Mat>{XYZr, XYZg, XYZb}, XYZ_rgbl);
  XYZ_rgbl = XYZ_rgbl.reshape(1, XYZ_rgbl.rows);
  Mat XYZw = Mat(illuminants.find(io)->second, true);
  solve(XYZ_rgbl, XYZw, Srgb);
  merge(std::vector<Mat>{Srgb.at<double>(0) * XYZr, Srgb.at<double>(1) * XYZg,
                         Srgb.at<double>(2) * XYZb},
        M_to);
  M_to = M_to.reshape(1, M_to.rows);
  M_from = M_to.inv();
};

/* *\ brief operations to or from XYZ.
 */
void RGBBase_::calOperations() {
  // rgb -> rgbl
  toL = [this](Mat rgb) -> Mat { return toLFunc(rgb); };

  // rgbl -> rgb
  fromL = [this](Mat rgbl) -> Mat { return fromLFunc(rgbl); };

  if (linear) {
    to = Operations({Operation(M_to.t())});
    from = Operations({Operation(M_from.t())});
  } else {
    to = Operations({Operation(toL), Operation(M_to.t())});
    from = Operations({Operation(M_from.t()), Operation(fromL)});
  }
}

Mat RGBBase_::toLFunc(Mat& /*rgb*/) { return Mat(); }

Mat RGBBase_::fromLFunc(Mat& /*rgbl*/) { return Mat(); }

/* *\ brief Base of Adobe RGB color space;
 */

Mat AdobeRGBBase_::toLFunc(Mat& rgb) { return gammaCorrection(rgb, gamma); }

Mat AdobeRGBBase_::fromLFunc(Mat& rgbl) {
  return gammaCorrection(rgbl, 1. / gamma);
}

/* *\ brief Base of sRGB color space;
 */

void sRGBBase_::calLinear() {
  alpha = a + 1;
  K0 = a / (gamma - 1);
  phi = (pow(alpha, gamma) * pow(gamma - 1, gamma - 1)) /
        (pow(a, gamma - 1) * pow(gamma, gamma));
  beta = K0 / phi;
}

/* *\ brief Used by toLFunc.
 */
double sRGBBase_::toLFuncEW(double& x) {
  if (x > K0) {
    return pow(((x + alpha - 1) / alpha), gamma);
  } else if (x >= -K0) {
    return x / phi;
  } else {
    return -(pow(((-x + alpha - 1) / alpha), gamma));
  }
}

/* *\ brief Linearization.
 *        see ColorSpace.pdf for details.
 *\ param rgb the input array, type of cv::Mat.
 *\ return the output array, type of cv::Mat.
 */
Mat sRGBBase_::toLFunc(Mat& rgb) {
  return elementWise(rgb,
                     [this](double a_) -> double { return toLFuncEW(a_); });
}

/* *\ brief Used by fromLFunc.
 */
double sRGBBase_::fromLFuncEW(double& x) {
  if (x > beta) {
    return alpha * pow(x, 1 / gamma) - (alpha - 1);
  } else if (x >= -beta) {
    return x * phi;
  } else {
    return -(alpha * pow(-x, 1 / gamma) - (alpha - 1));
  }
}

/* *\ brief Delinearization.
 *        see ColorSpace.pdf for details.
 *\ param rgbl the input array, type of cv::Mat.
 *\ return the output array, type of cv::Mat.
 */
Mat sRGBBase_::fromLFunc(Mat& rgbl) {
  return elementWise(rgbl,
                     [this](double a_) -> double { return fromLFuncEW(a_); });
}

/* *\ brief sRGB color space.
 *        data from https://en.wikipedia.org/wiki/SRGB.
 */

void sRGB_::setParameter() {
  xr = 0.64;
  yr = 0.33;
  xg = 0.3;
  yg = 0.6;
  xb = 0.15;
  yb = 0.06;
  a = 0.055;
  gamma = 2.4;
}

/* *\ brief Adobe RGB color space.
 */

void AdobeRGB_::setParameter() {
  xr = 0.64;
  yr = 0.33;
  xg = 0.21;
  yg = 0.71;
  xb = 0.15;
  yb = 0.06;
  gamma = 2.2;
}

/* *\ brief Wide-gamut RGB color space.
 *        data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space.
 */

void WideGamutRGB_::setParameter() {
  xr = 0.7347;
  yr = 0.2653;
  xg = 0.1152;
  yg = 0.8264;
  xb = 0.1566;
  yb = 0.0177;
  gamma = 2.2;
}

/* *\ brief ProPhoto RGB color space.
 *        data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space.
 */

void ProPhotoRGB_::setParameter() {
  xr = 0.734699;
  yr = 0.265301;
  xg = 0.159597;
  yg = 0.840403;
  xb = 0.036598;
  yb = 0.000105;
  gamma = 1.8;
}

/* *\ brief DCI-P3 RGB color space.
 *        data from https://en.wikipedia.org/wiki/DCI-P3.
 */

void DCI_P3_RGB_::setParameter() {
  xr = 0.68;
  yr = 0.32;
  xg = 0.265;
  yg = 0.69;
  xb = 0.15;
  yb = 0.06;
  gamma = 2.2;
}

/* *\ brief Apple RGB color space.
 *        data from
 * http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html.
 */

void AppleRGB_::setParameter() {
  xr = 0.625;
  yr = 0.34;
  xg = 0.28;
  yg = 0.595;
  xb = 0.155;
  yb = 0.07;
  gamma = 1.8;
}

/* *\ brief REC_709 RGB color space.
 *        data from https://en.wikipedia.org/wiki/Rec._709.
 */

void REC_709_RGB_::setParameter() {
  xr = 0.64;
  yr = 0.33;
  xg = 0.3;
  yg = 0.6;
  xb = 0.15;
  yb = 0.06;
  a = 0.099;
  gamma = 1 / 0.45;
}

/* *\ brief REC_2020 RGB color space.
 *        data from https://en.wikipedia.org/wiki/Rec._2020.
 */

void REC_2020_RGB_::setParameter() {
  xr = 0.708;
  yr = 0.292;
  xg = 0.17;
  yg = 0.797;
  xb = 0.131;
  yb = 0.046;
  a = 0.09929682680944;
  gamma = 1 / 0.45;
}

sRGB_ sRGB = sRGB_(false);
sRGB_ sRGBL = sRGB_(true);
AdobeRGB_ AdobeRGB = AdobeRGB_(false);
AdobeRGB_ AdobeRGBL = AdobeRGB_(true);
WideGamutRGB_ WideGamutRGB = WideGamutRGB_(false);
WideGamutRGB_ WideGamutRGBL = WideGamutRGB_(true);
ProPhotoRGB_ ProPhotoRGB = ProPhotoRGB_(false);
ProPhotoRGB_ ProPhotoRGBL = ProPhotoRGB_(true);
DCI_P3_RGB_ DCI_P3_RGB = DCI_P3_RGB_(false);
DCI_P3_RGB_ DCI_P3_RGBL = DCI_P3_RGB_(true);
AppleRGB_ AppleRGB = AppleRGB_(false);
AppleRGB_ AppleRGBL = AppleRGB_(true);
REC_709_RGB_ REC_709_RGB = REC_709_RGB_(false);
REC_709_RGB_ REC_709_RGBL = REC_709_RGB_(true);
REC_2020_RGB_ REC_2020_RGB = REC_2020_RGB_(false);
REC_2020_RGB_ REC_2020_RGBL = REC_2020_RGB_(true);

/* *\ brief Bind RGB with RGBL.
 */

ColorSpaceInitial::ColorSpaceInitial() {
  sRGB.bind(sRGBL);
  AdobeRGB.bind(AdobeRGBL);
  WideGamutRGB.bind(WideGamutRGBL);
  ProPhotoRGB.bind(ProPhotoRGBL);
  DCI_P3_RGB.bind(DCI_P3_RGBL);
  AppleRGB.bind(AppleRGBL);
  REC_709_RGB.bind(REC_709_RGBL);
  REC_2020_RGB.bind(REC_2020_RGBL);
}

ColorSpaceInitial color_space_initial;

/* *\ brief Enum of the possible types of CAMs.
 */

/* *\ brief XYZ color space.
 *        Chromatic adaption matrices.
 */
Operations XYZ::cam(IO dio, CAM method) {
  return (io == dio) ? Operations()
                     : Operations({Operation(cam_(io, dio, method).t())});
}
Mat XYZ::cam_(IO sio, IO dio, CAM method) const {
  if (sio == dio) {
    return Mat::eye(cv::Size(3, 3), CV_64FC1);
  }
  if (cams.count(std::make_tuple(dio, sio, method)) == 1) {
    return cams[std::make_tuple(dio, sio, method)];
  }

  // Function from http
  // ://www.brucelindbloom.com/index.html?ColorCheckerRGB.html.
  Mat XYZws = Mat(illuminants.find(dio)->second);
  Mat XYZWd = Mat(illuminants.find(sio)->second);
  Mat MA = MAs.at(method)[0];
  Mat MA_inv = MAs.at(method)[1];
  Mat M = MA_inv * Mat::diag((MA * XYZws) / (MA * XYZWd)) * MA;
  cams[std::make_tuple(dio, sio, method)] = M;
  cams[std::make_tuple(sio, dio, method)] = M.inv();
  return M;
}

/* *\ brief Lab color space.
 */
Lab::Lab(IO io_) : ColorSpace(io_, "Lab", true) {
  to = {Operation([this](Mat src) -> Mat { return tosrc(src); })};
  from = {Operation([this](Mat src) -> Mat { return fromsrc(src); })};
}

Vec3d Lab::fromxyz(cv::Vec3d& xyz) {
  double x = xyz[0] / illuminants.find(io)->second[0],
         y = xyz[1] / illuminants.find(io)->second[1],
         z = xyz[2] / illuminants.find(io)->second[2];
  auto f = [](double t) -> double {
    return t > t0 ? std::cbrt(t) : (m * t + c);
  };
  double fx = f(x), fy = f(y), fz = f(z);
  return {116. * fy - 16., 500 * (fx - fy), 200 * (fy - fz)};
}

/* *\ brief Calculate From.
 *\ param src the input array, type of cv::Mat.
 *\ return the output array, type of cv::Mat
 */
Mat Lab::fromsrc(Mat& src) {
  return channelWise(src,
                     [this](cv::Vec3d a) -> cv::Vec3d { return fromxyz(a); });
}

Vec3d Lab::tolab(cv::Vec3d& lab) {
  auto f_inv = [](double t) -> double {
    return t > delta ? pow(t, 3.0) : (t - c) / m;
  };
  double L = (lab[0] + 16.) / 116., a = lab[1] / 500., b = lab[2] / 200.;
  return {illuminants.find(io)->second[0] * f_inv(L + a),
          illuminants.find(io)->second[1] * f_inv(L),
          illuminants.find(io)->second[2] * f_inv(L - b)};
}

/* *\ brief Calculate To.
 *\ param src the input array, type of cv::Mat.
 *\ return the output array, type of cv::Mat
 */
Mat Lab::tosrc(Mat& src) {
  return channelWise(src,
                     [this](cv::Vec3d a) -> cv::Vec3d { return tolab(a); });
}

}  // namespace ccm
}  // namespace cv
