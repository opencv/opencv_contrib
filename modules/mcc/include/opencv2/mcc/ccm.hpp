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

#ifndef __OPENCV_MCC_CCM_HPP__
#define __OPENCV_MCC_CCM_HPP__

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// #include "opencv2/mcc/linearize.hpp"

namespace cv
{
namespace ccm
{

/**
    src :
            detected colors of ColorChecker patches;
            NOTICE: the color type is RGB not BGR, and the color values are in [0, 1];
            type: cv::Mat;
    constcolor :
            the Built-in color card;
            Supported list:
                Macbeth: Macbeth ColorChecker with 24 squares;
                Vinyl: DKK ColorChecker with 12 squares and 6 rectangle;
                DigitalSG: DigitalSG ColorChecker with 140 squares;
            type: enum CONST_COLOR;
    Mat colors_ :
           the reference color values
           and corresponding color space
           NOTICE: the color values are in [0, 1]
           type: cv::Mat
    ref_cs_ :
           the corresponding color space
           NOTICE: For the list of color spaces supported, see the notes below;
                  If the color type is some RGB, the format is RGB not BGR;
           type:enum COLOR_SPACE;
    cs_ :
            the absolute color space that detected colors convert to;
            NOTICE: it should be some RGB color space;
                    For the list of RGB color spaces supported, see the notes below;
            type: enum COLOR_SPACE;
    dst_ :
            the reference colors;
            NOTICE: custom color card are supported;
                    You should generate Color instance using reference color values and corresponding color space
                    For the list of color spaces supported, see the notes below;
                    If the color type is some RGB, the format is RGB not BGR, and the color values are in [0, 1];

    ccm_type :
            the shape of color correction matrix(CCM);
            Supported list:
                "CCM_3x3": 3x3 matrix;
                "CCM_4x3": 4x3 matrix;
            type: enum CCM_TYPE;
            default: CCM_3x3;
    distance :
            the type of color distance;
            Supported list:
                "CIE2000";
                "CIE94_GRAPHIC_ARTS";
                "CIE94_TEXTILES";
                "CIE76";
                "CMC_1TO1";
                "CMC_2TO1";
                "RGB" : Euclidean distance of rgb color space;
                "RGBL" : Euclidean distance of rgbl color space;
            type: enum DISTANCE_TYPE;
            default: CIE2000;
    linear_type :
            the method of linearization;
            NOTICE: see Linearization.pdf for details;
            Supported list:
                "IDENTITY_" : no change is made;
                "GAMMA": gamma correction;
                        Need assign a value to gamma simultaneously;
                "COLORPOLYFIT": polynomial fitting channels respectively;
                                Need assign a value to deg simultaneously;
                "GRAYPOLYFIT": grayscale polynomial fitting;
                                Need assign a value to deg and dst_whites simultaneously;
                "COLORLOGPOLYFIT": logarithmic polynomial fitting channels respectively;
                                Need assign a value to deg simultaneously;
                "GRAYLOGPOLYFIT": grayscale Logarithmic polynomial fitting;
                                Need assign a value to deg and dst_whites simultaneously;
            type: enum LINEAR_TYPE;
            default: IDENTITY_;
    gamma :
            the gamma value of gamma correction;
            NOTICE: only valid when linear is set to "gamma";
            type: double;
            default: 2.2;
    deg :
            the degree of linearization polynomial;
            NOTICE: only valid when linear is set to "COLORPOLYFIT", "GRAYPOLYFIT",
                    "COLORLOGPOLYFIT" and "GRAYLOGPOLYFIT";
            type: int;
            default: 3;
    saturated_threshold :
            the threshold to determine saturation;
            NOTICE: it is a tuple of [low, up];
                    The colors in the closed interval [low, up] are reserved to participate
                    in the calculation of the loss function and initialization parameters.
            type: std::vector<double>;
            default: { 0, 0.98 };
    ---------------------------------------------------
    There are some ways to set weights:
        1. set weights_list only;
        2. set weights_coeff only;
    see CCM.pdf for details;
    weights_list :
            the list of weight of each color;
            type: cv::Mat;
            default: empty array;
    weights_coeff :
            the exponent number of L* component of the reference color in CIE Lab color space;
            type: double;
            default: 0;
    ---------------------------------------------------
    initial_method_type :
            the method of calculating CCM initial value;
            see CCM.pdf for details;
            Supported list:
                'LEAST_SQUARE': least-squre method;
                'WHITE_BALANCE': white balance method;
            type: enum INITIAL_METHOD_TYPE;
    max_count, epsilon :
            used in MinProblemSolver-DownhillSolver;
            Terminal criteria to the algorithm;
            type: int, double;
            default: 5000, 1e-4;
    ---------------------------------------------------
    Supported Color Space:
            Supported list of RGB color spaces:
                sRGB;
                AdobeRGB;
                WideGamutRGB;
                ProPhotoRGB;
                DCI_P3_RGB;
                AppleRGB;
                REC_709_RGB;
                REC_2020_RGB;
            Supported list of linear RGB color spaces:
                sRGBL;
                AdobeRGBL;
                WideGamutRGBL;
                ProPhotoRGBL;
                DCI_P3_RGBL;
                AppleRGBL;
                REC_709_RGBL;
                REC_2020_RGBL;
            Supported list of non-RGB color spaces:
                Lab_D50_2;
                Lab_D65_2;
                XYZ_D50_2;
                XYZ_D65_2;
                XYZ_D65_10;
                XYZ_D50_10;
                XYZ_A_2;
                XYZ_A_10;
                XYZ_D55_2;
                XYZ_D55_10;
                XYZ_D75_2;
                XYZ_D75_10;
                XYZ_E_2;
                XYZ_E_10;
                Lab_D65_10;
                Lab_D50_10;
                Lab_A_2;
                Lab_A_10;
                Lab_D55_2;
                Lab_D55_10;
                Lab_D75_2;
                Lab_D75_10;
                Lab_E_2;
                Lab_E_10;
    ---------------------------------------------------
    Abbr.
        src, s: source;
        dst, d: destination;
        io: illuminant & observer;
        sio, dio: source of io; destination of io;
        rgbl: linear RGB
        cs: color space;
        cc: Colorchecker;
        M, m: matrix
        ccm: color correction matrix;
        cam: chromatic adaption matrix;
*/


/** @brief Enum of the possible types of ccm.
*/
enum CCM_TYPE
{
    CCM_3x3,
    CCM_4x3
};

/** @brief Enum of the possible types of initial method.
*/
enum INITIAL_METHOD_TYPE
{
    WHITE_BALANCE,
    LEAST_SQUARE
};
 /** @brief  Macbeth and Vinyl ColorChecker with 2deg D50 .
    */
enum CONST_COLOR {
    Macbeth,
    Vinyl,
    DigitalSG
};
enum COLOR_SPACE {
    sRGB,
    sRGBL,
    AdobeRGB,
    AdobeRGBL,
    WideGamutRGB,
    WideGamutRGBL,
    ProPhotoRGB,
    ProPhotoRGBL,
    DCI_P3_RGB,
    DCI_P3_RGBL,
    AppleRGB,
    AppleRGBL,
    REC_709_RGB,
    REC_709_RGBL,
    REC_2020_RGB,
    REC_2020_RGBL,
    XYZ_D65_2,
    XYZ_D65_10,
    XYZ_D50_2,
    XYZ_D50_10,
    XYZ_A_2,
    XYZ_A_10,
    XYZ_D55_2,
    XYZ_D55_10,
    XYZ_D75_2,
    XYZ_D75_10,
    XYZ_E_2,
    XYZ_E_10,
    Lab_D65_2,
    Lab_D65_10,
    Lab_D50_2,
    Lab_D50_10,
    Lab_A_2,
    Lab_A_10,
    Lab_D55_2,
    Lab_D55_10,
    Lab_D75_2,
    Lab_D75_10,
    Lab_E_2,
    Lab_E_10
};
enum LINEAR_TYPE
{
    IDENTITY_,
    GAMMA,
    COLORPOLYFIT,
    COLORLOGPOLYFIT,
    GRAYPOLYFIT,
    GRAYLOGPOLYFIT
};

/** @brief Enum of possibale functions to calculate the distance between
           colors.see https://en.wikipedia.org/wiki/Color_difference for details;*/
enum DISTANCE_TYPE
{
    CIE76,
    CIE94_GRAPHIC_ARTS,
    CIE94_TEXTILES,
    CIE2000,
    CMC_1TO1,
    CMC_2TO1,
    RGB,
    RGBL
};

/** @brief Core class of ccm model.
           produce a ColorCorrectionModel instance for inference.
*/

class CV_EXPORTS_W ColorCorrectionModel
{
public:
    ColorCorrectionModel(Mat src_, CONST_COLOR constcolor);
    ColorCorrectionModel(Mat src_, Mat colors_, COLOR_SPACE ref_cs_);
    ColorCorrectionModel(Mat src_, Mat colors_, COLOR_SPACE cs_, Mat colored_);
    CV_WRAP class Impl;
    CV_WRAP Ptr<Impl> p;
    CV_WRAP void setColorSpace(COLOR_SPACE cs_);
    CV_WRAP void setCCM(CCM_TYPE ccm_type_);
    CV_WRAP void setDistance(DISTANCE_TYPE distance_);
    CV_WRAP void setLinear(LINEAR_TYPE linear_type);
    CV_WRAP void setLinearGamma(double gamma);
    CV_WRAP void setLinearDegree(int deg);
    CV_WRAP void setSaturatedThreshold(double lower, double upper);//std::vector<double> saturated_threshold
    CV_WRAP void setWeightsList(Mat weights_list);
    CV_WRAP void setWeightCoeff(double weights_coeff);
    CV_WRAP void setInitialMethod(INITIAL_METHOD_TYPE initial_method_type);
    CV_WRAP void setMaxCount(int max_count_);
    CV_WRAP void setEpsilon(double epsilon_);
    CV_WRAP void run();

    // /** @brief Infer using fitting ccm.
    //     @param img the input image, type of cv::Mat.
    //     @param islinear default false.
    //     @return the output array, type of cv::Mat.
    // */
    CV_WRAP Mat infer(const Mat& img, bool islinear = false);


};

} // namespace ccm
} // namespace cv

#endif