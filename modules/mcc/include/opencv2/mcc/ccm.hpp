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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/mcc/linearize.hpp"

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
                Macbeth: Macbeth ColorChecker with 2deg D50;
                Vinyl: DKK ColorChecker with 2deg D50;
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
    colorspace :
            the absolute color space that detected colors convert to;
            NOTICE: it should be some RGB color space;
                    For the list of RGB color spaces supported, see the notes below;
            type: enum COLOR_SPACE;
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


/* *\ brief Enum of the possible types of ccm.
*/
enum CCM_TYPE
{
    CCM_3x3,
    CCM_4x3
};

/* *\ brief Enum of the possible types of initial method.
*/
enum INITIAL_METHOD_TYPE
{
    WHITE_BALANCE,
    LEAST_SQUARE
};


/* *\ brief Core class of ccm model.
    *        produce a ColorCorrectionModel instance for inference.
*/

class CV_EXPORTS_W ColorCorrectionModel
{
public:
    // detected colors, the referenceand the RGB colorspace for conversion
    Mat src;
    Color dst;
    Mat dist;
    RGBBase_& cs;
    Mat mask;

    // RGBl of detected data and the reference
    Mat src_rgbl;
    Mat dst_rgbl;

    // ccm type and shape
    CCM_TYPE ccm_type;
    int shape;

    // linear method and distance
    std::shared_ptr<Linear> linear;
    DISTANCE_TYPE distance;

    Mat weights;
    Mat ccm;
    Mat ccm0;
    int masked_len;
    double loss;
    int max_count;
    double epsilon;
    ColorCorrectionModel(cv::Mat src_, CONST_COLOR constcolor,  COLOR_SPACE cs_ = sRGB, CCM_TYPE ccm_type_ = CCM_3x3, DISTANCE_TYPE distance_ = CIE2000, LINEAR_TYPE linear_type = GAMMA,
        double gamma = 2.2, int deg = 3, std::vector<double> saturated_threshold = { 0, 0.98 }, cv::Mat weights_list = Mat(), double weights_coeff = 0,
        INITIAL_METHOD_TYPE initial_method_type = LEAST_SQUARE, int max_count_ = 5000, double epsilon_ = 1.e-4);

    ColorCorrectionModel(cv::Mat src_, Mat colors_, COLOR_SPACE  ref_cs_, COLOR_SPACE cs_ = sRGB, CCM_TYPE ccm_type_ = CCM_3x3, DISTANCE_TYPE distance_ = CIE2000, LINEAR_TYPE linear_type = GAMMA,
        double gamma = 2.2, int deg = 3, std::vector<double> saturated_threshold = { 0, 0.98 }, cv::Mat weights_list = Mat(), double weights_coeff = 0,
        INITIAL_METHOD_TYPE initial_method_type = LEAST_SQUARE, int max_count_ = 5000, double epsilon_ = 1.e-4);

    ColorCorrectionModel(cv::Mat src_, Color dst_, COLOR_SPACE cs_ = sRGB, CCM_TYPE ccm_type_ = CCM_3x3, DISTANCE_TYPE distance_ = CIE2000, LINEAR_TYPE linear_type = GAMMA,
        double gamma = 2.2, int deg = 3, std::vector<double> saturated_threshold = { 0, 0.98 }, cv::Mat weights_list = Mat(), double weights_coeff = 0,
        INITIAL_METHOD_TYPE initial_method_type = LEAST_SQUARE, int max_count_ = 5000, double epsilon_ = 1.e-4);

    ColorCorrectionModel(Mat src_, Color dst_, RGBBase_& cs_ , CCM_TYPE ccm_type_ = CCM_3x3, DISTANCE_TYPE distance_ = CIE2000, LINEAR_TYPE linear_type = GAMMA,
        double gamma = 2.2, int deg = 3, std::vector<double> saturated_threshold = { 0, 0.98 }, Mat weights_list = Mat(), double weights_coeff = 0,
        INITIAL_METHOD_TYPE initial_method_type = LEAST_SQUARE, int max_count_ = 5000, double epsilon_ = 1.e-4);


    /* *\ brief Make no change for CCM_3x3.
        *        convert cv::Mat A to [A, 1] in CCM_4x3.
        *\ param inp the input array, type of cv::Mat.
        *\ return the output array, type of cv::Mat
    */
    Mat prepare(const Mat& inp);

    /* *\ brief Calculate weights and mask.
        *\ param weights_list the input array, type of cv::Mat.
        *\ param weights_coeff type of double.
        *\ param saturate_list the input array, type of cv::Mat.
    */
    void calWeightsMasks(Mat weights_list, double weights_coeff, Mat saturate_mask);

    /* *\ brief Fitting nonlinear - optimization initial value by white balance.
        *        see CCM.pdf for details.
        *\ return the output array, type of Mat
    */
    Mat initialWhiteBalance(void);

    /* *\ brief Fitting nonlinear-optimization initial value by least square.
        *        see CCM.pdf for details
        *\ param fit if fit is True, return optimalization for rgbl distance function.
    */
    void initialLeastSquare(bool fit = false);

    double calc_loss_(Color color);
    double calc_loss(const Mat ccm_);

    /* *\ brief Fitting ccm if distance function is associated with CIE Lab color space.
        *        see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
        *        Set terminal criteria for solver is possible.
    */
    void fitting(void);

    /* *\ brief Infer using fitting ccm.
        *\ param img the input image, type of cv::Mat.
        *\ return the output array, type of cv::Mat.
    */
    Mat infer(const Mat& img, bool islinear = false);

    /* *\ brief Infer image and output as an BGR image with uint8 type.
        *        mainly for test or debug.
        *        input size and output size should be 255.
        *\ param imgfile path name of image to infer.
        *\ param islinear if linearize or not.
        *\ return the output array, type of cv::Mat.
    */
    Mat inferImage(std::string imgfile, bool islinear = false);

    /* *\ brief Loss function base on cv::MinProblemSolver::Function.
        *        see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
    */
    class LossFunction : public cv::MinProblemSolver::Function
    {
    public:
        ColorCorrectionModel* ccm_loss;
        LossFunction(ColorCorrectionModel* ccm) : ccm_loss(ccm) {};

        /* *\ brief Reset dims to ccm->shape.
        */
        int getDims() const CV_OVERRIDE
        {
            return ccm_loss->shape;
        }

        /* *\ brief Reset calculation.
        */
        double calc(const double* x) const CV_OVERRIDE
        {
            Mat ccm_(ccm_loss->shape, 1, CV_64F);
            for (int i = 0; i < ccm_loss->shape; i++)
            {
                ccm_.at<double>(i, 0) = x[i];
            }
            ccm_ = ccm_.reshape(0, ccm_loss->shape / 3);
            return ccm_loss->calc_loss(ccm_);
        }
    };
};

} // namespace ccm
} // namespace cv

#endif