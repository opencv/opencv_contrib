// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
    dst :
            the reference colors;
            NOTICE: Built-in color card or custom color card are supported;
                    Built-in:
                        Macbeth_D50_2: Macbeth ColorChecker with 2deg D50;
                        Macbeth_D65_2: Macbeth ColorChecker with 2deg D65;
                    Custom:
                        You should use Color
                        For the list of color spaces supported, see the notes below;
                        If the color type is some RGB, the format is RGB not BGR, and the color values are in [0, 1];
            type: Color;
    colorspace :
            the absolute color space that detected colors convert to;
            NOTICE: it should be some RGB color space;
                    For the list of RGB color spaces supported, see the notes below;
            type: ColorSpace;
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

            Supported IO (You can use Lab(io) or XYZ(io) to create color space):
                A_2;
                A_10;
                D50_2;
                D50_10;
                D55_2;
                D55_10;
                D65_2;
                D65_10;
                D75_2;
                D75_10;
                E_2;
                E_10;


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
class ColorCorrectionModel
{
public:
    // detected colors, the referenceand the RGB colorspace for conversion
    cv::Mat src;
    Color dst;
    RGBBase_ &cs;

    // ccm type and shape
    CCM_TYPE ccm_type;
    int shape;

    // linear method and distance
    std::shared_ptr<Linear> linear;
    DISTANCE_TYPE distance;

    cv::Mat weights;
    cv::Mat ccm;
    cv::Mat ccm0;

    int max_count;
    double epsilon;

    ColorCorrectionModel(cv::Mat src_, Color dst_, RGBBase_ &cs_, CCM_TYPE ccm_type_, DISTANCE_TYPE distance_, LINEAR_TYPE linear_type,
                         double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list, double weights_coeff,
                         INITIAL_METHOD_TYPE initial_method_type, int max_count_, double epsilon_) :
                         src(src_), dst(dst_), cs(cs_), ccm_type(ccm_type_), distance(distance_), max_count(max_count_), epsilon(epsilon_)
    {
        cv::Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
        this->linear = getLinear(gamma, deg, this->src, this->dst, saturate_mask, this->cs, linear_type);
        calWeightsMasks(weights_list, weights_coeff, saturate_mask);

        src_rgbl = this->linear->linearize(maskCopyTo(this->src, mask));
        this->dst = this->dst[mask];
        dst_rgbl = maskCopyTo(this->dst.to(*(this->cs.l)).colors, mask);

        // make no change for CCM_3x3, make change for CCM_4x3.
        src_rgbl = prepare(src_rgbl);

        // distance function may affect the loss function and the fitting function
        switch (this->distance)
        {
        case cv::ccm::RGBL:
            initialLeastSquare(true);
            break;
        default:
            switch (initial_method_type)
            {
            case cv::ccm::WHITE_BALANCE:
                initialWhiteBalance();
                break;
            case cv::ccm::LEAST_SQUARE:
                initialLeastSquare();
                break;
            default:
                throw std::invalid_argument{"Wrong initial_methoddistance_type!"};
                break;
            }
            break;
        }

        fitting();
    }

    /* *\ brief Make no change for CCM_3x3.
       *        convert cv::Mat A to [A, 1] in CCM_4x3.
       *\ param inp the input array, type of cv::Mat.
       *\ return the output array, type of cv::Mat
    */
    cv::Mat prepare(const cv::Mat &inp)
    {
        switch (ccm_type)
        {
        case cv::ccm::CCM_3x3:
            shape = 9;
            return inp;
        case cv::ccm::CCM_4x3:
        {
            shape = 12;
            cv::Mat arr1 = cv::Mat::ones(inp.size(), CV_64F);
            cv::Mat arr_out(inp.size(), CV_64FC4);
            cv::Mat arr_channels[3];
            split(inp, arr_channels);
            merge(std::vector<Mat>{arr_channels[0], arr_channels[1], arr_channels[2], arr1}, arr_out);
            return arr_out;
        }
        default:
            throw std::invalid_argument{"Wrong ccm_type!"};
            break;
        }
    };

    /* *\ brief Fitting nonlinear - optimization initial value by white balance.
       *        see CCM.pdf for details.
       *\ return the output array, type of cv::Mat
    */
    cv::Mat initialWhiteBalance(void)
    {
        cv::Mat schannels[3];
        split(src_rgbl, schannels);
        cv::Mat dchannels[3];
        split(dst_rgbl, dchannels);
        std::vector<double> initial_vec = { sum(dchannels[0])[0] / sum(schannels[0])[0], 0, 0, 0,
                                            sum(dchannels[1])[0] / sum(schannels[1])[0], 0, 0, 0,
                                            sum(dchannels[2])[0] / sum(schannels[2])[0], 0, 0, 0 };
        std::vector<double> initial_vec_(initial_vec.begin(), initial_vec.begin() + shape);
        cv::Mat initial_white_balance = cv::Mat(initial_vec_, true).reshape(0, shape / 3);

        return initial_white_balance;
    };

    /* *\ brief Fitting nonlinear-optimization initial value by least square.
       *        see CCM.pdf for details
       *\ param fit if fit is True, return optimalization for rgbl distance function.
    */
    void initialLeastSquare(bool fit = false)
    {
        cv::Mat A, B, w;
        if (weights.empty())
        {
            A = src_rgbl;
            B = dst_rgbl;
        }
        else
        {
            pow(weights, 0.5, w);
            cv::Mat w_;
            merge(std::vector<Mat>{w, w, w}, w_);
            A = w_.mul(src_rgbl);
            B = w_.mul(dst_rgbl);
        }
        solve(A.reshape(1, A.rows), B.reshape(1, B.rows), ccm0, DECOMP_SVD);

        // if fit is True, return optimalization for rgbl distance function.
        if (fit)
        {
            ccm = ccm0;
            cv::Mat residual = A.reshape(1, A.rows) * ccm.reshape(0, shape / 3) - B.reshape(1, B.rows);
            Scalar s = residual.dot(residual);
            double sum = s[0];
            loss = sqrt(sum / masked_len);
        }
    };

    /* *\ brief Loss function base on cv::MinProblemSolver::Function.
       *        see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
    */
    class LossFunction : public cv::MinProblemSolver::Function
    {
    public:
        ColorCorrectionModel *ccm_loss;
        LossFunction(ColorCorrectionModel *ccm) : ccm_loss(ccm){};

        /* *\ brief Reset dims to ccm->shape.
        */
        int getDims() const CV_OVERRIDE
        {
            return ccm_loss->shape;
        }

        /* *\ brief Reset calculation.
        */
        double calc(const double *x) const CV_OVERRIDE
        {
            cv::Mat ccm(ccm_loss->shape, 1, CV_64F);
            for (int i = 0; i < ccm_loss->shape; i++)
            {
                ccm.at<double>(i, 0) = x[i];
            }
            ccm = ccm.reshape(0, ccm_loss->shape / 3);
            Mat reshapecolor = ccm_loss->src_rgbl.reshape(1, 0) * ccm;
            cv::Mat dist = Color(reshapecolor.reshape(3, 0), ccm_loss->cs).diff(ccm_loss->dst, ccm_loss->distance);
            cv::Mat dist_;
            pow(dist, 2, dist_);
            if (!ccm_loss->weights.empty())
            {
                dist_ = ccm_loss->weights.mul(dist_);
            }
            Scalar ss = sum(dist_);
            return ss[0];
        }
    };

    /* *\ brief Fitting ccm if distance function is associated with CIE Lab color space.
       *        see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
       *        Set terminal criteria for solver is possible.
    */
    void fitting(void)
    {
        cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
        cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
        solver->setFunction(ptr_F);
        cv::Mat reshapeccm = ccm0.reshape(0, 1);
        cv::Mat step = cv::Mat::ones(reshapeccm.size(), CV_64F);
        solver->setInitStep(step * 10);
        /* TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_count, epsilon);
        solver->setTermCriteria(termcrit);*/
        double res = solver->minimize(reshapeccm);
        ccm = reshapeccm.reshape(0, shape);
        loss = pow((res / masked_len), 0.5);
        //std::cout << " ccm " << ccm << std::endl;
        //std::cout << " loss " << loss << std::endl;
    };

    /* *\ brief Infer using fitting ccm.
       *\ param img the input image, type of cv::Mat.
       *\ return the output array, type of cv::Mat.
    */
    cv::Mat infer(const cv::Mat &img, bool islinear = false)
    {
        if (!ccm.data)
        {
            throw "No CCM values!";
        }
        cv::Mat img_lin = linear->linearize(img);
        cv::Mat img_ccm(img_lin.size(), img_lin.type());
        cv::Mat ccm_ = ccm.reshape(0, shape / 3);
        img_ccm = multiple(prepare(img_lin), ccm_);
        if (islinear == true)
        {
            return img_ccm;
        }
        return cs.fromL(img_ccm);
    };

    /* *\ brief Infer image and output as an BGR image with uint8 type.
       *        mainly for test or debug.
       *        input size and output size should be 255.
       *\ param imgfile path name of image to infer.
       *\ param islinear if linearize or not.
       *\ return the output array, type of cv::Mat.
    */
    cv::Mat inferImage(std::string imgfile, bool islinear = false)
    {
        const int inp_size = 255;
        const int out_size = 255;
        cv::Mat img = imread(imgfile);
        cv::Mat img_;
        cvtColor(img, img_, COLOR_BGR2RGB);
        img_.convertTo(img_, CV_64F);
        img_ = img_ / inp_size;
        cv::Mat out = this->infer(img_, islinear);
        cv::Mat out_ = out * out_size;
        out_.convertTo(out_, CV_8UC3);
        cv::Mat img_out = min(max(out_, 0), out_size);
        cv::Mat out_img;
        cvtColor(img_out, out_img, COLOR_RGB2BGR);
        return out_img;
    };

private:
    cv::Mat mask;
    cv::Mat dist;
    int masked_len;
    double loss;

    // RGBl of detected data and the reference
    cv::Mat src_rgbl;
    cv::Mat dst_rgbl;

    /* *\ brief Calculate weights and mask.
       *\ param weights_list the input array, type of cv::Mat.
       *\ param weights_coeff type of double.
       *\ param saturate_list the input array, type of cv::Mat.
    */
    void calWeightsMasks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask)
    {
        // weights
        if (!weights_list.empty())
        {
            weights = weights_list;
        }
        else if (weights_coeff != 0)
        {
            pow(dst.toLuminant(dst.cs.io), weights_coeff, weights);
        }

        // masks
        cv::Mat weight_mask = cv::Mat::ones(src.rows, 1, CV_8U);
        if (!weights.empty())
        {
            weight_mask = weights > 0;
        }
        this->mask = (weight_mask) & (saturate_mask);

        // weights' mask
        if (!weights.empty())
        {
            cv::Mat weights_masked = maskCopyTo(this->weights, this->mask);
            weights = weights_masked / mean(weights_masked);
        }
        masked_len = sum(mask)[0];
    };
};

} // namespace ccm
} // namespace cv


#endif