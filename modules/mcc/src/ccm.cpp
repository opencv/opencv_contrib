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

#include "precomp.hpp"

namespace cv
{
namespace ccm
{

ColorCorrectionModel::ColorCorrectionModel(Mat src_, Mat colors_, const ColorSpace& ref_cs_, RGBBase_& cs_, CCM_TYPE ccm_type_, DISTANCE_TYPE distance_, LINEAR_TYPE linear_type,
    double gamma, int deg, std::vector<double> saturated_threshold, Mat weights_list, double weights_coeff,
    INITIAL_METHOD_TYPE initial_method_type, int max_count_, double epsilon_) :
    ColorCorrectionModel(src_, Color(colors_, ref_cs_), cs_, ccm_type_, distance_, linear_type,
    gamma, deg, saturated_threshold, weights_list, weights_coeff, initial_method_type, max_count_, epsilon_) {}

ColorCorrectionModel::ColorCorrectionModel(Mat src_, Color dst_, RGBBase_& cs_, CCM_TYPE ccm_type_, DISTANCE_TYPE distance_, LINEAR_TYPE linear_type,
    double gamma, int deg, std::vector<double> saturated_threshold, Mat weights_list, double weights_coeff,
    INITIAL_METHOD_TYPE initial_method_type, int max_count_, double epsilon_) :
    src(src_), dst(dst_), cs(cs_), ccm_type(ccm_type_), distance(distance_), max_count(max_count_), epsilon(epsilon_)
{
    Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    this->linear = getLinear(gamma, deg, this->src, this->dst, saturate_mask, this->cs, linear_type);
    calWeightsMasks(weights_list, weights_coeff, saturate_mask);

    src_rgbl = this->linear->linearize(maskCopyTo(this->src, mask));
    dst.colors = maskCopyTo(dst.colors, mask);
    dst_rgbl = this->dst.to(*(this->cs.l)).colors;

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
            throw std::invalid_argument{ "Wrong initial_methoddistance_type!" };
            break;
        }
        break;
    }

    fitting();
}

Mat ColorCorrectionModel::prepare(const Mat& inp)
{
    switch (ccm_type)
    {
    case cv::ccm::CCM_3x3:
        shape = 9;
        return inp;
    case cv::ccm::CCM_4x3:
    {
        shape = 12;
        Mat arr1 = Mat::ones(inp.size(), CV_64F);
        Mat arr_out(inp.size(), CV_64FC4);
        Mat arr_channels[3];
        split(inp, arr_channels);
        merge(std::vector<Mat>{arr_channels[0], arr_channels[1], arr_channels[2], arr1}, arr_out);
        return arr_out;
    }
    default:
        throw std::invalid_argument{ "Wrong ccm_type!" };
        break;
    }
}

void ColorCorrectionModel::calWeightsMasks(Mat weights_list, double weights_coeff, Mat saturate_mask)
{
    // weights
    if (!weights_list.empty())
    {
        weights = weights_list;
    }
    else if (weights_coeff != 0)
    {
        pow(dst.toLuminant(cs.io), weights_coeff, weights);
    }

    // masks
    Mat weight_mask = Mat::ones(src.rows, 1, CV_8U);
    if (!weights.empty())
    {
        weight_mask = weights > 0;
    }
    this->mask = (weight_mask) & (saturate_mask);

    // weights' mask
    if (!weights.empty())
    {
        Mat weights_masked = maskCopyTo(this->weights, this->mask);
        weights = weights_masked / mean(weights_masked)[0];
    }
    masked_len = (int)sum(mask)[0];
}

Mat ColorCorrectionModel::initialWhiteBalance(void)
{
    Mat schannels[3];
    split(src_rgbl, schannels);
    Mat dchannels[3];
    split(dst_rgbl, dchannels);
    std::vector<double> initial_vec = { sum(dchannels[0])[0] / sum(schannels[0])[0], 0, 0, 0,
                                        sum(dchannels[1])[0] / sum(schannels[1])[0], 0, 0, 0,
                                        sum(dchannels[2])[0] / sum(schannels[2])[0], 0, 0, 0 };
    std::vector<double> initial_vec_(initial_vec.begin(), initial_vec.begin() + shape);
    Mat initial_white_balance = Mat(initial_vec_, true).reshape(0, shape / 3);

    return initial_white_balance;
}

void ColorCorrectionModel::initialLeastSquare(bool fit)
{
    Mat A, B, w;
    if (weights.empty())
    {
        A = src_rgbl;
        B = dst_rgbl;
    }
    else
    {
        pow(weights, 0.5, w);
        Mat w_;
        merge(std::vector<Mat>{w, w, w}, w_);
        A = w_.mul(src_rgbl);
        B = w_.mul(dst_rgbl);
    }
    solve(A.reshape(1, A.rows), B.reshape(1, B.rows), ccm0, DECOMP_SVD);

    // if fit is True, return optimalization for rgbl distance function.
    if (fit)
    {
        ccm = ccm0;
        Mat residual = A.reshape(1, A.rows) * ccm.reshape(0, shape / 3) - B.reshape(1, B.rows);
        Scalar s = residual.dot(residual);
        double sum = s[0];
        loss = sqrt(sum / masked_len);
    }
}

double ColorCorrectionModel::calc_loss_(Color color)
{
    Mat distlist = color.diff(dst, distance);
    Color lab = color.to(Lab_D50_2);
    Mat dist_;
    pow(distlist, 2, dist_);
    if (!weights.empty())
    {
        dist_ = weights.mul(dist_);
    }
    Scalar ss = sum(dist_);
    return ss[0];
}

double ColorCorrectionModel::calc_loss(const Mat ccm_)
{
    Mat converted = src_rgbl.reshape(1, 0) * ccm_;
    Color color(converted.reshape(3, 0), *(cs.l));
    return calc_loss_(color);
}

void ColorCorrectionModel::fitting(void)
{
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
    solver->setFunction(ptr_F);
    Mat reshapeccm = ccm0.clone().reshape(0, 1);
    Mat step = Mat::ones(reshapeccm.size(), CV_64F);
    solver->setInitStep(step);
    TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_count, epsilon);
    solver->setTermCriteria(termcrit);
    double res = solver->minimize(reshapeccm);
    ccm = reshapeccm.reshape(0, shape/3);
    loss = pow((res / masked_len), 0.5);
    std::cout << " ccm " << ccm << std::endl;
    std::cout << " loss " << loss << std::endl;
}

Mat ColorCorrectionModel::infer(const Mat& img, bool islinear)
{
    if (!ccm.data)
    {
        throw "No CCM values!";
    }
    Mat img_lin = linear->linearize(img);
    Mat img_ccm(img_lin.size(), img_lin.type());
    Mat ccm_ = ccm.reshape(0, shape / 3);
    img_ccm = multiple(prepare(img_lin), ccm_);
    if (islinear == true)
    {
        return img_ccm;
    }
    return cs.fromL(img_ccm);
}

Mat ColorCorrectionModel::inferImage(std::string imgfile, bool islinear)
{
    const int inp_size = 255;
    const int out_size = 255;
    Mat img = imread(imgfile);
    Mat img_;
    cvtColor(img, img_, COLOR_BGR2RGB);
    img_.convertTo(img_, CV_64F);
    img_ = img_ / inp_size;
    Mat out = this->infer(img_, islinear);
    Mat out_ = out * out_size;
    out_.convertTo(out_, CV_8UC3);
    Mat img_out = min(max(out_, 0), out_size);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    return out_img;
}


} // namespace ccm
} // namespace cv
