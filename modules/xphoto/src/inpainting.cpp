/*M///////////////////////////////////////////////////////////////////////////////////////
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
//                       (3-clause BSD License)
//
// Copyright (C) 2000-2019, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
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
//   * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
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

#include <vector>
#include <stack>
#include <limits>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <time.h>
#include <functional>
#include <string>
#include <tuple>

#include "opencv2/xphoto.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"

#include "opencv2/core/types.hpp"
#include "opencv2/core/types_c.h"

#include "photomontage.hpp"
#include "annf.hpp"
#include "advanced_types.hpp"

namespace cv
{
namespace xphoto
{

    struct fsr_parameters
    {
        // default variables
        int block_size = 16;
        double conc_weighting = 0.5;
        double rhos[4] = { 0.80, 0.70, 0.66, 0.64 };
        double threshold_stddev_Y[3] = { 0.014, 0.030, 0.090 };
        double threshold_stddev_Cx[3] = { 0.006, 0.010, 0.028 };
        // quality profile dependent variables
        int block_size_min, fft_size, max_iter, min_iter, iter_const;
        double orthogonality_correction;
        fsr_parameters(const int quality)
        {
            if (quality == xphoto::INPAINT_FSR_BEST)
            {
                block_size_min = 2;
                fft_size = 64;
                max_iter = 400;
                min_iter = 50;
                iter_const = 2000;
                orthogonality_correction = 0.2;
            }
            else if (quality == xphoto::INPAINT_FSR_FAST)
            {
                block_size_min = 4;
                fft_size = 32;
                max_iter = 100;
                min_iter = 20;
                iter_const = 1000;
                orthogonality_correction = 0.5;
            }
            else
            {
                CV_Error(cv::Error::StsBadArg, "Unknown quality level set, supported: FAST, COMPROMISE, BEST");

            }
        }
    };


    static void
    icvBGR2YCbCr(const cv::Mat& bgr, cv::Mat& Y, cv::Mat& Cb, cv::Mat& Cr)
    {
        // same behavior as matlab rgb2ycbcr when rgb image is of type uint8
        int height = bgr.rows;
        int width = bgr.cols;
        Y = cv::Mat::zeros(height, width, CV_8U);
        Cb = cv::Mat::zeros(height, width, CV_8U);
        Cr = cv::Mat::zeros(height, width, CV_8U);

        cv::Mat channels_bgr[3];
        cv::split(bgr, channels_bgr);
        uchar* rData = (uchar*)channels_bgr[2].data;
        uchar* gData = (uchar*)channels_bgr[1].data;
        uchar* bData = (uchar*)channels_bgr[0].data;
        uchar* yData = (uchar*)Y.data;
        uchar* cbData = (uchar*)Cb.data;
        uchar* crData = (uchar*)Cr.data;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                uchar R = rData[y*channels_bgr[2].step1() + x];
                uchar G = gData[y*channels_bgr[1].step1() + x];
                uchar B = bData[y*channels_bgr[0].step1() + x];
                yData[y*Y.step1() + x] = cv::saturate_cast<uchar>(0.256788235294118 * R + 0.504129411764706 * G + 0.0979058823529412 * B + 16);
                cbData[y*Cb.step1() + x] = cv::saturate_cast<uchar>(-0.148223529411765 * R - 0.290992156862745* G + 0.439215686274510 * B + 128.0);
                crData[y*Cr.step1() + x] = cv::saturate_cast<uchar>(0.439215686274510 * R - 0.367788235294118 * G - 0.0714274509803922 * B + 128);
            }
        }
    }


    static void
    icvYCbCr2BGR(cv::Mat& Y, cv::Mat& Cb, cv::Mat& Cr, cv::Mat& bgr)
    {
        // same behavior as matlab ycbcr2rgb, when ycbcr image is of type uint8
        int height = Y.rows;
        int width = Y.cols;

        uchar* bgrData = (uchar*)bgr.data;
        double* yData = (double*)Y.data;
        double* cbData = (double*)Cb.data;
        double* crData = (double*)Cr.data;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                double Y_curr = yData[y*Y.step1() + x];
                double Cb_curr = cbData[y*Cb.step1() + x];
                double Cr_curr = crData[y*Cr.step1() + x];
                bgrData[y*bgr.step1() + 3 * x] = cv::saturate_cast<uchar>(1.16438356164384 * Y_curr + 2.01723263955646* Cb_curr + 0.000003054 * Cr_curr - 276.836305795032);
                bgrData[y*bgr.step1() + 3 * x + 1] = cv::saturate_cast<uchar>(1.16438356164384 * Y_curr - 0.391762539941450 * Cb_curr - 0.812968292162205 * Cr_curr + 135.575409522967);
                bgrData[y*bgr.step1() + 3 * x + 2] = cv::saturate_cast<uchar>(1.16438356164384 * Y_curr + 3.01124397411008e-07 * Cb_curr + 1.59602688733570 * Cr_curr - 222.921617109194);
            }
        }
    }


    static void
    icvSgnMat(const cv::Mat& src, cv::Mat& dst) {
        dst = cv::Mat::zeros(src.size(), CV_64F);
        double* srcData = (double*)src.data;
        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                double curr_val = srcData[y*src.step1() + x];
                if (curr_val > 0)
                {
                    srcData[y*src.step1() + x] = 1;
                }
                else if (curr_val)
                {
                    srcData[y*src.step1() + x] = -1;
                }
            }
        }
    }


    static double
    icvStandardDeviation(const cv::Mat& distorted_block_2d, const cv::Mat& error_mask_2d) {
        if (cv::countNonZero(error_mask_2d) < 1)
        {
            return NAN; // align to matlab behavior
            // block with no undistorted pixels shouldn't be chosen for processing (only if block_size_min is reached)
        }
        cv::Scalar tmp_stddev, tmp_mean;
        cv::Mat mask8u = error_mask_2d*2;
        mask8u.convertTo(mask8u, CV_8U);
        cv::meanStdDev(distorted_block_2d, tmp_mean, tmp_stddev, mask8u);
        double sigma_n = tmp_stddev[0] / 255;
        if (sigma_n < 0)
        {
            sigma_n = 0;
        }
        else if (sigma_n > 1)
        {
            sigma_n = 1;
        }
        return sigma_n;
    }

    static void
    icvExtrapolateBlock(cv::Mat& distorted_block, cv::Mat& error_mask, fsr_parameters& fsr_params, double rho, double normedStdDev, cv::Mat& extrapolated_block)
    {
        double fft_size = fsr_params.fft_size;
        double orthogonality_correction = fsr_params.orthogonality_correction;
        int M = distorted_block.rows;
        int N = distorted_block.cols;
        int fft_x_offset = cvFloor((fft_size - N) / 2);
        int fft_y_offset = cvFloor((fft_size - M) / 2);

        // weighting function
        cv::Mat w = cv::Mat::zeros(fsr_params.fft_size, fsr_params.fft_size, CV_64F);
        error_mask.copyTo(w(cv::Range(fft_y_offset, fft_y_offset + M), cv::Range(fft_x_offset, fft_x_offset + N)));
        for (int u = 0; u < fft_size; ++u)
        {
            for (int v = 0; v < fft_size; ++v)
            {
                w.at<double>(u, v) *= std::pow(rho, std::sqrt(std::pow(u + 0.5 - (fft_y_offset + M / 2), 2) + std::pow(v + 0.5 - (fft_x_offset + N / 2), 2)));
            }
        }
        cv::Mat W;
        cv::dft(w, W, cv::DFT_COMPLEX_OUTPUT);
        cv::Mat W_padded;
        cv::hconcat(W, W, W_padded);
        cv::vconcat(W_padded, W_padded, W_padded);

        // frequency weighting
        cv::Mat frequency_weighting = cv::Mat::ones(fsr_params.fft_size, fsr_params.fft_size / 2 + 1, CV_64F);
        for (int y = 0; y < fft_size; ++y)
        {
            for (int x = 0; x < (fft_size / 2 + 1); ++x)
            {
                double y2 = fft_size / 2 - std::abs(y - fft_size / 2);
                double x2 = fft_size / 2 - std::abs(x - fft_size / 2);
                frequency_weighting.at<double>(y, x) = 1 - std::sqrt(x2*x2 + y2 * y2)*std::sqrt(2) / fft_size;
            }
        }
        // pad image to fft window size
        cv::Mat f(cv::Size(fsr_params.fft_size, fsr_params.fft_size), CV_64F);
        distorted_block.copyTo(f(cv::Range(fft_y_offset, fft_y_offset + M), cv::Range(fft_x_offset, fft_x_offset + N)));

        // create initial model
        cv::Mat G = cv::Mat::zeros(fsr_params.fft_size, fsr_params.fft_size, CV_64FC2); // complex

        // calculate initial residual
        cv::Mat Rw_tmp, Rw;
        cv::dft(f.mul(w), Rw_tmp, cv::DFT_COMPLEX_OUTPUT);
        Rw = Rw_tmp(cv::Range(0, fsr_params.fft_size), cv::Range(0, fsr_params.fft_size / 2 + 1));

        // estimate ideal number of iterations (GenserIWSSIP2017)
        // calculate stddev if not available (e.g., for smallest block size)
        if (normedStdDev == 0) {
            normedStdDev = icvStandardDeviation(distorted_block, error_mask);
        }
        int num_iters = cvRound(fsr_params.iter_const * normedStdDev);
        if (num_iters < fsr_params.min_iter) {
            num_iters = fsr_params.min_iter;
        }
        else if (num_iters > fsr_params.max_iter) {
            num_iters = fsr_params.max_iter;
        }

        int iter_counter = 0;
        while (iter_counter < num_iters)
        { // Spectral Constrained FSE (GenserIWSSIP2018)
            cv::Mat projection_distances(Rw.size(), CV_64F);
            cv::Mat Rw_mag = cv::Mat(Rw.size(), CV_64F);
            std::vector<cv::Mat> channels(2);
            cv::split(Rw, channels);
            cv::magnitude(channels[0], channels[1], Rw_mag);
            projection_distances = Rw_mag.mul(frequency_weighting);

            double minVal, maxVal;
            int maxLocx = -1;
            int maxLocy = -1;
            cv::minMaxLoc(projection_distances, &minVal, &maxVal);

            for (int y = 0; y < projection_distances.rows; ++y)
            { // assure that first appearance of max Value is selected
                for (int x = 0; x < projection_distances.cols; ++x)
                {
                    if (std::abs(projection_distances.at<double>(y, x) - maxVal) < 0.001)
                    {
                        maxLocy = y;
                        maxLocx = x;
                        break;
                    }
                }
                if (maxLocy != -1)
                {
                    break;
                }
            }
            int bf2select = maxLocy + maxLocx * projection_distances.rows;
            int v = static_cast<int>(std::max(0.0, std::floor(bf2select / fft_size)));
            int u = static_cast<int>(std::max(0, bf2select % fsr_params.fft_size));


            // exclude second half of first and middle col
            if ((v == 0 && u > fft_size / 2) || (v == fft_size / 2 && u > fft_size / 2))
            {
                int u_prev = u;
                u = fsr_params.fft_size - u;
                Rw.at<std::complex<double> >(u, v) = std::conj(Rw.at<std::complex<double> >(u_prev, v));
            }

            // calculate complex conjugate solution
            int u_cj = -1;
            int v_cj = -1;
            // fill first lower col (copy from first upper col)
            if (u >= 1 && u < fft_size / 2 && v == 0)
            {
                u_cj = fsr_params.fft_size - u;
                v_cj = v;
            }
            // fill middle lower col (copy from first middle col)
            if (u >= 1 && u < fft_size / 2 && v == fft_size / 2)
            {
                u_cj = fsr_params.fft_size - u;
                v_cj = v;
            }
            // fill first row right (copy from first row left)
            if (u == 0 && v >= 1 && v < fft_size / 2)
            {
                u_cj = u;
                v_cj = fsr_params.fft_size - v;
            }
            // fill middle row right (copy from middle row left)
            if (u == fft_size / 2 && v >= 1 && v < fft_size / 2)
            {
                u_cj = u;
                v_cj = fsr_params.fft_size - v;
            }
            // fill cell upper right (copy from lower cell left)
            if (u >= fft_size / 2 + 1 && v >= 1 && v < fft_size / 2)
            {
                u_cj = fsr_params.fft_size - u;
                v_cj = fsr_params.fft_size - v;
            }
            // fill cell lower right (copy from upper cell left)
            if (u >= 1 && u < fft_size / 2 && v >= 1 && v < fft_size / 2)
            {
                u_cj = fsr_params.fft_size - u;
                v_cj = fsr_params.fft_size - v;
            }

            /// add coef to model and update residual
            if (u_cj != -1 && v_cj != -1)
            {
                std::complex< double> expansion_coefficient = orthogonality_correction * Rw.at< std::complex<double> >(u, v) / W.at<std::complex<double> >(0, 0);
                G.at< std::complex<double> >(u, v) += fft_size * fft_size * expansion_coefficient;
                G.at< std::complex<double> >(u_cj, v_cj) = std::conj(G.at< std::complex<double> >(u, v));

                cv::Mat expansion_mat(Rw.size(), CV_64FC2, cv::Scalar(expansion_coefficient.real(), expansion_coefficient.imag()));
                cv::Mat W_tmp1 = W_padded(cv::Range(fsr_params.fft_size - u, fsr_params.fft_size - u + Rw.rows), cv::Range(fsr_params.fft_size - v, fsr_params.fft_size - v + Rw.cols));
                cv::Mat W_tmp2 = W_padded(cv::Range(fsr_params.fft_size - u_cj, fsr_params.fft_size - u_cj + Rw.rows), cv::Range(fsr_params.fft_size - v_cj, fsr_params.fft_size - v_cj + Rw.cols));
                cv::Mat res_1(W_tmp1.size(), W_tmp1.type());
                cv::mulSpectrums(expansion_mat, W_tmp1, res_1, 0);
                expansion_mat.setTo(cv::Scalar(expansion_coefficient.real(), -expansion_coefficient.imag()));
                cv::Mat res_2(W_tmp1.size(), W_tmp1.type());
                cv::mulSpectrums(expansion_mat, W_tmp2, res_2, 0);
                Rw -= res_1 + res_2;

                ++iter_counter; // ... as two basis functions were added
            }
            else
            {
                std::complex<double> expansion_coefficient = orthogonality_correction * Rw.at< std::complex<double> >(u, v) / W.at< std::complex<double> >(0, 0);
                G.at< std::complex<double> >(u, v) += fft_size * fft_size * expansion_coefficient;
                cv::Mat expansion_mat(Rw.size(), CV_64FC2, cv::Scalar(expansion_coefficient.real(), expansion_coefficient.imag()));
                cv::Mat W_tmp = W_padded(cv::Range(fsr_params.fft_size - u, fsr_params.fft_size - u + Rw.rows), cv::Range(fsr_params.fft_size - v, fsr_params.fft_size - v + Rw.cols));
                cv::Mat res_tmp(W_tmp.size(), W_tmp.type());
                cv::mulSpectrums(expansion_mat, W_tmp, res_tmp, 0);
                Rw -= res_tmp;

            }
            ++iter_counter;
        }

        // get pixels from model
        cv::Mat g;
        cv::idft(G, g, cv::DFT_SCALE);

        // extract reconstructed pixels
        cv::Mat g_real(M, N, CV_64F);
        for (int x = 0; x < M; ++x)
        {
            for (int y = 0; y < N; ++y)
            {
                g_real.at<double>(x, y) = g.at< std::complex<double> >(fft_y_offset + x, fft_x_offset + y).real();
            }
        }
        g_real.copyTo(extrapolated_block);
        cv::Mat orig_samples;
        error_mask.convertTo(orig_samples, CV_8U);
        distorted_block.copyTo(extrapolated_block, orig_samples); // copy where orig_samples is nonzero
    }


    static void
    icvGetTodoBlocks(cv::Mat& sampled_img, cv::Mat& sampling_mask, std::vector< std::tuple< int, int > >& set_todo, int block_size, int block_size_min, int border_width, double homo_threshold, cv::Mat& set_process_this_block_size, std::vector< std::tuple< int, int > >& set_later, cv::Mat& sigma_n_array)
    {
        std::vector< std::tuple< int, int > > set_now;
        set_later.clear();
        size_t list_length = set_todo.size();
        int img_height = sampled_img.rows;
        int img_width = sampled_img.cols;
        cv::Mat reconstructed_img;
        sampled_img.copyTo(reconstructed_img);

        // calculate block lists
        for (size_t entry = 0; entry < list_length; ++entry)
        {
            int xblock_counter = std::get<0>(set_todo[entry]);
            int yblock_counter = std::get<1>(set_todo[entry]);

            int left_border = std::min(xblock_counter*block_size, border_width);
            int top_border = std::min(yblock_counter*block_size, border_width);
            int right_border = std::max(0, std::min(img_width - (xblock_counter + 1)*block_size, border_width));
            int bottom_border = std::max(0, std::min(img_height - (yblock_counter + 1)*block_size, border_width));

            // extract blocks from images
            cv::Mat distorted_block_2d = reconstructed_img(cv::Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), cv::Range(xblock_counter*block_size - left_border, std::min(img_width, (xblock_counter*block_size + block_size + right_border))));
            cv::Mat error_mask_2d = sampling_mask(cv::Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), cv::Range(xblock_counter*block_size - left_border, std::min(img_width, (xblock_counter*block_size + block_size + right_border))));

            // determine normalized and weighted standard deviation
            if (block_size > block_size_min)
            {
                double sigma_n = icvStandardDeviation(distorted_block_2d, error_mask_2d);
                sigma_n_array.at<double>( yblock_counter, xblock_counter) = sigma_n;

                // homogeneous case
                if (sigma_n < homo_threshold)
                {
                    set_now.emplace_back(xblock_counter, yblock_counter);
                    set_process_this_block_size.at<double>(yblock_counter, xblock_counter) = 255;

                }
                else
                {
                    int yblock_counter_quadernary = yblock_counter * 2;
                    int xblock_counter_quadernary = xblock_counter * 2;
                    int yblock_offset = 0;
                    int xblock_offset = 0;

                    for (int quader_counter = 0; quader_counter < 4; ++quader_counter)
                    {
                        if (quader_counter == 0)
                        {
                            yblock_offset = 0;
                            xblock_offset = 0;
                        }
                        else if (quader_counter == 1)
                        {
                            yblock_offset = 0;
                            xblock_offset = 1;
                        }
                        else if (quader_counter == 2)
                        {
                            yblock_offset = 1;
                            xblock_offset = 0;
                        }
                        else if (quader_counter == 3)
                        {
                            yblock_offset = 1;
                            xblock_offset = 1;
                        }

                        set_later.emplace_back(xblock_counter_quadernary + xblock_offset, yblock_counter_quadernary + yblock_offset);
                    }

                }
            }

        }
    }


    static void
    icvDetermineProcessingOrder(const cv::Mat& _sampled_img, const cv::Mat& _sampling_mask, const int quality, const std::string& channel, cv::Mat& reconstructed_img)
    {
        fsr_parameters fsr_params(quality);
        int block_size = fsr_params.block_size;
        int block_size_max = fsr_params.block_size;
        int block_size_min = fsr_params.block_size_min;
        double conc_weighting = fsr_params.conc_weighting;
        int fft_size = fsr_params.fft_size;
        double rho = fsr_params.rhos[0];
        cv::Mat sampled_img, sampling_mask;
        _sampled_img.convertTo(sampled_img, CV_64F);
        reconstructed_img = sampled_img.clone();

        int height = _sampling_mask.rows;
        int width = _sampling_mask.cols;
        uchar* maskData = (uchar*)_sampling_mask.data;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (maskData[y*_sampling_mask.step1() + x] > 0)
                {
                    maskData[y*_sampling_mask.step1() + x] = 1;
                }
                else
                {
                    maskData[y*_sampling_mask.step1() + x] = 0;
                }
            }
        }
        _sampling_mask.convertTo(sampling_mask, CV_64F);

        double threshold_stddev_LUT[3];
        if (channel == "Y")
        {
            std::copy(fsr_params.threshold_stddev_Y, fsr_params.threshold_stddev_Y + 3, threshold_stddev_LUT);
        }
        else if (channel == "Cx")
        {
            std::copy(fsr_params.threshold_stddev_Cx, fsr_params.threshold_stddev_Cx + 3, threshold_stddev_LUT);
        }
        else
        {
            CV_Error(cv::Error::StsBadArg, "channel type unsupported!");
        }


        double threshold_stddev = threshold_stddev_LUT[0];

        std::vector< std::tuple< int, int > > set_later;
        int img_height = sampled_img.rows;
        int img_width = sampled_img.cols;

        // inital scan of distorted blocks
        std::vector< std::tuple< int, int > > set_todo;
        int blocks_column = cvCeil(static_cast<double>(img_height) / block_size);
        int blocks_line = cvCeil(static_cast<double>(img_width) / block_size);
        for (int y = 0; y < blocks_column; ++y)
        {
            for (int x = 0; x < blocks_line; ++x)
            {
                cv::Mat curr_block = sampling_mask(cv::Range(y*block_size, std::min(img_height, (y + 1)*block_size)), cv::Range(x*block_size, std::min(img_width, (x + 1)*block_size)));
                double min_block, max_block;
                cv::minMaxLoc(curr_block, &min_block, &max_block);
                if (min_block == 0)
                {
                    set_todo.emplace_back(x, y);
                }
            }
        }

        // loop over all distorted blocks and extrapolate them depending on
        // their block size
        int border_width = 0;
        while (block_size >= block_size_min)
        {
            int blocks_per_column = cvCeil(img_height / block_size);
            int blocks_per_line = cvCeil(img_width / block_size);
            cv::Mat nen_array = cv::Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
            cv::Mat proc_array = cv::Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
            cv::Mat sigma_n_array = cv::Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
            cv::Mat set_process_this_block_size = cv::Mat::zeros(blocks_per_column, blocks_per_line, CV_64F);
            if (block_size > block_size_min)
            {
                if (block_size < block_size_max)
                {
                    set_todo = set_later;
                }
                border_width = cvFloor(fft_size - block_size) / 2;
                icvGetTodoBlocks(sampled_img, sampling_mask, set_todo, block_size, block_size_min, border_width, threshold_stddev, set_process_this_block_size, set_later, sigma_n_array);
            }
            else
            {
                set_process_this_block_size.setTo(cv::Scalar(255));
            }

            // if block to be extrapolated, increase nen of neighboring pixels
            for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
            {
                for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
                {
                    cv::Mat curr_block = sampling_mask(cv::Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), cv::Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size)));
                    double min_block, max_block;
                    cv::minMaxLoc(curr_block, &min_block, &max_block);
                    if (min_block == 0)
                    {
                        if (yblock_counter > 0 && xblock_counter > 0)
                        {
                            nen_array.at<double>(yblock_counter - 1, xblock_counter - 1)++;
                        }
                        if (yblock_counter > 0)
                        {
                            nen_array.at<double>(yblock_counter - 1, xblock_counter)++;
                        }
                        if (yblock_counter > 0 && xblock_counter < (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter - 1, xblock_counter + 1)++;
                        }
                        if (xblock_counter > 0)
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter - 1)++;
                        }
                        if (xblock_counter < (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter + 1)++;
                        }
                        if (yblock_counter < (blocks_per_column - 1) && xblock_counter>0)
                        {
                            nen_array.at<double>(yblock_counter + 1, xblock_counter - 1)++;
                        }
                        if (yblock_counter < (blocks_per_column - 1))
                        {
                            nen_array.at<double>(yblock_counter + 1, xblock_counter)++;
                        }
                        if (yblock_counter < (blocks_per_column - 1) && xblock_counter < (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter + 1, xblock_counter + 1)++;
                        }
                    }
                }
            }

            // determine if block itself has to be extrapolated
            for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
            {
                for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
                {
                    cv::Mat curr_block = sampling_mask(cv::Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), cv::Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size)));
                    double min_block, max_block;
                    cv::minMaxLoc(curr_block, &min_block, &max_block);
                    if (min_block != 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter) = -1;
                    }
                    else
                    {
                    // if border block, increase nen respectively
                        if (yblock_counter == 0 && xblock_counter == 0)
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                        }
                        if (yblock_counter == 0 && xblock_counter == (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                        }
                        if (yblock_counter == (blocks_per_column - 1) && xblock_counter == 0)
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                        }
                        if (yblock_counter == (blocks_per_column - 1) && xblock_counter == (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 5;
                        }
                        if (yblock_counter == 0 && xblock_counter != 0 && xblock_counter != (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                        }
                        if (yblock_counter == (blocks_per_column - 1) && xblock_counter != 0 && xblock_counter != (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                        }
                        if (yblock_counter != 0 && yblock_counter != (blocks_per_column - 1) && xblock_counter == 0)
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                        }
                        if (yblock_counter != 0 && yblock_counter != (blocks_per_column - 1) && xblock_counter == (blocks_per_line - 1))
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter) = nen_array.at<double>(yblock_counter, xblock_counter) + 3;
                        }
                    }
                }
            }

            // if all blocks have 8 not extrapolated neighbors, penalize nen of blocks without any known samples by one
            double min_nen_tmp, max_nen_tmp;
            cv::minMaxLoc(nen_array, &min_nen_tmp, &max_nen_tmp);
            if (min_nen_tmp == 8) {
                for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
                {
                    for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
                    {
                        cv::Mat curr_block = sampling_mask(cv::Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), cv::Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size)));
                        double min_block, max_block;
                        cv::minMaxLoc(curr_block, &min_block, &max_block);
                        if (max_block == 0)
                        {
                            nen_array.at<double>(yblock_counter, xblock_counter)++;
                        }
                    }
                }
            }

            // do actual processing per block
            int all_blocks_finished = 0;
            while (all_blocks_finished == 0) {
                // clear proc_array
                proc_array.setTo(cv::Scalar(1));

                // determine blocks to extrapolate
                double min_nen = 99;
                int bl_counter = 0;
                // add all homogeneous blocks that shall be processed to list
                // using same priority
                // begins with highest prioroty or lowest nen array value
                std::vector< std::tuple< int, int > > block_list;
                for (int yblock_counter = 0; yblock_counter < blocks_per_column; ++yblock_counter)
                {
                    for (int xblock_counter = 0; xblock_counter < blocks_per_line; ++xblock_counter)
                    {
                // decision if block contains errors
                        double tmp_val = nen_array.at<double>(yblock_counter, xblock_counter);
                        if (tmp_val >= 0 && tmp_val < min_nen && set_process_this_block_size.at<double>(yblock_counter, xblock_counter) == 255) {
                            bl_counter = 0;
                            block_list.clear();
                            min_nen = tmp_val;
                            proc_array.setTo(cv::Scalar(1));
                        }
                        if (tmp_val == min_nen && proc_array.at<double>(yblock_counter, xblock_counter) != 0 && set_process_this_block_size.at<double>(yblock_counter, xblock_counter) == 0) {
                            nen_array.at<double>(yblock_counter, xblock_counter) = -1;
                        }
                        if (tmp_val == min_nen && proc_array.at<double>(yblock_counter, xblock_counter) != 0 && set_process_this_block_size.at<double>(yblock_counter, xblock_counter) != 0) {
                            block_list.emplace_back(yblock_counter, xblock_counter);
                            bl_counter++;
                            // block neighboring blocks from processing
                            if (yblock_counter > 0 && xblock_counter > 0)
                            {
                                proc_array.at<double>(yblock_counter - 1, xblock_counter - 1) = 0;
                            }
                            if (yblock_counter > 0)
                            {
                                proc_array.at<double>(yblock_counter - 1, xblock_counter) = 0;
                            }
                             if (yblock_counter > 0 && xblock_counter > 0)
                            {
                                proc_array.at<double>(yblock_counter - 1, xblock_counter - 1) = 0;
                            }
                            if (yblock_counter > 0)
                            {
                                proc_array.at<double>(yblock_counter - 1, xblock_counter) = 0;
                            }
                            if (yblock_counter > 0 && xblock_counter < (blocks_per_line - 1))
                            {
                                proc_array.at<double>(yblock_counter - 1, xblock_counter + 1) = 0;
                            }
                            if (xblock_counter > 0)
                            {
                                proc_array.at<double>(yblock_counter, xblock_counter - 1) = 0;
                            }
                            if (xblock_counter < (blocks_per_line - 1))
                            {
                                proc_array.at<double>(yblock_counter, xblock_counter + 1) = 0;
                            }
                            if (yblock_counter < (blocks_per_column - 1) && xblock_counter > 0)
                            {
                                proc_array.at<double>(yblock_counter + 1, xblock_counter - 1) = 0;
                            }
                            if (yblock_counter < (blocks_per_column - 1))
                            {
                                proc_array.at<double>(yblock_counter + 1, xblock_counter) = 0;
                            }
                            if (yblock_counter < (blocks_per_column - 1) && xblock_counter < (blocks_per_line - 1))
                            {
                                proc_array.at<double>(yblock_counter + 1, xblock_counter + 1) = 0;
                            }
                        }
                    }
                }
                int max_bl_counter = bl_counter;
                block_list.emplace_back(-1, -1);
                if (bl_counter == 0)
                {
                    all_blocks_finished = 1;
                }
                // blockwise extrapolation of all blocks that can be processed in parallel
                for (bl_counter = 0; bl_counter < max_bl_counter; ++bl_counter)
                {
                    int yblock_counter = std::get<0>(block_list[bl_counter]);
                    int xblock_counter = std::get<1>(block_list[bl_counter]);

                    // calculation of the extrapolation area's borders
                    int left_border = std::min(xblock_counter*block_size, border_width);
                    int top_border = std::min(yblock_counter*block_size, border_width);
                    int right_border = std::max(0, std::min(img_width - (xblock_counter + 1)*block_size, border_width));
                    int bottom_border = std::max(0, std::min(img_height - (yblock_counter + 1)*block_size, border_width));

                    // extract blocks from images
                    cv::Mat distorted_block_2d = reconstructed_img(cv::Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), cv::Range(xblock_counter*block_size - left_border, std::min(img_width, (xblock_counter*block_size + block_size + right_border))));
                    cv::Mat error_mask_2d = sampling_mask(cv::Range(yblock_counter*block_size - top_border, std::min(img_height, (yblock_counter*block_size + block_size + bottom_border))), cv::Range(xblock_counter*block_size - left_border, std::min(img_width, xblock_counter*block_size + block_size + right_border)));
                    // get actual stddev value as it is needed to estimate the
                    // best number of iterations
                    double sigma_n_a = sigma_n_array.at<double>(yblock_counter, xblock_counter);

                    // actual extrapolation
                    cv::Mat extrapolated_block_2d;
                    icvExtrapolateBlock(distorted_block_2d, error_mask_2d, fsr_params, rho, sigma_n_a, extrapolated_block_2d);

                    // update image and mask
                    extrapolated_block_2d(cv::Range(top_border, extrapolated_block_2d.rows - bottom_border), cv::Range(left_border, extrapolated_block_2d.cols - right_border)).copyTo(reconstructed_img(cv::Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), cv::Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size))));

                    cv::Mat signs;
                    icvSgnMat(error_mask_2d(cv::Range(top_border, error_mask_2d.rows - bottom_border), cv::Range(left_border, error_mask_2d.cols - right_border)), signs);
                    cv::Mat tmp_mask = error_mask_2d(cv::Range(top_border, error_mask_2d.rows - bottom_border), cv::Range(left_border, error_mask_2d.cols - right_border)) + (1 - signs) *conc_weighting;
                    tmp_mask.copyTo(sampling_mask(cv::Range(yblock_counter*block_size, std::min(img_height, (yblock_counter + 1)*block_size)), cv::Range(xblock_counter*block_size, std::min(img_width, (xblock_counter + 1)*block_size))));

                    // update nen-array
                    nen_array.at<double>(yblock_counter, xblock_counter) = -1;
                    if (yblock_counter > 0 && xblock_counter > 0)
                    {
                        nen_array.at<double>(yblock_counter - 1, xblock_counter - 1)--;
                    }
                    if (yblock_counter > 0)
                    {
                        nen_array.at<double>(yblock_counter - 1, xblock_counter)--;
                    }
                    if (yblock_counter > 0 && xblock_counter < blocks_per_line - 1)
                    {
                        nen_array.at<double>(yblock_counter - 1, xblock_counter + 1)--;
                    }
                    if (xblock_counter > 0)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter - 1)--;
                    }
                    if (xblock_counter < blocks_per_line - 1)
                    {
                        nen_array.at<double>(yblock_counter, xblock_counter + 1)--;
                    }
                    if (yblock_counter < blocks_per_column - 1 && xblock_counter>0)
                    {
                        nen_array.at<double>(yblock_counter + 1, xblock_counter - 1)--;
                    }
                    if (yblock_counter < blocks_per_column - 1)
                    {
                        nen_array.at<double>(yblock_counter + 1, xblock_counter)--;
                    }
                    if (yblock_counter < blocks_per_column - 1 && xblock_counter < blocks_per_line - 1)
                    {
                        nen_array.at<double>(yblock_counter + 1, xblock_counter + 1)--;
                    }

                }

            }

            // set parameters for next extrapolation tasks (higher texture)
            block_size = block_size / 2;
            border_width = (fft_size - block_size) / 2;
            if (block_size == 8)
            {
                threshold_stddev = threshold_stddev_LUT[1];
                rho = fsr_params.rhos[1];
            }
            if (block_size == 4)
            {
                threshold_stddev = threshold_stddev_LUT[2];
                rho = fsr_params.rhos[2];
            }
            if (block_size == 2)
            {
                rho = fsr_params.rhos[3];
            }

            // terminate function - no heterogeneous blocks left
            if (set_later.empty())
            {
                break;
            }
        }
    }


    template <typename Tp, unsigned int cn>
    static void shiftMapInpaint( const Mat &_src, const Mat &_mask, Mat &dst,
        const int nTransform = 60, const int psize = 8, const cv::Point2i dsize = cv::Point2i(800, 600) )
    {
        /** Preparing input **/
        cv::Mat src, mask, img, dmask, ddmask;

        const float ls = std::max(/**/ std::min( /*...*/
            std::max(_src.rows, _src.cols)/float(dsize.x),
            std::min(_src.rows, _src.cols)/float(dsize.y)
                                               ), 1.0f /**/);


        cv::resize(_mask, mask, _mask.size()/ls, 0, 0, cv::INTER_NEAREST);
        cv::resize(_src,  src,  _src.size()/ls,  0, 0,    cv::INTER_AREA);

        src.convertTo( img, CV_32F );
        img.setTo(0, ~(mask > 0));

        cv::erode( mask,  dmask, cv::Mat(), cv::Point(-1,-1), 2);
        cv::erode(dmask, ddmask, cv::Mat(), cv::Point(-1,-1), 2);

        std::vector <Point2i> pPath;
        cv::Mat_<int> backref( ddmask.size(), int(-1) );

        for (int i = 0; i < ddmask.rows; ++i)
        {
            uchar *dmask_data = (uchar *) ddmask.template ptr<uchar>(i);
            int *backref_data = (int *) backref.template ptr< int >(i);

            for (int j = 0; j < ddmask.cols; ++j)
                if (dmask_data[j] == 0)
                {
                    backref_data[j] = int(pPath.size());
                     pPath.push_back( cv::Point(j, i) );
                }
        }

        /** ANNF computation **/
        std::vector <cv::Point2i> transforms( nTransform );
        dominantTransforms(img, transforms, nTransform, psize);
        transforms.push_back( cv::Point2i(0, 0) );

        /** Warping **/
        std::vector <std::vector <cv::Vec <float, cn> > > pointSeq( pPath.size() ); // source image transformed with transforms
        std::vector <int> labelSeq( pPath.size() );                                 // resulting label sequence
        std::vector <std::vector <int> >  linkIdx( pPath.size() );                  // neighbor links for pointSeq elements
        std::vector <std::vector <unsigned char > > maskSeq( pPath.size() );        // corresponding mask

        for (size_t i = 0; i < pPath.size(); ++i)
        {
            uchar xmask = dmask.template at<uchar>(pPath[i]);

            for (int j = 0; j < nTransform + 1; ++j)
            {
                cv::Point2i u = pPath[i] + transforms[j];

                unsigned char vmask = 0;
                cv::Vec <float, cn> vimg = 0;

                if ( u.y < src.rows && u.y >= 0
                &&   u.x < src.cols && u.x >= 0 )
                {
                    if ( xmask == 0 || j == nTransform )
                        vmask = mask.template at<uchar>(u);
                    vimg = img.template at<cv::Vec<float, cn> >(u);
                }

                maskSeq[i].push_back(vmask);
                pointSeq[i].push_back(vimg);

                if (vmask != 0)
                    labelSeq[i] = j;
            }

            cv::Point2i  p[] = {
                                 pPath[i] + cv::Point2i(0, +1),
                                 pPath[i] + cv::Point2i(+1, 0)
                               };

            for (uint j = 0; j < sizeof(p)/sizeof(cv::Point2i); ++j)
                if ( p[j].y < src.rows && p[j].y >= 0 &&
                     p[j].x < src.cols && p[j].x >= 0 )
                    linkIdx[i].push_back( backref(p[j]) );
                else
                    linkIdx[i].push_back( -1 );
        }

        /** Stitching **/
        photomontage( pointSeq, maskSeq, linkIdx, labelSeq );

        /** Upscaling **/
        if (ls != 1)
        {
            _src.convertTo( img, CV_32F );

            std::vector <Point2i> __pPath = pPath; pPath.clear();

            cv::Mat_<int> __backref( img.size(), -1 );

            std::vector <std::vector <cv::Vec <float, cn> > > __pointSeq = pointSeq; pointSeq.clear();
            std::vector <int> __labelSeq = labelSeq; labelSeq.clear();
            std::vector <std::vector <int> > __linkIdx = linkIdx; linkIdx.clear();
            std::vector <std::vector <unsigned char > > __maskSeq = maskSeq; maskSeq.clear();

            for (size_t i = 0; i < __pPath.size(); ++i)
            {
                cv::Point2i p[] = {
                    __pPath[i] + cv::Point2i(0, -1),
                    __pPath[i] + cv::Point2i(-1, 0)
                };

                for (uint j = 0; j < sizeof(p)/sizeof(cv::Point2i); ++j)
                    if ( p[j].y < src.rows && p[j].y >= 0 &&
                        p[j].x < src.cols && p[j].x >= 0 )
                        __linkIdx[i].push_back( backref(p[j]) );
                    else
                        __linkIdx[i].push_back( -1 );
            }

            for (size_t k = 0; k < __pPath.size(); ++k)
            {
                int clabel = __labelSeq[k];
                int nearSeam = 0;

                for (size_t i = 0; i < __linkIdx[k].size(); ++i)
                    nearSeam |= ( __linkIdx[k][i] == -1
                        || clabel != __labelSeq[__linkIdx[k][i]] );

                if (nearSeam != 0)
                    for (int i = 0; i < ls; ++i)
                        for (int j = 0; j < ls; ++j)
                        {
                            cv::Point2i u = ls*(__pPath[k] + transforms[__labelSeq[k]]) + cv::Point2i(j, i);

                            pPath.push_back( ls*__pPath[k] + cv::Point2i(j, i) );
                            labelSeq.push_back( 0 );

                            __backref(i, j) = int( pPath.size() );

                            cv::Point2i dv[] = {
                                                 cv::Point2i(0,  0),
                                                 cv::Point2i(-1, 0),
                                                 cv::Point2i(+1, 0),
                                                 cv::Point2i(0, -1),
                                                 cv::Point2i(0, +1)
                                               };

                            std::vector <cv::Vec <float, cn> > pointVec;
                                            std::vector <uchar> maskVec;

                            for (uint q = 0; q < sizeof(dv)/sizeof(cv::Point2i); ++q)
                                if (u.x + dv[q].x >= 0 && u.x + dv[q].x < img.cols
                                &&  u.y + dv[q].y >= 0 && u.y + dv[q].y < img.rows)
                                {
                                    pointVec.push_back(img.template at<cv::Vec <float, cn> >(u + dv[q]));
                                    maskVec.push_back(_mask.template at<uchar>(u + dv[q]));
                                }
                                else
                                {
                                    pointVec.push_back( cv::Vec <float, cn>::all(0) );
                                    maskVec.push_back( 0 );
                                }

                            pointSeq.push_back(pointVec);
                              maskSeq.push_back(maskVec);
                        }
                else
                {
                    cv::Point2i fromIdx = ls*(__pPath[k] + transforms[__labelSeq[k]]),
                                  toIdx = ls*__pPath[k];

                    for (int i = 0; i < ls; ++i)
                    {
                        cv::Vec <float, cn> *from = img.template ptr<cv::Vec <float, cn> >(fromIdx.y + i) + fromIdx.x;
                        cv::Vec <float, cn>   *to = img.template ptr<cv::Vec <float, cn> >(toIdx.y + i) + toIdx.x;

                        for (int j = 0; j < ls; ++j)
                            to[j] = from[j];
                    }
                }
            }


            for (size_t i = 0; i < pPath.size(); ++i)
            {
                cv::Point2i  p[] = {
                    pPath[i] + cv::Point2i(0, +1),
                    pPath[i] + cv::Point2i(+1, 0)
                };

                std::vector <int> linkVec;

                for (uint j = 0; j < sizeof(p)/sizeof(cv::Point2i); ++j)
                    if ( p[j].y < src.rows && p[j].y >= 0 &&
                        p[j].x < src.cols && p[j].x >= 0 )
                        linkVec.push_back( __backref(p[j]) );
                    else
                        linkVec.push_back( -1 );

                linkIdx.push_back(linkVec);
            }

            photomontage( pointSeq, maskSeq, linkIdx, labelSeq );
        }

        /** Writing result **/
        for (size_t i = 0; i < labelSeq.size(); ++i)
        {
            cv::Vec <float, cn> val = pointSeq[i][labelSeq[i]];
            img.template at<cv::Vec <float, cn> >(pPath[i]) = val;
        }
        img.convertTo( dst, dst.type() );
    }

    template <typename Tp, unsigned int cn>
    void inpaint(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)
    {
        dst.create( src.size(), src.type() );

        switch ( algorithmType )
        {
            case xphoto::INPAINT_SHIFTMAP:
                shiftMapInpaint <Tp, cn>(src, mask, dst);
                break;
            default:
                CV_Error_( CV_StsNotImplemented,
                    ("Unsupported algorithm type (=%d)", algorithmType) );
                break;
        }
    }

    /*! The function reconstructs the selected image area from known area.
    *  \param src : source image.
    *  \param mask : inpainting mask, 8-bit 1-channel image. Zero pixels indicate the area that needs to be inpainted.
    *  \param dst : destination image.
    *  \param algorithmType : inpainting method.
    */
    void inpaint(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)
    {
        CV_Assert( mask.channels() == 1 && mask.depth() == CV_8U );
        CV_Assert( src.rows == mask.rows && src.cols == mask.cols );

        switch ( algorithmType )
        {
            case xphoto::INPAINT_SHIFTMAP:
                switch ( src.type() )
                {
                    case CV_8SC1:
                        inpaint <char,   1>( src, mask, dst, algorithmType );
                        break;
                    case CV_8SC2:
                        inpaint <char,   2>( src, mask, dst, algorithmType );
                        break;
                    case CV_8SC3:
                        inpaint <char,   3>( src, mask, dst, algorithmType );
                        break;
                    case CV_8SC4:
                        inpaint <char,   4>( src, mask, dst, algorithmType );
                        break;
                    case CV_8UC1:
                        inpaint <uchar,  1>( src, mask, dst, algorithmType );
                        break;
                    case CV_8UC2:
                        inpaint <uchar,  2>( src, mask, dst, algorithmType );
                        break;
                    case CV_8UC3:
                        inpaint <uchar,  3>( src, mask, dst, algorithmType );
                        break;
                    case CV_8UC4:
                        inpaint <uchar,  4>( src, mask, dst, algorithmType );
                        break;
                    case CV_16SC1:
                        inpaint <short,  1>( src, mask, dst, algorithmType );
                        break;
                    case CV_16SC2:
                        inpaint <short,  2>( src, mask, dst, algorithmType );
                        break;
                    case CV_16SC3:
                        inpaint <short,  3>( src, mask, dst, algorithmType );
                        break;
                    case CV_16SC4:
                        inpaint <short,  4>( src, mask, dst, algorithmType );
                        break;
                    case CV_16UC1:
                        inpaint <ushort, 1>( src, mask, dst, algorithmType );
                        break;
                    case CV_16UC2:
                        inpaint <ushort, 2>( src, mask, dst, algorithmType );
                        break;
                    case CV_16UC3:
                        inpaint <ushort, 3>( src, mask, dst, algorithmType );
                        break;
                    case CV_16UC4:
                        inpaint <ushort, 4>( src, mask, dst, algorithmType );
                        break;
                    case CV_32SC1:
                        inpaint <int,    1>( src, mask, dst, algorithmType );
                        break;
                    case CV_32SC2:
                        inpaint <int,    2>( src, mask, dst, algorithmType );
                        break;
                    case CV_32SC3:
                        inpaint <int,    3>( src, mask, dst, algorithmType );
                        break;
                    case CV_32SC4:
                        inpaint <int,    4>( src, mask, dst, algorithmType );
                        break;
                    case CV_32FC1:
                        inpaint <float,  1>( src, mask, dst, algorithmType );
                        break;
                    case CV_32FC2:
                        inpaint <float,  2>( src, mask, dst, algorithmType );
                        break;
                    case CV_32FC3:
                        inpaint <float,  3>( src, mask, dst, algorithmType );
                        break;
                    case CV_32FC4:
                        inpaint <float,  4>( src, mask, dst, algorithmType );
                        break;
                    case CV_64FC1:
                        inpaint <double, 1>( src, mask, dst, algorithmType );
                        break;
                    case CV_64FC2:
                        inpaint <double, 2>( src, mask, dst, algorithmType );
                        break;
                    case CV_64FC3:
                        inpaint <double, 3>( src, mask, dst, algorithmType );
                        break;
                    case CV_64FC4:
                        inpaint <double, 4>( src, mask, dst, algorithmType );
                        break;
                    default:
                        CV_Error_( CV_StsNotImplemented,
                            ("Unsupported source image format (=%d)",
                            src.type()) );
                        break;
                }
                break;
            case xphoto::INPAINT_FSR_BEST:
            case xphoto::INPAINT_FSR_FAST:
                CV_Assert( src.channels() == 1 || src.channels() == 3 );
                double minRange, maxRange;
                switch ( src.type() )
                {
                    case CV_8UC1:
                        break;
                    case CV_8UC3:
                        break;
                    case CV_16UC1:
                        cv::minMaxLoc(src, &minRange, &maxRange);
                        if (minRange < 0 || maxRange > 65535)
                        {
                            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported source image format!");
                            break;
                        }
                        src.convertTo(src, CV_8UC1, 1/257.0);
                        break;
                    case CV_32FC1:
                    case CV_64FC1:
                        cv::minMaxLoc(src, &minRange, &maxRange);
                        if (minRange < 0 || maxRange > 1)
                        {
                            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported source image format!");
                            break;
                        }
                        src.convertTo(src, CV_8UC1, 255.0);
                        break;
                    case CV_16UC3:
                        cv::minMaxLoc(src, &minRange, &maxRange);
                        if (minRange < 0 || maxRange > 65535)
                        {
                            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported source image format!");
                            break;
                        }
                        src.convertTo(src, CV_8UC3, 1 / 257.0);
                        break;
                    case CV_32FC3:
                    case CV_64FC3:
                        cv::minMaxLoc(src, &minRange, &maxRange);
                        if (minRange < 0 || maxRange > 1)
                        {
                            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported source image format!");
                            break;
                        }
                        src.convertTo(src, CV_8UC3, 255.0);
                        break;
                    default:
                        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported source image format!");
                        break;
                }
                dst.create( src.size(), src.type() );
                if (src.channels() == 1)
                { // grayscale image
                    cv::Mat y_reconstructed;
                    icvDetermineProcessingOrder( src, mask, algorithmType, "Y", y_reconstructed );
                    int height = y_reconstructed.rows;
                    int width = y_reconstructed.cols;
                    double* yData = (double*)y_reconstructed.data;
                    uchar* recData = (uchar*)dst.data;
                    for (int y = 0; y < height; ++y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            recData[y*dst.step1() + x] = cv::saturate_cast<uchar>(yData[y*y_reconstructed.step1() + x]);
                        }
                    }
                }
                else if (src.channels() == 3)
                { // RGB image
                    cv::Mat y, cb, cr;
                    icvBGR2YCbCr( src, y, cb, cr );

                    cv::Mat y_reconstructed, cb_reconstructed, cr_reconstructed;
                    y = y.mul( mask );
                    cb = cb.mul( mask );
                    cr = cr.mul( mask );
                    icvDetermineProcessingOrder( y, mask, algorithmType, "Y", y_reconstructed );
                    icvDetermineProcessingOrder( cb, mask, algorithmType, "Cx", cb_reconstructed );
                    icvDetermineProcessingOrder( cr, mask, algorithmType, "Cx", cr_reconstructed );

                    icvYCbCr2BGR(y_reconstructed, cb_reconstructed, cr_reconstructed, dst);
                }
                break;
            default:
                CV_Error_( CV_StsNotImplemented,
                    ("Unsupported algorithm type (=%d)", algorithmType) );
                break;
        }
    }
}
}
