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
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __ANNF_HPP__
#define __ANNF_HPP__

#include "algo.hpp"

static void plusToMinusUpdate(const cv::Mat &current, cv::Mat &next, const int dx, const int dy)
{
    for (int i = 0; i < next.rows; ++i)
        for (int j = 0; j < next.cols; ++j)
        {
            int y = cv::borderInterpolate(i - dy, next.rows, cv::BORDER_CONSTANT);
            int x = cv::borderInterpolate(j - dx, next.cols, cv::BORDER_CONSTANT);

            next.at<float>(i, j) = -next.at<float>(y, x)
                + current.at<float>(i, j) - current.at<float>(y, x);
        }
}

static void minusToPlusUpdate(const cv::Mat &current, cv::Mat &next, const int dx, const int dy)
{
    for (int i = 0; i < next.rows; ++i)
        for (int j = 0; j < next.cols; ++j)
        {
            int y = cv::borderInterpolate(i - dy, next.rows, cv::BORDER_CONSTANT);
            int x = cv::borderInterpolate(j - dx, next.cols, cv::BORDER_CONSTANT);

            next.at<float>(i, j) = next.at<float>(y, x)
                - current.at<float>(i, j) + current.at<float>(y, x);
        }
}

static void getWHSeries(const cv::Mat &src, cv::Mat &dst, const int nProjections, const int psize = 8)
{
    CV_Assert(nProjections <= psize*psize && src.type() == CV_32FC3);
    CV_Assert( hamming_length(psize) == 1 );

    std::vector <cv::Mat> projections;

    cv::Mat proj;
    cv::boxFilter(proj, proj, CV_32F, cv::Size(psize, psize),
        cv::Point(-1,-1), false, cv::BORDER_REFLECT);

    projections.push_back(proj);

    std::vector <int> snake_idx(1, 0);
    std::vector <int> snake_idy(1, 0);

    for (int k = 1, num = 1; k < psize && num <= nProjections; ++k)
    {
        const int dx[] = { (k % 2 == 0) ? +1 : 0, (k % 2 == 0) ? 0 : -1};
        const int dy[] = { (k % 2 == 0) ? 0 : +1, (k % 2 == 0) ? -1 : 0};

        snake_idx.push_back(snake_idx[num - 1] - dx[1]);
        snake_idy.push_back(snake_idy[num++ - 1] - dy[1]);

        for (int i = 0; i < k && num < nProjections; ++i, ++num)
        {
            snake_idx.push_back(snake_idx[num - 1] + dx[0]);
            snake_idy.push_back(snake_idy[num - 1] + dy[0]);
        }

        for (int i = 0; i < k && num < nProjections; ++i, ++num)
        {
            snake_idx.push_back(snake_idx[num - 1] + dx[1]);
            snake_idy.push_back(snake_idy[num - 1] + dy[1]);
        }
    }

    for (int i = 1; i < nProjections; ++i)
    {
        int dx = (snake_idx[i] - snake_idx[i - 1]);
        int dy = (snake_idy[i] - snake_idy[i - 1]);

        dx <<= hamming_length(psize - 1) - hamming_length(snake_idx[i - 1] ^ snake_idx[i]);
        dy <<= hamming_length(psize - 1) - hamming_length(snake_idy[i - 1] ^ snake_idy[i]);

        if (i % 2 == 0)
            plusToMinusUpdate(proj, proj, dx, dy);
        else
            minusToPlusUpdate(proj, proj, dx, dy);
    }

    cv::merge(projections, dst);
}

#endif /* __ANNF_HPP__ */