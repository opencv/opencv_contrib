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

#ifndef __WHS_HPP__
#define __WHS_HPP__

static inline int hl(int x)
{
    int res = 0;
    while (x)
    {
        res += x&1;
        x >>= 1;
    }
    return res;
}

static inline int rp2(int x)
{
    int res = 1;
    while (res < x)
        res <<= 1;
    return res;
}

template <typename ForwardIterator>
static void generate_snake(ForwardIterator snake, const int n)
{
    cv::Point previous;
    if (n > 0)
    {
        previous = cv::Point(0, 0);
        *snake = previous;
    }

    for (int k = 1, num = 1; num <= n; ++k)
    {
        const cv::Point2i dv[] = { cv::Point2i( !(k&1),   (k&1) ),
                                   cv::Point2i( -(k&1), -!(k&1) ) };

        *snake = previous = previous - dv[1];
        ++num;

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < k && num < n; ++j)
            {
                *snake = previous = previous + dv[i];
                ++num;
            }
    }
}

static void nextProjection(std::vector <cv::Mat> &projections, const cv::Point &A,
                           const cv::Point &B, const int psize)
{
    int xsign = (A.x != B.x)*(hl(A.x&B.x) + (B.x > A.x))&1;
    int ysign = (A.y != B.y)*(hl(A.y&B.y) + (B.y > A.y))&1;
    bool plusToMinusUpdate = xsign || ysign;

    int dx = (A.x != B.x) << ( hl(psize - 1) - hl(A.x ^ B.x) );
    int dy = (A.y != B.y) << ( hl(psize - 1) - hl(A.y ^ B.y) );

    cv::Mat proj = projections[projections.size() - 1],
           nproj = -proj.clone();

    for (int i = dy; i < nproj.rows; ++i)
    {
        float *vxNext = nproj.ptr<float>(i - dy);
        float  *vNext = nproj.ptr<float>(i);

        float *vxCurrent = proj.ptr<float>(i - dy);

        if (plusToMinusUpdate)
            for (int j = dx; j < nproj.cols; ++j)
                vNext[j] += vxCurrent[j - dx] - vxNext[j - dx];
        else
            for (int j = dx; j < nproj.cols; ++j)
                vNext[j] -= vxCurrent[j - dx] - vxNext[j - dx];
    }
    projections.push_back(nproj);
}

static void rgb2whs(const cv::Mat &src, cv::Mat &dst, const int nProjections, const int psize)
{
    CV_Assert(nProjections <= psize*psize && src.type() == CV_32FC1);

    const int npsize = rp2(psize);
    std::vector <cv::Mat> projections;

    cv::Mat img, proj;
    cv::copyMakeBorder(src, img, npsize, npsize, npsize, npsize,
        cv::BORDER_CONSTANT, 0);
    cv::boxFilter(img, proj, CV_32F, cv::Size(npsize, npsize),
        cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    projections.push_back(proj);

    std::vector <cv::Point2i> snake_idx;
    generate_snake(std::back_inserter(snake_idx), nProjections);

    for (int i = 1; i < nProjections; ++i)
        nextProjection(projections, snake_idx[i - 1],
            snake_idx[i], npsize);

    int pad = 0;

    cv::merge(projections, img);
    img(cv::Rect(npsize + pad, npsize + pad, src.cols - pad,
        src.rows - pad)).copyTo(dst);
}


#endif /* __WHS_HPP__ */
