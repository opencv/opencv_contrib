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

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "opencv2/xphoto.hpp"

#include "opencv2/imgproc.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/types.hpp"

namespace cv
{
namespace xphoto
{

    void grayDctDenoising(const Mat &, Mat &, const double, const int);
    void rgbDctDenoising(const Mat &, Mat &, const double, const int);
    void dctDenoising(const Mat &, Mat &, const double, const int);


    namespace
    {
    struct GrayDctDenoisingBandInvoker : public ParallelLoopBody
    {
        GrayDctDenoisingBandInvoker(const Mat& _src, Mat& _dst, double _sigma, int _psize, int _nstripes)
            : src(_src), dst(_dst), psize(_psize), thresh(static_cast<float>(3.0 * _sigma)), nstripes(_nstripes) {}

        void operator()(const Range& range) const CV_OVERRIDE
        {
            const int rows = src.rows;
            const int cols = src.cols;
            const int maxPy = rows - psize;
            const int maxPx = cols - psize;

            const int r0 = (range.start * rows) / nstripes;
            const int r1 = (range.end * rows) / nstripes;
            if (r0 >= r1)
                return;

            Mat dstBand = dst.rowRange(r0, r1);
            dstBand.setTo(0.0f);

            const int pyStart = std::max(0, r0 - psize + 1);
            const int pyEnd = std::min(maxPy, r1);

            Mat patch(psize, psize, CV_32FC1);
            float* patchData = patch.ptr<float>();
            const int patchArea = psize * psize;
            Ptr<hal::DCT2D> dctFwd = hal::DCT2D::create(psize, psize, CV_32F, 0);
            Ptr<hal::DCT2D> dctInv = hal::DCT2D::create(psize, psize, CV_32F, DCT_INVERSE);

            for (int py = pyStart; py < pyEnd; ++py)
            {
                const int dy0 = std::max(0, r0 - py);
                const int dy1 = std::min(psize, r1 - py);

                for (int px = 0; px < maxPx; ++px)
                {
                    for (int dy = 0; dy < psize; ++dy)
                        std::memcpy(patch.ptr<float>(dy), src.ptr<float>(py + dy) + px, static_cast<size_t>(psize) * sizeof(float));

                    dctFwd->apply(patch.data, patch.step, patch.data, patch.step);
                    for (int k = 0; k < patchArea; ++k)
                        patchData[k] *= (std::fabs(patchData[k]) > thresh);
                    dctInv->apply(patch.data, patch.step, patch.data, patch.step);

                    for (int dy = dy0; dy < dy1; ++dy)
                    {
                        const float* pRow = patch.ptr<float>(dy);
                        float* dRow = dst.ptr<float>(py + dy) + px;
                        for (int dx = 0; dx < psize; ++dx)
                            dRow[dx] += pRow[dx];
                    }
                }
            }

            for (int y = r0; y < r1; ++y)
            {
                float* dRow = dst.ptr<float>(y);
                const int wy = std::min(y + 1, maxPy) - std::max(0, y - psize + 1);
                if (wy <= 0)
                {
                    for (int x = 0; x < cols; ++x)
                        dRow[x] = 0.0f;
                    continue;
                }
                for (int x = 0; x < cols; ++x)
                {
                    const int wx = std::min(x + 1, maxPx) - std::max(0, x - psize + 1);
                    const int count = wy * wx;
                    dRow[x] = (count > 0) ? (dRow[x] / static_cast<float>(count)) : 0.0f;
                }
            }
        }

        const Mat& src;
        Mat& dst;
        int psize;
        float thresh;
        int nstripes;

    private:
        GrayDctDenoisingBandInvoker& operator=(const GrayDctDenoisingBandInvoker&);
    };
    } // anonymous namespace

    void grayDctDenoising(const Mat &src, Mat &dst, const double sigma, const int psize)
    {
        CV_Assert( src.type() == CV_MAKE_TYPE(CV_32F, 1) );
        CV_CheckGT(psize, 0, "psize must be positive");
        CV_CheckGT(src.rows, psize, "src.rows must exceed psize");
        CV_CheckGT(src.cols, psize, "src.cols must exceed psize");

        Mat srcIn = src;
        if (dst.data == src.data)
            srcIn = src.clone();

        dst.create(srcIn.size(), CV_32FC1);

        const int maxStripes = std::max(1, std::min(srcIn.rows / std::max(1, psize), getNumThreads() * 4));
        parallel_for_(Range(0, maxStripes),
                      GrayDctDenoisingBandInvoker(srcIn, dst, sigma, psize, maxStripes),
                      maxStripes);
    }

    void rgbDctDenoising(const Mat &src, Mat &dst, const double sigma, const int psize)
    {
        CV_Assert( src.type() == CV_MAKE_TYPE(CV_32F, 3) );

        cv::Matx33f mt(pow(3.0f, -0.5f),  pow(3.0f, -0.5f),       pow(3.0f, -0.5f),
                       pow(2.0f, -0.5f),  0.0f,                 -pow(2.0f, -0.5f),
                       pow(6.0f, -0.5f), -2.0f*pow(6.0f, -0.5f),  pow(6.0f, -0.5f));

        cv::transform(src, dst, mt);

        Mat planeSrc(src.size(), CV_32FC1);
        Mat planeDst(src.size(), CV_32FC1);
        for (int c = 0; c < 3; ++c)
        {
            extractChannel(dst, planeSrc, c);
            grayDctDenoising(planeSrc, planeDst, sigma, psize);
            insertChannel(planeDst, dst, c);
        }

        cv::transform( dst, dst, mt.inv() );
    }

    /*! This function implements simple dct-based image denoising,
	 *	link: http://www.ipol.im/pub/art/2011/ys-dct/
     *
	 *  \param src : source image (rgb, or gray)
     *  \param dst : destination image
     *  \param sigma : expected noise standard deviation
     *  \param psize : size of block side where dct is computed
     */
    void dctDenoising(const Mat &src, Mat &dst, const double sigma, const int psize)
    {
        CV_Assert( src.channels() == 3 || src.channels() == 1 );

        int xtype = CV_MAKE_TYPE( CV_32F, src.channels() );
        if (src.type() == CV_32FC1)
        {
            grayDctDenoising(src, dst, sigma, psize);
            return;
        }
        if (src.type() == CV_32FC3)
        {
            rgbDctDenoising(src, dst, sigma, psize);
            return;
        }

        Mat img;
        src.convertTo(img, xtype);

        if ( img.type() == CV_32FC3 )
            rgbDctDenoising( img, img, sigma, psize );
        else if ( img.type() == CV_32FC1 )
            grayDctDenoising( img, img, sigma, psize );
        else
            CV_Error_( Error::StsNotImplemented,
            ("Unsupported source image format (=%d)", img.type()) );

        img.convertTo( dst, src.type() );
    }

}
}
