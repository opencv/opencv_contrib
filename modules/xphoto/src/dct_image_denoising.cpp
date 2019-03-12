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
#include "opencv2/core/core_c.h"

#include "opencv2/core/types.hpp"
#include "opencv2/core/types_c.h"

namespace cv
{
namespace xphoto
{

    void grayDctDenoising(const Mat &, Mat &, const double, const int);
    void rgbDctDenoising(const Mat &, Mat &, const double, const int);
    void dctDenoising(const Mat &, Mat &, const double, const int);


    struct grayDctDenoisingInvoker : public ParallelLoopBody
    {
    public:
        grayDctDenoisingInvoker(const Mat &src, std::vector <Mat> &patches, const double sigma, const int psize);
        ~grayDctDenoisingInvoker(){};

        void operator() (const Range &range) const CV_OVERRIDE;

    protected:
        const Mat &src;
        std::vector <Mat> &patches; // image decomposition into sliding patches

        const int psize; // size of block to compute dct
        const double sigma; // expected noise standard deviation
        const double thresh; // thresholding estimate

        void operator =(const grayDctDenoisingInvoker&) const {};
    };

    grayDctDenoisingInvoker::grayDctDenoisingInvoker(const Mat &_src, std::vector <Mat> &_patches,
                                                     const double _sigma, const int _psize)
        : src(_src), patches(_patches), psize(_psize), sigma(_sigma), thresh(3*_sigma) {}

    void grayDctDenoisingInvoker::operator() (const Range &range) const
    {
        for (int i = range.start; i <= range.end - 1; ++i)
        {
            int y = i / (src.cols - psize);
            int x = i % (src.cols - psize);

            Rect patchNum( x, y, psize, psize );

            Mat patch(psize, psize, CV_32FC1);
            src(patchNum).copyTo( patch );

            dct(patch, patch);
            float *data = (float *) patch.data;
            for (int k = 0; k < psize*psize; ++k)
                data[k] *= fabs(data[k]) > thresh;
            idct(patch, patches[i]);
        }
    }

    void grayDctDenoising(const Mat &src, Mat &dst, const double sigma, const int psize)
    {
        CV_Assert( src.type() == CV_MAKE_TYPE(CV_32F, 1) );

        int npixels = (src.rows - psize)*(src.cols - psize);

        std::vector <Mat> patches;
        for (int i = 0; i < npixels; ++i)
            patches.push_back( Mat(psize, psize, CV_32FC1) );
        parallel_for_( cv::Range(0, npixels),
            grayDctDenoisingInvoker(src, patches, sigma, psize) );

        Mat res( src.size(), CV_32FC1, 0.0f ),
            num( src.size(), CV_32FC1, 0.0f );

        for (int k = 0; k < npixels; ++k)
        {
            int i = k / (src.cols - psize);
            int j = k % (src.cols - psize);

            res( Rect(j, i, psize, psize) ) += patches[k];
            num( Rect(j, i, psize, psize) ) += Mat::ones(psize, psize, CV_32FC1);
        }
        res /= num;

        res.convertTo( dst, src.type() );
    }

    void rgbDctDenoising(const Mat &src, Mat &dst, const double sigma, const int psize)
    {
        CV_Assert( src.type() == CV_MAKE_TYPE(CV_32F, 3) );

        cv::Matx33f mt(cvInvSqrt(3.0f),  cvInvSqrt(3.0f),       cvInvSqrt(3.0f),
                       cvInvSqrt(2.0f),  0.0f,                 -cvInvSqrt(2.0f),
                       cvInvSqrt(6.0f), -2.0f*cvInvSqrt(6.0f),  cvInvSqrt(6.0f));

        cv::transform(src, dst, mt);

        std::vector <Mat> mv;
        split(dst, mv);

        for (size_t i = 0; i < mv.size(); ++i)
            grayDctDenoising(mv[i], mv[i], sigma, psize);

        merge(mv, dst);

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
        Mat img( src.size(), xtype );
        src.convertTo(img, xtype);

        if ( img.type() == CV_32FC3 )
            rgbDctDenoising( img, img, sigma, psize );
        else if ( img.type() == CV_32FC1 )
            grayDctDenoising( img, img, sigma, psize );
        else
            CV_Error_( CV_StsNotImplemented,
            ("Unsupported source image format (=%d)", img.type()) );

        img.convertTo( dst, src.type() );
    }

}
}
