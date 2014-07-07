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
    struct grayDctDenoisingInvoker : public ParallelLoopBody
    {
    public:
        grayDctDenoisingInvoker(const Mat_<float> &src, std::vector < Mat_<float> > &patches, const double sigma, const int psize);
        ~grayDctDenoisingInvoker();

        void operator() (const Range &range) const;

    private:
        const Mat_<float> &src;
        std::vector < Mat_<float> > &patches; // image decomposition into sliding patches

        const int psize; // size of block to compute dct
        const double sigma; // expected noise standard deviation
        const double thresh; // thresholding estimate
    };

    grayDctDenoisingInvoker::grayDctDenoisingInvoker(const Mat_<float> &src, std::vector < Mat_<float> > &patches,
                                                     const double sigma, const int psize)
        : src(src), patches(patches), sigma(sigma), thresh(3*sigma), psize(psize) {}
    grayDctDenoisingInvoker::~grayDctDenoisingInvoker(){}

    void grayDctDenoisingInvoker::operator() (const Range &range) const
    {
        for (int i = range.start; i <= range.end - 1; ++i)
        {
            int y = i / (src.cols - psize);
            int x = i % (src.cols - psize);

            Rect patchNum( x, y, psize, psize );

            Mat_<float> patch(psize, psize);
            src(patchNum).copyTo( patch );

            dct(patch, patch);
            float *data = (float *) patch.data;
            for (int k = 0; k < psize*psize; ++k)
                data[k] *= fabs(data[k]) > thresh;
            idct(patch, patches[i]);
        }
    }

    void grayDctDenoising(const Mat_<float> &src, Mat_<float> &dst, const double sigma, const int psize)
    {
        CV_Assert( src.channels() == 1 );

        //Mat_<float> res( src.size(), 0.0f ),
        //            num( src.size(), 0.0f );
        //
        //double threshold = 3*sigma;
        //
        //for (int i = 0; i <= src.rows - psize; ++i)
        //    for (int j = 0; j <= src.cols - psize; ++j)
        //    {
        //        Mat_<float> patch = src( Rect(j, i, psize, psize) ).clone();
        //
        //        dct(patch, patch);
        //        float * ptr = (float *) patch.data;
        //        for (int k = 0; k < psize*psize; ++k)
        //            if (fabs(ptr[k]) < threshold)
        //                ptr[k] = 0.0f;
        //        idct(patch, patch);
        //
        //        res( Rect(j, i, psize, psize) ) += patch;
        //        num( Rect(j, i, psize, psize) ) += Mat_<float>::ones(psize, psize);
        //    }
        //res /= num;
        //
        //res.convertTo( dst, src.type() );

        int npixels = (src.rows - psize)*(src.cols - psize);

        std::vector < Mat_<float> > patches;
        for (int i = 0; i < npixels; ++i)
            patches.push_back( Mat_<float>(psize, psize) );
        parallel_for_( cv::Range(0, npixels),
            grayDctDenoisingInvoker(src, patches, sigma, psize) );

        Mat_<float> res( src.size(), 0.0f ),
                    num( src.size(), 0.0f );

        for (int k = 0; k < npixels; ++k)
        {
            int i = k / (src.cols - psize);
            int j = k % (src.cols - psize);

            res( Rect(j, i, psize, psize) ) += patches[k];
            num( Rect(j, i, psize, psize) ) += Mat_<float>::ones(psize, psize);
        }
        res /= num;

        res.convertTo( dst, src.type() );
    }

    void rgbDctDenoising(const Mat_<Vec3f> &src, Mat_<Vec3f> &dst, const double sigma, const int psize)
    {
        CV_Assert( src.channels() == 3 );

        float M[] = {cvInvSqrt(3),  cvInvSqrt(3),       cvInvSqrt(3),
                     cvInvSqrt(2),  0.0f,              -cvInvSqrt(2),
                     cvInvSqrt(6), -2.0f*cvInvSqrt(6),  cvInvSqrt(6)};

        Mat_<Vec3f>::iterator outIt = dst.begin();

        for (Mat_<Vec3f>::const_iterator it = src.begin(); it != src.end(); ++it, ++outIt)
        {
            Vec3f rgb = *it;
            *outIt = Vec3f(M[0]*rgb[0] + M[1]*rgb[1] + M[2]*rgb[2],
                           M[3]*rgb[0] + M[4]*rgb[1] + M[5]*rgb[2],
                           M[6]*rgb[0] + M[7]*rgb[1] + M[8]*rgb[2]);
        }

        /*************************************/
        std::vector < Mat_<float> > mv;
        split(dst, mv);

        for (int i = 0; i < mv.size(); ++i)
            grayDctDenoising(mv[i], mv[i], sigma, psize);

        merge(mv, dst);
        /*************************************/

        for (Mat_<Vec3f>::iterator it = dst.begin(); it != dst.end(); ++it)
        {
            Vec3f rgb = *it;
            *it = Vec3f(M[0]*rgb[0] + M[3]*rgb[1] + M[6]*rgb[2],
                        M[1]*rgb[0] + M[4]*rgb[1] + M[7]*rgb[2],
                        M[2]*rgb[0] + M[5]*rgb[1] + M[8]*rgb[2]);
        }
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
            rgbDctDenoising( Mat_<Vec3f>(img), Mat_<Vec3f>(img), sigma, psize );
        else if ( img.type() == CV_32FC1 )
            grayDctDenoising( Mat_<float>(img), Mat_<float>(img), sigma, psize );
        else
            CV_Assert( false );

        img.convertTo( dst, src.type() );
    }

}