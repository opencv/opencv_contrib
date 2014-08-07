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
#include <time.h>

#include "opencv2/xphoto.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"

#include "opencv2/core/types.hpp"
#include "opencv2/core/types_c.h"


#include "photomontage.hpp"
#include "annf.hpp"


namespace cv
{
    template <typename Tp, unsigned int cn>
    static void shiftMapInpaint(const Mat &src, const Mat &mask, Mat &dst)
    {
        const int nTransform = 60; // number of dominant transforms for stitching
        const int psize = 8; // single ANNF patch size

        /** ANNF computation **/
        srand( unsigned(time(NULL)) );

        std::vector <Matx33f> transforms; // dominant transforms
        for (int i = 0; i < nTransform; ++i)
        {
            float dx = float( rand()%src.cols - src.cols/2 );
            float dy = float( rand()%src.rows - src.rows/2 );
            transforms.push_back( Matx33f( 1, 0, dx,
                                           0, 1, dy,
                                           0, 0,  1) );
        }

        /** Warping **/
        std::vector <Mat> images( nTransform + 1 ); // source image transformed with transforms
        std::vector <Mat> masks( nTransform + 1 );  // definition domain for current shift

        Mat_<uchar> invMask = 255 - mask;
        dilate(invMask, invMask, Mat(), Point(-1,-1), 2);

        src.convertTo( images[0], CV_32F );
        mask.copyTo( masks[0] );

        for (int i = 0; i < nTransform; ++i)
        {
            warpPerspective( images[0], images[i + 1], transforms[i],
                             images[0].size(), INTER_LINEAR );

            warpPerspective( masks[0], masks[i + 1], transforms[i],
                             masks[0].size(), INTER_NEAREST);
            masks[i + 1] &= invMask;
        }

        /** Stitching **/
        Mat photomontageResult;
        xphotoInternal::Photomontage < cv::Vec <float, cn> >( images, masks )
            .assignResImage(photomontageResult);
        photomontageResult.convertTo( dst, dst.type() );
    }

    template <typename Tp, unsigned int cn>
    void inpaint(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)
    {
        dst.create( src.size(), src.type() );

        switch ( algorithmType )
        {
            case INPAINT_SHIFTMAP:
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

        switch ( src.type() )
        {
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
    }
}
