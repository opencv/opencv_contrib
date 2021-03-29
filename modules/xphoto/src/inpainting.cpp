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

#include "inpainting_fsr.impl.hpp"

namespace cv
{
namespace xphoto
{
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

    static
    void inpaint_shiftmap(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)
    {
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
        }
    }

void inpaint(const Mat &src, const Mat &mask, Mat &dst, const int algorithmType)
{
    CV_Assert(!src.empty());
    CV_Assert(!mask.empty());
    CV_CheckTypeEQ(mask.type(), CV_8UC1, "");
    CV_Assert(src.rows == mask.rows && src.cols == mask.cols);

    switch (algorithmType)
    {
        case xphoto::INPAINT_SHIFTMAP:
            return inpaint_shiftmap(src, mask, dst, algorithmType);
        case xphoto::INPAINT_FSR_BEST:
        case xphoto::INPAINT_FSR_FAST:
            return inpaint_fsr(src, mask, dst, algorithmType);
    }
    CV_Error_(Error::StsNotImplemented, ("Unsupported inpainting algorithm type (=%d)", algorithmType));
}

}}  // namespace
