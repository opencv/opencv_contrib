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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

void cv::cuda::transpose(InputArray _src, OutputArray _dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);

    const int srcType = src.type();
    const size_t elemSize = src.elemSize();

    GpuMat dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);

    const bool isSupported =
      (elemSize == 1) || (elemSize == 2) || (elemSize == 3) || (elemSize == 4) ||
      (elemSize == 6) || (elemSize == 8) || (elemSize == 12) || (elemSize == 16);

    if (!isSupported)
        CV_Error(Error::StsUnsupportedFormat, "");
    else if (src.empty())
        CV_Error(Error::StsBadArg, "image is empty");

    if ((src.rows == 1) && (src.cols == 1))
        src.copyTo(dst, stream);
    else if (src.rows == 1)
        src.reshape(0, src.cols).copyTo(dst, stream);
    else if ((src.cols == 1) && src.isContinuous())
        src.reshape(0, src.cols).copyTo(dst, stream);
    else
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        if (!stream)
        {
          //native implementation
          if (srcType == CV_8UC1)
            nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
                dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_8UC3)
            nppSafeCall( nppiTranspose_8u_C3R(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_8UC4)
            nppSafeCall( nppiTranspose_8u_C4R(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_16UC1)
            nppSafeCall( nppiTranspose_16u_C1R(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_16UC3)
            nppSafeCall( nppiTranspose_16u_C3R(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_16UC4)
            nppSafeCall( nppiTranspose_16u_C4R(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_16SC1)
            nppSafeCall( nppiTranspose_16s_C1R(src.ptr<Npp16s>(), static_cast<int>(src.step),
              dst.ptr<Npp16s>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_16SC3)
            nppSafeCall( nppiTranspose_16s_C3R(src.ptr<Npp16s>(), static_cast<int>(src.step),
              dst.ptr<Npp16s>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_16SC4)
            nppSafeCall( nppiTranspose_16s_C4R(src.ptr<Npp16s>(), static_cast<int>(src.step),
              dst.ptr<Npp16s>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_32SC1)
            nppSafeCall( nppiTranspose_32s_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_32SC3)
            nppSafeCall( nppiTranspose_32s_C3R(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_32SC4)
            nppSafeCall( nppiTranspose_32s_C4R(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_32FC1)
            nppSafeCall( nppiTranspose_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step),
              dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_32FC3)
            nppSafeCall( nppiTranspose_32f_C3R(src.ptr<Npp32f>(), static_cast<int>(src.step),
              dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
          else if (srcType == CV_32FC4)
            nppSafeCall( nppiTranspose_32f_C4R(src.ptr<Npp32f>(), static_cast<int>(src.step),
              dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
          //reinterpretation
          else if (elemSize == 1)
            nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 2)
            nppSafeCall( nppiTranspose_16u_C1R(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 3)
            nppSafeCall( nppiTranspose_8u_C3R(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 4)
            nppSafeCall( nppiTranspose_32s_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 6)
            nppSafeCall( nppiTranspose_16u_C3R(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 8)
            nppSafeCall( nppiTranspose_16u_C4R(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 12)
            nppSafeCall( nppiTranspose_32s_C3R(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
          else if (elemSize == 16)
            nppSafeCall( nppiTranspose_32s_C4R(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
        }//end if (!stream)
        else//if (stream != 0)
        {
          NppStreamContext ctx;
          nppSafeCall( nppGetStreamContext(&ctx) );
          ctx.hStream = StreamAccessor::getStream(stream);

          //native implementation
          if (srcType == CV_8UC1)
            nppSafeCall( nppiTranspose_8u_C1R_Ctx(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_8UC3)
            nppSafeCall( nppiTranspose_8u_C3R_Ctx(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_8UC4)
            nppSafeCall( nppiTranspose_8u_C4R_Ctx(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_16UC1)
            nppSafeCall( nppiTranspose_16u_C1R_Ctx(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_16UC3)
            nppSafeCall( nppiTranspose_16u_C3R_Ctx(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_16UC4)
            nppSafeCall( nppiTranspose_16u_C4R_Ctx(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_16SC1)
            nppSafeCall( nppiTranspose_16s_C1R_Ctx(src.ptr<Npp16s>(), static_cast<int>(src.step),
              dst.ptr<Npp16s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_16SC3)
            nppSafeCall( nppiTranspose_16s_C3R_Ctx(src.ptr<Npp16s>(), static_cast<int>(src.step),
              dst.ptr<Npp16s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_16SC4)
            nppSafeCall( nppiTranspose_16s_C4R_Ctx(src.ptr<Npp16s>(), static_cast<int>(src.step),
              dst.ptr<Npp16s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_32SC1)
            nppSafeCall( nppiTranspose_32s_C1R_Ctx(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_32SC3)
            nppSafeCall( nppiTranspose_32s_C3R_Ctx(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_32SC4)
            nppSafeCall( nppiTranspose_32s_C4R_Ctx(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_32FC1)
            nppSafeCall( nppiTranspose_32f_C1R_Ctx(src.ptr<Npp32f>(), static_cast<int>(src.step),
              dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_32FC3)
            nppSafeCall( nppiTranspose_32f_C3R_Ctx(src.ptr<Npp32f>(), static_cast<int>(src.step),
              dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, ctx) );
          else if (srcType == CV_32FC4)
            nppSafeCall( nppiTranspose_32f_C4R_Ctx(src.ptr<Npp32f>(), static_cast<int>(src.step),
              dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, ctx) );
          //reinterpretation
          else if (elemSize == 1)
            nppSafeCall( nppiTranspose_8u_C1R_Ctx(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 2)
            nppSafeCall( nppiTranspose_16u_C1R_Ctx(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 3)
            nppSafeCall( nppiTranspose_8u_C3R_Ctx(src.ptr<Npp8u>(), static_cast<int>(src.step),
              dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 4)
            nppSafeCall( nppiTranspose_32s_C1R_Ctx(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 6)
            nppSafeCall( nppiTranspose_16u_C3R_Ctx(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 8)
            nppSafeCall( nppiTranspose_16u_C4R_Ctx(src.ptr<Npp16u>(), static_cast<int>(src.step),
              dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 12)
            nppSafeCall( nppiTranspose_32s_C3R_Ctx(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz, ctx) );
          else if (elemSize == 16)
            nppSafeCall( nppiTranspose_32s_C4R_Ctx(src.ptr<Npp32s>(), static_cast<int>(src.step),
              dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz, ctx) );
        }//end if (stream != 0)
    }//end if
}

#endif
