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
    const int srcDepth = src.depth();
    const int srcCn = src.channels();
    const size_t elemSize = src.elemSize();
    const size_t elemSize1 = src.elemSize1();

    GpuMat dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);

    const bool isNppiNativelySupported =
      (srcType == CV_8UC1)  || (srcType == CV_8UC3)  || (srcType == CV_8UC4)  ||
      (srcType == CV_16UC1) || (srcType == CV_16UC3) || (srcType == CV_16UC4) ||
      (srcType == CV_16SC1) || (srcType == CV_16SC3) || (srcType == CV_16SC4) ||
      (srcType == CV_32SC1) || (srcType == CV_32SC3) || (srcType == CV_32SC4) ||
      (srcType == CV_32FC1) || (srcType == CV_32FC3) || (srcType == CV_32FC4);
    const bool isElemSizeSupportedByNppi =
      (!(elemSize%1) && ((elemSize/1)<=4)) ||
      (!(elemSize%2) && ((elemSize/2)<=4)) ||
      (!(elemSize%4) && ((elemSize/4)<=4)) ||
      (!(elemSize%8) && ((elemSize/8)<=2));
    const bool isElemSizeSupportedByGridTranspose =
      (elemSize == 1) || (elemSize == 2) || (elemSize == 4) || (elemSize == 8);
    const bool isSupported = isNppiNativelySupported || isElemSizeSupportedByNppi || isElemSizeSupportedByGridTranspose;

    if (!isSupported)
      CV_Error(Error::StsUnsupportedFormat, "");
    else if (src.empty())
      CV_Error(Error::StsBadArg,"image is empty");

    if ((src.cols == 1) && (dst.cols == 1))
      src.copyTo(dst, stream);
    else if (((src.cols == 1) || (src.rows == 1)) && (src.cols*src.elemSize() == src.step))
      src.reshape(0, src.cols).copyTo(dst, stream);
    else if (isNppiNativelySupported)
    {
        NppStreamHandler h(StreamAccessor::getStream(stream));

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

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

        if (!stream)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }//end if (isNppiNativelySupported)
    else if (isElemSizeSupportedByNppi)
    {
      NppStreamHandler h(StreamAccessor::getStream(stream));

      NppiSize sz;
      sz.width  = src.cols;
      sz.height = src.rows;

      if (!(elemSize%1) && ((elemSize/1)==1))
        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
          dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%1) && ((elemSize/1)==2))
        nppSafeCall( nppiTranspose_16u_C1R(src.ptr<Npp16u>(), static_cast<int>(src.step),
          dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%1) && ((elemSize/1)==3))
        nppSafeCall( nppiTranspose_8u_C3R(src.ptr<Npp8u>(), static_cast<int>(src.step),
          dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%1) && ((elemSize/1)==4))
        nppSafeCall( nppiTranspose_8u_C4R(src.ptr<Npp8u>(), static_cast<int>(src.step),
          dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%2) && ((elemSize/2)==1))
        nppSafeCall( nppiTranspose_16u_C1R(src.ptr<Npp16u>(), static_cast<int>(src.step),
          dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%2) && ((elemSize/2)==2))
        nppSafeCall( nppiTranspose_8u_C4R(src.ptr<Npp8u>(), static_cast<int>(src.step),
          dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%2) && ((elemSize/2)==3))
        nppSafeCall( nppiTranspose_16u_C3R(src.ptr<Npp16u>(), static_cast<int>(src.step),
          dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%2) && ((elemSize/2)==4))
        nppSafeCall( nppiTranspose_16u_C4R(src.ptr<Npp16u>(), static_cast<int>(src.step),
          dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%4) && ((elemSize/4)==1))
        nppSafeCall( nppiTranspose_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step),
          dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%4) && ((elemSize/4)==2))
        nppSafeCall( nppiTranspose_16u_C4R(src.ptr<Npp16u>(), static_cast<int>(src.step),
          dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%4) && ((elemSize/4)==3))
        nppSafeCall( nppiTranspose_32f_C3R(src.ptr<Npp32f>(), static_cast<int>(src.step),
          dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%4) && ((elemSize/4)==4))
        nppSafeCall( nppiTranspose_32f_C4R(src.ptr<Npp32f>(), static_cast<int>(src.step),
          dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%8) && ((elemSize/8)==1))
        nppSafeCall( nppiTranspose_16u_C4R(src.ptr<Npp16u>(), static_cast<int>(src.step),
          dst.ptr<Npp16u>(), static_cast<int>(dst.step), sz) );
      else if (!(elemSize%8) && ((elemSize/8)==2))
        nppSafeCall( nppiTranspose_32f_C4R(src.ptr<Npp32f>(), static_cast<int>(src.step),
          dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

      if (!stream)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }//end if (isElemSizeSupportedByNppi)
    else if (isElemSizeSupportedByGridTranspose)
    {
      if (elemSize == 1)
        gridTranspose(globPtr<unsigned char>(src), globPtr<unsigned char>(dst), stream);
      else if (elemSize == 2)
        gridTranspose(globPtr<unsigned short>(src), globPtr<unsigned short>(dst), stream);
      else if (elemSize == 4)
        gridTranspose(globPtr<signed int>(src), globPtr<signed int>(dst), stream);
      else if (elemSize == 8)
        gridTranspose(globPtr<double>(src), globPtr<double>(dst), stream);
    }//end if (isElemSizeSupportedByGridTranspose)

    syncOutput(dst, _dst, stream);
}

#endif
