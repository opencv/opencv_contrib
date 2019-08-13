/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"

#ifdef HAVE_NVCUVID
using namespace cv;
using namespace cv::cudacodec;
using namespace cv::cudacodec::detail;

Codec RawToCodec(RawCodec raw_codec)
{
    switch (raw_codec)
    {
    case RawCodec::VideoCodec_MPEG1 :   return MPEG1;
    case RawCodec::VideoCodec_MPEG2 :   return MPEG2;
    case RawCodec::VideoCodec_MPEG4 :   return MPEG4;
    case RawCodec::VideoCodec_VC1   :   return VC1;
    case RawCodec::VideoCodec_H264  :   return H264;
    case RawCodec::VideoCodec_JPEG  :   return JPEG;
    case RawCodec::VideoCodec_HEVC  :   return HEVC;
    case RawCodec::VideoCodec_VP8   :   return VP8;
    case RawCodec::VideoCodec_VP9   :   return VP9;
    case RawCodec::VideoCodec_YUV420:   return Uncompressed_YUV420;
    case RawCodec::VideoCodec_YV12  :   return Uncompressed_YV12;
    case RawCodec::VideoCodec_NV12  :   return Uncompressed_NV12;
    case RawCodec::VideoCodec_YUYV  :   return Uncompressed_YUYV;
    case RawCodec::VideoCodec_UYVY  :   return Uncompressed_UYVY;
    default:    return NumCodecs;
    }
}

ChromaFormat RawToChromaFormat(RawPixelFormat raw_px_format)
{
    switch (raw_px_format)
    {
    case RawPixelFormat::VideoChromaFormat_Monochrome   :   return Monochrome;
    case RawPixelFormat::VideoChromaFormat_YUV420       :   return YUV420;
    case RawPixelFormat::VideoChromaFormat_YUV422       :   return YUV422;
    case RawPixelFormat::VideoChromaFormat_YUV444       :   return YUV444;
    }
}

cv::cudacodec::detail::FFmpegVideoSource::FFmpegVideoSource(const String& fname)
{
    cap.open(fname);
    if(!cap.isOpened())
        CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");

    RawCodec raw_codec = static_cast<RawCodec>(static_cast<int>(cap.get(CAP_PROP_INT_CODEC)));
    format_.codec = RawToCodec(raw_codec);

    format_.height = cap.get(CAP_PROP_FRAME_HEIGHT);
    format_.width = cap.get(CAP_PROP_FRAME_WIDTH);

    RawPixelFormat raw_px_format = static_cast<RawPixelFormat>(static_cast<int>(cap.get(CAP_PROP_INT_PX_FORMAT)));
    format_.chromaFormat = RawToChromaFormat(raw_px_format);
    format_.nBitDepthMinus8 = -1;
}

cv::cudacodec::detail::FFmpegVideoSource::~FFmpegVideoSource()
{
    if (cap.isOpened())
        cap.release();
}

FormatInfo cv::cudacodec::detail::FFmpegVideoSource::format() const
{
    return format_;
}

bool cv::cudacodec::detail::FFmpegVideoSource::getNextPacket(unsigned char** data, int* size, bool* bEndOfFile)
{
    *bEndOfFile = cap.readRaw(rawFrame);
    *data = rawFrame.data;
    *size = rawFrame.step;
    return *bEndOfFile;
}

#endif // HAVE_CUDA
