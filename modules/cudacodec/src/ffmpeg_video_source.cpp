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

Codec RawToCodec(double rawCodec)
{
    switch (static_cast<RawCodec>(static_cast<int>(rawCodec)))
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
    default:    return NumCodecs;
    }
}

void RawToChromaFormat(const double rawPxFormat, ChromaFormat &chromaFormat, int & nBitDepthMinus8)
{
    switch (static_cast<RawPixelFormat>(static_cast<int>(rawPxFormat)))
    {
    case RawPixelFormat::VideoChromaFormat_Monochrome   :
        chromaFormat =  Monochrome;
        nBitDepthMinus8 = 0;
        break;
    case RawPixelFormat::VideoChromaFormat_YUV420P10LE  :
    case RawPixelFormat::VideoChromaFormat_YUV420P12LE  :
        chromaFormat = YUV420;
        nBitDepthMinus8 = 1;
        break;
    case RawPixelFormat::VideoChromaFormat_YUV420       :
    case RawPixelFormat::VideoChromaFormat_YUVJ420      :
    case RawPixelFormat::VideoChromaFormat_YUVJ422      :   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
    case RawPixelFormat::VideoChromaFormat_YUVJ444      :   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        chromaFormat = YUV420;
        nBitDepthMinus8 = 0;
        break;
    case RawPixelFormat::VideoChromaFormat_YUV422       :
        chromaFormat = YUV422;
        nBitDepthMinus8 = 0;
        break;
    case RawPixelFormat::VideoChromaFormat_YUV444P10LE  :
    case RawPixelFormat::VideoChromaFormat_YUV444P12LE  :
        nBitDepthMinus8 = 0;
    case RawPixelFormat::VideoChromaFormat_YUV444       :
        chromaFormat = YUV444;
        nBitDepthMinus8 = 0;
        break;

    default:
        CV_LOG_WARNING(NULL, "ChromaFormat not recognized. Assuming 420");
        chromaFormat = YUV420;
        nBitDepthMinus8 = 0;
        break;
    }
}

cv::cudacodec::detail::FFmpegVideoSource::FFmpegVideoSource(const String& fname)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        CV_Error(Error::StsNotImplemented, "FFmpeg backend not found");

    cap.open(fname);
    if(!cap.isOpened())
        CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");

    format_.codec = RawToCodec(cap.get(CAP_PROP_INT_CODEC));
    format_.height = cap.get(CAP_PROP_FRAME_HEIGHT);
    format_.width = cap.get(CAP_PROP_FRAME_WIDTH);
    RawToChromaFormat(cap.get(CAP_PROP_INT_PX_FORMAT), format_.chromaFormat, format_.nBitDepthMinus8);
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

bool cv::cudacodec::detail::FFmpegVideoSource::getNextPacket(unsigned char** data, size_t* size, bool* bEndOfFile)
{
    const bool frameRead = cap.read(data, size);
    *bEndOfFile = !frameRead;
    return frameRead;
}

#endif // HAVE_CUDA
