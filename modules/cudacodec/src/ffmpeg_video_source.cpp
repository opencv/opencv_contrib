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

#ifndef CV_FOURCC_MACRO
#define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))
#endif

static std::string fourccToString(int fourcc)
{
    union {
        int u32;
        unsigned char c[4];
    } i32_c;
    i32_c.u32 = fourcc;
    return cv::format("%c%c%c%c",
        (i32_c.c[0] >= ' ' && i32_c.c[0] < 128) ? i32_c.c[0] : '?',
        (i32_c.c[1] >= ' ' && i32_c.c[1] < 128) ? i32_c.c[1] : '?',
        (i32_c.c[2] >= ' ' && i32_c.c[2] < 128) ? i32_c.c[2] : '?',
        (i32_c.c[3] >= ' ' && i32_c.c[3] < 128) ? i32_c.c[3] : '?');
}

static
Codec FourccToCodec(int codec)
{
    switch (codec)
    {
    case CV_FOURCC_MACRO('m', 'p', 'e', 'g'): // fallthru
    case CV_FOURCC_MACRO('M', 'P', 'G', '1'): return MPEG1;
    case CV_FOURCC_MACRO('M', 'P', 'G', '2'): return MPEG2;
    case CV_FOURCC_MACRO('X', 'V', 'I', 'D'): // fallthru
    case CV_FOURCC_MACRO('D', 'I', 'V', 'X'): return MPEG4;
    case CV_FOURCC_MACRO('W', 'V', 'C', '1'): return VC1;
    case CV_FOURCC_MACRO('H', '2', '6', '4'): // fallthru
    case CV_FOURCC_MACRO('h', '2', '6', '4'): // fallthru
    case CV_FOURCC_MACRO('a', 'v', 'c', '1'): return H264;
    case CV_FOURCC_MACRO('H', '2', '6', '5'): // fallthru
    case CV_FOURCC_MACRO('h', '2', '6', '5'): // fallthru
    case CV_FOURCC_MACRO('h', 'e', 'v', 'c'): return HEVC;
    case CV_FOURCC_MACRO('M', 'J', 'P', 'G'): return JPEG;
    case CV_FOURCC_MACRO('V', 'P', '8', '0'): return VP8;
    case CV_FOURCC_MACRO('V', 'P', '9', '0'): return VP9;
    default:
        break;
    }

    std::string msg = cv::format("Unknown codec FOURCC: 0x%08X (%s)", codec, fourccToString(codec).c_str());
    CV_LOG_WARNING(NULL, msg);
    CV_Error(Error::StsUnsupportedFormat, msg);
}

static
void FourccToChromaFormat(const int pixelFormat, ChromaFormat &chromaFormat, int & nBitDepthMinus8)
{
    switch (pixelFormat)
    {
    case CV_FOURCC_MACRO('I', '4', '2', '0'):
        chromaFormat = YUV420;
        nBitDepthMinus8 = 0;
        break;
    default:
        CV_LOG_WARNING(NULL, cv::format("ChromaFormat not recognized: 0x%08X (%s). Assuming I420", pixelFormat, fourccToString(pixelFormat).c_str()));
        chromaFormat = YUV420;
        nBitDepthMinus8 = 0;
        break;
    }
}

cv::cudacodec::detail::FFmpegVideoSource::FFmpegVideoSource(const String& fname)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        CV_Error(Error::StsNotImplemented, "FFmpeg backend not found");

    cap.open(fname, CAP_FFMPEG);
    if (!cap.isOpened())
        CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");

    if (!cap.set(CAP_PROP_FORMAT, -1))  // turn off video decoder (extract stream)
        CV_Error(Error::StsUnsupportedFormat, "Fetching of RAW video streams is not supported");
    CV_Assert(cap.get(CAP_PROP_FORMAT) == -1);

    int codec = (int)cap.get(CAP_PROP_FOURCC);
    int pixelFormat = (int)cap.get(CAP_PROP_CODEC_PIXEL_FORMAT);

    format_.codec = FourccToCodec(codec);
    format_.height = cap.get(CAP_PROP_FRAME_HEIGHT);
    format_.width = cap.get(CAP_PROP_FRAME_WIDTH);
    FourccToChromaFormat(pixelFormat, format_.chromaFormat, format_.nBitDepthMinus8);
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

bool cv::cudacodec::detail::FFmpegVideoSource::getNextPacket(unsigned char** data, size_t* size)
{
    cap >> rawFrame;
    *data = rawFrame.data;
    *size = rawFrame.total();
    return *size != 0;
}

#endif // HAVE_CUDA
