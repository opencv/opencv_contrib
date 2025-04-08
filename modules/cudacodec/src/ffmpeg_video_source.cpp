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
    case CV_FOURCC_MACRO('m', 'p', 'g', '1'): return MPEG1;
    case CV_FOURCC_MACRO('m', 'p', 'g', '2'): return MPEG2;
    case CV_FOURCC_MACRO('F', 'M', 'P', '4'): return MPEG4;
    case CV_FOURCC_MACRO('W', 'V', 'C', '1'): return VC1;
    case CV_FOURCC_MACRO('h', '2', '6', '4'): return H264;
    case CV_FOURCC_MACRO('h', 'e', 'v', 'c'): return HEVC;
    case CV_FOURCC_MACRO('M', 'J', 'P', 'G'): return JPEG;
    case CV_FOURCC_MACRO('V', 'P', '8', '0'): return VP8;
    case CV_FOURCC_MACRO('V', 'P', '9', '0'): return VP9;
    case CV_FOURCC_MACRO('A', 'V', '0', '1'): return AV1;
    default:
        break;
    }
    std::string msg = cv::format("Unknown codec FOURCC: 0x%08X (%s)", codec, fourccToString(codec).c_str());
    CV_LOG_WARNING(NULL, msg);
    CV_Error(Error::StsUnsupportedFormat, msg);
}

static
int StartCodeLen(unsigned char* data, const int sz) {
    if (sz >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1)
        return 3;
    else if (sz >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        return 4;
    else
        return 0;
}

bool ParamSetsExist(unsigned char* parameterSets, const int szParameterSets, unsigned char* data, const int szData) {
    const int paramSetStartCodeLen = StartCodeLen(parameterSets, szParameterSets);
    const int packetStartCodeLen = StartCodeLen(data, szData);
    // weak test to see if the parameter set has already been included in the RTP stream
    return paramSetStartCodeLen != 0 && packetStartCodeLen != 0 && parameterSets[paramSetStartCodeLen] == data[packetStartCodeLen];
}

cv::cudacodec::detail::FFmpegVideoSource::FFmpegVideoSource(const String& fname, const std::vector<int>& _videoCaptureParams, const int iMaxStartFrame)
    : videoCaptureParams(_videoCaptureParams)
{
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        CV_Error(Error::StsNotImplemented, "FFmpeg backend not found");

    videoCaptureParams.push_back(CAP_PROP_FORMAT);
    videoCaptureParams.push_back(-1);
    if (!cap.open(fname, CAP_FFMPEG, videoCaptureParams))
        CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");
    CV_Assert(cap.get(CAP_PROP_FORMAT) == -1);
    if (iMaxStartFrame) {
        CV_Assert(cap.set(CAP_PROP_POS_FRAMES, iMaxStartFrame));
        firstFrameIdx = static_cast<int>(cap.get(CAP_PROP_POS_FRAMES));
    }

    const int codecExtradataIndex = static_cast<int>(cap.get(CAP_PROP_CODEC_EXTRADATA_INDEX));
    Mat tmpExtraData;
    if (cap.retrieve(tmpExtraData, codecExtradataIndex) && tmpExtraData.total())
        extraData = tmpExtraData.clone();

    int codec = (int)cap.get(CAP_PROP_FOURCC);
    format_.codec = FourccToCodec(codec);
    format_.height = cap.get(CAP_PROP_FRAME_HEIGHT);
    format_.width = cap.get(CAP_PROP_FRAME_WIDTH);
    format_.displayArea = Rect(0, 0, format_.width, format_.height);
    format_.valid = false;
    format_.fps = cap.get(CAP_PROP_FPS);
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

void cv::cudacodec::detail::FFmpegVideoSource::updateFormat(const FormatInfo& videoFormat)
{
    format_ = videoFormat;
    format_.valid = true;
}

bool cv::cudacodec::detail::FFmpegVideoSource::get(const int propertyId, double& propertyVal) const
{
    propertyVal = cap.get(propertyId);
    if (propertyVal != 0.)
        return true;

    CV_Assert(videoCaptureParams.size() % 2 == 0);
    for (std::size_t i = 0; i < videoCaptureParams.size(); i += 2) {
        if (videoCaptureParams.at(i) == propertyId) {
            propertyVal = videoCaptureParams.at(i + 1);
            return true;
        }
    }

    return false;
}

bool cv::cudacodec::detail::FFmpegVideoSource::getNextPacket(unsigned char** data, size_t* size)
{
    cap >> rawFrame;
    *data = rawFrame.data;
    *size = rawFrame.total();
    if (iFrame++ == 0 && extraData.total()) {
        if (format_.codec == Codec::MPEG4 ||
            ((format_.codec == Codec::H264 || format_.codec == Codec::HEVC) && !ParamSetsExist(extraData.data, extraData.total(), *data, *size)))
        {
            const size_t nBytesToTrimFromData = format_.codec == Codec::MPEG4 ? 3 : 0;
            const size_t newSz = extraData.total() + *size - nBytesToTrimFromData;
            dataWithHeader = Mat(1, newSz, CV_8UC1);
            memcpy(dataWithHeader.data, extraData.data, extraData.total());
            memcpy(dataWithHeader.data + extraData.total(), (*data) + nBytesToTrimFromData, *size - nBytesToTrimFromData);
            *data = dataWithHeader.data;
            *size = newSz;
        }
    }

    return *size != 0;
}

bool cv::cudacodec::detail::FFmpegVideoSource::lastPacketContainsKeyFrame() const
{
    return cap.get(CAP_PROP_LRF_HAS_KEY_FRAME);
}

#endif // HAVE_CUDA
