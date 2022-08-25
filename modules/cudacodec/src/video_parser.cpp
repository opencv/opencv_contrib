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

cv::cudacodec::detail::VideoParser::VideoParser(VideoDecoder* videoDecoder, FrameQueue* frameQueue, const bool allowFrameDrop, const bool udpSource) :
    videoDecoder_(videoDecoder), frameQueue_(frameQueue), allowFrameDrop_(allowFrameDrop)
{
    if (udpSource) maxUnparsedPackets_ = 0;
    CUVIDPARSERPARAMS params;
    std::memset(&params, 0, sizeof(CUVIDPARSERPARAMS));

    params.CodecType              = videoDecoder->codec();
    params.ulMaxNumDecodeSurfaces = 1;
    params.ulMaxDisplayDelay      = 1; // this flag is needed so the parser will push frames out to the decoder as quickly as it can
    params.pUserData              = this;
    params.pfnSequenceCallback    = HandleVideoSequence;    // Called before decoding frames and/or whenever there is a format change
    params.pfnDecodePicture       = HandlePictureDecode;    // Called when a picture is ready to be decoded (decode order)
    params.pfnDisplayPicture      = HandlePictureDisplay;   // Called whenever a picture is ready to be displayed (display order)

    cuSafeCall( cuvidCreateVideoParser(&parser_, &params) );
}

bool cv::cudacodec::detail::VideoParser::parseVideoData(const unsigned char* data, size_t size, const bool rawMode, const bool containsKeyFrame, bool endOfStream)
{
    CUVIDSOURCEDATAPACKET packet;
    std::memset(&packet, 0, sizeof(CUVIDSOURCEDATAPACKET));

    if (endOfStream)
        packet.flags |= CUVID_PKT_ENDOFSTREAM;

    packet.payload_size = static_cast<unsigned long>(size);
    packet.payload = data;

    if (rawMode)
        currentFramePackets.push_back(RawPacket(data, size, containsKeyFrame));

    if (cuvidParseVideoData(parser_, &packet) != CUDA_SUCCESS)
    {
        CV_LOG_ERROR(NULL, "Call to cuvidParseVideoData failed!");
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    ++unparsedPackets_;
    if (maxUnparsedPackets_ && unparsedPackets_ > maxUnparsedPackets_)
    {
        CV_LOG_ERROR(NULL, "Maxium number of packets (" << maxUnparsedPackets_ << ") parsed without decoding a frame or reconfiguring the decoder, if reading from \
            a live source consider initializing with VideoReaderInitParams::udpSource == true.");
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    if (endOfStream)
        frameQueue_->endDecode();

    return !frameQueue_->isEndOfDecode();
}

int CUDAAPI cv::cudacodec::detail::VideoParser::HandleVideoSequence(void* userData, CUVIDEOFORMAT* format)
{
    VideoParser* thiz = static_cast<VideoParser*>(userData);

    thiz->unparsedPackets_ = 0;

    if (format->codec         != thiz->videoDecoder_->codec()       ||
        format->coded_width   != thiz->videoDecoder_->frameWidth()  ||
        format->coded_height  != thiz->videoDecoder_->frameHeight() ||
        format->chroma_format != thiz->videoDecoder_->chromaFormat()||
        format->bit_depth_luma_minus8 != thiz->videoDecoder_->nBitDepthMinus8() ||
        format->min_num_decode_surfaces != thiz->videoDecoder_->nDecodeSurfaces())
    {
        FormatInfo newFormat;
        newFormat.codec = static_cast<Codec>(format->codec);
        newFormat.chromaFormat = static_cast<ChromaFormat>(format->chroma_format);
        newFormat.nBitDepthMinus8 = format->bit_depth_luma_minus8;
        newFormat.ulWidth = format->coded_width;
        newFormat.ulHeight = format->coded_height;
        newFormat.fps = format->frame_rate.numerator / static_cast<float>(format->frame_rate.denominator);
        newFormat.targetSz = thiz->videoDecoder_->getTargetSz();
        newFormat.width = newFormat.targetSz.width ? newFormat.targetSz.width : format->coded_width;
        newFormat.height = newFormat.targetSz.height ? newFormat.targetSz.height : format->coded_height;
        newFormat.srcRoi = thiz->videoDecoder_->getSrcRoi();
        if (newFormat.srcRoi.empty()) {
            format->display_area.right = format->coded_width;
            format->display_area.bottom = format->coded_height;
            newFormat.displayArea = Rect(Point(format->display_area.left, format->display_area.top), Point(format->display_area.right, format->display_area.bottom));
        }
        else
            newFormat.displayArea = newFormat.srcRoi;
        newFormat.targetRoi = thiz->videoDecoder_->getTargetRoi();
        newFormat.ulNumDecodeSurfaces = min(!thiz->allowFrameDrop_ ? max(thiz->videoDecoder_->nDecodeSurfaces(), static_cast<int>(format->min_num_decode_surfaces)) :
            format->min_num_decode_surfaces * 2, 32);
        if (format->progressive_sequence)
            newFormat.deinterlaceMode = Weave;
        else
            newFormat.deinterlaceMode = Adaptive;
        int maxW = 0, maxH = 0;
        // AV1 has max width/height of sequence in sequence header
        if (format->codec == cudaVideoCodec_AV1 && format->seqhdr_data_length > 0)
        {
            CUVIDEOFORMATEX* vidFormatEx = (CUVIDEOFORMATEX*)format;
            maxW = vidFormatEx->av1.max_width;
            maxH = vidFormatEx->av1.max_height;
        }
        if (maxW < (int)format->coded_width)
            maxW = format->coded_width;
        if (maxH < (int)format->coded_height)
            maxH = format->coded_height;
        newFormat.ulMaxWidth = maxW;
        newFormat.ulMaxHeight = maxH;
        thiz->frameQueue_->init(newFormat.ulNumDecodeSurfaces);
        try
        {
            thiz->videoDecoder_->release();
            thiz->videoDecoder_->create(newFormat);
        }
        catch (const cv::Exception&)
        {
            CV_LOG_ERROR(NULL, "Attempt to reconfigure Nvidia decoder failed!");
            thiz->hasError_ = true;
            return false;
        }
    }

    return thiz->videoDecoder_->nDecodeSurfaces();
}

int CUDAAPI cv::cudacodec::detail::VideoParser::HandlePictureDecode(void* userData, CUVIDPICPARAMS* picParams)
{
    VideoParser* thiz = static_cast<VideoParser*>(userData);

    thiz->unparsedPackets_ = 0;

    bool isFrameAvailable = thiz->frameQueue_->waitUntilFrameAvailable(picParams->CurrPicIdx, thiz->allowFrameDrop_);
    if (!isFrameAvailable)
        return false;

    if (!thiz->videoDecoder_->decodePicture(picParams))
    {
        CV_LOG_ERROR(NULL, "Decoding failed!");
        thiz->hasError_ = true;
        return false;
    }

    return true;
}

int CUDAAPI cv::cudacodec::detail::VideoParser::HandlePictureDisplay(void* userData, CUVIDPARSERDISPINFO* picParams)
{
    VideoParser* thiz = static_cast<VideoParser*>(userData);

    thiz->unparsedPackets_ = 0;

    thiz->frameQueue_->enqueue(picParams, thiz->currentFramePackets);
    thiz->currentFramePackets.clear();
    return true;
}

#endif // HAVE_NVCUVID
