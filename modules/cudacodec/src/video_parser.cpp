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
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// \author Jeff Daily <jeff.daily@amd.com>
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

#ifdef HAVE_ROCDECODE

cv::cudacodec::detail::VideoParser::VideoParser(VideoDecoder* videoDecoder, FrameQueue* frameQueue, const bool allowFrameDrop, const bool udpSource) :
    videoDecoder_(videoDecoder), frameQueue_(frameQueue), allowFrameDrop_(allowFrameDrop)
{
    if (udpSource) maxUnparsedPackets_ = 0;
    const rocDecVideoCodec rocCodec = videoDecoder->codec();
    // Reject codecs rocDecode does not implement up front, with a clear message.
    // rocDecode declares MPEG1/2/4, VC1, VP8 and JPEG in its enum but its parser
    // currently decodes only AVC/HEVC/AV1/VP9; codecToRocDec already returns
    // NumCodecs for VC1 and the H.264 SVC/MVC extensions. This is the same guard
    // class as the per-device caps query in VideoDecoder::create -- a catchable
    // StsNotImplemented, never a crash -- so the multi-arch binary degrades
    // gracefully on an unsupported codec just as it does on a device with no VCN.
    const bool rocDecodeSupportsCodec = rocCodec == rocDecVideoCodec_AVC || rocCodec == rocDecVideoCodec_HEVC ||
                                        rocCodec == rocDecVideoCodec_AV1 || rocCodec == rocDecVideoCodec_VP9;
    if (!rocDecodeSupportsCodec) {
        CV_Error(Error::StsNotImplemented, std::string("cudacodec::VideoReader: the AMD rocDecode library does not support this codec. ") +
            "rocDecode decodes H.264 (AVC), H.265 (HEVC), AV1 and VP9; it lacks MPEG-1/2/4, VC-1, VP8 and MJPEG (use the FFmpeg cv::VideoCapture backend for those).");
    }
    RocdecParserParams params = {};
    params.codec_type = rocCodec;
    params.max_num_decode_surfaces = 1;
    params.max_display_delay = 1; // push frames to the decoder as quickly as possible
    params.user_data = this;
    params.pfn_sequence_callback = HandleVideoSequenceProc;
    params.pfn_decode_picture = HandlePictureDecodeProc;
    params.pfn_display_picture = HandlePictureDisplayProc;
    if (rocDecCreateVideoParser(&parser_, &params) != ROCDEC_SUCCESS)
        CV_Error(Error::StsNotImplemented, "cudacodec::VideoReader: rocDecCreateVideoParser failed (codec unsupported by the rocDecode parser).");
}

bool cv::cudacodec::detail::VideoParser::parseVideoData(const unsigned char* data, size_t size, const bool rawMode, const bool containsKeyFrame, bool endOfStream)
{
    RocdecSourceDataPacket packet = {};
    packet.flags = ROCDEC_PKT_TIMESTAMP;
    if (endOfStream)
        packet.flags |= ROCDEC_PKT_ENDOFSTREAM;
    packet.payload_size = static_cast<uint32_t>(size);
    packet.payload = data;

    if (rawMode)
        currentFramePackets.push_back(RawPacket(data, size, containsKeyFrame));

    rocDecStatus retVal = ROCDEC_SUCCESS;
    try {
        retVal = rocDecParseVideoData(parser_, &packet);
    }
    catch (const cv::Exception& e) {
        CV_LOG_ERROR(NULL, e.msg);
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    if (retVal != ROCDEC_SUCCESS) {
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    ++unparsedPackets_;
    if (maxUnparsedPackets_ && unparsedPackets_ > maxUnparsedPackets_) {
        CV_LOG_ERROR(NULL, "Maximum number of packets (" << maxUnparsedPackets_ << ") parsed without decoding a frame or reconfiguring the decoder, if reading from \
            a live source consider initializing with VideoReaderInitParams::udpSource == true.");
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    if (endOfStream)
        frameQueue_->endDecode();

    return !frameQueue_->isEndOfDecode();
}

int cv::cudacodec::detail::VideoParser::HandleVideoSequence(RocdecVideoFormat* format)
{
    unparsedPackets_ = 0;

    FormatInfo newFormat;
    newFormat.videoFullRangeFlag = format->video_signal_description.video_full_range_flag;
    newFormat.colorSpaceStandard = static_cast<ColorSpaceStandard>(format->video_signal_description.matrix_coefficients);
    newFormat.codec = videoDecoder_->format().codec; // keep the demuxer's codec (rocDecode and cudacodec enums differ)
    newFormat.chromaFormat = static_cast<ChromaFormat>(format->chroma_format);
    newFormat.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    newFormat.nBitDepthChromaMinus8 = format->bit_depth_chroma_minus8;
    newFormat.ulWidth = format->coded_width;
    newFormat.ulHeight = format->coded_height;
    newFormat.fps = format->frame_rate.denominator ? format->frame_rate.numerator / static_cast<float>(format->frame_rate.denominator) : 0;
    newFormat.targetSz = videoDecoder_->getTargetSz();
    newFormat.srcRoi = videoDecoder_->getSrcRoi();
    if (newFormat.srcRoi.empty()) {
        newFormat.displayArea = Rect(Point(format->display_area.left, format->display_area.top), Point(format->display_area.right, format->display_area.bottom));
        if (newFormat.targetSz.empty())
            newFormat.targetSz = Size((format->display_area.right - format->display_area.left), (format->display_area.bottom - format->display_area.top));
    }
    else
        newFormat.displayArea = newFormat.srcRoi;
    newFormat.width = newFormat.targetSz.width ? newFormat.targetSz.width : format->coded_width;
    newFormat.height = newFormat.targetSz.height ? newFormat.targetSz.height : format->coded_height;
    newFormat.targetRoi = videoDecoder_->getTargetRoi();
    newFormat.ulNumDecodeSurfaces = min(!allowFrameDrop_ ? max(videoDecoder_->nDecodeSurfaces(), static_cast<int>(format->min_num_decode_surfaces)) :
        static_cast<int>(format->min_num_decode_surfaces) * 2, 32);
    newFormat.deinterlaceMode = format->progressive_sequence ? Weave : Adaptive;
    int maxW = max(static_cast<int>(format->coded_width), 0);
    int maxH = max(static_cast<int>(format->coded_height), 0);
    newFormat.ulMaxWidth = maxW;
    newFormat.ulMaxHeight = maxH;
    newFormat.enableHistogram = videoDecoder_->enableHistogram();

    frameQueue_->waitUntilEmpty();
    int retVal = newFormat.ulNumDecodeSurfaces;
    if (videoDecoder_->inited()) {
        retVal = videoDecoder_->reconfigure(newFormat);
        if (retVal > 1 && newFormat.ulNumDecodeSurfaces != frameQueue_->getMaxSz())
            frameQueue_->resize(newFormat.ulNumDecodeSurfaces);
    }
    else {
        frameQueue_->init(newFormat.ulNumDecodeSurfaces);
        videoDecoder_->create(newFormat);
    }
    return retVal;
}

int cv::cudacodec::detail::VideoParser::HandlePictureDecode(RocdecPicParams* picParams)
{
    unparsedPackets_ = 0;

    bool isFrameAvailable = frameQueue_->waitUntilFrameAvailable(picParams->curr_pic_idx, allowFrameDrop_);
    if (!isFrameAvailable)
        return false;

    if (!videoDecoder_->decodePicture(picParams)) {
        CV_LOG_ERROR(NULL, "Decoding failed!");
        hasError_ = true;
        return false;
    }
    return true;
}

int cv::cudacodec::detail::VideoParser::HandlePictureDisplay(RocdecParserDispInfo* dispInfo)
{
    unparsedPackets_ = 0;

    // RocdecParserDispInfo has the same picture_index / progressive_frame /
    // top_field_first / repeat_first_field / pts fields as the compat
    // CUVIDPARSERDISPINFO the frame queue stores, so forward it directly.
    CUVIDPARSERDISPINFO disp;
    disp.picture_index = dispInfo->picture_index;
    disp.progressive_frame = dispInfo->progressive_frame;
    disp.top_field_first = dispInfo->top_field_first;
    disp.repeat_first_field = dispInfo->repeat_first_field;
    disp.pts = dispInfo->pts;
    frameQueue_->enqueue(&disp, currentFramePackets);
    currentFramePackets.clear();
    return true;
}

#endif // HAVE_ROCDECODE

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

    packet.flags = CUVID_PKT_TIMESTAMP;
    if (endOfStream)
        packet.flags |= CUVID_PKT_ENDOFSTREAM;

    packet.payload_size = static_cast<unsigned long>(size);
    packet.payload = data;

    if (rawMode)
        currentFramePackets.push_back(RawPacket(data, size, containsKeyFrame));

    CUresult retVal = CUDA_SUCCESS;
    try {
        retVal = cuvidParseVideoData(parser_, &packet);
    }
    catch(const cv::Exception& e) {
        CV_LOG_ERROR(NULL, e.msg);
        hasError_ = true;
        frameQueue_->endDecode();
        return false;
    }

    if (retVal != CUDA_SUCCESS) {
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

    FormatInfo newFormat;
    newFormat.videoFullRangeFlag = format->video_signal_description.video_full_range_flag;
    newFormat.colorSpaceStandard = static_cast<ColorSpaceStandard>(format->video_signal_description.matrix_coefficients);
    newFormat.codec = static_cast<Codec>(format->codec);
    newFormat.chromaFormat = static_cast<ChromaFormat>(format->chroma_format);
    newFormat.nBitDepthMinus8 = format->bit_depth_luma_minus8;
    newFormat.nBitDepthChromaMinus8 = format->bit_depth_chroma_minus8;
    newFormat.ulWidth = format->coded_width;
    newFormat.ulHeight = format->coded_height;
    newFormat.fps = format->frame_rate.numerator / static_cast<float>(format->frame_rate.denominator);
    newFormat.targetSz = thiz->videoDecoder_->getTargetSz();
    newFormat.srcRoi = thiz->videoDecoder_->getSrcRoi();
    if (newFormat.srcRoi.empty()) {
        newFormat.displayArea = Rect(Point(format->display_area.left, format->display_area.top), Point(format->display_area.right, format->display_area.bottom));
        if (newFormat.targetSz.empty())
            newFormat.targetSz = Size((format->display_area.right - format->display_area.left), (format->display_area.bottom - format->display_area.top));
    }
    else
        newFormat.displayArea = newFormat.srcRoi;
    newFormat.width = newFormat.targetSz.width ? newFormat.targetSz.width : format->coded_width;
    newFormat.height = newFormat.targetSz.height ? newFormat.targetSz.height : format->coded_height;
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
    newFormat.enableHistogram = thiz->videoDecoder_->enableHistogram();

    thiz->frameQueue_->waitUntilEmpty();
    int retVal = newFormat.ulNumDecodeSurfaces;
    if (thiz->videoDecoder_->inited()) {
        retVal = thiz->videoDecoder_->reconfigure(newFormat);
        if (retVal > 1 && newFormat.ulNumDecodeSurfaces != thiz->frameQueue_->getMaxSz())
            thiz->frameQueue_->resize(newFormat.ulNumDecodeSurfaces);
    }
    else {
        thiz->frameQueue_->init(newFormat.ulNumDecodeSurfaces);
        thiz->videoDecoder_->create(newFormat);
    }
    return retVal;
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
