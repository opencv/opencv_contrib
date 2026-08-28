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

using namespace cv::cudacodec::detail;

// cudacodec Codec ordering follows NVCUVID; rocDecode orders its codecs
// differently. Map explicitly so the hardware decoder receives the right codec.
rocDecVideoCodec cv::cudacodec::detail::VideoDecoder::codecToRocDec(const Codec codec) {
    switch (codec) {
    case Codec::MPEG1: return rocDecVideoCodec_MPEG1;
    case Codec::MPEG2: return rocDecVideoCodec_MPEG2;
    case Codec::MPEG4: return rocDecVideoCodec_MPEG4;
    case Codec::H264:  return rocDecVideoCodec_AVC;
    case Codec::HEVC:  return rocDecVideoCodec_HEVC;
    case Codec::AV1:   return rocDecVideoCodec_AV1;
    case Codec::VP8:   return rocDecVideoCodec_VP8;
    case Codec::VP9:   return rocDecVideoCodec_VP9;
    case Codec::JPEG:  return rocDecVideoCodec_JPEG;
    default:           return rocDecVideoCodec_NumCodecs; // VC1 / H264_SVC / H264_MVC: no rocDecode codec
    }
}

rocDecVideoChromaFormat cv::cudacodec::detail::VideoDecoder::chromaToRocDec(const ChromaFormat chroma) {
    switch (chroma) {
    case ChromaFormat::Monochrome: return rocDecVideoChromaFormat_Monochrome;
    case ChromaFormat::YUV420:     return rocDecVideoChromaFormat_420;
    case ChromaFormat::YUV422:     return rocDecVideoChromaFormat_422;
    case ChromaFormat::YUV444:     return rocDecVideoChromaFormat_444;
    default:                       return rocDecVideoChromaFormat_420;
    }
}

static const char* GetRocDecCodecString(rocDecVideoCodec codec) {
    switch (codec) {
    case rocDecVideoCodec_MPEG1: return "MPEG-1";
    case rocDecVideoCodec_MPEG2: return "MPEG-2";
    case rocDecVideoCodec_MPEG4: return "MPEG-4";
    case rocDecVideoCodec_AVC:   return "AVC/H.264";
    case rocDecVideoCodec_HEVC:  return "H.265/HEVC";
    case rocDecVideoCodec_AV1:   return "AV1";
    case rocDecVideoCodec_VP8:   return "VP8";
    case rocDecVideoCodec_VP9:   return "VP9";
    case rocDecVideoCodec_JPEG:  return "M-JPEG";
    default:                     return "Unsupported";
    }
}

static const char* GetRocDecChromaFormatString(rocDecVideoChromaFormat chroma) {
    switch (chroma) {
    case rocDecVideoChromaFormat_Monochrome: return "YUV 400 (Monochrome)";
    case rocDecVideoChromaFormat_420:        return "YUV 420";
    case rocDecVideoChromaFormat_422:        return "YUV 422";
    case rocDecVideoChromaFormat_444:        return "YUV 444";
    default:                                 return "Unknown";
    }
}

void cv::cudacodec::detail::VideoDecoder::create(const FormatInfo& videoFormat)
{
    {
        AutoLock autoLock(mtx_);
        videoFormat_ = videoFormat;
    }
    const rocDecVideoCodec codec = codecToRocDec(videoFormat_.codec);
    const rocDecVideoChromaFormat chromaFormat = chromaToRocDec(videoFormat_.chromaFormat);
    const int deviceId = ctx_; // the HIP device id, recorded by the reader

    if (codec == rocDecVideoCodec_NumCodecs) {
        CV_Error(Error::StsNotImplemented, "cudacodec::VideoReader: codec " + std::to_string(static_cast<int>(videoFormat_.codec)) +
            " is not supported by the AMD rocDecode hardware decoder (rocDecode lacks VC-1 and the H.264 SVC/MVC extensions).");
    }

    // Runtime guard. Query the hardware decoder's capabilities BEFORE creating the
    // decoder. This is the single branch that makes the same multi-arch binary safe
    // on a device with no VCN (e.g. gfx90a, where VA initialization yields
    // is_supported == 0) and functional on gfx1100 / gfx1201, and that rejects codecs
    // / dimensions / output formats the hardware cannot handle. It always throws a
    // catchable cv::Exception; it never crashes or aborts.
    RocdecDecodeCaps decodeCaps = {};
    decodeCaps.device_id = static_cast<uint8_t>(deviceId);
    decodeCaps.codec_type = codec;
    decodeCaps.chroma_format = chromaFormat;
    decodeCaps.bit_depth_minus_8 = videoFormat_.nBitDepthMinus8;
    const rocDecStatus capsStatus = rocDecGetDecoderCaps(&decodeCaps);
    if (capsStatus != ROCDEC_SUCCESS || !decodeCaps.is_supported) {
        std::string deviceName = "device " + std::to_string(deviceId);
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, deviceId) == hipSuccess)
            deviceName = std::string(prop.name) + " (" + prop.gcnArchName + ")";
        CV_Error(Error::StsNotImplemented, "cudacodec::VideoReader: " + deviceName + " has no supported hardware decoder for " +
            std::to_string(videoFormat_.nBitDepthMinus8 + 8) + " bit " + GetRocDecCodecString(codec) + " with " +
            GetRocDecChromaFormatString(chromaFormat) + " chroma (rocDecGetDecoderCaps is_supported=0). Compute-only GPUs (e.g. gfx90a) have no VCN video-decode engine.");
    }

    if (videoFormat_.enableHistogram)
        CV_Error(Error::StsNotImplemented, "cudacodec::VideoReader: luma histogram output is not available through the AMD rocDecode decoder.");

    // Output surface format: rocDecode decodes 4:2:0 to NV12 (8 bit) or P016 (>8 bit).
    surfaceFormat_ = videoFormat_.nBitDepthMinus8 ? rocDecVideoSurfaceFormat_P016 : rocDecVideoSurfaceFormat_NV12;
    if (chromaFormat == rocDecVideoChromaFormat_444)
        surfaceFormat_ = videoFormat_.nBitDepthMinus8 ? rocDecVideoSurfaceFormat_YUV444_16Bit : rocDecVideoSurfaceFormat_YUV444;
    if (!(decodeCaps.output_format_mask & (1 << surfaceFormat_))) {
        if (decodeCaps.output_format_mask & (1 << rocDecVideoSurfaceFormat_NV12))
            surfaceFormat_ = rocDecVideoSurfaceFormat_NV12;
        else if (decodeCaps.output_format_mask & (1 << rocDecVideoSurfaceFormat_P016))
            surfaceFormat_ = rocDecVideoSurfaceFormat_P016;
        else
            CV_Error(Error::StsNotImplemented, "cudacodec::VideoReader: the hardware decoder produces no output surface format the converter supports.");
    }
    videoFormat_.surfaceFormat = static_cast<SurfaceFormat>(surfaceFormat_);

    if (videoFormat_.ulWidth < decodeCaps.min_width || videoFormat_.ulHeight < decodeCaps.min_height ||
        videoFormat_.ulWidth > decodeCaps.max_width || videoFormat_.ulHeight > decodeCaps.max_height) {
        CV_Error(Error::StsNotImplemented, "cudacodec::VideoReader: resolution " + std::to_string(static_cast<int>(videoFormat_.ulWidth)) + "x" +
            std::to_string(static_cast<int>(videoFormat_.ulHeight)) + " is outside the hardware decoder's supported range [" +
            std::to_string(static_cast<int>(decodeCaps.min_width)) + "x" + std::to_string(static_cast<int>(decodeCaps.min_height)) + ", " +
            std::to_string(static_cast<int>(decodeCaps.max_width)) + "x" + std::to_string(static_cast<int>(decodeCaps.max_height)) + "].");
    }

    RocDecoderCreateInfo createInfo = {};
    createInfo.device_id = static_cast<uint8_t>(deviceId);
    createInfo.codec_type = codec;
    createInfo.chroma_format = chromaFormat;
    createInfo.output_format = surfaceFormat_;
    createInfo.bit_depth_minus_8 = videoFormat_.nBitDepthMinus8;
    createInfo.num_decode_surfaces = videoFormat_.ulNumDecodeSurfaces;
    createInfo.width = videoFormat_.ulWidth;
    createInfo.height = videoFormat_.ulHeight;
    createInfo.max_width = videoFormat_.ulMaxWidth ? videoFormat_.ulMaxWidth : videoFormat_.ulWidth;
    createInfo.max_height = videoFormat_.ulMaxHeight ? videoFormat_.ulMaxHeight : videoFormat_.ulHeight;
    createInfo.display_rect.left = videoFormat_.displayArea.x;
    createInfo.display_rect.top = videoFormat_.displayArea.y;
    createInfo.display_rect.right = videoFormat_.displayArea.x + videoFormat_.displayArea.width;
    createInfo.display_rect.bottom = videoFormat_.displayArea.y + videoFormat_.displayArea.height;
    createInfo.target_width = videoFormat_.width;
    createInfo.target_height = videoFormat_.height;
    createInfo.target_rect.left = videoFormat_.targetRoi.x;
    createInfo.target_rect.top = videoFormat_.targetRoi.y;
    createInfo.target_rect.right = videoFormat_.targetRoi.x + videoFormat_.targetRoi.width;
    createInfo.target_rect.bottom = videoFormat_.targetRoi.y + videoFormat_.targetRoi.height;
    createInfo.num_output_surfaces = 2;

    AutoLock autoLock(mtx_);
    if (rocDecCreateDecoder(&decoder_, &createInfo) != ROCDEC_SUCCESS)
        CV_Error(Error::StsError, "cudacodec::VideoReader: rocDecCreateDecoder failed.");
}

int cv::cudacodec::detail::VideoDecoder::reconfigure(const FormatInfo& videoFormat) {
    if (videoFormat.nBitDepthMinus8 != videoFormat_.nBitDepthMinus8)
        CV_Error(Error::StsUnsupportedFormat, "Reconfigure not supported for bit depth change");
    if (videoFormat.chromaFormat != videoFormat_.chromaFormat)
        CV_Error(Error::StsUnsupportedFormat, "Reconfigure not supported for chroma format change");

    const bool decodeResChange = !(videoFormat.ulWidth == videoFormat_.ulWidth && videoFormat.ulHeight == videoFormat_.ulHeight);
    if ((videoFormat.ulWidth > videoFormat_.ulMaxWidth) || (videoFormat.ulHeight > videoFormat_.ulMaxHeight)) {
        if (videoFormat.codec != Codec::VP9)
            CV_Error(Error::StsUnsupportedFormat, "Reconfigure not supported when width/height > maxwidth/maxheight");
    }
    if (!decodeResChange)
        return 1;

    {
        AutoLock autoLock(mtx_);
        videoFormat_.ulNumDecodeSurfaces = videoFormat.ulNumDecodeSurfaces;
        videoFormat_.ulWidth = videoFormat.ulWidth;
        videoFormat_.ulHeight = videoFormat.ulHeight;
        videoFormat_.targetRoi = videoFormat.targetRoi;
    }

    RocdecReconfigureDecoderInfo reconfigParams = {};
    reconfigParams.width = videoFormat_.ulWidth;
    reconfigParams.height = videoFormat_.ulHeight;
    reconfigParams.target_width = videoFormat_.width;
    reconfigParams.target_height = videoFormat_.height;
    reconfigParams.num_decode_surfaces = videoFormat_.ulNumDecodeSurfaces;
    reconfigParams.display_rect.left = videoFormat_.displayArea.x;
    reconfigParams.display_rect.top = videoFormat_.displayArea.y;
    reconfigParams.display_rect.right = videoFormat_.displayArea.x + videoFormat_.displayArea.width;
    reconfigParams.display_rect.bottom = videoFormat_.displayArea.y + videoFormat_.displayArea.height;
    reconfigParams.target_rect.left = videoFormat_.targetRoi.x;
    reconfigParams.target_rect.top = videoFormat_.targetRoi.y;
    reconfigParams.target_rect.right = videoFormat_.targetRoi.x + videoFormat_.targetRoi.width;
    reconfigParams.target_rect.bottom = videoFormat_.targetRoi.y + videoFormat_.targetRoi.height;

    AutoLock autoLock(mtx_);
    if (rocDecReconfigureDecoder(decoder_, &reconfigParams) != ROCDEC_SUCCESS)
        CV_Error(Error::StsError, "cudacodec::VideoReader: rocDecReconfigureDecoder failed.");
    return videoFormat_.ulNumDecodeSurfaces;
}

cv::cuda::GpuMat cv::cudacodec::detail::VideoDecoder::mapFrame(int picIdx, CUVIDPROCPARAMS& videoProcParams)
{
    RocdecProcParams procParams = {};
    procParams.progressive_frame = videoProcParams.progressive_frame;
    procParams.top_field_first = videoProcParams.top_field_first;

    void* devPtr[3] = { 0 };
    uint32_t horizontalPitch[3] = { 0 };
    if (rocDecGetVideoFrame(decoder_, picIdx, devPtr, horizontalPitch, &procParams) != ROCDEC_SUCCESS)
        CV_Error(Error::StsError, "cudacodec::VideoReader: rocDecGetVideoFrame failed.");

    // rocDecode lays out the decoded 4:2:0 surface as a single contiguous pitched
    // allocation: luma rows followed immediately by the interleaved-chroma rows,
    // with devPtr[1] == devPtr[0] + pitch * targetHeight. That is exactly the NV12
    // surface the YUV->RGB converter consumes, so wrap devPtr[0] as one GpuMat of
    // height * 3 / 2 rows without any copy (mirrors the cuvid mapFrame).
    const bool is16Bit = surfaceFormat_ == rocDecVideoSurfaceFormat_P016 || surfaceFormat_ == rocDecVideoSurfaceFormat_YUV444_16Bit;
    const bool is444 = surfaceFormat_ == rocDecVideoSurfaceFormat_YUV444 || surfaceFormat_ == rocDecVideoSurfaceFormat_YUV444_16Bit;
    const int height = is444 ? targetHeight() * 3 : targetHeight() * 3 / 2;
    const int type = (surfaceFormat_ == rocDecVideoSurfaceFormat_NV12 || surfaceFormat_ == rocDecVideoSurfaceFormat_YUV444) ? CV_8U : CV_16U;
    CV_UNUSED(is16Bit);
    return cuda::GpuMat(height, targetWidth(), type, devPtr[0], horizontalPitch[0]);
}

void cv::cudacodec::detail::VideoDecoder::release()
{
    AutoLock autoLock(mtx_);
    if (decoder_) {
        rocDecDestroyDecoder(decoder_);
        decoder_ = nullptr;
    }
}

#endif // HAVE_ROCDECODE

#ifdef HAVE_NVCUVID

#if (CUDART_VERSION >= 9000)
static const char* GetVideoCodecString(cudaVideoCodec eCodec) {
    static struct {
        cudaVideoCodec eCodec;
        const char* name;
    } aCodecName[] = {
        { cudaVideoCodec_MPEG1,     "MPEG-1"       },
        { cudaVideoCodec_MPEG2,     "MPEG-2"       },
        { cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,       "VC-1/WMV"     },
        { cudaVideoCodec_H264,      "AVC/H.264"    },
        { cudaVideoCodec_JPEG,      "M-JPEG"       },
        { cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
        { cudaVideoCodec_HEVC,      "H.265/HEVC"   },
        { cudaVideoCodec_VP8,       "VP8"          },
        { cudaVideoCodec_VP9,       "VP9"          },
        { cudaVideoCodec_AV1,       "AV1"          },
        { cudaVideoCodec_NumCodecs, "Invalid"      },
        { cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,      "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,      "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
    };

    if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
        return aCodecName[eCodec].name;
    }
    for (int i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
        if (eCodec == aCodecName[i].eCodec) {
            return aCodecName[eCodec].name;
        }
    }
    return "Unknown";
}
#endif

static const char* GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char* name;
    } aChromaFormatName[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

void cv::cudacodec::detail::VideoDecoder::create(const FormatInfo& videoFormat)
{
    {
        AutoLock autoLock(mtx_);
        videoFormat_ = videoFormat;
    }
    const cudaVideoCodec _codec = static_cast<cudaVideoCodec>(videoFormat_.codec);
    const cudaVideoChromaFormat _chromaFormat = static_cast<cudaVideoChromaFormat>(videoFormat_.chromaFormat);

    cudaVideoSurfaceFormat surfaceFormat = cudaVideoSurfaceFormat_NV12;
#if (CUDART_VERSION < 9000)
    if (videoFormat.nBitDepthMinus8 > 0) {
    std::ostringstream warning;
    warning << "NV12 (8 bit luma, 4 bit chroma) is currently the only supported decoder output format. Video input is " << videoFormat.nBitDepthMinus8 + 8 << " bit " \
            << std::string(GetVideoChromaFormatString(_chromaFormat)) << ".  Truncating luma to 8 bits";
        if (videoFormat.chromaFormat != YUV420)
            warning << " and chroma to 4 bits";
        CV_LOG_WARNING(NULL, warning.str());
    }
#else
    if (_chromaFormat == cudaVideoChromaFormat_420 || cudaVideoChromaFormat_Monochrome)
        surfaceFormat = videoFormat_.nBitDepthMinus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    else if (_chromaFormat == cudaVideoChromaFormat_444)
        surfaceFormat = videoFormat_.nBitDepthMinus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
    else if (_chromaFormat == cudaVideoChromaFormat_422) {
        surfaceFormat = videoFormat_.nBitDepthMinus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
        CV_LOG_WARNING(NULL, "YUV 4:2:2 is not currently supported, falling back to YUV 4:2:0.");
    }
#endif

    const cudaVideoCreateFlags videoCreateFlags = (_codec == cudaVideoCodec_JPEG || _codec == cudaVideoCodec_MPEG2) ?
                                            cudaVideoCreate_PreferCUDA :
                                            cudaVideoCreate_PreferCUVID;

    // Validate video format.  These are the currently supported formats via NVCUVID
    bool codecSupported =   cudaVideoCodec_MPEG1    == _codec ||
                            cudaVideoCodec_MPEG2    == _codec ||
                            cudaVideoCodec_MPEG4    == _codec ||
                            cudaVideoCodec_VC1      == _codec ||
                            cudaVideoCodec_H264     == _codec ||
                            cudaVideoCodec_JPEG     == _codec ||
                            cudaVideoCodec_H264_SVC == _codec ||
                            cudaVideoCodec_H264_MVC == _codec ||
                            cudaVideoCodec_YV12     == _codec ||
                            cudaVideoCodec_NV12     == _codec ||
                            cudaVideoCodec_YUYV     == _codec ||
                            cudaVideoCodec_UYVY     == _codec;

#if (CUDART_VERSION >= 6050)
    codecSupported |= cudaVideoCodec_HEVC == _codec;
#endif
#if (CUDART_VERSION >= 7050)
    codecSupported |= cudaVideoCodec_YUV420 == _codec;
#endif
#if  ((CUDART_VERSION == 7050) || (CUDART_VERSION >= 9000))
    codecSupported |= cudaVideoCodec_VP8 == _codec || cudaVideoCodec_VP9 == _codec;
#endif
#if (CUDART_VERSION >= 9000)
    codecSupported |= cudaVideoCodec_AV1;
#endif
    CV_Assert(codecSupported);
    CV_Assert(  cudaVideoChromaFormat_Monochrome == _chromaFormat ||
                cudaVideoChromaFormat_420        == _chromaFormat ||
                cudaVideoChromaFormat_422        == _chromaFormat ||
                cudaVideoChromaFormat_444        == _chromaFormat);

#if (CUDART_VERSION >= 9000)
    // Check video format is supported by GPU's hardware video decoder
    CUVIDDECODECAPS decodeCaps = {};
    decodeCaps.eCodecType = _codec;
    decodeCaps.eChromaFormat = _chromaFormat;
    decodeCaps.nBitDepthMinus8 = videoFormat.nBitDepthMinus8;
    cuSafeCall(cuCtxPushCurrent(ctx_));
    cuSafeCall(cuvidGetDecoderCaps(&decodeCaps));
    cuSafeCall(cuCtxPopCurrent(NULL));

    if (!decodeCaps.bIsSupported) {
        CV_Error(Error::StsUnsupportedFormat, std::to_string(decodeCaps.nBitDepthMinus8 + 8) + " bit " + GetVideoCodecString(_codec) + " with " + GetVideoChromaFormatString(_chromaFormat) + " chroma format is not supported by this GPU hardware video decoder.  Please refer to Nvidia's GPU Support Matrix to confirm your GPU supports hardware decoding of this video source.");
    }

    if (!(decodeCaps.nOutputFormatMask & (1 << surfaceFormat)))
    {
        if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
            surfaceFormat = cudaVideoSurfaceFormat_NV12;
        else if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
            surfaceFormat = cudaVideoSurfaceFormat_P016;
        else if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
            surfaceFormat = cudaVideoSurfaceFormat_YUV444;
        else if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
            surfaceFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
        else
            CV_Error(Error::StsUnsupportedFormat, "No supported output format found");
    }
    videoFormat_.surfaceFormat = static_cast<SurfaceFormat>(surfaceFormat);

    if (videoFormat.enableHistogram) {
        if (!decodeCaps.bIsHistogramSupported) {
            CV_Error(Error::StsBadArg, "Luma histogram output is not supported for current codec and/or on current device.");
        }

        if (decodeCaps.nCounterBitDepth != 32) {
            std::ostringstream error;
            error << "Luma histogram output disabled due to current device using " << decodeCaps.nCounterBitDepth << " bit bins. Histogram output only supports 32 bit bins.";
            CV_Error(Error::StsBadArg, error.str());
        }
        else {
            videoFormat_.nCounterBitDepth = decodeCaps.nCounterBitDepth;
            videoFormat_.nMaxHistogramBins = decodeCaps.nMaxHistogramBins;
        }
    }

    CV_Assert(videoFormat.ulWidth >= decodeCaps.nMinWidth &&
        videoFormat.ulHeight >= decodeCaps.nMinHeight &&
        videoFormat.ulWidth <= decodeCaps.nMaxWidth &&
        videoFormat.ulHeight <= decodeCaps.nMaxHeight);

    CV_Assert((videoFormat.width >> 4) * (videoFormat.height >> 4) <= decodeCaps.nMaxMBCount);
#else
    if (videoFormat.enableHistogram) {
        CV_Error(Error::StsBadArg, "Luma histogram output is not supported when CUDA Toolkit version <= 9.0.");
    }
#endif

    // Create video decoder
    CUVIDDECODECREATEINFO createInfo_ = {};
#if (CUDART_VERSION >= 9000)
    createInfo_.enableHistogram = videoFormat.enableHistogram;
    createInfo_.bitDepthMinus8 = videoFormat.nBitDepthMinus8;
    createInfo_.ulMaxWidth = videoFormat.ulMaxWidth;
    createInfo_.ulMaxHeight = videoFormat.ulMaxHeight;
#endif
    createInfo_.CodecType           = _codec;
    createInfo_.ulWidth             = videoFormat.ulWidth;
    createInfo_.ulHeight            = videoFormat.ulHeight;
    createInfo_.ulNumDecodeSurfaces = videoFormat.ulNumDecodeSurfaces;
    createInfo_.ChromaFormat    = _chromaFormat;
    createInfo_.OutputFormat    = surfaceFormat;
    createInfo_.DeinterlaceMode = static_cast<cudaVideoDeinterlaceMode>(videoFormat.deinterlaceMode);
    createInfo_.ulTargetWidth       = videoFormat.width;
    createInfo_.ulTargetHeight      = videoFormat.height;
    createInfo_.display_area.left   = videoFormat.displayArea.x;
    createInfo_.display_area.right  = videoFormat.displayArea.x + videoFormat.displayArea.width;
    createInfo_.display_area.top    = videoFormat.displayArea.y;
    createInfo_.display_area.bottom = videoFormat.displayArea.y + videoFormat.displayArea.height;
    createInfo_.target_rect.left    = videoFormat.targetRoi.x;
    createInfo_.target_rect.right   = videoFormat.targetRoi.x + videoFormat.targetRoi.width;
    createInfo_.target_rect.top     = videoFormat.targetRoi.y;
    createInfo_.target_rect.bottom  = videoFormat.targetRoi.y + videoFormat.targetRoi.height;
    createInfo_.ulNumOutputSurfaces = 2;
    createInfo_.ulCreationFlags     = videoCreateFlags;
    createInfo_.vidLock = lock_;
    cuSafeCall(cuCtxPushCurrent(ctx_));
    {
        AutoLock autoLock(mtx_);
        cuSafeCall(cuvidCreateDecoder(&decoder_, &createInfo_));
    }
    cuSafeCall(cuCtxPopCurrent(NULL));
}

int cv::cudacodec::detail::VideoDecoder::reconfigure(const FormatInfo& videoFormat) {
    if (videoFormat.nBitDepthMinus8 != videoFormat_.nBitDepthMinus8 || videoFormat.nBitDepthChromaMinus8 != videoFormat_.nBitDepthChromaMinus8) {
        CV_Error(Error::StsUnsupportedFormat, "Reconfigure Not supported for bit depth change");
    }

    if (videoFormat.chromaFormat != videoFormat_.chromaFormat) {
        CV_Error(Error::StsUnsupportedFormat, "Reconfigure Not supported for chroma format change");
    }

    const bool decodeResChange = !(videoFormat.ulWidth == videoFormat_.ulWidth && videoFormat.ulHeight == videoFormat_.ulHeight);

    if ((videoFormat.ulWidth > videoFormat_.ulMaxWidth) || (videoFormat.ulHeight > videoFormat_.ulMaxHeight)) {
        // For VP9, let driver  handle the change if new width/height > maxwidth/maxheight
        if (videoFormat.codec != Codec::VP9) {
            CV_Error(Error::StsUnsupportedFormat, "Reconfigure Not supported when width/height > maxwidth/maxheight");
        }
    }

    if (!decodeResChange)
        return 1;

    {
        AutoLock autoLock(mtx_);
        videoFormat_.ulNumDecodeSurfaces = videoFormat.ulNumDecodeSurfaces;
        videoFormat_.ulWidth = videoFormat.ulWidth;
        videoFormat_.ulHeight = videoFormat.ulHeight;
        videoFormat_.targetRoi = videoFormat.targetRoi;
    }

    CUVIDRECONFIGUREDECODERINFO reconfigParams = { 0 };
    reconfigParams.ulWidth = videoFormat_.ulWidth;
    reconfigParams.ulHeight = videoFormat_.ulHeight;
    reconfigParams.display_area.left = videoFormat_.displayArea.x;
    reconfigParams.display_area.right = videoFormat_.displayArea.x + videoFormat_.displayArea.width;
    reconfigParams.display_area.top = videoFormat_.displayArea.y;
    reconfigParams.display_area.bottom = videoFormat_.displayArea.y + videoFormat_.displayArea.height;
    reconfigParams.ulTargetWidth = videoFormat_.width;
    reconfigParams.ulTargetHeight = videoFormat_.height;
    reconfigParams.target_rect.left = videoFormat_.targetRoi.x;
    reconfigParams.target_rect.right = videoFormat_.targetRoi.x + videoFormat_.targetRoi.width;
    reconfigParams.target_rect.top = videoFormat_.targetRoi.y;
    reconfigParams.target_rect.bottom = videoFormat_.targetRoi.y + videoFormat_.targetRoi.height;
    reconfigParams.ulNumDecodeSurfaces = videoFormat_.ulNumDecodeSurfaces;

    cuSafeCall(cuCtxPushCurrent(ctx_));
    cuSafeCall(cuvidReconfigureDecoder(decoder_, &reconfigParams));
    cuSafeCall(cuCtxPopCurrent(NULL));
    CV_LOG_INFO(NULL, "Reconfiguring Decoder");
    return videoFormat_.ulNumDecodeSurfaces;
}

void cv::cudacodec::detail::VideoDecoder::release()
{
    if (decoder_)
    {
        cuvidDestroyDecoder(decoder_);
        decoder_ = 0;
    }
}

#endif // HAVE_NVCUVID
