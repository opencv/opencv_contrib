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
    const cudaVideoCodec _codec = static_cast<cudaVideoCodec>(videoFormat.codec);
    const cudaVideoChromaFormat _chromaFormat = static_cast<cudaVideoChromaFormat>(videoFormat.chromaFormat);
    if (videoFormat.nBitDepthMinus8 > 0) {
        std::ostringstream warning;
        warning << "NV12 (8 bit luma, 4 bit chroma) is currently the only supported decoder output format. Video input is " << videoFormat.nBitDepthMinus8 + 8 << " bit " \
            << std::string(GetVideoChromaFormatString(_chromaFormat)) << ".  Truncating luma to 8 bits";
        if (videoFormat.chromaFormat != YUV420)
            warning << " and chroma to 4 bits";
        CV_LOG_WARNING(NULL, warning.str());
    }
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

#if defined (HAVE_CUDA)
#if (CUDART_VERSION >= 6500)
    codecSupported |=       cudaVideoCodec_HEVC     == _codec;
#endif
#if  ((CUDART_VERSION == 7500) || (CUDART_VERSION >= 9000))
    codecSupported |=       cudaVideoCodec_VP8      == _codec ||
                            cudaVideoCodec_VP9      == _codec ||
                            cudaVideoCodec_AV1      == _codec ||
                            cudaVideoCodec_YUV420   == _codec;
#endif
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
    if (!(decodeCaps.bIsSupported && (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12)))){
        CV_Error(Error::StsUnsupportedFormat, "Video source is not supported by hardware video decoder");
        CV_LOG_ERROR(NULL, "Video source is not supported by hardware video decoder.");
    }
    CV_Assert(videoFormat.ulWidth >= decodeCaps.nMinWidth &&
        videoFormat.ulHeight >= decodeCaps.nMinHeight &&
        videoFormat.ulWidth <= decodeCaps.nMaxWidth &&
        videoFormat.ulHeight <= decodeCaps.nMaxHeight);

    CV_Assert((videoFormat.width >> 4)* (videoFormat.height >> 4) <= decodeCaps.nMaxMBCount);
#endif
    // Create video decoder
    CUVIDDECODECREATEINFO createInfo_ = {};
    createInfo_.CodecType           = _codec;
    createInfo_.ulWidth             = videoFormat.ulWidth;
    createInfo_.ulHeight            = videoFormat.ulHeight;
    createInfo_.ulNumDecodeSurfaces = videoFormat.ulNumDecodeSurfaces;
    createInfo_.ChromaFormat    = _chromaFormat;
    createInfo_.bitDepthMinus8 = videoFormat.nBitDepthMinus8;
    createInfo_.OutputFormat    = cudaVideoSurfaceFormat_NV12;
    createInfo_.DeinterlaceMode = static_cast<cudaVideoDeinterlaceMode>(videoFormat.deinterlaceMode);
    createInfo_.ulTargetWidth       = videoFormat.width;
    createInfo_.ulTargetHeight      = videoFormat.height;
    createInfo_.ulMaxWidth          = videoFormat.ulMaxWidth;
    createInfo_.ulMaxHeight         = videoFormat.ulMaxHeight;
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
    cuSafeCall(cuvidCreateDecoder(&decoder_, &createInfo_));
    cuSafeCall(cuCtxPopCurrent(NULL));
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
