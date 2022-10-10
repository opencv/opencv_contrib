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

namespace cv { namespace cudacodec {
using namespace cv::cuda;

#if !defined(HAVE_NVCUVENC)

Ptr<cudacodec::VideoWriter> createVideoWriter(const String&, const Size, const VideoWriterCodec, const double, const COLOR_FORMAT_CV, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const String&, const Size, const VideoWriterCodec, const double, const ENC_BUFFER_FORMAT, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const String&, const Size, const VideoWriterCodec, const double, const COLOR_FORMAT_CV, const EncoderParams&, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const String&, const Size, const VideoWriterCodec, const double, const ENC_BUFFER_FORMAT, const EncoderParams&, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const Ptr<EncoderCallback>&, const Size, const VideoWriterCodec codec, const double, const COLOR_FORMAT_CV, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const Ptr<EncoderCallback>&, const Size, const VideoWriterCodec codec, const double, const ENC_BUFFER_FORMAT, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const Ptr<EncoderCallback>&, const Size, const VideoWriterCodec, const double, const COLOR_FORMAT_CV, const EncoderParams&, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }
Ptr<cudacodec::VideoWriter> createVideoWriter(const Ptr<EncoderCallback>&, const Size, const VideoWriterCodec, const double, const ENC_BUFFER_FORMAT, const EncoderParams&, const cv::cuda::Stream&) { throw_no_cuda(); return Ptr<cv::cudacodec::VideoWriter>(); }

#else // !defined HAVE_NVCUVENC

ENC_BUFFER_FORMAT NvSurfaceFormat(const COLOR_FORMAT_CV format);
int NChannels(const COLOR_FORMAT_CV format);
int NChannels(const ENC_BUFFER_FORMAT format);
GUID CodecGuid(const VideoWriterCodec codec);
void FrameRate(const double fps, uint32_t& frameRateNum, uint32_t& frameRateDen);
GUID EncodingProfileGuid(const ENC_PROFILE encodingProfile);
GUID EncodingPresetGuid(const ENC_PRESET nvPreset);
bool Equal(const GUID& g1, const GUID& g2);

EncoderParams::EncoderParams() : nvPreset(ENC_PRESET_P3), tuningInfo(ENC_TUNING_INFO_HIGH_QUALITY), encodingProfile(ENC_CODEC_PROFILE_AUTOSELECT),
    rateControlMode(ENC_PARAMS_RC_VBR), multiPassEncoding(ENC_MULTI_PASS_DISABLED), constQp({ 0,0,0 }), averageBitRate(0), maxBitRate(0),
    targetQuality(30), gopLength(0)
{
};

bool operator==(const EncoderParams& lhs, const EncoderParams& rhs)
{
    return std::tie(lhs.nvPreset, lhs.tuningInfo, lhs.encodingProfile, lhs.rateControlMode, lhs.multiPassEncoding, lhs.constQp.qpInterB, lhs.constQp.qpInterP, lhs.constQp.qpIntra,
        lhs.averageBitRate, lhs.maxBitRate, lhs.targetQuality, lhs.gopLength) == std::tie(rhs.nvPreset, rhs.tuningInfo, rhs.encodingProfile, rhs.rateControlMode, rhs.multiPassEncoding, rhs.constQp.qpInterB, rhs.constQp.qpInterP, rhs.constQp.qpIntra,
            rhs.averageBitRate, rhs.maxBitRate, rhs.targetQuality, rhs.gopLength);
};

class RawVideoWriter : public EncoderCallback
{
public:
    RawVideoWriter(String fileName);
    ~RawVideoWriter();
    void onEncoded(std::vector<std::vector<uint8_t>> vPacket);
    void onEncodingFinished();
private:
    std::ofstream fpOut;
};

RawVideoWriter::RawVideoWriter(String fileName) {
    fpOut = std::ofstream(fileName, std::ios::out | std::ios::binary);
    if (!fpOut)
        CV_Error(Error::StsError, "Failed to open video file " + fileName + " for writing!");
}

void RawVideoWriter::onEncodingFinished() {
    fpOut.close();
}

RawVideoWriter::~RawVideoWriter() {
    onEncodingFinished();
}

void RawVideoWriter::onEncoded(std::vector<std::vector<uint8_t>> vPacket) {
    for (auto& packet : vPacket)
        fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
}

class VideoWriterImpl : public VideoWriter
{
public:
    VideoWriterImpl(const Ptr<EncoderCallback>& videoWriter, const Size frameSize, const VideoWriterCodec codec, const double fps,
        const COLOR_FORMAT_CV surfaceFormatCv, const Stream& stream = Stream::Null());
    VideoWriterImpl(const Ptr<EncoderCallback>& videoWriter, const Size frameSize, const VideoWriterCodec codec, const double fps,
        const ENC_BUFFER_FORMAT surfaceFormatNv, const Stream& stream = Stream::Null());
    VideoWriterImpl(const Ptr<EncoderCallback>& videoWriter, const Size frameSize, const VideoWriterCodec codec, const double fps,
        const COLOR_FORMAT_CV surfaceFormatCv, const EncoderParams& encoderParams, const Stream& stream = Stream::Null());
    VideoWriterImpl(const Ptr<EncoderCallback>& videoWriter, const Size frameSize, const VideoWriterCodec codec, const double fps,
        const ENC_BUFFER_FORMAT surfaceFormatNv, const EncoderParams& encoderParams, const Stream& stream = Stream::Null());
    ~VideoWriterImpl();
    void write(InputArray frame);
    EncoderParams getEncoderParams() const;
    void close();
private:
    void Init(const VideoWriterCodec codec, const double fps, const Size frameSz);
    void InitializeEncoder(const GUID codec, const double fps);
    void CopyToNvSurface(const InputArray src);

    Ptr<EncoderCallback> encoderCallback;
    COLOR_FORMAT_CV surfaceFormatCv = COLOR_FORMAT_CV::UNDEFINED;
    ENC_BUFFER_FORMAT surfaceFormatNv = ENC_BUFFER_FORMAT::BF_UNDEFINED;
    EncoderParams encoderParams;
    Stream stream = Stream::Null();
    Ptr<NvEncoderCuda> pEnc;
    std::vector<std::vector<uint8_t>> vPacket;
    int nSrcChannels = -1;
    CUcontext cuContext;
};

ENC_BUFFER_FORMAT NvSurfaceFormat(const COLOR_FORMAT_CV format) {
    switch (format) {
    case BGR: return BF_ARGB;
    case RGB: return BF_ABGR;
    case BGRA: return BF_ARGB;
    case RGBA: return BF_ABGR;
    case GRAY: return BF_NV12;
    default: return BF_UNDEFINED;
    }
}

int NChannels(const COLOR_FORMAT_CV format) {
    switch (format) {
    case BGR:
    case RGB: return 3;
    case RGBA:
    case BGRA: return 4;
    case GRAY: return 1;
    default: return 0;
    }
}

int NChannels(const ENC_BUFFER_FORMAT format) {
    if (format == ENC_BUFFER_FORMAT::BF_ARGB || format == ENC_BUFFER_FORMAT::BF_ABGR) return 4;
    else return 1;
}

VideoWriterImpl::VideoWriterImpl(const Ptr<EncoderCallback>& encoderCallBack_, const Size frameSz, const VideoWriterCodec codec, const double fps,
    const COLOR_FORMAT_CV surfaceFormatCv_, const EncoderParams& encoderParams_, const Stream& stream_) :
    encoderCallback(encoderCallBack_), surfaceFormatCv(surfaceFormatCv_), encoderParams(encoderParams_), stream(stream_)
{
    surfaceFormatNv = NvSurfaceFormat(surfaceFormatCv);
    if (surfaceFormatNv == BF_UNDEFINED) {
        String msg = cv::format("Unsupported input surface format: %i", surfaceFormatCv);
        CV_LOG_WARNING(NULL, msg);
        CV_Error(Error::StsUnsupportedFormat, msg);
    }
    nSrcChannels = NChannels(surfaceFormatCv);
    Init(codec, fps, frameSz);
}

VideoWriterImpl::VideoWriterImpl(const Ptr<EncoderCallback>& encoderCallBack_, const Size frameSz, const VideoWriterCodec codec, const double fps,
    const ENC_BUFFER_FORMAT surfaceFormatNv_, const EncoderParams& encoderParams_, const Stream& stream_) :
    encoderCallback(encoderCallBack_), surfaceFormatNv(surfaceFormatNv_), encoderParams(encoderParams_), stream(stream_)
{
    CV_Assert(surfaceFormatNv != BF_UNDEFINED);
    nSrcChannels = NChannels(surfaceFormatNv);
    Init(codec, fps, frameSz);
}

VideoWriterImpl::VideoWriterImpl(const Ptr<EncoderCallback>& encoderCallback, const Size frameSz, const VideoWriterCodec codec, const double fps,
    const COLOR_FORMAT_CV surfaceFormatCv, const Stream& stream) :
    VideoWriterImpl(encoderCallback, frameSz, codec, fps, surfaceFormatCv, EncoderParams(), stream)
{
}

VideoWriterImpl::VideoWriterImpl(const Ptr<EncoderCallback>& encoderCallback, const Size frameSz, const VideoWriterCodec codec, const double fps,
    const ENC_BUFFER_FORMAT surfaceFormatNv, const Stream& stream) :
    VideoWriterImpl(encoderCallback, frameSz, codec, fps, surfaceFormatNv, EncoderParams(), stream)
{
}

void VideoWriterImpl::close() {
    pEnc->EndEncode(vPacket);
    encoderCallback->onEncoded(vPacket);
    encoderCallback->onEncodingFinished();
}

VideoWriterImpl::~VideoWriterImpl() {
    close();
}

GUID CodecGuid(const VideoWriterCodec codec) {
    switch (codec) {
    case VideoWriterCodec::H264: return NV_ENC_CODEC_H264_GUID;
    case VideoWriterCodec::HEVC: return NV_ENC_CODEC_HEVC_GUID;
    default: break;
    }
    std::string msg = "Unknown codec: cudacodec::VideoWriter only supports VideoWriterCodec::H264 and VideoWriterCodec::HEVC";
    CV_LOG_WARNING(NULL, msg);
    CV_Error(Error::StsUnsupportedFormat, msg);
}

void VideoWriterImpl::Init(const VideoWriterCodec codec, const double fps, const Size frameSz) {
    // init context
    GpuMat temp(1, 1, CV_8UC1);
    temp.release();
    cuSafeCall(cuCtxGetCurrent(&cuContext));
    CV_Assert(nSrcChannels != 0);
    const GUID codecGuid = CodecGuid(codec);
    try {
        pEnc = new NvEncoderCuda(cuContext, frameSz.width, frameSz.height, (NV_ENC_BUFFER_FORMAT)surfaceFormatNv);
        InitializeEncoder(codecGuid, fps);
        const cudaStream_t cudaStream = cuda::StreamAccessor::getStream(stream);
        pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&cudaStream, (NV_ENC_CUSTREAM_PTR)&cudaStream);
    }
    catch (cv::Exception& e)
    {
        String msg = String("Error initializing Nvidia Encoder. Refer to Nvidia's GPU Support Matrix to confirm your GPU supports hardware encoding, ") +
            String("codec and surface format and check the encoder documentation to verify your choice of encoding paramaters are supported.") +
            e.msg;
        CV_Error(Error::GpuApiCallError, msg);
    }
    const Size encoderFrameSz(pEnc->GetEncodeWidth(), pEnc->GetEncodeHeight());
    CV_Assert(frameSz == encoderFrameSz);
}

void FrameRate(const double fps, uint32_t& frameRateNum, uint32_t& frameRateDen) {
    CV_Assert(fps >= 0);
    int frame_rate = (int)(fps + 0.5);
    int frame_rate_base = 1;
    while (fabs(((double)frame_rate / frame_rate_base) - fps) > 0.001) {
        frame_rate_base *= 10;
        frame_rate = (int)(fps * frame_rate_base + 0.5);
    }
    frameRateNum = frame_rate;
    frameRateDen = frame_rate_base;
}

GUID EncodingProfileGuid(const ENC_PROFILE encodingProfile) {
    switch (encodingProfile) {
    case(ENC_CODEC_PROFILE_AUTOSELECT): return NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
    case(ENC_H264_PROFILE_BASELINE): return NV_ENC_H264_PROFILE_BASELINE_GUID;
    case(ENC_H264_PROFILE_MAIN): return NV_ENC_H264_PROFILE_MAIN_GUID;
    case(ENC_H264_PROFILE_HIGH): return NV_ENC_H264_PROFILE_HIGH_GUID;
    case(ENC_H264_PROFILE_HIGH_444): return NV_ENC_H264_PROFILE_HIGH_444_GUID;
    case(ENC_H264_PROFILE_STEREO): return NV_ENC_H264_PROFILE_STEREO_GUID;
    case(ENC_H264_PROFILE_PROGRESSIVE_HIGH): return NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID;
    case(ENC_H264_PROFILE_CONSTRAINED_HIGH): return NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID;
    case(ENC_HEVC_PROFILE_MAIN): return NV_ENC_HEVC_PROFILE_MAIN_GUID;
    case(ENC_HEVC_PROFILE_MAIN10): return NV_ENC_HEVC_PROFILE_MAIN10_GUID;
    case(ENC_HEVC_PROFILE_FREXT): return NV_ENC_HEVC_PROFILE_FREXT_GUID;
    default: break;
    }
    std::string msg = "Unknown Encoding Profile.";
    CV_LOG_WARNING(NULL, msg);
    CV_Error(Error::StsUnsupportedFormat, msg);
}

GUID EncodingPresetGuid(const ENC_PRESET nvPreset) {
    switch (nvPreset) {
    case ENC_PRESET_P1: return NV_ENC_PRESET_P1_GUID;
    case ENC_PRESET_P2: return NV_ENC_PRESET_P2_GUID;
    case ENC_PRESET_P3: return NV_ENC_PRESET_P3_GUID;
    case ENC_PRESET_P4: return NV_ENC_PRESET_P4_GUID;
    case ENC_PRESET_P5: return NV_ENC_PRESET_P5_GUID;
    case ENC_PRESET_P6: return NV_ENC_PRESET_P6_GUID;
    case ENC_PRESET_P7: return NV_ENC_PRESET_P7_GUID;
    default: break;
    }
    std::string msg = "Unknown Nvidia Encoding Preset.";
    CV_LOG_WARNING(NULL, msg);
    CV_Error(Error::StsUnsupportedFormat, msg);
}

bool Equal(const GUID& g1, const GUID& g2) {
    if (std::tie(g1.Data1, g1.Data2, g1.Data3, g1.Data4) == std::tie(g2.Data1, g2.Data2, g2.Data3, g2.Data4))
        return true;
    return false;
}

void VideoWriterImpl::InitializeEncoder(const GUID codec, const double fps)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = {};
    initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    NV_ENC_CONFIG encodeConfig = {};
    encodeConfig.version = NV_ENC_CONFIG_VER;
    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, codec, EncodingPresetGuid(encoderParams.nvPreset), (NV_ENC_TUNING_INFO)encoderParams.tuningInfo);
    FrameRate(fps, initializeParams.frameRateNum, initializeParams.frameRateDen);
    initializeParams.encodeConfig->profileGUID = EncodingProfileGuid(encoderParams.encodingProfile);
    initializeParams.encodeConfig->rcParams.rateControlMode = (NV_ENC_PARAMS_RC_MODE)(encoderParams.rateControlMode + encoderParams.multiPassEncoding);
    initializeParams.encodeConfig->rcParams.constQP = { encoderParams.constQp.qpInterB, encoderParams.constQp.qpInterB,encoderParams.constQp.qpInterB };
    initializeParams.encodeConfig->rcParams.averageBitRate = encoderParams.averageBitRate;
    initializeParams.encodeConfig->rcParams.maxBitRate = encoderParams.maxBitRate;
    initializeParams.encodeConfig->rcParams.targetQuality = encoderParams.targetQuality;
    initializeParams.encodeConfig->gopLength = encoderParams.gopLength;
    if (Equal(codec, NV_ENC_CODEC_H264_GUID))
        initializeParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod = encoderParams.gopLength;
    else if (Equal(codec, NV_ENC_CODEC_HEVC_GUID))
        initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = encoderParams.gopLength;
    pEnc->CreateEncoder(&initializeParams);
}

void VideoWriterImpl::CopyToNvSurface(const InputArray src)
{
    const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();
    CV_Assert(src.isGpuMat() || src.isMat());
    if (surfaceFormatCv != COLOR_FORMAT_CV::UNDEFINED)
        CV_Assert(src.size() == Size(pEnc->GetEncodeWidth(), pEnc->GetEncodeHeight()));
    Npp8u* dst = (Npp8u*)encoderInputFrame->inputPtr;
    if (surfaceFormatCv == COLOR_FORMAT_CV::BGR || surfaceFormatCv == COLOR_FORMAT_CV::RGB) {
        GpuMat srcDevice;
        if (src.isGpuMat())
            srcDevice = src.getGpuMat();
        else {
            if (stream)
                srcDevice.upload(src, stream);
            else
                srcDevice.upload(src);
        }
        if (surfaceFormatCv == COLOR_FORMAT_CV::BGR) {
            GpuMat dstGpuMat(pEnc->GetEncodeHeight(), pEnc->GetEncodeWidth(), CV_8UC4, dst, encoderInputFrame->pitch);
            cuda::cvtColor(srcDevice, dstGpuMat, COLOR_BGR2BGRA, 0, stream);
        }
        else {
            GpuMat dstGpuMat(pEnc->GetEncodeHeight(), pEnc->GetEncodeWidth(), CV_8UC4, dst, encoderInputFrame->pitch);
            cuda::cvtColor(srcDevice, dstGpuMat, COLOR_RGB2RGBA, 0, stream);
        }
    }
    else if (surfaceFormatCv == COLOR_FORMAT_CV::GRAY) {
        const cudaMemcpyKind memcpyKind = src.isGpuMat() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        const void* srcPtr = src.isGpuMat() ? src.getGpuMat().data : src.getMat().data;
        const size_t srcPitch = src.isGpuMat() ? src.getGpuMat().step : src.getMat().step;
        const uint32_t chromaHeight = NvEncoder::GetChromaHeight(NV_ENC_BUFFER_FORMAT_NV12, pEnc->GetEncodeHeight());
        if (stream) {
            cudaMemcpy2DAsync(dst, encoderInputFrame->pitch, srcPtr, srcPitch, pEnc->GetEncodeWidth(), pEnc->GetEncodeHeight(), memcpyKind,
                cuda::StreamAccessor::getStream(stream));
            cudaMemset2DAsync(&dst[encoderInputFrame->pitch * pEnc->GetEncodeHeight()], encoderInputFrame->pitch, 128, pEnc->GetEncodeWidth(), chromaHeight,
                cuda::StreamAccessor::getStream(stream));
        }
        else {
            cudaMemcpy2D(dst, encoderInputFrame->pitch, srcPtr, srcPitch, pEnc->GetEncodeWidth(), pEnc->GetEncodeHeight(), memcpyKind);
            cudaMemset2D(&dst[encoderInputFrame->pitch * pEnc->GetEncodeHeight()], encoderInputFrame->pitch, 128, pEnc->GetEncodeWidth(), chromaHeight);
        }
    }
    else {
        void* srcPtr = src.isGpuMat() ? src.getGpuMat().data : src.getMat().data;
        const CUmemorytype cuMemoryType = src.isGpuMat() ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
        NvEncoderCuda::CopyToDeviceFrame(cuContext, srcPtr, static_cast<unsigned>(src.step()), (CUdeviceptr)encoderInputFrame->inputPtr, (int)encoderInputFrame->pitch, pEnc->GetEncodeWidth(),
            pEnc->GetEncodeHeight(), cuMemoryType, encoderInputFrame->bufferFormat, encoderInputFrame->chromaOffsets, encoderInputFrame->numChromaPlanes,
            false, cuda::StreamAccessor::getStream(stream));
    }
}

void VideoWriterImpl::write(const InputArray frame) {
    CV_Assert(frame.channels() == nSrcChannels);
    CopyToNvSurface(frame);
    pEnc->EncodeFrame(vPacket);
    encoderCallback->onEncoded(vPacket);
};

EncoderParams VideoWriterImpl::getEncoderParams() const {
    return encoderParams;
};

Ptr<VideoWriter> createVideoWriter(const String& fileName, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const COLOR_FORMAT_CV colorFormat, const Stream& stream)
{
    Ptr<EncoderCallback> rawVideoWriter = new RawVideoWriter(fileName);
    return createVideoWriter(rawVideoWriter, frameSize, codec, fps, colorFormat, stream);
}

Ptr<VideoWriter> createVideoWriter(const String& fileName, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const ENC_BUFFER_FORMAT bufferFormat, const Stream& stream)
{
    Ptr<EncoderCallback> rawVideoWriter = new RawVideoWriter(fileName);
    return createVideoWriter(rawVideoWriter, frameSize, codec, fps, bufferFormat, stream);
}

Ptr<VideoWriter> createVideoWriter(const String& fileName, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const COLOR_FORMAT_CV colorFormat, const EncoderParams& params, const Stream& stream)
{
    Ptr<EncoderCallback> rawVideoWriter = new RawVideoWriter(fileName);
    return createVideoWriter(rawVideoWriter, frameSize, codec, fps, colorFormat, params, stream);
}

Ptr<VideoWriter> createVideoWriter(const String& fileName, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const ENC_BUFFER_FORMAT bufferFormat, const EncoderParams& params, const Stream& stream)
{
    Ptr<EncoderCallback> rawVideoWriter = new RawVideoWriter(fileName);
    return createVideoWriter(rawVideoWriter, frameSize, codec, fps, bufferFormat, params, stream);
}

Ptr<VideoWriter> createVideoWriter(const Ptr<EncoderCallback>& encoderCallback, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const COLOR_FORMAT_CV colorFormat, const Stream& stream)
{
    return makePtr<VideoWriterImpl>(encoderCallback, frameSize, codec, fps, colorFormat, stream);
}

Ptr<VideoWriter> createVideoWriter(const Ptr<EncoderCallback>& encoderCallback, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const ENC_BUFFER_FORMAT bufferFormat, const Stream& stream)
{
    return makePtr<VideoWriterImpl>(encoderCallback, frameSize, codec, fps, bufferFormat, stream);
}

Ptr<VideoWriter> createVideoWriter(const Ptr<EncoderCallback>& encoderCallback, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const COLOR_FORMAT_CV colorFormat, const EncoderParams& params, const Stream& stream)
{
    return makePtr<VideoWriterImpl>(encoderCallback, frameSize, codec, fps, colorFormat, params, stream);
}

Ptr<VideoWriter> createVideoWriter(const Ptr<EncoderCallback>& encoderCallback, const Size frameSize, const VideoWriterCodec codec, const double fps,
    const ENC_BUFFER_FORMAT bufferFormat, const EncoderParams& params, const Stream& stream)
{
    return makePtr<VideoWriterImpl>(encoderCallback, frameSize, codec, fps, bufferFormat, params, stream);
}
#endif // !defined HAVE_NVCUVENC

}}
